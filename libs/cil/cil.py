from typing import Optional, Union, List, Tuple, Dict
import pathlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics.classification
from mmcv.runner import build_optimizer
from mmcv.utils.config import Config as mmcvConfig
from mmaction.datasets import build_dataset, RawframeDataset
from mmaction.models import build_model
from tqdm import tqdm

from ..module_hooks import OutputHook
from .memory_selection import Herding
from ..utils import print_mean_accuracy, build_lr_scheduler
from ..loader.comix_loader import BackgroundMixDataset


class CILDataModule(pl.LightningDataModule):
    def __init__(self, config: mmcvConfig):
        super().__init__()
        self.config = config

        self.batch_size = config.videos_per_gpu
        self.test_batch_size = config.testing_videos_per_gpu
        self.task_splits = config.task_splits
        self.work_dir = pathlib.Path(config.work_dir)

        self.num_classes_per_task = []
        accumulate_num_tasks = 0
        for i in self.task_splits:
            accumulate_num_tasks += len(i)
            self.num_classes_per_task.append(accumulate_num_tasks)

        self.ori_idx_to_inc_idx = {}
        for task_i in self.task_splits:
            for i in task_i:
                if i not in self.ori_idx_to_inc_idx:
                    self.ori_idx_to_inc_idx[i] = len(self.ori_idx_to_inc_idx)

        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.exemplar_dir = self.work_dir / 'exemplar'
        self.exemplar_dir.mkdir(exist_ok=True, parents=True)

        self.controller = None  # BaseCIL instance
        self.task_splits_ann_files = {'train': [], 'val': []}
        self.train_dataset = None
        self.val_datasets = []
        self.test_datasets = []
        self.features_extraction_dataset = None
        self.exemplar_datasets = []
        self._all_bg_files = set()

        # flag to control predict_dataloader
        self.predict_dataloader_mode = 'val'  # ['val', 'test']
        self.predict_dataset_task_idx = 0

    @property
    def current_task(self):
        return self.controller.current_task

    @property
    def num_tasks(self):
        return self.controller.num_tasks

    @property
    def exemplar_size(self):
        size = 0
        for ex in self.exemplar_datasets:
            size += len(ex)
        return size

    @property
    def all_bg_files(self):
        return self._all_bg_files

    def generate_annotation_file(self) -> None:
        train_ann_file = pathlib.Path(self.config.train_ann_file)
        val_ann_file = pathlib.Path(self.config.val_ann_file)
        destination = self.work_dir / 'task_splits'
        destination.mkdir(exist_ok=True, parents=True)

        for train_val, file_path in zip(['train', 'val'], [train_ann_file, val_ann_file]):
            with open(file_path, 'r') as f:
                data = f.readlines()

            annotation_ = {}
            for l in data:
                video_path, total_frames, label = l.strip().split()
                annotation_[video_path] = total_frames, int(label)

            # task_i_oracle = []
            for task_i, class_indices in enumerate(self.task_splits):
                class_indices = set(class_indices)
                task_i_data = []
                for video_path, (total_frames, label) in list(annotation_.items()):
                    if label in class_indices:
                        task_i_data.append((video_path, total_frames, self.ori_idx_to_inc_idx[label]))

                if task_i_data:
                    task_i_data_str = ['{} {} {}\n'.format(video_path, total_frames, label) for
                                       video_path, total_frames, label
                                       in task_i_data]
                    task_i_file_path = destination / self.config.cil_ann_file_template.format(train_val, task_i)
                    with open(task_i_file_path, 'w') as f:
                        f.writelines(task_i_data_str)

                    self.task_splits_ann_files[train_val].append(task_i_file_path)
                    print('create file at:', str(task_i_file_path))

    def collect_ann_files_from_work_dir(self):
        ann_files_dir = self.work_dir / 'task_splits'
        for task_i in range(self.num_tasks):
            self.task_splits_ann_files['train'].append(
                ann_files_dir / self.config.cil_ann_file_template.format('train', task_i))

            self.task_splits_ann_files['val'].append(
                ann_files_dir / self.config.cil_ann_file_template.format('val', task_i))

    def collect_exemplar_fron_work_dir(self):
        for task_idx in range(self.current_task):
            ann_file = self.exemplar_dir / 'exemplar_task_{}.txt'.format(task_idx)
            if ann_file.exists():
                exemplar_dataset = self.build_exemplar_dataset(str(ann_file))
                self.exemplar_datasets.append(exemplar_dataset)
            else:
                raise FileNotFoundError

    def build_validation_datasets(self):
        for i in range(self.num_tasks):
            self.config.data.val.ann_file = str(self.task_splits_ann_files['val'][i])
            val_ds = build_dataset(self.config.data.val)
            val_ds.test_mode = True
            self.val_datasets.append(val_ds)

    def build_cbf_dataset(self):
        dataset = build_dataset(self.config.data.train)  # TODO: create a exemplar data pipeline in config file
        dataset.video_infos = []

        if isinstance(dataset, BackgroundMixDataset):
            dataset.bg_files = []

        if self.config.keep_all_backgrounds:
            dataset = self.merge_dataset(dataset, self.exemplar_datasets)
            dataset.bg_files = list(self._all_bg_files)

        elif self.config.cbf_full_bg:
            dataset = self.merge_dataset(dataset, self.exemplar_datasets)
            all_bg_files_ = set(self.train_dataset.bg_files) | set(dataset.bg_files)
            dataset.bg_files = list(all_bg_files_)

        elif isinstance(dataset, RawframeDataset):
            dataset = self.merge_dataset(dataset, self.exemplar_datasets)

        else:
            raise NotImplementedError
        print('CBF dataset built ({} videos, {} background)'.format(len(dataset), len(dataset.bg_files)))
        return dataset

    def reload_train_dataset(self,
                             exemplar: Optional[Union[RawframeDataset, List[RawframeDataset]]] = None,
                             use_internal_exemplar=True):
        """
        this method should be used for reloading the train_dataset when moving to next task
        Note: the self.controller.current_task should be updated with new value before calling this method
        """
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        self.train_dataset = build_dataset(self.config.data.train)

        # self.config.data.val.ann_file = str(self.task_splits_ann_files['val'][self.current_task])
        # self.val_datasets.append(build_dataset(self.config.data.val))

        if use_internal_exemplar:
            self.train_dataset = self.merge_dataset(self.train_dataset, self.exemplar_datasets)

        elif exemplar is not None:
            self.train_dataset = self.merge_dataset(self.train_dataset, exemplar)

        if isinstance(self.train_dataset, BackgroundMixDataset) and self.config.keep_all_backgrounds:
            self._all_bg_files.update(self.train_dataset.bg_files)
            self.train_dataset.bg_files = list(self._all_bg_files)

    def get_training_set_at_task_i(self, taskIdx):
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][taskIdx])
        dataset = build_dataset(self.config.data.train)
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=True)

    # avoid override val_dataloader abstract method
    def get_val_dataloader(self, task_idx: int):
        return DataLoader(self.val_datasets[task_idx],
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False
                          )

    def get_test_dataloader(self, task_idx: int):
        return DataLoader(self.test_datasets[task_idx],
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False
                          )

    def predict_dataloader(self):
        if self.predict_dataloader_mode == 'val':
            print('[predict_dataloader] mode:', self.predict_dataloader_mode)
            loader = self.get_val_dataloader(self.predict_dataset_task_idx)
            # return self.get_val_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'test':
            print('[predict_dataloader] mode:', self.predict_dataloader_mode)
            loader = self.get_test_dataloader(self.predict_dataset_task_idx)
            # return self.get_val_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'feature_extraction':
            print('[predict_dataloader] mode:', self.predict_dataloader_mode)
            loader = self.features_extraction_dataloader()
            # return self.features_extraction_dataloader()
        else:
            raise ValueError
        print('Number of videos:', len(loader.dataset.video_infos))
        return loader

    def features_extraction_dataloader(self):
        """
        extracting features for memory selection
        """
        self.config.data.features_extraction.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        self.features_extraction_dataset = build_dataset(self.config.data.features_extraction)
        return DataLoader(self.features_extraction_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False
                          )

    def create_exemplar_ann_file(self, exemplar_meta: dict, task_idx=-1) -> str:
        if task_idx == -1:  # automatically infer the current task index using internal state from controller
            task_idx = self.current_task

        root_dir = pathlib.Path(self.config.data_root)
        ann_file = self.exemplar_dir / 'exemplar_task_{}.txt'.format(task_idx)
        with open(ann_file, 'w') as f:
            for classIdx, meta in exemplar_meta.items():
                for frame_dir, total_frames in zip(meta['frame_dir'], meta['total_frames']):
                    diff = pathlib.Path(frame_dir).relative_to(root_dir.absolute())
                    f.write('{} {} {}\n'.format(str(diff), total_frames, classIdx))

        return str(ann_file)

    def build_exemplar_dataset(self, ann_file: str):
        self.config.data.exemplar.ann_file = ann_file
        return build_dataset(self.config.data.exemplar)

    def build_exemplar_from_current_task(self, exemplar_meta: dict):
        ann_file = self.create_exemplar_ann_file(exemplar_meta)
        exemplar_dataset = self.build_exemplar_dataset(ann_file)
        self.exemplar_datasets.append(exemplar_dataset)

    def merge_dataset(self, source: RawframeDataset, targets: Union[RawframeDataset, List[RawframeDataset]]):
        """
        copy data info from target then merge to source. This method is useful
        """
        if isinstance(targets, list):
            for target_ in targets:
                source = self._merge_dataset(source, target_)
        else:
            source = self._merge_dataset(source, targets)
        return source

    @staticmethod
    def _merge_dataset(source: RawframeDataset, target_: Union[RawframeDataset, List[RawframeDataset]]):
        if type(source) != type(target_):
            raise ValueError('source and target must be the same type ')

        if isinstance(source, BackgroundMixDataset):
            source.video_infos.extend(target_.video_infos)
            source.bg_files.extend(target_.bg_files)
        elif isinstance(source, RawframeDataset):
            source.video_infos.extend(target_.video_infos)
        else:
            raise TypeError
        return source

    def store_bg_files(self, bg_files):
        self._all_bg_files.update(bg_files)


class BaseCIL(pl.LightningModule):
    def __init__(self,
                 config: mmcvConfig,
                 is_train=True,
                 use_nme_classifier=True
                 ):
        super().__init__()
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.controller = None  # BaseCIL instance
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = len(config.task_splits)
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        if hasattr(config, 'kd_modules_names'):
            self.use_kd = True
        else:
            self.use_kd = False
        self.current_model = build_model(config.model)  # current model

        if self.use_kd and is_train:
            self.current_model_kd_hooks = self.register_modules_hooks(self.current_model, config.kd_modules_names)
            self.kd_modules_names = config.kd_modules_names
            self.prev_model = build_model(config.model)
            self.prev_model_kd_hooks = self.register_modules_hooks(self.prev_model, config.kd_modules_names)
            self.prev_model.eval()
        else:
            self.current_model_kd_hooks = None
            self.kd_modules_names = []
            self.prev_model = None
            self.prev_model_kd_hooks = None

        # TODO: Consider move the representation extraction to respective model (self.current_model)
        self.extract_repr = False
        self._repr_module_name = config.repr_hook
        self._repr_hook = self.register_modules_hooks(self.current_model, [self._repr_module_name])

        # optimizers
        self.optimizer_mode = 'default'         # ['default', 'cbf']

        self.extract_meta = False

    # initialization
    def configure_optimizers(self):
        if self.optimizer_mode == 'default':
            print('build optimizer for incremental training')
            optimizer_config = self.config.optimizer
            lr_scheduler_config = self.config.lr_scheduler
        elif self.optimizer_mode == 'cbf':
            print('build optimizer for CBF training')
            optimizer_config = self.config.cbf_optimizer
            lr_scheduler_config = self.config.cbf_lr_scheduler

        else:
            raise ValueError

        optimizer = build_optimizer(self.current_model, optimizer_config)
        if lr_scheduler_config:
            return {
                    'optimizer': optimizer,
                    'lr_scheduler': build_lr_scheduler(optimizer, lr_scheduler_config),
                    }
        return optimizer

    @staticmethod
    def register_modules_hooks(model, modules_names: List[str]):
        output_hooks = OutputHook(model, modules_names, as_tensor=True)
        return output_hooks

    # properties
    @property
    def current_task(self):
        return self.controller.current_task

    def num_classes(self, task_idx=-1):
        return self.controller.num_classes(task_idx)

    # others
    def _extract_repr(self):
        # https://mmaction2.readthedocs.io/en/latest/_modules/mmaction/models/heads/tsm_head.html#TSMHead
        repr_ = self._repr_hook.get_layer_output(self._repr_module_name).flatten(1)
        repr_ = repr_.view(-1, self.current_model.cls_head.num_segments, repr_.size(1))
        repr_consensus = self.current_model.cls_head.consensus(repr_).squeeze(1)
        return repr_consensus

    # forwarding, training, validation and testing
    def forward(self, x):
        return self.current_model(x)

    def training_step(self, batch_data, batch_idx):
        previous_task_num_classes = self.num_classes(self.current_task - 1)
        # x.shape = (batch_size, channels, T, H, W)
        imgs, labels = batch_data['imgs'], batch_data['label']
        if 'blended' in batch_data:
            losses = self.current_model(imgs, labels, mixed_bg=batch_data['blended'])
        else:
            losses = self.current_model(imgs, labels)  # losses = {'loss_cls': loss_cls}
        if self.use_kd and self.current_task > 0:
            kd_loss = 0
            kd_criterion = nn.MSELoss()
            self.prev_model.eval()
            with torch.no_grad():
                self.prev_model.forward_test(imgs)

            for m_name in self.kd_modules_names:
                current_model_features = self.current_model_kd_hooks.get_layer_output(m_name)
                prev_model_features = self.prev_model_kd_hooks.get_layer_output(m_name).detach()

                if self.config.kd_exemplar_only:
                    indices = (batch_data['label'].view(-1) < previous_task_num_classes).nonzero().squeeze()
                    if indices.nelement():
                        kd_loss += kd_criterion(current_model_features[indices], prev_model_features[indices])
                else:
                    kd_loss += kd_criterion(current_model_features, prev_model_features)
            losses['kd_loss'] = kd_loss
        else:
            losses['kd_loss'] = 0.

        batch_size = imgs.shape[0]
        for loss_name, loss_value in losses.items():
            self.log('train_' + loss_name, losses[loss_name], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch_size)

        loss = losses['kd_loss'] + losses['loss_cls']
        if 'loss_bg_mixed' in losses:
            loss += losses['loss_bg_mixed']
        return loss

    def predict_step(self, batch_data, batch_idx, dataloader_idx: Optional[int] = None):
        x = batch_data['imgs']
        cls_score = self.current_model(x, return_loss=False)
        result = {'cls_score': cls_score,
                  'label': batch_data['label']
                  }
        if self.extract_repr:
            repr_ = self._extract_repr()
            batch_size = batch_data['imgs'].size(0)
            embedding_size = repr_.size(-1)
            repr_ = repr_.view(batch_size, -1, embedding_size)
            repr_ = F.normalize(repr_, p=2, dim=-1)
            repr_ = torch.mean(repr_, dim=1, keepdim=False)
            result['repr_'] = repr_
            assert result['repr_'].size(0) == result['cls_score'].size(0)
        if self.extract_meta:
            for k, v in batch_data.items():
                if k not in ['label', 'imgs', 'blended']:
                    result[k] = v
        return result


class CILTrainer:
    def __init__(self, config, dump_config=True):
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.starting_task = config.starting_task
        # self.ending_task = 0
        self._current_task = self.starting_task
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = len(config.task_splits)
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        self.data_module = CILDataModule(config)
        self.data_module.controller = self
        self.cil_model = BaseCIL(config)
        self.cil_model.controller = self

        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.starting_task == 0:
            self.data_module.generate_annotation_file()
            self.data_module.reload_train_dataset(exemplar=None, use_internal_exemplar=False)

        # resume training
        else:
            self.data_module.collect_ann_files_from_work_dir()
            try:
                self.data_module.collect_exemplar_fron_work_dir()
            except FileNotFoundError:
                for i in range(len(self.data_module.exemplar_datasets), self.starting_task):
                    self._current_task = i
                    print('Create exemplar for task {}'.format(i))
                    class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in
                                     self.task_splits[self.current_task]]
                    manager = Herding(budget_size=self.config.budget_size,
                                      class_indices=class_indices,
                                      cosine_distance=True,
                                      storing_methods=self.config.storing_methods,
                                      budget_type=self.config.budget_type)
                    prediction_with_meta = self._extract_features_for_constructing_exemplar()
                    exemplar_meta = manager.construct_exemplar(prediction_with_meta)
                    self.data_module.build_exemplar_from_current_task(exemplar_meta)
                self._current_task = self.starting_task

            # roll back to previous task_idx to load weights
            self._current_task -= 1
            self.cil_model.current_model.update_fc(self.num_classes())
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))
            self.cil_model.prev_model.update_fc(self.num_classes())
            self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
            self.cil_model.prev_model.eval()

            # back to starting task update the classifier
            self._current_task += 1
            self.cil_model.current_model.update_fc(self.num_classes())
            self.cil_model.prev_model.update_fc(self.num_classes())

            if self.config.keep_all_backgrounds:
                for i in range(self._current_task):
                    dataset = self.data_module.get_training_set_at_task_i(i)
                    self.data_module.store_bg_files(dataset.bg_files)
                print('{} background stored'.format(len(self.data_module.all_bg_files)))
            self.data_module.reload_train_dataset(use_internal_exemplar=True)

        self.data_module.build_validation_datasets()

        if dump_config:
            self.config.dump(self.work_dir / 'config.py')

    @property
    def current_task(self):
        return self._current_task

    @property
    def train_dataset(self):
        return self.data_module.train_dataset

    @property
    def val_dataset(self):
        return self.data_module.val_datasets[self._current_task]

    def num_classes(self, task_idx=-1):
        if task_idx == -1:
            return self.data_module.num_classes_per_task[self.current_task]
        else:
            return self.data_module.num_classes_per_task[task_idx]

    def train_task(self):
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.num_epochs_per_task,
                             # limit_train_batches=10,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # callbacks=[lr_monitor]
                             enable_checkpointing=False
                             )
        trainer.fit(self.cil_model, self.data_module)

    def train_cbf(self):
        # self.cil_model.current_model.freeze_backbone()
        print('Class Balance Fine-tuning. Freeze backbone: {}'.format(not self.config.cbf_train_backbone))
        cbf_dataset = self.data_module.build_cbf_dataset()
        loader = DataLoader(cbf_dataset,
                            batch_size=self.config.videos_per_gpu,
                            num_workers=self.config.workers_per_gpu,
                            pin_memory=True,
                            shuffle=True)

        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.cbf_num_epochs_per_task,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # limit_train_batches=10,
                             enable_checkpointing=False
                             )
        self.cil_model.optimizer_mode = 'cbf'
        if self.config.cbf_train_backbone:
            trainer.fit(self.cil_model, loader)
        else:
            self.cil_model.current_model.freeze_backbone()
            trainer.fit(self.cil_model, loader)
            self.cil_model.current_model.unfreeze_backbone()
        self.cil_model.optimizer_mode = 'default'

    def _resume(self):
        pass

    def train(self):
        while self._current_task < self.num_tasks:
            self.print_task_info()
            print('Start training for task {}'.format(self.current_task))
            self.train_task()

            # manage exemplar (for updating nme classifier and class balance finetuning)
            print('Create exemplar')
            class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in self.task_splits[self.current_task]]
            manager = Herding(budget_size=self.config.budget_size,
                              class_indices=class_indices,
                              cosine_distance=True,
                              storing_methods=self.config.storing_methods,
                              budget_type=self.config.budget_type)
            prediction_with_meta = self._extract_features_for_constructing_exemplar()
            exemplar_meta = manager.construct_exemplar(prediction_with_meta)
            self.data_module.build_exemplar_from_current_task(exemplar_meta)

            # train cbf (optional)
            if self._current_task > 0 and self.config.use_cbf:
                self.train_cbf()

            # testing
            self._testing(val_test='val')

            # saving model weights
            save_weight_destination = self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)
            torch.save(self.cil_model.current_model.state_dict(), save_weight_destination)
            print('save_model at:', str(save_weight_destination))

            # update model and prepare dataset for next task
            self._current_task += 1
            # 2. update current_model and prev_model
            if self._current_task < self.num_tasks:  # sanity check
                self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
                self.cil_model.prev_model.eval()
                self.cil_model.current_model.update_fc(self.num_classes())

                # update the fc layer of prev model here will simplify the training loops (load state dict)
                # In theory, prev model does not need a classifier, but the mmaction2 required a Classifier
                # for Recognizer2D class
                # this does not affect KD loss because the classifier does not contribute to KD loss
                self.cil_model.prev_model.update_fc(self.num_classes())

                # 3. prepare data for next task and update model
                self.data_module.reload_train_dataset(use_internal_exemplar=True)
            print('#####################################################################################\n')

    def print_task_info(self):
        print('Task {}, current heads: {}\n'
              'Training set size: {} (including {} samples from exemplar)'.format(self._current_task,
                                                                                  self.num_classes(),
                                                                                  len(self.data_module.train_dataset),
                                                                                  self.data_module.exemplar_size,
                                                                                  ))
        if isinstance(self.data_module.train_dataset, BackgroundMixDataset):
            print('Number of backgrounds:', len(self.data_module.train_dataset.bg_files))

    def _extract_features_for_constructing_exemplar(self):
        """
        extract features for constructing exemplar
        """
        # predictor = pl.Trainer()
        # trainer = pl.Trainer(gpus=self.config.gpu_ids,
        #                      default_root_dir=self.config.work_dir,
        #                      max_epochs=self.config.data.features_extraction_epochs,
        #                      logger=False,
        #                      strategy='dp'
        #                      )

        # loader = self.data_module.features_extraction_dataloader()
        self.cil_model.extract_repr = True  # to extract features
        self.cil_model.extract_meta = True  # and other meta data
        prediction_with_meta = []
        for _ in range(self.config.data.features_extraction_epochs):
            self.data_module.predict_dataloader_mode = 'feature_extraction'
            # pred_ = trainer.predict(model=self.cil_model, datamodule=self.data_module)
            pred_ = self.single_gpu_predict(self.data_module.predict_dataloader())

            prediction_with_meta.append(pred_)
        self.cil_model.extract_repr = False  # reset flag
        self.cil_model.extract_meta = False

        # collate data
        frame_dir = [batch_data['frame_dir'] for batch_data in prediction_with_meta[0]]
        frame_dir = [dir_name for batch_data in frame_dir for dir_name in batch_data]  # flatten list

        repr_ = []
        cls_score = []
        for i in range(self.config.data.features_extraction_epochs):
            for batch_data in prediction_with_meta[i]:
                repr_.append(batch_data['repr_'])
                cls_score.append(batch_data['cls_score'])
        repr_ = torch.cat(repr_, dim=0)
        repr_ = repr_.reshape(-1, self.config.data.features_extraction_epochs, repr_.size(1))

        cls_score = torch.cat(cls_score, dim=0)
        cls_score = cls_score.reshape(-1, self.config.data.features_extraction_epochs, cls_score.size(1))

        collated_prediction_with_meta = {
            # the data loader does not shuffle samples. So we can use the meta of the first epoch
            'frame_dir': frame_dir,
            'total_frames': torch.cat([batch_data['total_frames'] for batch_data in prediction_with_meta[0]], dim=0),
            'label': torch.cat([batch_data['label'] for batch_data in prediction_with_meta[0]], dim=0).squeeze(dim=1),
            'clip_len': torch.cat([batch_data['clip_len'] for batch_data in prediction_with_meta[0]], dim=0),
            'num_clips': torch.cat([batch_data['num_clips'] for batch_data in prediction_with_meta[0]], dim=0),
            'frame_inds': torch.cat([batch_data['frame_inds'] for batch_data in prediction_with_meta[0]], dim=0),
            'repr_': repr_,
            'cls_score': cls_score
        }
        return collated_prediction_with_meta

    def _testing(self, val_test='test', exemplar_class_means=None):
        print('Begin testing')
        # trainer = pl.Trainer(gpus=self.config.gpu_ids,
        #                      default_root_dir=self.config.work_dir,
        #                      max_epochs=self.config.data.features_extraction_epochs,
        #                      logger=False,
        #                      strategy='dp'
        #                      )
        if exemplar_class_means is not None:
            self.cil_model.extract_repr = True
        metric = torchmetrics.classification.Accuracy(num_classes=self.num_classes(), multiclass=True)

        cnn_accuracies = []
        nme_accuracies = []
        for task_idx in range(self.current_task + 1):
            self.data_module.predict_dataloader_mode = val_test
            self.data_module.predict_dataset_task_idx = task_idx

            # pred_ = trainer.predict(model=self.cil_model, datamodule=self.data_module)
            pred_ = self.single_gpu_predict(self.data_module.predict_dataloader())
            # collate data
            cls_score = []
            labels = []
            for batch_data in pred_:
                cls_score.extend(batch_data['cls_score'])
                labels.extend(batch_data['label'])
            cls_score = torch.stack(cls_score, dim=0)
            labels = torch.stack(labels, dim=0)

            preds = torch.argmax(cls_score, dim=1, keepdim=False)
            acc = metric(preds, labels)
            cnn_accuracies.append(acc.item() * 100)

            if exemplar_class_means is not None:
                repr_ = []
                for batch_data in pred_:
                    repr_.append(batch_data['repr_'])
                repr_ = torch.cat(repr_, dim=0)
                repr_ = repr_.reshape(-1, repr_.size(1))

                batch_size, dims = repr_.shape
                num_classes = exemplar_class_means.size(0)
                repr_broadcast = repr_.unsqueeze(dim=1).expand(batch_size, num_classes, dims)
                # dist.shape =  [batch_size, num_classes]
                # dist = torch.linalg.vector_norm(repr_broadcast - exemplar_class_means, ord=2, dim=2)
                # preds_nme = torch.argmin(dist, dim=1, keepdim=False)

                similarity = F.cosine_similarity(repr_broadcast, exemplar_class_means, dim=-1)
                preds_nme = torch.argmax(similarity, dim=1, keepdim=False)
                nme_acc = metric(preds_nme, labels)
                nme_accuracies.append(nme_acc.item() * 100)
        print('Accuracy across task:', cnn_accuracies)

        if exemplar_class_means is not None:
            print('Accuracy across task (nme):', nme_accuracies)
            self.cil_model.extract_repr = False     # reset flag
            return cnn_accuracies, nme_accuracies
        return cnn_accuracies

    def cil_testing(self, test_nme=False):
        tmp = self._current_task
        cnn_accuracies = []
        nme_accuracies = []
        for task_idx in range(self.num_tasks):
            self._current_task = task_idx
            self.data_module.config.data.test.ann_file = str(
                self.data_module.task_splits_ann_files['val'][self.current_task])
            self.data_module.test_datasets.append(build_dataset(self.config.data.test))

            self.cil_model.current_model.update_fc(self.num_classes())
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task))
            )
            if test_nme:
                exemplar_class_means = self._get_exemplar_class_means(task_idx)
                cnn_task_i_accuracies, nme_task_i_accuracies = self._testing(exemplar_class_means=exemplar_class_means)
                cnn_accuracies.append(cnn_task_i_accuracies)
                nme_accuracies.append(nme_task_i_accuracies)
            else:
                cnn_task_i_accuracies = self._testing()
                cnn_accuracies.append(cnn_task_i_accuracies)

        print('CNN accuracies')
        print(print_mean_accuracy(cnn_accuracies, [len(class_indices) for class_indices in self.task_splits]))
        if test_nme:
            print('NME accuracies')
            print(print_mean_accuracy(nme_accuracies, [len(class_indices) for class_indices in self.task_splits]))

        self._current_task = tmp  # reset state

    def _get_exemplar_class_means(self, task_idx):
        # load class mean from file
        exemplar_class_mean_file = self.ckpt_dir / 'exemplar_class_mean_task_{}.pt'.format(task_idx)
        if exemplar_class_mean_file.exists():
            class_means = torch.load(exemplar_class_mean_file)['class_means']

        # or extract class mean from exemplar
        else:
            print('Begin extract class mean from exemplar')
            raw_str_list = []
            for i in range(task_idx + 1):
                with open(self.data_module.exemplar_dir / 'exemplar_task_{}.txt'.format(i), 'r') as f:
                    raw_str_list.append(f.read().strip())

            raw_str = '\n'.join(raw_str_list)
            tmp_exemplars_file = self.data_module.exemplar_dir / 'tmp_exemplars.txt'
            with open(tmp_exemplars_file, 'w') as f:
                f.write(raw_str)
            # trainer = pl.Trainer(gpus=self.config.gpu_ids,
            #                      default_root_dir=self.config.work_dir,
            #                      max_epochs=1,
            #                      logger=False,
            #                      strategy='dp'
            #                      )
            self.cil_model.extract_repr = True
            self.cil_model.current_model.update_fc(self.num_classes())

            self.data_module.predict_dataloader_mode = 'feature_extraction'
            # pred_ = trainer.predict(model=self.cil_model, datamodule=self.data_module)
            pred_ = self.single_gpu_predict(self.data_module.predict_dataloader())
            repr_ = []
            for batch_data in pred_:
                repr_.append(batch_data['repr_'])
            repr_ = torch.cat(repr_, dim=0)
            repr_ = repr_.reshape(-1, repr_.size(1))

            label = torch.cat([batch_data['label'] for batch_data in pred_], dim=0).squeeze(dim=1)
            class_means = []
            for class_idx in range(self.num_classes()):
                indices = (label == class_idx).nonzero().squeeze()
                class_means.append(torch.mean(repr_[indices], dim=0))

            class_means = torch.stack(class_means, dim=0)
            torch.save({'class_means': class_means}, exemplar_class_mean_file)
        return class_means

    def single_gpu_predict(self, loader: DataLoader):
        predictions = []
        with torch.no_grad():
            self.cil_model.current_model.eval()
            device = torch.device('cuda')
            self.cil_model.to(device)
            for batch_idx, batch_data in tqdm(enumerate(loader), total=len(loader)):
                batch_data['imgs'] = batch_data['imgs'].to(device)
                batch_data['label'] = batch_data['label'].to(device)
                pred_ = self.cil_model.predict_step(batch_data, batch_idx)
                predictions.append(pred_)
        return predictions
