import copy
from typing import Optional, Union, List, Tuple, Dict
import pathlib

import torchmetrics.classification
from yacs.config import CfgNode
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mmcv.runner import build_optimizer
from mmcv.utils.config import Config as mmcvConfig
from mmaction.datasets import build_dataset, RawframeDataset
from mmaction.models import build_model
# from mmaction.core import OutputHook
from ..module_hooks import OutputHook
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

from .memory_selection import Herding


class CILDataModule(pl.LightningDataModule):
    def __init__(self, config: mmcvConfig):
        super().__init__()
        self.config = config

        self.batch_size = config.videos_per_gpu
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

        # set exist_ok=False to avoid accidentally overriding the previous experiment result
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.exemplar_dir = self.work_dir / 'exemplar'
        self.exemplar_dir.mkdir(exist_ok=True, parents=True)

        self.controller = None  # BaseCIL instance
        self.task_splits_ann_files = {'train': [], 'val': []}
        self.train_dataset = None
        self.val_datasets = []
        self.features_extraction_dataset = None
        self.exemplar_datasets = []


    @property
    def current_task(self):
        return self.controller.current_task

    @property
    def exemplar_size(self):
        size = 0
        for ex in self.exemplar_datasets:
            size += len(ex)
        return size

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

    def reload_dataset(self,
                       exemplar: Optional[Union[RawframeDataset, List[RawframeDataset]]]=None,
                       use_internal_exemplar=True):
        """
        this method should be used for reloading the train_dataset and val_dataset when moving to next task
        Note: the self.controller.current_task should be updated with new value before calling this method
        """
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        self.train_dataset = build_dataset(self.config.data.train)

        self.config.data.val.ann_file = str(self.task_splits_ann_files['val'][self.current_task])
        self.val_datasets.append(build_dataset(self.config.data.val))

        if use_internal_exemplar:
            self.train_dataset = self.merge_dataset(self.train_dataset, self.exemplar_datasets)

        elif exemplar is not None:
            self.train_dataset = self.merge_dataset(self.train_dataset, exemplar)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=True)

    # avoid override val_dataloader abstract method
    def get_val_dataloader(self, task_idx: int):
        return DataLoader(self.val_datasets[task_idx],
                          batch_size=self.batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False
                          )

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

    @staticmethod
    def merge_dataset(source: RawframeDataset, targets: Union[RawframeDataset, List[RawframeDataset]]):
        """
        copy data info from target then merge to source. This method is useful
        """
        if isinstance(targets, RawframeDataset):
            source.video_infos.extend(targets.video_infos)
        else:
            for target_ in targets:
                source.video_infos.extend(target_.video_infos)
        return source


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

    # initialization
    def configure_optimizers(self):
        optimizer = build_optimizer(self.current_model, self.config.optimizer)
        return optimizer

    @staticmethod
    def register_modules_hooks(model, modules_names: List[str]):
        output_hooks = OutputHook(model, modules_names, as_tensor=True)
        return output_hooks

    # properties
    @property
    def current_task(self):
        return self.controller.current_task

    # others
    def _extract_repr(self):
        # https://mmaction2.readthedocs.io/en/latest/_modules/mmaction/models/heads/tsm_head.html#TSMHead
        repr = self._repr_hook.get_layer_output(self._repr_module_name).flatten(1)
        repr = repr.view(-1, self.current_model.cls_head.num_segments, repr.size(1))
        repr_consensus = self.current_model.cls_head.consensus(repr).squeeze(1)
        return repr_consensus

    # forwarding, training, validation and testing
    def forward(self, x):
        return self.current_model(x)

    def training_step(self, batch_data, batch_idx):
        # x.shape = (batch_size, channels, T, H, W)
        imgs, labels = batch_data['imgs'], batch_data['label']
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
                kd_loss += kd_criterion(current_model_features, prev_model_features)
            losses['kd_loss'] = kd_loss
        else:
            losses['kd_loss'] = 0.

        loss = losses['loss_cls'] + losses['kd_loss']
        batch_size = imgs.shape[0]
        self.log('train_loss_cls', losses['loss_cls'], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch_size)
        self.log('train_loss_kd', losses['kd_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def predict_step(self, batch_data, batch_idx, dataloader_idx: Optional[int] = None):
        x = batch_data['imgs']
        cls_score = self.current_model(x, return_loss=False)
        # cls_score = torch.rand(x.size(0), self.current_model.cls_head.num_classes)
        if self.extract_repr:
            # metadata for constructing exemplar
            return {'frame_dir': batch_data['frame_dir'],
                    'total_frames': batch_data['total_frames'],
                    'label': batch_data['label'],
                    'clip_len': batch_data['clip_len'],
                    'num_clips': batch_data['num_clips'],
                    'frame_inds': batch_data['frame_inds'],
                    'cls_score': cls_score,
                    'repr': self._extract_repr()
                    # 'repr': torch.rand(x.size(0), 512)
                    }
        return {'cls_score': cls_score,
                'label': batch_data['label']
                }


class CILTrainer:
    def __init__(self, config):
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.starting_task = 0
        self.ending_task = 0
        self._current_task = self.starting_task
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = len(config.task_splits)
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        self.data_module = CILDataModule(config)
        self.data_module.controller = self
        self.data_module.generate_annotation_file()
        self.data_module.reload_dataset(exemplar=None, use_internal_exemplar=False)
        self.cil_model = BaseCIL(config)
        self.cil_model.controller = self

        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # properties
    @property
    def current_task(self):
        return self._current_task

    @property
    def train_dataset(self):
        return self.data_module.train_dataset

    @property
    def val_dataset(self):
        return self.data_module.val_dataset

    @property
    def num_classes(self):
        return self.data_module.num_classes_per_task[self.current_task]

    def train_task(self):
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.num_epochs_per_task,
                             # limit_train_batches=10,
                             accumulate_grad_batches=self.config.accumulate_grad_batches
                             )
        trainer.fit(self.cil_model, self.data_module)

    def train_cbf(self):
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
            self.train_cbf()

            # testing
            self._testing()

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
                self.cil_model.current_model.update_fc(self.num_classes)

                # update the fc layer of prev model here will simplify the training loops (load state dict)
                # In theory, prev model does not need a classifier, but the mmaction2 required a Classifier
                # for Recognizer2D class
                # this does not affect KD loss because the classifier does not contribute to KD loss
                self.cil_model.prev_model.update_fc(self.num_classes)

                # 3. prepare data for next task and update model
                self.data_module.reload_dataset(use_internal_exemplar=True)
            print('#####################################################################################\n')

    def print_task_info(self):
        print('Task {}, current heads: {}\n'
              'Training set size: {} (including {} samples from exemplar)'.format(self._current_task,
                                               self.num_classes,
                                               len(self.data_module.train_dataset),
                                               self.data_module.exemplar_size,
                                               ))

    def _extract_features_for_constructing_exemplar(self):
        """
        extract features for constructing exemplar
        """
        # predictor = pl.Trainer()
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.data.features_extraction_epochs,
                             logger=False
                             )

        loader = self.data_module.features_extraction_dataloader()
        self.cil_model.extract_repr = True  # to extract features and other meta data
        prediction_with_meta = []
        for _ in range(self.config.data.features_extraction_epochs):
            pred_ = trainer.predict(model=self.cil_model,
                                    dataloaders=loader)

            prediction_with_meta.append(pred_)
        self.cil_model.extract_repr = False  # reset flag

        # collate data
        frame_dir = [batch_data['frame_dir'] for batch_data in prediction_with_meta[0]]
        frame_dir = [dir_name for batch_data in frame_dir for dir_name in batch_data]  # flatten list

        repr = []
        cls_score = []
        for i in range(self.config.data.features_extraction_epochs):
            for batch_data in prediction_with_meta[i]:
                repr.append(batch_data['repr'])
                cls_score.append(batch_data['cls_score'])
        repr = torch.cat(repr, dim=0)
        repr = repr.reshape(-1, self.config.data.features_extraction_epochs, repr.size(1))

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
            'repr': repr,
            'cls_score': cls_score
        }
        return collated_prediction_with_meta

    def _testing(self):
        print('Begin testing')
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.data.features_extraction_epochs,
                             logger=False
                             )
        self.cil_model.extract_repr = False
        metric = torchmetrics.classification.Accuracy(num_classes=self.num_classes, multiclass=True)

        accumulate_acc = []
        for task_idx in range(self.current_task + 1):
            loader = self.data_module.get_val_dataloader(task_idx)
            pred_ = trainer.predict(model=self.cil_model,
                                    dataloaders=loader)
            # collate data
            cls_score = []
            labels = []
            for batch_data in pred_:
                cls_score.extend(batch_data['cls_score'])
                labels.extend(batch_data['label'])
            cls_score = torch.stack(cls_score, dim=0)
            labels = torch.stack(labels, dim=0)

            preds = torch.argmax(cls_score, dim=1, keepdim=False)
            accuracy = metric(preds, labels)
            accumulate_acc.append(accuracy.item())
        print('Accuracy across task:', accumulate_acc)
