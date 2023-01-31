from typing import Optional, Union, List, Tuple, Dict, Any
import pathlib
import os.path as osp
import shutil
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import WandbLogger
import torchmetrics.classification
from mmcv.runner import build_optimizer
from mmcv.utils.config import Config as mmcvConfig
from mmaction.datasets import build_dataset, RawframeDataset
from mmaction.models import build_model

from ..module_hooks import OutputHook
from .memory_selection import Herding
from ..utils import print_mean_accuracy, build_lr_scheduler, AverageMeter
from ..loader.comix_loader import BackgroundMixDataset


class CILDataModule(pl.LightningDataModule):
    def __init__(self, config: mmcvConfig):
        super().__init__()
        self.config = config

        self.batch_size = config.videos_per_gpu
        self.test_batch_size = config.testing_videos_per_gpu
        self.task_splits = config.task_splits
        self.work_dir = pathlib.Path(config.work_dir)

        self.accumulate_task_size_list = []
        accumulate_task_size = 0
        for i in self.task_splits:
            accumulate_task_size += len(i)
            self.accumulate_task_size_list.append(accumulate_task_size)

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

    def collect_exemplar_from_work_dir(self):
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
                          shuffle=True,
                          persistent_workers=True
                          )

    # avoid override val_dataloader abstract method
    def get_test_dataset(self, task_indices: Union[int, List, Tuple], val_test: str):
        assert val_test in ['val', 'test']

        if val_test == 'val':
            dataset_list = self.val_datasets
        else:
            dataset_list = self.test_datasets

        if isinstance(task_indices, int):
            return dataset_list[task_indices]

        assert len(task_indices) == 2
        starting_task, ending_task = task_indices  # ending task is inclusive
        dataset_list = dataset_list[starting_task: ending_task + 1]

        if val_test == 'val':
            cfg = copy.deepcopy(self.config.data.val)
        else:
            cfg = copy.deepcopy(self.config.data.test)

        cfg.ann_file = str(self.task_splits_ann_files['val'][starting_task])  # TODO: Need an empty ds constructor
        dataset = build_dataset(cfg)
        dataset.test_mode = True
        if len(dataset_list) > 1:
            for ds_ in dataset_list[1:]:
                dataset = self.merge_dataset(dataset, ds_)

        return dataset

    def get_val_dataloader(self, task_indices: Union[int, List, Tuple]):
        val_dataset = self.get_test_dataset(task_indices, 'val')
        return DataLoader(val_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def get_test_dataloader(self, task_indices: Union[int, List, Tuple]):
        test_dataset = self.get_test_dataset(task_indices, 'test')
        return DataLoader(test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    # def val_dataloader(self):
    #     return self.get_val_dataloader(self.current_task)

    def features_extraction_dataloader_on_train_dataset(self, task_idx: int):
        """
        extracting features for memory selection
        """
        self.config.data.features_extraction.ann_file = str(self.task_splits_ann_files['train'][task_idx])
        self.features_extraction_dataset = build_dataset(self.config.data.features_extraction)
        return DataLoader(self.features_extraction_dataset,
                          batch_size=self.batch_size,  # prediction takes less memory than training
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def features_extraction_dataloader_on_exemplar(self, task_idx: int):
        # raw_str_list = []
        # for i in range(task_idx + 1):
        #     with open(self.exemplar_dir / 'exemplar_task_{}.txt'.format(i), 'r') as f:
        #         raw_str_list.append(f.read().strip())
        #
        # raw_str = '\n'.join(raw_str_list)
        # tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        # with open(tmp_exemplars_file, 'w') as f:
        #     f.write(raw_str)
        """
        This method might be called multiple time in ddp_spawn. therefore, avoid write to file
        Solution: call method self.combine_all_exemplar_ann_files to create an annotation file which is a combination
        of all annotation files from all tasks so far
        """
        tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        cfg = copy.deepcopy(self.config.data.features_extraction)
        cfg.ann_file = str(tmp_exemplars_file)
        dataset = build_dataset(cfg)
        dataset.test_mode = True
        return DataLoader(dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def combine_all_exemplar_ann_files(self, task_idx: int):
        raw_str_list = []
        for i in range(task_idx + 1):
            with open(self.exemplar_dir / 'exemplar_task_{}.txt'.format(i), 'r') as f:
                raw_str_list.append(f.read().strip())

        raw_str = '\n'.join(raw_str_list)
        tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        with open(tmp_exemplars_file, 'w') as f:
            f.write(raw_str)

    def predict_dataloader(self):
        if self.predict_dataloader_mode == 'val':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.get_val_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'test':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.get_test_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'feature_extraction_on_train_dataset':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.features_extraction_dataloader_on_train_dataset(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'feature_extraction_on_exemplar':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.features_extraction_dataloader_on_exemplar(self.predict_dataset_task_idx)

        else:
            raise ValueError
        print('Number of videos:', len(loader.dataset.video_infos))
        return loader

    def create_exemplar_ann_file(self, exemplar_meta: dict, task_idx=-1) -> str:
        if task_idx == -1:  # automatically infer the current task index using internal state from controller
            task_idx = self.current_task

        """
        https://github.com/open-mmlab/mmaction2/blob/40643bce66e78fbe525c1922329e82480f2aae0b/mmaction/datasets/base.py#L74
        mmaction2 Base class convert data_prefix using realpath which will point to the source if data_prefix is a 
        symlink. 
        background directory should also be converted to realpath to make its behaviour consistent with the mmaction2 
        data_prefix 
        """
        root_dir = pathlib.Path(osp.realpath(self.config.data_root))
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
            if source.merge_bg_files:
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
        self.optimizer_mode = 'default'  # ['default', 'cbf']

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

    def num_classes(self, task_idx):
        return self.controller.num_classes(task_idx)

    @property
    def training_phase(self):
        return self.controller.training_phase

    @property
    def current_best(self):
        return self.controller.current_best

    @current_best.setter
    def current_best(self, value):
        self.controller.current_best = value

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
        losses = self.current_model(imgs, labels)  # losses = {'loss_cls': loss_cls}
        if self.use_kd and self.current_task > 0:
            total_kd_loss = 0
            kd_criterion = nn.MSELoss()
            self.prev_model.eval()
            with torch.no_grad():
                self.prev_model.forward_test(imgs)

            scale_factor = self.config.adaptive_scale_factors[self.current_task]
            for m_name, kd_weight in zip(self.kd_modules_names, self.config.kd_weight_by_module):
                current_model_features = self.current_model_kd_hooks.get_layer_output(m_name)
                prev_model_features = self.prev_model_kd_hooks.get_layer_output(m_name).detach()

                if self.config.kd_exemplar_only:
                    indices = (batch_data['label'].view(-1) < previous_task_num_classes).nonzero().squeeze()
                    if indices.nelement():
                        # kd_loss += scale_factor * kd_weight * kd_criterion(current_model_features[indices],
                        #                                                    prev_model_features[indices])
                        kd_loss = kd_criterion(current_model_features[indices], prev_model_features[indices])
                    else:
                        kd_loss = 0
                else:
                    # kd_loss += scale_factor * kd_weight * kd_criterion(current_model_features, prev_model_features)
                    kd_loss = kd_criterion(current_model_features, prev_model_features)
                losses[m_name] = kd_loss
                total_kd_loss += scale_factor * kd_weight * kd_loss
            losses['kd_loss'] = total_kd_loss
        else:
            losses['kd_loss'] = 0.

        batch_size = imgs.shape[0]
        for loss_name, loss_value in losses.items():
            # self.log('train_' + loss_name, losses[loss_name], on_step=True, on_epoch=True, prog_bar=True, logger=True,
            #          batch_size=batch_size)
            self.log('[{}_Task_{}]{}'.format(self.training_phase, self.current_task, loss_name),
                     losses[loss_name], logger=True, batch_size=batch_size)

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
            repr_ = repr_.view(batch_size, -1, embedding_size)      # (batch_size, num_crops, dim)
            repr_ = F.normalize(repr_, p=2, dim=-1)
            result['repr_'] = repr_                                 # (batch_size, num_crops, dim)
            result['mean_crops_repr_'] = torch.mean(repr_, dim=1, keepdim=False)    # (batch_size, dim)

            assert result['repr_'].size(0) == result['cls_score'].size(0)
        if self.extract_meta:
            for k, v in batch_data.items():
                if k not in ['label', 'imgs', 'blended']:
                    result[k] = v
        return result

    def validation_step(self, batch_data, batch_idx):
        x = batch_data['imgs']
        cls_score = self.current_model(x, return_loss=False)
        result = {'cls_score': cls_score,
                  'label': batch_data['label']
                  }
        return result

    def validation_epoch_end(self, validation_step_outputs):
        # collate data
        cls_score = []
        labels = []
        for batch_data in validation_step_outputs:
            cls_score.extend(batch_data['cls_score'])
            labels.extend(batch_data['label'])
        cls_score = torch.stack(cls_score, dim=0)
        labels = torch.stack(labels, dim=0)
        preds = torch.argmax(cls_score, dim=1, keepdim=False)

        metric = torchmetrics.classification.Accuracy(num_classes=self.num_classes(self.current_task), multiclass=True)
        metric.to(cls_score.device)
        cnn_accuracies = AverageMeter()
        ds_list = self.controller.data_module.val_datasets

        start = 0
        for task_idx in range(self.current_task + 1):
            num_samples = len(ds_list[task_idx])
            acc = metric(preds[start: start + num_samples], labels[start: start + num_samples])  # no shuffle
            cnn_accuracies.update(acc.item() * 100, num_samples)
            start += num_samples

        if self.current_best < cnn_accuracies.avg:
            print("Accuracy improve from {} to {}".format(self.current_best, cnn_accuracies.avg))
            self.current_best = cnn_accuracies.avg
            # saving model weights
            save_weight_destination = self.controller.ckpt_dir / 'ckpt_task_{}.pt'.format(self.current_task)
            torch.save(self.current_model.state_dict(), save_weight_destination)
            print('save_model at:', str(save_weight_destination))
        return cnn_accuracies


class CILTrainer:
    def __init__(self, config, dump_config=True):
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.starting_task = config.starting_task
        self._current_task = self.starting_task
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = min(len(config.task_splits), config.ending_task + 1)
        self.ending_task = config.ending_task  # TODO: fix this later
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        # setup data module
        self.data_module = CILDataModule(config)
        self.data_module.controller = self
        self.cil_model = BaseCIL(config)
        self.cil_model.controller = self

        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.data_module.generate_annotation_file()
        if self.starting_task == 0:
            self.data_module.reload_train_dataset(exemplar=None, use_internal_exemplar=False)

        # resume training
        else:
            self.data_module.collect_ann_files_from_work_dir()
            try:
                self.data_module.collect_exemplar_from_work_dir()
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
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))
            self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
            self.cil_model.prev_model.eval()

            # back to starting task update the classifier
            self._current_task += 1
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))

            if self.config.keep_all_backgrounds:
                for i in range(self._current_task):
                    dataset = self.data_module.get_training_set_at_task_i(i)
                    self.data_module.store_bg_files(dataset.bg_files)
                print('{} background stored'.format(len(self.data_module.all_bg_files)))
            self.data_module.reload_train_dataset(use_internal_exemplar=True)

        self.data_module.build_validation_datasets()

        # dump config
        if dump_config:
            self.config.dump(str(self.work_dir / 'config.py'))

        # select strategy based on number of gpus
        if isinstance(self.config.gpu_ids, list) and len(self.config.gpu_ids) > 1:
            self.strategy = 'ddp_spawn'
        elif isinstance(self.config.gpu_ids, int) and self.config.gpu_ids > 1:
            self.strategy = 'ddp_spawn'
        else:
            self.strategy = None

        # logger
        self.logger = WandbLogger(project='CILVideo')
        self.training_phase = None      # ['inc_step', 'cbf_step']

        if config.save_best:
            self.current_best = 0
        else:
            self.current_best = None

    @property
    def current_task(self):
        return self._current_task

    @property
    def train_dataset(self):
        return self.data_module.train_dataset

    @property
    def val_dataset(self):
        return self.data_module.val_datasets[self._current_task]

    def num_classes(self, task_idx: int):
        return self.data_module.accumulate_task_size_list[task_idx]

    def train_task(self):
        self.training_phase = 'inc_step'
        if self.config.save_best and self.current_task==0:
            val_dataloader = self.data_module.get_val_dataloader([0, self.current_task])
            self.current_best = 0  # reset for every task
        else:
            val_dataloader = None

        gradient_clip_val = None if self._current_task == 0 else 1.0
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.num_epochs_per_task,
                             # limit_train_batches=100,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # callbacks=[lr_monitor]
                             enable_checkpointing=False,
                             gradient_clip_val=gradient_clip_val,
                             strategy=self.strategy,
                             logger=self.logger,
                             log_every_n_steps=self.config.log_every_n_steps,
                             num_sanity_val_steps=0,
                             )
        trainer.fit(self.cil_model, self.data_module.train_dataloader(), val_dataloaders=val_dataloader)

    def train_cbf(self):
        self.training_phase = 'cbf_step'
        print('Class Balance Fine-tuning. Freeze backbone: {}'.format(not self.config.cbf_train_backbone))
        if self.config.save_best:
            val_dataloader = self.data_module.get_val_dataloader([0, self.current_task])
            self.current_best = 0   # reset for every task
        else:
            val_dataloader = None

        cbf_dataset = self.data_module.build_cbf_dataset()
        loader = DataLoader(cbf_dataset,
                            batch_size=self.config.videos_per_gpu,
                            num_workers=self.config.workers_per_gpu,
                            pin_memory=True,
                            shuffle=True,
                            persistent_workers=True)
        gradient_clip_val = None if self._current_task == 0 else 1.0
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.cbf_num_epochs_per_task,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # limit_train_batches=100,
                             gradient_clip_val=gradient_clip_val,
                             enable_checkpointing=False,
                             strategy=self.strategy,
                             logger=self.logger,
                             log_every_n_steps=self.config.log_every_n_steps,
                             num_sanity_val_steps=0
                             )
        self.cil_model.optimizer_mode = 'cbf'
        if self.config.cbf_train_backbone:
            trainer.fit(self.cil_model, loader, val_dataloaders=val_dataloader)
        else:
            self.cil_model.current_model.freeze_backbone()
            trainer.fit(self.cil_model, loader, val_dataloaders=val_dataloader)
            self.cil_model.current_model.unfreeze_backbone()
        self.cil_model.optimizer_mode = 'default'

    def _resume(self):
        pass

    def train(self):
        while self._current_task < self.num_tasks:
            self.print_task_info()
            print('Start training for task {}'.format(self.current_task))
            self.train_task()

            if self.config.save_best and self._current_task == 0:
                print("Load from best ckpt")
                self.cil_model.current_model.load_state_dict(
                    torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))


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

            # saving model weights
            if self.config.save_best:
                print("Load from best ckpt")
                self.cil_model.current_model.load_state_dict(
                    torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))
            else:
                print("Save last ckpt")
                save_weight_destination = self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)
                torch.save(self.cil_model.current_model.state_dict(), save_weight_destination)
                print('save_model at:', str(save_weight_destination))

            exemplar_class_means = self._get_exemplar_class_means(self.current_task, override_class_mean_ckpt=True)
            # testing
            self._testing(val_test='val', exemplar_class_means=exemplar_class_means,
                          task_indices=[0, self._current_task])

            # update model and prepare dataset for next task
            self._current_task += 1
            # 2. update current_model and prev_model
            if self._current_task < self.num_tasks:  # sanity check
                self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
                self.cil_model.prev_model.eval()
                self.cil_model.current_model.update_fc(self.num_classes(self._current_task))

                # update the fc layer of prev model here will simplify the training loops (load state dict)
                # In theory, prev model does not need a classifier, but the mmaction2 required a Classifier
                # for Recognizer2D class
                # this does not affect KD loss because the classifier does not contribute to KD loss
                self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))

                # 3. prepare data for next task and update model
                self.data_module.reload_train_dataset(use_internal_exemplar=True)
            print('#####################################################################################\n')

    def print_task_info(self):
        print('Task {}, current heads: {}\n'
              'Training set size: {} (including {} samples from exemplar)'.format(self._current_task,
                                                                                  self.num_classes(self._current_task),
                                                                                  len(self.data_module.train_dataset),
                                                                                  self.data_module.exemplar_size,
                                                                                  ))
        if isinstance(self.data_module.train_dataset, BackgroundMixDataset):
            print('Number of backgrounds:', len(self.data_module.train_dataset.bg_files))

    def _extract_features_for_constructing_exemplar(self):
        """
        extract features for constructing exemplar
        """
        self.cil_model.extract_repr = True  # to extract features
        self.cil_model.extract_meta = True  # and other meta data
        self.data_module.predict_dataloader_mode = 'feature_extraction_on_train_dataset'

        self.data_module.predict_dataset_task_idx = self.current_task
        pred_ = self.predict()

        self.cil_model.extract_repr = False  # reset flag
        self.cil_model.extract_meta = False

        # collate data
        repr_ = []
        cls_score = []
        for batch_data in pred_:
            repr_.append(batch_data['mean_crops_repr_'])
            cls_score.append(batch_data['cls_score'])
        repr_ = torch.cat(repr_, dim=0)
        repr_ = repr_.reshape(-1, self.config.data.features_extraction_epochs, repr_.size(1))

        cls_score = torch.cat(cls_score, dim=0)
        cls_score = cls_score.reshape(-1, self.config.data.features_extraction_epochs, cls_score.size(1))

        collated_prediction_with_meta = {
            'frame_dir': [frame_dir for batch_data in pred_ for frame_dir in batch_data['frame_dir']],
            'total_frames': torch.cat([batch_data['total_frames'] for batch_data in pred_], dim=0),
            'label': torch.cat([batch_data['label'] for batch_data in pred_], dim=0).squeeze(dim=1),
            'clip_len': torch.cat([batch_data['clip_len'] for batch_data in pred_], dim=0),
            'num_clips': torch.cat([batch_data['num_clips'] for batch_data in pred_], dim=0),
            'frame_inds': torch.cat([batch_data['frame_inds'] for batch_data in pred_], dim=0),
            'repr_': repr_,
            'cls_score': cls_score
        }
        return collated_prediction_with_meta

    def _testing(self, task_indices: Union[List[int], Tuple[int]], val_test='test', exemplar_class_means=None):
        assert len(task_indices) == 2
        print('Begin testing')
        if exemplar_class_means is not None:
            self.cil_model.extract_repr = True

        metric = torchmetrics.classification.Accuracy(num_classes=self.num_classes(task_indices[-1]), multiclass=True)
        self.data_module.predict_dataloader_mode = val_test
        self.data_module.predict_dataset_task_idx = task_indices
        pred_ = self.predict()

        # collate data
        cls_score = []
        labels = []
        for batch_data in pred_:
            cls_score.extend(batch_data['cls_score'])
            labels.extend(batch_data['label'])
        cls_score = torch.stack(cls_score, dim=0)
        labels = torch.stack(labels, dim=0)
        preds = torch.argmax(cls_score, dim=1, keepdim=False)

        cnn_accuracies = AverageMeter()
        nme_accuracies = AverageMeter()
        if val_test == 'val':
            ds_list = self.data_module.val_datasets
        else:
            ds_list = self.data_module.val_datasets

        start = 0
        for task_idx in range(self.current_task + 1):
            num_samples = len(ds_list[task_idx])
            acc = metric(preds[start: start + num_samples], labels[start: start + num_samples])  # no shuffle
            cnn_accuracies.update(acc.item() * 100, num_samples)
            start += num_samples

        if exemplar_class_means is not None:
            repr_ = []
            for batch_data in pred_:
                repr_.append(batch_data['repr_'])
            repr_ = torch.cat(repr_, dim=0)             # (num_samples, num_crops, dim)
            num_samples, num_crops, dim = repr_.size(0), repr_.size(1), repr_.size(2),
            repr_ = repr_.reshape(-1, repr_.size(2))    # (num_samples * num_crops, dim)

            num_classes = exemplar_class_means.size(0)

            # cosine distance
            size, dims = repr_.shape        # size = num_samples * num_crops
            repr_broadcast = repr_.unsqueeze(dim=1).expand(size, num_classes, dims)
            similarity = F.cosine_similarity(repr_broadcast, exemplar_class_means, dim=-1)
            similarity = torch.mean(similarity.reshape(num_samples, num_crops, num_classes), dim=1, keepdim=False)
            preds_nme = torch.argmax(similarity, dim=1, keepdim=False)

            # Euclidean distance
            # dist = torch.cdist(repr_, exemplar_class_means)        # num_samples * num_crops, num_classes
            # dist = torch.mean(dist.reshape(num_samples, num_crops, num_classes), dim=1, keepdim=False)
            # preds_nme = torch.argmin(dist, dim=1, keepdim=False)

            start = 0
            for task_idx in range(self.current_task + 1):
                num_samples = len(ds_list[task_idx])
                nme_acc = metric(preds_nme[start: start + num_samples], labels[start: start + num_samples])
                nme_accuracies.update(nme_acc.item() * 100, num_samples)
                start += num_samples
        print('Task {} Accuracies (CNN): {}\nAvg Accuracy (CNN): {}'.format(self.current_task,
                                                                            cnn_accuracies.values,
                                                                            cnn_accuracies.avg))

        if exemplar_class_means is not None:
            print('Task {} Accuracies (NME): {}\nAvg Accuracy (NME): {}'.format(self.current_task,
                                                                                nme_accuracies.values,
                                                                                nme_accuracies.avg))
            self.cil_model.extract_repr = False  # reset flag
            return cnn_accuracies, nme_accuracies
        return cnn_accuracies

    def cil_testing(self, test_nme=False):
        tmp = self._current_task
        cnn_accuracies = []
        nme_accuracies = []

        print('Build test dataset')
        for task_idx in range(self.num_tasks):
            self.data_module.config.data.test.ann_file = str(
                self.data_module.task_splits_ann_files['val'][task_idx])
            self.data_module.test_datasets.append(build_dataset(self.config.data.test))

        for task_idx in range(self.num_tasks):
            self._current_task = task_idx
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task))
            )
            if test_nme:
                exemplar_class_means = self._get_exemplar_class_means(task_idx, override_class_mean_ckpt=False)
                cnn_task_i_accuracies, nme_task_i_accuracies = self._testing(exemplar_class_means=exemplar_class_means,
                                                                             task_indices=[0, task_idx])
                cnn_accuracies.append(cnn_task_i_accuracies)
                nme_accuracies.append(nme_task_i_accuracies)
            else:
                cnn_task_i_accuracies = self._testing(task_indices=[0, task_idx])
                cnn_accuracies.append(cnn_task_i_accuracies)

        print('CNN accuracies')
        cnn_pretty_table = print_mean_accuracy(cnn_accuracies, [len(class_indices) for class_indices in
                                                                self.task_splits[
                                                                self.starting_task: self.ending_task + 1]])
        print(cnn_pretty_table)
        with open(self.work_dir / 'cnn_result.txt', 'w') as f:
            f.write('CNN Accuracies' + cnn_pretty_table + '\n')
        if test_nme:
            print('NME accuracies')
            nme_pretty_table = print_mean_accuracy(nme_accuracies, [len(class_indices) for class_indices in
                                                                    self.task_splits[
                                                                    self.starting_task: self.ending_task + 1]])
            print(nme_pretty_table)
            with open(self.work_dir / 'nme_result.txt', 'w') as f:
                f.write('NME Accuracies' + nme_pretty_table + '\n')

        self._current_task = tmp  # reset state

    def single_ckpt_testing(self, ckpt_file: str, test_nme=True):
        print("Load ckpt from", ckpt_file)
        self.cil_model.load_state_dict(torch.load(ckpt_file, map_location='cuda:0')['state_dict'])

        if test_nme:
            print('Create exemplar')
            class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in self.task_splits[self.current_task]]
            manager = Herding(budget_size=self.config.budget_size,
                              class_indices=class_indices,
                              cosine_distance=True,
                              storing_methods=self.config.storing_methods,
                              budget_type=self.config.budget_type)
            prediction_with_meta = self._extract_features_for_constructing_exemplar()
            exemplar_meta = manager.construct_exemplar(prediction_with_meta)

            exemplar_class_means = [exemplar_meta[class_idx]['class_mean'] for class_idx in range(len(exemplar_meta))]
            exemplar_class_means = torch.stack(exemplar_class_means, dim=0).squeeze(1)
        else:
            exemplar_class_means = None

        # build test datasets
        for task_idx in range(len(self.config.task_splits)):
            self._current_task = task_idx
            self.data_module.config.data.test.ann_file = str(
                self.data_module.task_splits_ann_files['val'][self.current_task])
            self.data_module.test_datasets.append(build_dataset(self.config.data.test))
        self._current_task = self.ending_task
        self._testing(val_test='test', exemplar_class_means=exemplar_class_means, task_indices=[0, self._current_task])

    def _get_exemplar_class_means(self, task_idx: int, override_class_mean_ckpt=False):
        # load class mean from file
        exemplar_class_mean_file = self.ckpt_dir / 'exemplar_class_mean_task_{}.pt'.format(task_idx)
        if not override_class_mean_ckpt and exemplar_class_mean_file.exists():
            print('Load class means (exemplar) from:', exemplar_class_mean_file)
            class_means = torch.load(exemplar_class_mean_file)['class_means']

        # or extract class mean from exemplar
        else:
            print('Begin extract class mean from exemplar')
            self.cil_model.extract_repr = True
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))

            self.data_module.combine_all_exemplar_ann_files(task_idx)  # prevent error from multiprocessing
            self.data_module.predict_dataloader_mode = 'feature_extraction_on_exemplar'
            self.data_module.predict_dataset_task_idx = task_idx
            pred_ = self.predict()
            repr_ = []
            for batch_data in pred_:
                repr_.append(batch_data['mean_crops_repr_'])
            repr_ = torch.cat(repr_, dim=0)
            repr_ = repr_.reshape(-1, repr_.size(1))

            label = torch.cat([batch_data['label'] for batch_data in pred_], dim=0).squeeze(dim=1)
            class_means = []
            for class_idx in range(self.num_classes(task_idx)):
                indices = (label == class_idx).nonzero().squeeze(dim=1)
                class_means.append(torch.mean(repr_[indices], dim=0))

            class_means = torch.stack(class_means, dim=0)
            torch.save({'class_means': class_means}, exemplar_class_mean_file)
        return class_means

    def predict(self):
        writer_tmp_dir = self.work_dir / 'tmp'
        writer_tmp_dir.mkdir(exist_ok=True)
        predict_writer = PredictWriter(writer_tmp_dir, write_interval='epoch')
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=1,
                             logger=False,
                             enable_checkpointing=False,
                             strategy=None,
                             callbacks=[predict_writer],
                             # limit_predict_batches=10
                             )
        trainer.predict(self.cil_model, datamodule=self.data_module)
        predictions = []
        for f in writer_tmp_dir.glob('predictions_rank_*'):
            predictions.extend(torch.load(f))

        print("Number of predictions:", len(predictions))
        print("removing", writer_tmp_dir)
        shutil.rmtree(writer_tmp_dir)
        return predictions


class PredictWriter(BasePredictionWriter):
    def __init__(self, output_dir: pathlib.Path, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
            self, trainer, pl_module: pl.LightningModule, predictions: List[Any], batch_indices: List[Any]
    ):
        save_location = self.output_dir / "predictions_rank_{}.pt".format(trainer.global_rank)
        torch.save(predictions[0], save_location)
        print("Predictions saved at", save_location)
