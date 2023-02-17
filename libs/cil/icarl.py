from typing import Optional, Union, List, Tuple, Dict, Any
import pathlib

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics.classification
from mmcv.runner import build_optimizer
from mmcv.utils.config import Config as mmcvConfig
from mmaction.models import build_model

from ..module_hooks import OutputHook
from ..utils import build_lr_scheduler, AverageMeter


class ICARLModel(pl.LightningModule):
    def __init__(self,
                 config: mmcvConfig,
                 is_train=True,
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

        self.current_model = build_model(config.model)  # current model

        if is_train:
            self.prev_model = build_model(config.model)
            self.prev_model.eval()
        else:
            self.prev_model = None

        self.extract_repr = False
        self._repr_module_name = config.repr_hook
        self._repr_hook = self.register_modules_hooks(self.current_model, [self._repr_module_name])
        self.extract_meta = False

    # initialization
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer
        lr_scheduler_config = self.config.lr_scheduler
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
        # x.shape = (batch_size, channels, T, H, W)
        imgs, targets = batch_data['imgs'], batch_data['label']
        cls_score = self.current_model(imgs, return_loss=False)
        targets = F.one_hot(targets.squeeze(dim=1), self.num_classes(self.current_task)).float()
        if self.current_task > 0:
            previous_task_num_classes = self.num_classes(self.current_task - 1)
            indices = (batch_data['label'].view(-1) < previous_task_num_classes).nonzero().squeeze(dim=1)
            if indices.nelement():
                with torch.no_grad():
                    prev_model_targets = self.prev_model(imgs[indices], return_loss=False)
                targets[indices] = prev_model_targets

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(cls_score, targets)

        batch_size = imgs.shape[0]
        self.log('[{}_Task_{}]{}'.format(self.training_phase, self.current_task, 'loss_cls'),
                 loss, logger=True, batch_size=batch_size)
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
