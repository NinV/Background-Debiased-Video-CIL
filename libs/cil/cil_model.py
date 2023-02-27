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
from ..utils import print_mean_accuracy, build_lr_scheduler, AverageMeter


class BaseCIL(pl.LightningModule):
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
        losses = self.current_model(imgs, labels, batch_data=batch_data)  # losses = {'loss_cls': loss_cls}
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
