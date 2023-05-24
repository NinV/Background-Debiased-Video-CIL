from typing import Optional, Union, List, Tuple, Dict, Any
import pathlib
import random
import numpy as np
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
from .icarl import ICARLModel


class ICARLVideoMix(ICARLModel):
    def training_step(self, batch_data, batch_idx):
        prob = self.config.video_mix_prob
        alpha = self.config.video_mix_alpha,
        # x.shape = (batch_size, channels, T, H, W)
        imgs, targets = batch_data['imgs'], batch_data['label']
        targets = F.one_hot(targets.squeeze(dim=1), self.num_classes(self.current_task)).float()
        imgs, targets = tubemix(imgs, targets, alpha, prob)
        cls_score = self.current_model(imgs, return_loss=False)
        if self.current_task > 0:
            previous_task_num_classes = self.num_classes(self.current_task - 1)
            indices = (batch_data['label'].view(-1) < previous_task_num_classes).nonzero().squeeze(dim=1)
            if indices.nelement():
                with torch.no_grad():
                    prev_model_targets = self.prev_model(imgs[indices], return_loss=False)
                    prev_model_targets = F.softmax(prev_model_targets,dim=1)
                targets[indices] = prev_model_targets

        # self.criterion = nn.BCEWithLogitsLoss()   # original ICaRL use BCE
        # new implementation or ICaRL use CrossEntropy
        loss = -torch.sum(targets * F.log_softmax(cls_score, dim=1), dim=1)
        loss = torch.mean(loss, dim=0)

        batch_size = imgs.shape[0]
        self.log('[{}_Task_{}]{}'.format(self.training_phase, self.current_task, 'loss_cls'),
                 loss, logger=True, batch_size=batch_size)
        return loss


def tubemix(x, y, alpha, prob):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        lam = np.random.beta(alpha, alpha)

        bbx1, bby1, bbx2, bby2 = rand_bbox(x[:, :, 0, :, :].size(), lam)
        x[:, :, :, bbx1:bbx2, bby1:bby2] = x[batch_idx, :, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        tube_y = y * lam + y[batch_idx] * (1 - lam)
        return x, tube_y
    else:
        return x, y


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2