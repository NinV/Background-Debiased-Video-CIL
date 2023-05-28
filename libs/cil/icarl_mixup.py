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


import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmaction.datasets.pipelines import Compose
from mmaction.datasets import RawframeDataset
from mmaction.datasets.builder import DATASETS


@DATASETS.register_module()
class MixUpDataset(RawframeDataset):
    """
    Args:
        ann_file (str): Path to the annotation file.
        det_file (str): Path to the human box detection result file.
    """

    def __init__(self,
                 ann_file,
                 mixup_prob=0.5,
                 alpha=0.5,
                 method='framewise',        # ['framewise', 'videowise']
                 num_segments=8,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 **kwargs):

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        randAug_pipeline_configs = [
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandAugment', n=2, m=10, prob=1),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
        ]
        super().__init__(ann_file,
                         randAug_pipeline_configs,
                         data_prefix,
                         test_mode,
                         filename_tmpl,
                         with_offset,
                         multi_class,
                         num_classes,
                         start_index,
                         modality,
                         sample_by_class,
                         power,
                         dynamic_length, **kwargs)
        self.randAug_pipeline = self.pipeline
        self.mixup_prob = mixup_prob
        self.alpha = alpha
        self.method = method
        self.num_segments = num_segments

        # pipeline for generating action video
        self.video_pipeline = Compose([
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
        ])

        if self.method == 'videowise':
            num_clips = 8
        else:
            num_clips = 1
        self.scene_video_pipeline = Compose([
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_clips),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            # dict(type='Resize', scale=(224, 224), keep_ratio=False),
        ])

        self.out_pipeline = Compose([
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            # dict(type='Collect', keys=['imgs', 'label', 'foreground_ratio', 'background_label'], meta_keys=[]),
            dict(type='Collect', keys=['imgs', 'label', 'background_label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label', 'background_label'])
        ])

    def prepare_train_frames(self, idx):
        results = self._prepare_frames(idx)
        if random.random() < self.mixup_prob:
            if self.method == 'framewise':
                results = self.framewise_mixup(results)
            elif self.method == 'videowise':
                results = self.videowise_mixup(results)
            else:
                raise ValueError
        else:
            results = self.randAug_pipeline(results)

            if self.method == 'framewise':
                results['background_label'] = np.ones(self.num_segments, dtype=int) * results['label']
            else:
                results['background_label'] = results['label']
        results = self.out_pipeline(results)
        results['alpha'] = self.alpha
        return results

    def _prepare_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return results

    def videowise_mixup(self, result):
        result = self.video_pipeline(result)

        # random sample a video from database to extract background
        scene_video_index = random.randrange(len(self.video_infos))
        scene_video = self._prepare_frames(scene_video_index)
        scene_video = self.video_pipeline(scene_video)

        for t in range(self.num_segments):
            result['imgs'][t] = self.alpha * result['imgs'][t] + (1 - self.alpha) * scene_video['imgs'][0]
        result['background_label'] = scene_video['label']
        return result

    def framewise_mixup(self, result):
        result = self.video_pipeline(result)
        result['background_label'] = []
        for t in range(self.num_segments):
            scene_video_index = random.randrange(len(self.video_infos))
            scene_video = self._prepare_frames(scene_video_index)
            scene_video = self.scene_video_pipeline(scene_video)
            result['imgs'][t] = self.alpha * result['imgs'][t] + (1 - self.alpha) * scene_video['imgs'][0]
            result['background_label'].append(scene_video['label'])
        return result

    def prepare_test_frames(self, idx):  # do not use this class for testing
        raise NotImplementedError


class ICARLMixUp(ICARLModel):
    def training_step(self, batch_data, batch_idx):
        # x.shape = (batch_size, channels, T, H, W)
        imgs, targets = batch_data['imgs'], batch_data['label']
        cls_score = self.current_model(imgs, return_loss=False)
        targets = F.one_hot(targets.squeeze(dim=1), self.num_classes(self.current_task)).float()
        background_labels = batch_data['background_label']
        background_labels = F.one_hot(background_labels, self.num_classes(self.current_task)).float()
        alpha = batch_data['alpha'].view(-1, 1)
        targets = alpha * targets + (1 - alpha) * torch.mean(background_labels, dim=1)

        if self.current_task > 0:
            previous_task_num_classes = self.num_classes(self.current_task - 1)
            indices = (batch_data['label'].view(-1) < previous_task_num_classes).nonzero().squeeze(dim=1)
            if indices.nelement():
                with torch.no_grad():
                    prev_model_targets = self.prev_model(imgs[indices], return_loss=False)
                    prev_model_targets = F.softmax(prev_model_targets, dim=1)
                targets[indices] = prev_model_targets

        # self.criterion = nn.BCEWithLogitsLoss()   # original ICaRL use BCE
        # new implementation or ICaRL use CrossEntropy
        loss = -torch.sum(targets * F.log_softmax(cls_score, dim=1), dim=1)
        loss = torch.mean(loss, dim=0)

        batch_size = imgs.shape[0]
        self.log('[{}_Task_{}]{}'.format(self.training_phase, self.current_task, 'loss_cls'),
                 loss, logger=True, batch_size=batch_size)
        return loss
