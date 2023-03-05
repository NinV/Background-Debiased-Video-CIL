import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmaction.datasets.pipelines import Compose
from mmaction.datasets import RawframeDataset
from mmaction.datasets.builder import DATASETS


@DATASETS.register_module()
class ActorCutMixDataset(RawframeDataset):
    """
    Args:
        ann_file (str): Path to the annotation file.
        det_file (str): Path to the human box detection result file.
    """
    def __init__(self,
                 ann_file,
                 det_file,
                 # randaug_pipeline,
                 # acm_pipeline,
                 acm_prob=1,
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

        # Load human detection bbox
        if det_file is not None:
            self.load_detections(det_file)
        self.acm_prob = acm_prob

        # pipeline for generating scene video
        self.scene_pipeline = Compose([
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='DetectionLoad', thres=0.4),
            dict(type='ResizeWithBox', scale=(-1, 256)),
            # dict(type='RandomResizedCropWithBox'),    # too extreme
            dict(type='FlipWithBox', flip_ratio=0.5),
            dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False),
            dict(type='ActorCutOut', fill_color=127),  # remove actor from video
        ])

        # pipeline for generating action video
        self.action_pipeline = Compose([
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='DetectionLoad', thres=0.4),
            dict(type='ResizeWithBox', scale=(-1, 256)),
            # dict(type='RandomResizedCropWithBox'),
            dict(type='FlipWithBox', flip_ratio=0.5),
            dict(type='ResizeWithBox', scale=(224, 224), keep_ratio=False),
            dict(type='BuildHumanMask'),
            dict(type='SceneCutOut', fill_color=127),  # remove scene from video
        ])

        self.out_pipeline = Compose([
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label', 'foreground_ratio', 'background_label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label', 'background_label'])
        ])

    def load_detections(self, det_file):
        """Load human detection results and merge it with self.video_infos"""
        dets = np.load(det_file, allow_pickle=True).item()
        if 'kinetics' in det_file:
            for idx in range(len(self.video_infos)):
                seq_name = self.video_infos[idx]['frame_dir'].split('/')[-1][:11]
                self.video_infos[idx]['all_detections'] = dets[seq_name]
        else:
            for idx in range(len(self.video_infos)):
                seq_name = self.video_infos[idx]['frame_dir'].split('/')[-1]
                self.video_infos[idx]['all_detections'] = dets[seq_name]

    def prepare_train_frames(self, idx):
        results = self._prepare_frames(idx)
        if random.random() < self.acm_prob:
            results = self.actor_cut_mix(results)
        else:
            results = self.randAug_pipeline(results)
            results['foreground_ratio'] = 1
            results['background_label'] = -1
        results = self.out_pipeline(results)
        return results

    def _prepare_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return results

    def actor_cut_mix(self, result):
        result = self.action_pipeline(result)

        # random sample a video from database to extract background
        scene_video_index = random.randrange(len(self.video_infos))
        scene_video = self._prepare_frames(scene_video_index)
        scene_video = self.scene_pipeline(scene_video)

        for frame_idx in range(len(result['imgs'])):
            actor_img = result['imgs'][frame_idx]
            scene_img = scene_video['imgs'][frame_idx]
            actor_mask = result['human_mask'][frame_idx]
            actor_cut_mix = actor_img * actor_mask + scene_img * (1 - actor_mask)
            result['imgs'][frame_idx] = actor_cut_mix
        # foreground_ratio = self._calc_foreground_ratio(result)
        result['foreground_ratio'] = self._calc_foreground_ratio(result)
        result['background_label'] = scene_video['label']
        return result

    @staticmethod
    def _calc_foreground_ratio(result):
        h, w = result['imgs'][0].shape[:2]
        num_segments = len(result['imgs'])

        total_area = num_segments * w * h
        foreground_area = 0
        for human_mask in result['human_mask']:
            # all human_mask channels are same
            foreground_area += human_mask[:, :, 0].sum()
        return foreground_area / total_area

    def prepare_test_frames(self, idx):     # do not use this class for testing
        raise NotImplementedError
