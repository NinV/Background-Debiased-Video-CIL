import pathlib
import os.path as osp
import random

from tqdm import tqdm
import numpy as np
import cv2
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, Resize, RandomCrop

from mmaction.datasets import BaseDataset
from mmaction.datasets.pipelines import SampleFrames

# from torch.utils.data import Dataset
from mmaction.datasets.builder import DATASETS, PIPELINES
import json


@DATASETS.register_module()
class ActivityNetDataset(BaseDataset):
    """
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self, ann_file, pipeline, data_prefix):
        super().__init__(ann_file, pipeline,
                         data_prefix=data_prefix,
                         multi_class=False,
                         num_classes=None,
                         start_index=0,
                         modality='RGB',
                         sample_by_class=False,
                         power=0,
                         dynamic_length=False
                         )

    def load_annotations(self):
        return self.load_json_annotations()


@PIPELINES.register_module()
class UntrimmedSampleFramesTSN(SampleFrames):
    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['end_frame'] - results['start_frame']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int) + results['start_frame']
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


if __name__ == '__main__':
    ann_file = 'configs/vclimb/ActivityNet_train_videos.json'
    pipelines = [
        dict(type='DecordInit'),
        dict(type='UntrimmedSampleFramesTSN', clip_len=1, frame_interval=1, num_clips=8),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]

    dataset = ActivityNetDataset(ann_file, pipelines,
                                 # bg_dir='bg_extract_2',
                                 data_prefix='/home/ninv/fiftyone/activitynet-100/train/'
                                 )
    video = dataset.prepare_train_frames(0)
    video_path = pathlib.Path(video['filename'])
    for i, frame in enumerate(video['imgs']):
        frame = torch.permute(frame, [1, 2, 0]).numpy()
        ret = cv2.imwrite('tmp_images/{}_{}.png'.format(video_path.with_suffix('').name, video['frame_inds'][i]),
                          cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(ret)
        # cv2.imshow('img', frame)
        # cv2.waitKey(0)
