import os
import pathlib
import os.path as osp
import random

from tqdm import tqdm
import numpy as np
import cv2
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, Resize, RandomCrop, RandomPerspective

from mmaction.datasets import RawframeDataset
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import RawFrameDecode

@DATASETS.register_module()
class BackgroundMixDataset(RawframeDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 bg_dir: str,
                 check_bg_dir=True,
                 bg_image_extension='.jpg',
                 bg_resize=256,
                 bg_crop_size=(224, 224),
                 bg_mean=[123.675, 116.28, 103.53],
                 bg_std=[58.395, 57.12, 57.375],
                 alpha=0.5,
                 prob=0.25,
                 with_randAug=False,
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
        super().__init__(ann_file,
                         pipeline,
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

        # extract background either from videos or from folders
        """
        https://github.com/open-mmlab/mmaction2/blob/40643bce66e78fbe525c1922329e82480f2aae0b/mmaction/datasets/base.py#L74
        mmaction2 Base class convert data_prefix using realpath which will point to the source if data_prefix is a 
        symlink. 
        background directory should also be converted to realpath to make its behaviour consistent with the mmaction2 
        data_prefix 
        """
        bg_dir = osp.realpath(bg_dir)
        self.bg_dir = pathlib.Path(bg_dir)
        self.bg_image_extension = bg_image_extension
        self.bg_dir.mkdir(exist_ok=True, parents=True)
        self.bg_pipeline = Compose([Resize(bg_resize),        # fit smaller edge
                                    RandomCrop(bg_crop_size),
                                    Normalize(bg_mean, bg_std)]
                                   )
        self.alpha = alpha
        self.prob = prob
        self.with_randAug = with_randAug
        self.bg_files = []
        if check_bg_dir:
            for idx, info in tqdm(enumerate(self.video_infos), total=len(self.video_infos),
                                  desc='check background images'):
                data_path = pathlib.Path(self.video_infos[idx]['frame_dir'])
                bg_image_file = (self.bg_dir / data_path.name).with_suffix(self.bg_image_extension)

                if not bg_image_file.exists():
                    bg_image_file = bg_extraction_tmf(data_path, bg_image_file)
                self.bg_files.append(str(bg_image_file))

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        result = super().prepare_train_frames(idx)
        result['bg_idx'] = -1

        # when randAug is in the pipeline, only apply BGMix when randAug is not applied
        if self.with_randAug:
            if not result['randAug']:
                result = self._mix_background(result)

        elif random.random() < self.prob:
            result = self._mix_background(result)

        # sanity check
        if self.with_randAug:
            if result['randAug']:
                assert result['bg_idx'] == -1
            else:
                assert result['bg_idx'] != -1
        return result

    def _mix_background(self, result):
        bg_idx = torch.randint(len(self.bg_files), (1,)).item()
        bg_img = read_image(self.bg_files[bg_idx]).float()
        bg_img = self.bg_pipeline(bg_img)
        bg_img = bg_img.view(1, bg_img.size(0), bg_img.size(1), bg_img.size(2))
        blend = result['imgs'] * (1 - self.alpha) + bg_img * self.alpha
        result['imgs'] = blend
        result['bg_idx'] = bg_idx
        return result


def bg_extraction_tmf(data_path: pathlib, dest: pathlib.Path, from_video=False):
    """
    extract background using median temporal filtering
    https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/
    """
    frames = []
    if from_video:
        raise NotImplementedError
    else:
        image_files = data_path.glob('*')
        for img_f in image_files:
            img = cv2.imread(str(img_f))
            frames.append(img)
        median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
        cv2.imwrite(str(dest),
                    median_frame)
    return median_frame


# @DATASETS.register_module()
class OmniSourceBackgroundMixDataset(RawframeDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 bg_dir: str,
                 check_bg_dir=True,
                 bg_image_extension='.jpg',
                 bg_resize=256,
                 bg_crop_size=(224, 224),
                 bg_type='type_a',
                 bg_mean=[123.675, 116.28, 103.53],
                 bg_std=[58.395, 57.12, 57.375],
                 alpha=0.5,
                 prob=0.25,
                 with_randAug=False,
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
        super().__init__(ann_file,
                         pipeline,
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

        # extract background either from videos or from folders
        """
        https://github.com/open-mmlab/mmaction2/blob/40643bce66e78fbe525c1922329e82480f2aae0b/mmaction/datasets/base.py#L74
        mmaction2 Base class convert data_prefix using realpath which will point to the source if data_prefix is a 
        symlink. 
        background directory should also be converted to realpath to make its behaviour consistent with the mmaction2 
        data_prefix 
        """
        bg_dir = osp.realpath(bg_dir)
        self.bg_dir = pathlib.Path(bg_dir)
        self.bg_dir.mkdir(exist_ok=True, parents=True)
        self.bg_pipeline = Compose([Resize(bg_resize),        # fit smaller edge
                                    RandomCrop(bg_crop_size),
                                    Normalize(bg_mean, bg_std)]
                                   )
        self.alpha = alpha
        self.prob = prob
        self.with_randAug = with_randAug
        self.bg_type = bg_type
        self.bg_image_extension = bg_image_extension
        self.bg_files = []
        if check_bg_dir:
            self.bg_files = []
            bg_file_list = list(self.bg_dir.rglob("*"))
            for bg_f in tqdm(bg_file_list, desc='check background images'):
                img = cv2.imread(str(bg_f))
                if img is not None:
                    self.bg_files.append(bg_f)
        else:
            self.bg_files = list(self.bg_dir.rglob("*"))

        self.tmp_bg_dir = pathlib.Path("out_bg_dir_ignore_nan")
        self.tmp_bg_dir.mkdir(exist_ok=True, parents=True)

    def _type_a_from_video(self):
        video = random.choice(self.video_infos)
        frame_index = random.randint(0, video['total_frames'] - 1)
        bg_img = read_image(osp.join(video['frame_dir'], self.filename_tmpl.format(frame_index))).float()
        return bg_img

    def _type_c_sim_cam_motion(self, video):
        cam_motion_pipeline = Compose([RandomPerspective(distortion_scale=0.5, p=1, fill=0)])

        transform_frames = []
        for frame_idx in range(self.start_index, video['total_frames'] + self.start_index):
            frame = read_image(osp.join(video['frame_dir'], self.filename_tmpl.format(frame_idx))).float()
            frame = cam_motion_pipeline(frame).permute(1, 2, 0).numpy()
            frame[frame==0] = np.nan
            transform_frames.append(frame)

        # median_frame = np.median(transform_frames, axis=0).astype(dtype=np.uint8)
        median_frame = np.nanmedian(transform_frames, axis=0).astype(dtype=np.uint8)
        bg_image_file = (self.tmp_bg_dir / pathlib.Path(video['frame_dir']).name).with_suffix(self.bg_image_extension)
        cv2.imwrite(str(bg_image_file), cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB))

    def _sample_from_bg_dirs(self):
        bg_idx = torch.randint(len(self.bg_files), (1,)).item()
        bg_img = read_image(str(self.bg_files[bg_idx])).float()
        return bg_img, bg_idx

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        result = super().prepare_train_frames(idx)
        result['bg_idx'] = -1

        # when randAug is in the pipeline, only apply BGMix when randAug is not applied
        if self.with_randAug:
            if not result['randAug']:
                result = self._mix_background(result)

        elif random.random() < self.prob:
            result = self._mix_background(result)

        # sanity check
        if self.with_randAug:
            if result['randAug']:
                assert result['bg_idx'] == -1
            else:
                assert result['bg_idx'] != -1
        return result

    def _mix_background(self, result):
        if self.bg_type == 'type_a':
            bg_img = self._type_a_from_video()
            bg_idx = None
        else:
            bg_img, bg_idx = self._sample_from_bg_dirs()

        bg_img = self.bg_pipeline(bg_img)
        bg_img = bg_img.view(1, bg_img.size(0), bg_img.size(1), bg_img.size(2))
        blend = result['imgs'] * (1 - self.alpha) + bg_img * self.alpha
        result['imgs'] = blend
        result['bg_idx'] = bg_idx
        return result


if __name__ == '__main__':
    ann_file = 'data/ucf101/ucf101_train_split_1_rawframes.txt'
    pipelines = [
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
    # dataset = BackgroundMixDataset(ann_file, pipelines, bg_dir='bg_extract_2', data_prefix='data/ucf101/rawframes/')
    # dataset.prepare_train_frames(0)
    # print()

    dataset = OmniSourceBackgroundMixDataset(ann_file, pipelines, bg_dir='bg_extract_2',
                                             data_prefix='data/ucf101/rawframes/',
                                             prob=2,
                                             with_randAug=False)
    dataset.prepare_train_frames(0)
    for i in tqdm(range(len(dataset.video_infos))):
        dataset._type_c_sim_cam_motion(dataset.video_infos[i])
    print()
