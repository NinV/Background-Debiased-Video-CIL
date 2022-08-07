import pathlib
import random

from tqdm import tqdm
import numpy as np
import cv2
import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Normalize, Resize, RandomCrop

from mmaction.datasets import RawframeDataset
from mmaction.datasets.builder import DATASETS


@DATASETS.register_module()
class BackgroundMixDataset(RawframeDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 bg_dir,
                 check_bg_dir=True,
                 bg_image_extension='.jpg',
                 bg_size=(224, 224),
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
        self.bg_dir = pathlib.Path(bg_dir)
        self.bg_image_extension = bg_image_extension
        self.bg_dir.mkdir(exist_ok=True, parents=True)
        self.bg_pipeline = Compose([RandomCrop(bg_size),
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

        # when randAug is in the pipeline, only apply BGMix when randAug is not applied
        if self.with_randAug and result['randAug']:
            return self._mix_background(result)

        if random.random() < self.prob:
            return self._mix_background(result)
        result['bg_idx'] = -1
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
    dataset = BackgroundMixDataset(ann_file, pipelines, bg_dir='bg_extract_2', data_prefix='data/ucf101/rawframes/')
    dataset.prepare_train_frames(0)
    print()
