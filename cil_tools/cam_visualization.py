import os
import argparse
import pathlib
from typing import List
import json

from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
from mmcv import Config
from mmcv.utils.config import Config as mmcvConfig

from pytorch_grad_cam.utils.image import show_cam_on_image
from libs.cil.cil import BaseCIL, CILDataModule, CILTrainer
import numpy as np
from mmaction.datasets import build_dataset
from mmaction.datasets.pipelines import Compose
from libs.gradCAM.base_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with CAM')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to load models')
    parser.add_argument('-t', '--model_at_step', type=int, required=True,
                        help='select the incremental step to load model')
    parser.add_argument('-d', '--data_at_step', type=int, required=True,
                        help='select the incremental step to load dataset')
    parser.add_argument('-p', '--prefix', default='', help='model prefix used to construct CAM output file name')
    parser.add_argument('--out_dir', default='output folder for CAM_visualization')
    parser.add_argument('--method', default='GradCAM', help='select from ["GradCAM", "GradCAM++"]')
    args = parser.parse_args()
    # cfg_dict are used for updating the configurations from config file
    cfg_dict = {}
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            cfg_dict[k] = v
    return args, cfg_dict


def get_num_classes(config:mmcvConfig, task_idx: int):
    num_classes = 0
    for task in config.task_splits[:task_idx+1]:
        num_classes += len(task)
    return num_classes


def get_model(config: mmcvConfig, task_idx: int):
    cil_model = BaseCIL(config)
    num_classes = get_num_classes(config, task_idx)
    print('Build model {} at task {} ({} classes)'.format(config.prefix, task_idx, num_classes))
    cil_model.current_model.update_fc(num_classes)
    cil_model.current_model.load_state_dict(torch.load(os.path.join(config.work_dir,
                                                                    'ckpt/ckpt_task_{}.pt'.format(task_idx))))
    cil_model.current_model.eval()
    return cil_model.current_model


def build_test_datasets(config: mmcvConfig):
    test_pipeline = [
        dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=8,
            test_mode=True),
        dict(type='RawFrameDecode'),
        # dict(type='Resize', scale=(-1, 256)),
        # dict(type='CenterCrop', crop_size=224),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['frame_dir', 'imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    work_dir = pathlib.Path(config.work_dir)
    if config.data_at_step != -1:
        return build_dataset_at_step_t(config.data.test, work_dir, config.data_at_step, test_pipeline)

    num_tasks = len(config.task_splits)
    test_datasets = []
    for i in range(num_tasks):
        ds = build_dataset_at_step_t(config.data.test, work_dir, config.data_at_step, test_pipeline)
        test_datasets.append(ds)
    return merge_dataset(test_datasets)


def build_dataset_at_step_t(dataset_config: mmcvConfig, work_dir: pathlib.Path, task_idx: int, pipeline):
    ann_file = str(work_dir / 'task_splits/val_task_{}.txt'.format(task_idx))
    dataset_config.ann_file = ann_file
    dataset_config.pipeline = pipeline
    return build_dataset(dataset_config)


def merge_dataset(datasets: List):
    ds_0 = datasets[0]
    for ds in datasets[1:]:
        ds_0.video_infos.extend(ds.video_infos)
        ds_0.bg_files.extend(ds.bg_files)
    return ds_0


def _class_idx_to_name(config: mmcvConfig):
    class_names = []
    with open(os.path.join(config.data_dir, 'classInd.txt'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            class_names.append(l.strip().split()[1])

    task_splits = config.task_splits
    original_class_indices = [idx for task in task_splits for idx in task]
    cil_class_idx_to_ori = {cil_cls_idx: ori_idx for cil_cls_idx, ori_idx in enumerate(original_class_indices)}

    class_idx_to_name = [class_names[cil_class_idx_to_ori[idx]] for idx in range(len(class_names))]
    return class_idx_to_name


def get_rgb_image(ds, idx):
    sample_pipeline = Compose([
        dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=8,
            test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ])
    ds_pipeline = ds.pipeline
    ds.pipeline = sample_pipeline
    rgb_images = ds[idx]['imgs']
    rgb_images = torch.permute(rgb_images, [0, 2, 3, 1]).numpy()
    ds.pipeline = ds_pipeline
    return rgb_images


def main():
    args, cfg_dict = parse_args()
    config = Config.fromfile(args.config)
    config.merge_from_dict(cfg_dict)

    model = get_model(config, args.model_at_step)
    test_ds = build_test_datasets(config)
    class_idx_to_name = _class_idx_to_name(config)

    if args.method == "GradCAM":
        cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]], use_cuda=True)
    elif args.method == "GradCAM++":
        cam = GradCAMPlusPlus(model=model, target_layers=[model.backbone.layer4[-1]], use_cuda=True)
    else:
        raise ValueError

    for sample_idx in tqdm(range(len(test_ds))):
        # sample_idx = 1004
        sample_data = test_ds[sample_idx]
        x = sample_data['imgs']
        x = torch.unsqueeze(x, dim=0).to('cuda')
        targets = [ClassifierOutputTarget(sample_data['label'])]
        preds = torch.argmax(model(x, return_loss=False))
        preds_str = class_idx_to_name[preds.item()]
        gt_str = class_idx_to_name[sample_data['label']]

        save_dir = pathlib.Path(args.out_dir) / str(sample_idx)
        save_dir.mkdir(parents=True, exist_ok=True)

        grayscale_cam = cam(input_tensor=x, targets=targets, return_loss=False)
        rgb_images = get_rgb_image(test_ds, sample_idx)

        cam_visualization_dir = save_dir / '{}_cam_visualization'.format(args.prefix)
        cam_visualization_dir.mkdir(parents=True, exist_ok=False)
        cam_mask_dir = save_dir / '{}_cam_mask'.format(args.prefix)
        cam_mask_dir.mkdir(parents=True, exist_ok=False)
        rawframe_dir = save_dir / 'rawframes'
        rawframe_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(8):
            visualization = show_cam_on_image(rgb_images[frame_idx] / 255, grayscale_cam[frame_idx], use_rgb=True)
            cv2.imwrite(str(cam_visualization_dir / 'frame_{}.jpg'.format(frame_idx)),
                        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            cv2.imwrite(str(cam_mask_dir / 'frame_{}.png'.format(frame_idx)), grayscale_cam[frame_idx] * 255)

            rawframe_dest = rawframe_dir / 'frame_{}.jpg'.format(frame_idx)
            if not rawframe_dest.exists():
                cv2.imwrite(str(rawframe_dest),
                            cv2.cvtColor(rgb_images[frame_idx], cv2.COLOR_RGB2BGR))

        prediction_json = save_dir / 'prediction.json'
        prediction_dict = {'{}'.format(args.prefix): preds_str,
                           'label': gt_str}
        if prediction_json.exists():
            with open(prediction_json, 'r') as f:
                old_pred = json.load(f)
            prediction_dict.update(old_pred)

        with open(prediction_json, 'w') as f:
            json.dump(prediction_dict, f)


if __name__ == '__main__':
    main()
