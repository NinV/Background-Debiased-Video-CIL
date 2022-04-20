import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.models import build_model
from mmaction.datasets import build_dataset
from mmaction.core import OutputHook

def parse_args():
    # config_file = 'configs/cil/tsm/tsm_r34_1x1x8_25e_ucf101_rgb_task_0.py'
    # ckpt_file = 'work_dirs/tsm_r34_1x1x8_25e_ucf101_hflip_rgb_task_0/epoch_50.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('ckpt_file')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_model(cfg, ckpt_file):
    model = build_model(cfg.model)
    load_checkpoint(model, ckpt_file, map_location='cpu')
    return model


def build_train_dataset(cfg, use_val_pipeline=True):
    # copy validation pipeline config to training pipeline config
    if use_val_pipeline:
        cfg.data.train.pipeline = cfg.data.val.pipeline
    train_dataset = build_dataset(cfg.data.train)
    return train_dataset


def single_prediction(model, sample, device):
    # cnn_layer_names = ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4']
    cnn_layer_names = []

    with torch.no_grad(), OutputHook(model, cnn_layer_names + ['cls_head.avg_pool', 'cls_head'], as_tensor=True) as output_hooks:
        model(sample['imgs'].unsqueeze(dim=0).to(device), return_loss=False)

        # https://mmaction2.readthedocs.io/en/latest/_modules/mmaction/models/heads/tsm_head.html#TSMHead
        repr = output_hooks.layer_outputs['cls_head.avg_pool'].flatten(1)
        repr = repr.view(-1, model.cls_head.num_segments, repr.size(1))
        repr_consensus = model.cls_head.consensus(repr).squeeze(1)
        cls_score = output_hooks.layer_outputs['cls_head']

    return cls_score, repr_consensus    # [N, num_classes], [N, dims] (dims=512 for resnet34)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    device = torch.device(args.device)

    model = load_model(cfg, args.ckpt_file)
    model = model.to(device).eval()
    train_dataset = build_train_dataset(cfg)

    features_by_class = {}
    for i in tqdm(range(len(train_dataset))):
        # if i == 10:
        #     break
        sample = train_dataset[i]
        sample_info = train_dataset.video_infos[i].copy()
        cls_score, repr_consensus = single_prediction(model, sample, device)

        # select sample with correct prediction
        if torch.argmax(cls_score).item() == sample_info['label']:
            sample_info['cls_score'] = cls_score.tolist()
            sample_info['repr_consensus'] = repr_consensus.tolist()
            try:
                features_by_class[sample_info['label']].append(sample_info)
            except KeyError:
                features_by_class[sample_info['label']] = [sample_info]
    import json
    with open('../features.json', 'w') as f:
        json.dump(features_by_class, f)


if __name__ == '__main__':
    main()