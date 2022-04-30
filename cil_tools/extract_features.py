import argparse
import pathlib

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.models import build_model
from mmaction.datasets import build_dataset
from mmaction.core import OutputHook

import libs     # for registering some modules

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Directory contains both config file and ckpt file')
    parser.add_argument('--config_file', default='config.py')
    parser.add_argument('--ckpt_file', default='latest.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dst', default='features/out.json')
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
    root_dir = pathlib.Path(args.root_dir)

    dst = pathlib.Path((root_dir / args.dst))
    dst.parent.mkdir(exist_ok=True, parents=True)

    cfg = Config.fromfile(str(root_dir / args.config_file))
    device = torch.device(args.device)

    model = load_model(cfg, str(root_dir / args.ckpt_file))
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
    data = {'features_by_class': features_by_class,
            'model_weights': next(model.cls_head.fc_cls.parameters()).data.tolist()}
    with open(dst, 'w') as f:
        json.dump(data, f)
    print('Saved features at:', dst)


if __name__ == '__main__':
    main()
