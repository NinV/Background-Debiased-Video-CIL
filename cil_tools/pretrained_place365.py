import argparse
import pickle

import numpy as np
from tqdm import tqdm
from os import path as osp
import torch
from mmcv import Config
from mmaction.datasets import RawframeDataset
from libs.module_hooks import OutputHook


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    return parser.parse_args()


def get_scene_mode():
    import torch
    import torchvision.models as models
    import os

    # th architecture to use
    arch = 'resnet50'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    output_hooks = OutputHook(model, ['avgpool'], as_tensor=False)
    return model, classes, output_hooks


# class SceneDatasetFromTMF
class SceneDatasetFromVideo(RawframeDataset):
    def __init__(self, ann_file, data_prefix, **kwargs):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

        pipeline_config = [
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label', 'frame_dir', 'total_frames', 'frame_inds'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]
        super().__init__(ann_file, pipeline_config, data_prefix=data_prefix, **kwargs)

    def prepare_train_frames(self, idx):
        result = super().prepare_train_frames(idx)
        # result['imgs'] = result['imgs'].squeeze(dim=0)
        return result

    def prepare_test_frames(self, idx):
        result = super().prepare_test_frames(idx)
        # result['imgs'] = result['imgs'].squeeze(dim=0)
        return result


if __name__ == '__main__':
    args = parse_args()
    config = Config.fromfile(args.config)

    model, classes, output_hooks = get_scene_mode()
    model.to('cuda')
    model.eval()

    train_ds = SceneDatasetFromVideo(
        ann_file=config.train_ann_file,
        data_prefix=config.data_root, test_mode=True)

    test_ds = SceneDatasetFromVideo(
        ann_file=config.val_ann_file,
        data_prefix=config.data_root, test_mode=True)

    all_features = {'train': {}, 'val': {}}
    for ds, ds_name in zip([train_ds, test_ds], ['train', 'val']):
        print('Extracting feature from {}_dataset'.format(ds_name))
        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(ds), total=len(ds)):
                input_img = batch_data['imgs'].to('cuda')
                model(input_img)
                features = output_hooks.get_layer_output('avgpool').squeeze()
                features = np.mean(features, axis=0, keepdims=False)
                label = batch_data['label']
                record = {
                    'frame_dir': batch_data['frame_dir'],
                    'total_frames': batch_data['total_frames'],
                    'frame_inds': batch_data['frame_inds'],
                    'features': features
                }
                try:
                    all_features[ds_name][label].append(record)
                except KeyError:
                    all_features[ds_name][label] = [record]
    with open('features_from_pretrained_place365.pkl', 'wb') as f:
        pickle.dump(all_features, f)

    # with open('features_from_pretrained_place365.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # print(data)