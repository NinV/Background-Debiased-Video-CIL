# base settings
gpu_ids = [0]

# 8 gpus setting
# videos_per_gpu = 12   # 8 gpus x 12 videos (single gpu setting)
# workers_per_gpu = 2   # 8 gpu setting

# single gpu setting for traning
videos_per_gpu = 48
workers_per_gpu = 4
accumulate_grad_batches = 2

# single gpu setting for testing
testing_videos_per_gpu = 1
testing_workers_per_gpu = 2

work_dir = 'work_dirs/bgmix_seed_1000_inc_class_10'


task_splits = [[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17,
                50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37],
               [95, 14, 71, 96, 99, 98, 2, 64, 66, 42],
               [22, 35, 86, 24, 34, 87, 21, 100, 0, 88],
               [27, 18, 94, 11, 12, 47, 25, 30, 46, 62],
               [69, 36, 61, 7, 63, 75, 5, 32, 4, 51],
               [48, 73, 93, 39, 67, 29, 97, 49, 57, 33]]


# select one of ['base', 'oracle', 'finetune']
methods = 'base'
starting_task = 0
ending_task = 5
use_nme_classifier = False
use_cbf = False
cbf_train_backbone = False
budget_size = 5
storing_methods = 'videos'
budget_type = 'class'
num_epochs_per_task = 50

# mmaction2 model config
# model settings
starting_num_classes = len(task_splits[0])
model = dict(
    type='CILBGMixedRecognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        depth=34,
        norm_eval=False,
        num_segments=8,
        shift_div=8),
    cls_head=dict(
        type='IncrementalTSMHead',
        num_classes=starting_num_classes,
        in_channels=512,
        inc_head_config=dict(type='LocalSimilarityClassifier',
                             out_features=starting_num_classes,
                             nb_proxies=1),
        num_segments=8,
        loss_cls=dict(type='LSCLoss'),
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
    ),
    # model training and testing settings
    prob=0.25,
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

kd_modules_names = ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4', 'cls_head.avg_pool']
repr_hook = 'cls_head.avg_pool'     # extract representation
kd_exemplar_only = False
# cil optimizer and lr_scheduler
optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructorImprovised',
    paramwise_cfg=dict(fc_lr_scale_factor=5.0),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# lr_scheduler = dict()
lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))
# lr_config = dict(policy='step', step=[20, 30])

# cbf optimizer and lr_scheduler
cbf_num_epochs_per_task = 50
cbf_optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructorImprovised',
    paramwise_cfg=dict(fc_lr_scale_factor=1.0),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
cbf_lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))

# dataset settings
data_root = 'data/ucf101/rawframes/'
test_split = 1
train_ann_file = 'data/ucf101/ucf101_train_split_{}_rawframes.txt'.format(test_split)
val_ann_file = 'data/ucf101/ucf101_val_split_{}_rawframes.txt'.format(test_split)
cil_ann_file_template = '{}_task_{}.txt'        # requre exact 2 placeholders

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandAugment', n=2, m=10),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='ThreeCrop', crop_size=256),
    dict(type='TenCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# for extracting features to construct exemplar set
# this pipeline should be similar to validation pipeline if only run one epoch
# In case we ran multiple epochs, this pipeline should be similar to train pipeline
features_extraction_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

dataset_type = 'BackgroundMixDataset'
background_dir = 'bg_extract'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file='',                    # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline,
        alpha=0.5
    ),
    val=dict(
        type=dataset_type,
        ann_file='',                    # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file='',                    # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),

    features_extraction=dict(
        type=dataset_type,
        ann_file='',                    # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=features_extraction_pipeline,
        test_mode=True),
    features_extraction_epochs=1,      # this value should be set to 1 if there's no randomness in pipeline

    exemplar=dict(
        type=dataset_type,
        ann_file='',                    # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline),
)

keep_all_backgrounds = False
cbf_full_bg = False
