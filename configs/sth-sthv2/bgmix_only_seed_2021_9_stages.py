# base settings
gpu_ids = 4

# single gpu setting for traning
videos_per_gpu = 12
workers_per_gpu = 2
accumulate_grad_batches = 1

# single gpu setting for testing
testing_videos_per_gpu = 1
testing_workers_per_gpu = 2

work_dir = 'work_dirs/sth-sthv2-seed_1000'

task_splits = [
    [147, 167, 0, 133, 66, 8, 77, 45, 28, 13, 139, 72, 74, 129, 34, 121, 141, 80, 104, 52, 42, 56, 79, 132, 148, 150,
     14, 111, 22, 35, 168, 23, 149, 2, 58, 160, 112, 10, 6, 118, 30, 153, 36, 65, 76, 155, 4, 68, 154, 64, 12, 91, 73,
     170, 59, 55, 81, 43, 145, 99, 96, 92, 24, 113, 69, 15, 135, 83, 41, 130, 146, 46, 171, 97, 16, 67, 39, 29, 86, 88,
     61, 48, 37, 158],
    [125, 60, 47, 26, 166, 173, 90, 38, 161, 165],
    [98, 3, 87, 95, 20, 32, 131, 18, 107, 127],
    [126, 31, 134, 136, 75, 122, 84, 137, 143, 138],
    [103, 105, 100, 9, 51, 162, 119, 108, 27, 115],
    [117, 156, 50, 89, 17, 78, 11, 53, 40, 82],
    [19, 106, 169, 114, 25, 164, 159, 172, 71, 142],
    [151, 5, 120, 163, 123, 54, 144, 49, 63, 124],
    [110, 1, 7, 101, 33, 70, 102, 140, 152, 93],
    [21, 157, 62, 44, 94, 109, 128, 57, 85, 116]]


# select one of ['base', 'oracle', 'finetune']
methods = 'base'
starting_task = 0
ending_task = 9
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
        pretrained='https://download.pytorch.org/models/resnet50-0676ba61.pth',
        depth=50,
        norm_eval=False,
        num_segments=8,
        shift_div=8),
    cls_head=dict(
        type='IncrementalTSMHead',
        num_classes=starting_num_classes,
        in_channels=2048,
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
repr_hook = 'cls_head.avg_pool'  # extract representation
kd_exemplar_only = False
# cil optimizer and lr_scheduler
optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))

# cbf optimizer and lr_scheduler
cbf_num_epochs_per_task = 50
cbf_optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
cbf_lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))

# dataset settings
data_root = '/local_datasets/something-somethingv2/rawframes/'
train_ann_file = '/local_datasets/something-somethingv2/sthv2_train_list_rawframes.txt'
val_ann_file = '/local_datasets/something-somethingv2/sthv2_val_list_rawframes.txt'
cil_ann_file_template = '{}_task_{}.txt'  # requre exact 2 placeholders

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
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
background_dir = '/local_datasets/something-somethingv2/bg_extract'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline,
        alpha=0.5
    ),
    val=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),

    features_extraction=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=features_extraction_pipeline,
        test_mode=True),
    features_extraction_epochs=1,  # this value should be set to 1 if there's no randomness in pipeline

    exemplar=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline),
)

keep_all_backgrounds = False
cbf_full_bg = False
