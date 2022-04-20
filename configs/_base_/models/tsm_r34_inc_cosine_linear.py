# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        depth=34,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='IncrementalTSMHead',
        inc_head_config=dict(type='CosineLinear', nb_proxy=3),
        loss_cls=dict(type='NCALoss'),
        num_classes=400,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
