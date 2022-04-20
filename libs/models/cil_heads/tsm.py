import torch
import torch.nn as nn
from torch.nn import functional as F
from mmaction.models.backbones.resnet_tsm import ResNetTSM
from mmaction.models.heads.tsm_head import TSMHead
from mmaction.models.builder import HEADS
from .inc_net import IncrementalNet, CosineIncrementalNet


inc_linear_layers = {
    'CosineLinear': CosineIncrementalNet,
    'SimpleLinear': IncrementalNet
}

@HEADS.register_module()
class IncrementalTSMHead(TSMHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 inc_head_config=dict(type='CosineLinear', nb_proxy=1),
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super(IncrementalTSMHead, self).__init__(
             num_classes,
             in_channels,
             num_segments=num_segments,
             loss_cls=loss_cls,
             spatial_type=spatial_type,
             consensus=consensus,
             dropout_ratio=dropout_ratio,
             init_std=init_std,
             is_shift=is_shift,
             temporal_pool=temporal_pool,
             **kwargs)

        self.inc_head_config = inc_head_config
        self.inc_head_config['in_features'] = in_channels

    def init_weights(self):
        # replace simple fc layer with incremental fc layer
        inc_head_config = self.inc_head_config.copy()
        head_type = inc_linear_layers[inc_head_config.pop('type')]
        self.fc_cls = head_type(**inc_head_config)
        self.fc_cls.update_fc(self.num_classes)

    def update_fc(self, nb_classes):
        if not hasattr(self.fc_cls, 'update_fc'):
            raise ValueError('Replace fc layer with incremental fc layer with "init_weights" method '
                             'before using "update_fc" method')

        self.fc_cls.update_fc(nb_classes)


class TSM(nn.Module):
    def __init__(self,
                 config,
                 is_NTCHW=True,
                 depth=34,
                 num_class=101,
                 average_clips='score'):
        super(TSM, self).__init__()

        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        self.average_clips = average_clips

        self.backbone = ResNetTSM(depth=depth)
        self.cls_head = IncrementalTSMHead(num_class, in_channels=512)
        self.repr_head = None

        # mmactions2 modules require run init_weight before using
        self.backbone.init_weights()
        self.cls_head.init_weights()

        # Pytorch-video format: (batch_size, channels, T, H, W) -> NCTHW
        # mmaction2 format: (batch_size, T, channels, H, W) -> NTCHW
        self.is_NTCHW = is_NTCHW

    def forward(self, imgs):
        # Pytorch-video format: (batch_size, channels, T, H, W)
        # mmaction2 format: (batch_size, T, channels, H, W)
        if not self.is_NTCHW:
            imgs = torch.permute(imgs, [0, 2, 1, 3, 4]).contiguous()

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)

        cls_score = self.cls_head(x, num_segs)
        assert cls_score.size(0) % batches == 0

        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score, cls_score.size()[0] // batches)

        if self.repr_head is None:
            return {'cls_score': cls_score}
        else:
            return {'cls_score': cls_score, 'repr': self.repr_head(x)}

    def extract_feat(self, imgs):
        """Extract features through a backbone.
        Args:
            imgs (torch.Tensor): The input images.
        Returns:
            torch.tensor: The extracted features.
        """
        if hasattr(self.backbone, 'features'):
            x = self.backbone.features(imgs)
        else:
            x = self.backbone(imgs)
        return x

    def average_clip(self, cls_score, num_segs=1):
        """Averaging class score over multiple clips (or test crops).

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class score.
        """
        if self.average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if self.average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_score = cls_score.mean(dim=1)
        return cls_score

    def update_fc(self, nb_classes):
        self.cls_head.update_fc(nb_classes)

        # device property on module is ambiguous
        # https://github.com/pytorch/pytorch/issues/7460
        # TODO: need test on Distributed Data Parallel setting
        self.cls_head.to(next(self.backbone.parameters()).device)
