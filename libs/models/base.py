from torch import nn
from torch.nn import functional as F
from mmaction.models.builder import RECOGNIZERS, build_head
from mmaction.models.recognizers import Recognizer2D


@RECOGNIZERS.register_module()
class CILRecognizer2D(Recognizer2D):
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs)

    def update_fc(self, nb_classes):
        """
        update the classifier with more classes
        """
        self.cls_head.update_fc(nb_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


@RECOGNIZERS.register_module()
class CILBGMixedRecognizer2D(CILRecognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 bg_mixed_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone,
                         cls_head=cls_head,
                         neck=neck,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)

        self.bg_mixed_head = build_head(bg_mixed_head) if bg_mixed_head else None

    def forward_train(self, imgs, labels, mixed_bg, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        mixed_bg = mixed_bg.reshape((-1,) + mixed_bg.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        x_mixed = self.extract_feat(mixed_bg)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
                x_mixed = nn.AdaptiveAvgPool2d(1)(x_mixed)
            x = x.reshape((x.shape[0], -1))
            x_mixed = x_mixed.reshape((x_mixed.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))
            x_mixed = x.reshape(x_mixed.shape + (1, 1))

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        if self.bg_mixed_head:
            mixed_cls_score = self.bg_mixed_head(x_mixed, num_segs)
            loss_bg_mixed = F.binary_cross_entropy(mixed_cls_score, cls_score.detach())

        else:
            mixed_cls_score = self.cls_head(x_mixed, num_segs)
            loss_bg_mixed = F.binary_cross_entropy_with_logits(mixed_cls_score, cls_score.detach())

        losses.update({'loss_bg_mixed': loss_bg_mixed})
        return losses
