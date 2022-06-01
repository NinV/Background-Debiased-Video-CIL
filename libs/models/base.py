from torch import nn
from mmaction.models.builder import RECOGNIZERS
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