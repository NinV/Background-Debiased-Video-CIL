import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models.builder import LOSSES


@LOSSES.register_module()
class ACMSmoothCE(nn.Module):
    def __init__(self, alpha: float = 4):
        super().__init__()
        self.alpha = alpha

    def forward(self, cls_score: torch.Tensor, labels: torch.Tensor,
                batch_data: dict,
                num_classes: int,
                **kwargs):
        action_labels = F.one_hot(labels, num_classes=num_classes)

        # since foreground_ratio = 1 -> lambda_ = 1 -> background_label does not have any effect.
        # therefore, it's okay to change background_label from -1 to 0
        background_labels = torch.squeeze(batch_data['background_label'], dim=1)
        background_labels[background_labels == -1] = 0
        background_labels = F.one_hot(background_labels, num_classes=num_classes)

        foreground_ratio = batch_data['foreground_ratio']
        lambda_ = 1 - (1 - foreground_ratio) ** self.alpha
        y = action_labels * lambda_ + (1 - lambda_) * background_labels
        loss = torch.sum(y * F.log_softmax(cls_score, dim=1), dim=1)
        loss = torch.mean(loss, dim=0)
        return loss
