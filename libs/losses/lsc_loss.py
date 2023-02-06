import torch
import torch.nn as nn
from torch.nn import functional as F
from mmaction.models.builder import LOSSES


@LOSSES.register_module()
class LSCLoss(nn.Module):
    """
    loss function for Local Similarity Classifier (LSC) (PODNet: https://arxiv.org/abs/2004.13513)
    """

    def __init__(self,
                 eta=1.0,
                 margin=0.6,
                 learnable_eta=True,
                 exclude_pos_denominator=True,
                 hinge_proxynca=True,
                 class_weights=None
                 ):
        super(LSCLoss, self).__init__()
        self.margin = margin
        self.exclude_pos_denominator = exclude_pos_denominator
        self.hinge_proxynca = hinge_proxynca
        self.class_weights = class_weights
        self.learnable_eta = learnable_eta

        self.eta = nn.Parameter(torch.Tensor([eta]), requires_grad=self.learnable_eta)

    def forward(self, similarities: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Args:
            similarities: (batch_size, num_classes)
            target: (batch_size,)
        """
        if self.exclude_pos_denominator:  # NCA-specific
            similarities = self.eta * (similarities - self.margin)

            batch_size = similarities.size(0)
            similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

            disable_pos = torch.zeros_like(similarities)
            disable_pos[torch.arange(batch_size), targets] = similarities[torch.arange(len(similarities)), targets]

            numerator = similarities[torch.arange(batch_size), targets]
            denominator = similarities - disable_pos

            losses = numerator - torch.log(torch.exp(denominator).sum(-1))
            if self.class_weights is not None:
                losses = self.class_weights[targets] * losses

            losses = -losses
            if self.hinge_proxynca:
                losses = torch.clamp(losses, min=0.)

            loss = torch.mean(losses)
            return loss
        return F.cross_entropy(similarities, targets, weight=self.class_weights, reduction='mean')