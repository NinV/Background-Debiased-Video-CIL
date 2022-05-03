from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmaction.models.builder import LOSSES


class LSC(nn.Module):
    """
    Local Similarity Classifier (PODNet: https://arxiv.org/abs/2004.13513)
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 nb_proxies: int = 3,
                 ):
        super(LSC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nb_proxies = nb_proxies

        self.weights = nn.Parameter(torch.empty(out_features, self.nb_proxies * in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (batch_size, dims)
        Returns:
        """
        # out: batch_size, nb_proxies * out_features
        out = F.cosine_similarity(x.view(x.size(0), 1, x.size(1)),  # (batch_size, dims) -> (batch_size, 1, dims)
                                  self.weights.view(1, self.nb_proxies * self.out_features, self.in_features),
                                  dim=2)
        out = self._reduce_proxies(out)
        return out

    def _reduce_proxies(self, similarities: torch.Tensor):
        similarities_per_class = similarities.reshape(-1, self.out_features, self.nb_proxies)
        proxy_scores = F.softmax(similarities_per_class, dim=2)
        return torch.sum(proxy_scores * similarities_per_class, dim=2, keepdim=False)

    def update_fc(self, nb_classes):
        new_weight = torch.empty(nb_classes, self.nb_proxies * self.in_features).type_as(self.weights.data)
        nn.init.kaiming_normal_(new_weight, nonlinearity='linear')
        new_weight[:self.out_features] = self.weights.data
        self.weights = nn.Parameter(new_weight, requires_grad=True)
        self.out_features = nb_classes

    def __repr__(self):
        return "LocalSimilarityClassifier(in_features: {}, out_features: {}, nb_proxies: {})".format(self.in_features,
                                                                                                     self.out_features,
                                                                                                     self.nb_proxies)


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

    def forward(self, similarities: torch.Tensor, targets: torch.Tensor):
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
