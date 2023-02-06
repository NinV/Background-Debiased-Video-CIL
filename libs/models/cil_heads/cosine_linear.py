import torch
import torch.nn as nn
from torch.nn import functional as F


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
