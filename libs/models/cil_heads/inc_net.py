import copy
import torch
from abc import ABC
from torch import nn
from .linears import SimpleLinear, SplitCosineLinear, CosineLinear


class BaseNet(nn.Module, ABC):
    def __init__(self, in_features):
        super(BaseNet, self).__init__()
        self.in_features = in_features
        self.fc = None


class IncrementalNet(BaseNet):
    def update_fc(self, nb_classes):
        fc = self._generate_fc(self.in_features, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        self.fc = fc

    def weight_align(self, increment):
        """
        Maintaining Discrimination and Fairness in Class Incremental Learning (https://arxiv.org/abs/1911.07053)
        """
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.fc.weight.data[-increment:, :] *= gamma

    @staticmethod
    def _generate_fc(in_features, out_features):
        fc = SimpleLinear(in_features, out_features)
        return fc

    def forward(self, x):
        out = self.fc(x)
        return out['logits']


class CosineIncrementalNet(BaseNet):
    def __init__(self, in_features, nb_proxy=1):
        super().__init__(in_features)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes):
        fc = self._generate_fc(self.in_features, nb_classes)
        if self.fc is not None:
            prev_out_features1 = self.fc.fc1.out_features
            fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
            fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
            fc.sigma.data = self.fc.sigma.data
        self.fc = fc

    def _generate_fc(self, in_features, out_features):
        if self.fc is None:
            fc = CosineLinear(in_features, out_features, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_features, prev_out_features, out_features - prev_out_features, self.nb_proxy)
        return fc

    def forward(self, x):
        out = self.fc(x)
        return out['logits']
