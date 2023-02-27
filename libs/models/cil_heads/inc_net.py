import torch
from torch import nn
import torch.nn.functional as F


class IncrementalNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(IncrementalNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def update_fc(self, nb_classes):
        new_weight = torch.empty(nb_classes, self.in_features).type_as(self.weight.data)
        nn.init.kaiming_normal_(new_weight, nonlinearity='linear')
        new_weight[:self.out_features] = self.weight.data
        self.weight = nn.Parameter(new_weight, requires_grad=True)

        new_bias = torch.empty(nb_classes).type_as(self.bias.data)
        nn.init.constant_(new_bias, 0)
        new_bias[:self.out_features] = self.bias.data
        self.bias = nn.Parameter(new_bias, requires_grad=True)

        self.out_features = nb_classes

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
