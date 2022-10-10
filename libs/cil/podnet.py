import torch
import torch.nn as nn
import torch.nn.functional as F


class PODNet(nn.Module):
    def __init__(self, pod_method):
        if pod_method not in ('pixels', 'channels', 'width', 'height', 'gap', 'spatial'):
            raise ValueError("Unknown method to collapse: {}".format(pod_method))
        super(PODNet, self).__init__()
        self.pod_method = pod_method
        self.criterion = nn.MSELoss()


    def forward(self, f1, f2):
        if f1.shape != f2.shape:
            raise ValueError("input features have to be same shape")

        if len(f1.shape) == 2 or self.pod_method == 'pixels':
            return self.criterion(f1, f2)

        # f1.shape = b, c, h, w 
        if self.pod_method == 'channels':
            f1 = f1.sum(dim=1).view(f1.shape[0], -1)  # shape of (b, w * h)
            f2 = f2.sum(dim=1).view(f2.shape[0], -1)
        
        elif self.pod_method == 'width':
            f1 = f1.sum(dim=3).view(f1.shape[0], -1)  # shape of (b, c * h)
            f2 = f2.sum(dim=3).view(f2.shape[0], -1)
            
        elif self.pod_method == 'height':
            f1 = f1.sum(dim=2).view(f1.shape[0], -1)  # shape of (b, c * w)
            f2 = f2.sum(dim=2).view(f2.shape[0], -1)
            
        elif self.pod_method == 'gap':
            f1 = F.adaptive_avg_pool2d(f1, (1, 1))[..., 0, 0]
            f2 = F.adaptive_avg_pool2d(f2, (1, 1))[..., 0, 0]
            
        else:   # spatial
            f1_w = f1.sum(dim=2).view(f1.shape[0], -1)  # shape of (b, c * h)
            f2_w = f2.sum(dim=2).view(f2.shape[0], -1)
            f1_h = f1.sum(dim=3).view(f1.shape[0], -1)  # shape of (b, c * h)
            f2_h = f2.sum(dim=3).view(f2.shape[0], -1)
            f1 = torch.cat([f1_h, f1_w], dim=-1)
            f2 = torch.cat([f2_h, f2_w], dim=-1)
            
        return self.criterion(f1, f2)

