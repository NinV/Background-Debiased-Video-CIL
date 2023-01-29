from typing import List
import numpy as np
import torch.optim.optimizer
from mmcv.utils.config import Config as mmcvConfig
from tabulate import tabulate


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.sizes = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.values.append(val)
        self.sizes.append(n)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_mean_accuracy(accuracies: List[AverageMeter], num_classes_per_task, floatfmt=".2f"):
    assert len(accuracies) == len(num_classes_per_task)
    num_tasks = len(num_classes_per_task)

    start = 0
    headers = ['range']
    for num_classes in num_classes_per_task:
        headers.append('{}-{}'.format(start, start + num_classes - 1))
        start += num_classes
    headers.append('Avg')
    table = []
    avg_acc = []
    for task_i in range(num_tasks):
        num_placeholders = num_tasks - task_i - 1
        table.append(['task {}'.format(task_i), *accuracies[task_i].values] + num_placeholders * [None] + [accuracies[task_i].avg])
        avg_acc.append(accuracies[task_i].avg)

    avg_acc = np.mean(avg_acc)
    table.append(['avg_acc'] + num_tasks * [None] + [avg_acc])
    return tabulate(table, headers=headers, floatfmt=[floatfmt] * 8, missingval='')


def build_lr_scheduler(optimizer, config: mmcvConfig):
    from torch.optim.lr_scheduler import MultiStepLR, StepLR, LinearLR, ExponentialLR, CosineAnnealingLR
    scheduler_types = {'StepLR': StepLR,
                       'MultiStepLR': MultiStepLR,
                       'LinearLR': LinearLR,
                       'ExponentialLR': ExponentialLR,
                       'CosineAnnealingLR': CosineAnnealingLR}
    constructor = scheduler_types[config.type]
    lr_scheduler = constructor(optimizer, **config['params'])
    return lr_scheduler
