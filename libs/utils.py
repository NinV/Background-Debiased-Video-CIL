import numpy as np
import torch.optim.optimizer
from mmcv.utils.config import Config as mmcvConfig
from tabulate import tabulate


def print_mean_accuracy(accuracies, num_classes_per_task, floatfmt=".2f"):
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
        avg_acc_task_i = (np.array(num_classes_per_task[:task_i + 1]) * accuracies[task_i]).sum()
        avg_acc_task_i /= np.sum(num_classes_per_task[:task_i + 1])
        num_placeholders = num_tasks - task_i - 1
        # table.append(['task {}'.format(task_i), *accuracies[task_i], ] + num_placeholders * ['-'] + [avg_acc_task_i])
        table.append(['task {}'.format(task_i), *accuracies[task_i], ] + num_placeholders * [None] + [avg_acc_task_i])
        avg_acc.append(avg_acc_task_i)

    avg_acc = np.mean(avg_acc)
    table.append(['avg_acc'] + num_tasks * [None] + [avg_acc])
    return tabulate(table, headers=headers, floatfmt=[floatfmt] * 8, missingval='')


def build_lr_scheduler(optimizer, config: mmcvConfig):
    from torch.optim.lr_scheduler import MultiStepLR, StepLR, LinearLR, ExponentialLR
    scheduler_types = {'StepLR': StepLR,
                       'MultiStepLR': MultiStepLR,
                       'LinearLR': LinearLR,
                       'ExponentialLR': ExponentialLR}
    constructor = scheduler_types[config.type]
    lr_scheduler = constructor(optimizer, **config['params'])
    return lr_scheduler


if __name__ == '__main__':
    print(print_mean_accuracy(
        accuracies=[
            [80.88003993034363],
            [72.55107164382935, 89.23512697219849],
            [68.30801367759705, 68.83852481842041, 90.04974961280823],
            [65.4269278049469, 61.18980050086975, 62.68656849861145, 91.73789024353027],
            [68.88422966003418, 62.0396614074707, 54.47761416435242, 71.22507095336914, 91.05263352394104],
            [58.19801092147827, 50.70821642875671, 53.98010015487671, 50.71225166320801, 65.78947305679321,
             88.65979313850403]
        ],
        num_classes_per_task=[51, 10, 10, 10, 10, 10],
        floatfmt=".2f"
    ))
