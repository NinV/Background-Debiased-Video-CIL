import numpy as np

def ceildiv(a, b):
    return -(a // -b)

seed = 2021
num_classes = 101
init_task_num_classes = 51
num_classes_per_task = 5

# generate random sequence
np.random.seed(seed)
class_list_total = np.arange(num_classes)
np.random.shuffle(class_list_total)
total_task_list = class_list_total.tolist()


num_tasks = ceildiv(num_classes - init_task_num_classes, num_classes_per_task) + 1

task_splits = []

for task_idx in range(num_tasks):
    if task_idx == 0:
        start, stop = 0, init_task_num_classes
    else:
        start = init_task_num_classes + (task_idx - 1) * num_classes_per_task
        stop = start + num_classes_per_task
    task_i_class_indices = total_task_list[start: stop]
    print('task {}:\n{}'.format(task_idx, task_i_class_indices))
    task_splits.append(task_i_class_indices)

print(task_splits)
