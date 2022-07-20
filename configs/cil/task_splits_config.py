def ceildiv(a, b):
    return -(a // -b)


"""
According to: Class-Incremental Learning for Action Recognition in Videos (https://arxiv.org/abs/2203.13611)
seed [1000, 1993, 2021] (UCF-101)
"""
class_sequence = {
    1000: [37, 97, 56, 55, 33, 84, 3, 4, 72, 59, 66, 48, 65, 91, 99, 39, 34, 22, 67, 74, 19, 35, 9, 86, 88, 63, 85,
           38, 54, 25, 57, 62, 83, 76, 6, 13, 2, 53, 8, 24, 44, 12, 100, 29, 5, 17, 15, 73, 47, 27, 46, 98, 96, 18,
           90, 75, 31, 95, 49, 43, 78, 23, 68, 16, 7, 26, 21, 50, 70, 32, 52, 11, 69, 93, 14, 79, 10, 80, 77, 81, 28,
           82, 30, 20, 41, 58, 42, 60, 36, 40, 45, 89, 0, 61, 1, 92, 94, 64, 71, 87, 51],

    1993: [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
           28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
           99, 98, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 100, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
           36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 97, 49, 57, 33],

    2021: [90, 2, 46, 4, 78, 8, 32, 22, 13, 60, 47, 80, 75, 74, 82, 56, 51, 30, 6, 35, 92, 28, 37, 84, 3, 23, 59, 98,
           61, 34, 68, 97, 45, 58, 31, 76, 72, 55, 81, 20, 43, 73, 77, 39, 69, 65, 9, 95, 27, 100, 67, 17, 71, 96, 64,
           11, 53, 89, 42, 40, 15, 83, 18, 99, 19, 36, 10, 25, 93, 41, 87, 14, 38, 79, 5, 52, 54, 50, 16, 49, 63, 48,
           66, 26, 1, 7, 33, 88, 70, 12, 24, 21, 29, 91, 62, 44, 86, 94, 0, 57, 85]
}
seed = 2021
total_classes = 101
init_task_num_classes = 51
num_classes_per_task = 10

num_tasks = ceildiv(total_classes - init_task_num_classes, num_classes_per_task) + 1

task_splits = []

for task_idx in range(num_tasks):
    class_indices = class_sequence[seed]

    if task_idx == 0:
        start, stop = 0, init_task_num_classes
    else:
        start = init_task_num_classes + (task_idx - 1) * num_classes_per_task
        stop = start + num_classes_per_task
    task_i_class_indices = class_indices[start: stop]
    print('task {}:\n{}'.format(task_idx, task_i_class_indices))
    task_splits.append(task_i_class_indices)

print(task_splits)
