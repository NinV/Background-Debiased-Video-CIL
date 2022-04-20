import pathlib
CLASS_INDICES = [[37, 97, 56, 55, 33, 84, 3, 4, 72, 59, 66, 48, 65, 91, 99, 39, 34, 22, 67, 74, 19, 35,
                  9, 86, 88, 63, 85, 38, 54, 25, 57, 62, 83, 76, 6, 13, 2, 53, 8, 24, 44, 12, 100, 29,
                  5, 17, 15, 73, 47, 27, 46],
                 [98, 96, 18, 90, 75, 31, 95, 49, 43, 78],
                 [23, 68, 16, 7, 26, 21, 50, 70, 32, 52],
                 [11, 69, 93, 14, 79, 10, 80, 77, 81, 28],
                 [82, 30, 20, 41, 58, 42, 60, 36, 40, 45],
                 [89, 0, 61, 1, 92, 94, 64, 71, 87, 51]]

test_split = 1
train_ann_file = pathlib.Path('data/ucf101/ucf101_train_split_{}_rawframes.txt'.format(test_split))
val_ann_file = pathlib.Path('data/ucf101/ucf101_val_split_{}_rawframes.txt'.format(test_split))
destination = pathlib.Path('data/ucf101/cil_annotations')
destination.mkdir(exist_ok=True, parents=True)
(destination / 'oracle').mkdir(exist_ok=True, parents=True)

ori_to_increment = {}
for task_i in CLASS_INDICES:
    for i in task_i:
        if i not in ori_to_increment:
            ori_to_increment[i] = len(ori_to_increment)

for file_path in [train_ann_file, val_ann_file]:
    with open(file_path, 'r') as f:
        data = f.readlines()

    annotation_ = {}
    for l in data:
        video_path, total_frames, label = l.strip().split()
        annotation_[video_path] = total_frames, int(label)

    task_i_oracle = []
    for task_i, class_indices in enumerate(CLASS_INDICES):
        class_indices = set(class_indices)
        task_i_data = []
        for video_path, (total_frames, label) in list(annotation_.items()):
            if label in class_indices:
                task_i_data.append((video_path, total_frames, ori_to_increment[label]))

        if task_i_data:
            task_i_data_str = ['{} {} {}\n'.format(video_path, total_frames, label) for video_path, total_frames, label
                               in task_i_data]
            task_i_file_path = destination / 'task_{}_{}'.format(task_i, file_path.name)
            with open(task_i_file_path, 'w') as f:
                f.writelines(task_i_data_str)
            print('create file at:', str(task_i_file_path))

            task_i_oracle.extend(task_i_data.copy())
            task_i_oracle_file_path = destination / 'oracle' / 'oracle_task_{}_{}'.format(task_i, file_path.name)
            task_i_oracle_data_str = ['{} {} {}\n'.format(video_path, total_frames, label) for video_path, total_frames, label
                               in task_i_oracle]
            with open(task_i_oracle_file_path, 'w') as f:
                f.writelines(task_i_oracle_data_str)
            print('create file at:', str(task_i_oracle_file_path))

import json
with open(destination / "class_indices_mapping.json", 'w') as f:
    json.dump(ori_to_increment, f)
print('create indice mapping file at:', str(destination / "class_indices_mapping.json"))
