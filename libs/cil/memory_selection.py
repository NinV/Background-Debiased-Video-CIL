from typing import List, Union, Dict
import pathlib
import random
import json, pickle
import torch
import torch.nn.functional as F


class Herding:
    """
    iCarl
    """
    def __init__(self,
                 budget_size: int,
                 class_indices: List[int],
                 cosine_distance: bool,
                 storing_methods='clips',
                 budget_type='class'):

        """
        storing_methods:
            video:  save the entire as a sample in exemplar
            clip:   divide a video into clips
            frame:  not supported yet
        """

        assert storing_methods in ['videos', 'clips', 'frames']
        assert budget_type in ['fixed', 'class']

        self.cosine_distance = cosine_distance
        self.storing_methods = storing_methods
        self.budget_type = budget_type
        self.budget_size = budget_size
        self.num_classes = len(class_indices)
        self.class_indices = class_indices

        if self.budget_type == 'fixed':
            self.num_exemplars_per_class = budget_size // self.num_classes
        else:
            self.num_exemplars_per_class = budget_size

    def construct_exemplar(self, prediction_with_meta, **kwargs):
        self._check_dimension(prediction_with_meta['repr_'], prediction_with_meta['label'])
        meta_by_class = self.split_meta_by_class(prediction_with_meta)
        exemplar_meta = {}
        with torch.no_grad():
            for classIdx, meta in meta_by_class.items():
                exemplar_meta[classIdx] = {'indices': [],
                                      'dist': []
                                      }

                features = meta['repr_']
                if self.storing_methods == 'videos':
                    if features.size(1) == 1:
                        features = features.squeeze(dim=1)
                    else:
                        # averaging all samples
                        # (videos, samples, dims) -> (videos, dims)
                        features = features.mean(1)  # TODO: may need to handle the cosine case

                elif self.storing_methods == 'clips':
                    features = features.view(-1, features.size(2), features.size(3))
                    if features.size(1) == 1:
                        features = features.squeeze(dim=1)
                    else:
                        # averaging all samples
                        # (videos, clips, samples, dims) -> (videos, clips, dims)
                        features = features.mean(2)  # TODO: may need to handle the cosine case

                        # (videos, clips, dims) -> (videos x clips, dims)
                        features = features.view(-1, features.size(-1))
                else:
                    raise NotImplementedError
                indexer = torch.arange(features.size(0))
                # features.shape = (n, dim)
                class_mean, normalized_features = self.calc_mean_features(features)

                moving_exemplar_mean = torch.zeros(1, features.size(-1))
                for n in range(1, self.num_exemplars_per_class + 1):
                    tmp_exemplar_means = moving_exemplar_mean * (n - 1) / n + normalized_features / n

                    if self.cosine_distance:
                        dist = 1 - torch.cosine_similarity(tmp_exemplar_means, class_mean, dim=1)
                    else:
                        dist = torch.pairwise_distance(tmp_exemplar_means, class_mean.squeeze(dim=0), p=2, )

                    row_index = torch.argmin(dist)
                    # sample_index = indexer[row_index]
                    moving_exemplar_mean = moving_exemplar_mean * (n - 1) / n + normalized_features[row_index] / n

                    exemplar_meta[classIdx]['indices'].append(indexer[row_index].item())
                    exemplar_meta[classIdx]['dist'].append(dist[row_index].item())

                    normalized_features = self._remove_rows(normalized_features, row_index)
                    indexer = self._remove_rows(indexer, row_index)

                exemplar_meta[classIdx]['class_mean'] = class_mean     # this class mean is calculated using fullset
        exemplar_meta = self._update_exemplar(exemplar_meta, meta_by_class)
        return exemplar_meta

    @staticmethod
    def _remove_rows(input_: torch.Tensor, removing_indices):
        selecting_indices = torch.ones(input_.shape[0], dtype=bool)
        selecting_indices[removing_indices] = False
        return input_[selecting_indices]

    def _update_exemplar(self, exemplar_meta: dict, meta_by_class: dict):
        for classIdx, meta in meta_by_class.items():
            sample_indices = exemplar_meta[classIdx]['indices']
            exemplar_meta[classIdx]['frame_dir'] = [meta['frame_dir'][i_] for i_ in sample_indices]
            exemplar_meta[classIdx]['total_frames'] = meta['total_frames'][sample_indices]
            exemplar_meta[classIdx]['label'] = meta['label'][sample_indices]
            exemplar_meta[classIdx]['clip_len'] = meta['clip_len'][sample_indices]
            exemplar_meta[classIdx]['frame_inds'] = meta['frame_inds'][sample_indices]

        return exemplar_meta

    def _check_dimension(self, all_features, labels):
        if all_features.size(0) != labels.size(0):
            raise ValueError('all_features and labels must have the same value of dim 0')

        if self.storing_methods == 'videos':
            if len(all_features.shape) != 3:
                raise ValueError('Expecting 3D features: (videos, samples, dims)')

        if self.storing_methods == 'clips':
            if len(all_features.shape) != 4:
                raise ValueError('Expecting 4D features: (videos, clips, samples, dims)')

        if self.storing_methods == 'frames':
            raise NotImplementedError('frame herding not supported yet')

    def split_meta_by_class(self, prediction_with_meta: dict):
        frame_dir = prediction_with_meta['frame_dir']
        meta_by_class = {}
        for i in self.class_indices:
            indices = (prediction_with_meta['label'] == i).nonzero(as_tuple=True)[0]
            meta_by_class[i] = {
                'frame_dir': [frame_dir[idx] for idx in indices],
                'total_frames': prediction_with_meta['total_frames'][indices],
                'label': prediction_with_meta['label'][indices],
                'clip_len': prediction_with_meta['clip_len'][indices],
                'num_clips': prediction_with_meta['num_clips'][indices],
                'frame_inds': prediction_with_meta['frame_inds'][indices],
                'repr_': prediction_with_meta['repr_'][indices],
                'cls_score': prediction_with_meta['cls_score'][indices],
            }
        return meta_by_class

    def calc_mean_features(self, features: torch.Tensor):
        """
        Args:
            features: (n , dims)
        """
        if self.cosine_distance:
            normalized_features = F.normalize(features, p=2, dim=-1)
        else:
            normalized_features = features

        mean = features.view(-1, features.size(-1)).mean(0, keepdim=True)
        if self.cosine_distance:
            return F.normalize(mean, p=2), normalized_features
        return mean, normalized_features


class PredefinedMemoryManager:
    def __init__(self,
                 exemplar_file: Union[str, pathlib.Path],
                 task_splits: List[List],
                 # strategy: str
                 ):
        self.exemplar_file = exemplar_file
        self.task_splits = task_splits
        # self.strategy = strategy

        self.ori_idx_to_inc_idx = {}
        for task in self.task_splits:
            for original_class_idx in task:
                if original_class_idx not in self.ori_idx_to_inc_idx:
                    self.ori_idx_to_inc_idx[original_class_idx] = len(self.ori_idx_to_inc_idx)

        with open(exemplar_file, 'rb') as f:
            exemplars_data_from_file = pickle.load(f)

        self.exemplar_list = [{} for _ in range(len(task_splits))]
        for task_idx, task in enumerate(task_splits):
            for original_class_idx in task:
                inc_class_idx = self.ori_idx_to_inc_idx[original_class_idx]

                # {inc_class_idx: (frame_dir, total_frames)}
                self.exemplar_list[task_idx][inc_class_idx] = exemplars_data_from_file[original_class_idx]

    def construct_exemplar(self, task_idx, **kwargs):
        exemplars = self.exemplar_list[task_idx]
        exemplar_meta = {}
        for class_idx in exemplars.keys():
            frame_dir = []
            total_frames = []
            exemplar_meta[class_idx] = {'frame_dir': frame_dir, 'total_frames': total_frames}
            records = exemplars[class_idx]['records']
            for r in records:
                frame_dir.append(r['frame_dir'])
                total_frames.append(r['total_frames'])
        return exemplar_meta


def main():
    task_splits = [[37, 97, 56, 55, 33, 84, 3, 4, 72, 59, 66, 48, 65, 91, 99, 39, 34, 22, 67, 74, 19, 35, 9, 86, 88, 63,
                    85, 38, 54, 25, 57, 62, 83, 76, 6, 13, 2, 53, 8, 24, 44, 12, 100, 29, 5, 17, 15, 73, 47, 27, 46],
                   [98, 96, 18, 90, 75, 31, 95, 49, 43, 78],
                   [23, 68, 16, 7, 26, 21, 50, 70, 32, 52],
                   [11, 69, 93, 14, 79, 10, 80, 77, 81, 28],
                   [82, 30, 20, 41, 58, 42, 60, 36, 40, 45],
                   [89, 0, 61, 1, 92, 94, 64, 71, 87, 51]]

    manager = PredefinedMemoryManager(exemplar_file='/home/ninv/MyProjects/mmaction2-cil/min_distance_set.pkl',
                                      task_splits=task_splits)

    exemplar_meta = manager.construct_exemplar(1)


if __name__ == '__main__':
    main()
