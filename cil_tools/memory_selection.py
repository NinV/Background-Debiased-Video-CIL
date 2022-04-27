import argparse
from typing import List, Dict, Optional, Union
import pathlib

import numpy as np
import json

import torch
from torch.nn import functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('--dst', default='exemplar.json')
    parser.add_argument('--method', default='cosine')
    parser.add_argument('--budget_size', type=int, default=20)
    return parser.parse_args()


class MemoryTracker:
    def __init__(self, store_method='min'):

        assert store_method in ['max', 'min']
        self._store_max = store_method == 'max'
        if self._store_max:
            self._best_score = float('-inf')
        else:
            self._best_score = float('inf')

        self._best_sample_features = None
        self._best_sample_video_path = None
        self._best_sample_idx = -1

    def update(self, score, sample_video_path, sample_features, sample_idx):
        if (self._store_max and self._best_score < score) or \
                (not self._store_max and self._best_score > score):
            self._best_score = score
            self._best_sample_video_path = sample_video_path
            self._best_sample_features = sample_features
            self._best_sample_idx = sample_idx

    def reset_params(self):
        if self._store_max:
            self._best_score = float('-inf')
        else:
            self._best_score = float('inf')

        self._best_sample_features = None
        self._best_sample_video_path = None

    def return_best(self):
        return self._best_sample_video_path, self._best_sample_features, self._best_score, self._best_sample_idx


class DataPool:
    def __init__(self,
                 video_paths: List[Union[pathlib.Path, str]],
                 all_features: List[torch.Tensor],
                 normalized_mean=False):
        assert len(video_paths) == len(all_features)

        self.video_paths = video_paths
        self.all_features = all_features
        self.normalized_mean = normalized_mean
        # initialize stats
        self._mean_features = None

        # update stats
        if video_paths:
            self.calc_stats()

    def calc_stats(self):
        self._mean_features = calc_mean(torch.stack(self.all_features, dim=0), self.normalized_mean)

    @property
    def mean_features(self):
        return self._mean_features

    def __getitem__(self, idx):
        return self.video_paths[idx], self.all_features[idx]

    def __len__(self):
        return len(self.video_paths)


class Memory:
    def __init__(self,
                 video_paths: List[Union[pathlib.Path, str]],
                 all_features: List[torch.Tensor],
                 normalized_mean=False):
        assert len(video_paths) == len(all_features)

        self.video_paths = video_paths
        self.all_features = all_features
        self.normalized_mean = normalized_mean

        # initialize stats
        self._mean_features = None

        # update stats
        if video_paths:
            self.calc_stats()

    def calc_stats(self):
        self._mean_features = calc_mean(torch.stack(self.all_features, dim=0), self.normalized_mean)
        return self.mean_features

    def update(self, sample_video_path, sample_features) -> None:
        self.all_features.append(sample_features)
        self.video_paths.append(sample_video_path)

        if self.normalized_mean:
            sample_features = F.normalize(sample_features, p=2, dim=0)

        if self._mean_features is None:
            self._mean_features = sample_features
            return

        self.calc_stats()

    def pop_last(self):
        n = len(self.video_paths)
        if n == 0:
            return
        self.video_paths.pop()
        last_features = self.all_features.pop()
        if n == 1:
            self._mean_features = None
        else:
            self._mean_features = (n * self._mean_features - last_features) / (n - 1)

    @property
    def mean_features(self):
        return self._mean_features

    def to_json(self, write_features=False):
        return {
            'video_paths': [str(path_) for path_ in self.video_paths],
            'normalized_mean': self.normalized_mean,
            'mean': self.calc_stats().tolist()
        }

    def __getitem__(self, idx):
        return self.video_paths[idx], self.all_features[idx]

    def __len__(self):
        return len(self.video_paths)


def calc_mean(input_, normalized_mean):
    if normalized_mean:
        features = F.normalize(input_, p=2, dim=1)
    else:
        features = input_
    return torch.mean(features, dim=0)


def calc_dist(memory: Memory, data_pool: DataPool, method_index: int) -> float:
    assert memory.normalized_mean == data_pool.normalized_mean
    if method_index == 0:  # euclidean distance of mean_features
        dist = F.pairwise_distance(data_pool.mean_features, memory.mean_features, p=2)

    elif method_index == 1:
        dist = 1 - F.cosine_similarity(data_pool.mean_features, memory.mean_features, dim=0)
    else:
        raise NotImplementedError
    return dist


def greedy_memory_selection(budget_size: int, data_pool: DataPool, memory: Memory, method_index: int):
    tracker = MemoryTracker()
    sample_indices = set(range(len(data_pool)))
    best_score_history = []
    while len(memory) < budget_size:
        for idx in sample_indices:
            sample_data = data_pool[idx]

            memory.update(*sample_data)
            dist = calc_dist(memory, data_pool, method_index)
            memory.pop_last()
            tracker.update(dist, *sample_data, idx)

        best_sample_video_path, best_sample_features, best_score, best_sample_idx = tracker.return_best()
        tracker.reset_params()

        memory.update(best_sample_video_path, best_sample_features)
        sample_indices.remove(best_sample_idx)
        best_score_history.append(best_score.item())

    # print(best_score_history)

def write_exemplar(exemplar):
    pass

def main():
    args = parse_args()

    assert args.method in ['euclidean', 'cosine']
    method_dict = {'euclidean': 0,
                   'cosine': 1,
                   }

    method_index = method_dict[args.method]
    if method_index == 0:
        normalized_mean = False
    else:
        normalized_mean = True

    with open(args.data_file, 'r') as f:
        data = json.load(f)

    exemplar = {}
    for class_label, info_per_class in data.items():
        video_paths = []
        all_features = []
        for sample_info in info_per_class:
            video_paths.append(sample_info['frame_dir'])
            all_features.append(torch.Tensor(sample_info['repr_consensus'][0]))  # TODO fix extract_features function

        # all_features = torch.Tensor(all_features)
        data_pool = DataPool(video_paths, all_features, normalized_mean)
        memory = Memory([], [], normalized_mean)

        greedy_memory_selection(args.budget_size, data_pool, memory, method_index)
        exemplar[int(class_label)] = memory

    for class_label, mem in exemplar.items():
        mem_json = mem.to_json()
        exemplar[class_label] = mem_json

    exemplar['method'] = args.method
    with open(args.dst, 'w') as f:
        json.dump(exemplar, f, indent=2)


if __name__ == '__main__':
    main()
