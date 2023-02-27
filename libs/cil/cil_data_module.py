from typing import Optional, Union, List, Tuple, Dict, Any
import pathlib
import os.path as osp
import shutil
import copy

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from mmcv.utils.config import Config as mmcvConfig
from mmaction.datasets import build_dataset, RawframeDataset
from ..loader import BackgroundMixDataset, ActorCutMixDataset


class CILDataModule(pl.LightningDataModule):
    def __init__(self, config: mmcvConfig):
        super().__init__()
        self.config = config

        self.batch_size = config.videos_per_gpu
        self.test_batch_size = config.testing_videos_per_gpu
        self.task_splits = config.task_splits
        self.work_dir = pathlib.Path(config.work_dir)

        self.accumulate_task_size_list = []
        accumulate_task_size = 0
        for i in self.task_splits:
            accumulate_task_size += len(i)
            self.accumulate_task_size_list.append(accumulate_task_size)

        self.ori_idx_to_inc_idx = {}
        for task_i in self.task_splits:
            for i in task_i:
                if i not in self.ori_idx_to_inc_idx:
                    self.ori_idx_to_inc_idx[i] = len(self.ori_idx_to_inc_idx)

        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.exemplar_dir = self.work_dir / 'exemplar'
        self.exemplar_dir.mkdir(exist_ok=True, parents=True)

        self.controller = None  # BaseCIL instance
        self.task_splits_ann_files = {'train': [], 'val': []}
        self.train_dataset = None
        self.val_datasets = []
        self.test_datasets = []
        self.features_extraction_dataset = None
        self.exemplar_datasets = []
        self._all_bg_files = set()

        # flag to control predict_dataloader
        self.predict_dataloader_mode = 'val'  # ['val', 'test']
        self.predict_dataset_task_idx = 0

    @property
    def current_task(self):
        return self.controller.current_task

    @property
    def num_tasks(self):
        return self.controller.num_tasks

    @property
    def exemplar_size(self):
        size = 0
        for ex in self.exemplar_datasets:
            size += len(ex)
        return size

    @property
    def all_bg_files(self):
        return self._all_bg_files

    def generate_annotation_file(self) -> None:
        train_ann_file = pathlib.Path(self.config.train_ann_file)
        val_ann_file = pathlib.Path(self.config.val_ann_file)
        destination = self.work_dir / 'task_splits'
        destination.mkdir(exist_ok=True, parents=True)

        for train_val, file_path in zip(['train', 'val'], [train_ann_file, val_ann_file]):
            with open(file_path, 'r') as f:
                data = f.readlines()

            annotation_ = {}
            for l in data:
                video_path, total_frames, label = l.strip().split()
                annotation_[video_path] = total_frames, int(label)

            # task_i_oracle = []
            for task_i, class_indices in enumerate(self.task_splits):
                class_indices = set(class_indices)
                task_i_data = []
                for video_path, (total_frames, label) in list(annotation_.items()):
                    if label in class_indices:
                        task_i_data.append((video_path, total_frames, self.ori_idx_to_inc_idx[label]))

                if task_i_data:
                    task_i_data_str = ['{} {} {}\n'.format(video_path, total_frames, label) for
                                       video_path, total_frames, label
                                       in task_i_data]
                    task_i_file_path = destination / self.config.cil_ann_file_template.format(train_val, task_i)
                    with open(task_i_file_path, 'w') as f:
                        f.writelines(task_i_data_str)

                    self.task_splits_ann_files[train_val].append(task_i_file_path)
                    print('create file at:', str(task_i_file_path))

    def collect_ann_files_from_work_dir(self):
        ann_files_dir = self.work_dir / 'task_splits'
        for task_i in range(self.num_tasks):
            self.task_splits_ann_files['train'].append(
                ann_files_dir / self.config.cil_ann_file_template.format('train', task_i))

            self.task_splits_ann_files['val'].append(
                ann_files_dir / self.config.cil_ann_file_template.format('val', task_i))

    def collect_exemplar_from_work_dir(self):
        for task_idx in range(self.current_task):
            ann_file = self.exemplar_dir / 'exemplar_task_{}.txt'.format(task_idx)
            if ann_file.exists():
                exemplar_dataset = self.build_exemplar_dataset(str(ann_file))
                self.exemplar_datasets.append(exemplar_dataset)
            else:
                raise FileNotFoundError

    def build_validation_datasets(self):
        for i in range(self.num_tasks):
            self.config.data.val.ann_file = str(self.task_splits_ann_files['val'][i])
            val_ds = build_dataset(self.config.data.val)
            val_ds.test_mode = True
            self.val_datasets.append(val_ds)

    def build_cbf_dataset(self):
        dataset = build_dataset(self.config.data.train)  # TODO: create a exemplar data pipeline in config file
        dataset.video_infos = []

        if isinstance(dataset, BackgroundMixDataset):
            dataset.bg_files = []

            if self.config.keep_all_backgrounds:
                dataset = self.merge_dataset(dataset, self.exemplar_datasets)
                dataset.bg_files = list(self._all_bg_files)

            elif self.config.cbf_full_bg:
                dataset = self.merge_dataset(dataset, self.exemplar_datasets)
                all_bg_files_ = set(self.train_dataset.bg_files) | set(dataset.bg_files)
                dataset.bg_files = list(all_bg_files_)

        elif isinstance(dataset, RawframeDataset):
            dataset = self.merge_dataset(dataset, self.exemplar_datasets)

        else:
            raise NotImplementedError

        if isinstance(dataset, BackgroundMixDataset):
            print('CBF dataset built ({} videos, {} background)'.format(len(dataset), len(dataset.bg_files)))
        else:
            print('CBF dataset built ({} videos)'.format(len(dataset)))
        return dataset

    def reload_train_dataset(self,
                             exemplar: Optional[Union[RawframeDataset, List[RawframeDataset]]] = None,
                             use_internal_exemplar=True):
        """
        this method should be used for reloading the train_dataset when moving to next task
        Note: the self.controller.current_task should be updated with new value before calling this method
        """
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        self.train_dataset = build_dataset(self.config.data.train)

        # self.config.data.val.ann_file = str(self.task_splits_ann_files['val'][self.current_task])
        # self.val_datasets.append(build_dataset(self.config.data.val))

        if use_internal_exemplar:
            self.train_dataset = self.merge_dataset(self.train_dataset, self.exemplar_datasets)

        elif exemplar is not None:
            self.train_dataset = self.merge_dataset(self.train_dataset, exemplar)

        if isinstance(self.train_dataset, BackgroundMixDataset) and self.config.keep_all_backgrounds:
            self._all_bg_files.update(self.train_dataset.bg_files)
            self.train_dataset.bg_files = list(self._all_bg_files)

    def get_training_set_at_task_i(self, taskIdx):
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][taskIdx])
        dataset = build_dataset(self.config.data.train)
        self.config.data.train.ann_file = str(self.task_splits_ann_files['train'][self.current_task])
        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=True,
                          persistent_workers=True
                          )

    # avoid override val_dataloader abstract method
    def get_test_dataset(self, task_indices: Union[int, List, Tuple], val_test: str):
        assert val_test in ['val', 'test']

        if val_test == 'val':
            dataset_list = self.val_datasets
        else:
            dataset_list = self.test_datasets

        if isinstance(task_indices, int):
            return dataset_list[task_indices]

        assert len(task_indices) == 2
        starting_task, ending_task = task_indices  # ending task is inclusive
        dataset_list = dataset_list[starting_task: ending_task + 1]

        if val_test == 'val':
            cfg = copy.deepcopy(self.config.data.val)
        else:
            cfg = copy.deepcopy(self.config.data.test)

        cfg.ann_file = str(self.task_splits_ann_files['val'][starting_task])  # TODO: Need an empty ds constructor
        dataset = build_dataset(cfg)
        dataset.test_mode = True
        if len(dataset_list) > 1:
            for ds_ in dataset_list[1:]:
                dataset = self.merge_dataset(dataset, ds_)

        return dataset

    def get_val_dataloader(self, task_indices: Union[int, List, Tuple]):
        val_dataset = self.get_test_dataset(task_indices, 'val')
        return DataLoader(val_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def get_test_dataloader(self, task_indices: Union[int, List, Tuple]):
        test_dataset = self.get_test_dataset(task_indices, 'test')
        return DataLoader(test_dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.testing_workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    # def val_dataloader(self):
    #     return self.get_val_dataloader(self.current_task)

    def features_extraction_dataloader_on_train_dataset(self, task_idx: int):
        """
        extracting features for memory selection
        """
        self.config.data.features_extraction.ann_file = str(self.task_splits_ann_files['train'][task_idx])
        self.features_extraction_dataset = build_dataset(self.config.data.features_extraction)
        return DataLoader(self.features_extraction_dataset,
                          batch_size=self.batch_size,  # prediction takes less memory than training
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def features_extraction_dataloader_on_exemplar(self, task_idx: int):
        # raw_str_list = []
        # for i in range(task_idx + 1):
        #     with open(self.exemplar_dir / 'exemplar_task_{}.txt'.format(i), 'r') as f:
        #         raw_str_list.append(f.read().strip())
        #
        # raw_str = '\n'.join(raw_str_list)
        # tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        # with open(tmp_exemplars_file, 'w') as f:
        #     f.write(raw_str)
        """
        This method might be called multiple time in ddp_spawn. therefore, avoid write to file
        Solution: call method self.combine_all_exemplar_ann_files to create an annotation file which is a combination
        of all annotation files from all tasks so far
        """
        tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        cfg = copy.deepcopy(self.config.data.features_extraction)
        cfg.ann_file = str(tmp_exemplars_file)
        dataset = build_dataset(cfg)
        dataset.test_mode = True
        return DataLoader(dataset,
                          batch_size=self.test_batch_size,
                          num_workers=self.config.workers_per_gpu,
                          pin_memory=True,
                          shuffle=False,
                          persistent_workers=True
                          )

    def combine_all_exemplar_ann_files(self, task_idx: int):
        raw_str_list = []
        for i in range(task_idx + 1):
            with open(self.exemplar_dir / 'exemplar_task_{}.txt'.format(i), 'r') as f:
                raw_str_list.append(f.read().strip())

        raw_str = '\n'.join(raw_str_list)
        tmp_exemplars_file = self.exemplar_dir / 'tmp_exemplars.txt'
        with open(tmp_exemplars_file, 'w') as f:
            f.write(raw_str)

    def predict_dataloader(self):
        if self.predict_dataloader_mode == 'val':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.get_val_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'test':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.get_test_dataloader(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'feature_extraction_on_train_dataset':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.features_extraction_dataloader_on_train_dataset(self.predict_dataset_task_idx)

        elif self.predict_dataloader_mode == 'feature_extraction_on_exemplar':
            print('[predict_dataloader] mode: {}, task: {}'.format(self.predict_dataloader_mode,
                                                                   self.predict_dataset_task_idx))
            loader = self.features_extraction_dataloader_on_exemplar(self.predict_dataset_task_idx)

        else:
            raise ValueError
        print('Number of videos:', len(loader.dataset.video_infos))
        return loader

    def create_exemplar_ann_file(self, exemplar_meta: dict, task_idx=-1) -> str:
        if task_idx == -1:  # automatically infer the current task index using internal state from controller
            task_idx = self.current_task

        """
        https://github.com/open-mmlab/mmaction2/blob/40643bce66e78fbe525c1922329e82480f2aae0b/mmaction/datasets/base.py#L74
        mmaction2 Base class convert data_prefix using realpath which will point to the source if data_prefix is a 
        symlink. 
        background directory should also be converted to realpath to make its behaviour consistent with the mmaction2 
        data_prefix 
        """
        root_dir = pathlib.Path(osp.realpath(self.config.data_root))
        ann_file = self.exemplar_dir / 'exemplar_task_{}.txt'.format(task_idx)
        with open(ann_file, 'w') as f:
            for classIdx, meta in exemplar_meta.items():
                for frame_dir, total_frames in zip(meta['frame_dir'], meta['total_frames']):
                    diff = pathlib.Path(frame_dir).relative_to(root_dir.absolute())
                    f.write('{} {} {}\n'.format(str(diff), total_frames, classIdx))

        return str(ann_file)

    def build_exemplar_dataset(self, ann_file: str):
        self.config.data.exemplar.ann_file = ann_file
        return build_dataset(self.config.data.exemplar)

    def build_exemplar_from_current_task(self, exemplar_meta: dict):
        ann_file = self.create_exemplar_ann_file(exemplar_meta)
        exemplar_dataset = self.build_exemplar_dataset(ann_file)
        self.exemplar_datasets.append(exemplar_dataset)

    def merge_dataset(self, source: RawframeDataset, targets: Union[RawframeDataset, List[RawframeDataset]]):
        """
        copy data info from target then merge to source. This method is useful
        """
        if isinstance(targets, list):
            for target_ in targets:
                source = self._merge_dataset(source, target_)
        else:
            source = self._merge_dataset(source, targets)
        return source

    # @staticmethod
    def _merge_dataset(self, source: RawframeDataset, target_: Union[RawframeDataset, List[RawframeDataset]]):
        # if type(source) != type(target_):
        #     raise ValueError('source and target must be the same type ')

        if isinstance(source, BackgroundMixDataset):
            source.video_infos.extend(target_.video_infos)
            if source.merge_bg_files:
                source.bg_files.extend(target_.bg_files)
        elif isinstance(source, ActorCutMixDataset):
            source.video_infos.extend(target_.video_infos)
            source.load_detections(self.config.det_file)

        elif isinstance(source, RawframeDataset):
            source.video_infos.extend(target_.video_infos)
        else:
            raise TypeError
        return source

    def store_bg_files(self, bg_files):
        self._all_bg_files.update(bg_files)
