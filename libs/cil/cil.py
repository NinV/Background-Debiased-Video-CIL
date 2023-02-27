from typing import Optional, Union, List, Tuple, Dict, Any
import pathlib
import shutil

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import WandbLogger
import torchmetrics.classification
from mmcv.utils.config import Config as mmcvConfig
from mmaction.datasets import build_dataset

from .cil_model import BaseCIL
from .cil_data_module import CILDataModule
from .memory_selection import Herding
from .icarl import ICARLModel
from ..utils import print_mean_accuracy, AverageMeter
from ..loader import BackgroundMixDataset


class CILTrainer:
    def __init__(self, config: mmcvConfig, dump_config=True):
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.starting_task = config.starting_task
        self._current_task = self.starting_task
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = min(len(config.task_splits), config.ending_task + 1)
        self.ending_task = config.ending_task  # TODO: fix this later
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        # setup data module
        self.data_module = CILDataModule(config)
        self.data_module.controller = self
        if config.methods == 'base':
            self.cil_model = BaseCIL(config)
        elif config.methods == 'icarl':
            self.cil_model = ICARLModel(config)
        else:
            raise ValueError
        self.cil_model.controller = self

        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.data_module.generate_annotation_file()
        if self.starting_task == 0:
            self.data_module.reload_train_dataset(exemplar=None, use_internal_exemplar=False)

        # resume training
        else:
            self.data_module.collect_ann_files_from_work_dir()
            try:
                self.data_module.collect_exemplar_from_work_dir()
            except FileNotFoundError:
                for i in range(len(self.data_module.exemplar_datasets), self.starting_task):
                    self._current_task = i
                    print('Create exemplar for task {}'.format(i))
                    class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in
                                     self.task_splits[self.current_task]]
                    manager = Herding(budget_size=self.config.budget_size,
                                      class_indices=class_indices,
                                      cosine_distance=True,
                                      storing_methods=self.config.storing_methods,
                                      budget_type=self.config.budget_type)
                    prediction_with_meta = self._extract_features_for_constructing_exemplar()
                    exemplar_meta = manager.construct_exemplar(prediction_with_meta)
                    self.data_module.build_exemplar_from_current_task(exemplar_meta)
                self._current_task = self.starting_task

            # roll back to previous task_idx to load weights
            self._current_task -= 1
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))
            self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
            self.cil_model.prev_model.eval()

            # back to starting task update the classifier
            self._current_task += 1
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))

            if self.config.keep_all_backgrounds:
                for i in range(self._current_task):
                    dataset = self.data_module.get_training_set_at_task_i(i)
                    self.data_module.store_bg_files(dataset.bg_files)
                print('{} background stored'.format(len(self.data_module.all_bg_files)))
            self.data_module.reload_train_dataset(use_internal_exemplar=True)

        self.data_module.build_validation_datasets()

        # dump config
        if dump_config:
            self.config.dump(str(self.work_dir / 'config.py'))

        # select strategy based on number of gpus
        if isinstance(self.config.gpu_ids, list) and len(self.config.gpu_ids) > 1:
            self.strategy = 'ddp_spawn'
        elif isinstance(self.config.gpu_ids, int) and self.config.gpu_ids > 1:
            self.strategy = 'ddp_spawn'
        else:
            self.strategy = None

        # logger
        self.logger = WandbLogger(project='CILVideo')
        self.training_phase = None      # ['inc_step', 'cbf_step']

        if config.save_best:
            self.current_best = 0
        else:
            self.current_best = None

    @property
    def current_task(self):
        return self._current_task

    @property
    def train_dataset(self):
        return self.data_module.train_dataset

    @property
    def val_dataset(self):
        return self.data_module.val_datasets[self._current_task]

    def num_classes(self, task_idx: int):
        return self.data_module.accumulate_task_size_list[task_idx]

    def train_task(self):
        self.training_phase = 'inc_step'
        if self.config.save_best and self.current_task==0:
            val_dataloader = self.data_module.get_val_dataloader([0, self.current_task])
            self.current_best = 0  # reset for every task
        else:
            val_dataloader = None

        gradient_clip_val = None if self._current_task == 0 else 1.0
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.num_epochs_per_task,
                             # limit_train_batches=100,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # callbacks=[lr_monitor]
                             enable_checkpointing=False,
                             gradient_clip_val=gradient_clip_val,
                             strategy=self.strategy,
                             logger=self.logger,
                             log_every_n_steps=self.config.log_every_n_steps,
                             num_sanity_val_steps=0,
                             )
        trainer.fit(self.cil_model, self.data_module.train_dataloader(), val_dataloaders=val_dataloader)

    def train_cbf(self):
        self.training_phase = 'cbf_step'
        print('Class Balance Fine-tuning. Freeze backbone: {}'.format(not self.config.cbf_train_backbone))
        if self.config.save_best:
            val_dataloader = self.data_module.get_val_dataloader([0, self.current_task])
            self.current_best = 0   # reset for every task
        else:
            val_dataloader = None

        cbf_dataset = self.data_module.build_cbf_dataset()
        loader = DataLoader(cbf_dataset,
                            batch_size=self.config.videos_per_gpu,
                            num_workers=self.config.workers_per_gpu,
                            pin_memory=True,
                            shuffle=True,
                            persistent_workers=True)
        gradient_clip_val = None if self._current_task == 0 else 1.0
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=self.config.cbf_num_epochs_per_task,
                             accumulate_grad_batches=self.config.accumulate_grad_batches,
                             # limit_train_batches=100,
                             gradient_clip_val=gradient_clip_val,
                             enable_checkpointing=False,
                             strategy=self.strategy,
                             logger=self.logger,
                             log_every_n_steps=self.config.log_every_n_steps,
                             num_sanity_val_steps=0
                             )
        self.cil_model.optimizer_mode = 'cbf'
        if self.config.cbf_train_backbone:
            trainer.fit(self.cil_model, loader, val_dataloaders=val_dataloader)
        else:
            self.cil_model.current_model.freeze_backbone()
            trainer.fit(self.cil_model, loader, val_dataloaders=val_dataloader)
            self.cil_model.current_model.unfreeze_backbone()
        self.cil_model.optimizer_mode = 'default'

    def _resume(self):
        pass

    def train(self):
        while self._current_task < self.num_tasks:
            self.print_task_info()
            print('Start training for task {}'.format(self.current_task))
            self.train_task()

            if self.config.save_best and self._current_task == 0:
                print("Load from best ckpt")
                self.cil_model.current_model.load_state_dict(
                    torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))


            # manage exemplar (for updating nme classifier and class balance finetuning)
            print('Create exemplar')
            class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in self.task_splits[self.current_task]]
            manager = Herding(budget_size=self.config.budget_size,
                              class_indices=class_indices,
                              cosine_distance=True,
                              storing_methods=self.config.storing_methods,
                              budget_type=self.config.budget_type)
            prediction_with_meta = self._extract_features_for_constructing_exemplar()
            exemplar_meta = manager.construct_exemplar(prediction_with_meta)
            self.data_module.build_exemplar_from_current_task(exemplar_meta)

            # train cbf (optional)
            if self._current_task > 0 and self.config.use_cbf:
                self.train_cbf()

            # saving model weights
            if self.config.save_best:
                print("Load from best ckpt")
                self.cil_model.current_model.load_state_dict(
                    torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)))
            else:
                print("Save last ckpt")
                save_weight_destination = self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task)
                torch.save(self.cil_model.current_model.state_dict(), save_weight_destination)
                print('save_model at:', str(save_weight_destination))

            exemplar_class_means = self._get_exemplar_class_means(self.current_task, override_class_mean_ckpt=True)
            # testing
            self._testing(val_test='val', exemplar_class_means=exemplar_class_means,
                          task_indices=[0, self._current_task])

            # update model and prepare dataset for next task
            self._current_task += 1
            # 2. update current_model and prev_model
            if self._current_task < self.num_tasks:  # sanity check
                self.cil_model.prev_model.load_state_dict(self.cil_model.current_model.state_dict())
                self.cil_model.prev_model.eval()
                self.cil_model.current_model.update_fc(self.num_classes(self._current_task))

                # update the fc layer of prev model here will simplify the training loops (load state dict)
                # In theory, prev model does not need a classifier, but the mmaction2 required a Classifier
                # for Recognizer2D class
                # this does not affect KD loss because the classifier does not contribute to KD loss
                self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))

                # 3. prepare data for next task and update model
                self.data_module.reload_train_dataset(use_internal_exemplar=True)
            print('#####################################################################################\n')

    def print_task_info(self):
        print('Task {}, current heads: {}\n'
              'Training set size: {} (including {} samples from exemplar)'.format(self._current_task,
                                                                                  self.num_classes(self._current_task),
                                                                                  len(self.data_module.train_dataset),
                                                                                  self.data_module.exemplar_size,
                                                                                  ))
        if isinstance(self.data_module.train_dataset, BackgroundMixDataset):
            print('Number of backgrounds:', len(self.data_module.train_dataset.bg_files))

    def _extract_features_for_constructing_exemplar(self):
        """
        extract features for constructing exemplar
        """
        self.cil_model.extract_repr = True  # to extract features
        self.cil_model.extract_meta = True  # and other meta data
        self.data_module.predict_dataloader_mode = 'feature_extraction_on_train_dataset'

        self.data_module.predict_dataset_task_idx = self.current_task
        pred_ = self.predict()

        self.cil_model.extract_repr = False  # reset flag
        self.cil_model.extract_meta = False

        # collate data
        repr_ = []
        cls_score = []
        for batch_data in pred_:
            repr_.append(batch_data['mean_crops_repr_'])
            cls_score.append(batch_data['cls_score'])
        repr_ = torch.cat(repr_, dim=0)
        repr_ = repr_.reshape(-1, self.config.data.features_extraction_epochs, repr_.size(1))

        cls_score = torch.cat(cls_score, dim=0)
        cls_score = cls_score.reshape(-1, self.config.data.features_extraction_epochs, cls_score.size(1))

        collated_prediction_with_meta = {
            'frame_dir': [frame_dir for batch_data in pred_ for frame_dir in batch_data['frame_dir']],
            'total_frames': torch.cat([batch_data['total_frames'] for batch_data in pred_], dim=0),
            'label': torch.cat([batch_data['label'] for batch_data in pred_], dim=0).squeeze(dim=1),
            'clip_len': torch.cat([batch_data['clip_len'] for batch_data in pred_], dim=0),
            'num_clips': torch.cat([batch_data['num_clips'] for batch_data in pred_], dim=0),
            'frame_inds': torch.cat([batch_data['frame_inds'] for batch_data in pred_], dim=0),
            'repr_': repr_,
            'cls_score': cls_score
        }
        return collated_prediction_with_meta

    def _testing(self, task_indices: Union[List[int], Tuple[int]], val_test='test', exemplar_class_means=None):
        assert len(task_indices) == 2
        print('Begin testing')
        if exemplar_class_means is not None:
            self.cil_model.extract_repr = True

        metric = torchmetrics.classification.Accuracy(num_classes=self.num_classes(task_indices[-1]), multiclass=True)
        self.data_module.predict_dataloader_mode = val_test
        self.data_module.predict_dataset_task_idx = task_indices
        pred_ = self.predict()

        # collate data
        cls_score = []
        labels = []
        for batch_data in pred_:
            cls_score.extend(batch_data['cls_score'])
            labels.extend(batch_data['label'])
        cls_score = torch.stack(cls_score, dim=0)
        labels = torch.stack(labels, dim=0)
        preds = torch.argmax(cls_score, dim=1, keepdim=False)

        cnn_accuracies = AverageMeter()
        nme_accuracies = AverageMeter()
        if val_test == 'val':
            ds_list = self.data_module.val_datasets
        else:
            ds_list = self.data_module.val_datasets

        start = 0
        for task_idx in range(self.current_task + 1):
            num_samples = len(ds_list[task_idx])
            acc = metric(preds[start: start + num_samples], labels[start: start + num_samples])  # no shuffle
            cnn_accuracies.update(acc.item() * 100, num_samples)
            start += num_samples

        if exemplar_class_means is not None:
            repr_ = []
            for batch_data in pred_:
                repr_.append(batch_data['repr_'])
            repr_ = torch.cat(repr_, dim=0)             # (num_samples, num_crops, dim)
            num_samples, num_crops, dim = repr_.size(0), repr_.size(1), repr_.size(2),
            repr_ = repr_.reshape(-1, repr_.size(2))    # (num_samples * num_crops, dim)

            num_classes = exemplar_class_means.size(0)

            # cosine distance
            size, dims = repr_.shape        # size = num_samples * num_crops
            repr_broadcast = repr_.unsqueeze(dim=1).expand(size, num_classes, dims)
            similarity = F.cosine_similarity(repr_broadcast, exemplar_class_means, dim=-1)
            similarity = torch.mean(similarity.reshape(num_samples, num_crops, num_classes), dim=1, keepdim=False)
            preds_nme = torch.argmax(similarity, dim=1, keepdim=False)

            # Euclidean distance
            # dist = torch.cdist(repr_, exemplar_class_means)        # num_samples * num_crops, num_classes
            # dist = torch.mean(dist.reshape(num_samples, num_crops, num_classes), dim=1, keepdim=False)
            # preds_nme = torch.argmin(dist, dim=1, keepdim=False)

            start = 0
            for task_idx in range(self.current_task + 1):
                num_samples = len(ds_list[task_idx])
                nme_acc = metric(preds_nme[start: start + num_samples], labels[start: start + num_samples])
                nme_accuracies.update(nme_acc.item() * 100, num_samples)
                start += num_samples
        print('Task {} Accuracies (CNN): {}\nAvg Accuracy (CNN): {}'.format(self.current_task,
                                                                            cnn_accuracies.values,
                                                                            cnn_accuracies.avg))

        if exemplar_class_means is not None:
            print('Task {} Accuracies (NME): {}\nAvg Accuracy (NME): {}'.format(self.current_task,
                                                                                nme_accuracies.values,
                                                                                nme_accuracies.avg))
            self.cil_model.extract_repr = False  # reset flag
            return cnn_accuracies, nme_accuracies
        return cnn_accuracies

    def cil_testing(self, test_nme=False):
        tmp = self._current_task
        cnn_accuracies = []
        nme_accuracies = []

        print('Build test dataset')
        for task_idx in range(self.num_tasks):
            self.data_module.config.data.test.ann_file = str(
                self.data_module.task_splits_ann_files['val'][task_idx])
            self.data_module.test_datasets.append(build_dataset(self.config.data.test))

        for task_idx in range(self.num_tasks):
            self._current_task = task_idx
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))
            self.cil_model.current_model.load_state_dict(
                torch.load(self.ckpt_dir / 'ckpt_task_{}.pt'.format(self._current_task))
            )
            if test_nme:
                exemplar_class_means = self._get_exemplar_class_means(task_idx, override_class_mean_ckpt=False)
                cnn_task_i_accuracies, nme_task_i_accuracies = self._testing(exemplar_class_means=exemplar_class_means,
                                                                             task_indices=[0, task_idx])
                cnn_accuracies.append(cnn_task_i_accuracies)
                nme_accuracies.append(nme_task_i_accuracies)
            else:
                cnn_task_i_accuracies = self._testing(task_indices=[0, task_idx])
                cnn_accuracies.append(cnn_task_i_accuracies)

        print('CNN accuracies')
        cnn_pretty_table = print_mean_accuracy(cnn_accuracies, [len(class_indices) for class_indices in
                                                                self.task_splits[
                                                                self.starting_task: self.ending_task + 1]])
        print(cnn_pretty_table)
        with open(self.work_dir / 'cnn_result.txt', 'w') as f:
            f.write('CNN Accuracies' + cnn_pretty_table + '\n')
        if test_nme:
            print('NME accuracies')
            nme_pretty_table = print_mean_accuracy(nme_accuracies, [len(class_indices) for class_indices in
                                                                    self.task_splits[
                                                                    self.starting_task: self.ending_task + 1]])
            print(nme_pretty_table)
            with open(self.work_dir / 'nme_result.txt', 'w') as f:
                f.write('NME Accuracies' + nme_pretty_table + '\n')

        self._current_task = tmp  # reset state

    def single_ckpt_testing(self, ckpt_file: str, test_nme=True):
        print("Load ckpt from", ckpt_file)
        self.cil_model.load_state_dict(torch.load(ckpt_file, map_location='cuda:0')['state_dict'])

        if test_nme:
            print('Create exemplar')
            class_indices = [self.data_module.ori_idx_to_inc_idx[idx] for idx in self.task_splits[self.current_task]]
            manager = Herding(budget_size=self.config.budget_size,
                              class_indices=class_indices,
                              cosine_distance=True,
                              storing_methods=self.config.storing_methods,
                              budget_type=self.config.budget_type)
            prediction_with_meta = self._extract_features_for_constructing_exemplar()
            exemplar_meta = manager.construct_exemplar(prediction_with_meta)

            exemplar_class_means = [exemplar_meta[class_idx]['class_mean'] for class_idx in range(len(exemplar_meta))]
            exemplar_class_means = torch.stack(exemplar_class_means, dim=0).squeeze(1)
        else:
            exemplar_class_means = None

        # build test datasets
        for task_idx in range(len(self.config.task_splits)):
            self._current_task = task_idx
            self.data_module.config.data.test.ann_file = str(
                self.data_module.task_splits_ann_files['val'][self.current_task])
            self.data_module.test_datasets.append(build_dataset(self.config.data.test))
        self._current_task = self.ending_task
        self._testing(val_test='test', exemplar_class_means=exemplar_class_means, task_indices=[0, self._current_task])

    def _get_exemplar_class_means(self, task_idx: int, override_class_mean_ckpt=False):
        # load class mean from file
        exemplar_class_mean_file = self.ckpt_dir / 'exemplar_class_mean_task_{}.pt'.format(task_idx)
        if not override_class_mean_ckpt and exemplar_class_mean_file.exists():
            print('Load class means (exemplar) from:', exemplar_class_mean_file)
            class_means = torch.load(exemplar_class_mean_file)['class_means']

        # or extract class mean from exemplar
        else:
            print('Begin extract class mean from exemplar')
            self.cil_model.extract_repr = True
            self.cil_model.current_model.update_fc(self.num_classes(self._current_task))

            self.data_module.combine_all_exemplar_ann_files(task_idx)  # prevent error from multiprocessing
            self.data_module.predict_dataloader_mode = 'feature_extraction_on_exemplar'
            self.data_module.predict_dataset_task_idx = task_idx
            pred_ = self.predict()
            repr_ = []
            for batch_data in pred_:
                repr_.append(batch_data['mean_crops_repr_'])
            repr_ = torch.cat(repr_, dim=0)
            repr_ = repr_.reshape(-1, repr_.size(1))

            label = torch.cat([batch_data['label'] for batch_data in pred_], dim=0).squeeze(dim=1)
            class_means = []
            for class_idx in range(self.num_classes(task_idx)):
                indices = (label == class_idx).nonzero().squeeze(dim=1)
                class_means.append(torch.mean(repr_[indices], dim=0))

            class_means = torch.stack(class_means, dim=0)
            torch.save({'class_means': class_means}, exemplar_class_mean_file)
        return class_means

    def predict(self):
        writer_tmp_dir = self.work_dir / 'tmp'
        writer_tmp_dir.mkdir(exist_ok=True)
        predict_writer = PredictWriter(writer_tmp_dir, write_interval='epoch')
        trainer = pl.Trainer(gpus=self.config.gpu_ids,
                             default_root_dir=self.config.work_dir,
                             max_epochs=1,
                             logger=False,
                             enable_checkpointing=False,
                             strategy=None,
                             callbacks=[predict_writer],
                             # limit_predict_batches=10
                             )
        trainer.predict(self.cil_model, datamodule=self.data_module)
        predictions = []
        for f in writer_tmp_dir.glob('predictions_rank_*'):
            predictions.extend(torch.load(f))

        print("Number of predictions:", len(predictions))
        print("removing", writer_tmp_dir)
        shutil.rmtree(writer_tmp_dir)
        return predictions


class PredictWriter(BasePredictionWriter):
    def __init__(self, output_dir: pathlib.Path, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
            self, trainer, pl_module: pl.LightningModule, predictions: List[Any], batch_indices: List[Any]
    ):
        save_location = self.output_dir / "predictions_rank_{}.pt".format(trainer.global_rank)
        torch.save(predictions[0], save_location)
        print("Predictions saved at", save_location)
