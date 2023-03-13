import pathlib
import torch
from pytorch_lightning.loggers import WandbLogger
from .memory_selection import PredefinedMemoryManager
from .icarl import ICARLModel
from .icarl_video_mix import ICARLVideoMix
from .cil import CILTrainer, CILDataModule, BaseCIL


class CustomMemCILTrainer(CILTrainer):
    def __init__(self, config, dump_config=True):
        self.config = config
        self.work_dir = pathlib.Path(config.work_dir)

        # class incremental learning setting
        self.starting_task = config.starting_task
        self._current_task = self.starting_task
        self.num_epoch_per_task = config.num_epochs_per_task
        self.task_splits = config.task_splits
        self.num_tasks = min(len(config.task_splits), config.ending_task + 1)
        self.ending_task = config.ending_task
        self.max_epochs = self.num_tasks * self.num_epoch_per_task

        # setup data module
        self.data_module = CILDataModule(config)
        self.data_module.controller = self
        if config.methods == 'base':
            self.cil_model = BaseCIL(config)
        elif config.methods == 'icarl':
            self.cil_model = ICARLModel(config)
        elif config.methods == 'icarl_video_mix':
            self.cil_model = ICARLVideoMix(config)
        else:
            raise ValueError
        self.cil_model.controller = self

        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.data_module.generate_annotation_file()
        self.exemplar_selector = PredefinedMemoryManager(exemplar_file=config.exemplar_file,
                                                         task_splits=config.task_splits)
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
                    exemplar_meta = self.exemplar_selector.construct_exemplar(self.current_task)
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
            exemplar_meta = self.exemplar_selector.construct_exemplar(task_idx=self.current_task)
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
                self.cil_model.prev_model.update_fc(self.num_classes(self._current_task))

                # 3. prepare data for next task and update model
                self.data_module.reload_train_dataset(use_internal_exemplar=True)
            print('#####################################################################################\n')
