import argparse
from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from libs.cil.cil import BaseCIL, CILDataModule, CILTrainer
import libs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def main():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable some redundant warning

    args = parse_args()
    config = Config.fromfile(args.config)
    # cfg.merge_from_dict(args.cfg_options)
    # data_module = CILDataModule(config)
    # model = BaseCIL(config, data_module)
    # data_module.prepare_data()
    # trainer = pl.Trainer(gpus=config.gpu_ids,
    #                      # accumulate_grad_batches=config.TRAIN.ACCUMULATE_GRAD_BATCHES,
    #                      # max_epochs=cl_model.max_epochs,
    #                      default_root_dir=config.work_dir,
    #                      limit_train_batches=10,
    #                      limit_val_batches=0.0,
    #                      # val_check_interval=10000
    #                      )
    # model.add_trainer(trainer)
    # trainer.fit(model, data_module)

    trainer = CILTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
