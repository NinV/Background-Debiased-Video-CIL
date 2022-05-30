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
    trainer = CILTrainer(config)
    trainer.cil_testing()


if __name__ == '__main__':
    main()
