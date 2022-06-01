import argparse
from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from libs.cil.cil import BaseCIL, CILDataModule, CILTrainer
import libs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')

    # other configs
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--videos_per_gpu', type=int)
    parser.add_argument('--workers_per_gpu', type=int)
    parser.add_argument('--accumulate_grad_batches', type=int)

    parser.add_argument('--gpu_ids', type=int, nargs='*', help='ids of gpus to use')
    parser.add_argument(
        '--starting_task', default=0, type=int,
        help='start training from selected i-th task. Previous checkpoint and other necessities will '
             'be loaded from work_dir')
    parser.add_argument('--use_cbf', action='store_true', help='Use Class Balance Finetune (CBF')
    parser.add_argument('--cbf_train_backbone', action='store_true', help='Unfreeze backbone when in CBF mode')
    args = parser.parse_args()

    # cfg_dict are used for updating the configurations from config file
    cfg_dict = {}
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            cfg_dict[k] = v
    return args, cfg_dict


def main():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)  # disable some redundant warning

    args, cfg_dict = parse_args()
    config = Config.fromfile(args.config)
    config.merge_from_dict(cfg_dict)
    trainer = CILTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
