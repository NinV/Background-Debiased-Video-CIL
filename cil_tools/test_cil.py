import argparse
from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from libs.cil.cil import BaseCIL, CILDataModule, CILTrainer
import libs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--gpu_ids', type=int, nargs='*', help='ids of gpus to use')

    # other configs
    parser.add_argument('--testing_videos_per_gpu', type=int)
    parser.add_argument('--testing_workers_per_gpu', type=int)
    parser.add_argument('--save_best', action='store_true', help='this flag do nothing')
    args = parser.parse_args()

    # cfg_dict are used for updating the configurations from config file
    cfg_dict = {}
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            cfg_dict[k] = v
    return args, cfg_dict

def main():
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable some redundant warning

    args, cfg_dict = parse_args()
    config = Config.fromfile(args.config)
    config.merge_from_dict(cfg_dict)
    config.starting_task = 0
    trainer = CILTrainer(config, dump_config=False)
    trainer.cil_testing(test_nme=True)


if __name__ == '__main__':
    main()
