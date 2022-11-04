import pathlib
import argparse
import cv2
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')

    parser.add_argument('-i', '--image_dir', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('--glob_pattern', default="*")
    args = parser.parse_args()
    return args


def get_predictor():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


if __name__ == '__main__':
    print("Parsing args")
    args = parse_args()
    print("Building model")
    predictor = get_predictor()

    image_dir = pathlib.Path(args.image_dir)
    our_dir = pathlib.Path(args.out_dir)
    our_dir.mkdir(exist_ok=False, parents=True)
    image_files = list(image_dir.glob(args.glob_pattern))

    all_outputs = []
    count = 0
    for im_file in tqdm(image_files):
        img = cv2.imread(str(im_file))
        outputs = predictor(img)
        outputs['im_file'] = str(im_file)
        all_outputs.append(outputs)
        if not 0 in outputs['instances'].pred_classes:
            shutil.copy(im_file, our_dir / im_file.name)
            count += 1

    with open("detection.json", "w") as f:
        json.dump(all_outputs, f)
