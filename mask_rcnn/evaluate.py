import argparse
import os
import sys

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import register_isaid, register_military


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    register_isaid()
    register_military()

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    evaluator = COCOEvaluator(args.dataset, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"))
    loader = build_detection_test_loader(cfg, args.dataset)
    inference_on_dataset(model, loader, evaluator)


if __name__ == "__main__":
    main()
