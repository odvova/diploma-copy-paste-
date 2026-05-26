import argparse
import os
import sys
import time
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

sys.path.insert(0, os.path.dirname(__file__))
from dataset import register_isaid, register_military


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mask_rcnn/configs/military_finetune.yaml")
    parser.add_argument("--weights", default="outputs/military/model_final.pth")
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5, help="frames to discard before timing")
    args = parser.parse_args()

    register_isaid()
    register_military()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(args.input)
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = args.max_frames or total_frames
    print(f"Video: {W}x{H} @ {fps_native:.1f} FPS, {total_frames} frames total")
    print(f"Running benchmark on first {min(max_frames, total_frames)} frames "
          f"({args.warmup} warmup)...")

    latencies = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        predictor(frame)
        t1 = time.perf_counter()
        if frame_idx >= args.warmup:
            latencies.append((t1 - t0) * 1000)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{min(max_frames, total_frames)} frames")

    cap.release()

    latencies = np.array(latencies)
    print("\n--- Inference Speed Results ---")
    print(f"Frames measured  : {len(latencies)}")
    print(f"Mean latency     : {latencies.mean():.1f} ms")
    print(f"Median latency   : {np.median(latencies):.1f} ms")
    print(f"Std dev          : {latencies.std():.1f} ms")
    print(f"Min latency      : {latencies.min():.1f} ms")
    print(f"Max latency      : {latencies.max():.1f} ms")
    print(f"Mean FPS         : {1000 / latencies.mean():.2f}")
    print(f"Native video FPS : {fps_native:.1f}")
    print(f"Realtime ratio   : {1000 / latencies.mean() / fps_native:.2f}x "
          f"({'faster' if 1000/latencies.mean() > fps_native else 'slower'} than real-time)")


if __name__ == "__main__":
    main()
