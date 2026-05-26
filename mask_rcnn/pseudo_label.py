"""
Extract frames from video(s), run Mask R-CNN at high confidence threshold,
and export a COCO-format JSON ready to upload to Roboflow for review.

Usage:
    python mask_rcnn/pseudo_label.py \
        --videos mask_rcnn/yt_input.mp4 mask_rcnn/yt_input2.mp4 \
        --out-dir mask_rcnn/pseudo_labeled \
        --sample-fps 1 \
        --threshold 0.7
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

sys.path.insert(0, os.path.dirname(__file__))
from dataset import register_isaid, register_military, MILITARY_CLASSES


def mask_to_polygon(binary_mask):
    """Convert a binary H×W mask to a list of COCO polygon coordinate lists."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        poly = c.flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


def build_predictor(config, weights, threshold):
    register_isaid()
    register_military()
    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    return DefaultPredictor(cfg)


def process_videos(videos, out_dir, sample_fps, threshold, config, weights):
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    predictor = build_predictor(config, weights, threshold)

    # COCO-format containers
    coco = {
        "info": {"description": "Pseudo-labeled military dataset"},
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "military"}
            for i, name in enumerate(MILITARY_CLASSES)
        ],
        "images": [],
        "annotations": [],
    }
    img_id = 1
    ann_id = 1

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(native_fps / sample_fps)))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"\n{video_path}: {total_frames} frames @ {native_fps:.0f}fps, "
              f"sampling every {step} frames (~{sample_fps}fps)")

        frame_idx = 0
        saved = 0
        annotated = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                instances = predictor(frame)["instances"].to("cpu")

                H, W = frame.shape[:2]
                fname = f"{video_name}_f{frame_idx:07d}.jpg"
                fpath = os.path.join(images_dir, fname)
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                coco["images"].append({
                    "id": img_id,
                    "file_name": fname,
                    "width": W,
                    "height": H,
                })

                # Frames with zero detections are saved as hard negatives (no annotations)
                if len(instances) > 0:
                    boxes = instances.pred_boxes.tensor.tolist()
                    classes = instances.pred_classes.tolist()
                    scores = instances.scores.tolist()
                    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

                    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                        x1, y1, x2, y2 = box
                        bw, bh = x2 - x1, y2 - y1
                        area = bw * bh

                        segmentation = []
                        if masks is not None:
                            segmentation = mask_to_polygon(masks[i])
                        if not segmentation:
                            segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls + 1,
                            "segmentation": segmentation,
                            "bbox": [x1, y1, bw, bh],
                            "area": float(area),
                            "iscrowd": 0,
                            "score": round(score, 4),
                        })
                        ann_id += 1
                        annotated += 1

                img_id += 1
                saved += 1

            frame_idx += 1
            if frame_idx % 1000 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  {frame_idx}/{total_frames} ({pct:.0f}%) — {saved} frames saved")

        cap.release()
        print(f"  Done: {saved} frames, {annotated} annotations")

    out_json = os.path.join(out_dir, "pseudo_labels.json")
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"\nTotal: {img_id - 1} images, {ann_id - 1} annotations")
    print(f"COCO JSON: {out_json}")
    print(f"Images:    {images_dir}")
    print("\nNext steps:")
    print("  1. Upload images/ folder to Roboflow as a new dataset version")
    print("  2. Import pseudo_labels.json as annotations (COCO format)")
    print("  3. Review predictions, fix/delete wrong boxes, add missed objects")
    print("  4. Export as COCO JSON and drop into detectron2/datasets/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--out-dir", default="mask_rcnn/pseudo_labeled")
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="frames per second to extract (default: 1)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="confidence threshold — higher = fewer but cleaner pseudo-labels")
    parser.add_argument("--config", default="mask_rcnn/configs/military_finetune.yaml")
    parser.add_argument("--weights", default="outputs/military/model_final.pth")
    args = parser.parse_args()

    process_videos(
        videos=args.videos,
        out_dir=args.out_dir,
        sample_fps=args.sample_fps,
        threshold=args.threshold,
        config=args.config,
        weights=args.weights,
    )


if __name__ == "__main__":
    main()
