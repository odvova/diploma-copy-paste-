"""
Mask R-CNN + IoU tracking — live video or save to file.

Usage examples:
  # Live display with tracking
  python mask_rcnn/infer.py --input mask_rcnn/yt_input2.mp4 --live

  # Save result to file
  python mask_rcnn/infer.py --input mask_rcnn/yt_input.mp4

  # Tune detection frequency (default: every 4 frames)
  python mask_rcnn/infer.py --input mask_rcnn/yt_input2.mp4 --live --detect-every 2
"""

import argparse
import os
import sys
import time

import torch
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, Boxes

sys.path.insert(0, os.path.dirname(__file__))

# ── Dataset constants ─────────────────────────────────────────────────────────
ALL_CLASSES = [
    "air-fighter",                  # 0
    "armoured personnel carrier",   # 1
    "bomber",                       # 2
    "soldier",                      # 3
    "tank",                         # 4
]
TARGET_IDS = {1, 3, 4}   # APC, soldier, tank
PER_CLASS_THRESH: dict = {}   # cid → min score; empty = use model threshold

CLASS_COLORS_BGR = {
    1: (255, 140,   0),   # APC     → orange
    3: (  0, 200,   0),   # soldier → green
    4: (  0, 100, 255),   # tank    → blue
}
LABEL_MAP = {1: "APC", 3: "Soldier", 4: "Tank"}

DETECT_EVERY = 1    # full Mask R-CNN every N frames
MAX_MISSED   = 1     # drop a track after this many missed detection frames

def _make_tracker():
    return cv2.legacy.TrackerCSRT_create()


# ── Registration ──────────────────────────────────────────────────────────────

def _register():
    for split, jf, ir in [
        ("military_train",
         "detectron2/datasets/military_train.json",
         "detectron2/datasets/train"),
        ("military_val",
         "detectron2/datasets/military_val.json",
         "detectron2/datasets/train"),
    ]:
        if split not in DatasetCatalog:
            register_coco_instances(split, {}, jf, ir)
            MetadataCatalog.get(split).set(
                thing_classes=ALL_CLASSES,
                thing_colors=[
                    (128,  64,  64),   # 0 air-fighter  (unused)
                    (255, 140,   0),   # 1 APC          orange
                    ( 64,  64, 128),   # 2 bomber       (unused)
                    (  0, 200,   0),   # 3 soldier      green
                    (  0, 100, 255),   # 4 tank         blue
                ],
            )


def build_predictor(config: str, weights: str, threshold: float):
    _register()
    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    meta = MetadataCatalog.get("military_val")
    return DefaultPredictor(cfg), meta


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_instances(instances):
    if len(instances) == 0:
        return instances
    cpu_classes = instances.pred_classes.cpu()
    cpu_scores  = instances.scores.cpu()
    keep = torch.zeros(len(instances), dtype=torch.bool)
    for cid in TARGET_IDS:
        thresh = PER_CLASS_THRESH.get(cid, 0.0)
        keep |= (cpu_classes == cid) & (cpu_scores >= thresh)
    return instances[keep]


# ── IoU tracker (no opencv-contrib required) ──────────────────────────────────

def _iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 1

    def __init__(self, frame: np.ndarray, box_xyxy, class_id: int, score: float, mask=None):
        self.id       = Track._next_id
        Track._next_id += 1
        self.class_id = class_id
        self.score    = score
        self.mask     = mask
        self.box      = tuple(int(v) for v in box_xyxy)   # (x1,y1,x2,y2)
        self.missed   = 0
        self._tracker = _make_tracker()
        x1, y1, x2, y2 = self.box
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def reinit(self, frame: np.ndarray, box_xyxy, score: float, mask):
        self.box    = tuple(int(v) for v in box_xyxy)
        self.score  = score
        self.mask   = mask
        self.missed = 0
        self._tracker = _make_tracker()
        x1, y1, x2, y2 = self.box
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def step(self, frame: np.ndarray) -> bool:
        """Advance tracker one frame. Returns False if tracking lost."""
        ok, bbox = self._tracker.update(frame)
        if ok:
            x, y, w, h = (int(v) for v in bbox)
            self.box = (x, y, x + w, y + h)
        else:
            self.missed += 1
        return bool(ok)


def refresh_tracks(tracks: list, frame: np.ndarray, instances) -> list:
    """
    Called on every detection frame.
    Matches Detectron2 detections → existing tracks by IoU + class.
    Unmatched detections → new tracks. Unmatched old tracks age out.
    """
    if len(instances) == 0:
        for t in tracks:
            t.missed += 1
        return [t for t in tracks if t.missed <= MAX_MISSED]

    boxes   = instances.pred_boxes.tensor.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    scores  = instances.scores.cpu().numpy()
    masks   = (instances.pred_masks.cpu().numpy()
               if instances.has("pred_masks") else [None] * len(boxes))

    matched_track_ids: set = set()
    det_matched = [False] * len(boxes)

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        best_iou, best_t = 0.2, None
        for t in tracks:
            if t.class_id != int(cls):
                continue
            v = _iou(box, t.box)
            if v > best_iou:
                best_iou, best_t = v, t
        if best_t and best_t.id not in matched_track_ids:
            matched_track_ids.add(best_t.id)
            det_matched[i] = True
            best_t.reinit(frame, box, float(scores[i]), masks[i])

    surviving: list = []
    for t in tracks:
        if t.id in matched_track_ids:
            surviving.append(t)
        else:
            t.missed += 1
            if t.missed <= MAX_MISSED:
                surviving.append(t)

    for i, matched in enumerate(det_matched):
        if not matched:
            surviving.append(Track(frame, boxes[i], int(classes[i]),
                                   float(scores[i]), masks[i]))
    return surviving


# ── Drawing ───────────────────────────────────────────────────────────────────

def _tracks_to_instances(tracks: list, H: int, W: int) -> Instances:
    """Convert Track list → Detectron2 Instances so Visualizer can render them."""
    inst = Instances((H, W))
    if not tracks:
        return inst
    inst.pred_boxes   = Boxes(torch.tensor([list(t.box) for t in tracks], dtype=torch.float32))
    inst.pred_classes = torch.tensor([t.class_id for t in tracks], dtype=torch.int64)
    inst.scores       = torch.tensor([t.score    for t in tracks], dtype=torch.float32)
    masks = [t.mask for t in tracks]
    if all(m is not None for m in masks):
        inst.pred_masks = torch.tensor(np.stack(masks), dtype=torch.bool)
    return inst


def render_tracks(frame_bgr: np.ndarray, tracks: list, meta) -> np.ndarray:
    """Draw solid masks + labels with fixed per-class colors (no Visualizer jitter)."""
    if not tracks:
        return frame_bgr
    vis = frame_bgr.copy()

    # draw masks first (all at once to avoid repeated addWeighted)
    mask_layer = vis.copy()
    for t in tracks:
        if t.mask is None:
            continue
        color = CLASS_COLORS_BGR.get(t.class_id, (200, 200, 200))
        mask_layer[t.mask.astype(bool)] = color
    vis = cv2.addWeighted(mask_layer, 0.72, vis, 0.28, 0)

    # draw boxes and labels on top
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in tracks:
        x1, y1, x2, y2 = t.box
        color = CLASS_COLORS_BGR.get(t.class_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label = LABEL_MAP.get(t.class_id, "?")
        cv2.putText(vis, label, (x1, y1 - 3),
                    font, 0.45, color, 1, cv2.LINE_AA)
    return vis


# ── Single image ──────────────────────────────────────────────────────────────

def run_image(predictor, meta, path: str, out_dir: str):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return
    instances = filter_instances(predictor(img)["instances"])
    rgb = img[:, :, ::-1]
    v = Visualizer(rgb, metadata=meta, scale=1.2, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(instances.to("cpu"))
    vis = out.get_image()[:, :, ::-1]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_pred.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}  ({len(instances)} detections)")


# ── Video — save to file ──────────────────────────────────────────────────────

def run_video_save(predictor, meta, path: str, out_dir: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_pred.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))

    tracks: list = []
    frame_idx = 0
    t_start = time.time()
    fps_display = 0.0
    fps_alpha = 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        is_det = (frame_idx % DETECT_EVERY == 0)

        if is_det:
            instances = filter_instances(predictor(frame)["instances"])
            tracks = refresh_tracks(tracks, frame, instances)
        else:
            for t in tracks:
                t.step(frame)
            tracks = [t for t in tracks if t.missed <= MAX_MISSED]

        dt = time.time() - t0
        fps_display = fps_display * (1 - fps_alpha) + (1 / max(dt, 1e-3)) * fps_alpha

        vis = render_tracks(frame, tracks, meta)
        writer.write(vis)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            pct = 100.0 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  avg {frame_idx/elapsed:.1f} fps")

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")


# ── Video — live window ───────────────────────────────────────────────────────

def run_video_live(predictor, meta, path: str):
    src = int(path) if path.isdigit() else path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return

    win = "Mask R-CNN + tracking  |  Q = quit   S = screenshot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    tracks: list = []
    fps_display = 0.0
    fps_alpha = 0.1
    frame_idx = 0
    screenshot_dir = "mask_rcnn/predictions"
    os.makedirs(screenshot_dir, exist_ok=True)

    print("Live inference + tracking — Q to quit, S to screenshot")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            if not str(src).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracks = []
                continue
            break

        is_det = (frame_idx % DETECT_EVERY == 0)
        if is_det:
            instances = filter_instances(predictor(frame)["instances"])
            tracks = refresh_tracks(tracks, frame, instances)
        else:
            for t in tracks:
                t.step(frame)
            tracks = [t for t in tracks if t.missed <= MAX_MISSED]

        dt = time.time() - t0
        fps_display = fps_display * (1 - fps_alpha) + (1 / max(dt, 1e-3)) * fps_alpha

        vis = render_tracks(frame, tracks, meta)
        cv2.imshow(win, vis)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snap = os.path.join(screenshot_dir, f"live_snap_{frame_idx:06d}.jpg")
            cv2.imwrite(snap, vis)
            print(f"Screenshot: {snap}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global DETECT_EVERY
    parser = argparse.ArgumentParser(
        description="Mask R-CNN + IoU tracking on images or video")
    parser.add_argument(
        "--config",
        default="detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--weights", default="outputs/military_v2/model_final.pth")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Image/video paths, or '0' for webcam")
    parser.add_argument("--output-dir", default="mask_rcnn/predictions")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--live", action="store_true",
                        help="Open live display window instead of saving")
    parser.add_argument("--detect-every", type=int, default=DETECT_EVERY,
                        help=f"Full detection every N frames (default: {DETECT_EVERY})")
    parser.add_argument("--classes", nargs="+", default=["soldier", "apc", "tank"],
                        choices=["soldier", "apc", "tank"],
                        help="Classes to detect (default: all three)")
    parser.add_argument("--thresholds", nargs="+", default=[],
                        metavar="CLASS:THRESH",
                        help="Per-class thresholds, e.g. soldier:0.70 apc:0.85")
    args = parser.parse_args()

    DETECT_EVERY = args.detect_every

    name_to_id = {"soldier": 3, "apc": 1, "tank": 4}
    global TARGET_IDS, PER_CLASS_THRESH
    TARGET_IDS = {name_to_id[c] for c in args.classes}

    PER_CLASS_THRESH = {}
    for spec in args.thresholds:
        name, val = spec.split(":")
        PER_CLASS_THRESH[name_to_id[name]] = float(val)

    # model threshold = lowest per-class value so all candidates pass through
    model_thresh = min(PER_CLASS_THRESH.values()) if PER_CLASS_THRESH else args.threshold

    print(f"Loading model: {args.weights}")
    predictor, meta = build_predictor(args.config, args.weights, model_thresh)
    print(f"Model loaded. Tracking: {', '.join(args.classes)}  (detect every {DETECT_EVERY} frames)")

    for path in args.input:
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv") or path.isdigit()

        if args.live and is_video:
            run_video_live(predictor, meta, path)
        elif is_video:
            print(f"Processing: {path}")
            run_video_save(predictor, meta, path, args.output_dir)
        else:
            run_image(predictor, meta, path, args.output_dir)


if __name__ == "__main__":
    main()
