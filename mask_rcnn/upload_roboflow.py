"""
Upload pseudo-labeled dataset to Roboflow project.
Converts COCO segmentation annotations to YOLO-seg .txt format (auto-detected by Roboflow).
Images with no detections are uploaded without an annotation file (hard negatives).
"""

import json
import os
import tempfile

from roboflow import Roboflow

API_KEY    = "CGF5qiFDQWlKY3Kr72d8"
WORKSPACE  = "volodymyr-kozariz"
PROJECT    = "soldier-recognition"
BATCH_NAME = "pseudo_labeled_v1"

COCO_JSON  = "mask_rcnn/pseudo_labeled/pseudo_labels.json"
IMAGES_DIR = "mask_rcnn/pseudo_labeled/images"


def coco_ann_to_yolo_seg(ann, img_w, img_h):
    """Convert one COCO segmentation annotation to a YOLO-seg line."""
    cls = ann["category_id"] - 1  # COCO 1-indexed → YOLO 0-indexed
    segs = ann.get("segmentation", [])
    if not segs:
        # Fall back to bbox polygon
        x, y, w, h = ann["bbox"]
        pts = [x, y, x+w, y, x+w, y+h, x, y+h]
    else:
        pts = segs[0]  # use first polygon

    norm = []
    for j in range(0, len(pts), 2):
        norm.append(f"{pts[j] / img_w:.6f}")
        norm.append(f"{pts[j+1] / img_h:.6f}")
    return f"{cls} " + " ".join(norm)


def main():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    total = len(coco["images"])
    print(f"Uploading {total} images to {WORKSPACE}/{PROJECT} (batch: {BATCH_NAME})")

    ok = skipped = errors = 0

    for i, img_info in enumerate(coco["images"], 1):
        img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img_anns = ann_by_image.get(img_info["id"], [])
        W, H = img_info["width"], img_info["height"]

        ann_path = None
        tmp_path = None

        if img_anns:
            lines = [coco_ann_to_yolo_seg(a, W, H) for a in img_anns]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp.write("\n".join(lines))
                tmp_path = tmp.name
            ann_path = tmp_path

        try:
            project.upload(
                image_path=img_path,
                annotation_path=ann_path,
                batch_name=BATCH_NAME,
            )
            ok += 1
        except Exception as e:
            print(f"  [ERR] {img_info['file_name']}: {e}")
            errors += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} — ok:{ok}  skipped:{skipped}  errors:{errors}")

    print(f"\nDone. {ok} uploaded, {skipped} skipped, {errors} errors.")


if __name__ == "__main__":
    main()
