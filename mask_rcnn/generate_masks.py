"""
Converts a COCO bbox-only dataset to instance segmentation format using SAM.
- Filters out the supercategory (id=0)
- Re-indexes categories to 0-based for Detectron2
- Generates polygon masks from bounding boxes via SAM
- Splits into train/val (80/20)
- Writes: datasets/military_train.json, datasets/military_val.json
"""

import argparse
import copy
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1:
        return None
    polygon = contour.flatten().tolist()
    if len(polygon) < 6:
        return None
    return polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="detectron2/datasets")
    parser.add_argument("--sam-checkpoint", default="mask_rcnn/checkpoints/sam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="detectron2/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ann_file = dataset_dir / "train" / "_annotations.coco.json"
    img_dir = dataset_dir / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    # Filter out supercategory (id=0), remap remaining to 0-indexed
    real_cats = [c for c in coco["categories"] if c["id"] != 0]
    old_to_new = {c["id"]: i for i, c in enumerate(real_cats)}
    new_categories = [
        {"id": i, "name": c["name"], "supercategory": "military"}
        for i, c in enumerate(real_cats)
    ]
    print(f"Classes: {[c['name'] for c in new_categories]}")

    valid_anns = [a for a in coco["annotations"] if a["category_id"] in old_to_new]
    img_ids_with_anns = set(a["image_id"] for a in valid_anns)
    valid_imgs = [img for img in coco["images"] if img["id"] in img_ids_with_anns]
    print(f"Images: {len(valid_imgs)}, Annotations: {len(valid_anns)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({args.sam_model_type}) on {device}...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    img_to_anns = {}
    for ann in valid_anns:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_annotations = []
    ann_id = 1
    for i, img_info in enumerate(valid_imgs):
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"  Missing: {img_path}, skipping")
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  Unreadable: {img_path}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        for ann in img_to_anns.get(img_info["id"], []):
            box = np.array(bbox_xywh_to_xyxy(ann["bbox"]))
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            polygon = mask_to_polygon(best_mask)
            if polygon is None:
                continue
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = ann_id
            new_ann["category_id"] = old_to_new[ann["category_id"]]
            new_ann["segmentation"] = [polygon]
            new_ann["iscrowd"] = 0
            new_annotations.append(new_ann)
            ann_id += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(valid_imgs)} images processed")

    print(f"Generated {len(new_annotations)} mask annotations")

    random.seed(args.seed)
    random.shuffle(valid_imgs)
    n_val = int(len(valid_imgs) * args.val_ratio)
    val_imgs = valid_imgs[:n_val]
    train_imgs = valid_imgs[n_val:]
    val_ids = set(img["id"] for img in val_imgs)
    train_anns = [a for a in new_annotations if a["image_id"] not in val_ids]
    val_anns = [a for a in new_annotations if a["image_id"] in val_ids]

    def save(imgs, anns, path):
        with open(path, "w") as f:
            json.dump({
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": new_categories,
                "images": imgs,
                "annotations": anns,
            }, f)

    save(train_imgs, train_anns, output_dir / "military_train.json")
    save(val_imgs, val_anns, output_dir / "military_val.json")
    print(f"Train: {len(train_imgs)} images, {len(train_anns)} anns")
    print(f"Val:   {len(val_imgs)} images, {len(val_anns)} anns")


if __name__ == "__main__":
    main()
