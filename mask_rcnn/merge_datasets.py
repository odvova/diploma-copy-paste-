"""
Merge Roboflow COCO exports into the existing military_train/val JSONs.

- Strips spurious 'Soldier-recognition' class
- Remaps Roboflow category IDs to match MILITARY_CLASSES (0-indexed)
- Copies new images into detectron2/datasets/train/
- Splits new data 80/20 and appends to military_train.json / military_val.json
"""

import json
import os
import random
import shutil

random.seed(42)

EXISTING_TRAIN = "detectron2/datasets/military_train.json"
EXISTING_VAL   = "detectron2/datasets/military_val.json"
IMAGES_DST     = "detectron2/datasets/train"

# Our canonical category mapping (0-indexed, matching MILITARY_CLASSES)
CANONICAL = {
    "air-fighter":               0,
    "armoured personnel carrier": 1,
    "bomber":                    2,
    "soldier":                   3,
    "tank":                      4,
}
SKIP_CLASSES = {"soldier-recognition"}  # spurious roboflow project-name class

ROBOFLOW_EXPORTS = [
    "/home/odvova/Downloads/Soldier recognition-pseudo_labeled_v1.coco/train",
    "/home/odvova/Downloads/Soldier recognition-pseudo_labeled_v1- Job 2.coco/train",
]


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def save_coco(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {path}  ({len(data['images'])} images, {len(data['annotations'])} annotations)")


def build_cat_remap(categories):
    """Map Roboflow category_id → canonical category_id. Returns None for skip."""
    remap = {}
    for cat in categories:
        name = cat["name"].strip().lower()
        if name in SKIP_CLASSES:
            remap[cat["id"]] = None
        else:
            canonical_id = CANONICAL.get(name)
            if canonical_id is None:
                print(f"  [WARN] unknown class '{cat['name']}' — skipping")
                remap[cat["id"]] = None
            else:
                remap[cat["id"]] = canonical_id
    return remap


def main():
    train = load_coco(EXISTING_TRAIN)
    val   = load_coco(EXISTING_VAL)

    next_img_id = max(i["id"] for i in train["images"] + val["images"]) + 1
    next_ann_id = max(a["id"] for a in train["annotations"] + val["annotations"]) + 1

    new_train_imgs, new_train_anns = [], []
    new_val_imgs,   new_val_anns   = [], []

    for export_dir in ROBOFLOW_EXPORTS:
        ann_file = os.path.join(export_dir, "_annotations.coco.json")
        coco = load_coco(ann_file)
        cat_remap = build_cat_remap(coco["categories"])

        ann_by_image = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        print(f"\n{export_dir}")
        print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

        imgs_added = anns_added = anns_skipped = 0

        for img in coco["images"]:
            src = os.path.join(export_dir, img["file_name"])
            if not os.path.exists(src):
                continue

            # Copy image with new unique name to avoid collisions
            new_fname = f"rf_{os.path.basename(img['file_name'])}"
            dst = os.path.join(IMAGES_DST, new_fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            new_id = next_img_id
            next_img_id += 1

            new_img = {
                "id": new_id,
                "file_name": new_fname,
                "width": img["width"],
                "height": img["height"],
            }

            new_anns = []
            for ann in ann_by_image.get(img["id"], []):
                mapped = cat_remap.get(ann["category_id"])
                if mapped is None:
                    anns_skipped += 1
                    continue
                new_ann = {
                    "id": next_ann_id,
                    "image_id": new_id,
                    "category_id": mapped,
                    "segmentation": ann.get("segmentation", []),
                    "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": 0,
                }
                new_anns.append(new_ann)
                next_ann_id += 1
                anns_added += 1

            # 80/20 split
            if random.random() < 0.8:
                new_train_imgs.append(new_img)
                new_train_anns.extend(new_anns)
            else:
                new_val_imgs.append(new_img)
                new_val_anns.extend(new_anns)

            imgs_added += 1

        print(f"  added: {imgs_added} images, {anns_added} annotations, {anns_skipped} skipped (spurious class)")

    train["images"]      += new_train_imgs
    train["annotations"] += new_train_anns
    val["images"]        += new_val_imgs
    val["annotations"]   += new_val_anns

    print(f"\nMerge summary:")
    print(f"  New train images: {len(new_train_imgs)}, annotations: {len(new_train_anns)}")
    print(f"  New val   images: {len(new_val_imgs)},   annotations: {len(new_val_anns)}")

    save_coco(train, EXISTING_TRAIN)
    save_coco(val,   EXISTING_VAL)


if __name__ == "__main__":
    main()
