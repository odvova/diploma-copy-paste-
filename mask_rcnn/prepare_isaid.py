"""
iSAID patch extraction for Detectron2.
Slices large aerial images into 800x800 patches (512px stride),
clips polygon annotations to each patch, and writes new COCO JSONs.

Usage:
    python mask_rcnn/prepare_isaid.py
    python mask_rcnn/prepare_isaid.py --patch-size 800 --stride 512 --min-area 10
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.validation import make_valid


def polygon_to_shapely(flat):
    pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if not p.is_valid:
        p = make_valid(p)
    return p if not p.is_empty else None


def shapely_to_polygon(geom):
    """Returns list of flat [x, y, x, y, ...] polygons (handles MultiPolygon)."""
    from shapely.geometry import MultiPolygon, GeometryCollection
    polys = []
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        if len(coords) >= 3:
            polys.append([c for pt in coords for c in pt])
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            polys.extend(shapely_to_polygon(part))
    return polys


def process_split(split, raw_root, out_root, patch_size, stride, min_area):
    img_dir = raw_root / split / "images"
    ann_file = raw_root / split / f"iSAID_{split}.json"
    out_img_dir = out_root / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    id_to_anns = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_images, new_annotations = [], []
    new_img_id = 0
    new_ann_id = 0

    total = len(coco["images"])
    for idx, img_info in enumerate(coco["images"]):
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        anns = id_to_anns.get(img_info["id"], [])

        # Precompute shapely polygons for this image's annotations
        ann_shapes = []
        for ann in anns:
            if not ann["segmentation"] or ann.get("iscrowd"):
                continue
            for flat in ann["segmentation"]:
                shp = polygon_to_shapely(flat)
                if shp is not None:
                    ann_shapes.append((ann, shp))

        xs = list(range(0, max(1, W - patch_size), stride)) + [max(0, W - patch_size)]
        ys = list(range(0, max(1, H - patch_size), stride)) + [max(0, H - patch_size)]
        xs = sorted(set(xs))
        ys = sorted(set(ys))

        for y0 in ys:
            for x0 in xs:
                x1 = min(x0 + patch_size, W)
                y1 = min(y0 + patch_size, H)
                patch_box = shapely_box(x0, y0, x1, y1)

                patch_anns = []
                for ann, shp in ann_shapes:
                    if not patch_box.intersects(shp):
                        continue
                    clipped = patch_box.intersection(shp)
                    if clipped.is_empty or clipped.area < min_area:
                        continue
                    polys = shapely_to_polygon(clipped)
                    polys = [[c - (x0 if i % 2 == 0 else y0) for i, c in enumerate(p)]
                              for p in polys]
                    polys = [p for p in polys if len(p) >= 6]
                    if not polys:
                        continue
                    all_x = [p[i] for p in polys for i in range(0, len(p), 2)]
                    all_y = [p[i] for p in polys for i in range(1, len(p), 2)]
                    bx, by = min(all_x), min(all_y)
                    bw, bh = max(all_x) - bx, max(all_y) - by
                    patch_anns.append({
                        "id": new_ann_id,
                        "image_id": new_img_id,
                        "category_id": ann["category_id"],
                        "segmentation": polys,
                        "bbox": [bx, by, bw, bh],
                        "area": clipped.area,
                        "iscrowd": 0,
                    })
                    new_ann_id += 1

                if not patch_anns:
                    continue

                pw, ph = x1 - x0, y1 - y0
                patch_img = img[y0:y1, x0:x1]
                fname = f"{img_path.stem}_{x0}_{y0}.png"
                cv2.imwrite(str(out_img_dir / fname), patch_img)

                new_images.append({
                    "id": new_img_id,
                    "file_name": fname,
                    "width": pw,
                    "height": ph,
                })
                new_annotations.extend(patch_anns)
                new_img_id += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{split}] {idx + 1}/{total} images, "
                  f"{new_img_id} patches, {new_ann_id} annotations")

    out_json = out_root / split / f"iSAID_{split}_patches.json"
    with open(out_json, "w") as f:
        json.dump({
            "categories": coco["categories"],
            "images": new_images,
            "annotations": new_annotations,
        }, f)

    print(f"[{split}] Done: {new_img_id} patches, {new_ann_id} annotations → {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="detectron2/datasets/iSAID_raw")
    parser.add_argument("--out-root", default="detectron2/datasets/iSAID")
    parser.add_argument("--patch-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-area", type=float, default=10.0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    for split in ("train", "val"):
        process_split(split, raw_root, out_root, args.patch_size, args.stride, args.min_area)


if __name__ == "__main__":
    main()
