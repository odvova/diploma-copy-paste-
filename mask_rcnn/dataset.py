from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

MILITARY_CLASSES = ["air-fighter", "armoured personnel carrier", "bomber", "soldier", "tank"]


def register_isaid(root: str = "detectron2/datasets/iSAID"):
    register_coco_instances(
        "isaid_train",
        {},
        f"{root}/train/iSAID_train_patches.json",
        f"{root}/train/images",
    )
    register_coco_instances(
        "isaid_val",
        {},
        f"{root}/val/iSAID_val_patches.json",
        f"{root}/val/images",
    )


def register_military(root: str = "detectron2/datasets"):
    register_coco_instances(
        "military_train",
        {},
        f"{root}/military_train.json",
        f"{root}/train",
    )
    register_coco_instances(
        "military_val",
        {},
        f"{root}/military_val.json",
        f"{root}/train",
    )
    MetadataCatalog.get("military_train").thing_classes = MILITARY_CLASSES
    MetadataCatalog.get("military_val").thing_classes = MILITARY_CLASSES
