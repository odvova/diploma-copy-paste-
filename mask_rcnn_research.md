# Mask R-CNN Research Notes — Aerial Combatant Detection

## Goal

Extend the diploma project with Mask R-CNN-based instance segmentation for combatant detection from FPV drone footage.
Core idea: bounding boxes (YOLOv7) lose shape/silhouette information. Mask R-CNN's pixel-level masks preserve it,
which matters when equipment load, body posture, and group formation are the primary distinguishing features
between a combatant and a civilian from the air.

---

## Training Pipeline

```
COCO pretrained weights
        ↓
Fine-tune on iSAID        ← aerial domain: overhead viewpoint, small objects
        ↓
Fine-tune on annotated    ← combatant-specific: soldiers, military vehicles
Roboflow military data
```

Each stage saves a checkpoint — if a later stage goes wrong, you don't redo everything from scratch.

**Note:** iSAID has no soldier/combatant classes (it covers vehicles, buildings, ships, etc.).
Stage 2 gives aerial domain knowledge; stage 3 is where combatant recognition comes from.

---

## Datasets

### 1. Roboflow — Military Vehicle Recognition (primary fine-tuning data)
- Real drone footage from the Russo-Ukrainian War
- Classes: soldiers, APCs, tanks, air fighters, bombers
- Diverse altitudes, angles, and lighting — realistic FPV conditions
- Needs mask annotation (see Annotation section below)
- https://universe.roboflow.com/militaryvehiclerecognition/military-vehicle-recognition

### 2. iSAID (pretraining)
- 655,451 instances across 15 categories in 2,806 high-resolution aerial images
- Standard benchmark for aerial instance segmentation
- Download: https://arxiv.org/pdf/1905.12886
- Dataset page: https://captain-whu.github.io/iSAID/

### 3. Mendeley Military Objects (optional supplement)
- ~7,985 images: tanks, drones, soldiers, civilians
- Mix of real images + GTA5 synthetic data
- https://data.mendeley.com/datasets/rcxfh67zkn/1

### 4. VisDrone (optional supplement — person detection)
- 261,908 frames + 10,209 static images from drone cameras
- 2.6M annotated pedestrians and vehicles
- Good for strengthening "person" detection from aerial viewpoints
- https://docs.ultralytics.com/datasets/detect/visdrone

---

## Annotation Pipeline (Roboflow → Masks)

Roboflow Military dataset ships with bounding boxes only.
Mask R-CNN needs polygon segmentation masks → use SAM2 to auto-generate them.

**Option A — Roboflow Auto-Annotate (easiest)**
- Built into Roboflow UI, SAM-powered
- Upload dataset → Auto-Annotate → export as COCO JSON format for Detectron2
- Zero code required

**Option B — Grounded SAM 2 (fully local, most powerful)**
- Text-prompt driven: feed `"soldier"`, `"tank"` etc. and it finds + masks matching objects
- Fully automatable, runs fine on RTX 4070 Ti Super
- Repo: https://github.com/IDEA-Research/Grounded-SAM-2

**Option C — CVAT + SAM (human-in-the-loop)**
- Open source annotation tool with SAM integration
- Good for reviewing and correcting auto-generated masks

Recommended: start with Option A for speed, fall back to Option B for any classes
where Roboflow's auto-annotator struggles.

---

## Software Stack

```bash
# PyTorch + CUDA (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Detectron2 (Mask R-CNN implementation)
# Build from source for latest version:
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2

# Optional: experiment tracking
pip install wandb
```

Detectron2 model zoo ships COCO-pretrained Mask R-CNN configs:
- `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` — ResNet-50 backbone (fast, good baseline)
- `COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml` — ResNet-101 (better accuracy, still fits in 16GB)

---

## Hardware

- **GPU:** RTX 4070 Ti Super — 16 GB VRAM
- **RAM:** 32 GB DDR5
- **OS:** Nobara Linux (Fedora-based, CUDA works natively via RPM Fusion)

| Config | Batch size | Est. speed |
|--------|-----------|------------|
| ResNet-50-FPN | 4–6 | ~4 img/sec |
| ResNet-101-FPN | 2–4 | ~2.5 img/sec |
| Swin-T/S backbone | 2–4 | ~2 img/sec |

Enable mixed precision (`fp16`) — cuts training time ~30% at no accuracy cost.

**Rough time estimates:**
- iSAID fine-tuning (~1,400 train images, 90k iterations): ~6–8 hours
- Roboflow military fine-tuning (~6,000 images, 30k iterations): ~3–4 hours

---

## Evaluation Metrics

Standard COCO metrics apply:
- **mAP@0.5** — primary detection metric
- **mAP@0.5:0.95** — stricter, averaged across IoU thresholds
- **mAP (mask)** — instance segmentation quality specifically

Compare against YOLOv7 (bounding box) baseline already in this repo
to quantify the benefit of shape-level segmentation for combatant classification.

---

## Research Contribution

The gap in existing literature is **low-altitude FPV drone combatant instance segmentation**.
Contributions of this work:

1. Annotated dataset: Roboflow military footage lifted from bounding boxes to polygon masks via SAM2
2. Trained Mask R-CNN model for aerial combatant detection
3. Comparison study: YOLOv7 (bounding box) vs Mask R-CNN (instance segmentation) on the same data

---

## Key References

1. He et al., "Mask R-CNN", ICCV 2017 — https://arxiv.org/abs/1703.06870
2. Wang et al., "iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images", 2019 — https://arxiv.org/abs/1905.12886
3. Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024 — https://arxiv.org/abs/2408.00714
4. Ren et al., "Faster R-CNN", NeurIPS 2015 — https://arxiv.org/abs/1506.01497
5. Qi et al., "Dynamic Snake Convolution", ICCV 2023 — https://arxiv.org/abs/2307.08388
6. Zheng et al., "Enhancing Geometric Factors: EIoU", IEEE Trans. Cybern. 2021
