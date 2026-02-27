# RetinaNet + ConvNeXt for Medicine Box Detection

A full deep learning pipeline for detecting 12 types of medicine boxes using **RetinaNet with ConvNeXt backbone**.

This project includes:

- Custom YOLO-style dataset loader (supports polygon & bbox)
- RetinaNet implementation from scratch
- ConvNeXt backbone
- COCO-style evaluation (mAP, AP@0.5, AP@0.75)
- Precision / Recall / F1 / Detection Accuracy
- Confusion Matrix
- GT vs Prediction visualization
- Full training & testing pipeline

---

## Problem

Detect 12 types of medicine boxes in images.

Challenges:

- Multi-class object detection
- Small & medium objects
- YOLO-style annotations + polygon format
- Class confusion between similar boxes

---

## Model Architecture

**Detector:** RetinaNet  
**Backbone:** ConvNeXt-Tiny  
**Feature Pyramid Network:** FPN  
**Loss:** Focal Loss + Smooth L1  

Why RetinaNet?

- Handles class imbalance well (Focal Loss)
- Strong one-stage detector
- Suitable for multi-class detection

---

## Project Structure

├── configs.py
├── train.py
├── test.py
├── test_confusion.py
├── test_visualize_gt_pred.py
├── models/
│ ├── retinanet.py
│ ├── convnext.py
│ ├── anchors.py
│ ├── losses.py
│ └── utils.py
├── data/
│ ├── yolo_dataset.py
│ └── transforms.py
├── dataset/
│ ├── train/
│ ├── valid/
│ └── test/
└── runs/
├── best.pt
├── metrics.png
├── confusion_matrix.png
└── compare_results/

## Dataset Format

- YOLO-style format: class x_center y_center width height
- Also supports polygon: class x1 y1 x2 y2 x3 y3 ...
The loader automatically converts polygon → bounding box.

## Training
```bash
uv sync
or
pip install -r requirements.txt
```
```bash
uv run -m train
or
python -m train
```

## Evaluation
```bash
uv run -m test
or
python -m test
```
```bash
uv run -m test_confusion
or
python -m test_confusion
```
```bash
uv run -m test_visualize_gt_pred
or
python -m test_visualize_gt_pred
```

## Metrics

The model evaluates using COCO metrics:

| Metric | Description |
|--------|------------|
| AP@0.50:0.95 | Main detection metric |
| AP@0.50 | Easier threshold |
| AP@0.75 | Stricter threshold |
| AR | Average recall |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-score | Harmonic mean |

---

## Hyperparameters

- Optimizer: AdamW
- Learning rate: configurable
- Gradient clipping: 0.5
- IoU threshold: 0.5
- Confidence threshold: 0.3

---

## Future Improvements

- Per-class AP reporting
- PR curve per class
- EMA weights
- Cosine scheduler
- Anchor auto-tuning (k-means)
- Mixed precision optimization
- TensorBoard logging

---

## Notes

Detection accuracy is computed as: TP / (TP + FP + FN)

## RetinaNet Architecture Diagram
                Input Image
                     │
                     ▼
            ConvNeXt Backbone
                     │
                     ▼
         Feature Pyramid Network (FPN)
                     │
      ┌──────────────┬──────────────┬
      ▼              ▼              ▼
     P3             P4             P5
      │              │              │
      ├────── Classification Head ──────┤
      └────── Box Regression Head ──────┘

### Detailed Pipeline
1. ConvNeXt Backbone
- Modern CNN architecture
- Large kernel depthwise convolutions
- LayerNorm instead of BatchNorm
- Inspired by Transformer design
- Outputs multi-scale feature maps
2. Feature Pyramid Network (FPN)
- Builds multi-scale feature pyramid:
``` {P3,P4,P5,P6,P7} ```

    Purpose:
    - P3 → detect small objects
    - P4–P5 → medium objects
    - P6–P7 → large objects
3. Detection Heads (Shared Across Levels)
- Classification Subnet
    - 4 convolution layers
    - Sigmoid activation
    - Focal Loss
    ```
    Output: A x C
    where:
        A = number of anchors
        C = 12 classes
    ```
- Box Regression Subnet
    - 4 convolution layers
    ```
    Predicts: (x,y,w,h)
    Relative to anchor boxes.
    ```
4. Anchor Mechanism
- Each pyramid level generates:
    - Multiple scales
    - Multiple aspect ratios
    - Dense anchor grid
- Total anchors per image: ~ 100k +