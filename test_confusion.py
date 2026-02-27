import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from configs import TrainConfig
from models.retinanet import RetinaNet
from data.yolo_dataset import YOLODetection
from data.transforms import Resize, Normalize, Compose


# ===============================
# CONFIG
# ===============================
WEIGHT_PATH = "runs/best.pt"
TEST_IMAGE_DIR = "dataset/test/images"
TEST_LABEL_DIR = "dataset/test/labels"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# ===============================
# IoU
# ===============================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0
    return inter / union


# ===============================
# Load model
# ===============================
cfg = TrainConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = RetinaNet(
    num_classes=cfg.num_classes,
    backbone_name="convnext_tiny",
    pretrained=False,
    fpn_out_channels=cfg.fpn_out_channels,
    anchor_sizes=cfg.anchor_sizes,
    anchor_ratios=cfg.anchor_ratios,
    anchor_scales=cfg.anchor_scales,
).to(device)

model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()

transform = Compose([
    Resize(cfg.img_size),
    Normalize()
])

dataset = YOLODetection(TEST_IMAGE_DIR, TEST_LABEL_DIR, transforms=transform)

num_classes = cfg.num_classes
background = num_classes

conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)


# ===============================
# Evaluation loop
# ===============================
for img_tensor, target in tqdm(dataset):

    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model([img_tensor[0]])

    pred_boxes = outputs[0]["boxes"].cpu().numpy()
    pred_scores = outputs[0]["scores"].cpu().numpy()
    pred_labels = outputs[0]["labels"].cpu().numpy()

    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()

    matched_gt = set()

    for box, score, pred_label in zip(pred_boxes, pred_scores, pred_labels):

        if score < CONF_THRESHOLD:
            continue

        best_iou = 0
        best_gt_idx = -1

        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched_gt:
                continue

            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou > IOU_THRESHOLD:
            gt_label = gt_labels[best_gt_idx]
            conf_matrix[gt_label, pred_label] += 1
            matched_gt.add(best_gt_idx)
        else:
            # False positive
            conf_matrix[background, pred_label] += 1

    # False negatives
    for i, gt_label in enumerate(gt_labels):
        if i not in matched_gt:
            conf_matrix[gt_label, background] += 1


# ===============================
# Plot confusion matrix
# ===============================
class_names = [f"class_{i}" for i in range(num_classes)] + ["background"]

plt.figure(figsize=(12,10))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Detection Confusion Matrix (IoU=0.5)")
plt.tight_layout()
plt.savefig("runs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved to runs/confusion_matrix.png")