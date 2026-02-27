import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUTPUT_DIR = "runs/compare_results"

CONF_THRESHOLD = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ===============================
# Transform (same as val)
# ===============================
transform = Compose([
    Resize(cfg.img_size),
    Normalize()
])

dataset = YOLODetection(TEST_IMAGE_DIR, TEST_LABEL_DIR, transforms=transform)

class_names = [f"class_{i}" for i in range(cfg.num_classes)]


# ===============================
# Loop through dataset
# ===============================
for idx in tqdm(range(len(dataset))):

    img_tensor, target = dataset[idx]

    img_tensor_input = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model([img_tensor_input[0]])

    pred_boxes = outputs[0]["boxes"].cpu().numpy()
    pred_scores = outputs[0]["scores"].cpu().numpy()
    pred_labels = outputs[0]["labels"].cpu().numpy()

    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()

    # Convert image tensor back to numpy for plotting
    img_np = img_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img_np)

    # ---------------------------
    # Draw GT (GREEN)
    # ---------------------------
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            f"GT: {class_names[label]}",
            color="lime",
            fontsize=8,
            backgroundcolor="black"
        )

    # ---------------------------
    # Draw Prediction (RED)
    # ---------------------------
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y2 + 10,
            f"Pred: {class_names[label]} {score:.2f}",
            color="yellow",
            fontsize=8,
            backgroundcolor="red"
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"img_{idx}.jpg"))
    plt.close()

print("Done. Results saved in runs/compare_results")
