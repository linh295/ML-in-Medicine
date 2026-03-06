# RetinaNet + ConvNeXt cho bài toán phát hiện hộp thuốc

Pipeline phát hiện đối tượng đa lớp (12 lớp hộp thuốc) dựa trên **RetinaNet + ConvNeXt-Tiny + FPN**, gồm đầy đủ các bước từ nạp dữ liệu YOLO, huấn luyện, đánh giá COCO, confusion matrix và trực quan hóa kết quả.

---

## 1) Tổng quan

### Mục tiêu
- Phát hiện chính xác vị trí và nhãn của 12 loại hộp thuốc.
- Xử lý dữ liệu nhãn thực tế có thể ở dạng YOLO bbox hoặc polygon.
- Cung cấp workflow có thể tái lập và dễ mở rộng cho nghiên cứu tiếp theo.

### Thành phần chính
- **Model**: RetinaNet (one-stage detector).
- **Backbone**: ConvNeXt-Tiny (qua `timm`).
- **Neck**: FPN (Feature Pyramid Network).
- **Loss**: Focal Loss (classification) + SmoothL1 (bbox regression).
- **Evaluation**: COCO mAP/AP/AR + Precision/Recall/F1 + confusion matrix.

---

## 2) Cấu trúc dự án

```text
retinanet_model/
├── configs.py
├── train.py
├── test.py
├── test_confusion.py
├── test_visualize_gt_pred.py
├── temp_gt.json
├── temp_pred.json
├── models/
│   ├── retinanet.py
│   ├── convnext_fpn.py
│   ├── anchors.py
│   ├── losses.py
│   └── utils.py
├── data/
│   ├── yolo_dataset.py
│   └── transforms.py
├── dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── runs/
│   ├── best.pt
│   ├── last.pt
│   ├── metrics.png
│   ├── confusion_matrix.png
│   └── compare_results/
└── requirements.txt
```

---

## 3) Dữ liệu và định dạng nhãn

### Hỗ trợ 2 loại annotation
1. **YOLO bbox**: `class x_center y_center width height`
2. **Polygon**: `class x1 y1 x2 y2 ...`

Trong `data/yolo_dataset.py`:
- Nếu dữ liệu normalized (`<=1.0`) thì được quy đổi về pixel.
- Polygon được chuyển thành bbox bằng `min/max` theo trục `x, y`.
- Box được clamp về biên ảnh và loại các box không hợp lệ/quá nhỏ.

### Lưu ý chuẩn dữ liệu
- Mỗi ảnh phải có file nhãn cùng tên trong thư mục `labels/`.
- Nhãn lớp nên nằm trong khoảng `[0, num_classes-1]` (mặc định `0..11`).

---

## 4) Tiền xử lý và augmentation

`data/transforms.py` triển khai:
- `Resize(size=640)`: resize ảnh về `640x640` và scale bbox tương ứng.
- `RandomHorizontalFlip(p=0.5)`: chỉ dùng ở train.
- `Normalize(mean,std)`: theo chuẩn ImageNet.

Pipelines mặc định:
- **Train**: `Resize -> RandomHorizontalFlip -> Normalize`
- **Validation/Test**: `Resize -> Normalize`

---

## 5) Kiến trúc mô hình

### 5.1 ConvNeXt + FPN
- Backbone ConvNeXt xuất đặc trưng nhiều mức (`C2..C5`).
- FPN thực hiện lateral + top-down fusion.
- Sinh pyramid `P2..P7`, trong đó model dùng `P3..P7` để detect.

### 5.2 Retina Head
Mỗi mức feature map có 2 nhánh:
- **Classification head**: dự đoán xác suất lớp.
- **Regression head**: dự đoán delta bbox so với anchor.

### 5.3 Anchor generator
Thiết lập mặc định (`configs.py`):
- `anchor_sizes = (32, 64, 128, 256, 512)`
- `anchor_ratios = (0.5, 1.0, 2.0)`
- `anchor_scales = (1.0, 2^(1/3), 2^(2/3))`

Số anchor mỗi vị trí: `3 ratios × 3 scales = 9`.

---

## 6) Loss và gán nhãn

Trong `models/retinanet.py`:
- Positive anchor: `IoU >= 0.5`
- Negative anchor: `IoU < 0.4`
- Ignore: `0.4 <= IoU < 0.5`

Loss tổng:
- `loss_cls`: Focal Loss (`alpha=0.25`, `gamma=2.0`)
- `loss_box`: SmoothL1
- `loss_total = loss_cls + loss_box`

---

## 7) Cấu hình huấn luyện mặc định

`configs.py -> TrainConfig` (mặc định):

| Tham số | Giá trị |
|---|---|
| `num_classes` | `12` |
| `img_size` | `640` |
| `batch_size` | `16` |
| `epochs` | `30` |
| `lr` | `1e-4` |
| `weight_decay` | `0.05` |
| `fpn_out_channels` | `256` |
| `score_thresh` | `0.05` |
| `nms_thresh` | `0.5` |
| `max_detections` | `300` |

---

## 8) Cài đặt môi trường

### Cách 1: dùng `uv` (khuyến nghị)
```bash
uv sync
```

### Cách 2: dùng pip
```bash
pip install -r requirements.txt
```

---

## 9) Cách chạy toàn bộ pipeline

### 9.1 Huấn luyện
```bash
python -m train
```
hoặc
```bash
uv run -m train
```

**Đầu ra chính**:
- `runs/last.pt`: trọng số epoch cuối
- `runs/best.pt`: trọng số có AP50:95 tốt nhất
- `runs/metrics.png`: đồ thị train/val loss, AP, AR

### 9.2 Đánh giá chỉ số TP/FP/FN (test script)
```bash
python -m test
```
In ra:
- TP, FP, FN
- Precision, Recall, F1-score
- Detection Accuracy

### 9.3 Confusion matrix
```bash
python -m test_confusion
```
Sinh file:
- `runs/confusion_matrix.png`

### 9.4 Trực quan hóa GT và prediction
```bash
python -m test_visualize_gt_pred
```
Sinh thư mục ảnh:
- `runs/compare_results/`

---

## 10) Ý nghĩa các metric

### COCO metrics (validation)
- **AP@0.50:0.95**: metric chính, phản ánh tổng quan chất lượng detection.
- **AP@0.50**: dễ hơn, thường cao hơn AP chính.
- **AP@0.75**: nghiêm ngặt hơn.
- **AR**: average recall.

### Detection metrics (test.py)
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`
- `Detection Accuracy = TP / (TP + FP + FN)`

---

## 11) Tuỳ chỉnh nhanh

### Đổi số lớp
Sửa trong `configs.py`:
```python
num_classes = 12
```

### Đổi ngưỡng score/NMS khi inference
Sửa trong `configs.py` hoặc script test:
- `score_thresh`
- `nms_thresh`
- `CONF_THRESHOLD` / `IOU_THRESHOLD`

### Đổi backbone
Trong `models/retinanet.py`, tham số `backbone_name` hiện dùng `convnext_tiny`.

---

## 12) Troubleshooting

### 1) Không thấy cải thiện AP
- Kiểm tra mapping class id trong label.
- Kiểm tra bbox có bị sai scale sau resize.
- Thử giảm `lr`, tăng `epochs`, hoặc tinh chỉnh `anchor_sizes`.

### 2) Lỗi `pycocotools`
- Cài lại đúng phiên bản trong `requirements.txt`.
- Đảm bảo môi trường Python đồng nhất (không trộn nhiều env).

### 3) OOM / thiếu VRAM
- Giảm `batch_size`.
- Giảm `img_size`.
- Tắt các process GPU khác.

### 4) Không sinh ra file trong `runs/`
- Kiểm tra quyền ghi thư mục.
- Xem log terminal xem có lỗi dừng sớm (NaN, thiếu dữ liệu, v.v.).

---

## 13) Hướng phát triển

- Báo cáo AP/AR theo từng lớp.
- PR-curve cho từng lớp.
- EMA weights.
- Cosine scheduler + warmup.
- K-means anchor tuning theo phân bố bbox thực tế.
- TensorBoard/W&B logging.
- Benchmark latency/FPS cho triển khai thực tế.

---

## 14) Tài liệu liên quan trong repo

- Báo cáo LaTeX chi tiết: `retinanet_project_report.tex`
- Bản PDF: `retinanet_project_report.pdf`

Nếu bạn cần, mình có thể tạo thêm README bản tiếng Anh hoặc thêm phần “quickstart 5 phút” ở đầu file.
