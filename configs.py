from dataclasses import dataclass

@dataclass
class TrainConfig:
    num_classes: int = 12

    # input
    img_size: int = 640
    batch_size: int = 16
    num_workers: int = 4

    # optimization
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0

    # retinanet / anchors
    fpn_out_channels: int = 256
    anchor_sizes: tuple = (32, 64, 128, 256, 512)   # P3..P7
    anchor_ratios: tuple = (0.5, 1.0, 2.0)
    anchor_scales: tuple = (1.0, 2**(1/3), 2**(2/3))

    # losses
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0

    # inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 300
