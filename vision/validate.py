"""Validate a trained YOLOv8 checkpoint.

Run from the repository root:
    python -m vision.validate --weights runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from vision.device import default_device


DEFAULT_DATA = Path("datasets/yolo_sidewalk/data.yaml")
DEFAULT_WEIGHTS = Path("runs/sidewalk/yolo11s_sidewalk/weights/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 sidewalk detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to YOLO data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for metrics.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="'0', 'mps', or 'cpu'. Auto-detected by default.")
    return parser.parse_args()


def validate() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(str(args.weights))

    # split="val" uses datasets/yolo_sidewalk/images/val from data.yaml.
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or default_device(),
        split="val",
        plots=True,
    )
    print(metrics)


if __name__ == "__main__":
    validate()
