"""Export the trained sidewalk YOLOv8n model to Core ML for the iOS app.

Run from the repository root on macOS (requires ultralytics and coremltools):
    python -m vision.export_coreml

The exported model includes Apple's NMS pipeline, so the app receives
per-box class confidences and normalized xywh coordinates directly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("runs/detect/runs/sidewalk/yolov8n_sidewalk-3/weights/best.pt")
DEFAULT_OUTPUT = Path("frontend/IOS_Swift/Glass/SidewalkDetector.mlpackage")

# The app analyzes upright portrait frames (1080x1920). A 640x384 (height x width)
# input is exactly what PyTorch rect inference uses for that aspect ratio at
# imgsz=640, so on-device results match the training/validation setup.
DEFAULT_IMGSZ = (640, 384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the sidewalk YOLO model to Core ML.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .mlpackage path.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=DEFAULT_IMGSZ,
        metavar=("HEIGHT", "WIDTH"),
        help="Model input size. Both values must be multiples of 32.",
    )
    return parser.parse_args()


def export() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.weights}")

    model = YOLO(str(args.weights))
    exported = Path(model.export(format="coreml", imgsz=list(args.imgsz), nms=True))

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(args.output))
    print(f"Core ML model written to: {args.output}")


if __name__ == "__main__":
    export()
