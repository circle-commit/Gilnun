"""Generate PyTorch reference detections for the on-device detector test.

The Swift test runs the Core ML model on the same portrait crops and checks that
it finds the same objects as the original PyTorch checkpoint.

Run from the repository root (requires ultralytics):
    python frontend/IOS_Swift/Tests/make_detector_golden.py
After retraining, pass the new checkpoint:
    python frontend/IOS_Swift/Tests/make_detector_golden.py --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = REPO_ROOT / "runs/sidewalk/yolo11s_sidewalk/weights/best.pt"
IMAGES_DIR = REPO_ROOT / "test_images"
OUTPUT = Path(__file__).resolve().parent / "detector_golden.json"

# Same thresholds the app passes to the Core ML NMS pipeline.
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 20


def portrait_crop_rect(width: int, height: int) -> tuple[int, int, int, int]:
    """Center crop to the 9:16 portrait aspect ratio of the app's camera frames."""

    crop_width = min(width, round(height * 9 / 16))
    left = (width - crop_width) // 2
    return left, 0, crop_width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PyTorch reference detections for the Swift detector test.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="Checkpoint the app's Core ML model was exported from.")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="Where to write the reference JSON.")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    cases = []

    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        x, y, crop_width, crop_height = portrait_crop_rect(width, height)
        crop = np.ascontiguousarray(image[y : y + crop_height, x : x + crop_width])

        result = model.predict(
            source=crop,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device="cpu",
            verbose=False,
        )[0]

        detections = [
            {
                "label": result.names[int(box.cls[0].item())],
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]
        cases.append(
            {
                "image": str(image_path.relative_to(REPO_ROOT)),
                "crop": [x, y, crop_width, crop_height],
                "detections": detections,
            }
        )

    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(case['detections']) for case in cases)} detections for {len(cases)} images to {args.out}")


if __name__ == "__main__":
    main()
