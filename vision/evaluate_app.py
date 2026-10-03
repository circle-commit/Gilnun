"""Measure detection quality the way the iOS app uses the model.

Ultralytics' mAP counts every labelled object, but most labels are tiny, far-away
objects that the app ignores. This script reports per-class precision and recall
under the app's conditions, on the test split by default (recordings that were kept out
of training; see scripts/convert_cvat_to_yolo.py):
- each validation image is center-cropped to the app's 9:16 portrait view,
- only boxes covering at least --min-area of that view count (labels and detections),
- detections use the app's confidence threshold, and a detection matches a label of
  the same class when their IoU is at least --match-iou.
Only classes the model knows are scored, so models with fewer classes compare fairly.

Run from the repository root (use --out to save a report for comparing models):
    python -m vision.evaluate_app --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt --out yolo11s.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from vision.device import default_device


DEFAULT_DATA = Path("datasets/yolo_sidewalk/data.yaml")
# The model in the app; pass --weights to score a new one against it.
DEFAULT_WEIGHTS = Path("runs/sidewalk/yolo11s_sidewalk/weights/best.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Same settings as frontend/IOS_Swift/Gilnun/ObjectDetector.swift and SceneAnalyzer.swift.
APP_CONFIDENCE = 0.25
APP_NMS_IOU = 0.5
APP_MAX_DETECTIONS = 20
APP_MIN_AREA = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector under the iOS app's conditions.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Checkpoint to evaluate (.pt).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml.")
    parser.add_argument("--split", default="test", help="Split in data.yaml to evaluate.")
    parser.add_argument("--conf", type=float, default=APP_CONFIDENCE, help="Confidence threshold.")
    parser.add_argument("--min-area", type=float, default=APP_MIN_AREA, help="Smallest box, as a fraction of the view.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU needed for a detection to match a label.")
    parser.add_argument("--full-frame", action="store_true", help="Use the whole image instead of the portrait crop.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images (quick check).")
    parser.add_argument("--device", default=None, help="'0', 'mps', or 'cpu'. Auto-detected by default.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report as JSON.")
    return parser.parse_args()


def view_rect(width: int, height: int, full_frame: bool) -> tuple[int, int, int, int]:
    """The app's portrait view: a centered 9:16 crop (same as make_detector_golden.py)."""

    if full_frame:
        return 0, 0, width, height
    crop_width = min(width, round(height * 9 / 16))
    return (width - crop_width) // 2, 0, crop_width, height


def area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def labels_in_view(label_path: Path, width: int, height: int, rect: tuple[int, int, int, int]) -> list[tuple[int, tuple]]:
    """Labels moved into view coordinates; boxes mostly outside the crop are dropped."""

    left, top, view_width, view_height = rect
    boxes = []
    lines = label_path.read_text().splitlines() if label_path.exists() else []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:])
        full = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
        clipped = (
            max(full[0], left) - left,
            max(full[1], top) - top,
            min(full[2], left + view_width) - left,
            min(full[3], top + view_height) - top,
        )
        if area(full) > 0 and area(clipped) / area(full) >= 0.5:
            boxes.append((cls, clipped))
    return boxes


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = config["names"]
    if args.split not in config:
        raise SystemExit(f"{args.data} has no {args.split} split")
    root = Path(config["path"]) if config.get("path") else args.data.parent
    val_dir = root / config[args.split]
    images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {val_dir}")

    model = YOLO(str(args.weights))
    device = args.device or default_device()
    # Match classes by name: the model's ids map onto the dataset's ids, and dataset
    # classes the model does not know are left out of the score.
    data_id = {name: class_id for class_id, name in names.items()}
    model_to_data = {model_id: data_id[name] for model_id, name in model.names.items() if name in data_id}
    scored = set(model_to_data.values())
    skipped = [names[class_id] for class_id in sorted(names) if class_id not in scored]
    if skipped:
        print(f"Not scored (unknown to this model): {', '.join(skipped)}")
    counts = defaultdict(lambda: {"labels": 0, "tp": 0, "fp": 0})

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        rect = view_rect(width, height, args.full_frame)
        left, top, view_width, view_height = rect
        view = image[top : top + view_height, left : left + view_width]
        view_area = view_width * view_height

        label_path = val_dir.parent.parent / "labels" / val_dir.name / f"{image_path.stem}.txt"
        labels = [
            (cls, box)
            for cls, box in labels_in_view(label_path, width, height, rect)
            if cls in scored and area(box) / view_area >= args.min_area
        ]

        result = model.predict(
            view,
            imgsz=640,
            conf=args.conf,
            iou=APP_NMS_IOU,
            max_det=APP_MAX_DETECTIONS,
            device=device,
            verbose=False,
        )[0]
        detections = [
            (model_to_data[int(cls)], float(score), tuple(float(v) for v in xyxy))
            for cls, score, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist())
            if int(cls) in model_to_data and area(tuple(xyxy)) / view_area >= args.min_area
        ]

        # Greedy matching per class, most confident detections first.
        for cls in {c for c, _ in labels} | {c for c, _, _ in detections}:
            remaining = [box for c, box in labels if c == cls]
            counts[cls]["labels"] += len(remaining)
            for _, _, box in sorted((d for d in detections if d[0] == cls), key=lambda d: -d[1]):
                best = max(remaining, key=lambda label: iou(box, label), default=None)
                if best is not None and iou(box, best) >= args.match_iou:
                    remaining.remove(best)
                    counts[cls]["tp"] += 1
                else:
                    counts[cls]["fp"] += 1

        if index % 500 == 0:
            print(f"{index}/{len(images)} images")

    report = {
        "weights": str(args.weights),
        "split": args.split,
        "images": len(images),
        "conf": args.conf,
        "min_area": args.min_area,
        "classes": {},
    }
    print(f"\n{'class':<16}{'labels':>8}{'recall':>9}{'precision':>11}")
    total = {"labels": 0, "tp": 0, "fp": 0}
    for cls in sorted(scored):
        c = counts[cls]
        recall = c["tp"] / c["labels"] if c["labels"] else None
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        report["classes"][names[cls]] = {**c, "recall": recall, "precision": precision}
        for key in total:
            total[key] += c[key]
        fmt = lambda value: f"{value:.3f}" if value is not None else "-"
        print(f"{names[cls]:<16}{c['labels']:>8}{fmt(recall):>9}{fmt(precision):>11}")

    overall_recall = total["tp"] / total["labels"] if total["labels"] else 0.0
    overall_precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    report["overall"] = {**total, "recall": overall_recall, "precision": overall_precision}
    print(f"{'all':<16}{total['labels']:>8}{overall_recall:>9.3f}{overall_precision:>11.3f}")

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
