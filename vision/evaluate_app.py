"""Measure detection quality the way the iOS app uses the model.

Ultralytics' mAP counts every labelled object, but most labels are tiny, far-away
objects that the app ignores. This script reports per-class precision and recall
under the app's conditions, on the test split by default (recordings that were kept out
of training; see scripts/convert_cvat_to_yolo.py):
- each validation image is center-cropped to the app's 9:16 portrait view,
- only boxes covering at least --min-area of that view count (labels and detections),
- detections use the app's confidence threshold, and a detection matches a label of
  the same class when their IoU is at least --match-iou;
- objects that the 9:16 crop cuts to less than half of their box are not scored either
  way: a detection of the visible part is neither a hit nor a false alarm.
Only classes the model knows are scored, so models with fewer classes compare fairly.
Datasets that label only some classes can say so in their data.yaml: `scored` lists the
classes to score, `merge` lists classes labeled as one (scored as the first), and boxes
in ignore/<split>/ mark objects the dataset labels but this model has no class for;
detections on them are not counted as false alarms (scripts/convert_first_person.py).

Run from the repository root (use --out to save a report for comparing models):
    python -m vision.evaluate_app --weights runs/sidewalk/yolo11s_sidewalk/weights/best.pt --out yolo11s.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
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


def labels_in_view(
    label_path: Path, width: int, height: int, rect: tuple[int, int, int, int]
) -> tuple[list[tuple[int, tuple]], list[tuple[int, tuple]]]:
    """Labels moved into view coordinates, and the cut-off ones separately.

    A box with less than half of it inside the crop is "cut off": too little of the
    object shows to require a detection, but detecting it is not a false alarm.
    """

    left, top, view_width, view_height = rect
    boxes = []
    cut_off = []
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
        if area(full) <= 0 or area(clipped) <= 0:
            continue
        if area(clipped) / area(full) >= 0.5:
            boxes.append((cls, clipped))
        else:
            cut_off.append((cls, clipped))
    return boxes, cut_off


def scoring_rules(config: dict, known: set[int]) -> tuple[set[int], Callable[[int], int]]:
    """Dataset classes to score, and the class each one is scored as.

    `known` holds the dataset ids of the classes the model knows. A data.yaml can narrow
    them with `scored` and have classes it labels as one scored together with `merge`.
    """

    data_id = {name: class_id for class_id, name in config["names"].items()}
    scored = set(known)
    if "scored" in config:
        scored &= {data_id[name] for name in config["scored"] if name in data_id}
    merged_into = {}
    for group in config.get("merge", []):
        members = [data_id[name] for name in group if name in data_id]
        for class_id in members:
            merged_into[class_id] = members[0]
    return scored, lambda class_id: merged_into.get(class_id, class_id)


def ignored_in_view(image_dir: Path, stem: str, width: int, height: int, rect: tuple[int, int, int, int]) -> list[tuple]:
    """Boxes from ignore/<split>/: objects the dataset labels as kinds no class stands for."""

    path = image_dir.parent.parent / "ignore" / image_dir.name / f"{stem}.txt"
    return [box for boxes in labels_in_view(path, width, height, rect) for _, box in boxes]


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
    scored, unify = scoring_rules(config, set(model_to_data.values()))
    skipped = [names[class_id] for class_id in sorted(names) if class_id not in scored]
    if skipped:
        print(f"Not scored (unknown to this model or not labeled in this dataset): {', '.join(skipped)}")
    counts = defaultdict(lambda: {"labels": 0, "tp": 0, "fp": 0})

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        rect = view_rect(width, height, args.full_frame)
        left, top, view_width, view_height = rect
        view = image[top : top + view_height, left : left + view_width]
        view_area = view_width * view_height

        label_path = val_dir.parent.parent / "labels" / val_dir.name / f"{image_path.stem}.txt"
        in_view, cut_off = labels_in_view(label_path, width, height, rect)
        labels = [(unify(cls), box) for cls, box in in_view if cls in scored and area(box) / view_area >= args.min_area]
        cut_off = [(unify(cls), box) for cls, box in cut_off]
        ignored = ignored_in_view(val_dir, image_path.stem, width, height, rect)

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
            (unify(model_to_data[int(cls)]), float(score), tuple(float(v) for v in xyxy))
            for cls, score, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist())
            if model_to_data.get(int(cls)) in scored and area(tuple(xyxy)) / view_area >= args.min_area
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
                elif not any(c == cls and iou(box, cut) >= args.match_iou for c, cut in cut_off) and not any(
                    iou(box, other) >= args.match_iou for other in ignored
                ):
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
    for cls in sorted({unify(class_id) for class_id in scored}):
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
