"""Add the objects the sidewalk labels missed, found by the current model (pseudo-labels).

The labels leave out many plainly visible objects, such as far poles, parked cars and
people in a crowd. On the test split, about 9 in 10 of the model's detections at
confidence 0.5-0.7 that no label covers were real objects (checked by hand). Training on
such images teaches the model that those objects are background. This script runs the
model over every image that training reads (train_balanced.txt and val_subset.txt) and
writes a copy of the dataset whose labels also hold the model's detections that:
- have confidence of at least --conf, or --weak-conf for WEAK_CLASSES;
- cover at least --min-area of the image (the app ignores smaller objects anyway);
- overlap no label: IoU below 0.3, and neither box half inside the other, so parts of a
  labeled object and boxes around several labeled objects are not added.
Images are not copied: images/ in the copy links to the original dataset's images.

Run from the repository root after scripts/build_balanced_train_list.py. It skips images
it has already labeled, so it can be run again after an interruption (delete the output
folder to start over with other settings):
    python scripts/pseudo_label.py
    python scripts/pack_training_data.py --dataset datasets/yolo_sidewalk_pl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.device import default_device  # noqa: E402


DEFAULT_DATASET = Path("datasets/yolo_sidewalk")
DEFAULT_OUTPUT = Path("datasets/yolo_sidewalk_pl")
DEFAULT_WEIGHTS = Path("runs/sidewalk/yolo11m_sidewalk/weights/best.pt")
LIST_FILES = ("data_balanced.yaml", "train_balanced.txt", "val_subset.txt")

# Rare classes the model is less sure about (test precision below 0.75 with few examples).
# Pole also scores below 0.75, but its unlabeled detections checked by hand were real poles.
WEAK_CLASSES = {
    "parking_meter", "traffic_light_controller", "power_controller", "carrier", "table",
    "chair", "bench", "barricade", "scooter", "potted_plant",
}
NMS_IOU = 0.5  # as in the app
MAX_LABEL_IOU = 0.3
MAX_INSIDE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add the model's confident detections that the labels missed.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset root with data_balanced.yaml.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Dataset copy to write.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Model that finds the missing objects.")
    parser.add_argument("--conf", type=float, default=0.5, help="Lowest confidence added.")
    parser.add_argument("--weak-conf", type=float, default=0.7, help="Lowest confidence added for WEAK_CLASSES.")
    parser.add_argument("--min-area", type=float, default=0.003, help="Smallest box added, as a fraction of the image.")
    parser.add_argument("--batch", type=int, default=64, help="Images per model call.")
    parser.add_argument("--device", default=None, help="'0', 'mps', or 'cpu'. Auto-detected by default.")
    return parser.parse_args()


def listed_images(list_path: Path) -> list[str]:
    """Unique image paths (relative to the dataset root) in a train or val list."""

    names = {line.strip().removeprefix("./") for line in list_path.read_text(encoding="utf-8").splitlines()}
    return sorted(name for name in names if name)


def label_path(root: Path, image: str) -> Path:
    split, name = Path(image).parts[-2:]
    return root / "labels" / split / f"{Path(name).stem}.txt"


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO label lines as (class, x1, y1, x2, y2) in normalized coordinates."""

    boxes = []
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            cls, cx, cy, w, h = int(parts[0]), *(float(value) for value in parts[1:])
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlaps(box: tuple[float, float, float, float], label: tuple[float, float, float, float]) -> bool:
    inter = area((max(box[0], label[0]), max(box[1], label[1]), min(box[2], label[2]), min(box[3], label[3])))
    if inter <= 0:
        return False
    union = area(box) + area(label) - inter
    return inter / union >= MAX_LABEL_IOU or inter / min(area(box), area(label)) >= MAX_INSIDE


def prepare_output(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for name in LIST_FILES:
        shutil.copy2(args.dataset / name, args.output / name)
    images = args.output / "images"
    if not images.exists():
        images.symlink_to(os.path.relpath((args.dataset / "images").resolve(), args.output.resolve()))


def main() -> None:
    args = parse_args()
    missing = [name for name in LIST_FILES if not (args.dataset / name).exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in {args.dataset}; run scripts/build_balanced_train_list.py first.")
    prepare_output(args)

    images = listed_images(args.dataset / "train_balanced.txt") + listed_images(args.dataset / "val_subset.txt")
    todo = [image for image in images if not label_path(args.output, image).exists()]
    print(f"{len(images)} images, {len(images) - len(todo)} already labeled, {len(todo)} to go")

    model = YOLO(str(args.weights))
    names = model.names
    device = args.device or default_device()
    for start in range(0, len(todo), args.batch):
        batch = todo[start : start + args.batch]
        results = model.predict(
            [str(args.dataset / image) for image in batch],
            imgsz=640,
            conf=min(args.conf, args.weak_conf),
            iou=NMS_IOU,
            device=device,
            verbose=False,
        )
        for image, result in zip(batch, results):
            labels = read_boxes(label_path(args.dataset, image))
            added = []
            for cls, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxyn.tolist()):
                cls, box = int(cls), tuple(box)
                if score < (args.weak_conf if names[cls] in WEAK_CLASSES else args.conf) or area(box) < args.min_area:
                    continue
                if any(overlaps(box, label[1:]) for label in labels + added):
                    continue
                added.append((cls, *box))
            original = label_path(args.dataset, image)
            text = original.read_text(encoding="utf-8") if original.exists() else ""
            if text and not text.endswith("\n"):
                text += "\n"
            for cls, x1, y1, x2, y2 in added:
                text += f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n"
            target = label_path(args.output, image)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".tmp")
            partial.write_text(text, encoding="utf-8")
            partial.replace(target)
        done = start + len(batch)
        if done % (args.batch * 100) < args.batch or done == len(todo):
            print(f"{done}/{len(todo)} images")

    # Count what was added, over every image (including ones labeled by an earlier run).
    added_by_class, changed = Counter(), 0
    for image in images:
        before = Counter(box[0] for box in read_boxes(label_path(args.dataset, image)))
        after = Counter(box[0] for box in read_boxes(label_path(args.output, image)))
        if after != before:
            changed += 1
        added_by_class.update(after - before)
    summary = {
        "weights": str(args.weights),
        "conf": args.conf,
        "weak_conf": args.weak_conf,
        "min_area": args.min_area,
        "images": len(images),
        "images_changed": changed,
        "added": {names[cls]: count for cls, count in added_by_class.most_common()},
    }
    (args.output / "pseudo_labels.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{changed} of {len(images)} images got new boxes ({sum(added_by_class.values())} in all):")
    for cls, count in added_by_class.most_common():
        print(f"  {names[cls]:<26}{count:>8}")


if __name__ == "__main__":
    main()
