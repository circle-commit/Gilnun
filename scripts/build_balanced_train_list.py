"""Write a class-balanced training list for the sidewalk YOLO dataset.

Rare classes such as e-scooters, wheelchairs and strollers appear in well under 1% of
the training images, so the model barely learns them. Repeat factor sampling (from
the LVIS paper) shows those images more often: every class gets
r(c) = max(1, sqrt(t / f(c))), where f(c) is the fraction of training images that
contain the class, and each image is repeated by the largest factor among its classes.

It also samples --val-images validation images for the check after every epoch; the
full validation split is large and its frames are close to training frames anyway. The
held-out test split is for vision/evaluate_app.py.

Run from the repository root after scripts/convert_cvat_to_yolo.py:
    python scripts/build_balanced_train_list.py
Then train with datasets/yolo_sidewalk/data_balanced.yaml.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_DATASET = Path("datasets/yolo_sidewalk")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a class-balanced YOLO training list.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset root containing data.yaml.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Classes in fewer than this fraction of images get repeated (t in r(c) = sqrt(t / f(c))).",
    )
    parser.add_argument("--max-repeat", type=float, default=10.0, help="Upper bound on how often one image repeats.")
    parser.add_argument("--val-images", type=int, default=8000, help="Validation images checked after every epoch.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for rounding repeats and sampling validation images.")
    return parser.parse_args()


def image_classes(images_dir: Path, labels_dir: Path) -> dict[Path, set[int]]:
    classes_by_image = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        lines = label_path.read_text().splitlines() if label_path.exists() else []
        classes_by_image[image_path] = {int(line.split()[0]) for line in lines if line.strip()}
    return classes_by_image


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]

    classes_by_image = image_classes(dataset / "images" / "train", dataset / "labels" / "train")
    if not classes_by_image:
        raise SystemExit(f"No training images found under {dataset / 'images' / 'train'}")

    total = len(classes_by_image)
    images_per_class = Counter(c for classes in classes_by_image.values() for c in classes)
    class_factor = {
        c: max(1.0, math.sqrt(args.threshold / (count / total)))
        for c, count in images_per_class.items()
    }

    rng = random.Random(args.seed)
    lines = []
    balanced_per_class = Counter()
    for image_path, classes in classes_by_image.items():
        factor = min(args.max_repeat, max((class_factor[c] for c in classes), default=1.0))
        # Stochastic rounding keeps the expected number of repeats equal to the factor.
        repeats = int(factor) + (1 if rng.random() < factor - int(factor) else 0)
        # "./" paths are resolved relative to this list file by Ultralytics.
        lines += [f"./images/train/{image_path.name}"] * repeats
        for c in classes:
            balanced_per_class[c] += repeats

    (dataset / "train_balanced.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    val_images = sorted(p.name for p in (dataset / "images" / "val").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    val_sample = sorted(random.Random(args.seed).sample(val_images, min(args.val_images, len(val_images))))
    (dataset / "val_subset.txt").write_text("".join(f"./images/val/{name}\n" for name in val_sample), encoding="utf-8")

    # Only what training reads, so the GPU server needs no test images.
    balanced_config = {"train": "train_balanced.txt", "val": "val_subset.txt", "names": names}
    (dataset / "data_balanced.yaml").write_text(
        yaml.safe_dump(balanced_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print(f"{total} training images -> {len(lines)} entries in {dataset / 'train_balanced.txt'}")
    print(f"{len(val_sample)} of {len(val_images)} validation images -> {dataset / 'val_subset.txt'}")
    print(f"{'class':<26}{'images':>8}{'repeat':>8}{'balanced':>10}")
    for c in sorted(names):
        if c in images_per_class:
            print(f"{names[c]:<26}{images_per_class[c]:>8}{class_factor[c]:>8.1f}{balanced_per_class[c]:>10}")


if __name__ == "__main__":
    main()
