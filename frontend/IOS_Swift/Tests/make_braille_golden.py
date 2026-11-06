"""Generate PyTorch reference masks for the on-device braille block segmenter test.

Portrait crops of validation images from datasets/braille_seg (made by
scripts/convert_surface_braille.py) are segmented by the PyTorch checkpoint, and the masks
are sampled at the centers of the app's BrailleMask grid. The images are AI Hub data and
stay out of the repository, so the Swift test skips cases whose image is missing.

Run from the repository root (requires ultralytics):
    .venv/bin/python frontend/IOS_Swift/Tests/make_braille_golden.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = REPO_ROOT / "runs/braille/yolo11n_braille/weights/best.pt"
IMAGES_DIR = REPO_ROOT / "datasets/braille_seg/images/val"
OUTPUT = Path(__file__).resolve().parent / "braille_golden.json"
COLUMNS, ROWS = 45, 80  # BrailleMask.columns, BrailleMask.rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PyTorch reference masks for the Swift segmenter test.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    label_dir = IMAGES_DIR.parents[1] / "labels" / IMAGES_DIR.name
    labeled = [p for p in images if (label_dir / f"{p.stem}.txt").read_text().strip()]
    unlabeled = [p for p in images if p not in set(labeled)]
    # Mostly frames with blocks, plus some without to check the model stays quiet.
    picked = labeled[:: max(1, len(labeled) // 12)][:12] + unlabeled[:: max(1, len(unlabeled) // 4)][:4]
    cases = []
    for image_path in picked:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        crop_width = min(width, round(height * 9 / 16))
        left = (width - crop_width) // 2
        crop = np.ascontiguousarray(image[:, left : left + crop_width])

        result = model.predict(crop, imgsz=(640, 384), conf=0.25, verbose=False)[0]
        mask = np.zeros((height, crop_width), np.uint8)
        if result.masks is not None:
            for polygon in result.masks.xy:
                if len(polygon):
                    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        xs = ((np.arange(COLUMNS) + 0.5) / COLUMNS * crop_width).astype(int)
        ys = ((np.arange(ROWS) + 0.5) / ROWS * height).astype(int)
        grid = mask[np.ix_(ys, xs)]
        cases.append({
            "image": str(image_path.relative_to(REPO_ROOT)),
            "crop": [left, 0, crop_width, height],
            "rows": ["".join(map(str, row)) for row in grid],
        })

    args.out.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases, {sum(any('1' in r for r in c['rows']) for c in cases)} with blocks -> {args.out}")


if __name__ == "__main__":
    main()
