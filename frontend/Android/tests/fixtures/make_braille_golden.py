"""PyTorch reference masks for the Android braille block segmenter test.

Like the iOS make_braille_golden.py, but on images that are in the repository: the
AI Hub validation photos the iOS golden uses stay out of it. The sidewalk test photos
(no blocks) and the drawn tactile paving fixtures are segmented by the PyTorch checkpoint
on the app's 360x640 portrait frames, and the masks are sampled at the centers of the
app's BrailleMask grid.

Run from the repository root (requires ultralytics):
    python frontend/Android/tests/fixtures/make_braille_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS = REPO_ROOT / "runs/braille/yolo11n_braille/weights/best.pt"
OUTPUT = Path(__file__).resolve().parent / "braille_golden.json"
COLUMNS, ROWS = 45, 80  # BrailleMask.COLUMNS, BrailleMask.ROWS
WIDTH, HEIGHT = 360, 640  # the frames the Android test feeds the segmenter


def frame(path: Path) -> np.ndarray:
    """Center 9:16 crop scaled to 360x640 with bilinear filtering, as the Kotlin test does."""

    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    crop_width = min(width, round(height * 9 / 16))
    left = (width - crop_width) // 2
    return cv2.resize(image[:, left : left + crop_width], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    model = YOLO(str(WEIGHTS))
    images = sorted((REPO_ROOT / "test_images").glob("*.jpg")) + sorted(
        (REPO_ROOT / "frontend/Android/tests/fixtures").glob("*.jpg")
    )
    cases = []
    for path in images:
        # Same mask rasterization as the iOS golden: the model's polygons, filled.
        result = model.predict(frame(path), imgsz=(640, 384), conf=0.25, device="cpu", verbose=False)[0]
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        if result.masks is not None:
            for polygon in result.masks.xy:
                if len(polygon):
                    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        xs = ((np.arange(COLUMNS) + 0.5) / COLUMNS * WIDTH).astype(int)
        ys = ((np.arange(ROWS) + 0.5) / ROWS * HEIGHT).astype(int)
        rows = ["".join(map(str, row)) for row in mask[np.ix_(ys, xs)]]
        cases.append({"image": str(path.relative_to(REPO_ROOT)), "rows": rows})
        print(path.name, sum(r.count("1") for r in rows), "cells")
    OUTPUT.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
