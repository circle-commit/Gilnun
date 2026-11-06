"""Convert the AI Hub 인도보행 영상 surface data (CVAT polygons) into a YOLO segmentation
dataset with one class, braille_guide_blocks, and zip it for uploading to Google Drive.

Every image with a braille block is kept, plus a share of images without one so the model
learns to stay quiet on ordinary pavement. Damaged blocks are still braille blocks, so they
share the class. Frames of one recording look alike, so val is whole recording folders
(every fifth Surface_XXX), never seen in training.

Run from the repository root (needs opencv-python, which ultralytics installs):
    .venv/bin/python scripts/convert_surface_braille.py
Then upload datasets/braille_seg.zip to Google Drive and open vision/train_braille_colab.ipynb.
The zip is AI Hub data: keep it in your own Drive and never share it.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


SOURCE_DIR = Path("datasets/15.인도보행영상/서페이스")
OUTPUT_DIR = Path("datasets/braille_seg")
LABEL = "braille_guide_blocks"
VAL_FOLDER_STEP = 5
NEGATIVE_RATIO = 0.3
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert surface polygons into a braille-block YOLO-seg dataset.")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help="Folder with the Surface_XXX recordings.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Dataset folder; a .zip is written next to it.")
    parser.add_argument("--size", type=int, default=640, help="Long side of saved images.")
    parser.add_argument("--negatives", type=float, default=NEGATIVE_RATIO, help="Share of images without blocks to keep.")
    return parser.parse_args()


def yolo_line(points: str, width: float, height: float) -> str:
    coords = []
    for pair in points.split(";"):
        x, y = map(float, pair.split(","))
        coords += [min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)]
    return "0 " + " ".join(f"{c:.5f}" for c in coords)


def main() -> None:
    args = parse_args()
    rng = random.Random(RANDOM_SEED)
    if args.output.exists():
        shutil.rmtree(args.output)
    counts = {"train": [0, 0], "val": [0, 0]}  # [with blocks, without]

    for xml_path in sorted(args.source.glob("Surface_*/*.xml")):
        folder = xml_path.parent
        split = "val" if int(folder.name.split("_")[1]) % VAL_FOLDER_STEP == 0 else "train"
        for image in ET.parse(xml_path).getroot().iter("image"):
            width, height = float(image.get("width")), float(image.get("height"))
            lines = [
                yolo_line(p.get("points"), width, height)
                for p in image.iter("polygon")
                if p.get("label") == LABEL
            ]
            if not lines and rng.random() >= args.negatives:
                continue
            source = folder / image.get("name")
            pixels = cv2.imread(str(source))
            if pixels is None:
                print(f"skipped unreadable {source}")
                continue
            scale = args.size / max(pixels.shape[:2])
            if scale < 1:
                pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            stem = f"{folder.name}_{source.stem}"
            for kind in ("images", "labels"):
                (args.output / kind / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output / "images" / split / f"{stem}.jpg"), pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (args.output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            counts[split][0 if lines else 1] += 1

    (args.output / "data.yaml").write_text(
        f"train: images/train\nval: images/val\nnames:\n  0: {LABEL}\n", encoding="utf-8"
    )
    for split, (positive, negative) in counts.items():
        print(f"{split}: {positive} with blocks, {negative} without")
    archive = shutil.make_archive(str(args.output), "zip", args.output.parent, args.output.name)
    print(f"{Path(archive).stat().st_size / 1024**2:.0f} MB -> {archive}")


if __name__ == "__main__":
    main()
