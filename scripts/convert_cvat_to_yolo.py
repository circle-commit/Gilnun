"""Convert the AI Hub 인도보행 영상 bounding-box data (CVAT XML) into a YOLO dataset.

Images are resized and saved as JPEG while they are copied, so the dataset is small
enough to upload to a GPU server: train and val images to --train-size on their long
side (the training image size), test images to --test-size, because
vision/evaluate_app.py crops the app's portrait view out of them.

Splits:
- test: recording folders after Bbox_0410 whose number is a multiple of 20, kept out of
  training entirely. Frames of one recording look alike, so only recordings a model has
  never seen show how it does on new scenes. The previous model was trained on
  Bbox_0001-Bbox_0410 (83,253 images), so these are new to it as well.
- train/val: every other image, split per image by the same seeded hash as before, so the
  previous model's training images are in train and its validation images in val.

Class ids 0-19 are the previous model's classes; the 8 classes after them were added.

Run from the repository root after scripts/download_aihub.py (needs opencv-python,
which ultralytics installs):
    python scripts/convert_cvat_to_yolo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


SOURCE_DIR = Path("datasets/15.인도보행영상/바운딩박스")
OUTPUT_DIR = Path("datasets/yolo_sidewalk")

VAL_RATIO = 0.2
RANDOM_SEED = 42
SPLITS = ("train", "val", "test")
PREVIOUS_MODEL_LAST_FOLDER = 410
TEST_FOLDER_STEP = 20

CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "scooter",
    "wheelchair",
    "stroller",
    "traffic_light",
    "traffic_sign",
    "pole",
    "bollard",
    "bench",
    "tree_trunk",
    "movable_signage",
    "potted_plant",
    "parking_meter",
    "stop",
    "table",
    # Added after the first model.
    "barricade",
    "chair",
    "fire_hydrant",
    "kiosk",
    "carrier",
    "dog",
    "traffic_light_controller",
    "power_controller",
]

class_to_id = {name: i for i, name in enumerate(CLASSES)}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageItem:
    source_path: Path
    # Folder and file name, e.g. Bbox_0001_MP_SEL_000001.jpg. The split is derived from it.
    key: str
    labels: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the CVAT bounding-box data into a YOLO dataset.")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help="Folder with the Bbox_XXXX recordings.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="YOLO dataset folder.")
    parser.add_argument("--train-size", type=int, default=640, help="Long side of train and val images.")
    parser.add_argument("--test-size", type=int, default=1280, help="Long side of test images.")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel processes.")
    return parser.parse_args()


def convert_box(width, height, xtl, ytl, xbr, ybr):
    x_center = ((xtl + xbr) / 2) / width
    y_center = ((ytl + ybr) / 2) / height
    box_width = (xbr - xtl) / width
    box_height = (ybr - ytl) / height
    return x_center, y_center, box_width, box_height


def folder_number(key: str) -> int:
    return int(key.split("_")[1])


def assign_split(key: str) -> str:
    number = folder_number(key)
    if number > PREVIOUS_MODEL_LAST_FOLDER and number % TEST_FOLDER_STEP == 0:
        return "test"
    rng = random.Random(f"{RANDOM_SEED}:{key}")
    return "val" if rng.random() < VAL_RATIO else "train"


def read_annotations(xml_path: Path) -> list[ImageItem]:
    items = []
    folder = xml_path.parent
    for image in ET.parse(xml_path).getroot().findall("image"):
        image_path = folder / image.attrib["name"]
        if not image_path.exists():
            logger.warning("image not found: %s", image_path)
            continue
        width = float(image.attrib["width"])
        height = float(image.attrib["height"])
        labels = []
        for box in image.findall("box"):
            label = box.attrib["label"]
            if label not in class_to_id:
                continue
            x_center, y_center, box_width, box_height = convert_box(
                width,
                height,
                float(box.attrib["xtl"]),
                float(box.attrib["ytl"]),
                float(box.attrib["xbr"]),
                float(box.attrib["ybr"]),
            )
            labels.append(f"{class_to_id[label]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        items.append(ImageItem(image_path, f"{folder.name}_{image_path.name}", tuple(labels)))
    return items


def convert_item(item: ImageItem, split: str, output: Path, long_side: int, quality: int) -> bool:
    """Write the resized JPEG and its label file. Returns False if the image could not be read."""

    stem = Path(item.key).stem
    label_path = output / "labels" / split / f"{stem}.txt"
    label_path.write_text("".join(f"{line}\n" for line in item.labels), encoding="utf-8")

    image_path = output / "images" / split / f"{stem}.jpg"
    if image_path.exists():
        return True
    image = cv2.imread(str(item.source_path), cv2.IMREAD_COLOR)
    if image is None:
        label_path.unlink()
        return False
    height, width = image.shape[:2]
    scale = long_side / max(width, height)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    encoded, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        label_path.unlink()
        return False
    # A temporary name that is not an image, so an interrupted run never leaves a
    # truncated image that training would pick up.
    partial = image_path.with_name(image_path.name + ".tmp")
    partial.write_bytes(data.tobytes())
    partial.replace(image_path)
    return True


def convert_batch(batch: list[tuple[ImageItem, str]], output: Path, sizes: dict[str, int], quality: int) -> list[str]:
    cv2.setNumThreads(1)  # One image per process; the pool provides the parallelism.
    return [item.key for item, split in batch if not convert_item(item, split, output, sizes[split], quality)]


def write_data_yaml(output: Path) -> None:
    # No `path:` key: Ultralytics then resolves the splits relative to this file.
    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    xml_paths = sorted(args.source.rglob("*.xml"))
    with ProcessPoolExecutor(args.workers) as pool:
        items = [item for items in pool.map(read_annotations, xml_paths, chunksize=8) for item in items]

    seen = set()
    unique_items = []
    for item in items:
        if item.key in seen:
            logger.warning("duplicate image skipped: %s", item.source_path)
            continue
        seen.add(item.key)
        unique_items.append(item)

    split_by_key = {item.key: assign_split(item.key) for item in unique_items}
    for split in SPLITS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    sizes = {"train": args.train_size, "val": args.train_size, "test": args.test_size}
    work = [(item, split_by_key[item.key]) for item in unique_items]
    batches = [work[i : i + 256] for i in range(0, len(work), 256)]
    unreadable = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert_batch, batch, args.output, sizes, args.quality) for batch in batches]
        for done, future in enumerate(futures, start=1):
            unreadable += future.result()
            if done % 100 == 0 or done == len(futures):
                logger.info("%d/%d images", min(done * 256, len(work)), len(work))
    for key in unreadable:
        logger.warning("unreadable image skipped: %s", key)
        del split_by_key[key]

    manifest = {"random_seed": RANDOM_SEED, "val_ratio": VAL_RATIO, "splits": dict(sorted(split_by_key.items()))}
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_data_yaml(args.output)

    split_counts = Counter(split_by_key.values())
    logger.info("Images: %s", ", ".join(f"{split} {split_counts[split]}" for split in SPLITS))
    images_per_class = {split: Counter() for split in SPLITS}
    for item in unique_items:
        if item.key in split_by_key:
            images_per_class[split_by_key[item.key]].update({int(line.split()[0]) for line in item.labels})
    logger.info("%-26s%9s%9s%9s", "class (images)", *SPLITS)
    for class_id, name in enumerate(CLASSES):
        logger.info("%-26s%9d%9d%9d", f"{class_id} {name}", *(images_per_class[split][class_id] for split in SPLITS))
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
