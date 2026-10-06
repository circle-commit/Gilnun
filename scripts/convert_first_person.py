"""Convert AI Hub 1인칭 시점 보행영상 (dataset 159) boxes into the sidewalk YOLO classes.

The dataset labels 79 kinds of objects in frames sampled from walking videos. Kinds that
match a sidewalk class are converted (CATEGORY_MAP). Boxes of other kinds, such as
traffic cones or litter, go to ignore/<split>/ so evaluation neither rewards nor
punishes detections on them. The dataset labels buses and trucks as "car" too, and
does not label traffic lights, traffic signs and some other sidewalk classes at all, so
data.yaml tells vision/evaluate_app.py which classes to score (`scored`) and to score
car, truck and bus as one (`merge`).

Run from the repository root after downloading with scripts/download_aihub.py:
    python scripts/convert_first_person.py --split Validation --out-split test
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import yaml


DEFAULT_ROOT = Path("datasets/aihub/109.1인칭 시점 보행영상 데이터/01.데이터")
DEFAULT_OUTPUT = Path("datasets/first_person")
SIDEWALK_DATA = Path("datasets/yolo_sidewalk/data.yaml")

# Dataset kind -> sidewalk class, checked against sample crops.
CATEGORY_MAP = {
    "person": "person",
    "car": "car",  # Buses and trucks are labeled "car" as well.
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "kickboard": "scooter",
    "stroller": "stroller",
    "powerpole": "pole",
    "tree": "tree_trunk",  # The boxes cover the trunk.
    "pot": "potted_plant",
    "bollard": "bollard",
    "bench": "bench",
    "table": "table",
    "chair": "chair",
    "barrigate": "barricade",
    "stop": "stop",  # Bus stops, as in the sidewalk dataset.
    "signboard": "movable_signage",
    "cart": "carrier",
}
MERGED = [["car", "truck", "bus"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert first-person walking video boxes into the sidewalk YOLO classes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="The dataset's 01.데이터 folder.")
    parser.add_argument("--split", choices=["Training", "Validation"], default="Validation", help="Dataset split to convert.")
    parser.add_argument("--viewpoint", default="Average_stature", help="Camera height folder (Average_stature is 165 cm).")
    parser.add_argument("--place", choices=["out", "in"], default="out", help="Outdoor or indoor recordings.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="YOLO dataset folder to write.")
    parser.add_argument("--out-split", default="test", help="Split name to write (images/<name>, labels/<name>).")
    parser.add_argument("--size", type=int, default=1280, help="Long side of the saved images.")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel processes.")
    return parser.parse_args()


def split_folder(root: Path, split: str) -> Path:
    return root / f"{1 if split == 'Training' else 2}.{split}" / "BBOX"


def yolo_line(cls: int, box: dict, width: int, height: int) -> str:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    return f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def convert_frame(job: tuple) -> bool:
    image_path, stem, boxes, output, split, size, quality, class_ids = job
    cv2.setNumThreads(1)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    height, width = image.shape[:2]

    labels, ignored = [], []
    for box in boxes:
        target = CATEGORY_MAP.get(box["category_name"])
        if box["box"]["w"] <= 0 or box["box"]["h"] <= 0:
            continue
        if target is None:
            ignored.append(yolo_line(-1, box["box"], width, height))
        else:
            labels.append(yolo_line(class_ids[target], box["box"], width, height))

    scale = size / max(width, height)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    encoded, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        return False
    (output / "labels" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in labels), encoding="utf-8")
    (output / "ignore" / split / f"{stem}.txt").write_text("".join(f"{line}\n" for line in ignored), encoding="utf-8")
    partial = output / "images" / split / f"{stem}.jpg.tmp"
    partial.write_bytes(data.tobytes())
    partial.replace(output / "images" / split / f"{stem}.jpg")
    return True


def main() -> None:
    args = parse_args()
    names = yaml.safe_load(SIDEWALK_DATA.read_text(encoding="utf-8"))["names"]
    class_ids = {name: int(cls) for cls, name in names.items()}

    base = split_folder(args.root, args.split)
    label_root = next(base.glob("라벨링데이터*")) / args.viewpoint / args.place
    image_root = next((base / "원천데이터").glob("*")) / args.viewpoint / args.place
    images = {path.stem: path for path in image_root.rglob("*.png")}
    print(f"{len(images)} images under {image_root}")

    frames, missing = {}, 0
    for label_path in sorted(label_root.rglob("*.json")):
        annotation = json.loads(label_path.read_text(encoding="utf-8"))["annotation"]
        for frame in annotation["annotations"]:
            name = Path(frame["atchOrgFileName"]).stem
            stem = f"{label_path.stem}__{name}"
            if stem not in images and name in images:
                stem = name  # A few frames name the image with its video already.
            if stem not in images:
                missing += 1
                continue
            boxes = frame.get("box") or []
            # Some frames are labeled twice; the fuller labeling includes the other one.
            if len(boxes) >= len(frames.get(stem, [])):
                frames[stem] = boxes
    print(f"{len(frames)} labeled frames with images, {missing} labeled frames without")

    kinds = Counter(box["category_name"] for boxes in frames.values() for box in boxes)
    jobs = [
        (images[stem], stem, boxes, args.output, args.out_split, args.size, args.quality, class_ids)
        for stem, boxes in frames.items()
    ]

    for folder in ("images", "labels", "ignore"):
        (args.output / folder / args.out_split).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        written = sum(pool.map(convert_frame, jobs, chunksize=32))
    print(f"{written} frames written to {args.output}")

    data_yaml = args.output / "data.yaml"
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if data_yaml.exists() else {}
    config[args.out_split] = f"images/{args.out_split}"
    config["names"] = names
    config["scored"] = sorted({target for target in CATEGORY_MAP.values()} | {name for group in MERGED for name in group}, key=class_ids.get)
    config["merge"] = MERGED
    data_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"\n{'kind':<20}{'boxes':>8}  sidewalk class")
    for kind, count in kinds.most_common():
        print(f"{kind:<20}{count:>8}  {CATEGORY_MAP.get(kind, '(ignored)')}")


if __name__ == "__main__":
    main()
