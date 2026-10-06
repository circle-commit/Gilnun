"""Break down detection errors under the iOS app's conditions.

Runs the model once over the test split (the app's 9:16 portrait view, objects covering
at least 1% of it, as in vision/evaluate_app.py) and caches every detection down to
--min-conf, so other thresholds or scoring rules can be studied with --reuse without
running the model again. Then it reports, per class:
- misses: nothing was detected there, or the object was detected as another class;
- false alarms: on background, on an object of another class, or a duplicate box;
the same scores by guidance group (classes the app warns about the same way), and saves
crops of the most confident false alarms and of missed objects for inspection.

Run from the repository root:
    python -m vision.analyze_errors
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from vision.device import default_device
from vision.evaluate_app import (
    APP_CONFIDENCE,
    APP_MAX_DETECTIONS,
    APP_MIN_AREA,
    APP_NMS_IOU,
    IMAGE_SUFFIXES,
    area,
    iou,
    labels_in_view,
    view_rect,
)


DEFAULT_WEIGHTS = Path("runs/sidewalk/yolo11s_sidewalk/weights/best.pt")
DEFAULT_DATA = Path("datasets/yolo_sidewalk/data.yaml")
DEFAULT_OUT = Path("runs/error_analysis")
MATCH_IOU = 0.5

# Classes the app warns about the same way: GuidanceEngine's risk sets.
GUIDANCE_GROUPS = {
    "vehicle": ["car", "truck", "bus"],
    "two_wheeler": ["bicycle", "motorcycle", "scooter"],
    "fixed_obstacle": ["pole", "bollard", "tree_trunk", "movable_signage", "barricade", "fire_hydrant"],
    "person_mobility": ["person", "wheelchair", "stroller", "carrier", "dog"],
    "low_risk": ["bench", "potted_plant", "traffic_light", "traffic_sign"],
    "other_obstacle": ["parking_meter", "stop", "table", "chair", "kiosk", "traffic_light_controller", "power_controller"],
}
GROUP_OF = {name: group for group, members in GUIDANCE_GROUPS.items() for name in members}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down detection errors under the app's conditions.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Checkpoint to analyze (.pt).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml.")
    parser.add_argument("--split", default="test", help="Split in data.yaml to analyze.")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Lowest confidence kept in the cache.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder for the cache, report and crops.")
    parser.add_argument("--reuse", action="store_true", help="Analyze the existing cache instead of running the model.")
    parser.add_argument("--device", default=None, help="'0', 'mps', or 'cpu'. Auto-detected by default.")
    return parser.parse_args()


# MARK: - Cache


def build_cache(args: argparse.Namespace) -> dict:
    """Detections down to --min-conf, in view coordinates, for every image."""

    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(config["path"]) if config.get("path") else args.data.parent
    image_dir = root / config[args.split]
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    model = YOLO(str(args.weights))
    device = args.device or default_device()

    entries = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        rect = view_rect(width, height, full_frame=False)
        left, top, view_width, view_height = rect
        view = image[top : top + view_height, left : left + view_width]
        view_area = view_width * view_height

        result = model.predict(
            view, imgsz=640, conf=args.min_conf, iou=APP_NMS_IOU, max_det=APP_MAX_DETECTIONS, device=device, verbose=False
        )[0]
        detections = [
            [int(cls), round(float(score), 4), *[round(v, 1) for v in xyxy]]
            for cls, score, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist())
            if area(tuple(xyxy)) / view_area >= APP_MIN_AREA
        ]
        entries.append({"image": image_path.name, "size": [width, height], "detections": detections})
        if index % 1000 == 0:
            print(f"{index}/{len(images)} images")

    return {
        "weights": str(args.weights),
        "image_dir": str(image_dir),
        "names": {int(k): v for k, v in config["names"].items()},
        "min_conf": args.min_conf,
        "images": entries,
    }


def load_cache(path: Path) -> dict:
    cache = json.loads(path.read_text(encoding="utf-8"))
    cache["names"] = {int(k): v for k, v in cache["names"].items()}
    return cache


def attach_labels(cache: dict) -> None:
    """Adds each image's labels and cut-off objects, by the same rules as evaluate_app."""

    image_dir = Path(cache["image_dir"])
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    for entry in cache["images"]:
        width, height = entry["size"]
        rect = view_rect(width, height, full_frame=False)
        view_area = rect[2] * rect[3]
        in_view, cut_off = labels_in_view(label_dir / f"{Path(entry['image']).stem}.txt", width, height, rect)
        entry["labels"] = [[cls, *box] for cls, box in in_view if area(box) / view_area >= APP_MIN_AREA]
        entry["cut_off"] = [[cls, *box] for cls, box in cut_off]


def is_cut_off(detection: list, entry: dict) -> bool:
    """A detection of an object the view cuts off is neither a hit nor a false alarm."""

    cls, _, *box = detection
    return any(c == cls and iou(tuple(box), tuple(cut)) >= MATCH_IOU for c, *cut in entry["cut_off"])


# MARK: - Matching


def match(labels: list, detections: list, key) -> tuple[list[bool], list[float | None]]:
    """Greedy matching per `key(class)`, most confident detection first, like evaluate_app.

    Returns whether each detection matched and, for each label, the confidence of the
    detection that matched it. Because higher-confidence detections are matched first,
    the result restricted to any threshold equals matching at that threshold.
    """

    matched = [False] * len(detections)
    label_conf: list[float | None] = [None] * len(labels)
    order = sorted(range(len(detections)), key=lambda i: -detections[i][1])
    for i in order:
        cls, conf, *box = detections[i]
        best, best_iou = None, MATCH_IOU
        for j, (label_cls, *label_box) in enumerate(labels):
            if label_conf[j] is not None or key(label_cls) != key(cls):
                continue
            overlap = iou(tuple(box), tuple(label_box))
            if overlap >= best_iou:
                best, best_iou = j, overlap
        if best is not None:
            matched[i] = True
            label_conf[best] = conf
    return matched, label_conf


def threshold_for(cls: int, thresholds: dict[int, float] | None) -> float:
    return (thresholds or {}).get(cls, APP_CONFIDENCE)


def score(cache: dict, thresholds: dict[int, float] | None = None, key=lambda cls: cls, images=None) -> dict:
    """True positives, false alarms and labels per `key`, at per-class thresholds."""

    counts = defaultdict(lambda: {"labels": 0, "tp": 0, "fp": 0})
    for entry in images if images is not None else cache["images"]:
        detections = [d for d in entry["detections"] if d[1] >= threshold_for(d[0], thresholds)]
        matched, label_conf = match(entry["labels"], detections, key)
        for (cls, *_), conf in zip(entry["labels"], label_conf):
            counts[key(cls)]["labels"] += 1
            if conf is not None:
                counts[key(cls)]["tp"] += 1
        for detection, is_match in zip(detections, matched):
            if not is_match and not is_cut_off(detection, entry):
                counts[key(detection[0])]["fp"] += 1
    return counts


def rates(count: dict) -> tuple[float | None, float | None]:
    recall = count["tp"] / count["labels"] if count["labels"] else None
    found = count["tp"] + count["fp"]
    precision = count["tp"] / found if found else None
    return recall, precision


# MARK: - Error breakdown


def breakdown(cache: dict, thresholds: dict[int, float] | None = None) -> tuple[dict, list, list]:
    """Why each label was missed and why each false alarm happened, at the app threshold."""

    names = cache["names"]
    reasons = defaultdict(Counter)
    false_alarms = []  # (conf, class name, reason, image, box)
    misses = []  # (class name, image, box)
    for entry in cache["images"]:
        labels = entry["labels"]
        detections = [d for d in entry["detections"] if d[1] >= threshold_for(d[0], thresholds)]
        matched, label_conf = match(labels, detections, key=lambda cls: cls)

        for (cls, *box), conf in zip(labels, label_conf):
            if conf is not None:
                continue
            others = [d for d in detections if d[0] != cls and iou(tuple(d[2:]), tuple(box)) >= MATCH_IOU]
            if others:
                other = max(others, key=lambda d: d[1])
                reasons[names[cls]][f"miss: seen as {names[other[0]]}"] += 1
            else:
                reasons[names[cls]]["miss: nothing detected"] += 1
                misses.append((names[cls], entry["image"], box))

        for detection, is_match in zip(detections, matched):
            if is_match:
                continue
            cls, conf, *box = detection
            if is_cut_off(detection, entry):
                reasons[names[cls]]["ignored: cut off by the view"] += 1
                continue
            overlaps = [(label[0], iou(tuple(box), tuple(label[1:]))) for label in labels]
            if any(label_cls == cls and overlap >= MATCH_IOU for label_cls, overlap in overlaps):
                reason = "false alarm: duplicate box"
            elif any(label_cls != cls and overlap >= MATCH_IOU for label_cls, overlap in overlaps):
                other = max((o for o in overlaps if o[0] != cls), key=lambda o: o[1])[0]
                reason = f"false alarm: on a {names[other]}"
            else:
                reason = "false alarm: background"
                false_alarms.append((conf, names[cls], entry["image"], box))
            reasons[names[cls]][reason] += 1
    return reasons, false_alarms, misses


def contact_sheet(items: list, image_dir: Path, path: Path, columns: int = 8, tile: int = 180) -> None:
    """Crops (with some context) in a grid, each captioned with its class and confidence."""

    tiles = []
    for caption, image_name, box in items:
        image = cv2.imread(str(image_dir / image_name))
        height, width = image.shape[:2]
        left, top, view_width, view_height = view_rect(width, height, full_frame=False)
        view = image[top : top + view_height, left : left + view_width]
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * 0.3, (y2 - y1) * 0.3
        cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        cx2, cy2 = int(min(view_width, x2 + pad_x)), int(min(view_height, y2 + pad_y))
        crop = view[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 255), 2)
        scale = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        canvas = np.full((tile + 22, tile, 3), 255, np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        cv2.putText(canvas, caption, (3, tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        tiles.append(canvas)
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "cache.json"
    if args.reuse and cache_path.exists():
        cache = load_cache(cache_path)
    else:
        cache = build_cache(args)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    attach_labels(cache)
    names = cache["names"]
    print(f"{len(cache['images'])} images, weights {cache['weights']}")

    by_class = score(cache)
    by_group = score(cache, key=lambda cls: GROUP_OF.get(names[cls], names[cls]))
    reasons, false_alarms, misses = breakdown(cache)

    print(f"\n{'class':<26}{'labels':>7}{'recall':>8}{'precision':>10}   top reasons")
    for cls, name in names.items():
        recall, precision = rates(by_class[cls])
        top = ", ".join(f"{reason} {count}" for reason, count in reasons[name].most_common(3))
        fmt = lambda value: f"{value:.3f}" if value is not None else "-"
        print(f"{name:<26}{by_class[cls]['labels']:>7}{fmt(recall):>8}{fmt(precision):>10}   {top}")

    print(f"\n{'guidance group':<26}{'labels':>7}{'recall':>8}{'precision':>10}")
    for group in GUIDANCE_GROUPS:
        recall, precision = rates(by_group[group])
        print(f"{group:<26}{by_group[group]['labels']:>7}{recall:>8.3f}{precision:>10.3f}")

    totals = Counter()
    for name_reasons in reasons.values():
        for reason, count in name_reasons.items():
            if reason.startswith("ignored"):
                continue
            if reason.startswith("miss: seen as"):
                reason = "miss: seen as another class"
            elif reason.startswith("false alarm: on a"):
                reason = "false alarm: on another class"
            totals[reason] += count
    print("\nall errors:", dict(totals.most_common()))

    image_dir = Path(cache["image_dir"])
    confident = sorted(false_alarms, key=lambda item: -item[0])[:64]
    contact_sheet([(f"{name} {conf:.2f}", image, box) for conf, name, image, box in confident], image_dir, args.out / "false_alarms.jpg")
    missed = sorted(misses, key=lambda item: -area(tuple(item[2])))[:64]
    contact_sheet([(name, image, box) for name, image, box in missed], image_dir, args.out / "misses.jpg")

    summary = {
        "weights": cache["weights"],
        "images": len(cache["images"]),
        "classes": {
            names[cls]: {**by_class[cls], "reasons": dict(reasons[names[cls]])} for cls in names
        },
        "groups": {group: dict(by_group[group]) for group in GUIDANCE_GROUPS},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport, crops and cache written to {args.out}")


if __name__ == "__main__":
    main()
