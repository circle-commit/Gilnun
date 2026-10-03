"""Pack what training reads into one tar for uploading to a GPU server.

That is data_balanced.yaml, its train and val lists, every training image and label,
and the validation images in val_subset.txt. The test split stays on this Mac, where
vision/evaluate_app.py scores the trained model. JPEG images do not compress further,
so the tar is not compressed.

Run from the repository root after scripts/build_balanced_train_list.py:
    python scripts/pack_training_data.py
On the server, unpack it inside the repository's datasets/ folder:
    tar -xf sidewalk_train.tar -C datasets/
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


DEFAULT_DATASET = Path("datasets/yolo_sidewalk")
DEFAULT_OUT = Path("datasets/sidewalk_train.tar")
LIST_FILES = ("data_balanced.yaml", "train_balanced.txt", "val_subset.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack the training data into one tar.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset root with data_balanced.yaml.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Tar file to write.")
    return parser.parse_args()


def listed_images(list_path: Path) -> list[str]:
    """Unique image paths (relative to the dataset root) in a train or val list."""

    names = {line.strip().removeprefix("./") for line in list_path.read_text(encoding="utf-8").splitlines()}
    return sorted(name for name in names if name)


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    missing = [name for name in LIST_FILES if not (dataset / name).exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in {dataset}; run scripts/build_balanced_train_list.py first.")

    images = listed_images(dataset / "train_balanced.txt") + listed_images(dataset / "val_subset.txt")
    files = list(LIST_FILES)
    for image in images:
        split, name = Path(image).parts[-2:]
        files += [image, f"labels/{split}/{Path(name).stem}.txt"]

    partial = args.out.with_name(args.out.name + ".partial")
    with tarfile.open(partial, "w") as tar:
        for index, name in enumerate(files, start=1):
            tar.add(dataset / name, arcname=f"{dataset.name}/{name}", recursive=False)
            if index % 50000 == 0:
                print(f"{index}/{len(files)} files")
    partial.replace(args.out)
    print(f"{len(images)} images, {args.out.stat().st_size / 1024**3:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
