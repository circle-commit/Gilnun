"""Train the sidewalk object detector with Ultralytics.

Run from the repository root, ideally on a CUDA GPU (see docs/retraining.md):
    python -m vision.train --data datasets/yolo_sidewalk/data_balanced.yaml
Continue an interrupted run from its last finished epoch, with the settings it started with:
    python -m vision.train --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from vision.device import default_device


DEFAULT_DATA = Path("datasets/yolo_sidewalk/data.yaml")
DEFAULT_PROJECT = Path("runs/sidewalk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sidewalk object detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Base checkpoint. YOLO11 keeps the app's Core ML export (NMS pipeline) unchanged.",
    )
    # One epoch of the balanced list is about 350,000 images, so fewer epochs are needed
    # than for a small dataset. 40 took about 8 hours on one A100.
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square training image size.")
    # Per-step overhead in the training loop dominates on an A100: batches of 32, 64 and
    # 128 trained 312, 397 and 499 images per second. 128 needs about 31 GB of GPU memory.
    parser.add_argument("--batch", type=int, default=128, help="Batch size. Lower this if the GPU runs out of memory.")
    parser.add_argument("--workers", type=int, default=16, help="Dataloader workers (capped at the CPU count).")
    parser.add_argument("--device", default=None, help="'0' for the first CUDA GPU, 'mps', or 'cpu'. Auto-detected by default.")
    parser.add_argument("--cache", choices=["ram", "disk"], default=None, help="Cache decoded images to speed up epochs.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output directory.")
    parser.add_argument("--name", default="yolo11s_sidewalk", help="Run name under project.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Continue an interrupted run from its last.pt (default: <project>/<name>/weights/last.pt).",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    if args.resume:
        last = Path(args.resume) if isinstance(args.resume, str) else args.project / args.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {last}")
        # The checkpoint keeps the run's data, epochs, batch and output folder.
        model = YOLO(str(last))
        model.train(resume=True)
        print(f"Training complete. Best checkpoint: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
        return

    data_path = args.data.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    device = args.device or default_device()
    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        # Ultralytics nests relative projects under runs/detect/; an absolute path keeps
        # results at runs/sidewalk/<name> in this repository.
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        # Deterministic CUDA kernels are slower; the fixed seed still makes runs comparable.
        deterministic=False,
        pretrained=True,
        cache=args.cache or False,
        plots=True,
        val=True,
        # Mixed precision roughly doubles speed on NVIDIA GPUs; it is not used on CPU or MPS.
        amp=device not in {"cpu", "mps"},
    )

    # The run directory gets a -2, -3, ... suffix when the name is already taken.
    print(f"Training complete. Best checkpoint: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    train()
