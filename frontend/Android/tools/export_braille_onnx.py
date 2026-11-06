"""Export the braille block segmentation model to ONNX for the Android app.

Run from the repository root (requires ultralytics, onnx and onnxslim):
    python frontend/Android/tools/export_braille_onnx.py

Android counterpart of the iOS export (`YOLO(best.pt).export(format="coreml",
imgsz=[640, 384], half=True)`, see `BrailleBlockSegmenter.swift`): the same YOLO11n-seg
checkpoint and 640x384 input. The model outputs raw predictions and mask prototypes; the
app merges the masks itself (`BrailleBlock.kt`). Weights are stored as float16 like the
detector's (see export_onnx.py), which halves the file without changing the math.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import store_weights_as_float16  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/braille/yolo11n_braille/weights/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "frontend/Android/app/src/main/assets/braille_block.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the braille block model to ONNX for Android.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .onnx path.")
    args = parser.parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.weights}")

    exported = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=[640, 384], opset=17, simplify=True))
    model = onnx.load(str(exported))
    converted = store_weights_as_float16(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    exported.unlink()
    print(f"ONNX model written to: {args.output} ({converted} weights stored as float16)")


if __name__ == "__main__":
    main()
