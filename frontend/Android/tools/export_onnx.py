"""Export the trained sidewalk YOLO model to ONNX for the Android app.

Run from the repository root (requires ultralytics, onnx and onnxslim):
    python frontend/Android/tools/export_onnx.py

Android counterpart of `vision/export_coreml.py`: same checkpoint, same 640x384
(height x width) input. NMS stays in the app (`ObjectDetector.kt`), so the model
outputs raw boxes. Weights are stored as float16 and cast back to float32 when the
model loads, which halves the file without changing the math the phone runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/sidewalk/yolo11m_sidewalk/weights/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "frontend/Android/app/src/main/assets/sidewalk_detector.onnx"

# Upright portrait frames at imgsz=640, as in PyTorch rect inference (see export_coreml.py).
DEFAULT_IMGSZ = (640, 384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the sidewalk YOLO model to ONNX for Android.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .onnx path.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=DEFAULT_IMGSZ,
        metavar=("HEIGHT", "WIDTH"),
        help="Model input size. Both values must be multiples of 32.",
    )
    return parser.parse_args()


def store_weights_as_float16(model: onnx.ModelProto) -> int:
    """Replaces float32 weights with float16 copies followed by a Cast back to float32.

    ONNX Runtime folds the casts when it loads the model, so inference stays float32.
    Returns the number of converted weights.
    """

    graph = model.graph
    casts = []
    converted = 0
    for index, initializer in enumerate(list(graph.initializer)):
        if initializer.data_type != TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(initializer)
        if values.size < 16:
            continue

        name = initializer.name
        half_name = f"{name}__float16"
        graph.initializer[index].CopyFrom(numpy_helper.from_array(values.astype(np.float16), half_name))
        casts.append(helper.make_node("Cast", [half_name], [name], to=TensorProto.FLOAT, name=f"{name}__to_float32"))
        converted += 1

    for cast in reversed(casts):
        graph.node.insert(0, cast)
    return converted


def export() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.weights}")

    model = YOLO(str(args.weights))
    exported = Path(model.export(format="onnx", imgsz=list(args.imgsz), opset=17, simplify=True, nms=False))

    onnx_model = onnx.load(str(exported))
    converted = store_weights_as_float16(onnx_model)
    onnx.checker.check_model(onnx_model)
    names = {prop.key: prop.value for prop in onnx_model.metadata_props}.get("names", "")
    if not names:
        raise RuntimeError("exported model has no class names in its metadata")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(args.output))
    exported.unlink()
    print(f"ONNX model written to: {args.output} ({converted} weights stored as float16)")
    print(f"Classes: {names}")


if __name__ == "__main__":
    export()
