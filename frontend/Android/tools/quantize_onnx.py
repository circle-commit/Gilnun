"""Quantize the Android detector to 8-bit integers so it runs faster on phone CPUs.

Run from the repository root after `export_onnx.py` (requires ultralytics, onnx and onnxruntime):
    python frontend/Android/tools/quantize_onnx.py

The convolutions of the backbone and neck get int8 weights (per channel) and uint8
activations in ONNX QDQ form, which ONNX Runtime runs with integer kernels on ARM.
The detection head's last convolutions and its box decoding stay float, because
quantizing them shifts boxes and confidences the most.

Activation ranges are calibrated on sidewalk photos that the detector tests do not
use (`test_images/` stays out), cut into many portrait crops like the app's frames.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image, ImageEnhance
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/sidewalk/yolo11m_sidewalk/weights/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "frontend/Android/app/src/main/assets/sidewalk_detector.onnx"
CALIBRATION_IMAGES = sorted((REPO_ROOT / "datasets/yolo_sidewalk/images/train").glob("*.jpg"))
IMGSZ = (640, 384)
# The first convolution and the attention block (C2PSA) lose the most accuracy in int8.
DEFAULT_KEEP_FLOAT = ["/model.0/", "/model.10/"]
# Quantizing the activations between convolutions too keeps tensors in int8 from layer to
# layer; with only the convolutions quantized, converting back and forth ate the speedup.
OP_TYPES = ["Conv", "Mul", "Sigmoid", "Add", "Concat", "MaxPool", "Resize", "Split"]


def letterbox(image: Image.Image, height: int, width: int) -> np.ndarray:
    """The app's preprocessing (`ObjectDetector.kt`): fit, center, gray padding, planar RGB 0-1."""

    gain = min(width / image.width, height / image.height)
    scaled_width, scaled_height = round(image.width * gain), round(image.height * gain)
    left = round((width - scaled_width) / 2 - 0.1)
    top = round((height - scaled_height) / 2 - 0.1)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(image.resize((scaled_width, scaled_height), Image.BILINEAR), (left, top))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def portrait_crops(image: Image.Image, rng: np.random.Generator, count: int) -> list[Image.Image]:
    """9:16 crops at random positions and sizes, some mirrored, darkened or brightened."""

    crops = []
    for _ in range(count):
        crop_height = int(image.height * rng.uniform(0.6, 1.0))
        crop_width = min(image.width, round(crop_height * 9 / 16))
        x = int(rng.integers(0, image.width - crop_width + 1))
        y = int(rng.integers(0, image.height - crop_height + 1))
        crop = image.crop((x, y, x + crop_width, y + crop_height))
        if rng.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        crop = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(0.5, 1.3)))
        crops.append(crop)
    return crops


class SidewalkCalibration(CalibrationDataReader):
    def __init__(self, input_name: str, crops_per_image: int) -> None:
        rng = np.random.default_rng(20261103)
        batches = []
        for path in CALIBRATION_IMAGES:
            image = Image.open(path).convert("RGB")
            batches += [letterbox(crop, *IMGSZ) for crop in portrait_crops(image, rng, crops_per_image)]
        self._inputs = iter([{input_name: batch} for batch in batches])
        print(f"Calibrating on {len(batches)} crops of {len(CALIBRATION_IMAGES)} photos")

    def get_next(self):
        return next(self._inputs, None)


def float_nodes(model: onnx.ModelProto, keep_float: list[str]) -> list[str]:
    """The detection head's final 1x1 convolutions, everything after them, and the layers
    named in `keep_float` (prefixes such as "/model.0/") stay float."""

    keep = []
    for node in model.graph.node:
        name = node.name
        if any(name.startswith(prefix) for prefix in keep_float):
            keep.append(name)
            continue
        final_conv = name.startswith("/model.23/cv2.") or name.startswith("/model.23/cv3.")
        final_conv = final_conv and name.endswith(".2/Conv")
        decoding = name.startswith("/model.23/") and not name.startswith(("/model.23/cv2.", "/model.23/cv3."))
        if final_conv or decoding:
            keep.append(name)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize the Android detector to int8.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .onnx path.")
    parser.add_argument("--crops-per-image", type=int, default=40, help="Calibration crops cut from each photo.")
    parser.add_argument(
        "--calibration",
        choices=["minmax", "percentile", "entropy"],
        # Percentile and entropy need far more memory (over 15 GB here); try them on a bigger machine.
        default="minmax",
        help="How activation ranges are chosen from the calibration crops.",
    )
    parser.add_argument(
        "--keep-float",
        nargs="*",
        default=DEFAULT_KEEP_FLOAT,
        help="Node name prefixes of layers to leave in float.",
    )
    args = parser.parse_args()
    if not CALIBRATION_IMAGES:
        raise FileNotFoundError("no calibration photos in datasets/yolo_sidewalk/images/train")

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        exported = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=list(IMGSZ), opset=17, simplify=True, nms=False))
        float_model = work_dir / "float.onnx"
        shutil.move(str(exported), float_model)
        prepared = work_dir / "prepared.onnx"
        quant_pre_process(str(float_model), str(prepared))

        model = onnx.load(str(prepared))
        input_name = model.graph.input[0].name
        excluded = float_nodes(model, args.keep_float)
        method = {
            "minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy,
        }[args.calibration]

        quantized = work_dir / "int8.onnx"
        quantize_static(
            str(prepared),
            str(quantized),
            SidewalkCalibration(input_name, args.crops_per_image),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=OP_TYPES,
            nodes_to_exclude=excluded,
            calibrate_method=method,
            # Histogram methods keep every intermediate output; flushing them in small groups
            # keeps calibration within a laptop's memory.
            extra_options={"CalibPercentile": 99.999, "CalibMaxIntermediateOutputs": 4}
            if method != CalibrationMethod.MinMax
            else {},
        )

        # Keep the class names the app reads from the metadata.
        result = onnx.load(str(quantized))
        names = {prop.key: prop.value for prop in onnx.load(str(float_model)).metadata_props}
        del result.metadata_props[:]
        for key, value in names.items():
            result.metadata_props.add(key=key, value=value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(result, str(args.output))

    print(f"int8 model written to: {args.output} ({len(excluded)} nodes kept float, {args.calibration} calibration)")


if __name__ == "__main__":
    main()
