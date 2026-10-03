from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from vision.tracker_logic import ApproachTracker, Detection

_DEFAULT_MODEL_PATH = (
    _REPO_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "sidewalk"
    / "yolov8n_sidewalk-3"
    / "weights"
    / "best.pt"
)

_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_TRACKER_LOCK = threading.Lock()
_MODEL: Any | None = None
_APPROACH_TRACKER = ApproachTracker(
    min_growth_ratio=1.28,
    center_band_only=False,
)

KOREAN_LABELS = {
    "person": "사람",
    "car": "차량",
    "truck": "트럭",
    "bus": "버스",
    "bicycle": "자전거",
    "motorcycle": "오토바이",
    "scooter": "전동 킥보드",
    "wheelchair": "휠체어",
    "stroller": "유모차",
    "traffic_light": "신호등",
    "traffic_sign": "교통 표지판",
    "pole": "기둥",
    "bollard": "볼라드 / 차단봉",
    "bench": "벤치",
    "tree_trunk": "나무",
    "movable_signage": "입간판",
    "potted_plant": "화분",
    "parking_meter": "주차 정산기",
    "stop": "버스 정류장",
    "table": "테이블",
    "barricade": "바리케이드",
    "chair": "의자",
    "fire_hydrant": "소화전",
    "kiosk": "가판대",
    "carrier": "카트",
    "dog": "강아지",
    "traffic_light_controller": "신호 제어함",
    "power_controller": "전기 분전함",
}


class VisionModelUnavailableError(RuntimeError):
    """Raised when YOLO inference cannot be used by the backend."""


def _load_yolo_model() -> Any:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        model_path = Path(os.getenv("YOLO_MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
        if not model_path.exists():
            raise VisionModelUnavailableError(f"YOLO checkpoint not found: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionModelUnavailableError(
                "Ultralytics is not installed. Install project requirements to enable vision inference."
            ) from exc

        model = YOLO(str(model_path))
        try:
            model.fuse()
        except Exception:
            pass

        _MODEL = model
        return _MODEL


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisionModelUnavailableError(
            "opencv-python-headless is not installed. Install project requirements to decode images."
        ) from exc

    cv2.setUseOptimized(True)
    cv2.setNumThreads(int(os.getenv("YOLO_CV2_THREADS", "2")))
    return cv2


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise VisionModelUnavailableError(
            "NumPy is not installed. Install project requirements to enable vision inference."
        ) from exc

    return np


def _image_to_bgr(image: bytes | bytearray | memoryview | Any) -> Any:
    cv2 = _cv2()
    np = _np()

    if isinstance(image, (bytes, bytearray, memoryview)):
        if not image:
            raise ValueError("Image payload is empty.")

        encoded = np.frombuffer(image, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image bytes.")
        height, width = image_bgr.shape[:2]
        print(f"[BBoxDebug] backend decoded upload frame={width}x{height}", file=sys.stderr)
        return image_bgr

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Image array is empty.")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image

        raise ValueError("Unsupported OpenCV image shape.")

    if hasattr(image, "convert"):
        image_rgb = image.convert("RGB")
        return cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)

    raise TypeError("Image must be upload bytes, a PIL image, or an OpenCV image array.")


def _position_from_bbox(bbox_xyxy: list[float], frame_width: int) -> str:
    x1, _y1, x2, _y2 = bbox_xyxy
    center_x = (x1 + x2) / 2.0

    if center_x < frame_width / 3:
        return "left"
    if center_x > frame_width * 2 / 3:
        return "right"
    return "center"


def _vertical_ratio_from_bbox(bbox_xyxy: list[float], frame_height: int) -> float:
    """Return how low the object sits in the frame; lower boxes are usually closer."""

    _x1, _y1, _x2, y2 = bbox_xyxy
    return round(max(0.0, min(1.0, y2 / max(1, frame_height))), 4)


def _area_ratio_from_bbox(bbox_xyxy: list[float], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    return round(min(1.0, bbox_area / frame_area), 6)


def _parse_yolo_result(result: Any, frame_width: int, frame_height: int) -> list[dict]:
    detections: list[dict] = []
    names = result.names
    print(
        f"[BBoxDebug] YOLO coordinate space frame={frame_width}x{frame_height} "
        f"orig_shape={getattr(result, 'orig_shape', None)}",
        file=sys.stderr,
    )

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox_xyxy = [float(value) for value in box.xyxy[0].tolist()]
        print(
            f"[BBoxDebug] YOLO raw label={names[class_id]} conf={confidence:.4f} "
            f"bbox_xyxy={[round(value, 2) for value in bbox_xyxy]}",
            file=sys.stderr,
        )

        if _area_ratio_from_bbox(bbox_xyxy, frame_width, frame_height) < 0.01:
            continue
        detections.append(
            {
                "label": str(names[class_id]),
                "korean_label": KOREAN_LABELS.get(str(names[class_id]), str(names[class_id])),
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(value, 2) for value in bbox_xyxy],
                "position": _position_from_bbox(bbox_xyxy, frame_width),
                "area_ratio": _area_ratio_from_bbox(bbox_xyxy, frame_width, frame_height),
                "vertical_ratio": _vertical_ratio_from_bbox(bbox_xyxy, frame_height),
                "approaching": False,
            }
        )

    return detections


def _mark_approaching_objects(detections: list[dict], frame_width: int) -> list[dict]:
    tracker_inputs = [
        Detection(
            label=str(detection["label"]),
            confidence=float(detection["confidence"]),
            bbox_xyxy=tuple(float(value) for value in detection["bbox_xyxy"]),
        )
        for detection in detections
    ]

    with _TRACKER_LOCK:
        alerts = _APPROACH_TRACKER.update(tracker_inputs, frame_width=frame_width)

    for alert in alerts:
        for detection in detections:
            if detection.get("label") != alert.label:
                continue
            if tuple(float(value) for value in detection.get("bbox_xyxy", ())) == alert.bbox_xyxy:
                detection["approaching"] = True
                detection["growth_ratio"] = alert.growth_ratio
                break

    return detections


def detect_objects(image: bytes | bytearray | memoryview | Any) -> list[dict]:
    """Run lazy YOLO inference and return JSON-friendly detection dictionaries."""

    image_bgr = _image_to_bgr(image)
    frame_height, frame_width = image_bgr.shape[:2]
    model = _load_yolo_model()

    with _INFERENCE_LOCK:
        results = model.predict(
            source=image_bgr,
            imgsz=416,
            conf=0.35,
            iou=0.5,
            max_det=20,
            device="cpu",
            half=False,
            verbose=False,
        )

    if not results:
        return []

    detections = _parse_yolo_result(results[0], int(frame_width), int(frame_height))
    return _mark_approaching_objects(detections, int(frame_width))


def warm_vision_model() -> None:
    """Optionally preload the YOLO model without running inference."""

    _load_yolo_model()
