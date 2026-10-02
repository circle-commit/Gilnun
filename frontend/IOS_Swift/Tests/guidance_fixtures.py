"""Generate fixtures that pin the iOS live-guidance port to the Python backend.

The Swift test replays every case and must produce exactly the same results.
Time and template choice are made deterministic on both sides.

Usage (standard library only):
    python3 frontend/IOS_Swift/Tests/guidance_fixtures.py OUTPUT.json
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from services import guidance_message_service as gms  # noqa: E402
from services import safety_service, vision_service  # noqa: E402
from vision import tracker_logic  # noqa: E402


FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
LABELS = list(vision_service.KOREAN_LABELS)
EVENT_TYPES = ["new_object", "closer", "risk_increased", "entered_front_zone", "approaching"]
BOUNDARY_AREAS = [0.01, 0.016, 0.018, 0.025, 0.035, 0.04, 0.055, 0.08, 0.09, 0.11, 0.16]
BOUNDARY_VERTICALS = [0.42, 0.48, 0.52, 0.54, 0.58, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.86, 0.88, 0.9]
ZERO_COOLDOWNS = {"global": 0.0, "object": 0.0, "situation": 0.0, "signature": 0.0, "stale_after": 12.0}
DEFAULT_COOLDOWNS = {"global": 1.5, "object": 6.0, "situation": 8.0, "signature": 10.0, "stale_after": 12.0}


class FakeClock:
    now = 0.0

    def monotonic(self) -> float:
        return self.now


CLOCK = FakeClock()
gms.time = types.SimpleNamespace(monotonic=CLOCK.monotonic)
tracker_logic.monotonic = CLOCK.monotonic

CAPTURED_TEMPLATES: list[list[list]] = []


def _first_template(weighted_templates):
    """Record the candidates and pick the first one, like Swift's `random` returning 0."""

    CAPTURED_TEMPLATES.append([[template, weight] for template, weight in weighted_templates])
    return weighted_templates[0][0]


gms._choose_weighted_template = _first_template


def make_event_tracker(cooldowns: dict) -> gms.GuidanceEventTracker:
    return gms.GuidanceEventTracker(
        global_cooldown_seconds=cooldowns["global"],
        object_cooldown_seconds=cooldowns["object"],
        situation_cooldown_seconds=cooldowns["situation"],
        signature_cooldown_seconds=cooldowns["signature"],
        stale_after_seconds=cooldowns["stale_after"],
    )


def position_for(bbox: list[float]) -> str:
    return vision_service._position_from_bbox(bbox, FRAME_WIDTH)


def random_bbox(rng: random.Random) -> list[float]:
    width = rng.uniform(20, 700)
    height = rng.uniform(20, 1400)
    x1 = rng.uniform(0, FRAME_WIDTH - width)
    y1 = rng.uniform(0, FRAME_HEIGHT - height)
    return [round(x1, 2), round(y1, 2), round(x1 + width, 2), round(y1 + height, 2)]


def random_detection(rng: random.Random) -> dict:
    """Detection with independent random fields, including exact threshold values."""

    label = rng.choice(LABELS + ["unknown_object"])
    area = rng.uniform(0, 0.25) if rng.random() < 0.7 else rng.choice(BOUNDARY_AREAS)
    vertical = rng.uniform(0, 1) if rng.random() < 0.7 else rng.choice(BOUNDARY_VERTICALS)
    return {
        "label": label,
        "korean_label": vision_service.KOREAN_LABELS.get(label, label),
        "confidence": round(rng.uniform(0.2, 1.0), 4),
        "bbox_xyxy": random_bbox(rng),
        "position": rng.choice(["left", "center", "right"]),
        "area_ratio": round(area, 6),
        "vertical_ratio": round(vertical, 4),
        "approaching": rng.random() < 0.25,
    }


def enriched(detections: list[dict]) -> list[dict]:
    return gms.enrich_and_prioritize_detections(detections, {})


def rules_cases(rng: random.Random) -> list[dict]:
    cases = []
    for _ in range(800):
        detection = random_detection(rng)
        result = enriched([detection])[0]
        cases.append(
            {
                "detection": detection,
                "front_danger_zone": result["front_danger_zone"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "distance_level": gms._distance_level(
                    detection["label"],
                    max(0.0, min(1.0, detection["area_ratio"])),
                    max(0.0, min(1.0, detection["vertical_ratio"])),
                ),
                "object_group": gms._object_group(detection["label"]),
                "particle": gms._particle_for(detection["korean_label"]),
            }
        )
    return cases


def priority_cases(rng: random.Random) -> list[dict]:
    cases = []
    for _ in range(150):
        detections = [random_detection(rng) for _ in range(rng.randint(1, 8))]
        # Duplicate a few rows so ties exercise the stable sort.
        if len(detections) > 1 and rng.random() < 0.3:
            detections.append(dict(detections[0]))
        for index, detection in enumerate(detections):
            detection["index"] = index
        order = [detection["index"] for detection in enriched(detections)]
        for detection in detections:
            del detection["index"]
        cases.append({"detections": detections, "order": order})
    return cases


def template_cases(rng: random.Random) -> list[dict]:
    cases = []
    for _ in range(800):
        detection = random_detection(rng)
        event_type = rng.choice(EVENT_TYPES)
        CAPTURED_TEMPLATES.clear()
        message = gms.build_detection_message(enriched([detection])[0], event_type)
        cases.append(
            {
                "detection": detection,
                "event_type": event_type,
                "templates": CAPTURED_TEMPLATES[0],
                "message": message,
            }
        )
    return cases


class MovingObject:
    """An object that persists across frames with slowly changing geometry."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.label = rng.choice(LABELS)
        self.width = rng.uniform(60, 500)
        self.height = rng.uniform(120, 1100)
        self.center_x = rng.uniform(self.width / 2, FRAME_WIDTH - self.width / 2)
        self.bottom = rng.uniform(self.height, FRAME_HEIGHT)
        self.confidence = rng.uniform(0.3, 0.95)
        self.approaching = False

    def step(self) -> None:
        rng = self.rng
        scale = rng.choice([1.0, 1.0, 1.05, 1.12, 1.25, 0.9])
        self.width = min(FRAME_WIDTH * 0.9, self.width * scale)
        self.height = min(FRAME_HEIGHT * 0.9, self.height * scale)
        self.center_x = min(FRAME_WIDTH - self.width / 2, max(self.width / 2, self.center_x + rng.uniform(-90, 90)))
        self.bottom = min(FRAME_HEIGHT, max(self.height, self.bottom + rng.uniform(-60, 120)))
        self.confidence = min(0.99, max(0.2, self.confidence + rng.uniform(-0.15, 0.15)))
        if rng.random() < 0.2:
            self.approaching = not self.approaching

    def bbox(self) -> list[float]:
        return [
            round(self.center_x - self.width / 2, 2),
            round(self.bottom - self.height, 2),
            round(self.center_x + self.width / 2, 2),
            round(self.bottom, 2),
        ]

    def detection(self) -> dict:
        bbox = self.bbox()
        return {
            "label": self.label,
            "korean_label": vision_service.KOREAN_LABELS[self.label],
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": bbox,
            "position": position_for(bbox),
            "area_ratio": vision_service._area_ratio_from_bbox(bbox, FRAME_WIDTH, FRAME_HEIGHT),
            "vertical_ratio": vision_service._vertical_ratio_from_bbox(bbox, FRAME_HEIGHT),
            "approaching": self.approaching,
        }


def next_time(rng: random.Random, now: float) -> float:
    return round(now + rng.choice([0.2, 0.2, 0.2, 0.2, 0.4, 1.0, 1.6, 3.0, 7.0, 13.0]), 3)


def event_sequences(rng: random.Random) -> list[dict]:
    sequences = []
    for sequence_index in range(60):
        cooldowns = ZERO_COOLDOWNS if sequence_index % 2 else DEFAULT_COOLDOWNS
        tracker = make_event_tracker(cooldowns)
        objects = [MovingObject(rng) for _ in range(rng.randint(1, 5))]
        now = 100.0
        frames = []
        for _ in range(40):
            now = next_time(rng, now)
            for obj in objects:
                obj.step()
            if rng.random() < 0.1:
                objects.append(MovingObject(rng))
            if len(objects) > 1 and rng.random() < 0.08:
                objects.pop(rng.randrange(len(objects)))

            detections = [obj.detection() for obj in objects if rng.random() > 0.15]
            CLOCK.now = now
            events = tracker.choose_events(enriched(detections), limit=2)
            frames.append(
                {
                    "now": now,
                    "detections": detections,
                    "events": [
                        {
                            "event": event["event"],
                            "label": event["detection"]["label"],
                            "position": event["detection"]["position"],
                            "message": event["message"],
                        }
                        for event in events
                    ],
                }
            )
        sequences.append({"cooldowns": cooldowns, "frames": frames})
    return sequences


def approach_sequences(rng: random.Random) -> list[dict]:
    sequences = []
    for _ in range(60):
        tracker = tracker_logic.ApproachTracker(min_growth_ratio=1.28, center_band_only=False)
        objects = [MovingObject(rng) for _ in range(rng.randint(1, 4))]
        now = 50.0
        frames = []
        for _ in range(30):
            now = round(now + rng.choice([0.2, 0.2, 0.2, 0.3, 0.6, 1.2]), 3)
            for obj in objects:
                obj.step()
            detections = [{"label": obj.label, "bbox": obj.bbox()} for obj in objects if rng.random() > 0.1]
            rng.shuffle(detections)

            # Feeding one detection per call is equivalent to one call per frame (expiry
            # uses the same timestamp) and tells us which detection raised each alert.
            alerts = []
            for index, detection in enumerate(detections):
                for alert in tracker.update(
                    [tracker_logic.Detection(detection["label"], 0.9, tuple(detection["bbox"]))],
                    frame_width=FRAME_WIDTH,
                    now=now,
                ):
                    alerts.append({"index": index, "track_id": alert.track_id, "growth_ratio": alert.growth_ratio})
            frames.append({"now": now, "detections": detections, "alerts": alerts})
        sequences.append({"frames": frames})
    return sequences


class _Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return list(self.value)


class _Box:
    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        self.cls = [_Value(class_id)]
        self.conf = [_Value(confidence)]
        self.xyxy = [_Value(bbox)]


class _YoloResult:
    """Just enough of an Ultralytics result for `vision_service._parse_yolo_result`."""

    names = dict(enumerate(LABELS))

    def __init__(self, raw_detections: list[dict]) -> None:
        self.boxes = [
            _Box(LABELS.index(raw["label"]), raw["confidence"], raw["bbox"]) for raw in raw_detections
        ]
        self.orig_shape = (FRAME_HEIGHT, FRAME_WIDTH)


def scene_sequences(rng: random.Random) -> list[dict]:
    sequences = []
    for _ in range(60):
        vision_service._APPROACH_TRACKER = tracker_logic.ApproachTracker(min_growth_ratio=1.28, center_band_only=False)
        gms.GUIDANCE_TRACKER = gms.GuidanceEventTracker()
        objects = [MovingObject(rng) for _ in range(rng.randint(1, 5))]
        now = 1000.0
        frames = []
        for _ in range(40):
            now = round(now + rng.choice([0.2, 0.2, 0.2, 0.2, 0.4, 1.0, 2.0, 13.0]), 3)
            for obj in objects:
                obj.step()
            if rng.random() < 0.1:
                objects.append(MovingObject(rng))

            raw = []
            for obj in objects:
                if rng.random() < 0.15:
                    continue
                jitter = [rng.uniform(-0.004, 0.004) for _ in range(4)]
                bbox = [value + offset for value, offset in zip(obj.bbox(), jitter)]
                raw.append({"label": obj.label, "confidence": obj.confidence + rng.uniform(-1e-5, 1e-5), "bbox": bbox})
            # Tiny boxes are dropped by the 1% area filter.
            if rng.random() < 0.3:
                x, y = rng.uniform(0, FRAME_WIDTH - 60), rng.uniform(0, FRAME_HEIGHT - 60)
                raw.append({"label": rng.choice(LABELS), "confidence": rng.uniform(0.35, 0.9), "bbox": [x, y, x + 50, y + 50]})
            raw.sort(key=lambda item: -item["confidence"])

            CLOCK.now = now
            with contextlib.redirect_stderr(io.StringIO()):
                detections = vision_service._parse_yolo_result(_YoloResult(raw), FRAME_WIDTH, FRAME_HEIGHT)
            detections = vision_service._mark_approaching_objects(detections, FRAME_WIDTH)
            prioritized = safety_service.prioritize_detections(detections)
            guidance = gms.build_guidance(prioritized)
            frames.append(
                {
                    "now": now,
                    "raw": raw,
                    "voice_guide": guidance["voice_guide"],
                    "detections": [
                        {
                            "label": detection["label"],
                            "position": detection["position"],
                            "approaching": detection["approaching"],
                            "risk_score": detection["risk_score"],
                            "risk_level": detection["risk_level"],
                        }
                        for detection in prioritized
                    ],
                }
            )
        sequences.append({"frames": frames})
    return sequences


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: guidance_fixtures.py OUTPUT.json")

    rng = random.Random(20261002)
    fixtures = {
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "rules": rules_cases(rng),
        "priority": priority_cases(rng),
        "templates": template_cases(rng),
        "event_sequences": event_sequences(rng),
        "approach_sequences": approach_sequences(rng),
        "scene_sequences": scene_sequences(rng),
    }
    Path(sys.argv[1]).write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
