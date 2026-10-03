from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from random import Random
from typing import Any


HIGHEST_RISK_LABELS = {"car", "truck", "bus"}
HIGH_RISK_LABELS = {"motorcycle", "scooter", "bicycle"}
MEDIUM_HIGH_RISK_LABELS = {"bollard", "pole", "movable_signage", "tree_trunk", "barricade", "fire_hydrant"}
MEDIUM_RISK_LABELS = {"person", "wheelchair", "stroller", "carrier", "dog"}
LOWER_RISK_LABELS = {"bench", "potted_plant", "traffic_sign", "traffic_light"}
IMMEDIATE_SPEECH_LABELS = HIGHEST_RISK_LABELS | {"motorcycle", "scooter"}
LOW_CONFIDENCE_SPEECH_LABELS = HIGHEST_RISK_LABELS | HIGH_RISK_LABELS | {"wheelchair", "stroller"}
NOISY_SPEECH_LABELS = {"pole", "tree_trunk", "traffic_sign", "traffic_light", "movable_signage"}
# Fixed objects without a risk set of their own. "stop" is a bus stop.
OTHER_OBSTACLE_LABELS = {
    "parking_meter",
    "stop",
    "table",
    "chair",
    "kiosk",
    "traffic_light_controller",
    "power_controller",
}
OBSTACLE_LABELS = MEDIUM_HIGH_RISK_LABELS | {"bench", "potted_plant"} | OTHER_OBSTACLE_LABELS
GUIDANCE_LABELS = (
    HIGHEST_RISK_LABELS
    | HIGH_RISK_LABELS
    | MEDIUM_HIGH_RISK_LABELS
    | MEDIUM_RISK_LABELS
    | LOWER_RISK_LABELS
    | OTHER_OBSTACLE_LABELS
)

POSITION_KO = {
    "left": "왼쪽",
    "center": "정면",
    "right": "오른쪽",
}

POSITION_FROM_KO = {
    "left": "왼쪽에서",
    "center": "정면에서",
    "right": "오른쪽에서",
}

RISK_LEVEL_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

OBJECT_GROUPS = {
    "vehicle": HIGHEST_RISK_LABELS,
    "mobility": HIGH_RISK_LABELS | {"wheelchair", "stroller", "carrier", "dog"},
    "person": {"person"},
    "obstacle": OBSTACLE_LABELS | MEDIUM_HIGH_RISK_LABELS,
    "traffic": {"traffic_sign", "traffic_light"},
}

EVENT_RANK = {
    "new_object": 0,
    "closer": 1,
    "risk_increased": 2,
    "entered_front_zone": 3,
    "approaching": 4,
}

_TEMPLATE_RANDOM = Random()


@dataclass(frozen=True)
class DistanceThresholds:
    very_close_area: float
    very_close_vertical: float
    close_area: float
    close_vertical: float
    near_area: float
    near_vertical: float


DEFAULT_DISTANCE_THRESHOLDS = DistanceThresholds(
    very_close_area=0.09,
    very_close_vertical=0.86,
    close_area=0.04,
    close_vertical=0.68,
    near_area=0.018,
    near_vertical=0.52,
)

DISTANCE_THRESHOLDS_BY_GROUP = {
    # Large objects occupy a lot of pixels even when they are not immediately near.
    "vehicle": DistanceThresholds(0.16, 0.90, 0.08, 0.76, 0.035, 0.58),
    "person": DistanceThresholds(0.11, 0.88, 0.055, 0.72, 0.025, 0.54),
    "mobility": DistanceThresholds(0.08, 0.84, 0.035, 0.64, 0.016, 0.48),
    # Small ground obstacles can be close while their boxes are still small.
    "ground_obstacle": DistanceThresholds(0.035, 0.80, 0.018, 0.60, 0.01, 0.42),
    "traffic": DistanceThresholds(0.08, 0.90, 0.035, 0.76, 0.016, 0.60),
}

GROUND_OBSTACLE_DISTANCE_LABELS = {
    "bollard",
    "pole",
    "tree_trunk",
    "movable_signage",
    "bench",
    "potted_plant",
    "parking_meter",
    "table",
    "fire_hydrant",
}


@dataclass
class SituationState:
    last_seen_at: float
    last_spoken_at: float
    risk_score: int
    risk_level: str
    position: str
    area_ratio: float
    approaching: bool
    seen_count: int


class GuidanceEventTracker:
    """Small in-memory tracker for event-based speech suppression.

    YOLO runs frame-by-frame and does not expose stable object IDs here, so this
    tracker uses a coarse situation key: object class and a bounding-box center
    bucket. That is enough to suppress repeated narration without introducing a
    heavy multi-object tracker.
    """

    def __init__(
        self,
        *,
        global_cooldown_seconds: float = 1.5,
        object_cooldown_seconds: float = 6.0,
        situation_cooldown_seconds: float = 8.0,
        signature_cooldown_seconds: float = 10.0,
        stale_after_seconds: float = 12.0,
    ) -> None:
        self.global_cooldown_seconds = global_cooldown_seconds
        self.object_cooldown_seconds = object_cooldown_seconds
        self.situation_cooldown_seconds = situation_cooldown_seconds
        self.signature_cooldown_seconds = signature_cooldown_seconds
        self.stale_after_seconds = stale_after_seconds
        self._states: dict[str, SituationState] = {}
        self._last_situation_at: dict[str, float] = {}
        self._last_signature_at: dict[str, float] = {}
        self._last_spoken_at = 0.0
        self._lock = threading.Lock()

    def choose_events(self, detections: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
        now = time.monotonic()
        candidates: list[dict[str, Any]] = []

        with self._lock:
            self._expire_old_states(now)
            for detection in detections:
                if not self._passes_speech_confidence(detection):
                    self._remember_seen(detection, now, speech_eligible=False)
                    continue

                if not self._is_temporally_confirmed(detection):
                    self._remember_seen(detection, now)
                    continue

                event = self._classify_event(detection, now)
                self._remember_seen(detection, now)
                if event:
                    candidates.append(event)

            selected = []
            for event in sorted(candidates, key=_event_sort_key):
                event["speech_signature"] = _speech_signature(event)
                message = build_detection_message(event["detection"], str(event["event"]))
                if not message:
                    continue

                event["message"] = message
                if not self._can_speak(event, now):
                    continue

                selected.append(event)
                self._mark_spoken(event, now)
                if len(selected) >= limit:
                    break

        return selected

    def _classify_event(self, detection: dict[str, Any], now: float) -> dict[str, Any] | None:
        label = str(detection.get("label", ""))
        if label not in GUIDANCE_LABELS:
            return None

        key = _object_key(detection)
        state = self._states.get(key)
        risk_score = int(detection.get("risk_score", 0))
        risk_level = str(detection.get("risk_level", risk_level_from_score(risk_score)))
        position = str(detection.get("position", "center"))
        area_ratio = float(detection.get("area_ratio", 0.0))
        is_center_front = bool(detection.get("front_danger_zone")) or position == "center"
        approaching = bool(detection.get("approaching"))

        if state is None:
            if risk_score >= 35 or is_center_front:
                return _event("new_object", detection, priority=45)
            return None

        if now - state.last_seen_at > self.stale_after_seconds:
            return _event("new_object", detection, priority=45)

        if state.last_spoken_at <= 0 and (risk_score >= 35 or is_center_front):
            return _event("new_object", detection, priority=45)

        if approaching and not state.approaching:
            return _event("approaching", detection, priority=95)

        if is_center_front and state.position != "center":
            return _event("entered_front_zone", detection, priority=85)

        score_delta = risk_score - state.risk_score
        level_delta = RISK_LEVEL_RANK.get(risk_level, 0) - RISK_LEVEL_RANK.get(state.risk_level, 0)
        area_delta = area_ratio - state.area_ratio

        if score_delta >= 18 or (level_delta >= 1 and score_delta >= 8):
            return _event("risk_increased", detection, priority=80 + max(level_delta, 0) * 5)

        if area_delta >= 0.035 and risk_score >= 45:
            return _event("closer", detection, priority=72)

        return None

    def _remember_seen(
        self,
        detection: dict[str, Any],
        now: float,
        *,
        speech_eligible: bool = True,
    ) -> None:
        key = _object_key(detection)
        previous = self._states.get(key)
        previous_seen_count = previous.seen_count if previous else 0
        self._states[key] = SituationState(
            last_seen_at=now,
            last_spoken_at=previous.last_spoken_at if previous else 0.0,
            risk_score=int(detection.get("risk_score", 0)),
            risk_level=str(
                detection.get(
                    "risk_level",
                    risk_level_from_score(int(detection.get("risk_score", 0))),
                )
            ),
            position=str(detection.get("position", "center")),
            area_ratio=float(detection.get("area_ratio", 0.0)),
            approaching=bool(detection.get("approaching")),
            seen_count=previous_seen_count + 1 if speech_eligible else previous_seen_count,
        )

    def _can_speak(self, event: dict[str, Any], now: float) -> bool:
        detection = event["detection"]
        key = _object_key(detection)
        situation_key = _situation_key(detection, str(event["event"]))
        state = self._states.get(key)

        if now - self._last_spoken_at < self.global_cooldown_seconds:
            return False

        object_cooldown = (
            2.5
            if event["event"] in {"approaching", "entered_front_zone"}
            else self.object_cooldown_seconds
        )
        if state and now - state.last_spoken_at < object_cooldown:
            return False

        if now - self._last_situation_at.get(situation_key, 0.0) < self.situation_cooldown_seconds:
            return False

        signature = str(event.get("speech_signature", ""))
        if signature and now - self._last_signature_at.get(signature, 0.0) < self.signature_cooldown_seconds:
            return False

        return True

    def _mark_spoken(self, event: dict[str, Any], now: float) -> None:
        detection = event["detection"]
        key = _object_key(detection)
        situation_key = _situation_key(detection, str(event["event"]))
        signature = str(event["speech_signature"])

        state = self._states.get(key)
        if state:
            state.last_spoken_at = now
        self._last_situation_at[situation_key] = now
        self._last_signature_at[signature] = now
        self._last_spoken_at = now

    def _expire_old_states(self, now: float) -> None:
        expired = [
            key
            for key, state in self._states.items()
            if now - state.last_seen_at > self.stale_after_seconds
        ]
        for key in expired:
            del self._states[key]

    def _passes_speech_confidence(self, detection: dict[str, Any]) -> bool:
        label = str(detection.get("label", ""))
        confidence = max(0.0, min(1.0, float(detection.get("confidence", 0.0))))

        if label in LOW_CONFIDENCE_SPEECH_LABELS:
            return confidence >= 0.25
        if label in NOISY_SPEECH_LABELS:
            return confidence >= 0.55
        return confidence >= 0.35

    def _is_temporally_confirmed(self, detection: dict[str, Any]) -> bool:
        label = str(detection.get("label", ""))
        confidence = max(0.0, min(1.0, float(detection.get("confidence", 0.0))))

        if label in IMMEDIATE_SPEECH_LABELS and confidence >= 0.35:
            return True
        if bool(detection.get("approaching")):
            return True

        state = self._states.get(_object_key(detection))
        seen_count = (state.seen_count + 1) if state else 1
        required_count = 3 if label in NOISY_SPEECH_LABELS else 2
        return seen_count >= required_count


GUIDANCE_TRACKER = GuidanceEventTracker()


def calculate_risk_score(detection: dict[str, Any]) -> int:
    label = str(detection.get("label", ""))
    position = str(detection.get("position", ""))
    confidence = max(0.0, min(1.0, float(detection.get("confidence", 0.0))))
    area_ratio = max(0.0, min(1.0, float(detection.get("area_ratio", 0.0))))
    vertical_ratio = max(0.0, min(1.0, float(detection.get("vertical_ratio", 0.0))))

    score = _base_risk(label)
    score += _position_bonus(position)
    score += min(24, int(area_ratio * 200))
    score += min(14, int(vertical_ratio * 14))
    score += int(confidence * 8)

    if detection.get("approaching"):
        score += 24 if position == "center" else 13

    if _is_front_danger_zone(detection):
        score += 10

    return max(0, min(100, score))


def enrich_and_prioritize_detections(
    detections: list[dict],
    korean_labels: dict[str, str],
) -> list[dict]:
    scored = []
    for detection in detections:
        normalized = dict(detection)
        label = str(normalized.get("label", ""))
        normalized["korean_label"] = korean_labels.get(
            label,
            str(normalized.get("korean_label", normalized.get("label", "장애물"))),
        )
        normalized["front_danger_zone"] = _is_front_danger_zone(normalized)
        normalized["risk_score"] = calculate_risk_score(normalized)
        normalized["risk_level"] = risk_level_from_score(int(normalized["risk_score"]))
        scored.append(normalized)

    return sorted(scored, key=_sort_detection_key)


def build_guidance(detections: list[dict[str, Any]]) -> dict[str, Any]:
    if not detections:
        return {
            "voice_guide": "",
            "warnings": [],
            "risk_level": "low",
            "selected_detection": None,
        }

    selected_detection = detections[0]
    events = GUIDANCE_TRACKER.choose_events(detections, limit=2)
    messages = [str(event["message"]) for event in events if event.get("message")]

    warning_messages = [
        build_warning_message(detection)
        for detection in detections[:2]
        if int(detection.get("risk_score", 0)) >= 55
    ]

    return {
        "voice_guide": " ".join(messages[:2]),
        "warnings": [message for message in warning_messages if message],
        "risk_level": str(selected_detection.get("risk_level", "low")),
        "selected_detection": selected_detection,
    }


def risk_level_from_score(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 68:
        return "high"
    if score >= 48:
        return "medium"
    return "low"


def build_warning_message(detection: dict[str, Any]) -> str:
    label = str(detection.get("label", ""))
    korean_label = str(detection.get("korean_label") or "장애물")
    if detection.get("approaching"):
        return f"{korean_label}{_particle_for(korean_label)} 가까워지고 있어요."
    if label in OBSTACLE_LABELS:
        return f"{korean_label} 주의"
    return f"{korean_label} 감지"


def build_detection_message(detection: dict[str, Any], event_type: str) -> str:
    label = str(detection.get("label", ""))
    korean_label = str(detection.get("korean_label") or "장애물")
    position = str(detection.get("position", "center"))
    position_ko = POSITION_KO.get(position, "정면")
    position_from_ko = POSITION_FROM_KO.get(position, "정면에서")
    risk_level = str(detection.get("risk_level", "low"))
    area_ratio = max(0.0, min(1.0, float(detection.get("area_ratio", 0.0))))
    vertical_ratio = max(0.0, min(1.0, float(detection.get("vertical_ratio", 0.0))))
    particle = _particle_for(korean_label)
    object_group = _object_group(label)
    approaching = bool(detection.get("approaching")) or event_type == "approaching"
    newly_detected = event_type == "new_object"
    distance_level = _distance_level(label, area_ratio, vertical_ratio)
    front_word = "앞쪽" if label in HIGHEST_RISK_LABELS else "앞"

    context = {
        "label": label,
        "korean_label": korean_label,
        "particle": particle,
        "position": position,
        "position_ko": position_ko,
        "position_from_ko": position_from_ko,
        "risk_level": risk_level,
        "object_group": object_group,
        "event_type": event_type,
        "approaching": approaching,
        "newly_detected": newly_detected,
        "distance_level": distance_level,
        "front_word": front_word,
    }

    templates = [
        ("왼쪽에 {korean_label}{particle} 있어요.", 18, _low_left, {"low", "side", "natural"}),
        ("오른쪽에 {korean_label}{particle} 있어요.", 18, _low_right, {"low", "side", "natural"}),
        ("{position_ko}에 {korean_label}{particle} 보여요.", 13, _medium_visible, {"medium", "new"}),
        ("{position_ko}에 {korean_label}{particle} 보여요. 조심해서 지나가 주세요.", 12, _medium_visible, {"medium", "action"}),
        ("{front_word}에 {korean_label}{particle} 있어요. 천천히 이동해 주세요.", 26, _front_obstacle, {"obstacle", "front", "action", "medium"}),
        ("{front_word}에 장애물이 있어요. 천천히 이동해 주세요.", 24, _front_obstacle, {"obstacle", "front", "action", "medium"}),
        ("정면에 {korean_label}{particle} 있어요. 조심해 주세요.", 24, _front_obstacle_named, {"obstacle", "front", "urgent"}),
        ("정면 {korean_label}{particle} 가까워지고 있어요. 잠시 멈춰 주세요.", 34, _center_approaching, {"approaching", "front", "urgent", "action"}),
        ("정면의 {korean_label}{particle} 가까워지고 있어요. 주의해 주세요.", 22, _center_approaching, {"approaching", "front", "urgent"}),
        ("주의해 주세요. {korean_label}{particle} 바로 앞에 있어요.", 34, _critical_front, {"critical", "front", "urgent", "action"}),
        ("잠시 멈춰 주세요. 정면에 {korean_label}{particle} 가까워요.", 30, _critical_front, {"critical", "front", "urgent", "action"}),
        ("{position_from_ko} {korean_label}{particle} 다가오고 있어요.", 18, _side_approaching, {"approaching", "side", "natural"}),
        ("{position_ko}의 {korean_label}{particle} 가까워지고 있어요.", 18, _side_approaching, {"approaching", "side"}),
        ("{position_from_ko} {korean_label}{particle} 지나가고 있어요.", 20, _side_person, {"person", "side", "natural"}),
        ("{position_ko}에 사람이 있어요. 살짝 주의해 주세요.", 11, _side_person, {"person", "side", "medium"}),
        ("{position_ko}에 {korean_label}{particle} 보여요.", 9, _side_person, {"person", "side", "new"}),
        ("{front_word}에 {korean_label}{particle} 보여요. 주의해 주세요.", 24, _front_vehicle, {"vehicle", "front", "new", "urgent"}),
        ("정면에 {korean_label}{particle} 가까워요. 조심해 주세요.", 22, _front_high_risk, {"front", "urgent", "near"}),
        ("{position_ko}에 {korean_label}{particle} 있어요. 주의해 주세요.", 13, _high_risk, {"urgent"}),
        ("{position_ko}에 {korean_label}{particle} 가까이 있어요.", 14, _high_near, {"urgent", "near", "natural"}),
        ("{position_ko}에 {korean_label}{particle} 있어요.", 8, _any_guidance, {"natural"}),
        ("{position_ko}에 {korean_label}{particle} 보여요.", 7, _any_guidance, {"new", "natural"}),
    ]

    weighted_templates = [
        (template, _template_weight(weight, context, tags))
        for template, weight, applies, tags in templates
        if applies(context)
    ]
    if not weighted_templates:
        weighted_templates = [("{position_ko}에 {korean_label}{particle} 있어요.", 1)]

    return _choose_weighted_template(weighted_templates).format(**context)


def _choose_weighted_template(weighted_templates: list[tuple[str, int]]) -> str:
    total = sum(max(0, weight) for _template, weight in weighted_templates)
    if total <= 0:
        return weighted_templates[0][0]

    cursor = _TEMPLATE_RANDOM.uniform(0, total)
    running = 0.0
    for template, weight in weighted_templates:
        running += max(0, weight)
        if cursor <= running:
            return template
    return weighted_templates[-1][0]


def _template_weight(base_weight: int, context: dict[str, Any], tags: set[str]) -> int:
    weight = base_weight
    event_type = str(context["event_type"])
    risk_level = str(context["risk_level"])
    object_group = str(context["object_group"])
    position = str(context["position"])
    distance_level = str(context["distance_level"])

    if event_type == "approaching":
        weight += 24 if "approaching" in tags else -4
    elif event_type in {"risk_increased", "closer", "entered_front_zone"}:
        weight += 18 if "urgent" in tags or "action" in tags else 2
    elif bool(context["newly_detected"]):
        weight += 12 if "new" in tags else 3

    if risk_level == "critical":
        weight += 34 if "critical" in tags else 18 if "urgent" in tags or "action" in tags else -8
    elif risk_level == "high":
        weight += 16 if "urgent" in tags or "action" in tags else 0
    elif risk_level == "medium":
        weight += 12 if "medium" in tags or "action" in tags else 2
    elif risk_level == "low":
        weight += 16 if "low" in tags or "natural" in tags else -6

    if position == "center":
        weight += 16 if "front" in tags or "action" in tags else 2
    elif position == "right":
        weight += 8 if "side" in tags else 1
    elif position == "left":
        weight += 7 if "side" in tags else 1

    if object_group == "vehicle":
        weight += 16 if "vehicle" in tags or "urgent" in tags else 0
    elif object_group in {"mobility", "obstacle"}:
        weight += 13 if "obstacle" in tags or "action" in tags else 3
    elif object_group == "person":
        weight += 12 if "person" in tags or "natural" in tags else 0

    if distance_level == "very_close":
        weight += 16 if "critical" in tags or "near" in tags or "action" in tags else -4
    elif distance_level == "close":
        weight += 10 if "near" in tags or "urgent" in tags or "action" in tags else 0
    elif distance_level == "far":
        weight += 8 if "low" in tags or "natural" in tags else -4

    return max(1, weight)


def _low_left(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["position"]) == "left"
        and str(context["risk_level"]) == "low"
    )


def _low_right(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["position"]) == "right"
        and str(context["risk_level"]) == "low"
    )


def _medium_visible(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["risk_level"]) == "medium"
        and str(context["distance_level"]) != "very_close"
    )


def _front_obstacle(context: dict[str, Any]) -> bool:
    return (
        str(context["position"]) == "center"
        and str(context["object_group"]) == "obstacle"
        and str(context["risk_level"]) in {"medium", "high", "critical"}
    )


def _front_obstacle_named(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["position"]) == "center"
        and str(context["object_group"]) == "obstacle"
        and str(context["risk_level"]) in {"high", "critical"}
    )


def _center_approaching(context: dict[str, Any]) -> bool:
    return bool(context["approaching"]) and str(context["position"]) == "center"


def _critical_front(context: dict[str, Any]) -> bool:
    return (
        str(context["risk_level"]) == "critical"
        and str(context["position"]) == "center"
        and str(context["distance_level"]) in {"close", "very_close"}
    )


def _side_approaching(context: dict[str, Any]) -> bool:
    return bool(context["approaching"]) and str(context["position"]) != "center"


def _side_person(context: dict[str, Any]) -> bool:
    return str(context["object_group"]) == "person" and str(context["position"]) != "center"


def _front_vehicle(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["object_group"]) == "vehicle"
        and str(context["position"]) == "center"
        and str(context["risk_level"]) != "critical"
    )


def _front_high_risk(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["position"]) == "center"
        and str(context["risk_level"]) in {"high", "critical"}
    )


def _high_risk(context: dict[str, Any]) -> bool:
    return not bool(context["approaching"]) and str(context["risk_level"]) == "high"


def _high_near(context: dict[str, Any]) -> bool:
    return (
        not bool(context["approaching"])
        and str(context["risk_level"]) == "high"
        and str(context["distance_level"]) in {"close", "very_close"}
    )


def _any_guidance(context: dict[str, Any]) -> bool:
    return not bool(context["approaching"]) and str(context["risk_level"]) in {"low", "medium"}


def _object_group(label: str) -> str:
    for group, labels in OBJECT_GROUPS.items():
        if label in labels:
            return group
    return "obstacle"


def _distance_level(label: str, area_ratio: float, vertical_ratio: float) -> str:
    thresholds = _distance_thresholds_for(label)
    if area_ratio >= thresholds.very_close_area or vertical_ratio >= thresholds.very_close_vertical:
        return "very_close"
    if area_ratio >= thresholds.close_area or vertical_ratio >= thresholds.close_vertical:
        return "close"
    if area_ratio >= thresholds.near_area or vertical_ratio >= thresholds.near_vertical:
        return "near"
    return "far"


def _distance_thresholds_for(label: str) -> DistanceThresholds:
    if label in GROUND_OBSTACLE_DISTANCE_LABELS:
        return DISTANCE_THRESHOLDS_BY_GROUP["ground_obstacle"]
    return DISTANCE_THRESHOLDS_BY_GROUP.get(_object_group(label), DEFAULT_DISTANCE_THRESHOLDS)


def _base_risk(label: str) -> int:
    if label in HIGHEST_RISK_LABELS:
        return 70
    if label in HIGH_RISK_LABELS:
        return 58
    if label in MEDIUM_HIGH_RISK_LABELS:
        return 48
    if label in MEDIUM_RISK_LABELS:
        return 36
    if label in LOWER_RISK_LABELS:
        return 22
    return 28


def _position_bonus(position: str) -> int:
    if position == "center":
        return 15
    if position == "right":
        return 6
    if position == "left":
        return 5
    return 0


def _is_front_danger_zone(detection: dict[str, Any]) -> bool:
    label = str(detection.get("label", ""))
    position = str(detection.get("position", ""))
    area_ratio = max(0.0, min(1.0, float(detection.get("area_ratio", 0.0))))
    vertical_ratio = max(0.0, min(1.0, float(detection.get("vertical_ratio", 0.0))))
    distance_level = _distance_level(label, area_ratio, vertical_ratio)
    return position == "center" and distance_level in {"close", "very_close"}


def _sort_detection_key(detection: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        -int(detection.get("risk_score", 0)),
        0 if detection.get("position") == "center" else 1,
        -float(detection.get("area_ratio", 0.0)),
        -float(detection.get("confidence", 0.0)),
    )


def _event(event_type: str, detection: dict[str, Any], priority: int) -> dict[str, Any]:
    return {
        "event": event_type,
        "detection": detection,
        "priority": priority,
    }


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int, float]:
    detection = event["detection"]
    return (
        -int(event["priority"]),
        -int(detection.get("risk_score", 0)),
        -float(detection.get("confidence", 0.0)),
    )


def _object_key(detection: dict[str, Any]) -> str:
    label = str(detection.get("label", ""))
    bbox = detection.get("bbox_xyxy") or []
    if len(bbox) >= 4:
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_bucket = int(center_x // 192)
    else:
        center_bucket = 0
    return f"{label}:{center_bucket}"


def _situation_key(detection: dict[str, Any], event_type: str) -> str:
    return f"{event_type}:{detection.get('label', '')}:{detection.get('position', '')}"


def _speech_signature(event: dict[str, Any]) -> str:
    detection = event["detection"]
    risk_level = str(detection.get("risk_level", "low"))
    event_type = str(event["event"])
    label = str(detection.get("label", ""))
    position = str(detection.get("position", "center"))
    distance_level = _distance_level(
        label,
        max(0.0, min(1.0, float(detection.get("area_ratio", 0.0)))),
        max(0.0, min(1.0, float(detection.get("vertical_ratio", 0.0)))),
    )

    return f"{event_type}:{label}:{position}:{risk_level}:{distance_level}"


def _particle_for(label: str) -> str:
    if not label:
        return "이"

    code = ord(label[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return "이" if (code - 0xAC00) % 28 else "가"

    return "이"
