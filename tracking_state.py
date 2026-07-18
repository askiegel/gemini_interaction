#!/usr/bin/env python3

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional


VISUAL_BEHAVIORS = {
    "FOLLOW_PERSON",
    "FIND_OBJECT",
}


def _number(
    value: Any,
) -> Optional[float]:
    """
    Convert a value to float without raising an exception.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(
    source: Dict[str, Any],
    *keys: str,
) -> Optional[float]:
    for key in keys:
        value = _number(source.get(key))

        if value is not None:
            return value

    return None


def _first_value(
    source: Dict[str, Any],
    *keys: str,
):
    for key in keys:
        value = source.get(key)

        if value is not None:
            return value

    return None


def _parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc,
        )

    return parsed.astimezone(timezone.utc)


def _age_ms(
    timestamp: Any,
) -> Optional[int]:
    parsed = _parse_timestamp(timestamp)

    if parsed is None:
        return None

    age_seconds = (
        datetime.now(timezone.utc) - parsed
    ).total_seconds()

    return max(
        0,
        int(round(age_seconds * 1000.0)),
    )


def _normalize_direction(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().upper()

    aliases = {
        "TURN_LEFT": "LEFT",
        "CENTERING_LEFT": "LEFT",
        "SEARCHING_LEFT": "LEFT",
        "TURN_RIGHT": "RIGHT",
        "CENTERING_RIGHT": "RIGHT",
        "SEARCHING_RIGHT": "RIGHT",
        "FORWARD": "CENTER",
        "APPROACHING": "CENTER",
        "CENTERED": "CENTER",
        "MAINTAINING_DISTANCE": "CENTER",
        "ARRIVED": "CENTER",
        "STOPPED": "STOP",
    }

    return aliases.get(text, text)


def _direction_from_state(
    state: Any,
) -> Optional[str]:
    text = str(state or "").strip().upper()

    if "LEFT" in text:
        return "LEFT"

    if "RIGHT" in text:
        return "RIGHT"

    if text in {
        "CENTERED",
        "APPROACHING",
        "MAINTAINING_DISTANCE",
        "ARRIVED",
        "TARGET_FOUND",
    }:
        return "CENTER"

    if text == "STOPPED":
        return "STOP"

    return None


def _distance_state(
    result: Dict[str, Any],
    target_area: Optional[float],
) -> Optional[str]:
    explicit = _first_value(
        result,
        "distance_state",
        "range_state",
    )

    if explicit is not None:
        return str(explicit).strip().upper()

    state = str(
        result.get("state") or ""
    ).strip().upper()

    if state in {
        "ARRIVED",
        "MAINTAINING_DISTANCE",
        "TOO_CLOSE",
        "TOO_FAR",
        "AT_DISTANCE",
    }:
        if state == "ARRIVED":
            return "AT_DISTANCE"

        if state == "MAINTAINING_DISTANCE":
            return "AT_DISTANCE"

        return state

    stop_area = _first_number(
        result,
        "stop_area",
        "arrival_area",
        "target_stop_area",
    )

    if (
        target_area is not None
        and stop_area is not None
    ):
        if target_area >= stop_area:
            return "AT_DISTANCE"

        return "TOO_FAR"

    if state == "APPROACHING":
        return "TOO_FAR"

    return None


def _identity_tracking_fields(
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract identity telemetry from a behavior result.

    Identity information may be present directly on the result, inside the
    selected vision result, or inside the original normalized detection.
    This function only normalizes existing data; it does not make identity
    decisions or alter matching behavior.
    """
    search_sources = []

    def add_source(value):
        if (
            isinstance(value, dict)
            and value not in search_sources
        ):
            search_sources.append(value)

    add_source(source)
    add_source(source.get("vision_result"))
    add_source(source.get("target"))

    for candidate in list(search_sources):
        add_source(candidate.get("raw_detection"))
        add_source(candidate.get("attributes"))

    def first_value(*keys):
        for candidate in search_sources:
            for key in keys:
                value = candidate.get(key)

                if value is not None:
                    return value

        return None

    diagnostics = first_value(
        "identity_diagnostics",
    )

    if not isinstance(diagnostics, dict):
        diagnostics = {}

    candidates = diagnostics.get("candidates")

    if not isinstance(candidates, list):
        candidates = []

    best_score = _number(
        diagnostics.get("best_score")
    )

    second_score = _number(
        diagnostics.get("second_score")
    )

    score_margin = _number(
        diagnostics.get("score_margin")
    )

    match_score = _number(
        first_value("identity_match_score")
    )

    if best_score is None:
        best_score = match_score

    return {
        "locked_entity_id": first_value(
            "locked_entity_id",
        ),
        "identity_id": first_value(
            "identity_id",
        ),
        "identity_status": first_value(
            "identity_status",
        ),
        "identity_match_score": match_score,
        "identity_ambiguous": bool(
            first_value("identity_ambiguous")
            or diagnostics.get("ambiguous")
        ),
        "identity_best_score": best_score,
        "identity_runner_up_score": second_score,
        "identity_score_margin": score_margin,
        "identity_decision": diagnostics.get(
            "decision"
        ),
    }


def empty_tracking_state(
    state: str = "IDLE",
) -> Dict[str, Any]:
    return {
        "active": False,
        "behavior": None,
        "state": state,
        "target_label": None,
        "target_confidence": None,
        "target_center_x": None,
        "target_center_y": None,
        "image_width": None,
        "image_height": None,
        "image_center_x": None,
        "horizontal_error": None,
        "center_tolerance_pixels": None,
        "target_area": None,
        "steering_direction": None,
        "distance_state": None,
        "vision_timestamp": None,
        "detection_age_ms": None,
        "locked_entity_id": None,
        "identity_id": None,
        "identity_status": None,
        "identity_match_score": None,
        "identity_ambiguous": False,
        "identity_best_score": None,
        "identity_runner_up_score": None,
        "identity_score_margin": None,
        "identity_decision": None,
        "bbox": None,
    }


def build_tracking_state(
    result: Optional[Dict[str, Any]],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize a BehaviorManager result into the runtime's public tracking
    contract.

    BehaviorManager remains responsible for selecting the bounded action.
    CognitiveRuntime owns the durable, dashboard-facing representation.
    """
    if not isinstance(result, dict):
        return deepcopy(
            previous
            or empty_tracking_state()
        )

    behavior = str(
        result.get("behavior") or ""
    ).strip().upper()

    state = str(
        result.get("state") or "UNKNOWN"
    ).strip().upper()

    if behavior not in VISUAL_BEHAVIORS:
        if behavior == "STOP":
            return empty_tracking_state(
                state="STOPPED",
            )

        return deepcopy(
            previous
            or empty_tracking_state()
        )

    target_data = result.get("target")

    if isinstance(target_data, dict):
        merged = {
            **target_data,
            **result,
        }
    else:
        merged = dict(result)

    identity = _identity_tracking_fields(
        merged
    )

    target_label = _first_value(
        merged,
        "target_label",
        "target_name",
        "label",
    )

    if target_label is None and isinstance(
        target_data,
        str,
    ):
        target_label = target_data

    target_center_x = _first_number(
        merged,
        "target_center_x",
        "center_x",
        "cx",
    )

    target_center_y = _first_number(
        merged,
        "target_center_y",
        "center_y",
        "cy",
    )

    image_width = _first_number(
        merged,
        "image_width",
        "frame_width",
    )

    image_height = _first_number(
        merged,
        "image_height",
        "frame_height",
    )

    image_center_x = _first_number(
        merged,
        "image_center_x",
        "frame_center_x",
    )

    if (
        image_center_x is None
        and image_width is not None
    ):
        image_center_x = image_width / 2.0

    horizontal_error = _first_number(
        merged,
        "horizontal_error",
        "center_error",
        "error_x",
        "x_error",
    )

    if (
        horizontal_error is None
        and target_center_x is not None
        and image_center_x is not None
    ):
        horizontal_error = (
            target_center_x - image_center_x
        )

    target_area = _first_number(
        merged,
        "target_area",
        "area",
    )

    confidence = _first_number(
        merged,
        "target_confidence",
        "confidence",
        "score",
    )

    tolerance = _first_number(
        merged,
        "center_tolerance_pixels",
        "center_tolerance",
        "tolerance_pixels",
    )

    steering_direction = _normalize_direction(
        _first_value(
            merged,
            "steering_direction",
            "direction",
            "turn_direction",
        )
    )

    if steering_direction is None:
        steering_direction = (
            _direction_from_state(state)
        )

    if (
        steering_direction is None
        and horizontal_error is not None
    ):
        effective_tolerance = (
            tolerance
            if tolerance is not None
            else 0.0
        )

        if horizontal_error < -effective_tolerance:
            steering_direction = "LEFT"
        elif horizontal_error > effective_tolerance:
            steering_direction = "RIGHT"
        else:
            steering_direction = "CENTER"

    vision_timestamp = _first_value(
        merged,
        "vision_timestamp",
        "detection_timestamp",
        "observation_timestamp",
        "timestamp",
        "last_seen",
    )

    detection_age_ms = _first_number(
        merged,
        "detection_age_ms",
    )

    if detection_age_ms is None:
        detection_age_ms = _age_ms(
            vision_timestamp
        )

    if detection_age_ms is not None:
        detection_age_ms = int(
            round(detection_age_ms)
        )

    bbox = _first_value(
        merged,
        "bbox",
        "bounding_box",
    )

    active = state not in {
        "STOPPED",
        "IDLE",
        "FAILED",
        "ERROR",
    }

    return {
        "active": active,
        "behavior": behavior or None,
        "state": state,
        "target_label": (
            str(target_label)
            if target_label is not None
            else None
        ),
        "target_confidence": confidence,
        "target_center_x": target_center_x,
        "target_center_y": target_center_y,
        "image_width": image_width,
        "image_height": image_height,
        "image_center_x": image_center_x,
        "horizontal_error": horizontal_error,
        "center_tolerance_pixels": tolerance,
        "target_area": target_area,
        "steering_direction": steering_direction,
        "distance_state": _distance_state(
            merged,
            target_area,
        ),
        "vision_timestamp": vision_timestamp,
        "detection_age_ms": detection_age_ms,
        "locked_entity_id": identity[
            "locked_entity_id"
        ],
        "identity_id": identity["identity_id"],
        "identity_status": identity[
            "identity_status"
        ],
        "identity_match_score": identity[
            "identity_match_score"
        ],
        "identity_ambiguous": identity[
            "identity_ambiguous"
        ],
        "identity_best_score": identity[
            "identity_best_score"
        ],
        "identity_runner_up_score": identity[
            "identity_runner_up_score"
        ],
        "identity_score_margin": identity[
            "identity_score_margin"
        ],
        "identity_decision": identity[
            "identity_decision"
        ],
        "bbox": bbox,
    }
