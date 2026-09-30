"""Offline tests for pure local reactive obstacle-avoidance decisions."""

import copy
import math
from pathlib import Path

import pytest

from local_motion_safety_envelope import build_local_motion_lidar_geometry
from local_reactive_obstacle_avoidance import (
    FORWARD_CLEAR,
    STOP_BLOCKED,
    TURN_LEFT,
    TURN_RIGHT,
    decide_forward_reaction,
)


SESSION = "reactive-session"


def _scan(default=1.5):
    return {
        "frame_id": "lidar_link", "angle_min": -math.pi,
        "angle_increment": math.tau / 80, "range_min": 0.02,
        "range_max": 8.0, "ranges": [default] * 80,
    }


def _sector_name(bearing):
    if -22.5 <= bearing < 22.5:
        return "front"
    if 22.5 <= bearing < 67.5:
        return "front_left"
    if 67.5 <= bearing < 112.5:
        return "left"
    if 112.5 <= bearing < 157.5:
        return "rear_left"
    if bearing >= 157.5 or bearing < -157.5:
        return "rear"
    if -157.5 <= bearing < -112.5:
        return "rear_right"
    if -112.5 <= bearing < -67.5:
        return "right"
    return "front_right"


def _state(*points, age=0.05):
    geometry = build_local_motion_lidar_geometry(_scan())
    for x_m, y_m in points:
        distance = math.hypot(x_m, y_m)
        point = {
            "x_m": x_m, "y_m": y_m, "distance_m": distance,
            "robot_bearing_deg": math.degrees(math.atan2(y_m, x_m)),
        }
        geometry["points"].append(point)
        sector = geometry["sectors"][_sector_name(point["robot_bearing_deg"])]
        sector["valid_sample_count"] += 1
        sector["minimum_distance_from_base_m"] = min(
            sector["minimum_distance_from_base_m"], distance,
        )
    return {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": SESSION, "received_monotonic_seconds": 10.0,
        "age_at_receipt_seconds": age, "effective_age_seconds": age,
        "local_motion_geometry": geometry,
        # Explicitly prove this planner does not consume global authority.
        "localization_validated": False,
    }


def _decide(snapshot):
    return decide_forward_reaction(snapshot, expected_session=SESSION, now=10.0)


def _set_side_clearance(snapshot, *, left, right):
    sectors = snapshot["local_motion_geometry"]["sectors"]
    for name in ("front_left", "left", "rear_left"):
        sectors[name]["minimum_distance_from_base_m"] = left
    for name in ("front_right", "right", "rear_right"):
        sectors[name]["minimum_distance_from_base_m"] = right


def test_clear_front_returns_forward_clear_without_turn_evaluation():
    result = _decide(_state())
    assert result["decision"] == FORWARD_CLEAR
    assert result["forward"]["permitted"] is True
    assert result["left"]["permitted"] is None
    assert result["right"]["permitted"] is None
    assert not {"linear_x", "angular_z", "duration"} & set(result)


def test_blocked_front_with_more_usable_left_space_turns_left():
    result = _decide(_state((0.60, 0.0), (0.20, -0.50)))
    assert result["decision"] == TURN_LEFT
    assert result["forward"]["permitted"] is False
    assert result["left"]["permitted"] is True
    assert result["right"]["permitted"] is True
    assert result["left"]["relevant_clearance_m"] > result["right"]["relevant_clearance_m"]


def test_blocked_front_with_more_usable_right_space_turns_right():
    result = _decide(_state((0.60, 0.0), (0.20, 0.50)))
    assert result["decision"] == TURN_RIGHT
    assert result["left"]["permitted"] is True
    assert result["right"]["permitted"] is True
    assert result["right"]["relevant_clearance_m"] > result["left"]["relevant_clearance_m"]


def test_equal_safe_sides_use_documented_left_tie_breaker():
    snapshot = _state((0.60, 0.0))
    _set_side_clearance(snapshot, left=1.0, right=1.0)
    result = _decide(snapshot)
    assert result["decision"] == TURN_LEFT
    assert result["reason"] == "usable_clearance_tie_left_preferred"


def test_front_and_both_turns_blocked_returns_stop():
    result = _decide(_state((0.60, 0.0), (0.20, 0.0)))
    assert result["decision"] == STOP_BLOCKED
    assert result["left"]["permitted"] is False
    assert result["right"]["permitted"] is False


@pytest.mark.parametrize("mutator", [
    lambda value: value.update(age_at_receipt_seconds=0.31),
    lambda value: value["local_motion_geometry"].update(valid=False, reason="invalid_lidar_geometry"),
    lambda value: value["local_motion_geometry"]["sectors"]["rear"].update(valid_sample_count=0),
    lambda value: value["local_motion_geometry"]["points"].append({"x_m": math.nan, "y_m": 0.0}),
])
def test_stale_invalid_missing_or_nonfinite_geometry_fails_closed(mutator):
    result = _decide((lambda snapshot: (mutator(snapshot), snapshot)[1])(_state((0.60, 0.0))))
    assert result["decision"] == STOP_BLOCKED


def test_human_like_front_obstacle_left_clear_right_constrained_turns_left():
    result = _decide(_state((0.60, 0.0), (0.20, -0.50)))
    assert result["decision"] == TURN_LEFT
    assert result["right"]["relevant_clearance_m"] == pytest.approx(math.hypot(0.20, -0.50))


def test_pure_local_decision_needs_no_map_camera_or_world_model_and_mutates_nothing():
    snapshot = _state((0.60, 0.0))
    original = copy.deepcopy(snapshot)
    result = _decide(snapshot)
    assert result["decision"] == TURN_LEFT
    assert snapshot == original
    source = Path(__file__).with_name("local_reactive_obstacle_avoidance.py").read_text()
    for forbidden in ("robot_bridge", "cmd_vel", "NavigateToPose", "BehaviorManager", "request_json("):
        assert forbidden not in source
