"""Offline tests for the advisory guarded-turn policy."""

import math

import pytest

from guarded_turn_policy import (
    MAX_ABSOLUTE_ANGULAR_SPEED,
    MAX_TURN_DURATION_SECONDS,
    validate_guarded_turn,
)
from local_motion_safety_envelope import build_local_motion_lidar_geometry


def _clear_geometry():
    return build_local_motion_lidar_geometry({
        "frame_id": "lidar_link", "angle_min": -math.pi,
        "angle_increment": math.tau / 80, "range_min": 0.02,
        "range_max": 8.0, "ranges": [1.5] * 80,
    })


def snapshot(*, front="CLEAR", left="CLEAR", front_left="CLEAR",
             right="CLEAR", front_right="CLEAR", session="session-1",
             age=0.05, now=10.0):
    def sector(state, clearance=1.0):
        return {"state": state, "available": True,
                "robust_clearance_m": clearance,
                "minimum_clearance_m": clearance}

    return {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": session, "received_monotonic_seconds": now,
        "age_at_receipt_seconds": age, "effective_age_seconds": age,
        "local_motion_geometry": _clear_geometry(),
        "sectors": {
            "front": sector(front), "front_left": sector(front_left),
            "front_right": sector(front_right), "left": sector(left),
            "right": sector(right),
        },
    }


_DEFAULT_STATE = object()


def turn(direction, state=_DEFAULT_STATE, speed=0.5, duration=0.4,
         target_directed=False):
    return validate_guarded_turn(
        direction, speed, duration, snapshot() if state is _DEFAULT_STATE else state,
        expected_session="session-1", now=10.0,
        target_directed=target_directed,
    )


def test_valid_left_turn_is_permitted_with_clear_directional_sectors():
    result = turn("LEFT")
    assert result["permitted"] is True
    assert result["angular_z"] == 0.5
    assert result["duration"] == 0.4




def test_target_directed_turn_uses_front_corridor_and_ignores_unrelated_side_caution():
    state = snapshot(left="CAUTION", front_left="CAUTION", right="CAUTION")
    result = turn("LEFT", state, target_directed=True)
    assert result["permitted"] is True
    assert result["reason"] == "turn_side_clear_advisory"




@pytest.mark.parametrize("direction,side", [("LEFT", "left"), ("RIGHT", "right")])
def test_target_directed_turn_allows_relevant_side_caution(direction, side):
    state = snapshot()
    state["sectors"][side]["state"] = "CAUTION"
    result = turn(direction, state, target_directed=True)
    assert result["permitted"] is True


@pytest.mark.parametrize("direction,side", [
    ("LEFT", "left"), ("LEFT", "front_left"),
    ("RIGHT", "right"), ("RIGHT", "front_right"),
])
def test_target_directed_turn_blocks_relevant_blocked_side(direction, side):
    state = snapshot()
    state["sectors"][side]["state"] = "BLOCKED"
    result = turn(direction, state, target_directed=True)
    assert result["permitted"] is False
    assert result["reason"] == "turn_side_not_clear"


def test_target_directed_turn_ignores_opposite_blocked_side():
    state = snapshot(right="BLOCKED", front_right="BLOCKED")
    assert turn("LEFT", state, target_directed=True)["permitted"] is True

def test_target_directed_turn_denies_nonclear_forward_corridor():
    result = turn("RIGHT", snapshot(front="CAUTION"), target_directed=True)
    assert result["permitted"] is False
    assert result["reason"] == "front_not_clear"

def test_valid_right_turn_is_permitted_with_clear_directional_sectors():
    result = turn("RIGHT")
    assert result["permitted"] is True
    assert result["angular_z"] == -0.5


def test_maximum_one_second_turn_is_permitted():
    result = turn("LEFT", duration=1.0)
    assert result["permitted"] is True
    assert result["duration"] == 1.0


def test_existing_half_second_turn_remains_permitted():
    result = turn("RIGHT", duration=0.5)
    assert result["permitted"] is True
    assert result["duration"] == 0.5


def test_duration_above_one_second_is_denied_without_clamping():
    result = turn("LEFT", duration=1.000001)
    assert result["permitted"] is False
    assert result["reason"] == "duration_exceeds_limit"
    assert result["duration"] == 1.000001


@pytest.mark.parametrize("direction", [None, "left", [], 1])
def test_invalid_direction_is_denied(direction):
    assert turn(direction)["permitted"] is False


def test_front_blocked_does_not_prevent_clear_in_place_turn():
    assert turn("LEFT", snapshot(front="BLOCKED"))["permitted"] is True
    assert turn("RIGHT", snapshot(front="BLOCKED"))["permitted"] is True


@pytest.mark.parametrize("front", [None, []])
def test_untrustworthy_front_container_is_denied(front):
    state = snapshot()
    state["sectors"]["front"] = front
    assert turn("LEFT", state)["permitted"] is False


@pytest.mark.parametrize("front_state", [[], {}, None, 123, "UNKNOWN", "clear", "UNRECOGNIZED"])
def test_untrustworthy_front_state_value_is_denied_without_exception(front_state):
    state = snapshot()
    state["sectors"]["front"]["state"] = front_state
    assert turn("LEFT", state)["permitted"] is False


def test_front_available_false_is_denied():
    state = snapshot()
    state["sectors"]["front"]["available"] = False
    assert turn("LEFT", state)["permitted"] is False


@pytest.mark.parametrize("field,value", [
    ("robust_clearance_m", None),
    ("robust_clearance_m", math.nan),
    ("robust_clearance_m", math.inf),
    ("robust_clearance_m", True),
    ("minimum_clearance_m", None),
    ("minimum_clearance_m", math.nan),
    ("minimum_clearance_m", math.inf),
    ("minimum_clearance_m", True),
])
def test_untrustworthy_front_metrics_are_denied(field, value):
    state = snapshot()
    state["sectors"]["front"][field] = value
    assert turn("LEFT", state)["permitted"] is False


@pytest.mark.parametrize("front", ["CAUTION", "BLOCKED"])
def test_legitimate_nonclear_front_allows_clear_turn_side(front):
    assert turn("LEFT", snapshot(front=front))["permitted"] is True


@pytest.mark.parametrize("state_name", ["BLOCKED", "CAUTION", "UNKNOWN"])
def test_blocked_caution_or_unknown_turn_side_is_denied(state_name):
    state = snapshot(left=state_name, front_left="CLEAR")
    result = turn("LEFT", state)
    assert result["permitted"] is False
    assert result["reason"] == "turn_side_not_clear"


def test_right_turn_side_is_gated_symmetrically():
    state = snapshot(right="CLEAR", front_right="CAUTION")
    assert turn("RIGHT", state)["permitted"] is False


@pytest.mark.parametrize("state", [None, {}, {"available": True, "valid": True, "reason": "fresh", "producer_session": "session-1", "sectors": []}])
def test_missing_or_malformed_lidar_is_denied(state):
    assert turn("LEFT", state)["permitted"] is False


def test_stale_unavailable_invalid_and_mismatched_lidar_are_denied():
    assert turn("LEFT", snapshot(age=0.31))["permitted"] is False
    unavailable = snapshot()
    unavailable.update(available=False, valid=False, reason="offline")
    assert turn("LEFT", unavailable)["permitted"] is False
    assert turn("LEFT", snapshot(session="other"))["permitted"] is False


@pytest.mark.parametrize("speed", [0, -0.1, math.nan, math.inf, -math.inf, True])
def test_invalid_angular_speed_is_denied(speed):
    assert turn("LEFT", speed=speed)["permitted"] is False


@pytest.mark.parametrize("duration", [0, -0.1, math.nan, math.inf, -math.inf, True])
def test_invalid_duration_is_denied(duration):
    assert turn("LEFT", duration=duration)["permitted"] is False


def test_hard_bounds_are_denied_without_clamping():
    too_fast = turn("LEFT", speed=MAX_ABSOLUTE_ANGULAR_SPEED + 0.01)
    too_long = turn("RIGHT", duration=MAX_TURN_DURATION_SECONDS + 0.01)
    assert too_fast["permitted"] is False and too_fast["angular_z"] is None
    assert too_long["permitted"] is False and too_long["duration"] == MAX_TURN_DURATION_SECONDS + 0.01


def test_direction_and_status_fields_are_structured():
    result = turn("LEFT", snapshot(front="BLOCKED"))
    assert result["direction"] == "LEFT"
    assert result["left_state"] == "CLEAR"
    assert result["front_left_state"] == "CLEAR"
    assert result["producer_session"] == "session-1"
    assert result["effective_age_seconds"] == pytest.approx(0.05)
    for name in ("front", "left", "front_left", "right", "front_right"):
        assert result[f"{name}_minimum_clearance_m"] == 1.0


def test_minimum_clearance_can_drive_caution_with_clear_robust_metric():
    state = snapshot()
    state["sectors"]["left"].update(
        state="CAUTION", robust_clearance_m=1.287, minimum_clearance_m=0.55
    )
    result = turn("LEFT", state)
    assert result["permitted"] is False
    assert result["reason"] == "turn_side_not_clear"
    assert result["left_robust_clearance_m"] == pytest.approx(1.287)
    assert result["left_minimum_clearance_m"] == pytest.approx(0.55)


def test_no_physical_execution_interface_is_used():
    result = turn("RIGHT")
    assert set(result) >= {"permitted", "reason", "angular_z", "duration"}
    assert "request" not in result
