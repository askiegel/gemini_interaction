"""Offline tests for pure envelope-backed local avoidance recommendations."""

import copy
import math

import pytest

from local_motion_safety_envelope import build_local_motion_lidar_geometry
from local_obstacle_policy import (
    LOCAL_AVOIDANCE_CANDIDATES,
    plan_local_obstacle_avoidance,
)


SESSION = "avoidance-session"


def scan(default=1.5):
    return {
        "frame_id": "lidar_link", "angle_min": -math.pi,
        "angle_increment": math.tau / 80, "range_min": 0.02,
        "range_max": 8.0, "ranges": [default] * 80,
    }


def state(*points, session=SESSION, age=0.05):
    geometry = build_local_motion_lidar_geometry(scan())
    for x_m, y_m in points:
        point = {
            "x_m": x_m, "y_m": y_m,
            "distance_m": math.hypot(x_m, y_m),
            "robot_bearing_deg": math.degrees(math.atan2(y_m, x_m)),
        }
        geometry["points"].append(point)
        bearing = point["robot_bearing_deg"]
        sectors = geometry["sectors"]
        name = (
            "front" if -22.5 <= bearing < 22.5 else
            "front_left" if 22.5 <= bearing < 67.5 else
            "left" if 67.5 <= bearing < 112.5 else
            "rear_left" if 112.5 <= bearing < 157.5 else
            "rear" if bearing >= 157.5 or bearing < -157.5 else
            "rear_right" if -157.5 <= bearing < -112.5 else
            "right" if -112.5 <= bearing < -67.5 else "front_right"
        )
        sectors[name]["valid_sample_count"] += 1
        sectors[name]["minimum_distance_from_base_m"] = min(
            sectors[name]["minimum_distance_from_base_m"], point["distance_m"],
        )
    return {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": session, "received_monotonic_seconds": 10.0,
        "age_at_receipt_seconds": age, "effective_age_seconds": age,
        "local_motion_geometry": geometry,
    }


def plan(snapshot):
    return plan_local_obstacle_avoidance(
        snapshot, expected_session=SESSION, now=10.0,
    )


def assert_selected_is_permitted(result):
    selected = result["selected_action"]
    if selected is not None:
        assert result["candidate_evaluations"][selected]["permitted"] is True


def test_forward_clear_selects_desired_motion():
    result = plan(state())
    assert result["selected_action"] == "forward"
    assert result["reason"] == "desired_motion_clear"
    assert set(result["candidate_evaluations"]) == set(LOCAL_AVOIDANCE_CANDIDATES)
    assert_selected_is_permitted(result)


def test_live_table_leg_does_not_unnecessarily_trigger_avoidance():
    result = plan(state((0.362, 0.482)))
    assert result["candidate_evaluations"]["forward"]["permitted"] is True
    assert result["selected_action"] == "forward"


def test_centered_front_blocker_selects_deterministic_safe_side():
    result = plan(state((0.60, 0.0)))
    assert result["candidate_evaluations"]["forward"]["permitted"] is False
    assert result["candidate_evaluations"]["left_turn"]["permitted"] is True
    assert result["selected_action"] == "forward_left"
    assert_selected_is_permitted(result)


def test_front_left_path_blocker_prefers_right_side_avoidance():
    result = plan(state((0.60, 0.25)))
    assert result["candidate_evaluations"]["forward"]["permitted"] is False
    assert result["blocking_point"]["y_m"] > 0
    assert result["selected_action"] == "forward_right"
    assert_selected_is_permitted(result)


def test_front_right_path_blocker_prefers_left_side_avoidance():
    result = plan(state((0.60, -0.25)))
    assert result["candidate_evaluations"]["forward"]["permitted"] is False
    assert result["blocking_point"]["y_m"] < 0
    assert result["selected_action"] == "forward_left"
    assert_selected_is_permitted(result)


def test_denied_left_candidate_selects_permitted_right_candidate():
    result = plan(state((0.60, 0.0), (0.45, 0.45)))
    assert result["candidate_evaluations"]["forward_left"]["permitted"] is False
    assert result["candidate_evaluations"]["forward_right"]["permitted"] is True
    assert result["selected_action"] == "forward_right"
    assert_selected_is_permitted(result)


def test_denied_right_candidate_selects_permitted_left_candidate():
    result = plan(state((0.60, 0.0), (0.45, -0.45)))
    assert result["candidate_evaluations"]["forward_right"]["permitted"] is False
    assert result["candidate_evaluations"]["forward_left"]["permitted"] is True
    assert result["selected_action"] == "forward_left"
    assert_selected_is_permitted(result)


def test_both_sides_with_footprint_violation_fails_closed():
    result = plan(state((0.60, 0.25), (0.60, -0.25), (0.10, 0.0)))
    assert result["selected_action"] is None
    assert result["reason"] == "operational_footprint_violated"
    assert all(not item["permitted"]
               for item in result["candidate_evaluations"].values())


@pytest.mark.parametrize("mutator,reason", [
    (lambda value: value.update(age_at_receipt_seconds=0.31), "stale"),
    (lambda value: value.update(producer_session="wrong"), "producer_session_mismatch"),
    (lambda value: value.update(local_motion_geometry={
        "valid": False, "reason": "invalid_lidar_geometry",
    }), "invalid_lidar_geometry"),
])
def test_untrusted_or_malformed_lidar_never_selects_action(mutator, reason):
    snapshot = state()
    mutator(snapshot)
    result = plan(snapshot)
    assert result["selected_action"] is None
    assert result["reason"] == reason
    assert all(not item["permitted"]
               for item in result["candidate_evaluations"].values())


def test_identical_state_returns_identical_recommendation_without_mutation():
    snapshot = state((0.60, 0.25))
    original = copy.deepcopy(snapshot)
    first = plan(snapshot)
    second = plan(snapshot)
    assert first["selected_action"] == second["selected_action"]
    assert first["reason"] == second["reason"]
    assert snapshot == original
