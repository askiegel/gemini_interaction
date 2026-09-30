"""Offline tests for the exact circular pure-rotation safety envelope."""

import math

import pytest

from local_motion_safety_envelope import build_local_motion_lidar_geometry
from rotational_swept_footprint import (
    ROTATIONAL_PROTECTED_RADIUS_M,
    evaluate_rotational_swept_footprint,
)


SESSION = "rotation-session"


def _geometry(default=1.5):
    return build_local_motion_lidar_geometry({
        "frame_id": "lidar_link", "angle_min": -math.pi,
        "angle_increment": math.tau / 80, "range_min": 0.02,
        "range_max": 8.0, "ranges": [default] * 80,
    })


def _state(geometry=None, *, age=0.05):
    return {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": SESSION, "received_monotonic_seconds": 10.0,
        "age_at_receipt_seconds": age, "effective_age_seconds": age,
        "local_motion_geometry": _geometry() if geometry is None else geometry,
    }


def _with_point(x_m, y_m):
    geometry = _geometry()
    bearing = math.degrees(math.atan2(y_m, x_m))
    point = {"x_m": x_m, "y_m": y_m, "distance_m": math.hypot(x_m, y_m),
             "robot_bearing_deg": bearing}
    geometry["points"].append(point)
    names = ("front", "front_left", "left", "rear_left", "rear",
             "rear_right", "right", "front_right")
    sector = names[int(((bearing + 22.5) % 360) // 45)]
    geometry["sectors"][sector]["valid_sample_count"] += 1
    return geometry


def _evaluate(direction="LEFT", geometry=None, speed=.25, duration=.5, age=.05):
    return evaluate_rotational_swept_footprint(
        _state(geometry, age=age), expected_session=SESSION, direction=direction,
        angular_speed=speed, duration=duration, now=10.0,
    )


@pytest.mark.parametrize("distance,permitted", [(.44, False), (.45, False), (.46, True)])
def test_exact_total_base_frame_radius(distance, permitted):
    result = _evaluate(geometry=_with_point(distance, 0.0))
    assert result["permitted"] is permitted
    assert result["protected_radius_m"] == pytest.approx(.45)


def test_known_live_0552_point_is_outside_the_total_circle():
    for direction in ("LEFT", "RIGHT"):
        result = _evaluate(direction, _with_point(0.0, .552))
        assert result["permitted"] is True
        assert result["reason"] == "rotational_protected_region_clear"


def test_left_and_right_have_identical_circular_collision_geometry():
    for geometry in (_with_point(.46, 0.0), _with_point(.44, 0.0)):
        left = _evaluate("LEFT", geometry)
        right = _evaluate("RIGHT", geometry)
        assert left["permitted"] is right["permitted"]
        assert left["reason"] == right["reason"]


def test_angle_is_reported_but_does_not_change_circular_collision_geometry():
    geometry = _with_point(.46, 0.0)
    short = _evaluate("LEFT", geometry, speed=.25, duration=.5)
    long = _evaluate("LEFT", geometry, speed=1.0, duration=math.pi / 6)
    assert short["permitted"] is long["permitted"] is True
    assert short["requested_angle_radians"] < long["requested_angle_radians"]
    assert short["angle_affects_collision_geometry"] is False
    assert long["angle_affects_collision_geometry"] is False


def test_genuine_circle_intersection_blocks_without_sector_label_authority():
    result = _evaluate("RIGHT", _with_point(.44, 0.0))
    assert result["permitted"] is False
    assert result["reason"] == "rotational_protected_region_violated"
    assert result["violating_point"]["x_m"] == pytest.approx(.44)


def test_stale_lidar_blocks_rotation():
    assert _evaluate(age=.31)["reason"] == "stale"


def test_invalid_or_insufficient_geometry_blocks_rotation():
    invalid = {"valid": False, "reason": "invalid_lidar_geometry"}
    assert _evaluate(geometry=invalid)["reason"] == "invalid_lidar_geometry"
    incomplete = _geometry()
    incomplete["sectors"]["rear_left"]["valid_sample_count"] = 0
    assert _evaluate(geometry=incomplete)["reason"] == "insufficient_lidar_samples"


def test_nonfinite_point_blocks_as_invalid_geometry():
    geometry = _geometry()
    geometry["points"].append({"x_m": math.nan, "y_m": 0.0})
    assert _evaluate(geometry=geometry)["reason"] == "invalid_lidar_geometry"


@pytest.mark.parametrize("direction", ["UP", None])
def test_invalid_direction_fails_closed(direction):
    assert _evaluate(direction)["permitted"] is False
