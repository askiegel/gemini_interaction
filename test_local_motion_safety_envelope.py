"""Offline tests for base-link bounded local-motion geometry only."""

import math

import pytest

from local_motion_safety_envelope import (
    LIDAR_TO_BASE_X_M,
    LIDAR_TO_BASE_Y_M,
    LOCAL_LIDAR_PROTECTED_RADIUS_M,
    build_local_motion_lidar_geometry,
    evaluate_local_motion_safety,
)


SESSION = "mayday-session"


def scan(default=1.5):
    return {"frame_id": "lidar_link", "angle_min": -math.pi,
            "angle_increment": math.tau / 80, "range_min": 0.02,
            "range_max": 8.0, "ranges": [default] * 80}


def set_robot_bearing(payload, bearing_deg, distance):
    raw = math.radians(bearing_deg) - math.pi / 2
    index = round((raw - payload["angle_min"]) / payload["angle_increment"])
    payload["ranges"][index % len(payload["ranges"])] = distance


def state(payload=None, *, session=SESSION, age=0.05):
    geometry = build_local_motion_lidar_geometry(payload or scan())
    return {"available": True, "valid": True, "reason": "fresh",
            "producer_session": session, "received_monotonic_seconds": 10.0,
            "age_at_receipt_seconds": age, "effective_age_seconds": age,
            "local_motion_geometry": geometry}


def evaluate(payload=None, **command):
    return evaluate_local_motion_safety(
        state(payload), expected_session=SESSION, now=10.0, duration=0.5,
        **command,
    )


def test_base_link_transform_accounts_for_lidar_offset_and_yaw():
    payload = scan()
    set_robot_bearing(payload, 0, 1.0)
    point = min(build_local_motion_lidar_geometry(payload)["points"],
                key=lambda item: abs(item["robot_bearing_deg"]))
    assert point["x_m"] == pytest.approx(LIDAR_TO_BASE_X_M + 1.0)
    assert point["y_m"] == pytest.approx(LIDAR_TO_BASE_Y_M)


def test_scanner_clear_range_can_still_violate_base_link_protected_radius():
    payload = scan()
    set_robot_bearing(payload, 0, 0.70)  # >0.67 from scanner, <0.67 from base.
    result = evaluate(payload, linear_x=0.1)
    assert result["permitted"] is False
    assert result["reason"] == "translation_protected_region_violated"


@pytest.mark.parametrize("bearing,command", [
    (0, {"linear_x": 0.1}), (180, {"linear_x": -0.1}),
    (90, {"linear_y": 0.1}), (-90, {"linear_y": -0.1}),
    (45, {"linear_x": 0.1, "linear_y": 0.1}),
])
def test_directional_obstacle_blocks_its_translation(bearing, command):
    payload = scan()
    set_robot_bearing(payload, bearing, 0.50)
    result = evaluate(payload, **command)
    assert result["permitted"] is False
    assert result["reason"] == "translation_protected_region_violated"


@pytest.mark.parametrize("bearing, blocking, nonblocking", [
    (45, {"linear_x": 0.1}, ({"linear_x": -0.1}, {"linear_y": -0.1})),
    (45, {"linear_y": 0.1}, ({"linear_x": -0.1}, {"linear_y": -0.1})),
    (-135, {"linear_x": -0.1}, ({"linear_x": 0.1}, {"linear_y": 0.1})),
    (-135, {"linear_y": -0.1}, ({"linear_x": 0.1}, {"linear_y": 0.1})),
    (0, {"linear_x": 0.1}, ({"linear_x": -0.1},)),
    (180, {"linear_x": -0.1}, ({"linear_x": 0.1},)),
    (90, {"linear_y": 0.1}, ({"linear_y": -0.1},)),
    (-90, {"linear_y": -0.1}, ({"linear_y": 0.1},)),
])
def test_nearby_obstacle_only_vetoes_translations_in_its_directional_neighborhood(
        bearing, blocking, nonblocking):
    payload = scan()
    set_robot_bearing(payload, bearing, 0.50)
    assert evaluate(payload, **blocking)["reason"] == (
        "translation_protected_region_violated"
    )
    for command in nonblocking:
        result = evaluate(payload, **command)
        assert result["permitted"] is True
        assert result["reason"] == "protected_region_clear"


def test_relevant_sector_point_outside_swept_tube_does_not_veto_translation():
    payload = scan()
    set_robot_bearing(payload, 45, 1.0)
    result = evaluate(payload, linear_x=0.1)
    assert "front_left" in result["required_sectors"]
    assert result["permitted"] is True
    assert result["reason"] == "protected_region_clear"


def test_rotation_uses_same_circular_geometry_for_left_and_right():
    payload = scan()
    set_robot_bearing(payload, 45, 0.50)
    left = evaluate(payload, angular_z=0.4)
    right = evaluate(payload, angular_z=-0.4)
    assert left["permitted"] is right["permitted"] is False
    assert left["reason"] == right["reason"] == "rotation_protected_region_violated"
    assert left["violating_point"] == right["violating_point"]


def test_rotation_geometry_permits_points_outside_protected_circle():
    result = evaluate(scan(), angular_z=0.4)
    assert result["permitted"] is True
    assert result["protected_radius_m"] == pytest.approx(LOCAL_LIDAR_PROTECTED_RADIUS_M)


@pytest.mark.parametrize("command, expected", [
    ({"linear_x": 0.1}, {"front", "front_left", "front_right"}),
    ({"linear_x": -0.1}, {"rear", "rear_left", "rear_right"}),
    ({"linear_y": 0.1}, {"front_left", "left", "rear_left"}),
    ({"linear_y": -0.1}, {"front_right", "right", "rear_right"}),
    ({"linear_x": 0.1, "linear_y": 0.1}, {"front", "front_left", "left"}),
    ({"linear_x": 0.1, "linear_y": -0.1}, {"front", "front_right", "right"}),
    ({"linear_x": -0.1, "linear_y": 0.1}, {"left", "rear_left", "rear"}),
    ({"linear_x": -0.1, "linear_y": -0.1}, {"rear", "rear_right", "right"}),
])
def test_required_sectors_follow_command_direction_with_wrap_safe_rear(command, expected):
    result = evaluate(scan(), **command)
    assert set(result["required_sectors"]) == expected
    if command.get("linear_x", 0) < 0:
        assert "rear" in result["required_sectors"]
        assert "front" not in result["required_sectors"]


def test_rotation_requires_all_eight_octants_for_both_directions():
    expected = {
        "front", "front_left", "left", "rear_left", "rear", "rear_right",
        "right", "front_right",
    }
    assert set(evaluate(angular_z=0.4)["required_sectors"]) == expected
    assert set(evaluate(angular_z=-0.4)["required_sectors"]) == expected


def test_reverse_direction_wrap_boundary_selects_rear_not_front():
    # atan2's signed representation can approach either side of ±180°.
    for y in (1e-12, -1e-12):
        result = evaluate(linear_x=-0.1, linear_y=y)
        assert "rear" in result["required_sectors"]
        assert "front" not in result["required_sectors"]


def test_self_return_is_excluded_before_base_link_safety_evaluation():
    payload = scan()
    set_robot_bearing(payload, -10, 0.10)
    geometry = build_local_motion_lidar_geometry(payload)
    assert geometry["self_return_filtered_count"] == 1
    assert evaluate_local_motion_safety(
        {**state(payload), "local_motion_geometry": geometry}, expected_session=SESSION,
        angular_z=0.4, duration=0.5, now=10.0,
    )["permitted"] is True


@pytest.mark.parametrize("mutator,reason", [
    (lambda value: value.update(age_at_receipt_seconds=0.31), "stale"),
    (lambda value: value.update(producer_session="other"), "producer_session_mismatch"),
    (lambda value: value.update(local_motion_geometry={"valid": False, "reason": "invalid_lidar_geometry"}), "invalid_lidar_geometry"),
])
def test_untrusted_state_fails_closed(mutator, reason):
    snapshot = state()
    mutator(snapshot)
    result = evaluate_local_motion_safety(snapshot, expected_session=SESSION,
                                          angular_z=0.4, duration=0.5, now=10.0)
    assert result["permitted"] is False
    assert result["reason"] == reason


def test_missing_or_wrong_frame_fails_closed():
    assert build_local_motion_lidar_geometry({})["reason"] == "unsupported_lidar_frame"
    snapshot = state()
    snapshot["local_motion_geometry"] = build_local_motion_lidar_geometry({})
    result = evaluate_local_motion_safety(snapshot, expected_session=SESSION,
                                          angular_z=0.4, duration=0.5, now=10.0)
    assert result["permitted"] is False
    assert result["reason"] == "unsupported_lidar_frame"


def test_policy_is_pure_and_returns_all_octant_diagnostics_without_transport():
    geometry = build_local_motion_lidar_geometry(scan())
    assert set(geometry["sectors"]) == {
        "front", "front_left", "left", "rear_left", "rear", "rear_right",
        "right", "front_right",
    }
    assert evaluate(angular_z=0.4)["permitted"] is True
