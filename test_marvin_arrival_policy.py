"""Offline contracts for calibrated pure Marvin visual arrival policy."""

from copy import deepcopy
import inspect

import pytest

from marvin_arrival_policy import (
    MARVIN_ARRIVAL_AREA_FRACTION,
    MARVIN_ARRIVAL_HEIGHT_FRACTION,
    evaluate_marvin_arrival,
)


STAMP = "2026-09-26T16:00:00+00:00"
IDENTITY = "marvin-identity"


def snapshot(**updates):
    value = {"tracking_mode": "LOCKED", "locked_identity_id": IDENTITY}
    value.update(updates)
    return value


def lock(*, height=264.0, width=188.0, **updates):
    value = {
        "found": True, "stale": False, "tracking_mode": "LOCKED",
        "identity_id": IDENTITY, "locked_identity_id": IDENTITY,
        "last_seen": STAMP, "image_width": 640.0, "image_height": 480.0,
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 100.0 + width, "y2": 100.0 + height},
    }
    value.update(updates)
    return value


def evaluate(result=None, state=None, **kwargs):
    selected_identity_id = kwargs.pop("selected_identity_id", IDENTITY)
    return evaluate_marvin_arrival(
        result if result is not None else lock(), state if state is not None else snapshot(),
        selected_identity_id=selected_identity_id, now=STAMP, **kwargs,
    )


def test_calibration_constants_match_live_measurement_exactly():
    assert MARVIN_ARRIVAL_HEIGHT_FRACTION == 0.545833
    assert MARVIN_ARRIVAL_AREA_FRACTION == 0.160339


def test_fresh_locked_same_identity_at_or_above_height_threshold_arrives():
    at_threshold = 480.0 * MARVIN_ARRIVAL_HEIGHT_FRACTION
    assert evaluate(lock(height=264.0))["arrived_at_marvin"] is True
    assert evaluate(lock(height=at_threshold))["arrived_at_marvin"] is True


def test_height_just_below_threshold_or_large_area_alone_never_arrives():
    assert evaluate(lock(height=480.0 * 0.545832))["arrived_at_marvin"] is False
    assert evaluate(lock(height=200.0, width=500.0))["arrived_at_marvin"] is False


def test_height_authorizes_even_when_area_corroboration_is_not_met():
    value = evaluate(lock(height=264.0, width=100.0))
    assert value["arrived_at_marvin"] is True
    assert value["area_threshold_met"] is False


def test_exact_area_threshold_is_diagnostic_and_met():
    width = MARVIN_ARRIVAL_AREA_FRACTION * 640.0 * 480.0 / 264.0
    value = evaluate(lock(height=264.0, width=width))
    assert value["arrived_at_marvin"] is True
    assert value["area_fraction"] >= MARVIN_ARRIVAL_AREA_FRACTION
    assert value["area_threshold_met"] is True


def test_lock_modes_identities_and_ambiguity_fail_closed():
    assert evaluate(lock(tracking_mode="WAITING_FOR_IDENTITY"), snapshot(tracking_mode="WAITING_FOR_IDENTITY"))["arrived_at_marvin"] is False
    assert evaluate(lock(tracking_mode="UNLOCKED"), snapshot(tracking_mode="UNLOCKED"))["arrived_at_marvin"] is False
    assert evaluate(lock(identity_id="other"))["arrived_at_marvin"] is False
    assert evaluate(lock(identity_ambiguous=True))["arrived_at_marvin"] is False
    assert evaluate(selected_identity_id=None)["arrived_at_marvin"] is False


def test_stale_or_missing_freshness_fails_closed():
    assert evaluate(lock(last_seen="2026-09-26T15:59:56+00:00"))["arrived_at_marvin"] is False
    assert evaluate(lock(last_seen=None))["arrived_at_marvin"] is False
    assert evaluate(lock(last_seen="not-a-timestamp"))["arrived_at_marvin"] is False


def test_missing_malformed_or_out_of_bounds_geometry_fails_closed():
    cases = (
        lock(bbox=None), lock(bbox={"x1": 1}), lock(image_width=None),
        lock(image_height=0), lock(image_width=-1),
        lock(bbox={"x1": -1, "y1": 1, "x2": 100, "y2": 300}),
        lock(bbox={"x1": 100, "y1": 1, "x2": 100, "y2": 300}),
    )
    assert all(not evaluate(item)["arrived_at_marvin"] for item in cases)


@pytest.mark.parametrize("bbox", (
    {"x1": 100, "y1": 100, "x2": 100, "y2": 300},
    {"x1": 100, "y1": 300, "x2": 300, "y2": 300},
    {"x1": 300, "y1": 100, "x2": 100, "y2": 300},
    {"x1": 100, "y1": 300, "x2": 300, "y2": 100},
    {"x1": 100, "y1": 100, "x2": 700, "y2": 300},
))
def test_all_degenerate_reversed_or_out_of_image_bboxes_fail_closed(bbox):
    assert evaluate(lock(bbox=bbox))["arrived_at_marvin"] is False


@pytest.mark.parametrize("updates", (
    {"image_width": None}, {"image_height": None},
    {"image_width": 0}, {"image_height": 0},
    {"image_width": -640}, {"image_height": -480},
))
def test_missing_zero_or_negative_image_dimensions_fail_closed(updates):
    assert evaluate(lock(**updates))["arrived_at_marvin"] is False


@pytest.mark.parametrize("updates", (
    {"found": False}, {"stale": True},
    {"tracking_mode": "WAITING_FOR_IDENTITY"},
    {"tracking_mode": "UNLOCKED"},
    {"identity_id": "another-identity"},
    {"identity_ambiguous": True},
))
def test_noncurrent_or_non_authoritative_lock_states_fail_closed(updates):
    state = snapshot(tracking_mode=updates.get("tracking_mode", "LOCKED"))
    assert evaluate(lock(**updates), state)["arrived_at_marvin"] is False


def test_preview_bridge_and_unrelated_extras_cannot_establish_arrival():
    malformed_lock = {"preview": True, "target": "marvin", "identity_confirmed": True}
    assert evaluate(malformed_lock)["arrived_at_marvin"] is False
    assert evaluate(lock(found=False), bridge_result={"classification": "same_identity_reacquired"})["arrived_at_marvin"] is False
    assert evaluate(lock(found=False), lidar_snapshot={"front": "BLOCKED"})["arrived_at_marvin"] is False
    assert evaluate(lock(found=False), preview_result={"target": "marvin", "identity_confirmed": True})["arrived_at_marvin"] is False
    assert evaluate(lock(found=False), tracker_id="reused-track", bbox_overlap=1.0)["arrived_at_marvin"] is False
    assert evaluate(lock(found=False), obstacle_state="CLEAR", clearance_m=99.0)["arrived_at_marvin"] is False


def test_deterministic_and_inputs_not_mutated():
    values = (lock(), snapshot())
    before = deepcopy(values)
    assert evaluate(*values) == evaluate(*values)
    assert values == before


def test_policy_has_no_transport_or_safety_dependencies():
    import marvin_arrival_policy
    source = inspect.getsource(marvin_arrival_policy).lower()
    for forbidden in ("lidar", "robot_bridge", "robot_client", "rospy", "rclpy", "stanford", "nav2", "local_motion", "obstacle_avoidance", "guarded_turn", "local_forward", "cmd_vel"):
        assert forbidden not in source
