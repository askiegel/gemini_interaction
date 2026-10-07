"""Offline camera/LiDAR association and strict V2 metric arrival tests."""

import math
import time
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from behavior_manager import BehaviorManager
from local_motion_safety_envelope import evaluate_local_motion_safety, build_local_motion_lidar_geometry
from marvin_lidar_standoff import evaluate_marvin_lidar_standoff
from test_find_marvin_runtime import _v2_preview, _v2_runtime
from test_local_motion_safety_envelope import geometry_with_point


CAMERA = {"fx_pixels": 320.0, "cx_pixels": 320.0, "image_width": 640,
          "x_m": 0.0, "y_m": 0.0, "yaw_degrees": 0.0, "range_uncertainty_m": 0.0}
SESSION = "v2-lidar"


def lidar_at(distance, *, sequence=1):
    geometry = build_local_motion_lidar_geometry({
        "frame_id": "lidar_link", "angle_min": -math.pi, "angle_increment": math.tau / 80,
        "range_min": .02, "range_max": 8., "ranges": [2.] * 80})
    geometry["points"].extend({"x_m": distance, "y_m": n * distance * .12 / 3} for n in range(-3, 4))
    return {"available": True, "valid": True, "reason": "fresh",
            "producer_session": SESSION, "effective_age_seconds": 0.0,
            "received_monotonic_seconds": time.monotonic(), "age_at_receipt_seconds": 0.0,
            "acquisition_sequence": sequence,
            "local_motion_geometry": geometry}


def runtime_at(distance):
    runtime, behavior = _v2_runtime(fresh_preview())
    runtime.running = True
    runtime.world_model.get_lidar_obstacles.return_value = lidar_at(distance)
    from marvin_target_range_association import MarvinTargetRangeAssociation
    runtime._marvin_target_range_association = MarvinTargetRangeAssociation()
    if distance <= .50:
        runtime._marvin_target_range_association.anchor = {
            "measured_distance_m": distance, "target_distance_m": distance,
            "producer_session": SESSION, "acquisition_sequence": 0,
            "source_frame_stamp_ns": 0, "translation_bound_m": 0.0}
    behavior.robot = Mock()
    behavior.robot.stop.return_value = {"ok": True}
    behavior.mark_strict_v2_action_dispatched = Mock(return_value=True)
    behavior.execute_single_marvin_approach_step = Mock(return_value={
        "ok": True, "motion_executed": True,
    })
    return runtime, behavior


def fresh_preview(stamp=101):
    preview = _v2_preview(0.0, stamp=stamp)
    now = datetime.now(timezone.utc).isoformat()
    preview.update(source_timestamp=now, vision_timestamp=now)
    return preview


@pytest.mark.parametrize("distance,decision", [
    (1.0, "FORWARD"), (0.75, "FORWARD"), (0.501, "FORWARD"),
    (0.50, "ARRIVED"), (0.49, "ARRIVED"), (0.45, "ARRIVED"), (0.44, "ARRIVED"),
])
def test_centered_metric_arrival_controls_forward_authorization(distance, decision):
    runtime, _ = runtime_at(distance)
    result = runtime.observe_find_marvin_v2()
    assert result["controller"]["decision"] == decision
    assert result["arrival"]["target_distance_m"] == pytest.approx(distance)
    assert result["arrival"]["hard_safety_condition"] is (distance <= 0.45)
    assert (runtime._marvin_alignment_observation is not None) is (decision == "FORWARD")


def test_large_visual_arrival_is_diagnostic_only_at_one_meter():
    runtime, behavior = runtime_at(1.0)
    for box in (behavior.preview["bbox"], behavior.preview["opencv_tracker"]["bbox"]):
        box.update(x1=100, y1=10, x2=540, y2=470)
    behavior.preview["opencv_tracker"]["center_y"] = 240
    result = runtime.observe_find_marvin_v2()
    assert result["visual_arrival"]["arrived_at_marvin"] is True
    assert result["arrival"]["arrived_at_marvin"] is False
    assert result["controller"]["decision"] == "FORWARD"


@pytest.mark.parametrize("changes", [
    {"valid": False}, {"reason": "stale"}, {"effective_age_seconds": 0.31},
    {"producer_session": "old"}, {"local_motion_geometry": {"valid": False}},
])
def test_invalid_target_lidar_blocks_current_observation(changes):
    runtime, _ = runtime_at(1.0)
    runtime.world_model.get_lidar_obstacles.return_value.update(changes)
    result = runtime.observe_find_marvin_v2()
    assert result["controller"]["decision"] == "BLOCKED"
    assert runtime._marvin_alignment_observation is None


def test_camera_calibration_is_mandatory_and_resolution_bound():
    tracker = _v2_preview(0.0)["opencv_tracker"]
    for model in (None, {}, dict(CAMERA, image_width=1280), dict(CAMERA, fx_pixels=0)):
        assert evaluate_marvin_lidar_standoff(tracker, lidar_at(1.0), model,
                                             expected_session=SESSION)["ok"] is False


def test_target_band_uses_camera_projection_not_global_minimum_or_front_center():
    tracker = _v2_preview(32.0)["opencv_tracker"]
    lidar = lidar_at(1.0)
    # The target center projects to -atan(0.1): positive image error is right.
    lidar["local_motion_geometry"]["points"] = [
        {"x_m": 1.0, "y_m": -0.1 + n / 1000} for n in range(-3, 4)
    ] + [{"x_m": 0.2, "y_m": 0.3}]  # unrelated front-left obstacle
    result = evaluate_marvin_lidar_standoff(tracker, lidar, CAMERA, expected_session=SESSION)
    assert result["ok"] is True and result["target_distance_m"] > 1.0
    assert result["point_count"] == 7
    assert result["target_bearing_degrees"] == pytest.approx(-math.degrees(math.atan(0.1)))
    # A changed yaw correctly makes those returns no longer associate.
    assert evaluate_marvin_lidar_standoff(tracker, lidar, dict(CAMERA, yaw_degrees=90),
                                         expected_session=SESSION)["ok"] is False


def test_range_uncertainty_stops_conservatively_and_missing_returns_block():
    tracker = _v2_preview(0.0)["opencv_tracker"]
    result = evaluate_marvin_lidar_standoff(tracker, lidar_at(0.51),
                                         dict(CAMERA, range_uncertainty_m=0.02), expected_session=SESSION)
    assert result["candidate_at_standoff"] is True
    assert result["arrived_at_marvin"] is False
    assert result["target_range_association_trusted"] is False
    missing = lidar_at(1.0)
    missing["local_motion_geometry"]["points"] = []
    assert evaluate_marvin_lidar_standoff(tracker, missing, CAMERA,
                                         expected_session=SESSION)["ok"] is False


def test_projection_accounts_for_camera_translation_and_requires_forward_target():
    tracker = _v2_preview(0.0)["opencv_tracker"]
    lidar = lidar_at(1.0)
    lidar["local_motion_geometry"]["points"] = [
        {"x_m": 1.0, "y_m": 0.10 + n / 1000.0} for n in range(-3, 4)
    ]
    result = evaluate_marvin_lidar_standoff(
        tracker, lidar, dict(CAMERA, x_m=0.10, y_m=0.10), expected_session=SESSION,
    )
    assert result["ok"] is True and result["target_distance_m"] > 1.0
    lidar["local_motion_geometry"]["points"] = [
        {"x_m": 1.0, "y_m": 1.0 + n / 1000.0} for n in range(-3, 4)
    ]
    sideways = evaluate_marvin_lidar_standoff(
        tracker, lidar, dict(CAMERA, yaw_degrees=45), expected_session=SESSION,
    )
    assert sideways["ok"] is False
    assert sideways["reason"] == "target_lidar_not_in_forward_sector"


@pytest.mark.parametrize("distance,max_duration", [(0.53, 0.30), (0.51, 0.10)])
def test_short_forward_bound_preserves_standoff_and_requires_new_camera_and_lidar(
        monkeypatch, distance, max_duration):
    monkeypatch.setattr("runtime.time.time_ns", lambda: 1000)
    runtime, behavior = runtime_at(distance)
    observation = runtime.observe_find_marvin_v2()
    stamp = observation["opencv_tracker"]["source_frame_stamp_ns"]
    first = runtime.execute_single_marvin_approach(linear_speed=0.10, duration=0.5,
                                                  source_frame_stamp_ns=stamp)
    assert first["motion_executed"] is True
    duration = behavior.execute_single_marvin_approach_step.call_args.kwargs["duration"]
    assert duration == pytest.approx(max_duration)
    assert duration <= max_duration + 1e-12
    assert distance - 0.10 * duration >= 0.50
    duplicate = runtime.execute_single_marvin_approach(linear_speed=0.10, duration=0.5,
                                                      source_frame_stamp_ns=stamp)
    assert duplicate["reason"] == "marvin_approach_observation_already_consumed"
    behavior.preview = fresh_preview(stamp=102)
    assert runtime.observe_find_marvin_v2()["controller"]["decision"] == "BLOCKED"
    runtime.world_model.get_lidar_obstacles.return_value = lidar_at(distance, sequence=2)
    newer = runtime.observe_find_marvin_v2()
    assert newer["controller"]["decision"] == "FORWARD"
    assert newer["opencv_tracker"]["source_frame_stamp_ns"] > stamp
    assert behavior.execute_single_marvin_approach_step.call_count == 1


@pytest.mark.parametrize("distance", [0.50, 0.49, 0.45])
def test_jit_distance_at_standoff_vetoes_cached_forward_authorization(monkeypatch, distance):
    monkeypatch.setattr("runtime.time.time_ns", lambda: 1000)
    runtime, behavior = runtime_at(1.0)
    observation = runtime.observe_find_marvin_v2()
    runtime.world_model.get_lidar_obstacles.return_value = lidar_at(distance)
    result = runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.5,
        source_frame_stamp_ns=observation["opencv_tracker"]["source_frame_stamp_ns"],
    )
    assert result["motion_executed"] is False
    assert result["target_standoff"]["hard_safety_condition"] is (distance <= 0.45)
    behavior.execute_single_marvin_approach_step.assert_not_called()


def test_turn_also_requires_new_lidar_evidence_before_later_approach(monkeypatch):
    from test_marvin_alignment_step import runtime as alignment_runtime, SESSION as turn_session, STAMP
    monkeypatch.setattr("runtime.time.time_ns", lambda: STAMP + 10)
    runtime, _ = alignment_runtime()
    runtime.marvin_camera_model = dict(CAMERA)
    runtime.world_model.lidar = dict(lidar_at(1.0), producer_session=turn_session)
    result = runtime.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.5, source_frame_stamp_ns=STAMP,
    )
    assert result["motion_executed"] is True
    tracker = _v2_preview(0.0, stamp=102)["opencv_tracker"]
    assert runtime._marvin_v2_lidar_arrival(tracker)["reason"] == "target_lidar_newer_observation_required"
    runtime.world_model.lidar["acquisition_sequence"] = 2
    assert runtime._marvin_v2_lidar_arrival(tracker)["ok"] is True


def test_other_sector_obstacle_still_vetoes_actual_guarded_forward():
    tracker = _v2_preview(0.0)["opencv_tracker"]
    lidar = lidar_at(1.0)
    geometry = geometry_with_point(0.30, 0.25)
    geometry["points"].extend(lidar["local_motion_geometry"]["points"])
    lidar["local_motion_geometry"] = geometry
    assert evaluate_marvin_lidar_standoff(tracker, lidar, CAMERA,
                                         expected_session=SESSION)["target_distance_m"] == 1.0
    assert evaluate_local_motion_safety(lidar, expected_session=SESSION,
                                       linear_x=0.10, duration=0.5)["permitted"] is False
    world = Mock()
    world.get_lidar_obstacles.return_value = lidar
    robot = Mock()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.5,
        target_tracker=tracker, camera_model=CAMERA,
    )
    assert result["motion_executed"] is False
    robot.move_forward.assert_not_called()


@pytest.mark.parametrize("distance,max_duration", [(0.53, 0.30), (0.51, 0.10)])
def test_behavior_rechecks_standoff_and_shrinks_bound_on_same_jit_safety_snapshot(
        monkeypatch, distance, max_duration):
    import behavior_manager as module
    world = Mock()
    world.get_lidar_obstacles.return_value = lidar_at(distance)
    robot = Mock()
    robot.move_forward.return_value = {"ok": True, "executed": True}
    manager = BehaviorManager(robot_client=robot, world_model=world)
    calls = []
    monkeypatch.setattr(module, "evaluate_local_motion_safety", lambda *_a, **kw:
                        calls.append(kw) or {"permitted": True, "geometry": {"valid": True}})
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.5,
        target_tracker=_v2_preview(0.0)["opencv_tracker"], camera_model=CAMERA,
    )
    assert result["motion_executed"] is True
    assert calls[0]["linear_x"] == robot.move_forward.call_args.kwargs["speed"] == 0.10
    duration = robot.move_forward.call_args.kwargs["seconds"]
    assert calls[0]["duration"] == duration == pytest.approx(max_duration)
    assert duration <= max_duration + 1e-12
    assert distance - 0.10 * duration >= 0.50
    assert world.get_lidar_obstacles.call_count == 1


@pytest.mark.parametrize("distance", [0.50, 0.49, 0.45])
def test_behavior_at_standoff_sends_no_forward_command(distance):
    world, robot = Mock(), Mock()
    world.get_lidar_obstacles.return_value = lidar_at(distance)
    manager = BehaviorManager(robot_client=robot, world_model=world)
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.50,
        target_tracker=_v2_preview(0.0)["opencv_tracker"], camera_model=CAMERA)
    assert result["motion_executed"] is False
    robot.move_forward.assert_not_called()
