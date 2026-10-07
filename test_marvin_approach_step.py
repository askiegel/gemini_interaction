"""Offline contracts for the one-shot, no-avoidance Marvin forward boundary."""

import threading
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager, FIND_MARVIN_FORWARD_SPEED_MPS
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


SESSION = "active-runtime-forward-session"
STAMP = 1791156612322562413
PRECISE_STAMP = 1791156612322562413


@pytest.fixture(autouse=True)
def camera_clock(monkeypatch):
    monkeypatch.setattr("runtime.time.time_ns", lambda: STAMP + 10)
    monkeypatch.setattr("runtime.time.monotonic", lambda: 100.0)


def lidar():
    from test_marvin_lidar_standoff import lidar_at
    return dict(lidar_at(1.0), producer_session=SESSION)


class Robot:
    def __init__(self, result=None):
        self.forward_calls = 0
        self.forward_requests = []
        self.stop_calls = 0
        self.result = result if result is not None else {"ok": True, "executed": True}

    def move_forward(self, *, speed, seconds):
        self.forward_calls += 1
        self.forward_requests.append((speed, seconds))
        return self.result

    def stop(self):
        self.stop_calls += 1
        return {"ok": True}


class World:
    def __init__(self, value=None):
        self.value = lidar() if value is None else value
        self.calls = []
        self.update_entity = Mock(side_effect=AssertionError("World Model write forbidden"))

    def get_lidar_obstacles(self, *, expected_session, **_kwargs):
        self.calls.append(expected_session)
        if isinstance(self.value, dict):
            import time
            self.value["received_monotonic_seconds"] = time.monotonic()
            self.value["acquisition_sequence"] = len(self.calls)
        return self.value


def safety(permitted=True):
    return {"permitted": permitted, "reason": "protected_region_clear" if permitted else "translation_protected_region_violated", "geometry": {"valid": True}}


def test_behavior_uses_existing_lidar_safety_then_one_forward_without_avoidance(monkeypatch):
    robot, world = Robot(), World()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    calls = []
    monkeypatch.setattr(behavior_manager_module, "evaluate_local_motion_safety",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or safety())
    manager.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.50)
    assert result["ok"] is result["motion_executed"] is True
    assert robot.forward_calls == 1 and world.calls == [SESSION] and len(calls) == 1
    assert robot.forward_requests == [(0.10, 0.50)]
    assert calls[0][1]["linear_x"] == 0.10 and calls[0][1]["duration"] == 0.50
    assert robot.forward_requests[0][0] * robot.forward_requests[0][1] == pytest.approx(0.050)
    manager.execute_local_obstacle_avoidance_step.assert_not_called()


def test_behavior_blocked_or_untrusted_lidar_vetoes_without_forward_or_avoidance(monkeypatch):
    for value, checked_safety in ((lidar(), safety(False)),
                                  ({**lidar(), "valid": False}, safety(False))):
        robot, world = Robot(), World(value)
        manager = BehaviorManager(robot_client=robot, world_model=world)
        monkeypatch.setattr(behavior_manager_module, "evaluate_local_motion_safety",
                            lambda *args, _safety=checked_safety, **kwargs: _safety)
        manager.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
        result = manager.execute_single_marvin_approach_step(
            expected_lidar_session=SESSION, linear_speed=0.10, duration=0.50)
        assert result["motion_executed"] is False and robot.forward_calls == 0
        manager.execute_local_obstacle_avoidance_step.assert_not_called()


def test_precise_translation_veto_prevents_transport_even_if_front_is_clear(monkeypatch):
    robot, world = Robot(), World()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    clear_front_diagnostic = {
        **lidar(),
        "sectors": {"front": {"available": True, "state": "CLEAR"}},
    }
    world.value = clear_front_diagnostic
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_local_motion_safety",
        lambda *_args, **_kwargs: safety(False),
    )

    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.50,
    )

    assert result["reason"] == "marvin_single_approach_translation_vetoed"
    assert robot.forward_calls == 0


def test_one_shot_accepts_only_complete_canonical_bounded_bridge_success(monkeypatch):
    canonical = {
        "ok": True, "action": "motion", "mode": "bounded",
        "linear_x": 0.10, "angular_z": 0.0, "duration": 0.50,
        "automatic_stop": True, "returned_immediately": False,
    }
    robot, world = Robot(canonical), World()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    monkeypatch.setattr(behavior_manager_module, "evaluate_local_motion_safety", lambda *_a, **_k: safety())
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.10, duration=0.50)
    assert result["ok"] is result["motion_executed"] is True
    assert result["forward_result"]["executed"] is True
    assert robot.forward_requests == [(0.10, 0.50)]


class RuntimeBehavior:
    def __init__(self, robot):
        self.robot = robot
        self.calls = []
        self.execute_find_marvin_controller = Mock(side_effect=AssertionError("controller forbidden"))
        self.execute_marvin_search_step = Mock(side_effect=AssertionError("search forbidden"))
        self.execute_marvin_pursuit_step = Mock(side_effect=AssertionError("pursuit forbidden"))
        self.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
        self.target_lock = SimpleNamespace(resolve=Mock(side_effect=AssertionError("TargetLock forbidden")))
        self.post_action_marks = []

    def execute_single_marvin_approach_step(self, **kwargs):
        assert kwargs["dispatch_guard"]() is True
        kwargs.pop("dispatch_guard")
        validator = kwargs.pop("target_range_validator")
        assert validator(kwargs["target_tracker"], self.current_lidar)["target_range_association_trusted"]
        self.calls.append(kwargs)
        return {"ok": True, "motion_executed": True, "forward_safety": safety()}

    def mark_strict_v2_action_dispatched(self, source_frame_stamp_ns, action):
        self.post_action_marks.append((source_frame_stamp_ns, action))
        return True


def active_runtime(*, lidar_value=None):
    runtime = object.__new__(CognitiveRuntime)
    runtime.running = True
    runtime._state_lock = threading.RLock()
    runtime._marvin_alignment_consumed_source_frame_stamps = set()
    runtime.marvin_camera_model = {
        "fx_pixels": 320.0, "cx_pixels": 320.0, "image_width": 640,
        "x_m": 0.0, "y_m": 0.0, "yaw_degrees": 0.0, "range_uncertainty_m": 0.0,
    }
    runtime._marvin_alignment_observation = {
        "source_frame_stamp_ns": STAMP,
        "received_monotonic_seconds": 100.0,
        "identity_confirmed": True,
        "identity_source": "gemini_marvin_candidate_selection",
        "opencv_tracker": {
            "active": True, "matched": True, "quality": 0.97,
            "threshold": 0.80, "source_frame_stamp_ns": STAMP,
            "received_monotonic_seconds": 100.0,
            "bbox": {"x1": 280, "y1": 1, "x2": 360, "y2": 300},
            "image_width": 640, "image_height": 480,
        },
        "controller_state": "VISUAL_READY_TO_APPROACH",
        "controller_decision": "FORWARD",
        "target_standoff": {"ok": True, "authority": "target_bearing_lidar",
                            "arrived_at_marvin": False, "target_distance_m": 1.0,
                            "target_range_association_trusted": True},
    }
    robot = Robot()
    runtime.behavior_manager = RuntimeBehavior(robot)
    runtime.world_model = World(lidar_value)
    runtime.lidar_worker = SimpleNamespace(session=SESSION, running=True)
    runtime.behavior_manager.current_lidar = runtime.world_model.value
    return runtime, robot


def test_runtime_uses_current_observation_once_then_explicitly_stops():
    runtime, robot = active_runtime()
    result = runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert result["ok"] is result["motion_executed"] is True
    assert result["actions_executed"] == 1 and robot.stop_calls == 1
    assert runtime.world_model.calls == [SESSION]
    assert runtime.behavior_manager.calls == [{
        "expected_lidar_session": SESSION, "linear_speed": 0.10, "duration": 0.5,
        "target_tracker": runtime._marvin_alignment_observation["opencv_tracker"],
        "camera_model": runtime.marvin_camera_model,
    }]
    assert runtime.behavior_manager.post_action_marks == [(STAMP, "forward")]
    second = runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert second["reason"] == "marvin_approach_observation_already_consumed"
    assert len(runtime.behavior_manager.calls) == 1


def test_runtime_requires_new_source_stamp_before_next_guarded_forward():
    runtime, robot = active_runtime()
    first = runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    runtime._marvin_alignment_observation = {
        **runtime._marvin_alignment_observation,
        "source_frame_stamp_ns": STAMP + 1,
        "opencv_tracker": {
            **runtime._marvin_alignment_observation["opencv_tracker"],
            "source_frame_stamp_ns": STAMP + 1,
        },
    }
    second = runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP + 1,
    )
    assert first["motion_executed"] is second["motion_executed"] is True
    assert len(runtime.behavior_manager.calls) == 2
    assert robot.stop_calls == 2


@pytest.mark.parametrize("age_ns,permitted", [
    (0, True), (1_000_000_000, True), (1_000_000_001, False), (-1, False),
])
def test_forward_enforces_exact_one_second_camera_age(monkeypatch, age_ns, permitted):
    runtime, _ = active_runtime()
    monkeypatch.setattr("runtime.time.monotonic", lambda: 100.0 + age_ns / 1e9)
    result = runtime.execute_single_marvin_approach(
        linear_speed=.10, duration=.5, source_frame_stamp_ns=STAMP)
    assert result["motion_executed"] is permitted
    assert len(runtime.behavior_manager.calls) == int(permitted)
    if not permitted:
        assert result["reason"] == "marvin_motion_observation_stale_or_preempted"
        assert runtime._marvin_alignment_observation is None


def test_runtime_rejects_noncalibrated_or_bad_lidar_without_forward():
    for kwargs in ({"linear_speed": 0.101, "duration": 0.50},
                   {"linear_speed": 0.10, "duration": 0.501},
                   {"linear_speed": 0.04, "duration": 0.50}):
        runtime, robot = active_runtime()
        result = runtime.execute_single_marvin_approach(
            **kwargs, source_frame_stamp_ns=STAMP,
        )
        assert result["motion_executed"] is False and robot.stop_calls == 0
        assert runtime.behavior_manager.calls == []
    runtime, robot = active_runtime(lidar_value={**lidar(), "reason": "stale"})
    assert runtime.execute_single_marvin_approach(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP,
    )["motion_executed"] is False
    assert runtime.behavior_manager.calls == [] and robot.stop_calls == 0


@pytest.mark.parametrize("speed,duration", [
    (0.08, 0.50), (0.101, 0.50), (0.10, 0.501),
])
@pytest.mark.parametrize("layer", ["runtime", "behavior"])
def test_exact_forward_contract_rejects_old_speed_excess_speed_and_excess_duration(
        speed, duration, layer):
    if layer == "runtime":
        runtime, robot = active_runtime()
        result = runtime.execute_single_marvin_approach(
            linear_speed=speed, duration=duration, source_frame_stamp_ns=STAMP)
        assert runtime.behavior_manager.calls == []
    else:
        robot, world = Robot(), World()
        manager = BehaviorManager(robot_client=robot, world_model=world)
        result = manager.execute_single_marvin_approach_step(
            expected_lidar_session=SESSION, linear_speed=speed, duration=duration)
        assert world.calls == []
    assert result["motion_executed"] is False
    assert robot.forward_calls == robot.stop_calls == 0


def test_forward_speed_change_preserves_safety_and_recovery_limits():
    from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
    from robot_bridge.forward_interlock import MAXIMUM_EFFECTIVE_AGE_SECONDS as interlock_age
    from local_motion_safety_envelope import LOCAL_LIDAR_PROTECTED_RADIUS_M
    from marvin_lidar_standoff import TARGET_STANDOFF_M
    from marvin_local_tracker import MarvinLocalTracker

    assert FIND_MARVIN_FORWARD_SPEED_MPS == BehaviorManager.FIND_FORWARD_SPEED == 0.10
    assert BehaviorManager.FIND_APPROACH_FORWARD_SPEED == 0.10
    assert BehaviorManager.FIND_APPROACH_FORWARD_SECONDS == 0.50
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == interlock_age == 0.30
    assert CognitiveRuntime.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS == 0.60
    assert CognitiveRuntime.MARVIN_NEW_LIDAR_POLL_SECONDS == 0.05
    assert CognitiveRuntime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0
    assert MarvinLocalTracker.MIN_MATCH_QUALITY == 0.80
    assert LOCAL_LIDAR_PROTECTED_RADIUS_M == 0.45
    assert TARGET_STANDOFF_M == 0.50


def test_endpoint_requires_exact_schema_and_calls_only_approach_method():
    runtime = SimpleNamespace(execute_single_marvin_approach=Mock(return_value={"ok": True}))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/approach-step"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={
        "linear_speed": 0.10, "duration": 0.50,
        "source_frame_stamp_ns": STAMP,
    })
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    runtime.execute_single_marvin_approach.assert_called_once_with(
        linear_speed=0.10, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert responses == [(200, {"ok": True})]
    handler.require_json_request = Mock(return_value={
        "linear_speed": 0.10, "duration": 0.50,
        "source_frame_stamp_ns": STAMP, "extra": True,
    })
    handler.do_POST()
    assert runtime.execute_single_marvin_approach.call_count == 1 and responses[-1][0] == 400


def _post_approach(runtime_value, stamp_value):
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/approach-step"
    handler.server = SimpleNamespace(runtime=runtime_value)
    handler.require_json_request = Mock(return_value={
        "linear_speed": 0.10, "duration": 0.50,
        "source_frame_stamp_ns": stamp_value,
    })
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    return responses[-1]


def test_approach_api_exact_string_stamp_dispatches_once_and_duplicate_is_rejected():
    runtime, robot = active_runtime()
    runtime._marvin_alignment_observation["source_frame_stamp_ns"] = PRECISE_STAMP
    runtime._marvin_alignment_observation["opencv_tracker"][
        "source_frame_stamp_ns"
    ] = PRECISE_STAMP

    first_status, first = _post_approach(runtime, str(PRECISE_STAMP))
    duplicate_status, duplicate = _post_approach(runtime, str(PRECISE_STAMP))

    assert first_status == 200 and first["motion_executed"] is True
    assert first["source_frame_stamp_ns"] == PRECISE_STAMP
    assert duplicate_status == 409
    assert duplicate["reason"] == "marvin_approach_observation_already_consumed"
    assert duplicate["actions_executed"] == 0
    assert len(runtime.behavior_manager.calls) == 1 and robot.stop_calls == 1


def test_approach_api_rounded_nearby_string_is_not_authorized():
    runtime, robot = active_runtime()
    runtime._marvin_alignment_observation["source_frame_stamp_ns"] = PRECISE_STAMP
    runtime._marvin_alignment_observation["opencv_tracker"][
        "source_frame_stamp_ns"
    ] = PRECISE_STAMP

    status, result = _post_approach(runtime, str(PRECISE_STAMP - 113))

    assert status == 409
    assert result["reason"] == "marvin_approach_observation_not_authorized"
    assert runtime.behavior_manager.calls == [] and robot.stop_calls == 0


def test_approach_api_rejects_inexact_or_malformed_stamp_inputs():
    invalid_values = (
        1.0, "1.0", "1e3", "1E3", "", "-1", -1, True, False,
        None, [], {}, " 1", "+1",
    )
    runtime = SimpleNamespace(
        execute_single_marvin_approach=Mock(return_value={"ok": True})
    )

    for invalid in invalid_values:
        status, result = _post_approach(runtime, invalid)
        assert status == 400
        assert result["ok"] is False

    runtime.execute_single_marvin_approach.assert_not_called()
