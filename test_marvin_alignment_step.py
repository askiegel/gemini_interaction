"""Offline contracts for the one-shot Marvin guarded alignment endpoint."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler
from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT


SESSION = "active-runtime-lidar-session"
STAMP = 101


class World:
    def __init__(self, lidar=None):
        self.lidar = lidar if lidar is not None else {
            "available": True, "valid": True, "reason": "fresh",
            "producer_session": SESSION,
            "local_motion_geometry": {"valid": True},
        }
        self.calls = []
        self.update_entity = Mock(side_effect=AssertionError("World Model identity write forbidden"))

    def get_lidar_obstacles(self, *, expected_session):
        self.calls.append(expected_session)
        return self.lidar


class Robot:
    def __init__(self):
        self.stop_calls = 0
        self.local_forward = Mock(side_effect=AssertionError("forward forbidden"))
        self.motion = Mock(side_effect=AssertionError("alignment must use guarded turn"))

    def stop(self):
        self.stop_calls += 1
        return {"ok": True, "stopped": True}


class Behavior:
    def __init__(self, robot):
        self.robot = robot
        self.calls = []
        self.execute_find_marvin_controller = Mock(side_effect=AssertionError("controller forbidden"))
        self.execute_marvin_search_step = Mock(side_effect=AssertionError("search forbidden"))
        self.execute_marvin_pursuit_step = Mock(side_effect=AssertionError("pursuit forbidden"))
        self.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
        self.target_lock = SimpleNamespace(resolve=Mock(side_effect=AssertionError("TargetLock mutation forbidden")))

    def _execute_target_directed_turn(self, direction, speed, duration, *, expected_lidar_session,
                                      safety_mode):
        self.calls.append((direction, speed, duration, expected_lidar_session, safety_mode))
        return {"ok": True, "permitted": True, "confirmed_forwarded": True}


def runtime(*, lidar=None):
    value = object.__new__(CognitiveRuntime)
    value.running = True
    value._state_lock = threading.RLock()
    value._marvin_alignment_consumed_source_frame_stamps = set()
    value._marvin_alignment_observation = strict_observation(STAMP, "TURN_RIGHT")
    robot = Robot()
    value.behavior_manager = Behavior(robot)
    value.world_model = World(lidar)
    value.lidar_worker = SimpleNamespace(session=SESSION, running=True)
    return value, robot


def strict_observation(stamp, decision):
    return {
        "source_frame_stamp_ns": stamp,
        "identity_confirmed": True,
        "identity_source": "gemini_marvin_candidate_selection",
        "opencv_tracker": {
            "active": True, "matched": True, "quality": 0.97,
            "threshold": 0.80,
            "bbox": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
            "source_frame_stamp_ns": stamp,
        },
        "controller_state": "VISUAL_READY_TO_ALIGN",
        "controller_decision": decision,
    }


def test_valid_right_uses_active_runtime_lidar_one_turn_and_explicit_stop():
    value, robot = runtime()
    result = value.execute_single_marvin_alignment(
        direction="right", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert result["ok"] is result["motion_executed"] is True
    assert result["actions_executed"] == 1
    assert value.world_model.calls == [SESSION]
    assert value.behavior_manager.calls == [("RIGHT", 0.25, 0.50, SESSION, ROTATIONAL_SWEPT_FOOTPRINT)]
    assert robot.stop_calls == 1
    assert robot.local_forward.call_count == 0
    for forbidden in (
        value.behavior_manager.execute_find_marvin_controller,
        value.behavior_manager.execute_marvin_search_step,
        value.behavior_manager.execute_marvin_pursuit_step,
        value.behavior_manager.execute_local_obstacle_avoidance_step,
        value.behavior_manager.target_lock.resolve,
        value.world_model.update_entity,
    ):
        assert forbidden.call_count == 0


def test_valid_left_is_the_only_other_allowed_direction():
    value, robot = runtime()
    value._marvin_alignment_observation = strict_observation(STAMP, "TURN_LEFT")
    result = value.execute_single_marvin_alignment(
        direction="LEFT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert result["ok"] is True
    assert value.behavior_manager.calls == [("LEFT", 0.25, 0.50, SESSION, ROTATIONAL_SWEPT_FOOTPRINT)]
    assert robot.stop_calls == 1


def test_endpoint_runtime_method_cannot_chain_a_second_turn():
    value, robot = runtime()
    first = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    second = value.execute_single_marvin_alignment(
        direction="LEFT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert first["ok"] is True
    assert second["reason"] == "marvin_alignment_observation_already_consumed"
    assert value.behavior_manager.calls == [("RIGHT", 0.25, 0.50, SESSION, ROTATIONAL_SWEPT_FOOTPRINT)]
    assert robot.stop_calls == 1


def test_newer_strict_observation_authorizes_one_new_alignment_without_restart():
    value, robot = runtime()
    first = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP,
    )
    value._marvin_alignment_observation = strict_observation(STAMP + 1, "TURN_RIGHT")
    second = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP + 1,
    )
    assert first["motion_executed"] is second["motion_executed"] is True
    assert len(value.behavior_manager.calls) == 2
    assert robot.stop_calls == 2


def test_alignment_rejects_direction_mismatch_and_non_strict_observations():
    value, robot = runtime()
    value._marvin_alignment_observation = strict_observation(STAMP, "TURN_LEFT")
    mismatch = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP,
    )
    assert mismatch["reason"] == "marvin_alignment_observation_not_authorized"
    value._marvin_alignment_observation = strict_observation(STAMP + 1, "TURN_RIGHT")
    value._marvin_alignment_observation["identity_source"] = "marvin_session_continuity"
    continuity = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP + 1,
    )
    assert continuity["reason"] == "marvin_alignment_observation_not_authorized"
    value._marvin_alignment_observation = strict_observation(STAMP + 2, "TURN_RIGHT")
    value._marvin_alignment_observation["opencv_tracker"]["quality"] = 0.79
    tracker = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP + 2,
    )
    assert tracker["reason"] == "marvin_alignment_observation_not_authorized"
    assert value.behavior_manager.calls == [] and robot.motion.call_count == 0


def test_invalid_direction_or_excessive_limits_never_read_lidar_or_turn():
    for kwargs in (
        {"direction": "forward", "angular_speed": 0.25, "duration": 0.50},
        {"direction": "RIGHT", "angular_speed": 0.251, "duration": 0.50},
        {"direction": "RIGHT", "angular_speed": 0.25, "duration": 0.501},
        {"direction": "RIGHT", "angular_speed": 0.0, "duration": 0.50},
        {"direction": "RIGHT", "angular_speed": 0.25, "duration": 0.0},
    ):
        value, robot = runtime()
        result = value.execute_single_marvin_alignment(**kwargs, source_frame_stamp_ns=STAMP)
        assert result["ok"] is result["motion_executed"] is False
        assert result["actions_executed"] == 0
        assert value.world_model.calls == value.behavior_manager.calls == []
        assert robot.stop_calls == 0


def test_missing_stale_or_invalid_runtime_lidar_vetoes_without_turn():
    cases = (
        None,
        {"available": True, "valid": True, "reason": "stale", "producer_session": SESSION, "local_motion_geometry": {"valid": True}},
        {"available": True, "valid": False, "reason": "fresh", "producer_session": SESSION, "local_motion_geometry": {"valid": True}},
        {"available": True, "valid": True, "reason": "fresh", "producer_session": SESSION, "local_motion_geometry": {"valid": False}},
    )
    for lidar in cases:
        value, robot = runtime()
        value.world_model.lidar = lidar
        result = value.execute_single_marvin_alignment(
            direction="RIGHT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
        )
        assert result["reason"] == "marvin_alignment_lidar_not_current"
        assert value.behavior_manager.calls == [] and robot.stop_calls == 0


def test_rotational_safety_veto_never_sends_bridge_motion():
    value, robot = runtime()
    value.behavior_manager._execute_target_directed_turn = Mock(return_value={
        "ok": False, "permitted": False, "confirmed_forwarded": False,
        "reason": "rotational_protected_region_violated",
        "rotational_swept_footprint": {
            "protected_radius_m": 0.45,
            "reason": "rotational_protected_region_violated",
        },
    })
    result = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert result["motion_executed"] is False
    retry = value.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50, source_frame_stamp_ns=STAMP,
    )
    assert retry["reason"] == "marvin_alignment_observation_already_consumed"
    value.behavior_manager._execute_target_directed_turn.assert_called_once_with(
        "RIGHT", 0.25, 0.50, expected_lidar_session=SESSION,
        safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
    )
    assert robot.motion.call_count == 0


def test_endpoint_requires_exact_json_and_delegates_only_to_runtime_method():
    runtime = SimpleNamespace(execute_single_marvin_alignment=Mock(return_value={"ok": True}))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/alignment-step"
    handler.server = SimpleNamespace(runtime=runtime)
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.require_json_request = Mock(return_value={
        "direction": "right", "angular_speed": 0.25, "duration": 0.50,
        "source_frame_stamp_ns": STAMP,
    })
    handler.do_POST()
    assert responses == [(200, {"ok": True})]
    runtime.execute_single_marvin_alignment.assert_called_once_with(
        direction="right", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=STAMP,
    )
    handler.require_json_request = Mock(return_value={
        "direction": "right", "angular_speed": 0.25, "duration": 0.50,
        "source_frame_stamp_ns": STAMP, "extra": True,
    })
    handler.do_POST()
    assert runtime.execute_single_marvin_alignment.call_count == 1
    assert responses[-1][0] == 400
