"""Offline contracts for the one-shot, no-avoidance Marvin forward boundary."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


SESSION = "active-runtime-forward-session"


def lidar():
    return {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": SESSION,
        "local_motion_geometry": {"valid": True},
    }


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
        expected_lidar_session=SESSION, linear_speed=0.08, duration=0.50)
    assert result["ok"] is result["motion_executed"] is True
    assert robot.forward_calls == 1 and world.calls == [SESSION] and len(calls) == 1
    assert robot.forward_requests == [(0.08, 0.50)]
    assert calls[0][1]["linear_x"] == 0.08 and calls[0][1]["duration"] == 0.50
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
            expected_lidar_session=SESSION, linear_speed=0.08, duration=0.50)
        assert result["motion_executed"] is False and robot.forward_calls == 0
        manager.execute_local_obstacle_avoidance_step.assert_not_called()


def test_one_shot_accepts_only_complete_canonical_bounded_bridge_success(monkeypatch):
    canonical = {
        "ok": True, "action": "motion", "mode": "bounded",
        "linear_x": 0.08, "angular_z": 0.0, "duration": 0.50,
        "automatic_stop": True, "returned_immediately": False,
    }
    robot, world = Robot(canonical), World()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    monkeypatch.setattr(behavior_manager_module, "evaluate_local_motion_safety", lambda *_a, **_k: safety())
    result = manager.execute_single_marvin_approach_step(
        expected_lidar_session=SESSION, linear_speed=0.08, duration=0.50)
    assert result["ok"] is result["motion_executed"] is True
    assert result["forward_result"]["executed"] is True
    assert robot.forward_requests == [(0.08, 0.50)]


class RuntimeBehavior:
    def __init__(self, robot):
        self.robot = robot
        self.calls = []
        self.execute_find_marvin_controller = Mock(side_effect=AssertionError("controller forbidden"))
        self.execute_marvin_search_step = Mock(side_effect=AssertionError("search forbidden"))
        self.execute_marvin_pursuit_step = Mock(side_effect=AssertionError("pursuit forbidden"))
        self.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
        self.target_lock = SimpleNamespace(resolve=Mock(side_effect=AssertionError("TargetLock forbidden")))

    def execute_single_marvin_approach_step(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "motion_executed": True, "forward_safety": safety()}


def active_runtime(*, lidar_value=None):
    runtime = object.__new__(CognitiveRuntime)
    runtime.running = True
    runtime._state_lock = threading.RLock()
    runtime._marvin_approach_step_consumed = False
    robot = Robot()
    runtime.behavior_manager = RuntimeBehavior(robot)
    runtime.world_model = World(lidar_value)
    runtime.lidar_worker = SimpleNamespace(session=SESSION, running=True)
    return runtime, robot


def test_runtime_uses_active_session_once_then_explicitly_stops():
    runtime, robot = active_runtime()
    result = runtime.execute_single_marvin_approach(linear_speed=0.08, duration=0.50)
    assert result["ok"] is result["motion_executed"] is True
    assert result["actions_executed"] == 1 and robot.stop_calls == 1
    assert runtime.world_model.calls == [SESSION]
    assert runtime.behavior_manager.calls == [{"expected_lidar_session": SESSION, "linear_speed": 0.08, "duration": 0.5}]
    second = runtime.execute_single_marvin_approach(linear_speed=0.08, duration=0.50)
    assert second["reason"] == "marvin_approach_step_already_consumed"
    assert len(runtime.behavior_manager.calls) == 1


def test_runtime_rejects_noncalibrated_or_bad_lidar_without_forward():
    for kwargs in ({"linear_speed": 0.081, "duration": 0.50},
                   {"linear_speed": 0.08, "duration": 0.501},
                   {"linear_speed": 0.04, "duration": 0.50}):
        runtime, robot = active_runtime()
        result = runtime.execute_single_marvin_approach(**kwargs)
        assert result["motion_executed"] is False and robot.stop_calls == 0
        assert runtime.behavior_manager.calls == []
    runtime, robot = active_runtime(lidar_value={**lidar(), "reason": "stale"})
    assert runtime.execute_single_marvin_approach(linear_speed=0.08, duration=0.50)["motion_executed"] is False
    assert runtime.behavior_manager.calls == [] and robot.stop_calls == 0


def test_endpoint_requires_exact_schema_and_calls_only_approach_method():
    runtime = SimpleNamespace(execute_single_marvin_approach=Mock(return_value={"ok": True}))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/approach-step"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={"linear_speed": 0.08, "duration": 0.50})
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    runtime.execute_single_marvin_approach.assert_called_once_with(linear_speed=0.08, duration=0.50)
    assert responses == [(200, {"ok": True})]
    handler.require_json_request = Mock(return_value={"linear_speed": 0.08, "duration": 0.50, "extra": True})
    handler.do_POST()
    assert runtime.execute_single_marvin_approach.call_count == 1 and responses[-1][0] == 400
