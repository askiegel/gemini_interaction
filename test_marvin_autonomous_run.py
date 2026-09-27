"""Offline contracts for the capped autonomous Marvin runtime boundary."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


SESSION = "active-lidar-session"


class Robot:
    def __init__(self):
        self.stop = Mock(return_value={"ok": True, "stopped": True})
        self.local_forward = Mock(side_effect=AssertionError("raw forward forbidden"))


class Behavior:
    def __init__(self, robot):
        self.robot = robot
        self.target_lock = SimpleNamespace(resolve=Mock(side_effect=AssertionError("TargetLock mutation forbidden")))
        self.execute_single_marvin_approach_step = Mock(side_effect=AssertionError("one-shot endpoint forbidden"))
        self.execute_single_marvin_alignment = Mock(side_effect=AssertionError("one-shot endpoint forbidden"))
        self.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance is controller-owned"))
        self.calls = []

    def execute_find_marvin_controller(self, provider, *, max_actions, dry_run, stop_after_action):
        self.calls.append((provider, max_actions, dry_run, stop_after_action))
        history = []
        for _ in range(max_actions):
            assert stop_after_action().get("ok") is True
            history.append({
                "pursuit_step_result": {"motion_executed": True},
                "stop_result": {"ok": True},
            })
        return {
            "ok": True,
            "actions_executed": max_actions,
            "history": history,
            "reason": "find_marvin_action_limit_reached",
        }


def active_runtime():
    value = object.__new__(CognitiveRuntime)
    value.running = True
    value._state_lock = threading.RLock()
    value._marvin_autonomous_run_consumed = False
    robot = Robot()
    value.behavior_manager = Behavior(robot)
    value.lidar_worker = SimpleNamespace(session=SESSION, running=True)
    value.world_model = SimpleNamespace(update_entity=Mock(side_effect=AssertionError("World Model write forbidden")))
    return value, robot


def test_capped_autonomous_run_uses_active_controller_and_stops_each_action():
    runtime, robot = active_runtime()
    result = runtime.execute_bounded_find_marvin_autonomous(max_actions=3)
    assert result["ok"] is result["motion_executed"] is True
    assert result["execution_authorized"] is True and result["actions_executed"] == 3
    assert len(runtime.behavior_manager.calls) == 1
    _provider, limit, dry_run, _stop = runtime.behavior_manager.calls[0]
    assert limit == 3 and dry_run is False and robot.stop.call_count == 3
    assert runtime.behavior_manager.target_lock.resolve.call_count == 0
    assert runtime.world_model.update_entity.call_count == 0
    assert robot.local_forward.call_count == 0


def test_cap_and_one_shot_guard_prevent_unbounded_or_repeated_runs():
    runtime, robot = active_runtime()
    assert runtime.execute_bounded_find_marvin_autonomous(max_actions=4)["reason"] == "marvin_autonomous_action_limit_invalid"
    assert runtime.behavior_manager.calls == [] and robot.stop.call_count == 0
    assert runtime.execute_bounded_find_marvin_autonomous(max_actions=3)["ok"] is True
    second = runtime.execute_bounded_find_marvin_autonomous(max_actions=1)
    assert second["reason"] == "marvin_autonomous_run_already_consumed"
    assert len(runtime.behavior_manager.calls) == 1


def test_runtime_refuses_missing_active_lidar_session_without_controller():
    runtime, robot = active_runtime()
    runtime.lidar_worker = SimpleNamespace(session=None, running=True)
    result = runtime.execute_bounded_find_marvin_autonomous(max_actions=3)
    assert result["reason"] == "marvin_autonomous_lidar_session_unavailable"
    assert runtime.behavior_manager.calls == [] and robot.stop.call_count == 0


def test_endpoint_requires_exact_schema_and_only_delegates_to_bounded_runtime_method():
    runtime = SimpleNamespace(execute_bounded_find_marvin_autonomous=Mock(return_value={"ok": True}))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/autonomous-run"
    handler.server = SimpleNamespace(runtime=runtime)
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.require_json_request = Mock(return_value={"max_actions": 3})
    handler.do_POST()
    runtime.execute_bounded_find_marvin_autonomous.assert_called_once_with(max_actions=3)
    assert responses == [(200, {"ok": True})]
    handler.require_json_request = Mock(return_value={"max_actions": 3, "extra": True})
    handler.do_POST()
    assert runtime.execute_bounded_find_marvin_autonomous.call_count == 1
    assert responses[-1][0] == 400
