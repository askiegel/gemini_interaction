"""Offline tests for one-action local reactive orchestration."""

import threading
from types import SimpleNamespace

import pytest

import runtime as runtime_module
from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime


SESSION = "local-reactive-session"


def _decision(value):
    return {
        "decision": value,
        "reason": "test_decision",
        "lidar_fresh": True,
        "geometry_valid": True,
        "forward": {"permitted": value == "FORWARD_CLEAR"},
        "left": {"permitted": value == "TURN_LEFT"},
        "right": {"permitted": value == "TURN_RIGHT"},
    }


class _MissionManager:
    def __init__(self, active=None):
        self.active = active

    def get_active_mission(self):
        return self.active


class _World:
    def __init__(self):
        self.calls = 0

    def get_lidar_obstacles(self, *, expected_session):
        self.calls += 1
        return {"producer_session": expected_session, "available": True,
                "valid": True, "reason": "fresh",
                "local_motion_geometry": {"valid": True}}


class _Facade:
    def __init__(self, *, goal_active=False, available=True):
        self.goal_active = goal_active
        self.available = available

    def get_localization_status(self):
        if not self.available:
            return {}
        return {"navigation": {
            "goal_active": self.goal_active,
            # Explicitly prove an unlocalized robot remains eligible.
            "localization_validated": False,
        }}


class _Robot:
    def __init__(self, *, zero=True, stop_ok=True, events=None):
        self.zero = zero
        self.stop_ok = stop_ok
        self.stop_calls = 0
        self.events = events if events is not None else []

    def status(self):
        if not self.zero:
            return {"ok": True, "ros_ready": True,
                    "motion": {"linear_x": 0.1, "angular_z": 0,
                               "streaming": False}}
        return {"ok": True, "ros_ready": True,
                "motion": {"linear_x": 0, "angular_z": 0,
                           "streaming": False}}

    def stop(self):
        self.stop_calls += 1
        self.events.append("stop")
        return {"ok": self.stop_ok}


class _Behavior:
    def __init__(self, *, forward_ok=True, turn_ok=True, events=None):
        self.forward_ok = forward_ok
        self.turn_ok = turn_ok
        self.events = events if events is not None else []
        self.forward_calls = []
        self.turn_calls = []

    def execute_guarded_local_forward(self, *, expected_lidar_session):
        self.events.append("forward")
        self.forward_calls.append(expected_lidar_session)
        return {"ok": self.forward_ok, "motion_executed": self.forward_ok,
                "jit_lidar_validation": True}

    def execute_guarded_turn(self, direction, angular_speed, duration, *,
                             expected_lidar_session, safety_mode):
        self.events.append("turn")
        self.turn_calls.append((direction, angular_speed, duration,
                                expected_lidar_session, safety_mode))
        return {"ok": self.turn_ok, "permitted": self.turn_ok,
                "confirmed_forwarded": self.turn_ok,
                "jit_lidar_validation": True}


def _runtime(*, robot=None, behavior=None, facade=None, active_mission=None):
    value = object.__new__(CognitiveRuntime)
    value.running = True
    value.robot_client = robot or _Robot()
    value.behavior_manager = behavior or _Behavior()
    value.world_model = _World()
    value.lidar_worker = SimpleNamespace(running=True, session=SESSION)
    value.localization_facade = facade or _Facade()
    value.mission_manager = _MissionManager(active_mission)
    value._state_lock = threading.RLock()
    value._active_localization_lock = threading.Lock()
    value._local_reactive_step_lock = threading.Lock()
    value._physical_action_lock = threading.Lock()
    value._behavior_execution_generation = None
    return value


def _install_decision(monkeypatch, value, events):
    def decide(state, *, expected_session, forward_linear_speed):
        assert state["producer_session"] == expected_session == SESSION
        assert forward_linear_speed == 0.10
        events.append("decision")
        return _decision(value)

    monkeypatch.setattr(runtime_module, "decide_forward_reaction", decide)


@pytest.mark.parametrize("value,expected_direction", [
    ("FORWARD_CLEAR", None),
    ("TURN_LEFT", "LEFT"),
    ("TURN_RIGHT", "RIGHT"),
])
def test_actionable_decision_executes_exactly_one_guarded_primitive_then_stop(
    monkeypatch, value, expected_direction,
):
    events = []
    robot = _Robot(events=events)
    behavior = _Behavior(events=events)
    runtime = _runtime(robot=robot, behavior=behavior)
    _install_decision(monkeypatch, value, events)

    result = runtime.run_local_reactive_step()

    assert result["ok"] is True
    assert result["action_attempted"] is True
    assert result["action_executed"] is True
    assert result["bridge_stopped"] is True
    assert events[0] == "decision" and events[-1] == "stop"
    assert len(behavior.forward_calls) + len(behavior.turn_calls) == 1
    if expected_direction is None:
        assert behavior.forward_calls == [SESSION]
        assert behavior.turn_calls == []
    else:
        assert behavior.forward_calls == []
        assert behavior.turn_calls == [(
            expected_direction, 0.25, 0.50, SESSION,
            "ROTATIONAL_SWEPT_FOOTPRINT",
        )]


def test_stop_blocked_executes_no_motion_but_stops_and_verifies_zero(monkeypatch):
    events = []
    robot = _Robot(events=events)
    behavior = _Behavior(events=events)
    runtime = _runtime(robot=robot, behavior=behavior)
    _install_decision(monkeypatch, "STOP_BLOCKED", events)

    result = runtime.run_local_reactive_step()

    assert result["ok"] is False
    assert result["action_attempted"] is False
    assert result["action_executed"] is False
    assert behavior.forward_calls == behavior.turn_calls == []
    assert robot.stop_calls == 1 and result["bridge_stopped"] is True


@pytest.mark.parametrize("value", ["FORWARD_CLEAR", "TURN_LEFT"])
def test_executor_veto_still_stops_without_fallback_action(monkeypatch, value):
    events = []
    robot = _Robot(events=events)
    behavior = _Behavior(forward_ok=False, turn_ok=False, events=events)
    runtime = _runtime(robot=robot, behavior=behavior)
    _install_decision(monkeypatch, value, events)

    result = runtime.run_local_reactive_step()

    assert result["ok"] is False
    assert result["action_attempted"] is True
    assert result["action_executed"] is False
    assert result["reason"] == "local_reactive_executor_vetoed"
    assert len(behavior.forward_calls) + len(behavior.turn_calls) == 1
    assert robot.stop_calls == 1 and result["bridge_stopped"] is True


def test_final_nonzero_bridge_fails_closed_after_executor_attempt(monkeypatch):
    events = []
    robot = _Robot(zero=True, events=events)
    behavior = _Behavior(events=events)
    runtime = _runtime(robot=robot, behavior=behavior)
    _install_decision(monkeypatch, "TURN_LEFT", events)
    original_stop = robot.stop

    def stop_then_report_nonzero():
        result = original_stop()
        robot.zero = False
        return result

    robot.stop = stop_then_report_nonzero
    result = runtime.run_local_reactive_step()
    assert result["ok"] is False
    assert result["bridge_stopped"] is False
    assert result["reason"] == "bridge_not_stopped_after_local_reactive_step"


def test_navigation_goal_or_other_physical_behavior_blocks_before_decision(monkeypatch):
    events = []
    runtime = _runtime(facade=_Facade(goal_active=True))
    _install_decision(monkeypatch, "FORWARD_CLEAR", events)
    result = runtime.run_local_reactive_step()
    assert result["reason"] == "navigation_goal_active"
    assert events == []

    runtime = _runtime(active_mission=object())
    _install_decision(monkeypatch, "FORWARD_CLEAR", events)
    result = runtime.run_local_reactive_step()
    assert result["reason"] == "physical_behavior_already_active"
    assert events == []


def test_active_localization_lock_blocks_local_step_before_decision(monkeypatch):
    events = []
    runtime = _runtime()
    runtime._active_localization_lock.acquire()
    try:
        _install_decision(monkeypatch, "FORWARD_CLEAR", events)
        result = runtime.run_local_reactive_step()
    finally:
        runtime._active_localization_lock.release()
    assert result["reason"] == "active_localization_already_running"
    assert events == []


def test_shared_physical_action_lock_blocks_another_local_step(monkeypatch):
    events = []
    runtime = _runtime()
    runtime._physical_action_lock.acquire()
    try:
        _install_decision(monkeypatch, "FORWARD_CLEAR", events)
        result = runtime.run_local_reactive_step()
    finally:
        runtime._physical_action_lock.release()
    assert result["reason"] == "physical_behavior_already_active"
    assert events == []


def test_actual_stale_lidar_decision_stops_without_executor_dispatch():
    class StaleWorld(_World):
        def get_lidar_obstacles(self, *, expected_session):
            return {"producer_session": expected_session, "available": True,
                    "valid": True, "reason": "fresh",
                    "effective_age_seconds": 0.31,
                    "local_motion_geometry": {"valid": True}}

    robot = _Robot()
    behavior = _Behavior()
    runtime = _runtime(robot=robot, behavior=behavior)
    runtime.world_model = StaleWorld()
    result = runtime.run_local_reactive_step()
    assert result["decision"] == "STOP_BLOCKED"
    assert result["action_executed"] is False
    assert behavior.forward_calls == behavior.turn_calls == []
    assert robot.stop_calls == 1 and result["bridge_stopped"] is True


def test_unlocalized_local_reactive_step_needs_no_map_camera_or_world_mutation(monkeypatch):
    events = []
    runtime = _runtime(facade=_Facade(goal_active=False))
    _install_decision(monkeypatch, "TURN_LEFT", events)
    result = runtime.run_local_reactive_step()
    assert result["ok"] is True
    assert runtime.localization_facade.get_localization_status()["navigation"]["localization_validated"] is False
    assert runtime.world_model.calls == 1


def test_guarded_local_forward_is_a_semantic_free_delegate(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []

    def existing(*, expected_lidar_session, linear_speed, duration):
        calls.append((expected_lidar_session, linear_speed, duration))
        return {"ok": True, "motion_executed": True}

    monkeypatch.setattr(manager, "execute_single_marvin_approach_step", existing)
    result = manager.execute_guarded_local_forward(expected_lidar_session=SESSION)
    assert result["motion_executed"] is True
    assert calls == [(SESSION, manager.FIND_APPROACH_FORWARD_SPEED,
                      manager.FIND_APPROACH_FORWARD_SECONDS)]
