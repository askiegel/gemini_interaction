"""Offline one-primitive contracts for Marvin pursuit coordination."""

from copy import deepcopy

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


SESSION = "marvin-pursuit-session"


class Robot:
    def __init__(self, result=None, error=None):
        self.forward_calls = 0
        self.result = result if result is not None else {"ok": True, "executed": True}
        self.error = error

    def local_forward(self):
        self.forward_calls += 1
        if self.error:
            raise self.error
        return self.result


class World:
    def __init__(self, lidar=None):
        self.lidar = lidar or {"producer_session": SESSION}
        self.calls = []

    def get_lidar_obstacles(self, *, expected_session, now=None):
        self.calls.append((expected_session, now))
        return self.lidar


def ready(**updates):
    value = {"ok": True, "state": "READY_TO_APPROACH", "pursuit_authorized": True}
    value.update(updates)
    return value


def safety(*, permitted, geometry=True):
    return {
        "permitted": permitted,
        "reason": "protected_region_clear" if permitted else "translation_protected_region_violated",
        "geometry": {"valid": True} if geometry else None,
    }


def invoke(monkeypatch, *, pursuit=None, forward_safety=None, robot=None,
           lidar=None, avoidance=None, avoidance_error=None, inputs=None):
    robot = robot or Robot()
    world = World(lidar)
    manager = BehaviorManager(robot_client=robot, world_model=world)
    manager.lidar_session = SESSION
    pursuit_calls, safety_calls, avoidance_calls = [], [], []

    def evaluate(*args, **kwargs):
        pursuit_calls.append((args, kwargs))
        return ready() if pursuit is None else pursuit

    def evaluate_safety(*args, **kwargs):
        safety_calls.append((args, kwargs))
        return safety(permitted=True) if forward_safety is None else forward_safety

    def avoid(**kwargs):
        avoidance_calls.append(kwargs)
        if avoidance_error:
            raise avoidance_error
        return avoidance if avoidance is not None else {
            "ok": True, "motion_executed": True, "replan_required": True,
            "executed_primitive": "left_turn", "reason": "avoidance_complete",
        }

    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", evaluate)
    monkeypatch.setattr(behavior_manager_module, "evaluate_local_motion_safety", evaluate_safety)
    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", avoid)
    inputs = inputs or (
        {"preview": "current"}, {"lock": "current"}, {"snapshot": "current"},
    )
    result = manager.execute_marvin_pursuit_step(*inputs, now=10.0)
    return result, robot, world, pursuit_calls, safety_calls, avoidance_calls


def assert_one_primitive(robot, avoidance_calls):
    assert robot.forward_calls + len(avoidance_calls) <= 1


def test_ready_and_clear_dispatches_one_forward_then_requires_replan(monkeypatch):
    result, robot, world, pursuit_calls, safety_calls, avoids = invoke(monkeypatch)
    assert result["ok"] is result["motion_executed"] is result["replan_required"] is True
    assert result["decision"] == "approach_forward"
    assert robot.forward_calls == 1 and avoids == []
    assert world.calls == [(SESSION, 10.0)] and len(pursuit_calls) == len(safety_calls) == 1
    assert_one_primitive(robot, avoids)


def test_ready_and_trusted_blockage_calls_only_one_avoidance_step(monkeypatch):
    result, robot, _world, _pursuit, _safety, avoids = invoke(
        monkeypatch, forward_safety=safety(permitted=False),
    )
    assert result["decision"] == "avoidance_required"
    assert result["motion_executed"] is result["replan_required"] is True
    assert robot.forward_calls == 0 and len(avoids) == 1
    assert_one_primitive(robot, avoids)


def test_every_non_authorized_state_never_reads_lidar_or_moves(monkeypatch):
    for state in ("SEARCHING", "CANDIDATE_SEEN", "MARVIN_LOCKED", "REACQUIRE_REQUIRED", "SAME_IDENTITY_REACQUIRED", "INSUFFICIENT_EVIDENCE"):
        result, robot, world, calls, safety_calls, avoids = invoke(
            monkeypatch, pursuit=ready(state=state, pursuit_authorized=False),
        )
        assert result["decision"] == "no_motion" and result["motion_executed"] is False
        assert robot.forward_calls == 0 and world.calls == safety_calls == avoids == []
        assert len(calls) == 1


def test_ready_named_state_without_authority_fails_closed(monkeypatch):
    result, robot, world, _calls, safety_calls, avoids = invoke(
        monkeypatch, pursuit=ready(pursuit_authorized=False),
    )
    assert result["reason"] == "marvin_pursuit_not_authorized"
    assert robot.forward_calls == 0 and world.calls == safety_calls == avoids == []


def test_stale_or_wrong_session_lidar_never_dispatches(monkeypatch):
    for lidar, forward_safety in (
        ({"producer_session": SESSION}, safety(permitted=False, geometry=False)),
        ({"producer_session": "wrong"}, safety(permitted=False)),
    ):
        result, robot, _world, _calls, _safety, avoids = invoke(
            monkeypatch, lidar=lidar, forward_safety=forward_safety,
        )
        assert result["reason"] == "marvin_pursuit_lidar_not_trusted"
        assert robot.forward_calls == 0 and avoids == []


def test_avoidance_failure_or_exception_never_falls_back_to_forward(monkeypatch):
    for avoidance, error in (({"ok": False, "motion_executed": False, "reason": "blocked"}, None), (None, RuntimeError("offline"))):
        result, robot, _world, _calls, _safety, avoids = invoke(
            monkeypatch, forward_safety=safety(permitted=False), avoidance=avoidance,
            avoidance_error=error,
        )
        assert result["motion_executed"] is False
        assert robot.forward_calls == 0 and len(avoids) == 1
        assert_one_primitive(robot, avoids)


def test_forward_failure_or_exception_never_falls_back_to_avoidance(monkeypatch):
    for robot in (Robot(result={"ok": False, "executed": False, "reason": "failed"}), Robot(error=RuntimeError("offline"))):
        result, used_robot, _world, _calls, _safety, avoids = invoke(monkeypatch, robot=robot)
        assert result["motion_executed"] is False
        assert used_robot.forward_calls == 1 and avoids == []
        assert_one_primitive(used_robot, avoids)


def test_fresh_evaluation_no_cached_authorization_determinism_and_no_mutation(monkeypatch):
    source = ({"preview": "current"}, {"lock": "current"}, {"snapshot": "current"})
    before = deepcopy(source)
    result, robot, _world, calls, _safety, avoids = invoke(monkeypatch, inputs=source)
    assert source == before and calls[0][0] == source and result["replan_required"] is True
    # A second call invokes the evaluator again; no READY state is retained.
    second, second_robot, _world, second_calls, _safety, second_avoids = invoke(monkeypatch)
    assert result == second and len(calls) == len(second_calls) == 1
    assert_one_primitive(robot, avoids)
    assert_one_primitive(second_robot, second_avoids)


def test_coordinator_has_no_mission_or_runtime_integration():
    source = open("behavior_manager.py", encoding="utf-8").read()
    start = source.index("    def execute_marvin_pursuit_step(")
    end = source.index("    def execute_local_obstacle_avoidance_step(", start)
    coordinator = source[start:end]
    for forbidden in ("self.execute_local_obstacle_avoidance_loop", "self._execute_find", "self.execute_behavior", "arrived_at_marvin"):
        assert forbidden not in coordinator
