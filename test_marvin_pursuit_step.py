"""Offline one-primitive contracts for Marvin pursuit coordination."""

from copy import deepcopy

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


SESSION = "marvin-pursuit-session"


class Interlock:
    def __init__(self, results=None):
        self.results = iter(results or [(True, "fresh_clear")])
        self.calls = 0

    def refresh(self):
        self.calls += 1
        return next(self.results)


class Robot:
    def __init__(self, result=None, error=None, interlock=None):
        self.forward_calls = 0
        self.forward_requests = []
        self.result = result if result is not None else {"ok": True, "executed": True}
        self.error = error
        self.forward_interlock = interlock or Interlock()

    def move_forward(self, *, speed, seconds):
        self.forward_calls += 1
        self.forward_requests.append((speed, seconds))
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


def visual_preview(*, centered=False):
    x1, x2 = (270.0, 370.0) if centered else (400.0, 500.0)
    return {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "target_found": True, "source": "marvin_local_tracker", "identity_confirmed": True,
        "source_timestamp": "2026-09-26T16:00:00+00:00",
        "image_width": 640.0, "image_height": 480.0,
        "bbox": {"x1": x1, "y1": 20.0, "x2": x2, "y2": 220.0},
        "marvin_continuity": {"tracker_id": 1, "tracker_source": "diagnostic"},
    }


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
        if isinstance(forward_safety, list):
            return forward_safety.pop(0)
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
    assert robot.forward_requests == [(0.08, 0.50)]
    assert world.calls == [(SESSION, 10.0), (SESSION, 10.0)]
    assert len(pursuit_calls) == 1 and len(safety_calls) == 2
    assert robot.forward_interlock.calls == 1
    assert_one_primitive(robot, avoids)


def test_visual_off_center_dispatches_one_guarded_alignment_before_any_forward(monkeypatch):
    robot = Robot()
    manager = BehaviorManager(robot_client=robot, world_model=World())
    manager.lidar_session = SESSION
    turns = []
    monkeypatch.setattr(
        manager, "_execute_target_directed_turn",
        lambda *args, **kwargs: turns.append((args, kwargs)) or {
            "ok": True, "permitted": True,
        },
    )
    result = manager.execute_marvin_pursuit_step(
        visual_preview(), {}, {"tracking_mode": "UNLOCKED"}, now="2026-09-26T16:00:00+00:00",
    )
    assert result["ok"] is result["motion_executed"] is result["replan_required"] is True
    assert result["decision"] == "align_right"
    assert robot.forward_calls == 0
    assert turns == [(("RIGHT", 0.25, 0.50), {"expected_lidar_session": SESSION})]


def test_visual_centered_uses_existing_lidar_gated_forward_path(monkeypatch):
    result, robot, _world, _calls, safety_calls, avoids = invoke(
        monkeypatch,
        pursuit={"ok": True, "state": "VISUAL_READY_TO_APPROACH", "pursuit_authorized": True},
        inputs=(visual_preview(centered=True), {}, {"tracking_mode": "UNLOCKED"}),
    )
    assert result["decision"] == "approach_forward"
    assert robot.forward_calls == 1 and len(safety_calls) == 2 and avoids == []
    assert robot.forward_requests == [(0.08, 0.50)]


def test_canonical_bounded_bridge_result_is_normalized_and_replans(monkeypatch):
    bridge_result = {
        "ok": True, "action": "motion", "mode": "bounded",
        "linear_x": 0.08, "angular_z": 0.0, "duration": 0.50,
        "automatic_stop": True, "returned_immediately": False,
    }
    result, robot, _world, _calls, _safety, avoids = invoke(
        monkeypatch, robot=Robot(result=bridge_result),
    )
    assert result["ok"] is result["motion_executed"] is result["replan_required"] is True
    assert result["forward_result"]["executed"] is True
    assert robot.forward_requests == [(0.08, 0.50)] and avoids == []


def test_partial_bridge_success_never_normalizes(monkeypatch):
    malformed = {
        "ok": True, "action": "motion", "mode": "bounded",
        "linear_x": 0.08, "angular_z": 0.0, "duration": 0.50,
        "automatic_stop": True,
    }
    for bridge_result in (malformed, {"ok": False}):
        result, robot, _world, _calls, _safety, avoids = invoke(
            monkeypatch, robot=Robot(result=bridge_result),
        )
        assert result["ok"] is result["motion_executed"] is False
        assert result["replan_required"] is False
        assert robot.forward_requests == [(0.08, 0.50)] and avoids == []


def test_pre_dispatch_stale_veto_never_calls_robot_and_requests_replan(monkeypatch):
    stale = {"permitted": False, "reason": "stale", "geometry": None}
    result, robot, world, _calls, safety_calls, avoids = invoke(
        monkeypatch, forward_safety=[safety(permitted=True), stale],
    )
    assert result["ok"] is True and result["motion_executed"] is False
    assert result["replan_required"] is result["stale_replan"] is True
    assert result["stale_replan_classification"] == "NONPHYSICAL_STALE_REPLAN"
    assert result["action_budget_consumed"] is False
    assert robot.forward_calls == 0 and avoids == []
    assert world.calls == [(SESSION, 10.0), (SESSION, 10.0)]
    assert len(safety_calls) == 2


def test_post_transport_stale_invalidation_is_physical_or_uncertain_replan(monkeypatch):
    bridge_result = {
        "ok": False, "forwarded": True, "transport_attempted": True,
        "bounded_forward_invalidated": True, "reason": "stale",
        "transport_result": {
            "ok": True, "action": "motion", "mode": "bounded",
            "linear_x": 0.08, "angular_z": 0.0, "duration": 0.50,
            "automatic_stop": True, "returned_immediately": False,
        },
    }
    result, robot, _world, _calls, _safety, avoids = invoke(
        monkeypatch, robot=Robot(result=bridge_result),
    )
    assert result["ok"] is result["replan_required"] is True
    assert result["stale_replan_classification"] == "PHYSICAL_OR_UNCERTAIN_STALE_REPLAN"
    assert result["action_budget_consumed"] is result["motion_executed"] is True
    assert result["motion_possible"] is True
    assert robot.forward_requests == [(0.08, 0.50)] and avoids == []


def test_verified_no_transport_stale_result_is_nonphysical_replan(monkeypatch):
    result, robot, _world, _calls, _safety, avoids = invoke(
        monkeypatch,
        robot=Robot(result={"ok": False, "forwarded": False, "reason": "stale"}),
    )
    assert result["ok"] is result["replan_required"] is True
    assert result["motion_executed"] is False
    assert result["stale_replan_classification"] == "NONPHYSICAL_STALE_REPLAN"
    assert result["action_budget_consumed"] is False
    assert robot.forward_calls == 1 and avoids == []


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
