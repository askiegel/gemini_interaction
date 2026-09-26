"""Offline one-primitive tests for local-obstacle avoidance orchestration."""

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


SESSION = "avoidance-session"


class WorldModel:
    def __init__(self):
        self.calls = []

    def get_lidar_obstacles(self, *, expected_session, now=None):
        self.calls.append((expected_session, now))
        return {"authoritative": True, "producer_session": expected_session}


class Robot:
    def __init__(self, result=None, error=None):
        self.forward_calls = 0
        self.result = result if result is not None else {
            "ok": True, "executed": True,
        }
        self.error = error

    def local_forward(self):
        self.forward_calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def planner_result(action, *, ok=True, producer_session=SESSION,
                   selected_permitted=True, turn_permitted=True,
                   reason="planned"):
    evaluations = {
        "forward": {"permitted": True},
        "forward_left": {"permitted": True},
        "forward_right": {"permitted": True},
        "left_turn": {"permitted": turn_permitted},
        "right_turn": {"permitted": turn_permitted},
    }
    if action in evaluations:
        evaluations[action] = {"permitted": selected_permitted}
    elif action is not None:
        evaluations[action] = {"permitted": selected_permitted}
    return {
        "ok": ok,
        "selected_action": action,
        "producer_session": producer_session,
        "reason": reason,
        "candidate_evaluations": evaluations,
    }


def invoke(monkeypatch, planned, *, robot=None, turn_result=None,
           turn_error=None):
    robot = robot or Robot()
    world = WorldModel()
    manager = BehaviorManager(robot_client=robot, world_model=world)
    manager.lidar_session = SESSION
    planner_calls = []
    turn_calls = []

    def planner(state, *, expected_session, now):
        planner_calls.append((state, expected_session, now))
        return planned

    def turn(*args, **kwargs):
        turn_calls.append((args, kwargs))
        if turn_error is not None:
            raise turn_error
        return turn_result if turn_result is not None else {
            "ok": True, "permitted": True, "confirmed_forwarded": True,
        }

    monkeypatch.setattr(
        behavior_manager_module, "plan_local_obstacle_avoidance", planner,
    )
    monkeypatch.setattr(manager, "execute_guarded_turn", turn)
    result = manager.execute_local_obstacle_avoidance_step(now=10.0)
    return result, robot, world, planner_calls, turn_calls


def assert_at_most_one_primitive(robot, turns):
    assert robot.forward_calls + len(turns) <= 1


def test_forward_dispatches_exactly_one_local_forward(monkeypatch):
    result, robot, world, planner_calls, turns = invoke(
        monkeypatch, planner_result("forward"),
    )
    assert result["ok"] is True
    assert result["executed_primitive"] == "forward"
    assert result["replan_required"] is True
    assert robot.forward_calls == 1 and turns == []
    assert world.calls == [(SESSION, 10.0)]
    assert len(planner_calls) == 1
    assert_at_most_one_primitive(robot, turns)


def test_left_turn_dispatches_exactly_once(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("left_turn"),
    )
    assert result["ok"] is True and result["executed_primitive"] == "left_turn"
    assert turns[0][0] == ("LEFT",)
    assert robot.forward_calls == 0 and result["replan_required"] is True
    assert_at_most_one_primitive(robot, turns)


def test_right_turn_dispatches_exactly_once(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("right_turn"),
    )
    assert result["ok"] is True and result["executed_primitive"] == "right_turn"
    assert turns[0][0] == ("RIGHT",)
    assert robot.forward_calls == 0
    assert_at_most_one_primitive(robot, turns)


def test_forward_left_becomes_one_left_turn_then_requires_replan(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("forward_left"),
    )
    assert result["ok"] is True
    assert result["executed_primitive"] == "left_turn"
    assert result["reason"] == "turned_for_forward_left_replan"
    assert result["replan_required"] is True
    assert turns[0][0] == ("LEFT",) and robot.forward_calls == 0
    assert_at_most_one_primitive(robot, turns)


def test_forward_right_becomes_one_right_turn_then_requires_replan(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("forward_right"),
    )
    assert result["ok"] is True
    assert result["executed_primitive"] == "right_turn"
    assert result["reason"] == "turned_for_forward_right_replan"
    assert result["replan_required"] is True
    assert turns[0][0] == ("RIGHT",) and robot.forward_calls == 0
    assert_at_most_one_primitive(robot, turns)


def test_none_or_planner_failure_never_dispatches(monkeypatch):
    for planned in (
        planner_result(None, ok=False, reason="no_safe_local_avoidance"),
        planner_result("forward", ok=False, reason="planner_failure"),
    ):
        result, robot, _world, _planner, turns = invoke(monkeypatch, planned)
        assert result["ok"] is False and result["motion_executed"] is False
        assert robot.forward_calls == 0 and turns == []
        assert_at_most_one_primitive(robot, turns)


def test_stale_or_wrong_session_planner_result_never_dispatches(monkeypatch):
    for planned in (
        planner_result("forward", ok=False, reason="stale"),
        planner_result("forward", producer_session="wrong", reason="stale"),
    ):
        result, robot, _world, _planner, turns = invoke(monkeypatch, planned)
        assert result["ok"] is False
        assert robot.forward_calls == 0 and turns == []


def test_selected_candidate_denial_and_diagonal_turn_denial_fail_closed(monkeypatch):
    cases = (
        planner_result("forward", selected_permitted=False),
        planner_result("forward_left", turn_permitted=False),
    )
    for planned in cases:
        result, robot, _world, _planner, turns = invoke(monkeypatch, planned)
        assert result["ok"] is False
        assert robot.forward_calls == 0 and turns == []


def test_unsupported_action_never_dispatches(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("reverse"),
    )
    assert result["reason"] == "unsupported_local_avoidance_action"
    assert robot.forward_calls == 0 and turns == []


def test_primitive_failure_never_falls_back(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch,
        planner_result("forward"),
        robot=Robot(result={"ok": False, "executed": False, "reason": "bridge_failed"}),
    )
    assert result["ok"] is False and result["reason"] == "bridge_failed"
    assert robot.forward_calls == 1 and turns == []
    assert_at_most_one_primitive(robot, turns)


def test_primitive_exception_never_falls_back(monkeypatch):
    result, robot, _world, _planner, turns = invoke(
        monkeypatch, planner_result("forward"), robot=Robot(error=RuntimeError("offline")),
    )
    assert result["ok"] is False
    assert result["reason"] == "local_avoidance_primitive_exception"
    assert robot.forward_calls == 1 and turns == []
    assert_at_most_one_primitive(robot, turns)


def test_existing_pure_planner_module_is_not_modified_by_execution(monkeypatch):
    planned = planner_result("forward_right")
    original = {name: dict(value) for name, value in planned["candidate_evaluations"].items()}
    result, _robot, _world, _planner, _turns = invoke(monkeypatch, planned)
    assert result["planner"] is planned
    assert planned["candidate_evaluations"] == original
