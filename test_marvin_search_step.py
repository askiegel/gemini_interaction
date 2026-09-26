"""Offline one-turn contracts for Marvin search execution coordination."""

from copy import deepcopy
import inspect

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


SESSION = "marvin-search-session"


def planned(action, *, ok=True, identity="marvin-1", reason="planned"):
    return {
        "ok": ok, "selected_search_action": action,
        "selected_identity_id": identity, "reason": reason,
    }


def turn_result(*, ok=True, permitted=True, confirmed=True, reason="turn_complete"):
    return {
        "ok": ok, "permitted": permitted,
        "confirmed_forwarded": confirmed, "reason": reason,
    }


def invoke(monkeypatch, planner_output, *, turn_output=None, turn_error=None,
           selected_identity_id="marvin-1"):
    manager = BehaviorManager(robot_client=object())
    manager.lidar_session = SESSION
    planner_calls, turn_calls = [], []

    def planner(*args, **kwargs):
        planner_calls.append((args, kwargs))
        if isinstance(planner_output, Exception):
            raise planner_output
        return planner_output

    def turn(*args, **kwargs):
        turn_calls.append((args, kwargs))
        if turn_error:
            raise turn_error
        return turn_output if turn_output is not None else turn_result()

    monkeypatch.setattr(behavior_manager_module, "plan_marvin_search_step", planner)
    monkeypatch.setattr(manager, "execute_guarded_turn", turn)
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, prior_search_history=[],
        selected_identity_id=selected_identity_id, preview_result=None, now=10.0,
    )
    return result, manager, planner_calls, turn_calls


def assert_one_turn(calls):
    assert len(calls) <= 1


def test_left_plan_dispatches_one_canonical_left_guarded_turn(monkeypatch):
    result, manager, planner_calls, turns = invoke(monkeypatch, planned("turn_left"))
    assert result["ok"] is result["motion_executed"] is result["replan_required"] is True
    assert result["executed_primitive"] == "guarded_turn_left"
    assert turns == [(("LEFT", manager.SEARCH_TURN_SPEED, manager.SEARCH_TURN_SECONDS), {"expected_lidar_session": SESSION, "now": 10.0})]
    assert len(planner_calls) == 1
    assert_one_turn(turns)


def test_right_plan_dispatches_one_canonical_right_guarded_turn(monkeypatch):
    result, manager, _planner, turns = invoke(monkeypatch, planned("turn_right"))
    assert result["executed_primitive"] == "guarded_turn_right"
    assert turns[0][0] == ("RIGHT", manager.SEARCH_TURN_SPEED, manager.SEARCH_TURN_SECONDS)
    assert result["replan_required"] is True
    assert_one_turn(turns)


def test_nonmotion_actions_never_turn(monkeypatch):
    for action in ("preview_only", "reacquired", "search_complete", "fail_closed"):
        result, _manager, _planner, turns = invoke(monkeypatch, planned(action))
        assert result["motion_executed"] is False and turns == []
        assert_one_turn(turns)


def test_malformed_unknown_or_planner_exception_never_turn(monkeypatch):
    for output in (None, planned("unknown"), RuntimeError("offline")):
        result, _manager, _planner, turns = invoke(monkeypatch, output)
        assert result["motion_executed"] is False and turns == []


def test_left_or_right_failure_never_falls_back(monkeypatch):
    for action in ("turn_left", "turn_right"):
        result, _manager, _planner, turns = invoke(
            monkeypatch, planned(action), turn_output=turn_result(ok=False, reason="denied"),
        )
        assert result["motion_executed"] is False and len(turns) == 1
        assert_one_turn(turns)


def test_turn_exception_never_dispatches_a_second_primitive(monkeypatch):
    result, _manager, _planner, turns = invoke(
        monkeypatch, planned("turn_left"), turn_error=RuntimeError("offline"),
    )
    assert result["reason"] == "marvin_search_guarded_turn_exception"
    assert len(turns) == 1
    assert_one_turn(turns)


def test_selected_identity_mismatch_fails_closed_before_turn(monkeypatch):
    result, _manager, _planner, turns = invoke(
        monkeypatch, planned("turn_left", identity="marvin-2"),
    )
    assert result["reason"] == "marvin_search_selected_identity_changed"
    assert turns == []


def test_inputs_not_mutated_and_result_is_deterministic(monkeypatch):
    source = {"state": "SEARCHING"}
    history = [{"selected_search_action": "turn_left"}]
    before = deepcopy((source, history))
    first, _manager, _planner, first_turns = invoke(monkeypatch, planned("preview_only"))
    second, _manager, _planner, second_turns = invoke(monkeypatch, planned("preview_only"))
    assert first == second and first_turns == second_turns == []
    assert (source, history) == before


def test_coordinator_has_no_pursuit_loop_or_direct_transport():
    source = inspect.getsource(behavior_manager_module)
    start = source.index("    def execute_marvin_search_step(")
    end = source.index("    def execute_find_marvin_controller(", start)
    coordinator = source[start:end]
    for forbidden in (
        "robot.local_forward", "execute_marvin_pursuit_step",
        "execute_local_obstacle_avoidance_step", "execute_local_obstacle_avoidance_loop",
        "while true", "arrived_at_marvin",
    ):
        assert forbidden not in coordinator.lower()
    assert coordinator.count("self.execute_guarded_turn(") == 1
