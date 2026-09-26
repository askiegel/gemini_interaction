"""Offline bounded-controller contracts for Find-Marvin pursuit."""

from copy import deepcopy

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


def evidence(identity="marvin-1"):
    return {
        "preview_result": {"preview": "fresh"},
        "target_lock_result": {"lock": "fresh"},
        "target_lock_snapshot": {"snapshot": "fresh"},
        "selected_identity_id": identity,
    }


def pursuit(state="READY_TO_APPROACH", authorized=True, identity="marvin-1"):
    return {
        "ok": True,
        "state": state,
        "pursuit_authorized": authorized,
        "selected_identity_id": identity,
        "entity_id": "entity-1",
        "fresh": True,
        "geometry_usable": True,
    }


def successful_step():
    return {
        "ok": True, "decision": "approach_forward",
        "executed_primitive": "forward", "motion_executed": True,
        "replan_required": True,
    }


def invoke(monkeypatch, states, *, steps=None, max_actions=6, identities=None):
    manager = BehaviorManager(robot_client=object())
    provider_calls, evaluator_calls, step_calls = [], [], []
    state_values = iter(states)
    identity_values = iter(identities or ["marvin-1"] * len(states))
    step_values = iter(steps or [successful_step()] * max_actions)

    def provider():
        provider_calls.append(True)
        return evidence(next(identity_values))

    def evaluate(*args, **kwargs):
        evaluator_calls.append((args, kwargs))
        value = next(state_values)
        if isinstance(value, Exception):
            raise value
        return value

    def step(*args, **kwargs):
        step_calls.append((args, kwargs))
        value = next(step_values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", evaluate)
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", step)
    result = manager.execute_find_marvin_controller(provider, max_actions=max_actions, now=10.0)
    return result, provider_calls, evaluator_calls, step_calls


def test_three_ready_iterations_are_fresh_and_action_limited(monkeypatch):
    result, providers, evaluations, steps = invoke(
        monkeypatch, [pursuit(), pursuit(), pursuit()], max_actions=3,
    )
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert result["actions_executed"] == len(providers) == len(evaluations) == len(steps) == 3
    assert len(result["history"]) == 3


def test_default_bound_stops_an_infinite_ready_stream_at_six(monkeypatch):
    result, providers, evaluations, steps = invoke(
        monkeypatch, [pursuit()] * 6,
    )
    assert result["max_actions"] == result["actions_executed"] == 6
    assert len(providers) == len(evaluations) == len(steps) == 6


def test_smaller_bound_permits_exactly_one_action(monkeypatch):
    result, _providers, _evaluations, steps = invoke(monkeypatch, [pursuit()], max_actions=1)
    assert result["actions_executed"] == len(steps) == 1


def test_all_nonready_states_pause_without_a_pursuit_step(monkeypatch):
    for state in ("SEARCHING", "CANDIDATE_SEEN", "MARVIN_LOCKED", "REACQUIRE_REQUIRED", "SAME_IDENTITY_REACQUIRED", "INSUFFICIENT_EVIDENCE"):
        result, providers, evaluations, steps = invoke(
            monkeypatch, [pursuit(state=state, authorized=False)],
        )
        assert result["completed"] is False and result["actions_executed"] == 0
        assert len(providers) == len(evaluations) == 1 and steps == []
    assert result["reason"] == "insufficient_evidence"


def test_ready_without_authority_stops_without_a_step(monkeypatch):
    result, _providers, _evaluations, steps = invoke(
        monkeypatch, [pursuit(authorized=False)],
    )
    assert result["reason"] == "find_marvin_pursuit_not_authorized" and steps == []


def test_step_failure_exception_no_motion_and_no_replan_stop_immediately(monkeypatch):
    cases = (
        ([{"ok": False}], "find_marvin_pursuit_step_failed"),
        ([RuntimeError("offline")], "find_marvin_pursuit_step_exception"),
        ([{"ok": True, "motion_executed": False, "replan_required": True}], "find_marvin_pursuit_step_no_motion"),
        ([{"ok": True, "motion_executed": True, "replan_required": False}], "find_marvin_pursuit_step_replan_required"),
    )
    for steps, reason in cases:
        result, providers, evaluations, calls = invoke(
            monkeypatch, [pursuit(), pursuit()], steps=steps, max_actions=2,
        )
        assert result["reason"] == reason and result["actions_executed"] == len(calls) == 1
        assert len(providers) == len(evaluations) == 1


def test_ready_then_reacquire_stops_after_one_prior_action(monkeypatch):
    result, _providers, evaluations, steps = invoke(
        monkeypatch, [pursuit(), pursuit("REACQUIRE_REQUIRED", False)], max_actions=3,
    )
    assert result["reason"] == "reacquire_required"
    assert result["actions_executed"] == len(steps) == 1 and len(evaluations) == 2
    assert len(result["history"]) == 2


def test_identity_change_between_actions_fails_closed(monkeypatch):
    result, _providers, evaluations, steps = invoke(
        monkeypatch, [pursuit(), pursuit(identity="marvin-2")], identities=["marvin-1", "marvin-2"], max_actions=3,
    )
    assert result["reason"] == "find_marvin_identity_changed"
    assert result["actions_executed"] == len(steps) == 1 and len(evaluations) == 2


def test_invalid_limits_and_invalid_provider_never_call_a_step(monkeypatch):
    for limit in (0, -1, True, 1.5, "6", None):
        manager = BehaviorManager(robot_client=object())
        calls = []
        monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
        result = manager.execute_find_marvin_controller(lambda: evidence(), max_actions=limit)
        assert result["reason"] == "invalid_find_marvin_action_limit" and calls == []
    manager = BehaviorManager(robot_client=object())
    assert manager.execute_find_marvin_controller(None)["reason"] == "find_marvin_state_provider_unavailable"


def test_malformed_or_exceptional_state_provider_fails_closed(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
    assert manager.execute_find_marvin_controller(lambda: None)["reason"] == "find_marvin_state_evidence_malformed"
    def broken_provider():
        raise RuntimeError("offline")
    assert manager.execute_find_marvin_controller(broken_provider)["reason"] == "find_marvin_state_provider_exception"
    assert calls == []


def test_pursuit_evaluation_exception_fails_closed_without_a_step(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_pursuit_state",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
    result = manager.execute_find_marvin_controller(lambda: evidence())
    assert result["reason"] == "find_marvin_pursuit_evaluation_exception"
    assert result["actions_executed"] == 0 and calls == []


def test_deterministic_history_and_inputs_not_mutated(monkeypatch):
    fixture = evidence()
    before = deepcopy(fixture)
    manager = BehaviorManager(robot_client=object())
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", lambda *a, **k: pursuit())
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: successful_step())
    first = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    second = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    assert first == second and fixture == before


def test_controller_source_owns_no_primitive_or_arrival_integration():
    source = open("behavior_manager.py", encoding="utf-8").read()
    start = source.index("    def execute_find_marvin_controller(")
    end = source.index("    def execute_marvin_pursuit_step(", start)
    controller = source[start:end]
    for forbidden in (
        "robot.local_forward", "execute_guarded_turn",
        "execute_local_obstacle_avoidance_step",
        "execute_local_obstacle_avoidance_loop", "arrived_at_marvin",
        "self._execute_find", "self.execute_behavior",
    ):
        assert forbidden not in controller
    assert controller.count("self.execute_marvin_pursuit_step(") == 1
