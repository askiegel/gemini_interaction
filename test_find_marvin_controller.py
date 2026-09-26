"""Offline bounded-controller contracts for Find-Marvin routing."""

from copy import deepcopy

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager


def evidence(identity="marvin-1"):
    return {"preview_result": {"preview": "fresh"}, "target_lock_result": {"lock": "fresh"},
            "target_lock_snapshot": {"snapshot": "fresh"}, "selected_identity_id": identity}


def pursuit(state="READY_TO_APPROACH", authorized=True, identity="marvin-1"):
    return {"ok": True, "state": state, "pursuit_authorized": authorized,
            "selected_identity_id": identity, "entity_id": "entity-1", "fresh": True,
            "geometry_usable": True}


def successful_pursuit():
    return {"ok": True, "decision": "approach_forward", "executed_primitive": "forward",
            "motion_executed": True, "replan_required": True}


def successful_search(action="turn_left"):
    return {"ok": True, "decision": "search_turn", "search_action": action,
            "executed_primitive": "guarded_turn_left", "motion_executed": True,
            "replan_required": True, "planner": {"selected_search_action": action}}


def not_arrived(identity="marvin-1"):
    return {"ok": True, "arrived_at_marvin": False,
            "selected_identity_id": identity}


def arrived(identity="marvin-1"):
    return {"ok": True, "arrived_at_marvin": True,
            "selected_identity_id": identity, "identity_authorized": True,
            "fresh": True, "geometry_valid": True,
            "reason": "marvin_visual_standoff_reached"}


def invoke(monkeypatch, states, *, search_steps=None, pursuit_steps=None, max_actions=6,
           identities=None, arrivals=None):
    manager = BehaviorManager(robot_client=object())
    provider_calls, evaluator_calls, arrival_calls, search_calls, pursuit_calls = [], [], [], [], []
    state_values = iter(states)
    identity_values = iter(identities or ["marvin-1"] * len(states))
    search_values = iter(search_steps or [successful_search()] * max_actions)
    pursuit_values = iter(pursuit_steps or [successful_pursuit()] * max_actions)
    arrival_values = iter(arrivals) if arrivals is not None else None

    def provider():
        provider_calls.append(True)
        return evidence(next(identity_values))

    def evaluate(*args, **kwargs):
        evaluator_calls.append((args, kwargs))
        value = next(state_values)
        if isinstance(value, Exception):
            raise value
        return value

    def search(*args, **kwargs):
        search_calls.append((args, kwargs))
        value = next(search_values)
        if isinstance(value, Exception):
            raise value
        return value

    def pursuit_step(*args, **kwargs):
        pursuit_calls.append((args, kwargs))
        value = next(pursuit_values)
        if isinstance(value, Exception):
            raise value
        return value

    def arrival(*args, **kwargs):
        arrival_calls.append((args, kwargs))
        if arrival_values is None:
            return not_arrived(kwargs.get("selected_identity_id"))
        value = next(arrival_values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", evaluate)
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_arrival", arrival)
    monkeypatch.setattr(manager, "execute_marvin_search_step", search)
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", pursuit_step)
    result = manager.execute_find_marvin_controller(provider, max_actions=max_actions, now=10.0)
    return result, provider_calls, evaluator_calls, arrival_calls, search_calls, pursuit_calls


def test_searching_and_reacquire_route_only_to_search(monkeypatch):
    for state in ("SEARCHING", "REACQUIRE_REQUIRED"):
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit(state, False)], max_actions=1)
        assert len(searches) == result["actions_executed"] == 1
        assert pursuits == [] and result["reason"] == "find_marvin_action_limit_reached"


def test_ready_authorized_routes_only_to_pursuit(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, [pursuit()], max_actions=1)
    assert searches == [] and len(pursuits) == result["actions_executed"] == 1


def test_arrival_stops_before_any_search_or_pursuit_executor(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit()], arrivals=[arrived()], max_actions=1)
    assert result["ok"] is result["completed"] is result["arrived_at_marvin"] is True
    assert result["reason"] == "arrived_at_marvin"
    assert result["actions_executed"] == 0
    assert len(providers) == len(evaluations) == len(arrivals) == 1
    assert searches == pursuits == []
    assert result["history"][-1]["route"] == "arrival"


def test_arrival_claim_in_searching_state_fails_closed_without_motion(monkeypatch):
    result, _providers, _evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False)], arrivals=[arrived()], max_actions=1)
    assert len(arrivals) == 1 and searches == pursuits == []
    assert result["reason"] == "marvin_arrival_evaluation_inconsistent"
    assert result["actions_executed"] == 0


def test_nonrouting_states_execute_no_executor(monkeypatch):
    for state in ("CANDIDATE_SEEN", "MARVIN_LOCKED", "SAME_IDENTITY_REACQUIRED", "INSUFFICIENT_EVIDENCE"):
        result, providers, evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit(state=state, authorized=False)])
        assert result["actions_executed"] == 0 and len(providers) == len(evaluations) == 1
        assert searches == pursuits == []


def test_search_then_ready_routes_one_executor_per_fresh_iteration(monkeypatch):
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit()], max_actions=2)
    assert len(providers) == len(evaluations) == 2
    assert len(searches) == len(pursuits) == 1
    assert [entry["route"] for entry in result["history"]] == ["search", "pursuit"]
    assert result["actions_executed"] == 2


def test_arrival_after_pursuit_uses_fresh_second_iteration_without_extra_action(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit()], arrivals=[not_arrived(), arrived()], max_actions=2)
    assert len(providers) == len(evaluations) == len(arrivals) == 2
    assert searches == [] and len(pursuits) == result["actions_executed"] == 1
    assert result["completed"] is result["arrived_at_marvin"] is True


def test_arrival_after_search_then_pursuit_stops_before_next_action(monkeypatch):
    result, _providers, _evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit(), pursuit()],
        arrivals=[not_arrived(), not_arrived(), arrived()], max_actions=3)
    assert len(arrivals) == 3 and len(searches) == len(pursuits) == 1
    assert result["actions_executed"] == 2 and result["reason"] == "arrived_at_marvin"


def test_pursuit_then_reacquire_routes_to_search(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit("REACQUIRE_REQUIRED", False)], max_actions=2)
    assert len(searches) == len(pursuits) == 1
    assert [entry["route"] for entry in result["history"]] == ["pursuit", "search"]


def test_candidate_or_bridge_only_stops_without_follow_on_motion(monkeypatch):
    for state in ("CANDIDATE_SEEN", "SAME_IDENTITY_REACQUIRED"):
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit("SEARCHING", False), pursuit(state, False)], max_actions=3)
        assert len(searches) == 1 and pursuits == []
        assert result["actions_executed"] == 1


def test_identity_change_between_search_and_pursuit_fails_closed(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit(identity="marvin-2")],
        identities=["marvin-1", "marvin-2"], max_actions=3)
    assert result["reason"] == "find_marvin_identity_changed"
    assert len(searches) == 1 and pursuits == []


def test_global_budget_is_shared_across_search_and_pursuit(monkeypatch):
    states = [pursuit("SEARCHING", False), pursuit("SEARCHING", False), pursuit(), pursuit()]
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, states, max_actions=4)
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert len(searches) == len(pursuits) == 2
    assert result["actions_executed"] == len(searches) + len(pursuits) == 4


def test_default_budget_stops_an_alternating_stream_at_six(monkeypatch):
    states = [pursuit("SEARCHING", False), pursuit()] * 3
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, states)
    assert result["max_actions"] == result["actions_executed"] == 6
    assert len(providers) == len(evaluations) == len(searches) + len(pursuits) == 6


def test_arrival_check_does_not_consume_budget_or_add_final_observation(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit()], max_actions=1)
    assert result["actions_executed"] == len(pursuits) == 1 and searches == []
    assert len(providers) == len(evaluations) == len(arrivals) == 1
    assert result["reason"] == "find_marvin_action_limit_reached"


def test_executor_exceptions_or_failures_have_no_fallback(monkeypatch):
    cases = (
        (pursuit("SEARCHING", False), [RuntimeError("offline")], None, "find_marvin_search_step_exception"),
        (pursuit("SEARCHING", False), [{"ok": False}], None, "find_marvin_search_step_failed"),
        (pursuit(), None, [RuntimeError("offline")], "find_marvin_pursuit_step_exception"),
        (pursuit(), None, [{"ok": False}], "find_marvin_pursuit_step_failed"),
    )
    for state, search_steps, pursuit_steps, reason in cases:
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [state], search_steps=search_steps, pursuit_steps=pursuit_steps)
        assert result["reason"] == reason and result["actions_executed"] == 1
        assert len(searches) + len(pursuits) == 1


def test_arrival_evaluator_exception_or_malformed_result_stops_before_motion(monkeypatch):
    cases = (
        [RuntimeError("offline")], [None],
        [{"ok": False, "arrived_at_marvin": False, "selected_identity_id": "marvin-1"}],
        [{"ok": True, "arrived_at_marvin": True, "selected_identity_id": "other"}],
    )
    for arrivals in cases:
        result, _providers, _evaluations, arrival_calls, searches, pursuits = invoke(
            monkeypatch, [pursuit()], arrivals=arrivals, max_actions=1)
        assert len(arrival_calls) == 1 and searches == pursuits == []
        assert result["actions_executed"] == 0
        assert result["reason"] in {"marvin_arrival_evaluation_failed", "marvin_arrival_evaluation_inconsistent"}


def test_arrival_receives_no_lidar_authority(monkeypatch):
    _result, _providers, _evaluations, calls, _searches, _pursuits = invoke(
        monkeypatch, [pursuit()], max_actions=1)
    assert len(calls) == 1
    assert set(calls[0][1]) == {"selected_identity_id", "now"}


def test_no_motion_replan_false_and_search_complete_stop(monkeypatch):
    cases = (
        (pursuit("SEARCHING", False), [{"ok": True, "motion_executed": False}], None, "find_marvin_search_step_no_motion"),
        (pursuit("SEARCHING", False), [{"ok": True, "search_action": "turn_left", "motion_executed": True, "replan_required": False}], None, "find_marvin_search_step_replan_required"),
        (pursuit("SEARCHING", False), [{"ok": True, "search_action": "search_complete", "motion_executed": False}], None, "find_marvin_search_complete"),
        (pursuit(), None, [{"ok": True, "motion_executed": False}], "find_marvin_pursuit_step_no_motion"),
        (pursuit(), None, [{"ok": True, "motion_executed": True, "replan_required": False}], "find_marvin_pursuit_step_replan_required"),
    )
    for state, search_steps, pursuit_steps, reason in cases:
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [state], search_steps=search_steps, pursuit_steps=pursuit_steps)
        assert result["reason"] == reason and len(searches) + len(pursuits) == 1


def test_invalid_limits_provider_and_evaluator_fail_closed(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    monkeypatch.setattr(manager, "execute_marvin_search_step", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
    for limit in (0, -1, True, 1.5, "6", None):
        assert manager.execute_find_marvin_controller(lambda: evidence(), max_actions=limit)["reason"] == "invalid_find_marvin_action_limit"
    assert manager.execute_find_marvin_controller(None)["reason"] == "find_marvin_state_provider_unavailable"
    assert manager.execute_find_marvin_controller(lambda: None)["reason"] == "find_marvin_state_evidence_malformed"
    assert manager.execute_find_marvin_controller(
        lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )["reason"] == "find_marvin_state_provider_exception"
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert manager.execute_find_marvin_controller(lambda: evidence())["reason"] == "find_marvin_pursuit_evaluation_exception"
    assert calls == []


def test_search_history_advances_only_for_successful_turns(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, _pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit("SEARCHING", False)], max_actions=2)
    assert searches[0][1]["prior_search_history"] == []
    assert searches[1][1]["prior_search_history"] == [{"selected_search_action": "turn_left"}]
    assert result["history"][0]["search_action"] == "turn_left"


def test_deterministic_history_and_inputs_not_mutated(monkeypatch):
    fixture = evidence()
    before = deepcopy(fixture)
    manager = BehaviorManager(robot_client=object())
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", lambda *a, **k: pursuit())
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: successful_pursuit())
    first = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    second = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    assert first == second and fixture == before


def test_controller_source_delegates_arrival_without_geometry_or_motion_logic():
    source = open("behavior_manager.py", encoding="utf-8").read()
    start = source.index("    def execute_find_marvin_controller(")
    end = source.index("    def execute_marvin_pursuit_step(", start)
    controller = source[start:end]
    for forbidden in ("robot.local_forward", "execute_guarded_turn", "execute_local_obstacle_avoidance_step",
                      "execute_local_obstacle_avoidance_loop", "0.545833", "0.160339", "bbox_width",
                      "height_fraction", "area_fraction", "lidar", "while true",
                      "self._execute_find", "self.execute_behavior"):
        assert forbidden not in controller.lower()
    assert controller.count("self.execute_marvin_search_step(") == 1
    assert controller.count("self.execute_marvin_pursuit_step(") == 1
    assert controller.count("evaluate_marvin_arrival(") == 1
