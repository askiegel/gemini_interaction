"""Offline Find Marvin contracts for the generic local-progress handoff."""

from datetime import datetime, timezone
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import behavior_manager as behavior_module
from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime


IDENTITY = "marvin-identity-1"


def _evidence():
    return {
        "preview_result": {"target": "marvin", "fresh": True},
        "target_lock_result": {"found": True, "identity_id": IDENTITY},
        "target_lock_snapshot": {
            "tracking_mode": "LOCKED",
            "locked_identity_id": IDENTITY,
        },
        "selected_identity_id": IDENTITY,
    }


def _pursuit(state="READY_TO_APPROACH"):
    return {
        "ok": True,
        "state": state,
        "pursuit_authorized": state in {"READY_TO_APPROACH", "VISUAL_READY_TO_APPROACH"},
        "selected_identity_id": IDENTITY,
        "entity_id": "marvin-entity-1",
        "fresh": True,
        "geometry_usable": True,
    }


def _progress(terminal="LOCAL_PROGRESS_COMPLETE", *, actions=1, mode="DIRECT_FORWARD"):
    return {
        "ok": terminal == "LOCAL_PROGRESS_COMPLETE",
        "action": "local_progress_with_avoidance",
        "terminal_state": terminal,
        "mode": mode,
        "physical_actions": actions,
        "max_physical_actions": 4,
        "bridge_stopped": terminal != "LOCAL_PROGRESS_OWNERSHIP_REJECTED",
        "reason": terminal.lower(),
        "nested_result": {
            "terminal_state": "PATH_CLEAR" if terminal == "LOCAL_PROGRESS_COMPLETE" else terminal,
            "physical_actions": actions,
            "history": [{"decision": "FORWARD_CLEAR"}] * actions,
        },
    }


def _controller(monkeypatch, *, states, pursuit_steps, max_actions=6,
                search_steps=None, stop=None):
    manager = BehaviorManager(robot_client=object())
    state_values = iter(states)
    step_values = iter(pursuit_steps)
    search_values = iter(search_steps or [])
    provider_calls = []
    pursuit_calls = []
    search_calls = []
    stops = []

    def provider():
        provider_calls.append(True)
        return _evidence()

    def evaluate(*_args, **_kwargs):
        return _pursuit(next(state_values))

    def arrival(*_args, **kwargs):
        return {
            "ok": True,
            "arrived_at_marvin": False,
            "selected_identity_id": kwargs.get("selected_identity_id", IDENTITY),
            "fresh": True,
        }

    def pursuit_step(*_args, **kwargs):
        pursuit_calls.append(kwargs)
        return next(step_values)

    def search_step(*_args, **_kwargs):
        search_calls.append(True)
        return next(search_values)

    monkeypatch.setattr(behavior_module, "evaluate_marvin_pursuit_state", evaluate)
    monkeypatch.setattr(behavior_module, "evaluate_marvin_arrival", arrival)
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", pursuit_step)
    monkeypatch.setattr(manager, "execute_marvin_search_step", search_step)
    stop_callback = stop or (lambda: stops.append(True) or {"ok": True})
    result = manager.execute_find_marvin_controller(
        provider,
        max_actions=max_actions,
        stop_after_action=stop_callback,
    )
    return result, provider_calls, pursuit_calls, search_calls, stops


@pytest.mark.parametrize(
    ("terminal", "actions", "stop_expected"),
    [
        ("LOCAL_PROGRESS_BLOCKED", 0, 1),
        ("LOCAL_PROGRESS_SAFETY_VETO", 0, 1),
        ("LOCAL_PROGRESS_EXECUTION_FAILED", 2, 1),
        ("LOCAL_PROGRESS_OWNERSHIP_REJECTED", 0, 0),
        ("LOCAL_PROGRESS_MAX_STEPS_REACHED", 4, 1),
    ],
)
def test_handoff_terminals_propagate_without_another_controller_action(
    monkeypatch, terminal, actions, stop_expected,
):
    result, providers, pursuits, searches, stops = _controller(
        monkeypatch,
        states=["READY_TO_APPROACH"],
        pursuit_steps=[{
            "ok": True,
            "decision": "local_progress_handoff",
            "motion_executed": actions > 0,
            "replan_required": False,
            "local_progress_terminal": terminal,
            "local_progress_physical_actions": actions,
            "local_progress_result": _progress(terminal, actions=actions),
            "stop_required": terminal != "LOCAL_PROGRESS_OWNERSHIP_REJECTED",
        }],
    )

    assert result["reason"] == "marvin_local_progress_terminal"
    assert result["local_progress_terminal"] == terminal
    assert result["actions_executed"] == actions
    assert len(providers) == 1
    assert len(pursuits) == 1
    assert searches == []
    assert len(stops) == stop_expected


def test_successful_handoff_counts_nested_actions_then_rechecks_identity(monkeypatch):
    step = {
        "ok": True,
        "decision": "local_progress_handoff",
        "executed_primitive": "local_progress_with_avoidance",
        "motion_executed": True,
        "replan_required": True,
        "local_progress_terminal": "LOCAL_PROGRESS_COMPLETE",
        "local_progress_physical_actions": 3,
        "local_progress_result": _progress(
            actions=3, mode="BOUNDED_AVOIDANCE",
        ),
    }
    result, providers, pursuits, searches, stops = _controller(
        monkeypatch,
        states=["READY_TO_APPROACH", "REACQUIRE_REQUIRED"],
        pursuit_steps=[step],
    )

    assert result["reason"] == "marvin_local_progress_complete"
    assert result["completed"] is False
    assert result["arrived_at_marvin"] is False
    assert result["actions_executed"] == 3
    assert result["post_progress_pursuit_state"] == "REACQUIRE_REQUIRED"
    assert len(providers) == 2
    assert len(pursuits) == 1
    assert searches == []
    assert len(stops) == 1
    assert result["history"][-1]["selected_action"] == "perception_reassessment"


def test_four_nested_actions_fit_budget_and_force_fresh_identity_recheck(monkeypatch):
    successful_search = {
        "ok": True,
        "motion_executed": True,
        "replan_required": True,
        "search_action": "turn_left",
    }
    terminal_step = {
        "ok": True,
        "decision": "local_progress_handoff",
        "motion_executed": True,
        "replan_required": True,
        "local_progress_terminal": "LOCAL_PROGRESS_COMPLETE",
        "local_progress_physical_actions": 4,
        "local_progress_result": _progress(
            "LOCAL_PROGRESS_COMPLETE", actions=4,
            mode="BOUNDED_AVOIDANCE",
        ),
    }
    result, _providers, pursuits, searches, stops = _controller(
        monkeypatch,
        states=["SEARCHING", "SEARCHING", "READY_TO_APPROACH", "READY_TO_APPROACH"],
        pursuit_steps=[terminal_step],
        search_steps=[successful_search, successful_search],
    )

    assert len(searches) == 2
    assert len(pursuits) == 1
    assert pursuits[0]["local_progress_action_budget_remaining"] == 4
    assert result["actions_executed"] == 6
    assert result["reason"] == "marvin_local_progress_complete"
    assert result["completed"] is False
    assert len(_providers) == 4
    assert len(stops) == 3


def test_pursuit_step_invokes_only_the_handoff_after_forward_is_authorized(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.world_model = object()
    manager._current_lidar_session = lambda: "producer-session"
    callback = Mock(return_value=_progress(actions=1))
    manager.marvin_local_progress_with_avoidance_handler = callback
    monkeypatch.setattr(
        behavior_module,
        "evaluate_marvin_pursuit_state",
        lambda *_args, **_kwargs: _pursuit("READY_TO_APPROACH"),
    )

    result = manager.execute_marvin_pursuit_step(
        {}, {}, {}, selected_identity_id=IDENTITY,
        local_progress_action_budget_remaining=6,
    )

    callback.assert_called_once_with(remaining_actions=6)
    assert result["local_progress_terminal"] == "LOCAL_PROGRESS_COMPLETE"
    assert result["local_progress_physical_actions"] == 1
    assert result["motion_executed"] is True
    assert result["replan_required"] is True


def test_insufficient_remaining_budget_does_not_start_handoff(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.world_model = object()
    manager._current_lidar_session = lambda: "producer-session"
    callback = Mock()
    manager.marvin_local_progress_with_avoidance_handler = callback
    monkeypatch.setattr(
        behavior_module,
        "evaluate_marvin_pursuit_state",
        lambda *_args, **_kwargs: _pursuit("READY_TO_APPROACH"),
    )

    result = manager.execute_marvin_pursuit_step(
        {}, {}, {}, selected_identity_id=IDENTITY,
        local_progress_action_budget_remaining=3,
    )

    callback.assert_not_called()
    assert result["local_progress_budget_blocked"] is True
    assert result["motion_executed"] is False


def test_find_marvin_runtime_authorization_is_mission_thread_and_generation_scoped():
    runtime = object.__new__(CognitiveRuntime)
    mission = SimpleNamespace(
        mission_type="FIND_OBJECT", target="marvin", mission_id="mission-7",
    )
    runtime._state_lock = threading.RLock()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: mission)
    runtime._control_generation = 9
    runtime._behavior_execution_generation = 9
    runtime._behavior_execution_thread_id = threading.get_ident()
    runtime._find_marvin_progress_context = threading.local()
    runtime.MAX_REACTIVE_STEPS = 4
    calls = []

    def run_handoff():
        calls.append(runtime._active_localization_has_behavior_owner())
        return _progress(actions=1)

    runtime.run_local_progress_with_avoidance = run_handoff

    result = runtime._run_find_marvin_local_progress(remaining_actions=6)

    assert result["terminal_state"] == "LOCAL_PROGRESS_COMPLETE"
    assert calls == [False]


def test_unrelated_caller_cannot_authorize_find_marvin_handoff():
    runtime = object.__new__(CognitiveRuntime)
    mission = SimpleNamespace(
        mission_type="FIND_OBJECT", target="backpack", mission_id="other",
    )
    runtime._state_lock = threading.RLock()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: mission)
    runtime._control_generation = 2
    runtime._behavior_execution_generation = 2
    runtime._behavior_execution_thread_id = threading.get_ident()
    runtime._find_marvin_progress_context = threading.local()
    runtime.MAX_REACTIVE_STEPS = 4
    runtime.run_local_progress_with_avoidance = Mock()

    result = runtime._run_find_marvin_local_progress(remaining_actions=6)

    assert result["terminal_state"] == "LOCAL_PROGRESS_OWNERSHIP_REJECTED"
    runtime.run_local_progress_with_avoidance.assert_not_called()


@pytest.mark.parametrize("ownership_failure", ["generation", "thread"])
def test_find_marvin_handoff_rejects_wrong_generation_or_execution_thread(
    ownership_failure,
):
    runtime = object.__new__(CognitiveRuntime)
    mission = SimpleNamespace(
        mission_type="FIND_OBJECT", target="marvin", mission_id="mission-8",
    )
    runtime._state_lock = threading.RLock()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: mission)
    runtime._control_generation = 4
    runtime._behavior_execution_generation = 3 if ownership_failure == "generation" else 4
    runtime._behavior_execution_thread_id = (
        threading.get_ident() + 100 if ownership_failure == "thread"
        else threading.get_ident()
    )
    runtime._find_marvin_progress_context = threading.local()
    runtime.MAX_REACTIVE_STEPS = 4
    runtime.run_local_progress_with_avoidance = Mock()

    result = runtime._run_find_marvin_local_progress(remaining_actions=6)

    assert result["terminal_state"] == "LOCAL_PROGRESS_OWNERSHIP_REJECTED"
    runtime.run_local_progress_with_avoidance.assert_not_called()


@pytest.mark.parametrize(
    ("terminal", "expected_state", "expected_ok"),
    [
        ("LOCAL_PROGRESS_BLOCKED", "FIND_MARVIN_SAFE_INCOMPLETE", True),
        ("LOCAL_PROGRESS_SAFETY_VETO", "FIND_MARVIN_SAFE_INCOMPLETE", True),
        ("LOCAL_PROGRESS_MAX_STEPS_REACHED", "FIND_MARVIN_SAFE_INCOMPLETE", True),
        ("LOCAL_PROGRESS_OWNERSHIP_REJECTED", "FIND_MARVIN_SAFE_INCOMPLETE", True),
        ("LOCAL_PROGRESS_EXECUTION_FAILED", "FIND_MARVIN_FAILED", False),
    ],
)
def test_normal_mission_surfaces_handoff_terminal_without_retry(
    terminal, expected_state, expected_ok,
):
    runtime = object.__new__(CognitiveRuntime)
    mission = SimpleNamespace(
        mission_type="FIND_OBJECT", target="marvin", mission_id="mission-11",
    )
    runtime._state_lock = threading.RLock()
    runtime._marvin_controller_lock = threading.RLock()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: mission)
    runtime._control_generation = 3
    runtime.running = True
    episode_calls = []
    controller = {
        "ok": True,
        "completed": False,
        "arrived_at_marvin": False,
        "reason": "marvin_local_progress_terminal",
        "local_progress_terminal": terminal,
        "local_progress_result": _progress(terminal, actions=2),
        "actions_executed": 2,
    }
    runtime._execute_bounded_find_marvin_episode = lambda **_kwargs: (
        episode_calls.append(True)
        or {
            "ok": True,
            "execution_authorized": True,
            "actions_executed": 2,
            "motion_executed": True,
            "controller_result": controller,
        }
    )

    result = runtime._execute_normal_marvin_find_mission_locked(
        mission, control_generation=3,
    )

    assert result["ok"] is expected_ok
    assert result["completed"] is True
    assert result["arrived_at_marvin"] is False
    assert result["state"] == expected_state
    assert result["local_progress_terminal"] == terminal
    assert len(episode_calls) == 1
