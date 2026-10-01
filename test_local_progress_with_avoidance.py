"""Offline tests for the high-level local-progress handoff coordinator."""

import threading
from unittest.mock import Mock

import pytest

import runtime as runtime_module
from runtime import CognitiveRuntime


SESSION = "test-lidar-session"


def _decision(name, reason="test_decision"):
    return {"decision": name, "reason": reason, "forward": {"permitted": name == "FORWARD_CLEAR"}}


def _single_step(
    decision="FORWARD_CLEAR", *, action_attempted=True, action_executed=True,
    bridge_stopped=True, reason="local_reactive_action_complete",
):
    return {
        "decision": decision,
        "action_attempted": action_attempted,
        "action_executed": action_executed,
        "bridge_stopped": bridge_stopped,
        "reason": reason,
    }


def _bounded(terminal_state, *, physical_actions=0, bridge_stopped=True,
             reason="bounded_test_result", history=None):
    return {
        "terminal_state": terminal_state,
        "physical_actions": physical_actions,
        "bridge_stopped": bridge_stopped,
        "reason": reason,
        "history": history if history is not None else [{"decision": "test_only"}],
    }


def _runtime():
    runtime = object.__new__(CognitiveRuntime)
    runtime.running = True
    runtime._local_progress_with_avoidance_lock = threading.Lock()
    runtime._local_reactive_lidar_state = Mock(return_value=(SESSION, {"fresh": True}))
    runtime.run_local_reactive_step = Mock()
    runtime.run_bounded_local_reactive_avoidance = Mock()
    return runtime


def test_clear_path_uses_exactly_one_direct_single_step(monkeypatch):
    runtime = _runtime()
    runtime.run_local_reactive_step.return_value = _single_step()
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("FORWARD_CLEAR"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["terminal_state"] == "LOCAL_PROGRESS_COMPLETE"
    assert result["mode"] == "DIRECT_FORWARD"
    assert result["physical_actions"] == 1
    runtime.run_local_reactive_step.assert_called_once_with()
    runtime.run_bounded_local_reactive_avoidance.assert_not_called()


def test_turnable_initial_scene_hands_off_without_a_pre_handoff_turn(monkeypatch):
    runtime = _runtime()
    runtime.run_bounded_local_reactive_avoidance.return_value = _bounded(
        "PATH_CLEAR", physical_actions=3,
        history=[{"decision": "FORWARD_CLEAR", "action_executed": True}],
    )
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("TURN_RIGHT"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["terminal_state"] == "LOCAL_PROGRESS_COMPLETE"
    assert result["mode"] == "BOUNDED_AVOIDANCE"
    assert result["physical_actions"] == 3
    assert result["nested_result"]["history"][0]["decision"] == "FORWARD_CLEAR"
    runtime.run_local_reactive_step.assert_not_called()
    # The initial TURN_RIGHT is routing-only: no direction is supplied to the
    # bounded coordinator, whose first decision must be freshly sensed.
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()


def test_initial_stop_blocked_returns_without_any_lower_level_action(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("STOP_BLOCKED", "no_safe_local_action"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["ok"] is True
    assert result["terminal_state"] == "LOCAL_PROGRESS_BLOCKED"
    assert result["mode"] == "NO_MOTION"
    assert result["physical_actions"] == 0
    runtime.run_local_reactive_step.assert_not_called()
    runtime.run_bounded_local_reactive_avoidance.assert_not_called()


@pytest.mark.parametrize(
    ("bounded_terminal", "expected_terminal", "ok"),
    [
        ("SAFETY_VETO", "LOCAL_PROGRESS_SAFETY_VETO", False),
        ("BLOCKED", "LOCAL_PROGRESS_BLOCKED", True),
        ("OWNERSHIP_REJECTED", "LOCAL_PROGRESS_OWNERSHIP_REJECTED", False),
        ("EXECUTION_FAILED", "LOCAL_PROGRESS_EXECUTION_FAILED", False),
    ],
)
def test_bounded_fail_closed_terminals_propagate_without_retry(
    monkeypatch, bounded_terminal, expected_terminal, ok,
):
    runtime = _runtime()
    runtime.run_bounded_local_reactive_avoidance.return_value = _bounded(bounded_terminal)
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("TURN_LEFT"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["terminal_state"] == expected_terminal
    assert result["ok"] is ok
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()
    runtime.run_local_reactive_step.assert_not_called()


def test_max_steps_propagates_four_actions_without_a_fifth(monkeypatch):
    runtime = _runtime()
    runtime.run_bounded_local_reactive_avoidance.return_value = _bounded(
        "MAX_STEPS_REACHED", physical_actions=4,
    )
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("TURN_LEFT"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["ok"] is True
    assert result["terminal_state"] == "LOCAL_PROGRESS_MAX_STEPS_REACHED"
    assert result["physical_actions"] == 4
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()
    runtime.run_local_reactive_step.assert_not_called()


@pytest.mark.parametrize("initial_decision", ["FORWARD_CLEAR", "TURN_LEFT"])
def test_unlocalized_mapless_cameraless_paths_need_only_local_lidar(monkeypatch, initial_decision):
    runtime = _runtime()
    if initial_decision == "FORWARD_CLEAR":
        runtime.run_local_reactive_step.return_value = _single_step()
    else:
        runtime.run_bounded_local_reactive_avoidance.return_value = _bounded(
            "PATH_CLEAR", physical_actions=2,
        )
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision(initial_decision))

    result = runtime.run_local_progress_with_avoidance()

    assert result["terminal_state"] == "LOCAL_PROGRESS_COMPLETE"
    # This intentionally minimal fake has no localization, AMCL, map, Nav2,
    # Home, camera, or semantic-identity attributes.


def test_handoff_lock_rejects_overlap_without_stopping_or_delegating(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("FORWARD_CLEAR"))
    runtime._local_progress_with_avoidance_lock.acquire()
    try:
        result = runtime.run_local_progress_with_avoidance()
    finally:
        runtime._local_progress_with_avoidance_lock.release()

    assert result["terminal_state"] == "LOCAL_PROGRESS_OWNERSHIP_REJECTED"
    runtime.run_local_reactive_step.assert_not_called()
    runtime.run_bounded_local_reactive_avoidance.assert_not_called()


def test_direct_jit_veto_propagates_without_bounded_retry(monkeypatch):
    runtime = _runtime()
    runtime.run_local_reactive_step.return_value = _single_step(
        action_executed=False,
        reason="local_reactive_executor_vetoed",
    )
    monkeypatch.setattr(runtime_module, "decide_forward_reaction",
                        lambda *_args, **_kwargs: _decision("FORWARD_CLEAR"))

    result = runtime.run_local_progress_with_avoidance()

    assert result["terminal_state"] == "LOCAL_PROGRESS_SAFETY_VETO"
    assert result["physical_actions"] == 0
    runtime.run_local_reactive_step.assert_called_once_with()
    runtime.run_bounded_local_reactive_avoidance.assert_not_called()
