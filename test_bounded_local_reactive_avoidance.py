"""Offline tests for four-cycle local reactive avoidance orchestration."""

import threading

import pytest

from runtime import CognitiveRuntime


def _step(
    decision,
    *,
    action_attempted=False,
    action_executed=False,
    bridge_stopped=True,
    reason=None,
):
    return {
        "decision": decision,
        "action_attempted": action_attempted,
        "action_executed": action_executed,
        "bridge_stopped": bridge_stopped,
        "reason": reason or (
            "local_reactive_action_complete"
            if action_executed else "local_reactive_executor_vetoed"
        ),
        # Deliberately large/raw-looking data that must not be copied into the
        # bounded episode history.
        "decision_evidence": {"raw_lidar": "not-in-history"},
        "executor_result": {"raw_transport": "not-in-history"},
    }


def _runtime(*steps):
    runtime = object.__new__(CognitiveRuntime)
    runtime.running = True
    runtime._bounded_local_reactive_avoidance_lock = threading.Lock()
    calls = []
    sequence = iter(steps)

    def run_step():
        calls.append(len(calls) + 1)
        return next(sequence)

    runtime.run_local_reactive_step = run_step
    return runtime, calls


def test_initially_clear_path_executes_one_forward_and_stops_episode():
    runtime, calls = _runtime(_step(
        "FORWARD_CLEAR", action_attempted=True, action_executed=True,
    ))

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "PATH_CLEAR"
    assert result["steps_attempted"] == result["physical_actions"] == 1
    assert result["forward_steps_executed"] == 1
    assert result["turns_executed"] == 0
    assert result["bridge_stopped"] is True
    assert calls == [1]


@pytest.mark.parametrize("turn", ["TURN_LEFT", "TURN_RIGHT"])
def test_blocked_turn_then_fresh_forward_ends_path_clear(turn):
    runtime, calls = _runtime(
        _step(turn, action_attempted=True, action_executed=True),
        _step("FORWARD_CLEAR", action_attempted=True, action_executed=True),
    )

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "PATH_CLEAR"
    assert result["steps_attempted"] == 2
    assert result["physical_actions"] == 2
    assert result["turns_executed"] == 1
    assert result["forward_steps_executed"] == 1
    assert [entry["decision"] for entry in result["history"]] == [
        turn, "FORWARD_CLEAR",
    ]
    # A second single-step call is the required fresh sense/decision boundary.
    assert calls == [1, 2]


def test_stop_blocked_terminates_without_motion_or_reconsideration():
    runtime, calls = _runtime(_step("STOP_BLOCKED", reason="local_reactive_stop_blocked"))

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["ok"] is True
    assert result["terminal_state"] == "BLOCKED"
    assert result["physical_actions"] == 0
    assert result["steps_attempted"] == 1
    assert calls == [1]


@pytest.mark.parametrize("decision", ["TURN_RIGHT", "FORWARD_CLEAR"])
def test_jit_executor_veto_terminates_without_alternate_action(decision):
    runtime, calls = _runtime(_step(
        decision,
        action_attempted=True,
        action_executed=False,
        reason="local_reactive_executor_vetoed",
    ))

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "SAFETY_VETO"
    assert result["physical_actions"] == 0
    assert calls == [1]


def test_failed_bridge_zero_proof_terminates_before_any_next_step():
    runtime, calls = _runtime(_step(
        "TURN_LEFT", action_attempted=True, action_executed=True,
        bridge_stopped=False,
        reason="bridge_not_stopped_after_local_reactive_step",
    ))

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "EXECUTION_FAILED"
    assert result["physical_actions"] == 1
    assert result["bridge_stopped"] is False
    assert calls == [1]


def test_four_turns_are_the_hard_step_and_physical_action_bound():
    runtime, calls = _runtime(*[
        _step("TURN_LEFT" if index % 2 == 0 else "TURN_RIGHT",
              action_attempted=True, action_executed=True)
        for index in range(4)
    ])

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "MAX_STEPS_REACHED"
    assert result["steps_attempted"] == 4
    assert result["physical_actions"] == 4
    assert result["turns_executed"] == 4
    assert result["forward_steps_executed"] == 0
    assert result["max_steps"] == 4
    assert calls == [1, 2, 3, 4]


def test_preownership_rejection_does_not_continue_or_claim_bridge_zero():
    runtime, calls = _runtime(_step(
        None,
        bridge_stopped=False,
        reason="physical_behavior_already_active",
    ))

    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "OWNERSHIP_REJECTED"
    assert result["physical_actions"] == 0
    assert result["bridge_stopped"] is False
    assert calls == [1]


def test_episode_lock_rejects_overlapping_bounded_avoidance_without_step_call():
    runtime, calls = _runtime(_step(
        "FORWARD_CLEAR", action_attempted=True, action_executed=True,
    ))
    runtime._bounded_local_reactive_avoidance_lock.acquire()
    try:
        result = runtime.run_bounded_local_reactive_avoidance()
    finally:
        runtime._bounded_local_reactive_avoidance_lock.release()

    assert result["terminal_state"] == "OWNERSHIP_REJECTED"
    assert result["reason"] == "bounded_reactive_avoidance_already_running"
    assert calls == []


def test_unlocalized_mapless_cameraless_orchestration_uses_only_single_step_result():
    runtime, calls = _runtime(_step(
        "FORWARD_CLEAR", action_attempted=True, action_executed=True,
    ))
    # The object intentionally has no localization, map, camera, Nav2, or
    # motion-executor attributes.  Episode orchestration needs none of them.
    result = runtime.run_bounded_local_reactive_avoidance()

    assert result["terminal_state"] == "PATH_CLEAR"
    assert calls == [1]
    assert set(result["history"][0]) == {
        "step", "decision", "action_attempted", "action_executed",
        "reason", "bridge_stopped",
    }
