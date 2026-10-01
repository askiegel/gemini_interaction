"""Offline coverage for Find Object's runtime local-progress handoff."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime


def _target():
    return {
        "cx": 320.0,
        "cy": 240.0,
        "area": 12000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "confidence": 0.95,
        "bbox": {"x1": 260.0, "y1": 180.0, "x2": 380.0, "y2": 300.0},
        "source_timestamp": "synthetic-frame-1",
    }


def _progress(terminal="LOCAL_PROGRESS_COMPLETE", *, mode="DIRECT_FORWARD",
              physical_actions=1, nested=None):
    return {
        "ok": terminal == "LOCAL_PROGRESS_COMPLETE",
        "terminal_state": terminal,
        "mode": mode,
        "physical_actions": physical_actions,
        "reason": terminal.lower(),
        "nested_result": nested or {"terminal_state": "PATH_CLEAR"},
    }


def _manager(progress_results, confirmations=None):
    manager = object.__new__(BehaviorManager)
    manager.FIND_APPROACH_MAX_CHUNKS = 4
    manager.local_progress_with_avoidance_handler = Mock(
        side_effect=list(progress_results)
    )
    manager._execution_is_current = lambda: True
    manager._target_is_fresh_and_acquired = lambda _target: True
    manager._publish_tracking_state = lambda _result: None
    manager._vision_timestamp_is_iso = lambda _timestamp: False
    promoted = _target()
    manager._confirm_find_target_with_semantic = Mock(
        side_effect=list(confirmations or [(promoted, "confirmed", {"fresh": True})])
    )
    manager._promote_confirmed_target = lambda target: target
    return manager


def _run_approach(manager, initial=None):
    target = initial or _target()
    return manager._execute_find_object_approach(
        "backpack",
        target,
        {"behavior": "FIND_OBJECT"},
        centering_attempted=0,
        centering_completed=0,
        last_guarded_result=None,
        telemetry={},
    )


def test_direct_forward_handoff_counts_one_action_and_does_not_claim_arrival():
    manager = _manager([_progress()])
    manager.FIND_APPROACH_MAX_CHUNKS = 1

    result = _run_approach(manager)

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert result["ok"] is True
    assert result["physical_actions"] == 1
    assert result["local_progress_physical_actions"] == 1
    assert result["target_found"] is True
    assert result["state"] not in {"ARRIVED", "OBJECT_FOUND", "MISSION_COMPLETE"}
    manager.local_progress_with_avoidance_handler.assert_called_once_with()
    manager._confirm_find_target_with_semantic.assert_called_once()


def test_bounded_avoidance_counts_nested_actions_and_returns_after_perception():
    nested = {
        "terminal_state": "PATH_CLEAR",
        "physical_actions": 3,
        "history": [
            {"decision": "TURN_RIGHT"},
            {"decision": "TURN_RIGHT"},
            {"decision": "FORWARD_CLEAR"},
        ],
    }
    manager = _manager([_progress(mode="BOUNDED_AVOIDANCE", physical_actions=3, nested=nested)])

    result = _run_approach(manager)

    assert result["state"] == "APPROACH_AVOIDANCE_COMPLETE"
    assert result["ok"] is True
    assert result["completed"] is False
    assert result["physical_actions"] == 3
    assert result["approach_chunks_completed"] == 1
    assert result["target_found"] is True
    assert result["local_progress_result"]["nested_result"]["history"][-1]["decision"] == "FORWARD_CLEAR"
    manager.local_progress_with_avoidance_handler.assert_called_once_with()
    manager._confirm_find_target_with_semantic.assert_called_once()


@pytest.mark.parametrize(
    ("terminal", "expected_state", "actions"),
    [
        ("LOCAL_PROGRESS_BLOCKED", "APPROACH_BLOCKED", 0),
        ("LOCAL_PROGRESS_SAFETY_VETO", "APPROACH_BLOCKED", 0),
        ("LOCAL_PROGRESS_MAX_STEPS_REACHED", "APPROACH_BLOCKED", 4),
        ("LOCAL_PROGRESS_OWNERSHIP_REJECTED", "APPROACH_BLOCKED", 0),
        ("LOCAL_PROGRESS_EXECUTION_FAILED", "APPROACH_FAILED", 1),
    ],
)
def test_noncomplete_handoff_terminals_stop_without_retry(terminal, expected_state, actions):
    manager = _manager([_progress(terminal, mode="BOUNDED_AVOIDANCE", physical_actions=actions)])

    result = _run_approach(manager)

    assert result["state"] == expected_state
    assert result["ok"] is False
    assert result["physical_actions"] == actions
    assert result["target_found"] is True
    manager.local_progress_with_avoidance_handler.assert_called_once_with()
    manager._confirm_find_target_with_semantic.assert_not_called()


def test_target_loss_after_avoidance_uses_normal_post_motion_confirmation():
    nested = {"terminal_state": "PATH_CLEAR", "physical_actions": 2}
    manager = _manager(
        [_progress(mode="BOUNDED_AVOIDANCE", physical_actions=2, nested=nested)],
        confirmations=[(None, "target_reconfirmation_failed", {"detections": []})],
    )

    result = _run_approach(manager)

    assert result["state"] == "TARGET_LOST_AFTER_APPROACH"
    assert result["target_found"] is False
    assert result["physical_actions"] == 2
    manager.local_progress_with_avoidance_handler.assert_called_once_with()


def test_four_nested_actions_do_not_start_a_second_episode_or_fifth_action():
    manager = _manager([_progress(
        "LOCAL_PROGRESS_MAX_STEPS_REACHED",
        mode="BOUNDED_AVOIDANCE",
        physical_actions=4,
        nested={"terminal_state": "MAX_STEPS_REACHED", "physical_actions": 4},
    )])

    result = _run_approach(manager)

    assert result["physical_actions"] == 4
    assert result["state"] == "APPROACH_BLOCKED"
    manager.local_progress_with_avoidance_handler.assert_called_once_with()


def test_outer_approach_budget_counts_nested_actions_before_bounded_handoff():
    direct = _progress(mode="DIRECT_FORWARD", physical_actions=1)
    bounded = _progress(
        terminal="LOCAL_PROGRESS_MAX_STEPS_REACHED",
        mode="BOUNDED_AVOIDANCE",
        physical_actions=4,
        nested={"terminal_state": "MAX_STEPS_REACHED", "physical_actions": 4},
    )
    confirmations = [(_target(), "confirmed", {"fresh": True}) for _ in range(4)]
    manager = _manager([direct, direct, direct, bounded], confirmations=confirmations)

    result = _run_approach(manager)

    assert result["state"] == "APPROACH_BLOCKED"
    assert result["physical_actions"] == 7
    assert result["maximum_local_progress_physical_actions"] == 7
    assert result["local_progress_calls"] == 4
    assert manager.local_progress_with_avoidance_handler.call_count == 4


def test_runtime_authorizes_handoff_only_on_current_find_object_owner_thread():
    runtime = object.__new__(CognitiveRuntime)
    active = SimpleNamespace(mission_id="find-1", mission_type="FIND_OBJECT")
    runtime._state_lock = threading.RLock()
    runtime._control_generation = 5
    runtime._behavior_execution_generation = 5
    runtime._behavior_execution_thread_id = threading.get_ident()
    runtime._find_object_progress_context = threading.local()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: active)
    runtime.MAX_REACTIVE_STEPS = 4
    runtime.run_local_progress_with_avoidance = Mock(
        side_effect=lambda: {
            "owner_seen_as_busy": runtime._active_localization_has_behavior_owner()
        }
    )

    result = runtime._run_find_object_local_progress()

    assert result == {"owner_seen_as_busy": False}
    runtime.run_local_progress_with_avoidance.assert_called_once_with()
    assert runtime._active_localization_has_behavior_owner() is True


def test_runtime_callback_rejects_other_owner_without_calling_coordinator():
    runtime = object.__new__(CognitiveRuntime)
    active = SimpleNamespace(mission_id="marvin-1", mission_type="FIND_OBJECT")
    runtime._state_lock = threading.RLock()
    runtime._control_generation = 5
    runtime._behavior_execution_generation = 5
    runtime._behavior_execution_thread_id = threading.get_ident() + 1
    runtime._find_object_progress_context = threading.local()
    runtime.mission_manager = SimpleNamespace(get_active_mission=lambda: active)
    runtime.MAX_REACTIVE_STEPS = 4
    runtime.run_local_progress_with_avoidance = Mock()

    result = runtime._run_find_object_local_progress()

    assert result["terminal_state"] == "LOCAL_PROGRESS_OWNERSHIP_REJECTED"
    assert result["physical_actions"] == 0
    runtime.run_local_progress_with_avoidance.assert_not_called()
