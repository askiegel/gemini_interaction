"""Offline tests for the narrow one-step local-reactive Runtime API."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime_api import RuntimeAPIHandler


def _result(decision, *, ok=True, bridge_stopped=True, reason="complete"):
    return {
        "ok": ok,
        "decision": decision,
        "action_attempted": decision != "STOP_BLOCKED",
        "action_executed": decision != "STOP_BLOCKED" and ok,
        "reason": reason,
        "decision_evidence": {"decision": decision},
        "executor_result": {},
        "stop_result": {"ok": bridge_stopped},
        "bridge_stopped": bridge_stopped,
    }


class _ForbiddenBehavior:
    def __getattr__(self, name):
        raise AssertionError("endpoint must not access BehaviorManager directly")


class _Runtime:
    def __init__(self, result):
        self.run_local_reactive_step = Mock(return_value=result)
        self.behavior_manager = _ForbiddenBehavior()
        self.robot_client = _ForbiddenBehavior()


def _call(body, result):
    runtime = _Runtime(result)
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/local-reactive-step"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value=body)
    handler.send_json = Mock()
    handler.do_POST()
    return handler, runtime


@pytest.mark.parametrize("decision", [
    "FORWARD_CLEAR", "TURN_LEFT", "TURN_RIGHT",
])
def test_empty_request_invokes_coordinator_once_and_passes_result_through(decision):
    result = _result(decision)
    handler, runtime = _call({}, result)
    assert handler.send_json.call_args.args == (200, result)
    runtime.run_local_reactive_step.assert_called_once_with()


def test_stop_blocked_with_verified_bridge_zero_is_a_valid_safe_outcome():
    result = _result("STOP_BLOCKED", ok=False, bridge_stopped=True,
                     reason="local_reactive_stop_blocked")
    handler, runtime = _call({}, result)
    assert handler.send_json.call_args.args == (200, result)
    runtime.run_local_reactive_step.assert_called_once_with()


@pytest.mark.parametrize("result", [
    _result("TURN_LEFT", ok=False, bridge_stopped=False,
            reason="bridge_not_stopped_after_local_reactive_step"),
    _result(None, ok=False, bridge_stopped=False,
            reason="physical_behavior_already_active"),
])
def test_coordinator_safety_or_busy_failure_uses_existing_conflict_convention(result):
    handler, runtime = _call({}, result)
    assert handler.send_json.call_args.args == (409, result)
    runtime.run_local_reactive_step.assert_called_once_with()


@pytest.mark.parametrize("body", [
    {"direction": "LEFT"},
    {"speed": 0.25},
    {"duration": 0.50},
    {"distance": 1.0},
    {"linear_x": 0.08},
    {"angular_z": 0.25},
    {"angle": 0.1},
    {"turn": "LEFT"},
    {"safety_mode": "ROTATIONAL_SWEPT_FOOTPRINT"},
])
def test_nonempty_motion_or_control_request_is_rejected_without_coordinator(body):
    handler, runtime = _call(body, _result("FORWARD_CLEAR"))
    status, response = handler.send_json.call_args.args
    assert status == 400
    assert response["ok"] is False
    runtime.run_local_reactive_step.assert_not_called()


def test_endpoint_does_not_require_global_localization_or_map_or_camera():
    result = _result("TURN_LEFT")
    result["localization_validated"] = False
    handler, runtime = _call({}, result)
    assert handler.send_json.call_args.args == (200, result)
    runtime.run_local_reactive_step.assert_called_once_with()
