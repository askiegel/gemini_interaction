"""Offline tests for the narrow bounded local-reactive Runtime API."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime_api import RuntimeAPIHandler


def _result(terminal_state, *, bridge_stopped=True):
    return {
        "ok": terminal_state in {
            "PATH_CLEAR", "BLOCKED", "MAX_STEPS_REACHED",
        },
        "action": "bounded_local_reactive_avoidance",
        "terminal_state": terminal_state,
        "steps_attempted": 2,
        "physical_actions": 2,
        "max_steps": 4,
        "turns_executed": 1,
        "forward_steps_executed": 1,
        "bridge_stopped": bridge_stopped,
        "history": [],
    }


class _ForbiddenDirectMotion:
    def __getattr__(self, name):
        raise AssertionError("endpoint must not access direct motion authority")


class _Runtime:
    def __init__(self, result):
        self.run_bounded_local_reactive_avoidance = Mock(return_value=result)
        self.run_local_reactive_step = _ForbiddenDirectMotion()
        self.behavior_manager = _ForbiddenDirectMotion()
        self.robot_client = _ForbiddenDirectMotion()


def _call(body, result):
    runtime = _Runtime(result)
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/local-reactive-avoidance"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value=body)
    handler.send_json = Mock()
    handler.do_POST()
    return handler, runtime


@pytest.mark.parametrize("terminal_state", [
    "PATH_CLEAR", "BLOCKED", "MAX_STEPS_REACHED",
])
def test_verified_normal_terminal_outcome_passes_through_as_success(terminal_state):
    result = _result(terminal_state)
    handler, runtime = _call({}, result)

    assert handler.send_json.call_args.args == (200, result)
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()


def test_verified_safety_veto_is_a_valid_safe_terminal_response():
    result = _result("SAFETY_VETO")
    handler, runtime = _call({}, result)

    assert handler.send_json.call_args.args == (200, result)
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()


@pytest.mark.parametrize("terminal_state", [
    "OWNERSHIP_REJECTED", "EXECUTION_FAILED",
])
def test_ownership_or_execution_failure_uses_existing_non_success_convention(
    terminal_state,
):
    result = _result(terminal_state, bridge_stopped=False)
    handler, runtime = _call({}, result)

    assert handler.send_json.call_args.args == (409, result)
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()


def test_safe_terminal_requires_verified_bridge_zero():
    result = _result("PATH_CLEAR", bridge_stopped=False)
    handler, runtime = _call({}, result)

    assert handler.send_json.call_args.args == (409, result)
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()


@pytest.mark.parametrize("body", [
    {"max_steps": 5},
    {"steps": 5},
    {"retries": 1},
    {"direction": "LEFT"},
    {"speed": 0.25},
    {"duration": 0.50},
    {"distance": 1.0},
    {"linear_x": 0.08},
    {"angular_z": 0.25},
    {"turn": "RIGHT"},
    {"angle": 0.1},
    {"safety_mode": "ROTATIONAL_SWEPT_FOOTPRINT"},
])
def test_nonempty_control_or_motion_body_is_rejected_without_execution(body):
    handler, runtime = _call(body, _result("PATH_CLEAR"))

    status, response = handler.send_json.call_args.args
    assert status == 400
    assert response["ok"] is False
    runtime.run_bounded_local_reactive_avoidance.assert_not_called()


def test_endpoint_does_not_require_global_localization_map_or_camera():
    result = _result("PATH_CLEAR")
    result["localization_validated"] = False
    handler, runtime = _call({}, result)

    assert handler.send_json.call_args.args == (200, result)
    runtime.run_bounded_local_reactive_avoidance.assert_called_once_with()
