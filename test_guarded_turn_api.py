"""Offline tests for the explicit guarded-turn Runtime API endpoint."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from runtime_api import RuntimeAPIHandler


SESSION = "runtime-session"


class FakeWorker:
    session = SESSION


class FakeBehaviorManager:
    def __init__(self, result=None):
        self.result = result or {
            "ok": True,
            "permitted": True,
            "reason": "turn_side_clear_advisory",
            "direction": "LEFT",
            "angular_z": 0.5,
            "duration": 0.4,
            "transport_attempted": True,
            "forwarded": True,
            "confirmed_forwarded": True,
            "generation": 3,
            "generation_invalidated": False,
            "pending_turn": False,
            "active_turn": False,
            "inhibited": False,
            "monitor_reason": "turn_side_clear_advisory",
            "producer_session": SESSION,
            "effective_age_seconds": 0.05,
            "left_state": "CLEAR",
            "front_left_state": "CLEAR",
            "stop_count": 0,
        }
        self.calls = []

    def execute_guarded_turn(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return dict(self.result)


def call_post(body, *, manager=None, worker=None):
    manager = manager or FakeBehaviorManager()
    runtime = SimpleNamespace(
        behavior_manager=manager,
        lidar_worker=worker or FakeWorker(),
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/guarded-turn"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value=body)
    handler.send_json = Mock()
    handler.do_POST()
    status, result = handler.send_json.call_args.args
    return status, result, manager, handler


def test_left_request_uses_exact_runtime_session_and_passthrough_result():
    status, result, manager, _ = call_post({
        "direction": "LEFT",
        "angular_speed": 0.5,
        "duration": 0.4,
    })

    assert status == 200
    assert result["ok"] is True
    assert manager.calls == [(
        ("LEFT", 0.5, 0.4),
        {"expected_lidar_session": SESSION},
    )]


def test_right_request_is_forwarded_unchanged():
    status, _, manager, _ = call_post({
        "direction": "RIGHT",
        "angular_speed": 0.25,
        "duration": 0.2,
    })

    assert status == 200
    assert manager.calls[0] == (
        ("RIGHT", 0.25, 0.2),
        {"expected_lidar_session": SESSION},
    )


def test_caller_cannot_override_session_or_add_motion_fields():
    for body in (
        {
            "direction": "LEFT",
            "angular_speed": 0.5,
            "duration": 0.4,
            "producer_session": "attacker-session",
        },
        {
            "direction": "LEFT",
            "angular_speed": 0.5,
            "duration": 0.4,
            "linear_x": 0.0,
        },
        {
            "direction": "LEFT",
            "angular_speed": 0.5,
            "duration": 0.4,
            "streaming": False,
        },
    ):
        status, result, manager, _ = call_post(body)
        assert status == 400
        assert result["ok"] is False
        assert manager.calls == []


def test_missing_and_malformed_fields_are_rejected_before_execution():
    bodies = (
        {},
        {"direction": "LEFT", "angular_speed": 0.5},
        {"direction": "LEFT", "duration": 0.4},
        {"direction": [], "angular_speed": 0.5, "duration": 0.4},
        {"direction": "LEFT", "angular_speed": "0.5", "duration": 0.4},
        {"direction": "LEFT", "angular_speed": True, "duration": 0.4},
        {"direction": "LEFT", "angular_speed": 0.5, "duration": float("nan")},
    )
    for body in bodies:
        status, result, manager, _ = call_post(body)
        assert status == 400
        assert result["ok"] is False
        assert manager.calls == []


def test_unsupported_direction_is_rejected_before_execution():
    status, result, manager, _ = call_post({
        "direction": "FORWARD",
        "angular_speed": 0.5,
        "duration": 0.4,
    })

    assert status == 400
    assert result["ok"] is False
    assert manager.calls == []


def test_safety_denial_is_returned_without_becoming_transport_success():
    denial = {
        "ok": False,
        "permitted": False,
        "forwarded": False,
        "transport_attempted": False,
        "reason": "turn_side_not_clear",
    }
    manager = FakeBehaviorManager(denial)

    status, result, _, _ = call_post({
        "direction": "LEFT",
        "angular_speed": 0.5,
        "duration": 0.4,
    }, manager=manager)

    assert status == 200
    assert result == denial
    assert result["permitted"] is False
    assert result["transport_attempted"] is False


def test_transport_failure_and_concurrent_denial_are_surfaced_json_safely():
    failure = {
        "ok": False,
        "permitted": True,
        "reason": "transport_exception",
        "transport_attempted": True,
        "delivery_uncertain": True,
        "transport_error": "timed out",
        "transport_error_type": "TimeoutError",
        "stop_fallback_attempted": True,
    }
    for result_value, reason in (
        (failure, "transport_exception"),
        ({
            "ok": False,
            "permitted": False,
            "reason": "turn_already_active",
            "transport_attempted": False,
        }, "turn_already_active"),
    ):
        manager = FakeBehaviorManager(result_value)
        status, result, _, _ = call_post({
            "direction": "RIGHT",
            "angular_speed": 0.5,
            "duration": 0.4,
        }, manager=manager)
        assert status == 200
        assert result["reason"] == reason
        json.dumps(result)


def test_endpoint_does_not_call_robot_transport_or_local_policy():
    manager = FakeBehaviorManager()
    robot = SimpleNamespace(motion=Mock(), stop=Mock())
    runtime = SimpleNamespace(
        behavior_manager=manager,
        lidar_worker=FakeWorker(),
        robot_client=robot,
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/guarded-turn"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={
        "direction": "LEFT",
        "angular_speed": 0.5,
        "duration": 0.4,
    })
    handler.send_json = Mock()

    handler.do_POST()

    robot.motion.assert_not_called()
    robot.stop.assert_not_called()
    assert len(manager.calls) == 1
