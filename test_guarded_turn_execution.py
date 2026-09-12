"""Offline tests for the explicit guarded bounded-turn execution boundary."""

import json

from behavior_manager import BehaviorManager
from local_obstacle_policy import recommend_local_avoidance


SESSION = "session-1"


def snapshot(*, front="CLEAR", left="CLEAR", front_left="CLEAR",
             right="CLEAR", front_right="CLEAR"):
    def sector(state, clearance=1.0):
        return {
            "state": state,
            "available": True,
            "robust_clearance_m": clearance,
            "minimum_clearance_m": clearance,
        }

    return {
        "available": True,
        "valid": True,
        "reason": "fresh",
        "producer_session": SESSION,
        "received_monotonic_seconds": 10.0,
        "age_at_receipt_seconds": 0.05,
        "sectors": {
            "front": sector(front),
            "front_left": sector(front_left),
            "front_right": sector(front_right),
            "left": sector(left),
            "right": sector(right),
        },
    }


class FakeWorldModel:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def get_lidar_obstacles(self, *, expected_session, now=None):
        self.calls.append((expected_session, now))
        return self.state


class FakeRobot:
    def __init__(self, result=None, error=None, stop_error=None):
        self.result = result or {"ok": True, "action": "motion"}
        self.error = error
        self.stop_error = stop_error
        self.motion_calls = []
        self.stop_calls = 0

    def motion(self, **payload):
        self.motion_calls.append(payload)
        if self.error:
            raise self.error
        return self.result

    def stop(self):
        self.stop_calls += 1
        if self.stop_error:
            raise self.stop_error
        return {"ok": True, "action": "stop"}


def manager(state, robot=None):
    return BehaviorManager(
        robot_client=robot or FakeRobot(),
        world_model=FakeWorldModel(state),
    )


def execute(state, direction="LEFT", speed=0.5, duration=0.4, robot=None):
    return manager(state, robot).execute_guarded_turn(
        direction,
        speed,
        duration,
        expected_lidar_session=SESSION,
        now=10.0,
    )


def test_valid_left_forwards_once_with_positive_angular_z():
    robot = FakeRobot()
    result = execute(snapshot(), robot=robot)
    assert result["ok"] is True
    assert result["permitted"] is True
    assert result["forwarded"] is True
    assert robot.motion_calls == [{
        "linear_x": 0.0,
        "angular_z": 0.5,
        "duration": 0.4,
        "streaming": False,
    }]
    assert robot.stop_calls == 0


def test_valid_right_forwards_once_with_negative_angular_z():
    robot = FakeRobot()
    result = execute(snapshot(), direction="RIGHT", robot=robot)
    assert result["ok"] is True
    assert robot.motion_calls[0]["angular_z"] == -0.5


def test_front_blocked_with_clear_turn_side_can_forward():
    robot = FakeRobot()
    result = execute(snapshot(front="BLOCKED"), robot=robot)
    assert result["permitted"] is True
    assert len(robot.motion_calls) == 1


def test_denials_make_zero_motion_calls():
    cases = [
        snapshot(front="CAUTION", left="BLOCKED"),
        snapshot(front="CAUTION", left="CAUTION"),
        snapshot(front="CAUTION", left="UNKNOWN"),
        snapshot(front="CAUTION", left="CLEAR", front_left="BLOCKED"),
        snapshot(front="CAUTION", left="CLEAR", front_left="UNKNOWN"),
    ]
    for state in cases:
        robot = FakeRobot()
        result = execute(state, robot=robot)
        assert result["permitted"] is False
        assert result["forwarded"] is False
        assert result["transport_attempted"] is False
        assert result["stop_fallback_attempted"] is False
        assert robot.motion_calls == []
        assert robot.stop_calls == 0


def test_stale_unavailable_and_session_mismatch_deny_before_transport():
    for mutate in (
        lambda state: state.update(age_at_receipt_seconds=0.31),
        lambda state: state.update(available=False, valid=False),
        lambda state: state.update(producer_session="other"),
    ):
        state = snapshot()
        mutate(state)
        robot = FakeRobot()
        result = execute(state, robot=robot)
        assert result["permitted"] is False
        assert result["transport_attempted"] is False
        assert robot.motion_calls == []
        assert robot.stop_calls == 0


def test_speed_and_duration_bounds_deny_without_transport():
    for speed, duration in ((1.01, 0.4), (0.5, 0.51), (0.0, 0.4), (0.5, 0.0)):
        robot = FakeRobot()
        result = execute(snapshot(), speed=speed, duration=duration, robot=robot)
        assert result["permitted"] is False
        assert robot.motion_calls == []


def test_transport_failure_is_explicit():
    robot = FakeRobot(result={"ok": False, "error": "bridge rejected"})
    result = execute(snapshot(), robot=robot)
    assert result["permitted"] is True
    assert result["forwarded"] is False
    assert result["confirmed_forwarded"] is False
    assert result["transport_attempted"] is True
    assert result["delivery_uncertain"] is True
    assert result["ok"] is False
    assert result["reason"] == "transport_failed"
    assert result["stop_fallback_attempted"] is True
    assert result["stop_fallback_result"]["ok"] is True
    assert robot.stop_calls == 1


def test_transport_exception_is_explicit():
    error = TimeoutError("timed out")
    robot = FakeRobot(error=error)
    result = execute(snapshot(), robot=robot)
    assert result["permitted"] is True
    assert result["forwarded"] is False
    assert result["confirmed_forwarded"] is False
    assert result["transport_attempted"] is True
    assert result["delivery_uncertain"] is True
    assert result["ok"] is False
    assert result["reason"] == "transport_exception"
    assert result["transport_error"] == str(error)
    assert result["transport_error_type"] == "TimeoutError"
    assert result["stop_fallback_attempted"] is True
    assert result["stop_fallback_result"]["ok"] is True
    assert robot.stop_calls == 1


def test_transport_exception_and_stop_exception_both_remain_visible():
    motion_error = TimeoutError("motion timed out")
    stop_error = RuntimeError("stop unavailable")
    robot = FakeRobot(error=motion_error, stop_error=stop_error)
    result = execute(snapshot(), robot=robot)
    assert result["reason"] == "transport_exception"
    assert result["transport_error"] == str(motion_error)
    assert result["transport_error_type"] == "TimeoutError"
    assert result["stop_fallback_attempted"] is True
    assert result["stop_fallback_error"] == str(stop_error)
    assert result["stop_fallback_error_type"] == "RuntimeError"
    assert result["transport_result"]["ok"] is False
    assert robot.stop_calls == 1


def test_stop_remains_unconditional_when_turn_is_denied():
    robot = FakeRobot()
    behavior = manager(snapshot(front="CAUTION", left="BLOCKED"), robot)
    denied = behavior.execute_guarded_turn(
        "LEFT", 0.5, 0.4, expected_lidar_session=SESSION, now=10.0,
    )
    stopped = behavior._execute_stop()
    assert denied["permitted"] is False
    assert stopped["ok"] is True
    assert robot.motion_calls == []
    assert robot.stop_calls == 1


def test_no_forward_or_reverse_transport_is_used():
    robot = FakeRobot()
    result = execute(snapshot(), robot=robot)
    assert result["ok"] is True
    assert robot.motion_calls[0]["linear_x"] == 0.0


def test_advisory_recommendation_does_not_execute_a_turn():
    robot = FakeRobot()
    recommendation = recommend_local_avoidance(
        snapshot(), expected_session=SESSION, now=10.0,
    )
    assert recommendation["recommendation"] == "FORWARD"
    assert robot.motion_calls == []


def test_all_guarded_turn_result_variants_are_json_serializable():
    successful = execute(snapshot())
    denied = execute(snapshot(front="CAUTION", left="BLOCKED"))
    motion_exception = execute(
        snapshot(), robot=FakeRobot(error=TimeoutError("motion timed out")),
    )
    both_exceptions = execute(
        snapshot(),
        robot=FakeRobot(
            error=TimeoutError("motion timed out"),
            stop_error=RuntimeError("stop unavailable"),
        ),
    )
    explicit_failure = execute(
        snapshot(), robot=FakeRobot(result={"ok": False, "error": "rejected"}),
    )

    for result in (
        successful,
        denied,
        motion_exception,
        both_exceptions,
        explicit_failure,
    ):
        json.dumps(result)
