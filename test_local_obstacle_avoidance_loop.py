"""Offline bounded-loop tests; the loop never owns a motion primitive."""

import pytest

from behavior_manager import BehaviorManager


SESSION = "loop-session"


class NoMotionRobot:
    def local_forward(self):
        raise AssertionError("loop must not call local_forward directly")

    def motion(self, *args, **kwargs):
        raise AssertionError("loop must not call motion directly")


def step(sequence, *, ok=True, motion_executed=True, replan_required=True,
         session=SESSION, reason="bounded_local_avoidance_primitive_complete"):
    return {
        "ok": ok,
        "motion_executed": motion_executed,
        "replan_required": replan_required,
        "reason": reason,
        "lidar_acquisition_sequence": sequence,
        "planner": {"producer_session": session},
    }


def invoke(monkeypatch, stream, *, max_steps=3, session=SESSION):
    manager = BehaviorManager(robot_client=NoMotionRobot())
    calls = []
    values = iter(stream)

    def coordinator(**kwargs):
        calls.append(kwargs)
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", coordinator)
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=session, max_steps=max_steps, now=10.0,
    )
    return result, calls


def test_one_successful_step_then_failure_stops_without_step_three(monkeypatch):
    failed = step(101, ok=False, motion_executed=False, reason="no_safe_local_avoidance")
    result, calls = invoke(monkeypatch, [step(100), failed, step(102)])
    assert result["ok"] is False
    assert result["reason"] == "no_safe_local_avoidance"
    assert result["steps_executed"] == 1
    assert len(calls) == len(result["steps"]) == 2


def test_three_successful_steps_stop_at_hard_limit(monkeypatch):
    result, calls = invoke(monkeypatch, [step(100), step(101), step(102)])
    assert result["ok"] is True and result["completed"] is False
    assert result["reason"] == "local_avoidance_step_limit_reached"
    assert result["steps_executed"] == 3
    assert len(calls) == 3
    assert [call["minimum_lidar_acquisition_sequence"] for call in calls] == [None, 100, 101]


def test_max_steps_one_makes_exactly_one_coordinator_call(monkeypatch):
    result, calls = invoke(monkeypatch, [step(100), step(101)], max_steps=1)
    assert result["reason"] == "local_avoidance_step_limit_reached"
    assert result["steps_executed"] == len(calls) == 1


@pytest.mark.parametrize("limit", [0, -1, 1.5, "3", True, None])
def test_invalid_step_limits_fail_closed_without_coordinator_call(monkeypatch, limit):
    result, calls = invoke(monkeypatch, [], max_steps=limit)
    assert result["ok"] is False
    assert result["reason"] == "invalid_local_avoidance_step_limit"
    assert calls == result["steps"] == []


def test_new_sequence_between_steps_is_required_and_recorded(monkeypatch):
    result, calls = invoke(monkeypatch, [step(100), step(101), step(102)])
    assert [item["lidar_acquisition_sequence"] for item in result["steps"]] == [100, 101, 102]
    assert calls[1]["minimum_lidar_acquisition_sequence"] == 100
    assert calls[2]["minimum_lidar_acquisition_sequence"] == 101


@pytest.mark.parametrize("next_sequence", [100, 99])
def test_same_or_older_sequence_blocks_before_another_motion(monkeypatch, next_sequence):
    blocked = step(
        next_sequence,
        ok=False,
        motion_executed=False,
        reason="fresh_lidar_after_motion_unavailable",
    )
    result, calls = invoke(monkeypatch, [step(100), blocked, step(102)])
    assert result["reason"] == "fresh_lidar_after_motion_unavailable"
    assert result["steps_executed"] == 1
    assert len(calls) == 2
    assert result["steps"][1]["motion_executed"] is False


def test_producer_session_change_stops_before_next_primitive(monkeypatch):
    changed = step(101, ok=False, motion_executed=False, session="other",
                   reason="producer_session_mismatch")
    result, calls = invoke(monkeypatch, [step(100), changed, step(102)])
    assert result["reason"] == "producer_session_mismatch"
    assert result["steps_executed"] == 1 and len(calls) == 2


def test_step_exception_is_captured_without_later_call(monkeypatch):
    result, calls = invoke(monkeypatch, [step(100), RuntimeError("offline"), step(102)])
    assert result["ok"] is False
    assert result["reason"] == "local_avoidance_step_exception"
    assert result["steps_executed"] == 1 and len(calls) == 2
    assert result["steps"][1]["motion_executed"] is False


def test_no_motion_or_no_replan_result_terminates_without_another_step(monkeypatch):
    no_motion = step(100, motion_executed=False, reason="no_motion_needed")
    result, calls = invoke(monkeypatch, [no_motion, step(101)])
    assert result["reason"] == "local_avoidance_motion_not_executed"
    assert len(calls) == 1

    no_replan = step(100, replan_required=False)
    result, calls = invoke(monkeypatch, [no_replan, step(101)])
    assert result["ok"] is True
    assert result["reason"] == "local_avoidance_replan_not_required"
    assert len(calls) == 1


def test_exact_history_and_determinism(monkeypatch):
    first, first_calls = invoke(monkeypatch, [step(10), step(11), step(12)])
    second, second_calls = invoke(monkeypatch, [step(10), step(11), step(12)])
    assert first == second
    assert len(first["steps"]) == first["steps_executed"] == 3
    assert len(first_calls) == len(second_calls) == 3
