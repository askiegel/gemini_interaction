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


def invoke(monkeypatch, stream, *, max_steps=3, session=SESSION,
           freshness_results=None):
    manager = BehaviorManager(robot_client=NoMotionRobot())
    calls = []
    values = iter(stream)
    wait_calls = []
    waits = iter(freshness_results or [])

    def coordinator(**kwargs):
        calls.append(kwargs)
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", coordinator)
    def wait_for_newer(**kwargs):
        wait_calls.append(kwargs)
        try:
            return next(waits)
        except StopIteration:
            return {
                "ok": True,
                "snapshot": {"producer_session": session,
                             "acquisition_sequence": kwargs["previous_sequence"] + 1},
                "acquisition_sequence": kwargs["previous_sequence"] + 1,
                "wait_elapsed_seconds": 0.0,
                "poll_count": 1,
                "reason": "newer_lidar_snapshot_available",
            }
    monkeypatch.setattr(manager, "_wait_for_newer_lidar_snapshot", wait_for_newer)
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=session, max_steps=max_steps, now=10.0,
    )
    return result, calls, wait_calls


def test_one_successful_step_then_failure_stops_without_step_three(monkeypatch):
    failed = step(101, ok=False, motion_executed=False, reason="no_safe_local_avoidance")
    result, calls, waits = invoke(monkeypatch, [step(100), failed, step(102)])
    assert result["ok"] is False
    assert result["reason"] == "no_safe_local_avoidance"
    assert result["steps_executed"] == 1
    assert len(calls) == len(result["steps"]) == 2
    assert len(waits) == 1


def test_three_successful_steps_stop_at_hard_limit(monkeypatch):
    result, calls, waits = invoke(monkeypatch, [step(100), step(101), step(102)])
    assert result["ok"] is True and result["completed"] is False
    assert result["reason"] == "local_avoidance_step_limit_reached"
    assert result["steps_executed"] == 3
    assert len(calls) == 3
    assert [call["minimum_lidar_acquisition_sequence"] for call in calls] == [None, 100, 101]
    assert [call["previous_sequence"] for call in waits] == [100, 101]
    assert len(result["freshness_waits"]) == 2


def test_max_steps_one_makes_exactly_one_coordinator_call(monkeypatch):
    result, calls, waits = invoke(monkeypatch, [step(100), step(101)], max_steps=1)
    assert result["reason"] == "local_avoidance_step_limit_reached"
    assert result["steps_executed"] == len(calls) == 1
    assert waits == []


@pytest.mark.parametrize("limit", [0, -1, 1.5, "3", True, None])
def test_invalid_step_limits_fail_closed_without_coordinator_call(monkeypatch, limit):
    result, calls, waits = invoke(monkeypatch, [], max_steps=limit)
    assert result["ok"] is False
    assert result["reason"] == "invalid_local_avoidance_step_limit"
    assert calls == result["steps"] == waits == []


def test_new_sequence_between_steps_is_required_and_recorded(monkeypatch):
    result, calls, waits = invoke(monkeypatch, [step(100), step(101), step(102)])
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
    result, calls, waits = invoke(monkeypatch, [step(100), blocked, step(102)])
    assert result["reason"] == "fresh_lidar_after_motion_unavailable"
    assert result["steps_executed"] == 1
    assert len(calls) == 2
    assert result["steps"][1]["motion_executed"] is False


def test_producer_session_change_stops_before_next_primitive(monkeypatch):
    changed = step(101, ok=False, motion_executed=False, session="other",
                   reason="producer_session_mismatch")
    result, calls, waits = invoke(monkeypatch, [step(100), changed, step(102)])
    assert result["reason"] == "producer_session_mismatch"
    assert result["steps_executed"] == 1 and len(calls) == 2


def test_step_exception_is_captured_without_later_call(monkeypatch):
    result, calls, waits = invoke(monkeypatch, [step(100), RuntimeError("offline"), step(102)])
    assert result["ok"] is False
    assert result["reason"] == "local_avoidance_step_exception"
    assert result["steps_executed"] == 1 and len(calls) == 2
    assert result["steps"][1]["motion_executed"] is False


def test_no_motion_or_no_replan_result_terminates_without_another_step(monkeypatch):
    no_motion = step(100, motion_executed=False, reason="no_motion_needed")
    result, calls, waits = invoke(monkeypatch, [no_motion, step(101)])
    assert result["reason"] == "local_avoidance_motion_not_executed"
    assert len(calls) == 1

    no_replan = step(100, replan_required=False)
    result, calls, waits = invoke(monkeypatch, [no_replan, step(101)])
    assert result["ok"] is True
    assert result["reason"] == "local_avoidance_replan_not_required"
    assert len(calls) == 1


def test_exact_history_and_determinism(monkeypatch):
    first, first_calls, first_waits = invoke(monkeypatch, [step(10), step(11), step(12)])
    second, second_calls, second_waits = invoke(monkeypatch, [step(10), step(11), step(12)])
    assert first == second
    assert len(first["steps"]) == first["steps_executed"] == 3
    assert len(first_calls) == len(second_calls) == 3


class SequenceWorldModel:
    def __init__(self, values):
        self.values = iter(values)
        self.read_count = 0

    def get_lidar_obstacles(self, *, expected_session, now=None):
        self.read_count += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


class FakeMonotonic:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def snapshot(sequence, *, session=SESSION):
    return {"producer_session": session, "acquisition_sequence": sequence}


def wait_manager(monkeypatch, values):
    manager = BehaviorManager(
        robot_client=NoMotionRobot(), world_model=SequenceWorldModel(values),
    )
    clock = FakeMonotonic()
    monkeypatch.setattr("behavior_manager.time.monotonic", clock.monotonic)
    monkeypatch.setattr("behavior_manager.time.sleep", clock.sleep)
    return manager, clock


def wait(manager, **overrides):
    args = {
        "expected_lidar_session": SESSION,
        "previous_sequence": 100,
        "timeout_seconds": 1.0,
        "poll_interval_seconds": 0.05,
    }
    args.update(overrides)
    return manager._wait_for_newer_lidar_snapshot(**args)


def test_wait_accepts_immediately_new_snapshot_without_sleep(monkeypatch):
    manager, clock = wait_manager(monkeypatch, [snapshot(101)])
    result = wait(manager)
    assert result["ok"] is True
    assert result["acquisition_sequence"] == 101
    assert result["poll_count"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("values,expected_polls", [
    ([snapshot(100), snapshot(101)], 2),
    ([snapshot(100), snapshot(100), snapshot(101)], 3),
])
def test_wait_polls_until_sequence_advances(monkeypatch, values, expected_polls):
    manager, clock = wait_manager(monkeypatch, values)
    result = wait(manager)
    assert result["ok"] is True
    assert result["acquisition_sequence"] == 101
    assert result["poll_count"] == expected_polls
    assert len(clock.sleeps) == expected_polls - 1
    assert all(duration == pytest.approx(0.05) for duration in clock.sleeps)


def test_wait_times_out_by_monotonic_deadline_and_stops_loop(monkeypatch):
    manager, clock = wait_manager(monkeypatch, [snapshot(100)] * 8)
    result, calls, waits = invoke(
        monkeypatch,
        [step(100), step(101)],
        freshness_results=[{
            "ok": False,
            "reason": "fresh_lidar_after_motion_timeout",
            "previous_acquisition_sequence": 100,
            "last_acquisition_sequence": 100,
            "wait_elapsed_seconds": 1.0,
            "poll_count": 21,
        }],
    )
    assert result["reason"] == "fresh_lidar_after_motion_timeout"
    assert result["steps_executed"] == 1
    assert len(calls) == 1
    assert len(waits) == 1

    # Exercise the real helper's monotonic deadline independently.
    wait_result = wait(manager, timeout_seconds=0.12, poll_interval_seconds=0.05)
    assert wait_result["ok"] is False
    assert wait_result["reason"] == "fresh_lidar_after_motion_timeout"
    assert wait_result["wait_elapsed_seconds"] == pytest.approx(0.12)
    assert clock.now == pytest.approx(0.12)
    assert clock.sleeps == pytest.approx([0.05, 0.05, 0.02])


def test_real_loop_waits_for_new_sequences_before_next_coordinator_call(monkeypatch):
    world = SequenceWorldModel([snapshot(101), snapshot(102)])
    manager = BehaviorManager(robot_client=NoMotionRobot(), world_model=world)
    clock = FakeMonotonic()
    monkeypatch.setattr("behavior_manager.time.monotonic", clock.monotonic)
    monkeypatch.setattr("behavior_manager.time.sleep", clock.sleep)
    sequences = iter([100, 101, 102])
    coordinator_calls = []

    def coordinator(**kwargs):
        coordinator_calls.append(kwargs)
        return step(next(sequences))

    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", coordinator)
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=SESSION, max_steps=3,
    )
    assert result["reason"] == "local_avoidance_step_limit_reached"
    assert result["steps_executed"] == len(coordinator_calls) == 3
    assert [entry["lidar_acquisition_sequence"] for entry in result["steps"]] == [100, 101, 102]
    assert [entry["acquisition_sequence"] for entry in result["freshness_waits"]] == [101, 102]
    assert [call["minimum_lidar_acquisition_sequence"] for call in coordinator_calls] == [None, 100, 101]
    assert world.read_count == 2


def test_real_loop_timeout_does_not_dispatch_second_coordinator_step(monkeypatch):
    world = SequenceWorldModel([snapshot(100)] * 8)
    manager = BehaviorManager(robot_client=NoMotionRobot(), world_model=world)
    clock = FakeMonotonic()
    monkeypatch.setattr("behavior_manager.time.monotonic", clock.monotonic)
    monkeypatch.setattr("behavior_manager.time.sleep", clock.sleep)
    coordinator_calls = []

    def coordinator(**kwargs):
        coordinator_calls.append(kwargs)
        return step(100)

    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", coordinator)
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=SESSION, max_steps=3,
        lidar_wait_timeout_seconds=0.12,
        lidar_poll_interval_seconds=0.05,
    )
    assert result["reason"] == "fresh_lidar_after_motion_timeout"
    assert result["steps_executed"] == len(coordinator_calls) == 1
    assert len(result["steps"]) == 1
    assert result["freshness_waits"][0]["poll_count"] == 3
    assert world.read_count == 3
    assert clock.sleeps == pytest.approx([0.05, 0.05, 0.02])


def test_real_loop_session_change_stops_before_next_coordinator_step(monkeypatch):
    world = SequenceWorldModel([snapshot(101, session="new-session")])
    manager = BehaviorManager(robot_client=NoMotionRobot(), world_model=world)
    clock = FakeMonotonic()
    monkeypatch.setattr("behavior_manager.time.monotonic", clock.monotonic)
    monkeypatch.setattr("behavior_manager.time.sleep", clock.sleep)
    calls = []

    def coordinator(**kwargs):
        calls.append(kwargs)
        return step(100)

    monkeypatch.setattr(manager, "execute_local_obstacle_avoidance_step", coordinator)
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=SESSION, max_steps=3,
    )
    assert result["reason"] == "lidar_producer_session_changed"
    assert result["producer_session"] == SESSION
    assert result["steps_executed"] == len(calls) == 1
    assert world.read_count == 1
    assert clock.sleeps == []


def test_coordinator_minimum_sequence_guard_remains_after_successful_wait(monkeypatch):
    result, calls, waits = invoke(
        monkeypatch,
        [step(100), step(100)],
    )
    assert len(waits) == 1
    # The loop still sends the last accepted sequence to the coordinator. Its
    # own authoritative read/guard remains responsible for rejecting equality.
    assert calls[1]["minimum_lidar_acquisition_sequence"] == 100
    assert result["reason"] == "fresh_lidar_after_motion_unavailable"
    assert result["steps_executed"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("sequence,reason", [
    (99, "lidar_acquisition_sequence_regressed"),
    (None, "malformed_lidar_acquisition_sequence"),
    ("101", "malformed_lidar_acquisition_sequence"),
])
def test_wait_fails_closed_on_regressed_or_malformed_sequence(monkeypatch, sequence, reason):
    manager, _clock = wait_manager(monkeypatch, [snapshot(sequence)])
    result = wait(manager)
    assert result["ok"] is False
    assert result["reason"] == reason
    assert result["poll_count"] == 1


def test_wait_fails_immediately_when_producer_session_changes(monkeypatch):
    manager, clock = wait_manager(monkeypatch, [snapshot(101, session="changed")])
    result = wait(manager)
    assert result["ok"] is False
    assert result["reason"] == "lidar_producer_session_changed"
    assert result["poll_count"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("value,reason", [
    (None, "malformed_lidar_snapshot"),
    (RuntimeError("read failed"), "world_model_read_failed"),
])
def test_wait_fails_closed_on_malformed_snapshot_or_read_exception(monkeypatch, value, reason):
    manager, clock = wait_manager(monkeypatch, [value])
    result = wait(manager)
    assert result["ok"] is False
    assert result["reason"] == reason
    assert result["poll_count"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("timeout,interval", [(0, 0.05), (-1, 0.05), (float("inf"), 0.05), (1, 0), (1, 2)])
def test_loop_rejects_invalid_wait_bounds_without_motion(monkeypatch, timeout, interval):
    manager = BehaviorManager(robot_client=NoMotionRobot())
    result = manager.execute_local_obstacle_avoidance_loop(
        expected_lidar_session=SESSION, max_steps=3,
        lidar_wait_timeout_seconds=timeout,
        lidar_poll_interval_seconds=interval,
    )
    assert result["reason"] == "invalid_lidar_freshness_wait_bounds"
    assert result["steps"] == []
