"""Offline tests for the explicit guarded bounded-turn execution boundary."""

import json
import copy
import threading
import time

import behavior_manager as behavior_manager_module
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


class SequenceWorldModel:
    def __init__(self, states):
        self.states = [copy.deepcopy(state) for state in states]
        self.calls = []

    def get_lidar_obstacles(self, *, expected_session, now=None):
        self.calls.append((expected_session, now))
        if self.states:
            state = self.states.pop(0)
            self.last_state = copy.deepcopy(state)
        return copy.deepcopy(getattr(self, "last_state", snapshot()))


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


class BlockingRobot(FakeRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.motion_started = threading.Event()
        self.release_motion = threading.Event()

    def motion(self, **payload):
        self.motion_calls.append(payload)
        self.motion_started.set()
        self.release_motion.wait(timeout=2.0)
        if self.error:
            raise self.error
        return self.result


class ReturningRobot(FakeRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.motion_started = threading.Event()
        self.motion_returned = threading.Event()

    def motion(self, **payload):
        self.motion_calls.append(payload)
        self.motion_started.set()
        result = self.result
        self.motion_returned.set()
        return result


class FirstStopFailsRobot(BlockingRobot):
    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise self.stop_error or RuntimeError("immediate stop unavailable")
        return {"ok": True, "action": "stop"}


class FirstStopRejectsRobot(BlockingRobot):
    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            return {"ok": False, "error": "stop rejected"}
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
    assert result["transient_stale_observed"] is False
    assert result["transient_stale_recovered"] is False


def test_valid_right_forwards_once_with_negative_angular_z():
    robot = FakeRobot()
    result = execute(snapshot(), direction="RIGHT", robot=robot)
    assert result["ok"] is True
    assert robot.motion_calls[0]["angular_z"] == -0.5


def test_one_second_approved_window_is_forwarded_without_extra_wait(monkeypatch):
    def finish_window_immediately(monitor):
        with monitor._lock:
            monitor._window_complete = True
        return True

    monkeypatch.setattr(
        behavior_manager_module._GuardedTurnMonitor,
        "wait_for_window",
        finish_window_immediately,
    )
    robot = FakeRobot()
    result = execute(snapshot(), duration=1.0, robot=robot)
    assert result["ok"] is True
    assert result["duration"] == 1.0
    assert robot.motion_calls == [{
        "linear_x": 0.0,
        "angular_z": 0.5,
        "duration": 1.0,
        "streaming": False,
    }]


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
    for speed, duration in ((1.01, 0.4), (0.5, 1.01), (0.0, 0.4), (0.5, 0.0)):
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


def run_blocked_turn(state, robot, direction="LEFT"):
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                direction,
                0.5,
                0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert robot.motion_started.wait(timeout=1.0)
    return world, behavior, worker, result_box


def wait_for_stop(robot, expected=1):
    deadline = time.monotonic() + 1.0
    while robot.stop_calls < expected and time.monotonic() < deadline:
        time.sleep(0.005)
    assert robot.stop_calls >= expected


def test_left_and_right_clear_turns_complete_without_spurious_stop():
    for direction in ("LEFT", "RIGHT"):
        robot = BlockingRobot()
        world, behavior, worker, result_box = run_blocked_turn(
            snapshot(), robot, direction,
        )
        robot.release_motion.set()
        worker.join(timeout=1.0)
        result = result_box[0]
        assert result["ok"] is True
        assert result["active_turn"] is False
        assert result["pending_turn"] is False
        assert result["generation_invalidated"] is False
        assert result["monitor_running"] is False
        assert robot.stop_calls == 0
        assert world.calls


def test_early_http_return_keeps_monitor_alive_until_duration_window():
    robot = ReturningRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert robot.motion_returned.wait(timeout=1.0)
    time.sleep(0.05)
    assert worker.is_alive()
    assert robot.stop_calls == 0
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["window_complete"] is True
    assert result["monitor_running"] is False
    assert result["stop_count"] == 0


def test_unsafe_transition_after_http_return_stops_before_window_end():
    robot = ReturningRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert robot.motion_returned.wait(timeout=1.0)
    state["sectors"]["left"]["state"] = "BLOCKED"
    wait_for_stop(robot, 1)
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["stop_count"] == 2
    assert robot.stop_calls == 2


def test_normal_synchronous_transport_completion_after_window_succeeds():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 1)
    assert robot.release_motion.is_set() is False
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is True
    assert result["generation_invalidated"] is False
    assert result["reason"] != "turn_window_expired"
    assert result["deadline_stop_attempted"] is True
    assert result["normal_completion"] is True
    assert result["completed_after_deadline"] is True
    assert result["stop_count"] == 1
    assert robot.stop_calls == 1


def test_unsafe_transition_during_transport_completion_wait_stops_and_reasserts():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 1)
    state["sectors"]["left"]["state"] = "CAUTION"
    wait_for_stop(robot, 2)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "turn_side_not_clear"
    assert result["stop_count"] == 3
    assert robot.stop_calls == 3


def test_left_turn_right_side_caution_does_not_invalidate_and_first_snapshot_is_retained():
    robot = BlockingRobot()
    state = snapshot(right="CAUTION", front_right="CAUTION")
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    state["sectors"]["left"]["state"] = "CAUTION"
    wait_for_stop(robot, 1)
    state["sectors"]["left"]["state"] = "BLOCKED"
    state["sectors"]["front_left"]["state"] = "UNKNOWN"
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["permitted"] is True
    assert result["transport_result"]["ok"] is True
    assert result["ok"] is False
    assert result["monitor_validation"]["reason"] == "turn_side_not_clear"
    assert result["monitor_validation"]["left_state"] == "CAUTION"
    assert result["monitor_validation"]["front_left_state"] == "CLEAR"
    assert result["monitor_validation"]["right_state"] == "CAUTION"
    assert result["monitor_validation"]["front_right_state"] == "CAUTION"
    assert result["stop_events"][-1]["monitor_validation"]["left_state"] == "CAUTION"


def test_operator_stop_during_transport_completion_wait_reasserts():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 1)
    stop_result = behavior._execute_stop()
    assert stop_result["ok"] is True
    assert robot.stop_calls == 2
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "operator_stop"
    assert result["stop_count"] == 3
    assert robot.stop_calls == 3


def test_transport_completion_timeout_is_distinct_and_reasserted(monkeypatch):
    monkeypatch.setattr(
        behavior_manager_module._GuardedTurnMonitor,
        "TRANSPORT_COMPLETION_ALLOWANCE_SECONDS",
        0.10,
    )
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 2)
    assert robot.release_motion.is_set() is False
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "transport_completion_timeout"
    assert result["transport_completion_timed_out"] is True
    assert result["stop_count"] == 3
    assert robot.stop_calls == 3


def test_failed_deadline_stop_response_invalidates_and_reasserts():
    robot = FirstStopRejectsRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 1)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "deadline_stop_failed"
    assert result["reason"] == "deadline_stop_failed"
    assert result["normal_completion"] is False
    assert result["deadline_stop_result"] == {
        "ok": False,
        "error": "stop rejected",
    }
    assert result["stop_fallback_attempted"] is True
    assert result["stop_fallback_result"]["ok"] is True
    assert len(robot.motion_calls) == 1
    assert robot.stop_calls == 2


def test_failed_deadline_stop_exception_is_json_safe_and_reasserted():
    error = RuntimeError("deadline stop unavailable")
    robot = FirstStopFailsRobot(stop_error=error)
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    wait_for_stop(robot, 1)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "deadline_stop_failed"
    assert result["normal_completion"] is False
    assert result["deadline_stop_error"] == str(error)
    assert result["deadline_stop_error_type"] == "RuntimeError"
    assert result["stop_fallback_attempted"] is True
    assert result["stop_fallback_result"]["ok"] is True
    json.dumps(result)
    assert robot.stop_calls == 2


def test_only_one_guarded_turn_owns_the_slot_at_a_time():
    robot = BlockingRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    first_result = []
    first = threading.Thread(
        target=lambda: first_result.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    first.start()
    assert robot.motion_started.wait(timeout=1.0)
    second = behavior.execute_guarded_turn(
        "RIGHT", 0.5, 0.4,
        expected_lidar_session=SESSION,
        now=10.0,
    )
    assert second["permitted"] is False
    assert second["reason"] == "turn_already_active"
    assert second["transport_attempted"] is False
    assert len(robot.motion_calls) == 1
    robot.release_motion.set()
    first.join(timeout=1.0)
    assert first_result[0]["generation_invalidated"] is False
    third = behavior.execute_guarded_turn(
        "RIGHT", 0.5, 0.4,
        expected_lidar_session=SESSION,
        now=10.0,
    )
    assert third["ok"] is True
    assert len(robot.motion_calls) == 2


def test_operator_stop_invalidates_pending_turn_and_reasserts_after_return():
    robot = BlockingRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert robot.motion_started.wait(timeout=1.0)
    stop_result = behavior._execute_stop()
    assert stop_result["ok"] is True
    assert robot.stop_calls == 1
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "operator_stop"
    assert result["active_turn"] is False
    assert result["pending_turn"] is False
    assert robot.stop_calls == 2


def test_operator_stop_before_transport_boundary_prevents_motion(monkeypatch):
    robot = FakeRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    entered = threading.Event()
    release = threading.Event()
    original = behavior_manager_module._GuardedTurnMonitor.begin_transport

    def gated_begin(monitor, dispatch_started):
        entered.set()
        release.wait(timeout=1.0)
        return original(monitor, dispatch_started)

    monkeypatch.setattr(
        behavior_manager_module._GuardedTurnMonitor,
        "begin_transport",
        gated_begin,
    )
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert entered.wait(timeout=1.0)
    assert robot.motion_calls == []
    behavior._execute_stop()
    assert robot.stop_calls == 1
    release.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["transport_attempted"] is False
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "operator_stop"
    assert result["pending_turn"] is False
    assert result["active_turn"] is False
    assert result["stop_count"] == 1
    assert robot.motion_calls == []
    next_result = behavior.execute_guarded_turn(
        "RIGHT", 0.5, 0.4,
        expected_lidar_session=SESSION,
        now=10.0,
    )
    assert next_result["ok"] is True


def test_lidar_invalidation_before_transport_boundary_prevents_motion(monkeypatch):
    robot = FakeRobot()
    state = snapshot()
    world = FakeWorldModel(state)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    entered = threading.Event()
    release = threading.Event()
    original = behavior_manager_module._GuardedTurnMonitor.begin_transport

    def gated_begin(monitor, dispatch_started):
        entered.set()
        release.wait(timeout=1.0)
        return original(monitor, dispatch_started)

    monkeypatch.setattr(
        behavior_manager_module._GuardedTurnMonitor,
        "begin_transport",
        gated_begin,
    )
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, 0.4,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert entered.wait(timeout=1.0)
    state["sectors"]["left"]["state"] = "BLOCKED"
    wait_for_stop(robot, 1)
    release.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["transport_attempted"] is False
    assert result["generation_invalidated"] is True
    assert result["monitor_reason"] == "turn_side_not_clear"
    assert result["stop_count"] == 1
    assert robot.stop_calls == 1
    assert robot.motion_calls == []


def test_transport_after_atomic_boundary_retains_two_stop_protection():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    assert result_box == []
    state["sectors"]["left"]["state"] = "CAUTION"
    wait_for_stop(robot, 1)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["transport_began"] is True
    assert result["transport_attempted"] is True
    assert result["generation_invalidated"] is True
    assert robot.stop_calls == 2


def test_unsafe_transition_while_motion_pending_gets_immediate_and_post_return_stops():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    state["sectors"]["left"]["state"] = "CAUTION"
    wait_for_stop(robot, 1)
    assert robot.release_motion.is_set() is False
    assert robot.stop_calls == 1
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["inhibited"] is True
    assert result["stop_count"] == 2
    assert robot.stop_calls == 2
    assert result["active_turn"] is False
    assert result["pending_turn"] is False


def test_invalidated_turn_is_not_replayed_after_lidar_returns_clear():
    robot = BlockingRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    state["sectors"]["left"]["state"] = "BLOCKED"
    wait_for_stop(robot, 1)
    state["sectors"]["left"]["state"] = "CLEAR"
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert robot.motion_calls and len(robot.motion_calls) == 1
    assert robot.stop_calls == 2


def test_front_blocked_alone_does_not_stop_clear_escape_turn():
    robot = BlockingRobot()
    world, behavior, worker, result_box = run_blocked_turn(
        snapshot(front="BLOCKED"), robot,
    )
    robot.release_motion.set()
    worker.join(timeout=1.0)
    assert result_box[0]["ok"] is True
    assert robot.stop_calls == 0


def test_stale_session_invalid_and_malformed_states_stop_during_turn():
    mutations = (
        lambda state: state.update(age_at_receipt_seconds=0.31),
        lambda state: state.update(producer_session="other"),
        lambda state: state.update(available=False, valid=False),
        lambda state: state["sectors"].update(left=[]),
        lambda state: state["sectors"]["left"].update(state="UNKNOWN"),
        lambda state: state["sectors"].update(front=[]),
        lambda state: state["sectors"]["front"].update(state="UNKNOWN"),
    )
    for mutate in mutations:
        robot = BlockingRobot()
        state = snapshot()
        world, behavior, worker, result_box = run_blocked_turn(state, robot)
        mutate(state)
        wait_for_stop(robot, 1)
        robot.release_motion.set()
        worker.join(timeout=1.0)
        result = result_box[0]
        assert result["generation_invalidated"] is True
        assert robot.stop_calls == 2


def _run_sequence_turn(states, robot=None, duration=0.4):
    robot = robot or BlockingRobot()
    world = SequenceWorldModel(states)
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            behavior.execute_guarded_turn(
                "LEFT", 0.5, duration,
                expected_lidar_session=SESSION,
                now=10.0,
            )
        )
    )
    worker.start()
    assert robot.motion_started.wait(timeout=1.0)
    return robot, world, behavior, worker, result_box


def test_transient_stale_gap_recovers_without_stop():
    stale = snapshot()
    stale["age_at_receipt_seconds"] = 0.31
    fresh = snapshot()
    robot, world, _behavior, worker, result_box = _run_sequence_turn(
        [snapshot(), snapshot(), stale, stale, fresh],
    )
    deadline = time.monotonic() + 1.0
    while len(world.calls) < 5 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(world.calls) >= 5
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is True
    assert result["generation_invalidated"] is False
    assert result["transient_stale_observed"] is True
    assert result["transient_stale_recovered"] is True
    assert result["transient_stale_max_duration_seconds"] <= 0.15
    assert robot.stop_calls == 0


def test_fresh_lidar_resets_stale_grace_for_a_later_gap():
    stale = snapshot()
    stale["age_at_receipt_seconds"] = 0.31
    fresh = snapshot()
    robot, world, _behavior, worker, result_box = _run_sequence_turn(
        [snapshot(), snapshot(), stale, fresh, stale, fresh],
    )
    deadline = time.monotonic() + 1.0
    while len(world.calls) < 6 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(world.calls) >= 6
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is True
    assert result["generation_invalidated"] is False
    assert result["transient_stale_observed"] is True
    assert result["transient_stale_recovered"] is True
    assert robot.stop_calls == 0


def test_persistent_stale_gap_stops_after_single_grace_window():
    stale = snapshot()
    stale["age_at_receipt_seconds"] = 0.31
    robot, world, _behavior, worker, result_box = _run_sequence_turn(
        [snapshot(), snapshot(), stale],
    )
    wait_for_stop(robot, 1)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["ok"] is False
    assert result["generation_invalidated"] is True
    assert result["reason"] == "stale"
    assert result["transient_stale_observed"] is True
    assert result["transient_stale_recovered"] is False
    assert result["transient_stale_max_duration_seconds"] > 0.15
    assert robot.stop_calls == 2


def test_non_stale_failures_are_not_granted_stale_grace():
    stale = snapshot()
    stale["age_at_receipt_seconds"] = 0.31
    mismatched = snapshot()
    mismatched["producer_session"] = "other"
    unavailable = snapshot()
    unavailable.update(available=False, valid=False, reason="unavailable")
    unsafe = snapshot(left="BLOCKED")
    for failure in (mismatched, unavailable, unsafe):
        robot, world, _behavior, worker, result_box = _run_sequence_turn(
            [snapshot(), snapshot(), stale, failure],
        )
        wait_for_stop(robot, 1)
        robot.release_motion.set()
        worker.join(timeout=1.0)
        result = result_box[0]
        assert result["ok"] is False
        assert result["generation_invalidated"] is True
        assert result["reason"] != "stale"
        assert result["transient_stale_observed"] is True
        assert result["transient_stale_recovered"] is False
        assert robot.stop_calls == 2


def test_monitor_stop_exception_is_captured_and_cleanup_completes():
    robot = FirstStopFailsRobot()
    state = snapshot()
    world, behavior, worker, result_box = run_blocked_turn(state, robot)
    state["sectors"]["front_left"]["state"] = "CAUTION"
    wait_for_stop(robot, 1)
    robot.release_motion.set()
    worker.join(timeout=1.0)
    result = result_box[0]
    assert result["generation_invalidated"] is True
    assert result["monitor_running"] is False
    assert result["stop_count"] == 2
    assert result["stop_events"][0]["error"] == "immediate stop unavailable"
    assert robot.stop_calls == 2


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
