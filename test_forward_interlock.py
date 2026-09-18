"""Offline fail-closed interlock and client choke-point tests."""

import threading
import time
from unittest.mock import Mock

import pytest

from robot_bridge.client import RobotBridgeClient
from robot_bridge.forward_interlock import ForwardMotionInterlock, evaluate_lidar_state


def state(*, front="CLEAR", age=0.1, available=True, valid=True, session="s"):
    return {
        "available": available, "valid": valid, "reason": "fresh",
        "effective_age_seconds": age, "producer_session": session,
        "sectors": {"front": {"state": front, "available": True}},
    }


class FakeInterlock:
    def __init__(self, permitted=True):
        self.permitted = permitted
        self.active = False
        self.calls = []

    def begin_positive_dispatch(self, *, streaming):
        self.calls.append(("gate", streaming))
        if not self.permitted:
            raise PermissionError("unsafe")
        return 1

    def finalize_positive_dispatch(self, generation, transport_result):
        self.active = transport_result.get("ok") is True
        return True

    def stop_active(self):
        self.active = False


@pytest.mark.parametrize("front,age,available,valid,reason", [
    ("CLEAR", 0.30, True, True, "fresh_clear"),
    ("CLEAR", 0.300001, True, True, "stale_lidar"),
    ("CLEAR", -0.1, True, True, "stale_lidar"),
    ("CAUTION", 0.1, True, True, "front_not_clear"),
    ("BLOCKED", 0.1, True, True, "front_not_clear"),
    ("UNKNOWN", 0.1, True, True, "front_not_clear"),
    ("CLEAR", 0.1, False, False, "fresh"),
    ("CLEAR", 0.1, True, False, "fresh"),
])
def test_policy(front, age, available, valid, reason):
    permitted, actual = evaluate_lidar_state(state(front=front, age=age, available=available, valid=valid), "s")
    assert permitted is (reason == "fresh_clear")
    assert actual == reason


def test_default_positive_and_bounded_positive_are_denied_without_post():
    client = RobotBridgeClient(base_url="http://robot.invalid")
    client._request = Mock(return_value={"ok": True})
    assert client.streaming_motion(linear_x=0.1)["ok"] is False
    assert client.move_forward()["ok"] is False
    client._request.assert_not_called()


def test_clear_streaming_and_bounded_dispatches():
    client = RobotBridgeClient(base_url="http://robot.invalid", forward_interlock=FakeInterlock())
    client._request = Mock(return_value={"ok": True})
    assert client.streaming_motion(linear_x=0.1)["ok"] is True
    assert client.move_forward()["ok"] is True
    assert client._request.call_count == 2


def test_bounded_forward_uses_pending_guard_and_clears_on_completion():
    current = state()
    reader = Mock(return_value=current)
    interlock = ForwardMotionInterlock(
        reader,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    observed = {}

    def transport(method, path, payload=None):
        del method
        assert path == "/motion"
        observed["pending"] = interlock.status()["pending_forward"]
        observed["payload"] = payload
        return {"ok": True, "forwarded": True}

    client._request = transport
    assert interlock.refresh() == (True, "fresh_clear")
    result = client.move_forward(speed=0.08, seconds=0.50)

    assert result["ok"] is True
    assert observed["pending"] is True
    assert observed["payload"] == {
        "linear_x": 0.08,
        "angular_z": 0.0,
        "duration": 0.5,
    }
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False
    assert reader.call_args.kwargs["expected_session"] == "s"
    interlock.stop()


@pytest.mark.parametrize("front,reason", [
    ("CAUTION", "front_not_clear"),
    ("BLOCKED", "front_not_clear"),
])
def test_bounded_forward_denied_before_transport(front, reason):
    current = state(front=front)
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})
    interlock.refresh()

    result = client.move_forward(speed=0.08, seconds=0.50)

    assert result["ok"] is False
    assert result["error"] == reason
    client._request.assert_not_called()
    interlock.stop()


@pytest.mark.parametrize(("invalidation", "reason"), [
    ("front", "front_not_clear"),
    ("stale", "stale_lidar"),
])
def test_bounded_forward_pending_invalidation_stops_and_does_not_replay(
    invalidation, reason
):
    current = state()
    entered = threading.Event()
    release = threading.Event()
    events = []
    client = RobotBridgeClient(base_url="http://robot.invalid")
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=client.stop,
    )
    client.configure_forward_interlock(interlock)

    def transport(method, path, payload=None):
        del method, payload
        events.append(path)
        if path == "/motion":
            entered.set()
            assert release.wait(2)
            return {"ok": True}
        return {"ok": True}

    client._request = transport
    interlock.refresh()
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "value", client.move_forward(speed=0.08, seconds=0.50)
        )
    )
    thread.start()
    assert entered.wait(1)
    if invalidation == "front":
        current["sectors"]["front"]["state"] = "CAUTION"
    else:
        current["effective_age_seconds"] = 0.31
    interlock.refresh()
    assert interlock.status()["inhibited"] is True
    assert events == ["/motion", "/stop"]
    release.set()
    thread.join(2)

    assert not thread.is_alive()
    assert events == ["/motion", "/stop", "/stop"]
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False
    assert result["value"]["ok"] is False
    assert result["value"]["bounded_forward_invalidated"] is True
    assert result["value"]["reason"] == reason
    assert result["value"]["transport_result"]["ok"] is True
    interlock.stop()


def test_bounded_forward_session_change_fails_closed_during_dispatch():
    current = state()
    entered = threading.Event()
    release = threading.Event()
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=stop,
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )

    def transport(method, path, payload=None):
        del method, payload
        if path == "/motion":
            entered.set()
            assert release.wait(2)
        return {"ok": True}

    client._request = transport
    interlock.refresh()
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "value", client.move_forward(speed=0.08, seconds=0.50)
        )
    )
    thread.start()
    assert entered.wait(1)
    current["producer_session"] = "new-session"
    interlock.refresh()
    assert interlock.status()["reason"] == "producer_session_mismatch"
    assert stop.call_count == 1
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert stop.call_count == 2
    assert result["value"]["ok"] is False
    assert result["value"]["reason"] == "producer_session_mismatch"
    assert result["value"]["bounded_forward_invalidated"] is True
    interlock.stop()


def test_bounded_forward_rejects_angular_or_invalid_duration():
    interlock = Mock()
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})

    mixed = client.motion(
        linear_x=0.08,
        angular_z=0.1,
        duration=0.50,
    )
    invalid = client.motion(
        linear_x=0.08,
        angular_z=0.0,
        duration=0.0,
    )

    assert mixed["error"] == "bounded_forward_requires_zero_angular"
    assert invalid["error"] == "invalid_bounded_forward_duration"
    interlock.begin_positive_dispatch.assert_not_called()
    client._request.assert_not_called()


def test_bounded_transport_failure_is_not_replayed():
    current = state()
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(
        return_value={"ok": False, "error": "bridge rejected"}
    )
    interlock.refresh()

    result = client.move_forward(speed=0.08, seconds=0.50)

    assert result["ok"] is False
    assert result["error"] == "bridge rejected"
    assert client._request.call_count == 1
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False
    interlock.stop()


def test_independent_bounded_requests_get_fresh_epochs_and_tokens():
    current = state()
    reader = Mock(return_value=current)
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(
        reader,
        expected_session="s",
        stop_callback=stop,
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})
    tokens = []
    original_begin = interlock.begin_positive_dispatch

    def record_begin(**kwargs):
        token = original_begin(**kwargs)
        tokens.append(token)
        return token

    interlock.begin_positive_dispatch = record_begin
    interlock.refresh()
    first = client.move_forward(speed=0.08, seconds=0.50)
    assert first["ok"] is True
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False

    # A second request requires a new LiDAR authorization tick and token.
    assert interlock.refresh() == (True, "fresh_clear")
    second = client.move_forward(speed=0.08, seconds=0.50)
    assert second["ok"] is True
    assert tokens[0] != tokens[1]
    assert client._request.call_count == 2
    assert reader.call_count >= 2
    assert all(
        call.kwargs["expected_session"] == "s"
        for call in reader.call_args_list
    )
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False
    interlock.stop()


def test_bounded_request_without_refresh_is_denied_then_refresh_reauthorizes():
    current = state()
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})
    interlock.refresh()

    first = client.move_forward(speed=0.08, seconds=0.50)
    second = client.move_forward(speed=0.08, seconds=0.50)
    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"] == (
        "bounded_forward_authorization_refresh_required"
    )
    assert client._request.call_count == 1

    assert interlock.refresh() == (True, "fresh_clear")
    third = client.move_forward(speed=0.08, seconds=0.50)
    assert third["ok"] is True
    assert client._request.call_count == 2
    interlock.stop()


def test_dispatch_bookkeeping_is_clean_after_repeated_normal_requests():
    current = state()
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})

    for _ in range(10):
        interlock.refresh()
        assert client.move_forward(speed=0.08, seconds=0.50)["ok"]

    assert interlock._dispatch_outcomes == {}
    assert interlock._dispatch_epochs == {}
    assert interlock._invalidation_reasons == {}
    interlock.stop()


def test_changed_session_blocks_later_bounded_request_without_reconfiguration():
    current = state()
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=Mock(return_value={"ok": True}),
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(return_value={"ok": True})
    interlock.refresh()
    assert client.move_forward(speed=0.08, seconds=0.50)["ok"] is True

    current["producer_session"] = "new-session"
    assert interlock.refresh() == (False, "producer_session_mismatch")
    second = client.move_forward(speed=0.08, seconds=0.50)

    assert second["ok"] is False
    assert second["error"] == "producer_session_mismatch"
    assert client._request.call_count == 1
    interlock.stop()


def test_bounded_transport_exception_is_uncertain_and_stops():
    current = state()
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(
        lambda **_: current,
        expected_session="s",
        stop_callback=stop,
    )
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
        forward_interlock=interlock,
    )
    client._request = Mock(side_effect=TimeoutError("delivery uncertain"))
    interlock.refresh()

    with pytest.raises(TimeoutError, match="delivery uncertain"):
        client.move_forward(speed=0.08, seconds=0.50)

    assert client._request.call_count == 1
    assert stop.call_count == 1
    assert interlock.status()["pending_forward"] is False
    assert interlock.status()["active_forward"] is False
    assert interlock.status()["inhibited"] is True
    assert interlock.status()["reason"] == "transport_exception"
    interlock.stop()


@pytest.mark.parametrize("linear_x", [0.0, -0.1])
def test_zero_and_reverse_retain_existing_transport(linear_x):
    client = RobotBridgeClient(base_url="http://robot.invalid")
    client._request = Mock(return_value={"ok": True})
    assert client.motion(linear_x=linear_x)["ok"] is True
    client._request.assert_called_once()


def test_rotation_and_stop_are_not_gated():
    client = RobotBridgeClient(base_url="http://robot.invalid")
    client._request = Mock(return_value={"ok": True})
    assert client.turn_left()["ok"] is True
    assert client.stop()["ok"] is True
    assert [call.args[1] for call in client._request.call_args_list] == ["/motion", "/stop"]


def test_monitor_stops_active_stream_without_new_command():
    current = state()
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=stop, interval=0.01)
    interlock.start()
    deadline = time.time() + 1
    while not interlock.status()["forward_permitted"] and time.time() < deadline:
        time.sleep(0.005)
    generation = interlock.begin_positive_dispatch(streaming=True)
    assert interlock.finalize_positive_dispatch(generation, {"ok": True})
    current["sectors"]["front"]["state"] = "CAUTION"
    deadline = time.time() + 1
    while not stop.called and time.time() < deadline:
        time.sleep(0.005)
    assert stop.called
    assert interlock.status()["inhibited"]
    assert interlock.status()["active_forward"] is False
    interlock.stop()


@pytest.mark.parametrize("unsafe", ["CAUTION", "BLOCKED"])
def test_active_stream_unsafe_transition_explicitly_stops(unsafe):
    current = state()
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=stop)
    interlock.refresh()
    generation = interlock.begin_positive_dispatch(streaming=True)
    interlock.finalize_positive_dispatch(generation, {"ok": True})
    current["sectors"]["front"]["state"] = unsafe
    interlock.refresh()
    assert stop.call_count == 1
    assert interlock.status()["inhibited"]
    interlock.stop()


def test_active_stream_reader_failure_explicitly_stops():
    current = state()
    stop = Mock(return_value={"ok": True})
    failing = {"value": False}
    def reader(**_):
        if failing["value"]:
            raise OSError("reader failed")
        return current
    interlock = ForwardMotionInterlock(reader, expected_session="s", stop_callback=stop)
    interlock.refresh()
    generation = interlock.begin_positive_dispatch(streaming=True)
    interlock.finalize_positive_dispatch(generation, {"ok": True})
    failing["value"] = True
    interlock.refresh()
    assert stop.call_count == 1
    assert interlock.status()["reason"] == "reader_exception"
    interlock.stop()


def test_reader_failure_stops_and_does_not_rearm():
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(lambda **_: (_ for _ in ()).throw(OSError("reader")), expected_session="s", stop_callback=stop, interval=0.01)
    assert interlock.refresh() == (False, "reader_exception")
    assert not interlock.status()["forward_permitted"]
    interlock.stop()


def test_stop_failure_leaves_inhibited_and_new_clear_needs_new_request():
    current = state()
    stop = Mock(side_effect=OSError("stop failed"))
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=stop)
    interlock.refresh()
    generation = interlock.begin_positive_dispatch(streaming=True)
    interlock.finalize_positive_dispatch(generation, {"ok": True})
    current["sectors"]["front"]["state"] = "BLOCKED"
    interlock.refresh()
    assert interlock.status()["inhibited"] and interlock.status()["last_stop_error"]
    current["sectors"]["front"]["state"] = "CLEAR"
    interlock.refresh()
    assert interlock.status()["forward_permitted"]
    interlock.stop()


def test_pending_guard_serializes_inhibition_before_dispatch():
    current = state()
    entered = threading.Event()
    release = threading.Event()
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=stop)
    interlock.refresh()
    generation = interlock.begin_positive_dispatch(streaming=True)
    current["sectors"]["front"]["state"] = "CAUTION"
    assert interlock.refresh() == (False, "front_not_clear")
    assert interlock.status()["inhibited"]
    assert interlock.finalize_positive_dispatch(generation, {"ok": True}) is False
    assert stop.call_count == 2
    interlock.stop()


@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
@pytest.mark.parametrize("stop_failure", [None, "immediate_response", "immediate_exception", "final_response", "final_exception"])
@pytest.mark.parametrize("clear_while_pending", [False, True])
def test_cross_thread_invalidated_transport_has_two_stop_phases(outcome, stop_failure, clear_while_pending):
    current = state()
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    events = []
    reader = Mock(side_effect=lambda **_: current)
    client = RobotBridgeClient(base_url="http://robot.invalid")
    interlock = ForwardMotionInterlock(reader, expected_session="s", stop_callback=client.stop)
    client.configure_forward_interlock(interlock)
    interlock.refresh()
    stop_count = 0

    def transport(method, path, payload=None):
        nonlocal stop_count
        # A separate thread must acquire the lock during every HTTP operation.
        acquired = threading.Event()
        def probe():
            with interlock._lock:
                acquired.set()
        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        assert acquired.wait(1), "interlock lock held during HTTP"
        probe_thread.join(1)
        if path == "/motion":
            events.append("motion_enter")
            entered.set()
            assert release.wait(3)
            events.append("motion_return")
            returned.set()
            if outcome == "exception":
                raise TimeoutError("remote receipt unknown")
            return {"ok": outcome == "success"}
        stop_count += 1
        phase = "final" if returned.is_set() else "immediate"
        events.append(phase + "_stop")
        if stop_failure == phase + "_exception":
            raise OSError(phase + " stop failed")
        if stop_failure == phase + "_response":
            return {"ok": False, "error": phase + " stop failed"}
        return {"ok": True}

    client._request = transport
    result = {}
    def dispatch():
        try:
            result["value"] = client.streaming_motion(linear_x=0.1)
        except Exception as exc:
            result["exception"] = exc
    motion_thread = threading.Thread(target=dispatch)
    motion_thread.start()
    try:
        assert entered.wait(1)
        current["sectors"]["front"]["state"] = "BLOCKED"
        monitor = threading.Thread(target=interlock.refresh)
        monitor.start()
        monitor.join(1)
        assert not monitor.is_alive()
        assert interlock.status()["inhibited"]
        assert stop_count == 1
        assert not returned.is_set() and motion_thread.is_alive()
        interlock.refresh()  # Repeated unsafe ticks cannot add pre-return STOPs.
        assert stop_count == 1
        if clear_while_pending:
            current["sectors"]["front"]["state"] = "CLEAR"
            interlock.refresh()
            assert interlock.status()["inhibited"]
            assert client.streaming_motion(linear_x=0.1)["forwarded"] is False
        reader.reset_mock()
    finally:
        release.set()
        motion_thread.join(2)
    assert not motion_thread.is_alive()
    reader.assert_not_called()
    assert events == ["motion_enter", "immediate_stop", "motion_return", "final_stop"]
    assert stop_count == 2
    if outcome == "exception":
        assert isinstance(result["exception"], TimeoutError)
    else:
        assert result["value"] == {"ok": outcome == "success"}
    status = interlock.status()
    assert not status["active_forward"] and not status["pending_forward"]
    assert status["inhibited"]
    if stop_failure:
        assert "stop failed" in status["last_stop_error"]
    assert client.streaming_motion(linear_x=0.1)["forwarded"] is False
    current["sectors"]["front"]["state"] = "CLEAR"
    interlock.refresh()
    assert interlock.status()["forward_permitted"]
    assert not interlock.status()["active_forward"]
    assert events.count("motion_enter") == 1  # CLEAR never replays.
    client._request = Mock(return_value={"ok": True})
    assert client.streaming_motion(linear_x=0.1)["ok"]
    client._request.assert_called_once()
    assert interlock.status()["active_forward"]
    interlock.stop()


def test_clear_transport_becomes_active_then_unsafe_stops_once():
    current = state()
    client = RobotBridgeClient(base_url="http://robot.invalid")
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=client.stop)
    client.configure_forward_interlock(interlock)
    client._request = Mock(return_value={"ok": True})
    interlock.refresh()
    assert client.streaming_motion(linear_x=0.1)["ok"]
    assert interlock.status()["active_forward"]
    assert not interlock.status()["pending_forward"]
    assert [call.args[1] for call in client._request.call_args_list] == ["/motion"]
    current["sectors"]["front"]["state"] = "BLOCKED"
    monitor = threading.Thread(target=interlock.refresh)
    monitor.start()
    monitor.join(1)
    assert not monitor.is_alive()
    interlock.refresh()
    interlock.stop()
    assert [call.args[1] for call in client._request.call_args_list] == ["/motion", "/stop"]
    assert interlock.status()["inhibited"]
    assert not interlock.status()["active_forward"]


@pytest.mark.parametrize("change", ["reader", "stale"])
def test_cross_thread_pending_failure_stops_and_does_not_replay(change):
    current = state()
    blocked = threading.Event()
    release = threading.Event()
    stop = Mock(return_value={"ok": True})
    failed = {"value": False}
    def reader(**_):
        if failed["value"]:
            if change == "reader":
                raise OSError("reader failed")
            current["effective_age_seconds"] = 0.31
        return current
    interlock = ForwardMotionInterlock(reader, expected_session="s", stop_callback=stop)
    interlock.refresh()
    client = RobotBridgeClient(base_url="http://robot.invalid", forward_interlock=interlock)
    def transport(method, path, payload=None):
        del method, payload
        if path == "/motion":
            blocked.set()
            release.wait(2)
        return {"ok": True}
    client._request = transport
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.setdefault("result", client.streaming_motion(linear_x=0.1)))
    thread.start()
    assert blocked.wait(1)
    failed["value"] = True
    monitor = threading.Thread(target=interlock.refresh)
    monitor.start()
    monitor.join(1)
    assert stop.call_count == 1
    assert interlock.status()["inhibited"]
    release.set()
    thread.join(1)
    assert interlock.status()["active_forward"] is False
    assert not interlock.status()["forward_permitted"]
    interlock.stop()


def test_motion_dispatch_does_not_read_lidar_or_world_model():
    reader = Mock(return_value=state())
    interlock = ForwardMotionInterlock(reader, expected_session="s", stop_callback=Mock())
    interlock.refresh()
    reader.reset_mock()
    client = RobotBridgeClient(base_url="http://robot.invalid", forward_interlock=interlock)
    client._request = Mock(return_value={"ok": True})
    assert client.streaming_motion(linear_x=0.1, angular_z=0.2)["ok"] is True
    reader.assert_not_called()
    client._request.assert_called_once()
    interlock.stop()


def test_new_clear_request_can_rearm_without_replay():
    current = state(front="CAUTION")
    stop = Mock(return_value={"ok": True})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session="s", stop_callback=stop)
    interlock.refresh()
    assert not interlock.status()["forward_permitted"]
    current["sectors"]["front"]["state"] = "CLEAR"
    interlock.refresh()
    client = RobotBridgeClient(base_url="http://robot.invalid", forward_interlock=interlock)
    client._request = Mock(return_value={"ok": True})
    assert client.streaming_motion(linear_x=0.1)["ok"] is True
    assert client._request.call_count == 1
    interlock.stop()


def test_status_is_observable():
    interlock = ForwardMotionInterlock(lambda **_: state(), expected_session="s", stop_callback=Mock())
    status = interlock.status()
    assert status["configured"] and status["inhibited"] and status["front_state"] == "UNKNOWN"
    interlock.stop()


def test_reason_and_malformed_state_are_fail_closed():
    not_fresh = state()
    not_fresh["reason"] = "cached"
    assert evaluate_lidar_state(not_fresh, "s") == (False, "not_fresh")
    malformed = state()
    malformed["sectors"] = []
    assert evaluate_lidar_state(malformed, "s") == (False, "malformed_lidar_state")
    contradictory = state()
    contradictory["sectors"]["front"]["available"] = False
    assert evaluate_lidar_state(contradictory, "s") == (False, "front_not_clear")
