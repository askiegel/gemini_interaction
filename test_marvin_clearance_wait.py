"""Bounded wait/resume behavior for a blocked Find Marvin scan turn."""

from copy import deepcopy

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime


SESSION = "clearance-session"


def _blocked(*, age=0.05, session=SESSION, reason="rotational_protected_region_violated"):
    footprint = {
        "permitted": False,
        "reason": reason,
        "model": "base_link_circular_rotational_envelope",
        "protected_radius_m": 0.45,
        "geometry": {
            "valid": True, "frame_id": behavior_manager_module.EXPECTED_LIDAR_FRAME,
            "points": [], "sectors": {},
        },
        "violating_point": {"x_m": 0.418, "y_m": 0.0},
    }
    return {
        "ok": False,
        "permitted": False,
        "reason": reason,
        "validation_reason": reason,
        "producer_session": session,
        "effective_age_seconds": age,
        "rotational_swept_footprint": footprint,
        "transport_attempted": False,
    }


class _Robot:
    def __init__(self, status=None, stop_result=None):
        self._status = status or {
            "ok": True,
            "ros_ready": True,
            "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
        }
        self.stop_result = stop_result or {"ok": True}
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return dict(self.stop_result)

    def status(self):
        return dict(self._status)


def _manager():
    manager = BehaviorManager(robot_client=_Robot())
    manager.lidar_session = SESSION
    manager.begin_find_marvin_room_scan("mission", source_frame_stamp_ns=100)
    manager._room_scan_update(scan_turn_index=23)
    return manager


def _planner(monkeypatch):
    monkeypatch.setattr(
        behavior_manager_module,
        "plan_marvin_search_step",
        lambda *_args, **_kwargs: {
            "ok": True, "selected_search_action": "turn_left",
            "selected_identity_id": "marvin", "reason": "continue_search",
        },
    )


def _monitor_stopped(*, age=0.05, session=SESSION, reason="rotational_protected_region_violated"):
    monitor_validation = _blocked(age=age, session=session, reason=reason)
    return {
        "ok": False, "permitted": True, "reason": reason,
        "validation_reason": "rotational_swept_footprint_clear",
        "rotational_swept_footprint": {
            "permitted": True, "reason": "rotational_swept_footprint_clear",
            "model": "base_link_circular_rotational_envelope",
            "protected_radius_m": 0.45,
        },
        "monitor_reason": reason, "generation_invalidated": True,
        "transport_began": True, "transport_accepted": True,
        "transport_returned": True, "transport_result": {"ok": True},
        "delivery_uncertain": False, "transport_error": None,
        "stop_fallback_attempted": True,
        "stop_fallback_result": {"ok": True}, "stop_fallback_error": None,
        "monitor_validation": monitor_validation,
        "stop_events": [{
            "source": "monitor", "result": {"ok": True},
            "monitor_validation": monitor_validation,
        }],
    }


def test_rotational_occupancy_waits_then_resumes_same_left_index(monkeypatch):
    manager = _manager()
    _planner(monkeypatch)
    now = {"value": 100.0}
    monkeypatch.setattr(behavior_manager_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        behavior_manager_module.time,
        "sleep",
        lambda duration: now.__setitem__("value", now["value"] + duration),
    )
    checks = []

    def guarded(direction, speed, duration, **kwargs):
        checks.append((direction, speed, duration, kwargs))
        return _blocked() if len(checks) == 1 else {
            "ok": True, "permitted": True, "confirmed_forwarded": True,
            "reason": "rotational_swept_footprint_clear",
        }

    monkeypatch.setattr(manager, "execute_guarded_turn", guarded)
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )

    assert result["ok"] is True and result["motion_executed"] is True
    assert result["pending_scan_turn_index"] == 23
    assert result["clearance_recheck_count"] == 1
    assert checks[0][:3] == checks[1][:3] == ("LEFT", 0.25, 1.0)
    assert checks[1][3]["now"] is None
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23
    assert manager._room_scan_snapshot()["clearance_wait_active"] is False
    assert manager.robot.stop_calls == 1


def test_stays_blocked_until_deadline_then_safe_timeout_without_advancing(monkeypatch):
    manager = _manager()
    _planner(monkeypatch)
    manager.MARVIN_CLEARANCE_RECHECK_INTERVAL_SECONDS = 30.0
    now = {"value": 500.0}
    monkeypatch.setattr(behavior_manager_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        behavior_manager_module.time,
        "sleep",
        lambda duration: now.__setitem__("value", now["value"] + duration),
    )
    calls = []
    monkeypatch.setattr(
        manager,
        "execute_guarded_turn",
        lambda *_args, **kwargs: (
            calls.append(kwargs)
            or (_blocked() if not kwargs.get("validate_only") else _blocked())
        ),
    )
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )

    assert result["ok"] is True
    assert result["clearance_wait_timed_out"] is True
    assert result["reason"] == "find_marvin_clearance_wait_timeout"
    assert result["clearance_wait_elapsed_seconds"] == 60.0
    assert result["pending_scan_turn_index"] == 23
    assert result["bridge_status"]["motion"] == {
        "linear_x": 0, "angular_z": 0, "streaming": False,
    }
    assert len(calls) == 3
    assert calls[-1]["validate_only"] is True
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23
    assert manager._room_scan_snapshot()["clearance_wait_active"] is False
    assert manager.robot.stop_calls == 3  # each blocked check plus deadline confirmation


def test_active_monitor_stop_waits_then_retries_same_index(monkeypatch):
    manager = _manager()
    _planner(monkeypatch)
    now = {"value": 20.0}
    monkeypatch.setattr(behavior_manager_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        behavior_manager_module.time, "sleep",
        lambda duration: now.__setitem__("value", now["value"] + duration),
    )
    calls = []

    def guarded(direction, speed, duration, **kwargs):
        calls.append((direction, speed, duration, kwargs))
        if len(calls) == 1:
            return _monitor_stopped()
        return {
            "ok": True, "permitted": True, "confirmed_forwarded": True,
            "reason": "rotational_swept_footprint_clear",
        }

    monkeypatch.setattr(manager, "execute_guarded_turn", guarded)
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    scan = manager._room_scan_snapshot()
    assert result["ok"] is True and result["motion_executed"] is True
    assert result["clearance_wait_origin"] == "active_turn_monitor"
    assert result["interrupted_turn_detected"] is True
    assert result["interrupted_turn_monitor_reason"] == "rotational_protected_region_violated"
    assert result["pending_scan_turn_index"] == 23
    assert calls[0][:3] == calls[1][:3] == ("LEFT", 0.25, 1.0)
    assert calls[1][3]["now"] is None
    assert manager.robot.stop_calls == 1
    assert scan["scan_turn_index"] == 23
    assert scan["clearance_wait_active"] is False
    assert scan["clearance_wait_started_monotonic"] is None


def test_active_monitor_wait_classifier_fails_closed_on_uncertain_evidence():
    accepted = _monitor_stopped()
    assert BehaviorManager._is_active_turn_rotational_clearance_stop(accepted, SESSION)
    variants = []
    for key, value in (
        ("delivery_uncertain", True),
        ("transport_returned", False),
        ("transport_accepted", False),
        ("stop_fallback_result", {"ok": False}),
        ("monitor_reason", "lidar_stale"),
    ):
        variant = deepcopy(accepted)
        variant[key] = value
        variants.append(variant)
    for age, session, point in (
        (0.31, SESSION, {"x_m": 0.1, "y_m": 0.0}),
        (0.05, "other", {"x_m": 0.1, "y_m": 0.0}),
        (0.05, SESSION, None),
    ):
        variant = deepcopy(accepted)
        validation = variant["monitor_validation"]
        validation["effective_age_seconds"] = age
        validation["producer_session"] = session
        validation["rotational_swept_footprint"]["violating_point"] = point
        variant["stop_events"][0]["monitor_validation"] = deepcopy(validation)
        variants.append(variant)
    assert all(
        not BehaviorManager._is_active_turn_rotational_clearance_stop(item, SESSION)
        for item in variants
    )


def test_lost_mission_authority_after_monitor_stop_does_not_enter_wait(monkeypatch):
    manager = _manager()
    manager.execution_authorization_provider = lambda: False
    _planner(monkeypatch)
    monkeypatch.setattr(manager, "execute_guarded_turn", lambda *_a, **_k: _monitor_stopped())
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    assert result["reason"] == "find_marvin_clearance_wait_preempted"
    assert manager._room_scan_snapshot()["clearance_wait_active"] is False
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23


def test_repeated_active_monitor_stop_keeps_original_deadline(monkeypatch):
    manager = _manager()
    _planner(monkeypatch)
    manager.MARVIN_CLEARANCE_RECHECK_INTERVAL_SECONDS = 35.0
    now = {"value": 100.0}
    monkeypatch.setattr(behavior_manager_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        behavior_manager_module.time, "sleep",
        lambda duration: now.__setitem__("value", now["value"] + duration),
    )
    calls = []

    def guarded(_direction, _speed, _duration, **kwargs):
        calls.append(kwargs)
        if len(calls) in (1, 2):
            if len(calls) == 2:
                now["value"] += 5.0
            return _monitor_stopped()
        return _blocked()

    monkeypatch.setattr(manager, "execute_guarded_turn", guarded)
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    assert result["clearance_wait_timed_out"] is True
    assert result["clearance_wait_elapsed_seconds"] == 60.0
    assert result["clearance_wait_origin"] == "active_turn_monitor"
    assert result["pending_scan_turn_index"] == 23
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23
    assert len(calls) == 3 and calls[-1]["validate_only"] is True


def test_stale_or_malformed_safety_result_does_not_enter_clearance_wait(monkeypatch):
    for unsafe in (
        _blocked(age=0.31),
        _blocked(session="other"),
        _blocked(reason="invalid_lidar_geometry"),
        {**_blocked(), "rotational_swept_footprint": {"permitted": False}},
    ):
        manager = _manager()
        _planner(monkeypatch)
        calls = []
        monkeypatch.setattr(manager, "execute_guarded_turn", lambda *_a, **_k: calls.append(True) or unsafe)
        result = manager.execute_marvin_search_step(
            {"state": "SEARCHING"}, scan_turn_index=23,
            selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
        )
        assert result["ok"] is False
        assert len(calls) == 1
        assert manager._room_scan_snapshot()["clearance_wait_active"] is False
        assert manager.robot.stop_calls == 0
        assert manager._room_scan_snapshot()["scan_turn_index"] == 23


def test_stop_or_zero_failure_prevents_wait_rechecks(monkeypatch):
    manager = _manager()
    manager.robot = _Robot(stop_result={"ok": False})
    _planner(monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "execute_guarded_turn", lambda *_a, **_k: calls.append(True) or _blocked())
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    assert result["reason"] == "clearance_wait_stop_failed"
    assert len(calls) == 1
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23

    manager = _manager()
    manager.robot = _Robot(status={
        "ok": True, "ros_ready": True,
        "motion": {"linear_x": 0, "angular_z": 0.1, "streaming": False},
    })
    _planner(monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "execute_guarded_turn", lambda *_a, **_k: calls.append(True) or _blocked())
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    assert result["reason"] == "clearance_wait_bridge_not_zero"
    assert len(calls) == 1
    assert manager._room_scan_snapshot()["scan_turn_index"] == 23

    manager = _manager()
    manager.robot = _Robot(stop_result={"ok": False})
    _planner(monkeypatch)
    calls = []
    monkeypatch.setattr(
        manager, "execute_guarded_turn",
        lambda *_a, **_k: calls.append(True) or _monitor_stopped(),
    )
    result = manager.execute_marvin_search_step(
        {"state": "SEARCHING"}, scan_turn_index=23,
        selected_identity_id="marvin", preview_result={"source_frame_stamp_ns": 123},
    )
    assert result["reason"] == "clearance_wait_stop_failed"
    assert len(calls) == 1
    assert manager._room_scan_snapshot()["clearance_wait_active"] is False


def test_new_mission_clears_clearance_wait_and_pending_index():
    manager = _manager()
    manager._room_scan_update(
        clearance_wait_active=True,
        clearance_wait_started_monotonic=123.0,
        pending_scan_turn_index=23,
        pending_scan_direction="LEFT",
        search_state="FIND_MARVIN_WAITING_FOR_CLEARANCE",
    )
    fresh = manager.begin_find_marvin_room_scan("mission-b", source_frame_stamp_ns=900)
    assert fresh["scan_turn_index"] == 0
    assert fresh["clearance_wait_active"] is False
    assert fresh["clearance_wait_started_monotonic"] is None
    assert fresh["pending_scan_turn_index"] is None
    assert fresh["pending_scan_direction"] is None


def test_runtime_accepts_only_explicit_stopped_clearance_timeout():
    terminal = {
        "route": "search", "action_budget_consumed": False,
        "pending_scan_turn_index": 23,
        "stop_result": {"ok": True},
        "search_step_result": {
            "clearance_wait_timed_out": True,
            "motion_executed": False,
            "reason": "find_marvin_clearance_wait_timeout",
            "clearance_wait_origin": "pre_turn",
        },
    }
    controller = {
        "completed": True, "arrived_at_marvin": False,
        "history": [terminal],
        "clearance_wait": {
            "decision": "clearance_wait_timeout",
            "clearance_wait_origin": "pre_turn",
            "pending_scan_turn_index": 23,
            "bridge_status": {
                "ok": True, "ros_ready": True,
                "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
            },
        },
    }
    assert CognitiveRuntime._marvin_clearance_timeout_is_safe(controller)
    controller["clearance_wait"]["bridge_status"]["motion"]["angular_z"] = 0.2
    assert not CognitiveRuntime._marvin_clearance_timeout_is_safe(controller)


def test_runtime_accepts_active_monitor_timeout_with_attempted_turn_accounting():
    terminal = {
        "route": "search", "action_budget_consumed": True,
        "pending_scan_turn_index": 2, "stop_result": {"ok": True},
        "search_step_result": {
            "clearance_wait_timed_out": True, "motion_executed": False,
            "reason": "find_marvin_clearance_wait_timeout",
            "clearance_wait_origin": "active_turn_monitor",
            "interrupted_turn_detected": True,
            "interrupted_turn_monitor_reason": "rotational_protected_region_violated",
        },
    }
    controller = {
        "completed": True, "arrived_at_marvin": False, "history": [terminal],
        "clearance_wait": {
            "decision": "clearance_wait_timeout",
            "clearance_wait_origin": "active_turn_monitor",
            "pending_scan_turn_index": 2,
            "bridge_status": {
                "ok": True, "ros_ready": True,
                "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
            },
        },
    }
    assert CognitiveRuntime._marvin_clearance_timeout_is_safe(controller)
    terminal["search_step_result"]["interrupted_turn_detected"] = False
    assert not CognitiveRuntime._marvin_clearance_timeout_is_safe(controller)
