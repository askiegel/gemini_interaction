"""Bounded wait/resume behavior for a blocked Find Marvin scan turn."""

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime


SESSION = "clearance-session"


def _blocked(*, age=0.05, session=SESSION, reason="rotational_protected_region_violated"):
    footprint = {
        "permitted": False,
        "reason": reason,
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
        },
    }
    controller = {
        "completed": True, "arrived_at_marvin": False,
        "history": [terminal],
        "clearance_wait": {
            "decision": "clearance_wait_timeout",
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
