"""Offline tests for the single Runtime-owned Find Marvin admission view."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


def lidar_state(front_state="CLEAR", *, age=0.08, session="lidar-session",
                valid=True, reason="fresh", worker_error=None):
    sectors = {
        name: {"valid_sample_count": 5}
        for name in ("front", "front_left", "left", "rear_left",
                     "rear", "rear_right", "right", "front_right")
    }
    sectors["front"] = {
        "available": True, "state": front_state, "valid_sample_count": 5,
    }
    return {
        "available": valid,
        "valid": valid,
        "reason": reason,
        "effective_age_seconds": age,
        "acquisition_sequence": 42,
        "producer_session": session,
        "worker_error": worker_error,
        "sectors": sectors,
        "local_motion_geometry": {"valid": True, "sectors": sectors},
    }


def make_runtime(state=None, *, front_state="CLEAR", cached_reason="fresh_clear",
                 interlock_session="lidar-session", worker_error=None):
    runtime = object.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime._lidar_error = None
    runtime.last_error = None
    runtime.running = True
    runtime.lidar_worker = SimpleNamespace(
        session="lidar-session", running=True, sequence=42,
        last_error=worker_error,
    )
    state = state or lidar_state(
        front_state, worker_error=worker_error,
    )
    runtime.world_model = SimpleNamespace(
        robot_state={"runtime_state": "IDLE"},
        get_lidar_obstacles=Mock(return_value=state),
    )
    runtime.mission_manager = SimpleNamespace(
        get_active_mission=Mock(return_value=None),
        get_queue=Mock(return_value=[]),
    )
    runtime.forward_interlock = SimpleNamespace(status=Mock(return_value={
        "configured": True,
        "monitor_running": True,
        "producer_session": interlock_session,
        "active_forward": False,
        "pending_forward": False,
        "reason": cached_reason,
        "forward_permitted": cached_reason == "fresh_clear",
    }))
    runtime.robot_client = SimpleNamespace(status=Mock(return_value={
        "ok": True,
        "status": "READY",
        "ros_ready": True,
        "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
    }))
    return runtime


def test_snapshot_reads_one_lidar_sample_and_returns_decision_evidence():
    runtime = make_runtime(cached_reason="stale_lidar")

    snapshot = runtime.get_find_marvin_admission_snapshot()

    runtime.world_model.get_lidar_obstacles.assert_called_once_with(
        expected_session="lidar-session",
    )
    runtime.forward_interlock.status.assert_called_once_with()
    runtime.robot_client.status.assert_called_once_with()
    assert snapshot["admission_ready"] is True
    assert snapshot["lidar"]["acquisition_sequence"] == 42
    assert snapshot["lidar"]["age_seconds"] == 0.08
    assert snapshot["forward_interlock"]["age_seconds"] == 0.08
    assert snapshot["lidar"]["producer_session"] == "lidar-session"
    assert snapshot["forward_interlock"]["producer_session"] == "lidar-session"
    assert snapshot["forward_interlock"]["forward_permitted"] is True
    assert snapshot["forward_interlock"]["monitor_reported_reason"] == "stale_lidar"
    assert snapshot["evaluation_timestamp"]
    assert snapshot["reasons"] == []


def test_read_only_runtime_endpoint_returns_the_same_snapshot():
    runtime = make_runtime()
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/admission-snapshot"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.send_json = Mock()

    handler.do_GET()

    status, payload = handler.send_json.call_args.args
    assert status == 200
    assert payload["admission_ready"] is True
    runtime.world_model.get_lidar_obstacles.assert_called_once()
    runtime.forward_interlock.status.assert_called_once()


@pytest.mark.parametrize("front_state", ["CLEAR", "CAUTION", "BLOCKED"])
def test_valid_front_caution_states_do_not_block_normal_admission(front_state):
    snapshot = make_runtime(front_state=front_state).get_find_marvin_admission_snapshot()

    assert snapshot["admission_ready"] is True
    assert snapshot["lidar"]["front_state"] == front_state
    assert snapshot["forward_interlock"]["front_state"] == front_state


def test_unknown_front_state_fails_closed():
    snapshot = make_runtime(front_state="UNKNOWN").get_find_marvin_admission_snapshot()

    assert snapshot["admission_ready"] is False
    assert "LiDAR front sector is unavailable" in snapshot["reasons"]
    assert snapshot["forward_interlock"]["reason"] == "front_unavailable_or_unknown"


def test_stale_lidar_reports_exact_age_and_rejects():
    stale = lidar_state(age=0.304, valid=False, reason="stale")
    snapshot = make_runtime(stale).get_find_marvin_admission_snapshot()

    assert snapshot["admission_ready"] is False
    assert snapshot["lidar"]["age_seconds"] == 0.304
    assert snapshot["lidar"]["reason"] == "stale"
    assert "LiDAR is not fresh" in snapshot["reasons"]
    assert snapshot["forward_interlock"]["reason"] == "stale"


def test_session_mismatch_is_rejected_and_diagnosed():
    mismatch = lidar_state(session="old-session")
    snapshot = make_runtime(mismatch, interlock_session="old-session").get_find_marvin_admission_snapshot()

    assert snapshot["admission_ready"] is False
    assert snapshot["lidar"]["session_matches"] is False
    assert snapshot["forward_interlock"]["session_matches"] is False
    assert "LiDAR producer session does not match" in snapshot["reasons"]
    assert snapshot["forward_interlock"]["reason"] == "producer_session_mismatch"


def test_worker_error_is_preserved_and_rejects_admission():
    failed = lidar_state(valid=False, reason="telemetry_timeout",
                         worker_error="serial read timed out")
    runtime = make_runtime(failed, worker_error="serial read timed out")

    snapshot = runtime.get_find_marvin_admission_snapshot()

    assert snapshot["admission_ready"] is False
    assert snapshot["lidar"]["reason"] == "telemetry_timeout"
    assert snapshot["lidar"]["worker_error"] == "serial read timed out"
    assert "LiDAR is unavailable" in snapshot["reasons"]
