"""Offline reporting checks: loopback HTTP only, no robot/service/motion calls."""

import json
import threading
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mission_manager import MissionManager
from runtime import CognitiveRuntime
from runtime_api import create_server
from runtime_reporting import status_summary
from voice_relay.server import VoiceRelayHandler
from diagnostics import DiagnosticsManager


class RetainedEvidence(dict):
    def __deepcopy__(self, _memo):
        raise AssertionError("frequent reporting copied retained evidence")

    def items(self):
        raise AssertionError("frequent reporting traversed retained evidence")


def reporting_runtime():
    runtime = CognitiveRuntime.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime.started_at = None
    runtime.running = True
    runtime.mission_manager = MissionManager()
    runtime.world_model = SimpleNamespace(robot_state={"runtime_state": "IDLE"})
    runtime.last_result = {"mission_id": "mission-old", "state": "BLOCKED", "reason": "stale",
                           "history": RetainedEvidence(), "progress_diagnostics": RetainedEvidence()}
    runtime.tracking_state = {"active": False, "bbox": {"x1": 10, "x2": 20}, "state": "STOPPED"}
    runtime.last_error = None
    runtime.forward_interlock = None
    runtime._lidar_status = Mock(return_value={"available": True, "valid": True,
        "effective_age_seconds": .06, "acquisition_sequence": 123, "producer_session": "producer"})
    runtime._retain_marvin_diagnostic = Mock(side_effect=AssertionError("diagnostic snapshot requested"))
    runtime.robot_client = Mock()
    return runtime


def test_real_summary_never_touches_retained_evidence_or_robot():
    runtime = reporting_runtime()
    result = runtime.get_status_summary()
    assert result["last_result"]["state"] == "BLOCKED"
    assert result["last_result"]["diagnostics_available"] is True
    assert "history" not in result["last_result"]
    assert "progress_diagnostics" not in result["last_result"]
    assert "marvin_progress_diagnostics" not in result
    assert result["lidar_perception"]["acquisition_sequence"] == 123
    assert result["tracking"]["bbox"] == {"x1": 10, "x2": 20}
    assert len(json.dumps(result)) < 2000
    runtime._retain_marvin_diagnostic.assert_not_called()
    assert runtime.robot_client.mock_calls == []


def test_summary_queue_is_bounded_but_count_is_exact():
    runtime = reporting_runtime()
    runtime.mission_manager.mission_queue = [SimpleNamespace(to_dict=lambda: {
        "mission_id": "queued", "history": RetainedEvidence()}) for _ in range(40)]
    result = runtime.get_status_summary()
    assert result["queue_count"] == 40
    assert len(result["queue"]) == 20
    assert all("history" not in mission for mission in result["queue"])


def test_result_projection_keeps_legacy_wrapped_summary_without_history():
    result = status_summary({"last_result": {"result": {"state": "ARRIVED",
        "history": RetainedEvidence()}, "history": RetainedEvidence()}, "queue": []})
    assert result["last_result"]["result"] == {"state": "ARRIVED"}


@pytest.fixture
def api():
    runtime = reporting_runtime()
    runtime.get_status = Mock(return_value={"ok": True, "marvin_progress_diagnostics": {
        "actions": [{"source_frame_stamp_ns": 1791252377488820744, "raw_points": [1, 2, 3]}]}})
    server = create_server(runtime=runtime, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield runtime, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("endpoint", ["/status", "/missions"])
def test_frequent_api_uses_summary_without_full_status(api, endpoint):
    runtime, url = api
    with urllib.request.urlopen(url + endpoint, timeout=2) as response:
        result = json.load(response)
        assert response.headers["Cache-Control"] == "no-store"
    runtime.get_status.assert_not_called()
    assert result["last_result"]["reason"] == "stale"
    assert "progress_diagnostics" not in result["last_result"]


def test_full_diagnostics_remain_available_on_explicit_request(api):
    runtime, url = api
    with urllib.request.urlopen(url + "/status?details=1", timeout=2) as response:
        result = json.load(response)
        assert response.headers["Cache-Control"] == "no-store"
    runtime.get_status.assert_called_once()
    action = result["marvin_progress_diagnostics"]["actions"][0]
    assert action["raw_points"] == [1, 2, 3]
    assert action["source_frame_stamp_ns"] == "1791252377488820744"


def test_submission_acknowledgement_does_not_copy_prior_diagnostics(api):
    # The server and submission are entirely fake; no mission or motion runs.
    runtime, url = api
    runtime.submit_text = Mock(return_value={"intent": {"intent": "STOP"},
        "mission": {"mission_id": "offline-fake", "status": "COMPLETED"}})
    request = urllib.request.Request(url + "/missions", method="POST",
        headers={"Content-Type": "application/json"}, data=b'{"command":"offline-fake"}')
    with urllib.request.urlopen(request, timeout=2) as response:
        result = json.load(response)
        assert response.status == 202
    assert result["accepted"] is True
    assert result["mission"]["mission_id"] == "offline-fake"
    assert "progress_diagnostics" not in result["runtime"]["last_result"]
    runtime.get_status.assert_not_called()


def test_relay_status_does_not_make_redundant_missions_request():
    handler = object.__new__(VoiceRelayHandler)
    summary = reporting_runtime().get_status_summary()

    def fetch(_method, url, **_kwargs):
        assert not url.endswith("/missions")
        data = summary if ":8770/status" in url else {"ok": True, "ros_ready": True,
            "motion": {"linear_x": 0, "angular_z": 0, "streaming": False}}
        return {"ok": True, "status_code": 200, "data": data, "error": None}

    with patch("voice_relay.server.request_json", side_effect=fetch) as request:
        result = handler.dashboard_status()
    assert request.call_count == 3
    assert result["missions"]["last_result"]["reason"] == "stale"
    assert result["robot"]["motion"]["streaming"] is False


def test_relay_details_proxy_is_explicit_and_read_only():
    handler = object.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/runtime-details"
    handler.send_json = Mock()
    with patch("voice_relay.server.request_json", return_value={"ok": True,
            "status_code": 200, "data": {"heavy": "retained"}, "error": None}) as request:
        handler.do_GET()
    assert request.call_args.args[0] == "GET"
    assert request.call_args.args[1].endswith("/status?details=1")
    assert handler.send_json.call_args.args == (200, {"heavy": "retained"})


def test_relay_json_has_no_store_header():
    handler = object.__new__(VoiceRelayHandler)
    handler.send_response = Mock()
    handler.send_cors_headers = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.wfile = Mock()
    handler.send_json(200, {"available": True})
    assert ("Cache-Control", "no-store") in [call.args for call in handler.send_header.call_args_list]


def test_health_diagnostics_use_summary_and_preserve_full_queue_count(tmp_path):
    runtime = reporting_runtime()
    runtime.get_status = Mock(side_effect=AssertionError("full reporting called"))
    runtime.loop_interval = .1
    runtime.world_model.get_entities = lambda: []
    runtime.mission_manager.mission_queue = [SimpleNamespace(to_dict=lambda: {
        "mission_id": "queued"}) for _ in range(40)]
    config = SimpleNamespace(get_config=lambda: {}, robot_bridge_url="http://invalid",
                             camera_relay_url="http://invalid")
    manager = DiagnosticsManager(runtime, config, project_dir=tmp_path)
    manager._probe_json = Mock(return_value={"online": True, "data": {}})
    manager._git_value = Mock(return_value="")
    result = manager.collect()
    assert result["runtime"]["queue_length"] == 40
    runtime.get_status.assert_not_called()
