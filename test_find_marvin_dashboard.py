"""Offline contract tests for the dedicated Find Marvin dashboard control."""

from pathlib import Path
from unittest.mock import Mock, patch

from voice_relay.server import VoiceRelayHandler


ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "voice_relay" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "voice_relay" / "operator_console.css").read_text(encoding="utf-8")
JS = (ROOT / "voice_relay" / "operator_console.js").read_text(encoding="utf-8")
SERVER = (ROOT / "voice_relay" / "server.py").read_text(encoding="utf-8")


def safe_admission_snapshot():
    return {
        "ok": True,
        "evaluation_timestamp": "2026-10-02T12:00:00+00:00",
        "admission_ready": True,
        "reasons": [],
        "runtime": {
            "running": True,
            "state": "IDLE",
            "last_error": None,
            "active_mission": None,
            "queue": [],
            "queue_count": 0,
        },
        "lidar": {
            "running": True, "available": True, "valid": True,
            "reason": "fresh", "age_seconds": 0.08,
            "acquisition_sequence": 17,
            "producer_session": "lidar-session",
            "expected_session": "lidar-session", "session_matches": True,
            "worker_error": None, "front_state": "CLEAR",
            "local_motion_geometry_valid": True,
            "required_sectors_valid": True,
        },
        "forward_interlock": {
            "configured": True, "monitor_running": True,
            "age_seconds": 0.08, "reason": "fresh_clear",
            "front_state": "CLEAR", "forward_permitted": True,
            "producer_session": "lidar-session", "session_matches": True,
            "active_forward": False, "pending_forward": False,
        },
        "bridge": {
            "connected": True,
            "status": "READY",
            "ros_ready": True,
            "linear_x": 0, "angular_z": 0, "streaming": False,
        },
    }


def test_dedicated_route_constructs_one_fixed_find_object_mission():
    handler = object.__new__(VoiceRelayHandler)
    accepted = {
        "ok": True,
        "accepted": True,
        "mission": {"mission_id": "mission-marvin"},
    }
    with patch(
        "voice_relay.server.request_json",
        side_effect=[
            {"ok": True, "status_code": 200,
             "data": safe_admission_snapshot(), "error": None},
            {"ok": True, "status_code": 202,
             "data": accepted, "error": None},
        ],
    ) as request:
        status_code, payload = handler.submit_find_marvin(execute=True)

    assert status_code == 202
    assert payload == accepted
    assert request.call_count == 2
    assert request.call_args_list[0].args == (
        "GET",
        "http://127.0.0.1:8770/find-marvin/admission-snapshot",
    )
    assert request.call_args_list[0].kwargs == {"timeout": 5.0}
    assert request.call_args_list[1].args == (
        "POST",
        "http://127.0.0.1:8770/missions",
    )
    assert request.call_args_list[1].kwargs == {
        "payload": {
            "source_text": "Find Marvin.",
            "intent": {
                "intent": "FIND_OBJECT",
                "speech": "Find Marvin.",
                    "target": "marvin",
            },
        },
        "timeout": 15.0,
    }


def test_preflight_uses_one_runtime_snapshot_and_returns_its_exact_diagnostics():
    handler = object.__new__(VoiceRelayHandler)
    snapshot = safe_admission_snapshot()
    snapshot.update(admission_ready=False, reasons=["LiDAR is not fresh"])
    snapshot["lidar"].update(
        available=False, valid=False, reason="stale", age_seconds=0.304,
        acquisition_sequence=21, worker_error="telemetry timeout",
    )
    snapshot["forward_interlock"].update(
        forward_permitted=False, reason="stale", age_seconds=0.304,
    )
    with patch(
        "voice_relay.server.request_json",
        return_value={"ok": True, "status_code": 200,
                      "data": snapshot, "error": None},
    ) as request:
        status_code, payload = handler.submit_find_marvin(execute=True)

    assert status_code == 409
    assert payload["reasons"] == ["LiDAR is not fresh"]
    assert payload["admission_snapshot"] == snapshot
    request.assert_called_once_with(
        "GET",
        "http://127.0.0.1:8770/find-marvin/admission-snapshot",
        timeout=5.0,
    )


def test_unsafe_preflight_never_forwards_mission():
    handler = object.__new__(VoiceRelayHandler)
    unsafe = safe_admission_snapshot()
    unsafe["runtime"]["active_mission"] = {"mission_type": "FOLLOW_PERSON"}
    unsafe.update(admission_ready=False, reasons=[
        "another mission is active", "LiDAR is not fresh",
    ])
    unsafe["lidar"]["reason"] = "stale"
    unsafe["forward_interlock"]["forward_permitted"] = False

    with patch("voice_relay.server.request_json", return_value={
        "ok": True, "status_code": 200, "data": unsafe, "error": None,
    }) as request:
        status_code, payload = handler.submit_find_marvin(execute=True)

    assert status_code == 409
    assert payload["accepted"] is False
    assert "another mission is active" in payload["reasons"]
    assert "LiDAR is not fresh" in payload["reasons"]
    request.assert_called_once_with(
        "GET",
        "http://127.0.0.1:8770/find-marvin/admission-snapshot",
        timeout=5.0,
    )


def test_normal_mission_admits_healthy_clear_caution_or_blocked_front():
    for front_state in ("CLEAR", "CAUTION", "BLOCKED"):
        handler = object.__new__(VoiceRelayHandler)
        status = safe_admission_snapshot()
        status["lidar"]["front_state"] = front_state
        status["forward_interlock"]["front_state"] = front_state
        with patch(
            "voice_relay.server.request_json",
            side_effect=[
                {"ok": True, "status_code": 200, "data": status, "error": None},
                {"ok": True, "status_code": 202, "data": {"accepted": True}, "error": None},
            ],
        ) as request:
            status_code, payload = handler.submit_find_marvin(execute=True)
        assert status_code == 202
        assert payload["accepted"] is True
        assert request.call_count == 2


def test_normal_mission_rejects_bad_lidar_session_or_incomplete_geometry():
    for field, value, reason in (
        ("session_matches", False, "LiDAR producer session does not match"),
        ("local_motion_geometry_valid", False, "LiDAR local geometry is invalid"),
        ("required_sectors_valid", False, "LiDAR required sectors are incomplete"),
    ):
        handler = object.__new__(VoiceRelayHandler)
        status = safe_admission_snapshot()
        status["admission_ready"] = False
        status["reasons"] = [reason]
        status["lidar"][field] = value
        with patch("voice_relay.server.request_json", return_value={
            "ok": True, "status_code": 200, "data": status, "error": None,
        }) as request:
            status_code, payload = handler.submit_find_marvin(execute=True)
        assert status_code == 409
        assert reason in payload["reasons"]
        request.assert_called_once()


def test_normal_mission_rejects_unknown_front_as_missing_sector_evidence():
    handler = object.__new__(VoiceRelayHandler)
    status = safe_admission_snapshot()
    status["admission_ready"] = False
    status["reasons"] = ["LiDAR front sector is unavailable"]
    status["lidar"]["front_state"] = "UNKNOWN"
    with patch("voice_relay.server.request_json", return_value={
        "ok": True, "status_code": 200, "data": status, "error": None,
    }) as request:
        status_code, payload = handler.submit_find_marvin(execute=True)
    assert status_code == 409
    assert "LiDAR front sector is unavailable" in payload["reasons"]
    request.assert_called_once()


def test_normal_mission_still_rejects_unhealthy_lidar():
    for field, value, reason in (
        ("running", False, "LiDAR worker is not running"),
        ("available", False, "LiDAR is unavailable"),
        ("valid", False, "LiDAR is invalid"),
        ("reason", "stale", "LiDAR is not fresh"),
    ):
        handler = object.__new__(VoiceRelayHandler)
        status = safe_admission_snapshot()
        status["admission_ready"] = False
        status["reasons"] = [reason]
        status["lidar"][field] = value
        with patch("voice_relay.server.request_json", return_value={
            "ok": True, "status_code": 200, "data": status, "error": None,
        }) as request:
            status_code, payload = handler.submit_find_marvin(execute=True)
        assert status_code == 409
        assert reason in payload["reasons"]
        request.assert_called_once()


def test_missing_or_malformed_preflight_fields_fail_closed():
    handler = object.__new__(VoiceRelayHandler)
    with patch("voice_relay.server.request_json", return_value={
        "ok": True, "status_code": 200,
        "data": {"ok": True, "admission_ready": True}, "error": None,
    }) as request:
        status_code, payload = handler.submit_find_marvin(execute=True)

    assert status_code == 503
    assert payload["accepted"] is False
    assert payload["reasons"] == ["find_marvin_admission_snapshot_malformed"]
    request.assert_called_once()


def test_route_and_browser_cannot_supply_arbitrary_mission_fields():
    assert 'path == "/dashboard/find-marvin"' in SERVER
    assert 'allowed_fields = {"execute"}' in SERVER
    assert "Find Marvin accepts only execution authorization." in SERVER
    assert 'body: JSON.stringify({})' in HTML
    assert '"target": "marvin"' in SERVER
    assert '"target": "marvin"' not in HTML

    handler = object.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-marvin"
    handler.read_json_body = lambda: {"target": "anything else"}
    handler.send_json = Mock()
    handler.submit_find_marvin = Mock()
    handler.do_POST()
    handler.send_json.assert_called_once()
    assert handler.send_json.call_args.args[0] == 400
    handler.submit_find_marvin.assert_not_called()


def test_status_exposes_existing_safety_telemetry():
    required = (
        'runtime.get("lidar_perception", {})',
        'runtime.get("forward_interlock", {})',
        'robot.get("motion", {})',
    )
    for marker in required:
        assert marker in SERVER


def test_find_marvin_button_and_client_preflight_exist():
    assert 'id="findMarvinButton"' in HTML
    assert "Find Marvin" in HTML
    assert "Marvin (teddy bear)" in HTML
    assert "function findMarvinPreflight(status)" in HTML
    assert "function findMarvinMissionPreflight(status)" in HTML
    assert 'findMarvinReadiness(status, {requireClearFront: false})' in HTML
    assert "const preflight = findMarvinMissionPreflight(status);" in HTML
    assert "const ordinary = findMarvinPreflight(status);" in HTML
    assert "&& !missions.active" in HTML
    assert "findMarvinButton.disabled" in HTML
    assert 'fetch("/dashboard/find-marvin"' in HTML
    assert "centering_turn_chunks_attempted" in HTML
    assert "approach_chunks_completed" in HTML


def test_camera_and_read_only_lidar_are_responsive_companions():
    assert 'class="camera-lidar-pair"' in HTML
    assert 'id="cameraImage"' in HTML
    assert 'id="missionLidarCanvas"' in HTML
    assert ".camera-lidar-pair {" in CSS
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in CSS
    assert "@media (max-width: 980px)" in CSS
    assert 'const LIDAR_URL = "/dashboard/lidar";' in JS
    assert "LiDAR is stale, invalid, or unavailable. Scan points hidden." in JS
    assert 'context.fillText("FORWARD ↑"' in JS
    assert ".mission-lidar-message[hidden] {\n    display: none;\n}" in CSS


def test_companion_lidar_geometry_matches_production_sector_convention():
    feature = JS.split(
        "/* Mission camera companion: read-only robot-relative LiDAR */", 1
    )[1].split("/*", 1)[0]
    assert "const SCAN_TO_ROBOT_ROTATION_RADIANS = Math.PI / 2;" in feature
    assert "let rawAngle = Number(scan.angle_min);" in feature
    assert "rawAngle += increment;" in feature
    assert "rawAngle + SCAN_TO_ROBOT_ROTATION_RADIANS" in feature
    assert "x: centerX - Math.sin(bearing) * range * scale" in feature
    assert "y: centerY - Math.cos(bearing) * range * scale" in feature

    # Robot-bearing zero is up. Positive is left; negative is right.
    import math
    center_x, center_y, distance = 100.0, 100.0, 20.0

    def point(bearing):
        return (
            center_x - math.sin(bearing) * distance,
            center_y - math.cos(bearing) * distance,
        )

    forward = point(0.0)
    left = point(math.radians(20))
    right = point(math.radians(-20))
    assert forward == (center_x, center_y - distance)
    assert left[0] < center_x
    assert right[0] > center_x


def test_companion_front_wedge_mirrors_production_twenty_degree_bounds():
    feature = JS.split(
        "/* Mission camera companion: read-only robot-relative LiDAR */", 1
    )[1].split("/*", 1)[0]
    assert "const FRONT_SECTOR_HALF_RADIANS = 20 * Math.PI / 180;" in feature
    assert "-Math.PI / 2 - FRONT_SECTOR_HALF_RADIANS" in feature
    assert "-Math.PI / 2 + FRONT_SECTOR_HALF_RADIANS" in feature
    assert "Math.PI * .7" not in feature
    assert "Math.PI * .3" not in feature


def test_companion_lidar_introduces_no_ros_or_navigation_control():
    feature = JS.split(
        "/* Mission camera companion: read-only robot-relative LiDAR */", 1
    )[1].split("/*", 1)[0]
    forbidden = (
        "cmd_vel", "NavigateToPose", "navigation-goal", "click-to-go",
        'method: "POST"', "/motion", "/stop",
    )
    for marker in forbidden:
        assert marker not in feature
