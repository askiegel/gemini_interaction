"""Offline contract tests for the dedicated Find Marvin dashboard control."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from voice_relay.server import VoiceRelayHandler


ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "voice_relay" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "voice_relay" / "operator_console.css").read_text(encoding="utf-8")
JS = (ROOT / "voice_relay" / "operator_console.js").read_text(encoding="utf-8")
SERVER = (ROOT / "voice_relay" / "server.py").read_text(encoding="utf-8")


def safe_status():
    return {
        "runtime": {
            "connected": True,
            "running": True,
            "last_error": None,
            "lidar": {
                "running": True,
                "available": True,
                "valid": True,
                "reason": "fresh",
                "front_state": "CLEAR",
            },
            "forward_interlock": {
                "configured": True,
                "monitor_running": True,
                "forward_permitted": True,
                "reason": "fresh_clear",
                "active_forward": False,
                "pending_forward": False,
            },
        },
        "missions": {"active": None, "queue_count": 0},
        "robot": {
            "connected": True,
            "status": "READY",
            "ros_ready": True,
            "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
        },
    }


def test_dedicated_route_constructs_one_fixed_find_object_mission():
    handler = object.__new__(VoiceRelayHandler)
    handler.dashboard_status = safe_status
    accepted = {
        "ok": True,
        "accepted": True,
        "mission": {"mission_id": "mission-marvin"},
    }
    with patch(
        "voice_relay.server.request_json",
        return_value={
            "ok": True,
            "status_code": 202,
            "data": accepted,
            "error": None,
        },
    ) as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 202
    assert payload == accepted
    request.assert_called_once_with(
        "POST",
        "http://127.0.0.1:8770/missions",
        payload={
            "source_text": "Find Marvin.",
            "intent": {
                "intent": "FIND_OBJECT",
                "speech": "Find Marvin.",
                "target": "marvin",
            },
        },
        timeout=15.0,
    )


def test_unsafe_preflight_never_forwards_mission():
    handler = object.__new__(VoiceRelayHandler)
    unsafe = safe_status()
    unsafe["missions"]["active"] = {"mission_type": "FOLLOW_PERSON"}
    unsafe["runtime"]["lidar"]["reason"] = "stale"
    unsafe["runtime"]["forward_interlock"]["forward_permitted"] = False
    handler.dashboard_status = lambda: unsafe

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    assert "another mission is active" in payload["reasons"]
    assert "LiDAR is not fresh" in payload["reasons"]
    request.assert_not_called()


@pytest.mark.parametrize(
    "front_state,forward_permitted,reason",
    [
        ("CLEAR", True, "fresh_clear"),
        ("BLOCKED", False, "front_not_clear"),
        ("CAUTION", False, "front_not_clear"),
    ],
)
def test_front_motion_inhibition_does_not_block_submission(
    front_state, forward_permitted, reason
):
    handler = object.__new__(VoiceRelayHandler)
    status = safe_status()
    status["runtime"]["lidar"]["front_state"] = front_state
    interlock = status["runtime"]["forward_interlock"]
    interlock["forward_permitted"] = forward_permitted
    interlock["reason"] = reason
    handler.dashboard_status = lambda: status

    with patch(
        "voice_relay.server.request_json",
        return_value={
            "ok": True,
            "status_code": 202,
            "data": {"ok": True, "accepted": True},
            "error": None,
        },
    ) as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 202
    assert payload["accepted"] is True
    request.assert_called_once()


@pytest.mark.parametrize("field", ["active_forward", "pending_forward"])
def test_active_or_pending_forward_still_rejects_submission(field):
    handler = object.__new__(VoiceRelayHandler)
    status = safe_status()
    status["runtime"]["forward_interlock"][field] = True
    handler.dashboard_status = lambda: status

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    request.assert_not_called()


@pytest.mark.parametrize(
    "lidar_update",
    [
        {"reason": "stale"},
        {"valid": False},
        {"available": False},
    ],
)
def test_unhealthy_lidar_still_rejects_submission(lidar_update):
    handler = object.__new__(VoiceRelayHandler)
    status = safe_status()
    status["runtime"]["lidar"].update(lidar_update)
    handler.dashboard_status = lambda: status

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    request.assert_not_called()


@pytest.mark.parametrize("mission_field", ["active", "queue_count"])
def test_nonquiescent_mission_state_still_rejects_submission(mission_field):
    handler = object.__new__(VoiceRelayHandler)
    status = safe_status()
    status["missions"][mission_field] = (
        {"mission_type": "FOLLOW_PERSON"}
        if mission_field == "active"
        else 1
    )
    handler.dashboard_status = lambda: status

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    request.assert_not_called()


@pytest.mark.parametrize(
    "motion_update",
    [
        {"linear_x": 0.01},
        {"angular_z": 0.01},
        {"streaming": True},
    ],
)
def test_robot_motion_or_streaming_still_rejects_submission(motion_update):
    handler = object.__new__(VoiceRelayHandler)
    status = safe_status()
    status["robot"]["motion"].update(motion_update)
    handler.dashboard_status = lambda: status

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    request.assert_not_called()


def test_missing_or_malformed_preflight_fields_fail_closed():
    handler = object.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: {
        "runtime": {}, "missions": {}, "robot": {}
    }

    with patch("voice_relay.server.request_json") as request:
        status_code, payload = handler.submit_find_marvin()

    assert status_code == 409
    assert payload["accepted"] is False
    assert payload["reasons"]
    request.assert_not_called()


def test_route_and_browser_cannot_supply_arbitrary_mission_fields():
    assert 'path == "/dashboard/find-marvin"' in SERVER
    assert "if payload != {}:" in SERVER
    assert "accepts no browser-supplied mission fields" in SERVER
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
    assert "Marvin (semantic; detector alias teddy bear)" in HTML
    assert "function findMarvinPreflight(status)" in HTML
    assert "&& !missions.active" in HTML
    assert "findMarvinButton.disabled" in HTML
    assert 'fetch("/dashboard/find-marvin"' in HTML
    assert "centering_turn_chunks_attempted" in HTML
    assert "approach_chunks_completed" in HTML


def test_client_submission_gate_allows_front_motion_inhibition():
    preflight = HTML.split("function findMarvinPreflight(status)", 1)[1].split(
        "function clearTrackingOverlay()", 1
    )[0]
    submission_gate = preflight.split(
        "const forwardMotionInhibited", 1
    )[0]
    assert 'lidar.reason === "fresh"' in submission_gate
    assert "interlock.active_forward === false" in submission_gate
    assert "interlock.pending_forward === false" in submission_gate
    assert 'lidar.front_state === "CLEAR"' not in submission_gate
    assert "interlock.forward_permitted === true" not in submission_gate
    assert 'interlock.reason === "fresh_clear"' not in submission_gate
    assert (
        "Ready to find Marvin. Forward motion currently inhibited by LiDAR."
        in preflight
    )


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
