#!/usr/bin/env python3

from unittest.mock import patch

from voice_relay.server import VoiceRelayHandler


TRACKING = {
    "active": True,
    "behavior": "FOLLOW_PERSON",
    "state": "CENTERING_LEFT",
    "target_label": "person",
    "target_confidence": 0.91,
    "target_center_x": 186.0,
    "target_center_y": 240.0,
    "image_width": 640.0,
    "image_height": 480.0,
    "image_center_x": 320.0,
    "horizontal_error": -134.0,
    "center_tolerance_pixels": 95.0,
    "target_area": 47290.0,
    "steering_direction": "LEFT",
    "distance_state": "TOO_FAR",
    "vision_timestamp": "2026-07-16T00:22:18Z",
    "detection_age_ms": 184,
    "bbox": {
        "x1": 100,
        "y1": 80,
        "x2": 272,
        "y2": 355,
    },
}


def fake_request_json(
    method,
    url,
    payload=None,
    timeout=5.0,
):
    if url.endswith("/status") and ":8770" in url:
        return {
            "ok": True,
            "status_code": 200,
            "data": {
                "running": True,
                "runtime_state": "MISSION_ACTIVE",
                "tracking": TRACKING,
                "last_error": None,
            },
            "error": None,
        }

    if url.endswith("/missions"):
        return {
            "ok": True,
            "status_code": 200,
            "data": {
                "active_mission": {
                    "mission_type": "FOLLOW_PERSON",
                    "target": "person",
                },
                "queue": [],
                "history_count": 2,
                "last_result": {
                    "behavior": "FOLLOW_PERSON",
                    "state": "CENTERING_LEFT",
                },
            },
            "error": None,
        }

    if ":8000" in url:
        return {
            "ok": True,
            "status_code": 200,
            "data": {
                "camera_running": True,
                "description": "Person detected.",
                "detections": [],
                "camera_url": (
                    "http://192.168.68.127:8091/"
                    "camera/latest.jpg"
                ),
                "timestamp": "2026-07-16T00:22:18Z",
                "last_error": None,
            },
            "error": None,
        }

    if ":8090" in url:
        return {
            "ok": True,
            "status_code": 200,
            "data": {
                "ok": True,
                "ros_ready": True,
                "status": "READY",
                "ros_error": None,
            },
            "error": None,
        }

    raise AssertionError(
        f"Unexpected request URL: {url}"
    )


def main():
    handler = object.__new__(
        VoiceRelayHandler
    )

    with patch(
        "voice_relay.server.request_json",
        side_effect=fake_request_json,
    ):
        status = handler.dashboard_status()

    assert status["ok"] is True
    assert status["runtime"]["connected"] is True
    assert status["runtime"]["running"] is True
    assert status["runtime"]["tracking"] == TRACKING

    tracking = status["runtime"]["tracking"]

    assert tracking["state"] == "CENTERING_LEFT"
    assert tracking["horizontal_error"] == -134.0
    assert tracking["target_area"] == 47290.0
    assert tracking["steering_direction"] == "LEFT"
    assert tracking["distance_state"] == "TOO_FAR"

    print("PASS: dashboard status forwards tracking unchanged")
    print("PASS: tracking state is runtime authoritative")
    print("PASS: steering and distance state are preserved")
    print()
    print("Dashboard tracking status test passed.")


if __name__ == "__main__":
    main()
