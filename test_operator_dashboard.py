#!/usr/bin/env python3

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
HTML_FILE = PROJECT_DIR / "voice_relay" / "index.html"
SERVER_FILE = PROJECT_DIR / "voice_relay" / "server.py"


def main():
    html = HTML_FILE.read_text(
        encoding="utf-8"
    )

    server = SERVER_FILE.read_text(
        encoding="utf-8"
    )

    html_requirements = [
        "Mini Pupper 2 Cognitive Robotics Platform",
        "Start Listening",
        "STOP ROBOT",
        "Find Backpack",
        "Follow Me",
        "Runtime state",
        "Behavior state",
        "Detection",
        "/dashboard/status",
        "/stop",
        "SpeechRecognition",
        "setInterval",
        "const tracking = runtime.tracking || {};",
        "tracking.target_label",
        "tracking.state",
        "tracking.horizontal_error",
        "tracking.target_area",
    ]

    for requirement in html_requirements:
        assert requirement in html, requirement

    forbidden_dashboard_logic = [
        "lastResult.horizontal_error",
        "targetDetection.area",
        "active.mission_type ===",
    ]

    for forbidden in forbidden_dashboard_logic:
        assert forbidden not in html, forbidden

    server_requirements = [
        'path == "/dashboard/status"',
        'path == "/stop"',
        "COGNITIVE_RUNTIME_URL",
        "VISION_SERVER_URL",
        "ROBOT_BRIDGE_URL",
        "dashboard_status",
        "run_voice_command",
        'runtime.get("tracking", {})',
    ]

    for requirement in server_requirements:
        assert requirement in server, requirement

    print("PASS: dashboard forwards runtime tracking state")
    print("PASS: dashboard renders runtime target label")
    print("PASS: dashboard renders runtime behavior state")
    print("PASS: dashboard renders runtime horizontal error")
    print("PASS: dashboard renders runtime target area")
    print("PASS: dashboard no longer derives tracking metrics")
    print("PASS: dashboard preserves voice and STOP controls")
    print()
    print("Operator Dashboard runtime integration test passed.")


if __name__ == "__main__":
    main()
