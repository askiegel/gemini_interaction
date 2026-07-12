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
    ]

    for requirement in html_requirements:
        assert requirement in html, requirement

    server_requirements = [
        'path == "/dashboard/status"',
        'path == "/stop"',
        "COGNITIVE_RUNTIME_URL",
        "VISION_SERVER_URL",
        "ROBOT_BRIDGE_URL",
        "dashboard_status",
        "run_voice_command",
    ]

    for requirement in server_requirements:
        assert requirement in server, requirement

    print("PASS: dashboard includes live service health")
    print("PASS: dashboard includes mission visibility")
    print("PASS: dashboard includes perception status")
    print("PASS: dashboard preserves voice recognition")
    print("PASS: dashboard includes quick commands")
    print("PASS: dashboard includes Stop Robot action")
    print()
    print("Operator Dashboard offline test passed.")


if __name__ == "__main__":
    main()
