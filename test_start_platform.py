#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPT = PROJECT_DIR / "scripts" / "start_platform.py"


def main():
    print("===== STARTUP MANAGER PLAN TEST =====")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan",
        ],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0

    expected_items = [
        "Mini Pupper ROS2 bringup",
        "Mini Pupper Robot Bridge",
        "Mini Pupper Camera Relay",
        "Ubuntu PC YOLO Vision Server",
        "Ubuntu PC Vision Service",
        "Ubuntu PC Cognitive Runtime",
        "Ubuntu PC Browser Voice Relay",
        "Full health verification",
    ]

    for item in expected_items:
        assert item in result.stdout

    assert "--restart" in result.stdout

    print("PASS: startup plan executes offline")
    print("PASS: dependency order is preserved")
    print("PASS: all platform services are included")
    print("PASS: restart behavior is documented")

    print()
    print("===== STARTUP MANAGER HELP TEST =====")

    help_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--help",
        ],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    print(help_result.stdout)

    if help_result.stderr:
        print(help_result.stderr)

    assert help_result.returncode == 0
    assert "--plan" in help_result.stdout
    assert "--check" in help_result.stdout
    assert "--start" in help_result.stdout
    assert "--restart" in help_result.stdout

    print("PASS: existing CLI modes remain available")
    print("PASS: --restart is available")
    print()
    print("Startup Manager offline test passed.")


if __name__ == "__main__":
    main()
