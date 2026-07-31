#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
STARTUP_SCRIPT = PROJECT_DIR / "scripts" / "start_platform.py"


def main():
    test_environment = os.environ.copy()
    test_environment["MINI_PUPPER_HOST"] = "192.0.2.25"
    test_environment.pop("CAMERA_RELAY_URL", None)
    test_environment.pop("VISION_CAMERA_URL", None)

    code = f"""
import importlib.util
import os
from pathlib import Path

path = Path({str(STARTUP_SCRIPT)!r})
spec = importlib.util.spec_from_file_location(
    "portable_camera_startup",
    path,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

expected = "http://192.0.2.25:8091/camera/latest.jpg"

assert module.ROBOT_HOST == "192.0.2.25"
assert module.CAMERA_RELAY_URL == expected
assert os.environ["VISION_CAMERA_URL"] == expected

print("Resolved Camera Relay:", module.CAMERA_RELAY_URL)
print("Vision Camera URL:", os.environ["VISION_CAMERA_URL"])
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_DIR,
        env=test_environment,
        text=True,
        capture_output=True,
        check=False,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0

    print("PASS: Vision Server inherits the resolved Camera Relay URL.")
    print("PASS: No fixed robot subnet is required.")


if __name__ == "__main__":
    main()
