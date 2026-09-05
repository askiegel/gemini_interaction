#!/usr/bin/env python3

import argparse
import os
import runpy
from pathlib import Path


ROOT = (
    Path.home()
    / "robot_services"
    / "cognitive"
)

START_PLATFORM = (
    ROOT
    / "scripts"
    / "start_platform.py"
)


PID_NAMES = {
    "vision-server":
        "vision_server.pid",
    "vision-service":
        "vision_service.pid",
    "cognitive-runtime":
        "runtime_api.pid",
    "voice-relay":
        "voice_relay.pid",
}


def load_platform():
    ns = runpy.run_path(
        str(START_PLATFORM)
    )

    # Preserve the exact URL configuration used by
    # the existing platform startup implementation.
    for name in (
        "ROBOT_BRIDGE_URL",
        "CAMERA_RELAY_URL",
        "VISION_SERVER_URL",
        "COGNITIVE_RUNTIME_URL",
        "VOICE_RELAY_URL",
    ):
        value = ns.get(name)

        if isinstance(value, str):
            os.environ[name] = value

    camera = ns.get(
        "CAMERA_RELAY_URL"
    )

    if not isinstance(camera, str):
        raise RuntimeError(
            "CAMERA_RELAY_URL unavailable."
        )

    # server.py explicitly consumes this name.
    os.environ[
        "VISION_CAMERA_URL"
    ] = camera

    return ns


def require(path, label):
    path = Path(path)

    if not path.is_file():
        raise RuntimeError(
            f"{label} not found: {path}"
        )

    return path


def record_pid(ns, role):
    run_dir = Path(
        ns["PLATFORM_RUN_DIR"]
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pid_file = (
        run_dir
        / PID_NAMES[role]
    )

    pid_file.write_text(
        f"{os.getpid()}\n",
        encoding="utf-8",
    )


def run(role):
    ns = load_platform()

    project = Path(
        ns["PROJECT_DIR"]
    )

    vision = Path(
        ns["VISION_SERVER_DIR"]
    )

    cognitive_python = require(
        project
        / ".venv"
        / "bin"
        / "python3",
        "Cognitive Python",
    )

    if role == "vision-server":
        executable = require(
            vision
            / ".venv"
            / "bin"
            / "uvicorn",
            "Vision uvicorn",
        )

        cwd = vision

        argv = [
            str(executable),
            "server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]

    elif role == "vision-service":
        require(
            project
            / "vision_service.py",
            "Vision Service",
        )

        executable = cognitive_python
        cwd = project

        argv = [
            str(executable),
            "vision_service.py",
            "--poll-interval",
            "0.35",
        ]

    elif role == "cognitive-runtime":
        require(
            project
            / "runtime_api.py",
            "Runtime API",
        )

        executable = cognitive_python
        cwd = project

        argv = [
            str(executable),
            "runtime_api.py",
        ]

    elif role == "voice-relay":
        require(
            project
            / "voice_relay"
            / "server.py",
            "Voice Relay",
        )

        executable = cognitive_python
        cwd = project

        argv = [
            str(executable),
            "voice_relay/server.py",
        ]

    else:
        raise RuntimeError(
            f"Unknown role: {role}"
        )

    record_pid(
        ns,
        role,
    )

    os.chdir(
        cwd
    )

    os.execvpe(
        str(executable),
        argv,
        os.environ.copy(),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "role",
        choices=tuple(
            PID_NAMES
        ),
    )

    args = parser.parse_args()

    run(
        args.role
    )


if __name__ == "__main__":
    main()
