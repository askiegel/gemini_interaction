#!/usr/bin/env python3

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
PLATFORM_LOG_DIR = PROJECT_DIR / "logs" / "platform"
PLATFORM_RUN_DIR = PROJECT_DIR / ".run"

VISION_SERVER_DIR = Path.home() / "vision_server"

ROBOT_HOST = os.getenv(
    "MINI_PUPPER_HOST",
    "192.168.68.124",
)

ROBOT_USER = os.getenv(
    "MINI_PUPPER_USER",
    "ubuntu",
)

ROBOT_SSH_TARGET = f"{ROBOT_USER}@{ROBOT_HOST}"

ROBOT_BRIDGE_URL = os.getenv(
    "ROBOT_BRIDGE_URL",
    f"http://{ROBOT_HOST}:8090",
)

CAMERA_RELAY_URL = os.getenv(
    "CAMERA_RELAY_URL",
    f"http://{ROBOT_HOST}:8091/camera/latest.jpg",
)

VISION_SERVER_URL = os.getenv(
    "VISION_SERVER_URL",
    "http://127.0.0.1:8000/detections/latest",
)

COGNITIVE_RUNTIME_URL = os.getenv(
    "COGNITIVE_RUNTIME_URL",
    "http://127.0.0.1:8770",
)

VOICE_RELAY_URL = os.getenv(
    "VOICE_RELAY_URL",
    "http://127.0.0.1:8765/status",
)


def heading(title: str):
    print()
    print("=" * 52)
    print(title)
    print("=" * 52)


def http_request(
    url: str,
    timeout: float = 3.0,
) -> Optional[bytes]:
    request = urllib.request.Request(
        url,
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            if 200 <= response.status < 300:
                return response.read()

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
    ):
        return None

    return None


def read_json_url(
    url: str,
    timeout: float = 3.0,
):
    body = http_request(
        url,
        timeout=timeout,
    )

    if body is None:
        return None

    try:
        return json.loads(
            body.decode("utf-8")
        )
    except json.JSONDecodeError:
        return None


def wait_for_url(
    name: str,
    url: str,
    timeout_seconds: float,
):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if http_request(url, timeout=2.0) is not None:
            print(f"PASS: {name} is ready")
            return True

        time.sleep(1.0)

    print(
        f"FAIL: {name} did not become ready: {url}"
    )
    return False


def process_is_running(pid_file: Path):
    if not pid_file.exists():
        return False

    try:
        pid = int(
            pid_file.read_text().strip()
        )
    except (OSError, ValueError):
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_file(pid_file: Path) -> Optional[int]:
    """Return a valid integer PID from a PID file."""

    if not pid_file.exists():
        return None

    try:
        value = pid_file.read_text(
            encoding="utf-8"
        ).strip()

        return int(value)

    except (OSError, ValueError):
        return None


def remove_pid_file(pid_file: Path):
    """Remove a stale or no-longer-needed PID file."""

    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(
            f"WARN: Could not remove PID file "
            f"{pid_file}: {exc}"
        )


def wait_for_process_exit(
    pid: int,
    timeout_seconds: float = 8.0,
) -> bool:
    """Wait until a process no longer exists."""

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True

        time.sleep(0.2)

    return False


def stop_pid(
    name: str,
    pid: int,
):
    """Stop one process, escalating to SIGKILL if required."""

    if pid <= 1 or pid == os.getpid():
        return

    try:
        os.kill(pid, 0)
    except OSError:
        return

    print(
        f"STOP:  {name} PID={pid}"
    )

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise RuntimeError(
            f"Unable to stop {name} PID {pid}: {exc}"
        ) from exc

    if wait_for_process_exit(
        pid,
        timeout_seconds=8.0,
    ):
        print(
            f"PASS:  {name} PID={pid} stopped"
        )
        return

    print(
        f"WARN:  {name} PID={pid} did not stop; "
        "sending SIGKILL"
    )

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise RuntimeError(
            f"Unable to kill {name} PID {pid}: {exc}"
        ) from exc

    if not wait_for_process_exit(
        pid,
        timeout_seconds=4.0,
    ):
        raise RuntimeError(
            f"{name} PID {pid} is still running."
        )

    print(
        f"PASS:  {name} PID={pid} killed"
    )


def process_matches(
    pid: int,
    cwd: Path,
    command_markers: Iterable[str],
) -> bool:
    """
    Return True when a Linux process has the expected working directory
    and all command-line markers.

    This catches services that were started manually without a PID file.
    """

    proc_dir = Path("/proc") / str(pid)

    try:
        process_cwd = Path(
            os.readlink(proc_dir / "cwd")
        ).resolve()

        command_line = (
            proc_dir
            .joinpath("cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(
                "utf-8",
                errors="replace",
            )
        )

    except (
        FileNotFoundError,
        PermissionError,
        OSError,
    ):
        return False

    try:
        expected_cwd = cwd.resolve()
    except OSError:
        expected_cwd = cwd

    if process_cwd != expected_cwd:
        return False

    return all(
        marker in command_line
        for marker in command_markers
    )


def find_matching_processes(
    cwd: Path,
    command_markers: Iterable[str],
):
    """Find service processes matching a working directory and command."""

    markers = tuple(command_markers)
    matches = []

    proc_root = Path("/proc")

    if not proc_root.exists():
        return matches

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue

        pid = int(entry.name)

        if pid == os.getpid():
            continue

        if process_matches(
            pid,
            cwd,
            markers,
        ):
            matches.append(pid)

    return sorted(set(matches))


def stop_local_process(
    name: str,
    cwd: Path,
    pid_file: Path,
    command_markers: Iterable[str],
):
    """
    Stop a managed local service.

    The PID file is checked first, followed by a /proc scan so manually
    started or orphaned service instances are also removed.
    """

    stopped_pids = set()

    pid = read_pid_file(pid_file)

    if pid is not None:
        stop_pid(
            name,
            pid,
        )
        stopped_pids.add(pid)

    for matching_pid in find_matching_processes(
        cwd,
        command_markers,
    ):
        if matching_pid in stopped_pids:
            continue

        stop_pid(
            name,
            matching_pid,
        )
        stopped_pids.add(matching_pid)

    remove_pid_file(pid_file)

    if not stopped_pids:
        print(
            f"PASS:  {name} was not running"
        )


def port_is_open(
    host: str,
    port: int,
    timeout: float = 0.5,
) -> bool:
    """Check whether a TCP port accepts connections."""

    try:
        with socket.create_connection(
            (host, int(port)),
            timeout=timeout,
        ):
            return True
    except OSError:
        return False


def wait_for_port_free(
    name: str,
    host: str,
    port: int,
    timeout_seconds: float = 10.0,
):
    """Wait until a local service port is no longer listening."""

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if not port_is_open(
            host,
            port,
        ):
            print(
                f"PASS:  {name} port {port} is free"
            )
            return True

        time.sleep(0.25)

    print(
        f"FAIL:  {name} port {port} is still in use"
    )
    return False


def restart_local_services():
    """
    Stop local platform services so current source code is loaded.

    Remote Mini Pupper services are not forcibly restarted. The normal
    startup process reconnects to the robot and starts any missing remote
    service before completing full health verification.
    """

    heading("STOPPING LOCAL PLATFORM SERVICES")

    stop_local_process(
        name="Browser Voice Relay",
        cwd=PROJECT_DIR,
        pid_file=(
            PLATFORM_RUN_DIR
            / "voice_relay.pid"
        ),
        command_markers=(
            "voice_relay/server.py",
        ),
    )

    stop_local_process(
        name="Cognitive Runtime",
        cwd=PROJECT_DIR,
        pid_file=(
            PLATFORM_RUN_DIR
            / "runtime_api.pid"
        ),
        command_markers=(
            "runtime_api.py",
        ),
    )

    stop_local_process(
        name="Vision Service",
        cwd=PROJECT_DIR,
        pid_file=(
            PLATFORM_RUN_DIR
            / "vision_service.pid"
        ),
        command_markers=(
            "vision_service.py",
        ),
    )

    stop_local_process(
        name="YOLO Vision Server",
        cwd=VISION_SERVER_DIR,
        pid_file=(
            PLATFORM_RUN_DIR
            / "vision_server.pid"
        ),
        command_markers=(
            "uvicorn",
            "server:app",
        ),
    )

    heading("VERIFYING LOCAL PORTS")

    checks = [
        (
            "YOLO Vision Server",
            "127.0.0.1",
            8000,
        ),
        (
            "Browser Voice Relay",
            "127.0.0.1",
            8765,
        ),
        (
            "Cognitive Runtime",
            "127.0.0.1",
            8770,
        ),
    ]

    failures = []

    for name, host, port in checks:
        if not wait_for_port_free(
            name,
            host,
            port,
        ):
            failures.append(
                f"{name} port {port}"
            )

    if failures:
        raise RuntimeError(
            "Unable to restart because these resources "
            "remain in use: "
            + ", ".join(failures)
        )


def start_local_process(
    name: str,
    command,
    cwd: Path,
    pid_file: Path,
    log_file: Path,
):
    if process_is_running(pid_file):
        pid = pid_file.read_text().strip()
        print(
            f"SKIP: {name} already has running PID {pid}"
        )
        return

    if not cwd.exists():
        raise RuntimeError(
            f"{name} directory does not exist: {cwd}"
        )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pid_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        log_file,
        "ab",
        buffering=0,
    ) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    pid_file.write_text(
        f"{process.pid}\n"
    )

    print(
        f"START: {name} PID={process.pid}"
    )
    print(
        f"LOG:   {log_file}"
    )


def start_remote_stack():
    """
    Start missing Mini Pupper services in one SSH session.

    The SSH command may request the Mini Pupper password when SSH keys are
    not configured.
    """
    remote_script = r'''
set -e

RUN_DIR="$HOME/robot_bridge/.run"
LOG_DIR="$HOME/robot_bridge/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

is_running() {
    local pid_file="$1"

    [ -f "$pid_file" ] || return 1

    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)

    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

if is_running "$RUN_DIR/ros2_bringup.pid"; then
    echo "SKIP: ROS2 bringup already running"
else
    nohup bash -lc '
        source /opt/ros/humble/setup.bash
        source "$HOME/ros2_ws/install/setup.bash"
        export ROS_DOMAIN_ID=42
        export ROS_LOCALHOST_ONLY=0
        exec ros2 launch mini_pupper_bringup bringup.launch.py
    ' > "$LOG_DIR/ros2_bringup.log" 2>&1 &

    echo $! > "$RUN_DIR/ros2_bringup.pid"
    echo "START: ROS2 bringup PID=$(cat "$RUN_DIR/ros2_bringup.pid")"
fi

sleep 4

if is_running "$RUN_DIR/robot_bridge.pid"; then
    echo "SKIP: Robot Bridge already running"
else
    cd "$HOME/robot_bridge"

    nohup bash -lc '
        source /opt/ros/humble/setup.bash
        source "$HOME/ros2_ws/install/setup.bash"
        export ROS_DOMAIN_ID=42
        export ROS_LOCALHOST_ONLY=0
        cd "$HOME/robot_bridge"
        exec python3 app.py
    ' > "$LOG_DIR/robot_bridge.log" 2>&1 &

    echo $! > "$RUN_DIR/robot_bridge.pid"
    echo "START: Robot Bridge PID=$(cat "$RUN_DIR/robot_bridge.pid")"
fi

if is_running "$RUN_DIR/camera_relay.pid"; then
    echo "SKIP: Camera Relay already running"
else
    cd "$HOME/robot_bridge"

    nohup bash -lc '
        source /opt/ros/humble/setup.bash
        source "$HOME/ros2_ws/install/setup.bash"
        export ROS_DOMAIN_ID=42
        export ROS_LOCALHOST_ONLY=0
        cd "$HOME/robot_bridge"
        exec python3 -u camera_relay.py
    ' > "$LOG_DIR/camera_relay.log" 2>&1 &

    echo $! > "$RUN_DIR/camera_relay.pid"
    echo "START: Camera Relay PID=$(cat "$RUN_DIR/camera_relay.pid")"
fi
'''

    print(
        f"Connecting to {ROBOT_SSH_TARGET}..."
    )

    result = subprocess.run(
        [
            "ssh",
            ROBOT_SSH_TARGET,
            "bash -s",
        ],
        input=remote_script,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Remote Mini Pupper startup failed."
        )


def service_status():
    bridge = read_json_url(
        f"{ROBOT_BRIDGE_URL}/status"
    )

    camera_ok = (
        http_request(
            CAMERA_RELAY_URL,
            timeout=3.0,
        )
        is not None
    )

    vision = read_json_url(
        VISION_SERVER_URL
    )

    runtime = read_json_url(
        f"{COGNITIVE_RUNTIME_URL}/health"
    )

    voice = read_json_url(
        VOICE_RELAY_URL
    )

    return {
        "robot_bridge": {
            "ready": bool(
                bridge
                and bridge.get("ok")
                and bridge.get("ros_ready")
            ),
            "details": bridge,
        },
        "camera_relay": {
            "ready": camera_ok,
            "url": CAMERA_RELAY_URL,
        },
        "vision_server": {
            "ready": vision is not None,
            "camera_running": (
                vision.get("camera_running")
                if vision
                else None
            ),
            "details": vision,
        },
        "cognitive_runtime": {
            "ready": bool(
                runtime
                and runtime.get("ok")
            ),
            "details": runtime,
        },
        "voice_relay": {
            "ready": bool(
                voice
                and voice.get("ok")
            ),
            "details": voice,
        },
        "vision_service": {
            "ready": process_is_running(
                PLATFORM_RUN_DIR
                / "vision_service.pid"
            ),
        },
    }


def print_status(status):
    labels = {
        "robot_bridge": "Robot Bridge / ROS2",
        "camera_relay": "Camera Relay",
        "vision_server": "YOLO Vision Server",
        "vision_service": "Vision Service",
        "cognitive_runtime": "Cognitive Runtime",
        "voice_relay": "Browser Voice Relay",
    }

    heading("MINI PUPPER COGNITIVE PLATFORM")

    all_ready = True

    for key, label in labels.items():
        ready = bool(
            status.get(
                key,
                {},
            ).get("ready")
        )

        all_ready = all_ready and ready

        marker = "PASS" if ready else "FAIL"
        print(f"{marker:4}  {label}")

    print()

    if all_ready:
        print("SYSTEM READY")
    else:
        print("SYSTEM NOT READY")

    return all_ready


def print_plan():
    heading("STARTUP PLAN")

    print("1. Mini Pupper ROS2 bringup")
    print("2. Mini Pupper Robot Bridge")
    print("3. Mini Pupper Camera Relay")
    print("4. Ubuntu PC YOLO Vision Server")
    print("5. Ubuntu PC Vision Service")
    print("6. Ubuntu PC Cognitive Runtime")
    print("7. Ubuntu PC Browser Voice Relay")
    print("8. Full health verification")
    print()
    print(
        "--start starts services that are missing."
    )
    print(
        "--restart reloads all local service code, "
        "then verifies the complete stack."
    )


def start_platform():
    PLATFORM_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLATFORM_RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    heading("REMOTE MINI PUPPER SERVICES")
    start_remote_stack()

    if not wait_for_url(
        "Robot Bridge",
        f"{ROBOT_BRIDGE_URL}/status",
        timeout_seconds=30,
    ):
        raise RuntimeError(
            "Robot Bridge health check failed."
        )

    if not wait_for_url(
        "Camera Relay",
        CAMERA_RELAY_URL,
        timeout_seconds=30,
    ):
        raise RuntimeError(
            "Camera Relay health check failed."
        )

    heading("LOCAL YOLO VISION SERVER")

    if http_request(VISION_SERVER_URL) is None:
        uvicorn_path = (
            VISION_SERVER_DIR
            / ".venv"
            / "bin"
            / "uvicorn"
        )

        if not uvicorn_path.exists():
            raise RuntimeError(
                f"Uvicorn was not found: {uvicorn_path}"
            )

        start_local_process(
            name="YOLO Vision Server",
            command=[
                str(uvicorn_path),
                "server:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=VISION_SERVER_DIR,
            pid_file=(
                PLATFORM_RUN_DIR
                / "vision_server.pid"
            ),
            log_file=(
                PLATFORM_LOG_DIR
                / "vision_server.log"
            ),
        )
    else:
        print("SKIP: YOLO Vision Server already ready")

    if not wait_for_url(
        "YOLO Vision Server",
        VISION_SERVER_URL,
        timeout_seconds=45,
    ):
        raise RuntimeError(
            "YOLO Vision Server health check failed."
        )

    heading("LOCAL PERSISTENT SERVICES")

    cognitive_python = (
        PROJECT_DIR
        / ".venv"
        / "bin"
        / "python3"
    )

    if not cognitive_python.exists():
        raise RuntimeError(
            f"Cognitive Python was not found: "
            f"{cognitive_python}"
        )

    local_env = os.environ.copy()
    local_env["VISION_SERVER_URL"] = (
        VISION_SERVER_URL
    )
    local_env["COGNITIVE_RUNTIME_URL"] = (
        COGNITIVE_RUNTIME_URL
    )

    os.environ.update(local_env)

    start_local_process(
        name="Vision Service",
        command=[
            str(cognitive_python),
            "vision_service.py",
            "--poll-interval",
            "0.35",
        ],
        cwd=PROJECT_DIR,
        pid_file=(
            PLATFORM_RUN_DIR
            / "vision_service.pid"
        ),
        log_file=(
            PLATFORM_LOG_DIR
            / "vision_service.log"
        ),
    )

    if http_request(
        f"{COGNITIVE_RUNTIME_URL}/health"
    ) is None:
        start_local_process(
            name="Cognitive Runtime",
            command=[
                str(cognitive_python),
                "runtime_api.py",
            ],
            cwd=PROJECT_DIR,
            pid_file=(
                PLATFORM_RUN_DIR
                / "runtime_api.pid"
            ),
            log_file=(
                PLATFORM_LOG_DIR
                / "runtime_api.log"
            ),
        )
    else:
        print("SKIP: Cognitive Runtime already ready")

    if not wait_for_url(
        "Cognitive Runtime",
        f"{COGNITIVE_RUNTIME_URL}/health",
        timeout_seconds=30,
    ):
        raise RuntimeError(
            "Cognitive Runtime health check failed."
        )

    if http_request(VOICE_RELAY_URL) is None:
        start_local_process(
            name="Browser Voice Relay",
            command=[
                str(cognitive_python),
                "voice_relay/server.py",
            ],
            cwd=PROJECT_DIR,
            pid_file=(
                PLATFORM_RUN_DIR
                / "voice_relay.pid"
            ),
            log_file=(
                PLATFORM_LOG_DIR
                / "voice_relay.log"
            ),
        )
    else:
        print("SKIP: Browser Voice Relay already ready")

    if not wait_for_url(
        "Browser Voice Relay",
        VOICE_RELAY_URL,
        timeout_seconds=30,
    ):
        raise RuntimeError(
            "Browser Voice Relay health check failed."
        )

    time.sleep(2.0)

    status = service_status()

    if not print_status(status):
        raise RuntimeError(
            "One or more platform services are not ready."
        )

    print()
    print("Voice interface:")
    print("  http://localhost:8765")
    print()
    print("Runtime status:")
    print(
        f"  {COGNITIVE_RUNTIME_URL}/status"
    )
    print()
    print("Logs:")
    print(f"  {PLATFORM_LOG_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Start and verify the Mini Pupper 2 "
            "Cognitive Robotics Platform."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--plan",
        action="store_true",
        help="Print the startup order without starting services.",
    )

    group.add_argument(
        "--check",
        action="store_true",
        help="Check the current service state.",
    )

    group.add_argument(
        "--start",
        action="store_true",
        help="Start missing services and verify the complete stack.",
    )

    group.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Restart local services with current source code "
            "and verify the complete stack."
        ),
    )

    args = parser.parse_args()

    try:
        if args.plan:
            print_plan()
            return

        if args.check:
            ready = print_status(
                service_status()
            )

            raise SystemExit(
                0 if ready else 1
            )

        if args.start:
            print_plan()
            start_platform()
            return

        if args.restart:
            print_plan()

            PLATFORM_LOG_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            PLATFORM_RUN_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            restart_local_services()
            start_platform()
            return

    except KeyboardInterrupt:
        print()
        print("Startup interrupted.")
        raise SystemExit(130)

    except Exception as exc:
        print()
        print(f"STARTUP FAILED: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
