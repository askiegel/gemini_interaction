#!/usr/bin/env python3

import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


VOICE_RELAY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VOICE_RELAY_DIR.parent

ROBOT_HOST = os.getenv(
    "MAYDAY_SSH_HOST",
    "ubuntu@192.168.68.124",
)

ROBOT_BRIDGE_URL = os.getenv(
    "ROBOT_BRIDGE_URL",
    "http://192.168.68.124:8090",
).rstrip("/")

NAV_ASSET = (
    VOICE_RELAY_DIR
    / "tony2_navigation_assets"
    / "mayday_guarded_navigation.yaml"
)

NAV_RUNTIME = (
    VOICE_RELAY_DIR
    / "tony2_navigation_runtime.py"
)

EXPECTED_MAX_VEL_X = 0.50
EXPECTED_MIN_VEL_X = 0.0
EXPECTED_GOAL_DISTANCE = 5.0
EXPECTED_TIMEOUT = 120.0


def utc_now():
    return (
        datetime.datetime.now(
            datetime.timezone.utc
        )
        .isoformat()
    )


def run_command(
    args,
    *,
    cwd=None,
    timeout=15,
):
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def ssh(script, timeout=20):
    remote_command = (
        "bash -lc "
        + shlex.quote(script)
    )

    return run_command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            ROBOT_HOST,
            remote_command,
        ],
        timeout=timeout,
    )


def request_json(
    method,
    url,
    *,
    timeout=5,
):
    request = urllib.request.Request(
        url,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            body = response.read().decode(
                "utf-8",
                errors="replace",
            )

            return {
                "ok": True,
                "status": response.status,
                "data": json.loads(body),
                "error": None,
            }

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        try:
            data = json.loads(body)
        except Exception:
            data = None

        return {
            "ok": False,
            "status": exc.code,
            "data": data,
            "error": body or str(exc),
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "data": None,
            "error": str(exc),
        }


# STARTUP_LIVE_PROGRESS_BEGIN
STARTUP_CHECK_PLAN = [{'id': 'dashboard_owner', 'label': 'Tony2 dashboard process', 'required': True},
 {'id': 'cognitive_source', 'label': 'Cognitive source state', 'required': True},
 {'id': 'tony2_cognitive_boot',
  'label': 'Tony2 cognitive boot services',
  'required': True},
 {'id': 'camera_vision_pipeline',
  'label': 'Camera to Vision pipeline',
  'required': True},
 {'id': 'camera_relay_boot_service',
  'label': 'Camera Relay boot service',
  'required': True},
 {'id': 'guarded_asset', 'label': 'Guarded navigation asset', 'required': True},
 {'id': 'gait_threshold', 'label': 'Mayday navigation speed', 'required': True},
 {'id': 'mayday_reachable', 'label': 'Mayday / Robot Bridge', 'required': True},
 {'id': 'hardware_bringup', 'label': 'Hardware bringup service', 'required': True},
 {'id': 'hardware_nodes', 'label': 'Required ROS hardware nodes', 'required': True},
 {'id': 'bridge_process', 'label': 'Robot Bridge process', 'required': True},
 {'id': 'bridge_boot_service', 'label': 'Robot Bridge boot service', 'required': True},
 {'id': 'cmd_vel_chain', 'label': '/cmd_vel command chain', 'required': True},
 {'id': 'locomotion_chain',
  'label': 'CHAMP -> servo trajectory chain',
  'required': True},
 {'id': 'lidar', 'label': 'LD06 LiDAR /scan', 'required': True},
 {'id': 'motion_zero', 'label': 'Physical motion state', 'required': True},
 {'id': 'navigation_clean', 'label': 'Persistent Nav2 / AMCL ready', 'required': True},
 {'id': 'current_pose_localization',
  'label': 'Current physical pose localized',
  'required': True},
 {'id': 'navigation_envelope',
  'label': 'Guarded navigation envelope',
  'required': True}]
# STARTUP_LIVE_PROGRESS_END

def add_check(
    checks,
    check_id,
    label,
    passed,
    detail,
    *,
    required=True,
):
    item = {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "required": bool(required),
        "detail": str(detail),
    }

    checks.append(
        item
    )

    return item


def parse_float(
    text,
    name,
):
    match = re.search(
        rf"^\s*{re.escape(name)}:\s*"
        r"([-+]?[0-9]*\.?[0-9]+)\s*$",
        text,
        re.MULTILINE,
    )

    if not match:
        return None

    return float(
        match.group(1)
    )


def expected_asset_hash():
    text = NAV_RUNTIME.read_text(
        encoding="utf-8"
    )

    filename = (
        "mayday_guarded_navigation.yaml"
    )

    filename_position = text.find(
        filename
    )

    if filename_position < 0:
        return None

    matches = list(
        re.finditer(
            r"\b[0-9a-f]{64}\b",
            text,
        )
    )

    if not matches:
        return None

    nearest = min(
        matches,
        key=lambda item: abs(
            item.start()
            - filename_position
        ),
    )

    if (
        abs(
            nearest.start()
            - filename_position
        )
        > 1000
    ):
        return None

    return nearest.group(0)


def local_dashboard_snapshot():
    head = run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_DIR,
    )

    branch = run_command(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=PROJECT_DIR,
    )

    status = run_command(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=PROJECT_DIR,
    )

    sockets = run_command(
        [
            "ss",
            "-ltnp",
        ],
        timeout=5,
    )

    owner_pid = None

    for line in (
        sockets["stdout"].splitlines()
    ):
        if ":8765" not in line:
            continue

        match = re.search(
            r"pid=(\d+)",
            line,
        )

        if match:
            owner_pid = int(
                match.group(1)
            )
            break

    return {
        "head": head["stdout"],
        "branch": branch["stdout"],
        "dirty": bool(
            status["stdout"].strip()
        ),
        "pid": os.getpid(),
        "port_owner_pid": owner_pid,
        "cmdline": (
            Path(
                f"/proc/{os.getpid()}/cmdline"
            )
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        ),
    }


def robot_snapshot():
    service_script = r'''
set +e

echo "bringup_enabled=$(systemctl is-enabled mayday-bringup.service 2>/dev/null || true)"
echo "bringup_active=$(systemctl is-active mayday-bringup.service 2>/dev/null || true)"

echo "bridge_service_enabled=$(systemctl is-enabled mayday-robot-bridge.service 2>/dev/null || true)"
echo "bridge_service_active=$(systemctl is-active mayday-robot-bridge.service 2>/dev/null || true)"

echo "bridge_service_pid=$(systemctl show mayday-robot-bridge.service -p MainPID --value 2>/dev/null || true)"

PORT_PID="$(
    ss -ltnp 2>/dev/null \
        | grep -E '[:.]8090[[:space:]]' \
        | sed -nE 's/.*pid=([0-9]+).*/\1/p' \
        | head -n1
)"

echo "bridge_port_pid=${PORT_PID}"

if [[ "$PORT_PID" =~ ^[0-9]+$ ]]; then
    CMD="$(
        tr '\0' ' ' \
            < "/proc/$PORT_PID/cmdline" \
            2>/dev/null
    )"

    echo "bridge_cmdline=${CMD}"
else
    echo "bridge_cmdline="
fi
'''

    service_result = ssh(
        service_script,
        timeout=10,
    )

    values = {}

    for line in (
        service_result[
            "stdout"
        ].splitlines()
    ):
        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1,
        )

        values[
            key.strip()
        ] = value.strip()

    ros_script = r'''
set +e

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "__NODES__"
timeout 8 ros2 node list --no-daemon 2>&1 || true

echo "__CMDVEL__"
timeout 8 ros2 topic info /cmd_vel --verbose 2>&1 || true

echo "__QUADRUPED__"
timeout 8 ros2 node info /quadruped_controller_node 2>&1 || true

echo "__SERVO__"
timeout 8 ros2 node info /servo_interface 2>&1 || true

echo "__SCAN__"
timeout 8 ros2 topic info /scan --verbose 2>&1 || true
'''

    ros_result = ssh(
        ros_script,
        timeout=40,
    )

    sections = {
        "nodes": "",
        "cmdvel": "",
        "quadruped": "",
        "servo": "",
        "scan": "",
    }

    marker_map = {
        "__NODES__": "nodes",
        "__CMDVEL__": "cmdvel",
        "__QUADRUPED__": "quadruped",
        "__SERVO__": "servo",
        "__SCAN__": "scan",
    }

    current = None
    buckets = {
        key: []
        for key in sections
    }

    for line in (
        ros_result[
            "stdout"
        ].splitlines()
    ):
        if line in marker_map:
            current = marker_map[
                line
            ]
            continue

        if current is not None:
            buckets[
                current
            ].append(line)

    for key in sections:
        sections[key] = "\n".join(
            buckets[key]
        ).strip()

    return {
        "ssh_ok": (
            service_result["ok"]
            and ros_result["ok"]
        ),
        "service": values,
        "ros": sections,
        "service_error": (
            service_result[
                "stderr"
            ]
        ),
        "ros_error": (
            ros_result[
                "stderr"
            ]
        ),
    }



# _MAYDAY_FULL_STARTUP_PROOF_V2

def _startup_local_systemd_state(service):
    import subprocess

    def command(*args):
        result = subprocess.run(
            list(args),
            text=True,
            capture_output=True,
            check=False,
            timeout=8,
        )

        return (
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )

    enabled_rc, enabled, enabled_err = command(
        "systemctl",
        "is-enabled",
        service,
    )

    active_rc, active, active_err = command(
        "systemctl",
        "is-active",
        service,
    )

    pid_rc, pid_text, pid_err = command(
        "systemctl",
        "show",
        service,
        "-p",
        "MainPID",
        "--value",
    )

    try:
        pid = int(pid_text)
    except Exception:
        pid = 0

    return {
        "service": service,
        "enabled": enabled,
        "active": active,
        "main_pid": pid,
        "ok": (
            enabled_rc == 0
            and enabled == "enabled"
            and active_rc == 0
            and active == "active"
            and pid_rc == 0
            and pid > 0
        ),
        "errors": [
            value
            for value in (
                enabled_err,
                active_err,
                pid_err,
            )
            if value
        ],
    }


def _startup_remote_systemd_state(service):
    import subprocess
    import urllib.parse

    host = urllib.parse.urlparse(
        ROBOT_BRIDGE_URL
    ).hostname

    if not host:
        raise RuntimeError(
            "Robot Bridge hostname unavailable."
        )

    target = f"ubuntu@{host}"

    remote = (
        f"systemctl is-enabled {service}; "
        f"systemctl is-active {service}; "
        f"systemctl show {service} "
        "-p MainPID --value"
    )

    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            target,
            remote,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=12,
    )

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    enabled = (
        lines[0]
        if len(lines) >= 1
        else ""
    )

    active = (
        lines[1]
        if len(lines) >= 2
        else ""
    )

    try:
        pid = int(
            lines[2]
            if len(lines) >= 3
            else "0"
        )
    except Exception:
        pid = 0

    return {
        "service": service,
        "enabled": enabled,
        "active": active,
        "main_pid": pid,
        "ssh_returncode":
            result.returncode,
        "stderr":
            result.stderr.strip(),
        "ok": (
            result.returncode == 0
            and enabled == "enabled"
            and active == "active"
            and pid > 0
        ),
    }


def _startup_http_json(url, timeout=5):
    import json
    import urllib.request

    request = urllib.request.Request(
        url,
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        status = getattr(
            response,
            "status",
            200,
        )

        payload = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    return {
        "http_status": status,
        "payload": payload,
    }




_STARTUP_LOCALIZATION_EVIDENCE = None


def clear_startup_localization_evidence():
    global _STARTUP_LOCALIZATION_EVIDENCE

    _STARTUP_LOCALIZATION_EVIDENCE = None


def set_startup_localization_evidence(value):
    import copy

    global _STARTUP_LOCALIZATION_EVIDENCE

    _STARTUP_LOCALIZATION_EVIDENCE = (
        copy.deepcopy(value)
    )


def get_startup_localization_evidence():
    import copy

    return copy.deepcopy(
        _STARTUP_LOCALIZATION_EVIDENCE
    )


def prove_ready(
    navigation=None,
    progress_callback=None,
):
    checks = []

    def record_check(
        *args,
        **kwargs,
    ):
        item = add_check(
            *args,
            **kwargs,
        )

        if progress_callback is not None:
            progress_callback(
                dict(item)
            )

        return item

    try:
        local = (
            local_dashboard_snapshot()
        )

        dashboard_ok = (
            local["pid"]
            == local[
                "port_owner_pid"
            ]
            and (
                "voice_relay/server.py"
                in local["cmdline"]
            )
        )

        record_check(
            checks,
            "dashboard_owner",
            "Tony2 dashboard process",
            dashboard_ok,
            (
                f"PID {local['pid']} "
                f"owns :8765; "
                f"HEAD {local['head'][:12]}"
            ),
        )

        record_check(
            checks,
            "cognitive_source",
            "Cognitive source state",
            (
                local["branch"]
                == "main"
                and not local[
                    "dirty"
                ]
            ),
            (
                f"branch={local['branch']} "
                f"clean={not local['dirty']} "
                f"HEAD={local['head'][:12]}"
            ),
        )

    except Exception as exc:
        local = {}

        record_check(
            checks,
            "dashboard_owner",
            "Tony2 dashboard process",
            False,
            exc,
        )

        record_check(
            checks,
            "cognitive_source",
            "Cognitive source state",
            False,
            exc,
        )


    try:
        tony2_services = (
            "mayday-vision-server.service",
            "mayday-vision-service.service",
            "mayday-cognitive-runtime.service",
            "mayday-voice-relay.service",
        )

        tony2_states = {
            service:
                _startup_local_systemd_state(
                    service
                )
            for service in tony2_services
        }

        tony2_boot_ok = all(
            state.get("ok") is True
            for state in tony2_states.values()
        )

        record_check(
            checks,
            "tony2_cognitive_boot",
            "Tony2 cognitive boot services",
            tony2_boot_ok,
            str(tony2_states),
        )

    except Exception as exc:
        record_check(
            checks,
            "tony2_cognitive_boot",
            "Tony2 cognitive boot services",
            False,
            exc,
        )

    try:
        vision_result = (
            _startup_http_json(
                "http://127.0.0.1:8000"
                "/detections/latest",
                timeout=6,
            )
        )

        vision_payload = (
            vision_result.get("payload")
            or {}
        )

        camera_vision_ok = (
            vision_result.get(
                "http_status"
            )
            == 200
            and isinstance(
                vision_payload,
                dict,
            )
            and vision_payload.get(
                "camera_running"
            )
            is True
            and not vision_payload.get(
                "last_error"
            )
        )

        record_check(
            checks,
            "camera_vision_pipeline",
            "Camera to Vision pipeline",
            camera_vision_ok,
            (
                f"http="
                f"{vision_result.get('http_status')} "
                f"camera_running="
                f"{vision_payload.get('camera_running')} "
                f"last_error="
                f"{vision_payload.get('last_error')}"
            ),
        )

    except Exception as exc:
        record_check(
            checks,
            "camera_vision_pipeline",
            "Camera to Vision pipeline",
            False,
            exc,
        )

    try:
        camera_relay_state = (
            _startup_remote_systemd_state(
                "mayday-camera-relay.service"
            )
        )

        record_check(
            checks,
            "camera_relay_boot_service",
            "Camera Relay boot service",
            (
                camera_relay_state.get(
                    "ok"
                )
                is True
            ),
            str(camera_relay_state),
        )

    except Exception as exc:
        record_check(
            checks,
            "camera_relay_boot_service",
            "Camera Relay boot service",
            False,
            exc,
        )

    try:
        asset_text = (
            NAV_ASSET.read_text(
                encoding="utf-8"
            )
        )

        actual_hash = (
            hashlib.sha256(
                NAV_ASSET.read_bytes()
            )
            .hexdigest()
        )

        expected_hash = (
            expected_asset_hash()
        )

        record_check(
            checks,
            "guarded_asset",
            "Guarded navigation asset",
            (
                expected_hash
                is not None
                and actual_hash
                == expected_hash
            ),
            (
                f"sha256={actual_hash[:12]} "
                f"expected="
                f"{(expected_hash or 'missing')[:12]}"
            ),
        )

        min_vel = parse_float(
            asset_text,
            "min_vel_x",
        )

        max_vel = parse_float(
            asset_text,
            "max_vel_x",
        )

        max_speed = parse_float(
            asset_text,
            "max_speed_xy",
        )

        speed_ok = (
            min_vel
            == EXPECTED_MIN_VEL_X
            and max_vel
            == EXPECTED_MAX_VEL_X
            and max_speed
            == EXPECTED_MAX_VEL_X
        )

        record_check(
            checks,
            "gait_threshold",
            "Mayday navigation speed",
            speed_ok,
            (
                f"min={min_vel} "
                f"max={max_vel} "
                f"max_xy={max_speed}"
            ),
        )

    except Exception as exc:
        record_check(
            checks,
            "guarded_asset",
            "Guarded navigation asset",
            False,
            exc,
        )

        record_check(
            checks,
            "gait_threshold",
            "Mayday navigation speed",
            False,
            exc,
        )

    bridge = request_json(
        "GET",
        f"{ROBOT_BRIDGE_URL}/status",
        timeout=5,
    )

    bridge_data = (
        bridge.get("data")
        if isinstance(
            bridge.get("data"),
            dict,
        )
        else {}
    )

    record_check(
        checks,
        "mayday_reachable",
        "Mayday / Robot Bridge",
        (
            bridge["ok"]
            and bridge_data.get(
                "status"
            )
            == "READY"
            and bridge_data.get(
                "ros_ready"
            )
            is True
        ),
        (
            f"http={bridge.get('status')} "
            f"status={bridge_data.get('status')} "
            f"ros_ready={bridge_data.get('ros_ready')}"
        ),
    )

    robot = robot_snapshot()

    service = robot[
        "service"
    ]

    ros = robot["ros"]

    record_check(
        checks,
        "hardware_bringup",
        "Hardware bringup service",
        (
            service.get(
                "bringup_enabled"
            )
            == "enabled"
            and service.get(
                "bringup_active"
            )
            == "active"
        ),
        (
            f"enabled="
            f"{service.get('bringup_enabled')} "
            f"active="
            f"{service.get('bringup_active')}"
        ),
    )

    expected_nodes = (
        "/quadruped_controller_node",
        "/servo_interface",
        "/LD06",
        "/robot_state_publisher",
        "/state_estimation_node",
        "/base_to_footprint_ekf",
        "/footprint_to_odom_ekf",
    )

    missing_nodes = [
        node
        for node in expected_nodes
        if node not in ros[
            "nodes"
        ]
    ]

    record_check(
        checks,
        "hardware_nodes",
        "Required ROS hardware nodes",
        not missing_nodes,
        (
            "all expected nodes present"
            if not missing_nodes
            else (
                "missing: "
                + ", ".join(
                    missing_nodes
                )
            )
        ),
    )

    port_pid = service.get(
        "bridge_port_pid"
    )

    bridge_cmdline = (
        service.get(
            "bridge_cmdline",
            "",
        )
    )

    bridge_process_ok = (
        bool(port_pid)
        and port_pid.isdigit()
        and "app.py"
        in bridge_cmdline
    )

    record_check(
        checks,
        "bridge_process",
        "Robot Bridge process",
        bridge_process_ok,
        (
            f"PID={port_pid or 'NONE'} "
            f"owns :8090; "
            f"cmd={bridge_cmdline or 'NONE'}"
        ),
    )

    bridge_service_ok = (
        service.get(
            "bridge_service_enabled"
        )
        == "enabled"
        and service.get(
            "bridge_service_active"
        )
        == "active"
    )

    record_check(
        checks,
        "bridge_boot_service",
        "Robot Bridge boot service",
        bridge_service_ok,
        (
            f"enabled="
            f"{service.get('bridge_service_enabled')} "
            f"active="
            f"{service.get('bridge_service_active')}"
        ),
        required=True,
    )

    cmdvel = ros[
        "cmdvel"
    ]

    cmdvel_ok = (
        "Publisher count: 1"
        in cmdvel
        and "Subscription count: 1"
        in cmdvel
        and "robot_bridge_publisher"
        in cmdvel
        and "quadruped_controller_node"
        in cmdvel
    )

    record_check(
        checks,
        "cmd_vel_chain",
        "/cmd_vel command chain",
        cmdvel_ok,
        (
            "1 Robot Bridge publisher -> "
            "1 quadruped controller subscriber"
            if cmdvel_ok
            else "unexpected /cmd_vel wiring"
        ),
    )

    quadruped = ros[
        "quadruped"
    ]

    servo = ros[
        "servo"
    ]

    trajectory_topic = (
        "/joint_group_effort_controller/"
        "joint_trajectory"
    )

    locomotion_ok = (
        trajectory_topic
        in quadruped
        and trajectory_topic
        in servo
    )

    record_check(
        checks,
        "locomotion_chain",
        "CHAMP -> servo trajectory chain",
        locomotion_ok,
        (
            "quadruped publishes and "
            "servo_interface subscribes"
            if locomotion_ok
            else "trajectory chain incomplete"
        ),
    )

    scan = ros["scan"]

    scan_ok = (
        "Publisher count:"
        in scan
        and not (
            "Publisher count: 0"
            in scan
        )
    )

    record_check(
        checks,
        "lidar",
        "LD06 LiDAR /scan",
        scan_ok,
        (
            "LaserScan publisher available"
            if scan_ok
            else "/scan publisher unavailable"
        ),
    )

    motion = (
        bridge_data.get(
            "motion"
        )
        or {}
    )

    try:
        motion_zero = (
            float(
                motion.get(
                    "linear_x",
                    999,
                )
            )
            == 0.0
            and float(
                motion.get(
                    "angular_z",
                    999,
                )
            )
            == 0.0
            and motion.get(
                "streaming"
            )
            is False
        )
    except Exception:
        motion_zero = False

    record_check(
        checks,
        "motion_zero",
        "Physical motion state",
        motion_zero,
        (
            f"linear={motion.get('linear_x')} "
            f"angular={motion.get('angular_z')} "
            f"streaming={motion.get('streaming')}"
        ),
    )

    if not isinstance(
        navigation,
        dict,
    ):
        navigation = {}

    pids = (
        navigation.get("pids")
        or {}
    )

    nav_clean = (
        navigation.get("state")
        == "READY"
        and navigation.get(
            "running"
        )
        is True
        and navigation.get(
            "owned"
        )
        is True
        and navigation.get(
            "map_server_enabled"
        )
        is True
        and navigation.get(
            "localization_enabled"
        )
        is True
        and navigation.get(
            "planner_enabled"
        )
        is True
        and navigation.get(
            "controller_enabled"
        )
        is True
        and navigation.get(
            "navigator_enabled"
        )
        is True
        and navigation.get(
            "goal_active"
        )
        is False
        and navigation.get(
            "motion_output_connected"
        )
        is False
        and pids.get(
            "supervisor"
        )
        is not None
        and pids.get(
            "probe"
        )
        is not None
        and pids.get(
            "goal"
        )
        is None
    )

    record_check(
        checks,
        "navigation_clean",
        "Persistent Nav2 / AMCL ready",
        nav_clean,
        (
            f"state={navigation.get('state')} "
            f"running={navigation.get('running')} "
            f"goal={navigation.get('goal_active')} "
            f"output="
            f"{navigation.get('motion_output_connected')} "
            f"pids={pids}"
        ),
    )

    try:
        evidence = (
            get_startup_localization_evidence()
        )

        localization = (
            evidence.get("localization")
            if isinstance(
                evidence,
                dict,
            )
            else None
        )

        evidence_navigation = (
            evidence.get("navigation")
            if isinstance(
                evidence,
                dict,
            )
            else None
        )

        diagnostic = (
            localization.get("diagnostic")
            if isinstance(
                localization,
                dict,
            )
            else None
        )

        final_pose = (
            localization.get("final_pose")
            if isinstance(
                localization,
                dict,
            )
            else None
        )

        uncertainty = (
            localization.get("uncertainty")
            if isinstance(
                localization,
                dict,
            )
            else None
        )

        evidence_pids = (
            evidence_navigation.get("pids")
            if isinstance(
                evidence_navigation,
                dict,
            )
            else {}
        ) or {}

        current_pids = (
            navigation.get("pids")
            or {}
        )

        localization_result_ok = (
            isinstance(
                evidence,
                dict,
            )
            and evidence.get(
                "action"
            )
            == "OPERATOR_POSE_VALIDATED"
            and isinstance(
                localization,
                dict,
            )
            and localization.get(
                "ok"
            )
            is True
            and localization.get(
                "trusted"
            )
            is True
            and localization.get(
                "frame_id"
            )
            == "map"
            and localization.get(
                "localization_method"
            )
            == "amcl_global"
            and localization.get(
                "search_scope"
            )
            == "full_saved_map"
            and localization.get(
                "seed_pose_used"
            )
            is False
            and localization.get(
                "global_localization_requested"
            )
            is True
            and localization.get(
                "initial_pose_supplied"
            )
            is False
            and localization.get(
                "stationary_required"
            )
            is True
            and localization.get(
                "navigation_goal_executed"
            )
            is False
            and localization.get(
                "motion_enabled"
            )
            is False
            and isinstance(
                diagnostic,
                dict,
            )
            and diagnostic.get(
                "covariance_tight"
            )
            is True
            and diagnostic.get(
                "global_search_completed"
            )
            is True
            and diagnostic.get(
                "seed_pose_applied"
            )
            is False
            and diagnostic.get(
                "alignment_good"
            )
            is True
            and diagnostic.get(
                "trusted"
            )
            is True
            and isinstance(
                final_pose,
                dict,
            )
            and isinstance(
                uncertainty,
                dict,
            )
            and evidence_pids.get(
                "supervisor"
            )
            == current_pids.get(
                "supervisor"
            )
            and evidence_pids.get(
                "probe"
            )
            == current_pids.get(
                "probe"
            )
            and evidence_pids.get(
                "goal"
            )
            is None
        )

        numeric_values = (
            final_pose.get("x"),
            final_pose.get("y"),
            final_pose.get("yaw_rad"),
            uncertainty.get("sigma_x_m"),
            uncertainty.get("sigma_y_m"),
            uncertainty.get("sigma_yaw_rad"),
        )

        import math as _startup_math

        localization_numbers_ok = (
            localization_result_ok
            and all(
                isinstance(
                    value,
                    (int, float),
                )
                and _startup_math.isfinite(
                    float(value)
                )
                for value in numeric_values
            )
        )

        localization_result = (
            _startup_http_json(
                f"{ROBOT_BRIDGE_URL}"
                "/telemetry/localization",
                timeout=6,
            )
        )

        localization_payload = (
            localization_result.get(
                "payload"
            )
            or {}
        )

        live_telemetry = (
            localization_payload.get(
                "telemetry"
            )
            if isinstance(
                localization_payload,
                dict,
            )
            else None
        )

        live_amcl_ok = (
            localization_result.get(
                "http_status"
            )
            == 200
            and isinstance(
                localization_payload,
                dict,
            )
            and localization_payload.get(
                "ok"
            )
            is True
            and localization_payload.get(
                "runtime_active"
            )
            is True
            and localization_payload.get(
                "topic"
            )
            == "/amcl_pose"
            and isinstance(
                live_telemetry,
                dict,
            )
            and live_telemetry.get(
                "available"
            )
            is True
        )

        current_pose_ok = (
            localization_numbers_ok
            and live_amcl_ok
        )

        record_check(
            checks,
            "current_pose_localization",
            "Current physical pose localized",
            current_pose_ok,
            (
                f"action="
                f"{evidence.get('action') if isinstance(evidence, dict) else None} "
                f"trusted="
                f"{localization.get('trusted') if isinstance(localization, dict) else None} "
                f"alignment="
                f"{diagnostic.get('alignment_good') if isinstance(diagnostic, dict) else None} "
                f"covariance="
                f"{diagnostic.get('covariance_tight') if isinstance(diagnostic, dict) else None} "
                f"global="
                f"{diagnostic.get('global_search_completed') if isinstance(diagnostic, dict) else None} "
                f"seed="
                f"{localization.get('seed_pose_used') if isinstance(localization, dict) else None} "
                f"amcl_live="
                f"{live_amcl_ok} "
                f"pose={final_pose} "
                f"uncertainty={uncertainty}"
            ),
        )

    except Exception as exc:
        record_check(
            checks,
            "current_pose_localization",
            "Current physical pose localized",
            False,
            exc,
        )


    envelope_ok = (
        navigation.get(
            "maximum_goal_distance_meters"
        )
        == EXPECTED_GOAL_DISTANCE
        and navigation.get(
            "execution_timeout_seconds"
        )
        == EXPECTED_TIMEOUT
    )

    record_check(
        checks,
        "navigation_envelope",
        "Guarded navigation envelope",
        envelope_ok,
        (
            f"distance="
            f"{navigation.get('maximum_goal_distance_meters')}m "
            f"timeout="
            f"{navigation.get('execution_timeout_seconds')}s"
        ),
    )

    required = [
        item
        for item in checks
        if item["required"]
    ]

    passed_required = sum(
        1
        for item in required
        if item["passed"]
    )

    ready = (
        passed_required
        == len(required)
        and len(required) > 0
    )

    return {
        "ok": True,
        "ready": ready,
        "verdict": (
            "READY_FOR_NAVIGATION"
            if ready
            else "NOT_READY"
        ),
        "proved_at": utc_now(),
        "passed_required": passed_required,
        "required_checks": len(
            required
        ),
        "total_checks": len(
            checks
        ),
        "checks": checks,
    }


def ensure_robot_runtime():
    steps = []

    bringup = ssh(
        r'''
set +e

if systemctl is-active --quiet mayday-bringup.service; then
    echo "bringup already active"
    exit 0
fi

sudo -n systemctl start mayday-bringup.service
''',
        timeout=15,
    )

    steps.append(
        {
            "name": "hardware_bringup",
            "ok": bringup["ok"],
            "detail": (
                bringup["stdout"]
                or bringup["stderr"]
            ),
        }
    )

    bridge = request_json(
        "GET",
        f"{ROBOT_BRIDGE_URL}/status",
        timeout=3,
    )

    if not bridge["ok"]:
        start_bridge = ssh(
            r'''
set +e

if systemctl list-unit-files \
    | grep -q '^mayday-robot-bridge.service'; then

    sudo -n systemctl start mayday-robot-bridge.service

    sleep 1

    if systemctl is-active --quiet mayday-robot-bridge.service; then
        echo "Robot Bridge started by systemd"
        exit 0
    fi
fi

PORT_PID="$(
    ss -ltnp 2>/dev/null \
        | grep -E '[:.]8090[[:space:]]' \
        | sed -nE 's/.*pid=([0-9]+).*/\1/p' \
        | head -n1
)"

if [[ "$PORT_PID" =~ ^[0-9]+$ ]]; then
    echo "Robot Bridge already owns 8090"
    exit 0
fi

cd "$HOME/robot_bridge" || exit 1

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset FASTDDS_DEFAULT_PROFILES_FILE

nohup \
    /usr/bin/python3 \
    -u \
    app.py \
    >/tmp/mayday_robot_bridge_startup.log \
    2>&1 \
    </dev/null &

echo "Robot Bridge fallback PID=$!"
''',
            timeout=15,
        )

        steps.append(
            {
                "name": "robot_bridge",
                "ok": start_bridge["ok"],
                "detail": (
                    start_bridge["stdout"]
                    or start_bridge["stderr"]
                ),
            }
        )

    else:
        steps.append(
            {
                "name": "robot_bridge",
                "ok": True,
                "detail": (
                    "Robot Bridge already reachable"
                ),
            }
        )

    bridge_ready = False

    for _ in range(30):
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/status",
            timeout=2,
        )

        if (
            response["ok"]
            and isinstance(
                response.get("data"),
                dict,
            )
            and response[
                "data"
            ].get("ros_ready")
            is True
        ):
            bridge_ready = True
            break

        time.sleep(0.25)

    steps.append(
        {
            "name": "bridge_health",
            "ok": bridge_ready,
            "detail": (
                "READY / ros_ready=true"
                if bridge_ready
                else (
                    "Robot Bridge did not "
                    "become ROS-ready"
                )
            ),
        }
    )

    if bridge_ready:
        for endpoint in (
            "/navigation/stop",
            "/planning/stop",
            "/localization/stop",
            "/mapping/stop",
            "/stop",
        ):
            response = request_json(
                "POST",
                (
                    ROBOT_BRIDGE_URL
                    + endpoint
                ),
                timeout=8,
            )

            steps.append(
                {
                    "name": (
                        "safe_stop:"
                        + endpoint
                    ),
                    "ok": (
                        response[
                            "status"
                        ]
                        in (
                            200,
                            404,
                        )
                    ),
                    "detail": (
                        f"HTTP "
                        f"{response.get('status')}"
                    ),
                }
            )

    return steps


def prepare_session():
    steps = ensure_robot_runtime()

    return {
        "ok": all(
            step["ok"]
            for step in steps
            if not step[
                "name"
            ].startswith(
                "safe_stop:"
            )
        ),
        "action": "prepare_session",
        "prepared_at": utc_now(),
        "steps": steps,
        "message": (
            "Session preparation completed "
            "without starting navigation."
        ),
    }
