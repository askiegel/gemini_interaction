#!/usr/bin/env bash
set -e
set -o pipefail

ROBOT="${ROBOT:-http://192.168.68.124:8090}"
CAMERA="${CAMERA:-http://192.168.68.124:8091}"
DASHBOARD="${DASHBOARD:-http://127.0.0.1:8765}"
VISION="${VISION:-http://127.0.0.1:8000}"
RUNTIME="${RUNTIME:-http://127.0.0.1:8770}"
MAYDAY="${MAYDAY:-ubuntu@192.168.68.124}"

SCRIPT_DIRECTORY=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)

PROJECT_ROOT=$(
    cd -- "$SCRIPT_DIRECTORY/.." &&
    pwd
)

MODE="${1:---check}"

case "$MODE" in
    --check|--prepare)
        ;;
    *)
        echo "Usage: $0 [--check|--prepare]" >&2
        exit 2
        ;;
esac

prepared=false

safe_stop() {
    curl -fsS --connect-timeout 2 --max-time 40 \
        -X POST "$ROBOT/navigation/stop" \
        >/dev/null 2>&1 || true

    curl -fsS --connect-timeout 2 --max-time 40 \
        -X POST "$ROBOT/planning/stop" \
        >/dev/null 2>&1 || true

    curl -fsS --connect-timeout 2 --max-time 20 \
        -X POST "$ROBOT/mapping/stop" \
        >/dev/null 2>&1 || true

    curl -fsS --connect-timeout 2 --max-time 20 \
        -X POST "$ROBOT/localization/stop" \
        >/dev/null 2>&1 || true

    curl -fsS --connect-timeout 2 --max-time 10 \
        -X POST "$ROBOT/stop" \
        >/dev/null 2>&1 || true
}

cleanup() {
    rm -f \
        /tmp/mayday_session_status.json \
        /tmp/mayday_session_camera.jpg \
        /tmp/mayday_session_lidar.json \
        /tmp/mayday_session_stop.json \
        /tmp/mayday_session_vision.json \
        /tmp/mayday_session_runtime.json

    if [ "$prepared" = true ]; then
        safe_stop
    fi
}

trap cleanup EXIT

echo "===== PREPARE MAYDAY SESSION ====="
date
hostname
echo "Mode: $MODE"

if [ "$MODE" = "--prepare" ]; then
    echo
    echo "===== ESTABLISH SAFE STOPPED STATE ====="
    safe_stop
    prepared=true
    echo "PASS: Optional runtimes were stopped."
    echo "PASS: Safety zero was requested."

    echo
    echo "===== START COMPLETE COGNITIVE PLATFORM ====="

    python3         "$PROJECT_ROOT/scripts/start_platform.py"         --start

    echo "PASS: Complete platform startup finished."
else
    echo
    echo "INFO: Check mode performs no stop requests."
fi

echo
echo "===== VERIFY NETWORK SERVICES ====="

curl -fsS --connect-timeout 2 --max-time 10 \
    "$ROBOT/status" \
    >/tmp/mayday_session_status.json

curl -fsS --connect-timeout 2 --max-time 10 \
    "$CAMERA/camera/latest.jpg" \
    >/tmp/mayday_session_camera.jpg

curl -fsS --connect-timeout 2 --max-time 10 \
    "$DASHBOARD/" \
    >/dev/null

curl -fsS --connect-timeout 2 --max-time 10 \
    "$VISION/detections/latest" \
    >/tmp/mayday_session_vision.json

curl -fsS --connect-timeout 2 --max-time 10 \
    "$RUNTIME/health" \
    >/tmp/mayday_session_runtime.json

camera_bytes=$(
    wc -c </tmp/mayday_session_camera.jpg
)

test "$camera_bytes" -gt 1000

echo "PASS: Robot Bridge is reachable."
echo "PASS: Camera is live (${camera_bytes} bytes)."
echo "PASS: Dashboard is reachable."

python3 - <<'__VERIFY_COMPLETE_PC_PLATFORM_F092__'
import json
from pathlib import Path

vision = json.loads(
    Path("/tmp/mayday_session_vision.json").read_text(
        encoding="utf-8"
    )
)

if vision.get("last_error"):
    raise SystemExit(
        "ERROR: Vision reports: "
        + str(vision["last_error"])
    )

if vision.get("camera_running") is False:
    raise SystemExit("ERROR: Vision camera is not running.")

runtime = json.loads(
    Path("/tmp/mayday_session_runtime.json").read_text(
        encoding="utf-8"
    )
)

if not runtime.get("ok"):
    raise SystemExit("ERROR: Cognitive Runtime is unhealthy.")

if runtime.get("runtime_running") is False:
    raise SystemExit("ERROR: Cognitive Runtime is not running.")

print("PASS: Vision Server is available and error-free.")
print("PASS: Cognitive Runtime is healthy.")
__VERIFY_COMPLETE_PC_PLATFORM_F092__

echo
echo "===== VERIFY ROBOT RUNTIME ====="

ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=5 \
    "$MAYDAY" \
    'bash -s' <<'__REMOTE_MAYDAY_SESSION_CHECK_92C1__'
set -e
set -o pipefail

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

test "$ROS_DISTRO" = "humble"
test "$ROS_DOMAIN_ID" = "42"
test "$ROS_LOCALHOST_ONLY" = "0"

echo "PASS: ROS_DISTRO=$ROS_DISTRO"
echo "PASS: ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "PASS: ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"

mapfile -t bringup_pids < <(
    pgrep -f \
        '[r]os2 launch mini_pupper_bringup bringup.launch.py' ||
    true
)

if [ "${#bringup_pids[@]}" -ne 1 ]; then
    echo "ERROR: Expected exactly one standard bringup owner."
    printf 'Bringup PID: %s\n' "${bringup_pids[@]:-NONE}"
    exit 1
fi

echo "PASS: Exactly one standard bringup owner: ${bringup_pids[0]}"

required_patterns=(
    '[r]obot_state_publisher/robot_state_publisher'
    '[c]hamp_base/quadruped_controller_node'
    '[c]hamp_base/state_estimation_node'
    '[r]obot_localization/ekf_node.*base_to_footprint_ekf'
    '[r]obot_localization/ekf_node.*footprint_to_odom_ekf'
    '[l]dlidar_stl_ros2_node'
    '[s]ervo_interface'
    '[i]mu_interface'
)

for pattern in "${required_patterns[@]}"
do
    if ! pgrep -af "$pattern" >/dev/null; then
        echo "ERROR: Required standard ROS process is missing:"
        echo "$pattern"
        exit 1
    fi
done

echo "PASS: Required standard ROS processes are running."

forbidden_pattern='[g]uarded_navigation.launch.py|[p]lanning.launch.py|[l]ocalization.launch.py|[n]av2_map_server/map_server|[n]av2_amcl/amcl|[n]av2_planner/planner_server|[n]av2_controller/controller_server|[n]av2_bt_navigator/bt_navigator|[s]lam_toolbox|[c]artographer'

if pgrep -af "$forbidden_pattern"; then
    echo "ERROR: Nav2, localization, planning, or SLAM is active."
    exit 1
fi

echo "PASS: Nav2, planning, localization, and SLAM are absent."

bridge_pid=$(
    pgrep -f \
        '[p]ython3.*app.py' |
    head -n 1
)

test -n "$bridge_pid"

bridge_cwd=$(
    readlink -f "/proc/$bridge_pid/cwd"
)

bridge_command=$(
    tr '\0' ' ' <"/proc/$bridge_pid/cmdline"
)

test "$bridge_cwd" = "$HOME/robot_bridge"

echo "PASS: Robot Bridge identity verified: PID $bridge_pid"
echo "Robot Bridge command: $bridge_command"

echo
echo "===== REPOSITORY REPORT ====="

for repository in \
    "$HOME/robot_bridge" \
    "$HOME/ros2_ws/src/mini_pupper_ros" \
    "$HOME/ros2_ws/src/champ/champ"
do
    if [ ! -d "$repository/.git" ]; then
        echo "INFO: Repository unavailable: $repository"
        continue
    fi

    echo
    echo "Repository: $repository"
    git -C "$repository" status --short --branch

    if [ -n "$(git -C "$repository" diff --cached --name-only)" ]; then
        echo "ERROR: Staged repository changes were found."
        exit 1
    fi
done

echo
echo "PASS: No robot repository contains staged changes."
__REMOTE_MAYDAY_SESSION_CHECK_92C1__

echo
echo "===== VERIFY LIDAR TELEMETRY ====="

curl -fsS --connect-timeout 2 --max-time 10 \
    "$ROBOT/telemetry/lidar" \
    >/tmp/mayday_session_lidar.json

python3 - <<'__VERIFY_MAYDAY_LIDAR_B615__'
import json
from pathlib import Path

payload = json.loads(
    Path("/tmp/mayday_session_lidar.json").read_text(
        encoding="utf-8"
    )
)

telemetry = payload.get("telemetry", payload)

if not telemetry.get("available"):
    raise SystemExit("ERROR: LiDAR telemetry is unavailable.")

age = telemetry.get("age_seconds")

if not isinstance(age, (int, float)):
    raise SystemExit("ERROR: LiDAR age is unavailable.")

if age >= 2.0:
    raise SystemExit(
        f"ERROR: LiDAR telemetry is stale: {age:.3f}s"
    )

print(f"PASS: LiDAR is live: age {age:.3f}s.")
__VERIFY_MAYDAY_LIDAR_B615__

echo
echo "===== VERIFY SPEECH ====="

python3 - <<'__VERIFY_MAYDAY_SPEECH_5A12__'
import json
from pathlib import Path

payload = json.loads(
    Path("/tmp/mayday_session_status.json").read_text(
        encoding="utf-8"
    )
)

speech = payload.get("speech")

if not isinstance(speech, dict):
    raise SystemExit("ERROR: Speech status is unavailable.")

if not speech.get("available"):
    raise SystemExit("ERROR: Speech is unavailable.")

if speech.get("last_error"):
    raise SystemExit(
        "ERROR: Speech reports an error: "
        + str(speech["last_error"])
    )

print("PASS: Speech is available and error-free.")
__VERIFY_MAYDAY_SPEECH_5A12__

if [ "$MODE" = "--prepare" ]; then
    echo
    echo "===== VERIFY FINAL ZERO VELOCITY ====="

    curl -fsS --connect-timeout 2 --max-time 10 \
        -X POST "$ROBOT/stop" \
        >/tmp/mayday_session_stop.json

    python3 - <<'__VERIFY_MAYDAY_STOP_04DC__'
import json
from pathlib import Path

payload = json.loads(
    Path("/tmp/mayday_session_stop.json").read_text(
        encoding="utf-8"
    )
)

result = payload.get("stop_result", {})

if not payload.get("ok"):
    raise SystemExit("ERROR: Final safety stop failed.")

if result.get("linear_x") != 0.0:
    raise SystemExit("ERROR: Linear velocity is not zero.")

if result.get("angular_z") != 0.0:
    raise SystemExit("ERROR: Angular velocity is not zero.")

print("PASS: Mayday velocity is zero.")
__VERIFY_MAYDAY_STOP_04DC__
fi

echo
echo "===== SESSION READY ====="
echo "Standard ROS is healthy."
echo "Vision, Cognitive Runtime, dashboard, Robot Bridge, camera, LiDAR, and speech are available."
echo "Nav2 and SLAM are stopped."

if [ "$MODE" = "--prepare" ]; then
    echo "Mayday is stationary."
else
    echo "Run with --prepare before live hardware testing."
fi
