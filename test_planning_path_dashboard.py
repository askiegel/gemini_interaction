from pathlib import Path


ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "voice_relay/index.html").read_text(
    encoding="utf-8"
)
CSS = (
    ROOT / "voice_relay/operator_console.css"
).read_text(encoding="utf-8")
SERVER = (
    ROOT / "voice_relay/server.py"
).read_text(encoding="utf-8")

START = HTML.index(
    "/* Isolated guarded read-only planning "
    "path controller */"
)
END = HTML.index("</script>", START)
CONTROL = HTML[START:END]


def test_planning_path_workspace_exists():
    identifiers = (
        "planningPathCanvas",
        "planningPathOverlayStatus",
        "planningPathState",
        "startPlanningButton",
        "stopPlanningButton",
        "computePlanningPathButton",
        "planningSelectedGoal",
        "planningPathSummary",
        "planningPoseUncertainty",
        "planningReadiness",
    )

    for identifier in identifiers:
        assert HTML.count(
            f'id="{identifier}"'
        ) == 1


def test_planning_proxies_are_fixed():
    assert (
        'f"{ROBOT_BRIDGE_URL}/planning/status"'
        in SERVER
    )
    assert (
        'f"{ROBOT_BRIDGE_URL}/planning/{action}"'
        in SERVER
    )
    assert (
        '"/planning/compute-path"'
        in SERVER
    )
    assert 'action not in ("start", "stop")' in SERVER


def test_compute_proxy_validates_exact_payload():
    for field in (
        "goal_x",
        "goal_y",
        "goal_yaw",
    ):
        assert f'"{field}"' in SERVER

    assert "math.isfinite" in SERVER
    assert "planner_id" not in (
        SERVER[
            SERVER.index(
                "def planning_compute_path"
            ):
            SERVER.index(
                "def localization_status"
            )
        ]
    )


def test_planning_method_scope_is_guarded():
    assert (
        '"/dashboard/planning-control"'
        in SERVER
    )
    assert (
        '"/dashboard/planning-start"'
        in SERVER
    )
    assert (
        '"/dashboard/planning-stop"'
        in SERVER
    )
    assert (
        '"/dashboard/planning-compute-path"'
        in SERVER
    )


def test_controller_uses_read_only_contract():
    for marker in (
        "path.read_only !== true",
        "path.executed !== false",
        "path.motion_enabled !== false",
        'path.frame_id !== "map"',
    ):
        assert marker in CONTROL


def test_goal_selection_uses_map_metadata():
    for marker in (
        "canvasToMap",
        "occupancyMap.origin",
        "occupancyMap.resolution",
        "occupancyMap.cells[index] !== 0",
        "goal_x: selectedGoal.x",
        "goal_y: selectedGoal.y",
        "goal_yaw: selectedGoal.yaw",
    ):
        assert marker in CONTROL


def test_high_uncertainty_blocks_compute():
    assert "MAX_POSITION_STD_METERS" in CONTROL
    assert "MAX_YAW_STD_RADIANS" in CONTROL
    assert "uncertaintyBlocked" in CONTROL
    assert (
        "Blocked — pose uncertain"
        in CONTROL
    )


def test_stopped_state_clears_path():
    assert "function clearPath()" in CONTROL
    assert "selectedGoal = null" in CONTROL
    assert "latestPose = null" in CONTROL
    assert (
        'endpoint === STOP_ENDPOINT'
        in CONTROL
    )


def test_path_is_drawn_as_overlay_only():
    assert "drawPath" in CONTROL
    assert 'context.strokeStyle = "#a855f7"' in CONTROL
    assert (
        "#planningPathCanvas"
        in CSS
    )
    assert "background: transparent" in CSS


def test_no_execution_or_motion_capability():
    forbidden = (
        "NavigateToPose",
        "FollowPath",
        "cmd_vel",
        "controller_server",
        "bt_navigator",
        "behavior_server",
        "waypoint_follower",
        "velocity_smoother",
    )

    for marker in forbidden:
        assert marker not in CONTROL


def test_existing_saved_map_controller_remains_read_only():
    start = (
        ROOT
        / "voice_relay/operator_console.js"
    ).read_text(
        encoding="utf-8"
    ).index(
        "/* Read-only saved occupancy-map "
        "visualization */"
    )
    source = (
        ROOT
        / "voice_relay/operator_console.js"
    ).read_text(encoding="utf-8")
    end = source.index(
        "/* Read-only localization pose overlay */",
        start,
    )
    saved_map = source[start:end]

    assert 'method: "POST"' not in saved_map
    assert "/planning/" not in saved_map


def test_safety_copy_is_visible():
    assert (
        "This panel cannot execute the path"
        in HTML
    )
    assert (
        "or move Mayday"
        in HTML
    )

def test_initializer_proxy_is_fixed():
    start = SERVER.index(
        "def planning_initialize_localization"
    )
    end = SERVER.index(
        "def planning_compute_path",
        start,
    )
    initializer = SERVER[start:end]

    assert (
        '"/planning/initialize-localization"'
        in initializer
    )
    assert 'payload={}' in initializer
    assert 'timeout=75.0' in initializer

    for marker in (
        "request_payload",
        "goal_x",
        "goal_y",
        "goal_yaw",
        "initial_pose",
        "planner_id",
        "cmd_vel",
    ):
        assert marker not in initializer


def test_initializer_route_is_post_only():
    get_start = SERVER.index(
        "def do_GET(self):"
    )
    post_start = SERVER.index(
        "def do_POST(self):"
    )
    get_source = SERVER[get_start:post_start]
    post_source = SERVER[post_start:]

    route = (
        "/dashboard/planning-"
        "initialize-localization"
    )

    assert route not in get_source
    assert route in post_source


def test_start_runs_fixed_initializer_sequence():
    assert (
        'const INITIALIZE_ENDPOINT ='
        in CONTROL
    )
    assert (
        '"/dashboard/planning-'
        'initialize-localization"'
        in CONTROL
    )
    assert (
        "async function startPlanning()"
        in CONTROL
    )

    start = CONTROL.index(
        "async function startPlanning()"
    )
    end = CONTROL.index(
        "async function performAction(",
        start,
    )
    handler = CONTROL[start:end]

    start_fetch = handler.index(
        "START_ENDPOINT"
    )
    initialize_fetch = handler.index(
        "INITIALIZE_ENDPOINT"
    )

    assert start_fetch < initialize_fetch
    assert "nomotion_updates_requested" in handler
    assert "!== 20" in handler
    assert "stationary_required" in handler
    assert "path_computed" in handler
    assert "path_executed" in handler
    assert "navigation_goal_executed" in handler
    assert "motion_enabled" in handler


def test_initializer_failure_stops_planning():
    start = CONTROL.index(
        "async function startPlanning()"
    )
    end = CONTROL.index(
        "async function performAction(",
        start,
    )
    handler = CONTROL[start:end]

    assert "if (planningStarted)" in handler
    assert "STOP_ENDPOINT" in handler
    assert "planningRunning = false" in handler
    assert "latestPose = null" in handler
    assert "selectedGoal = null" in handler
    assert "computedPath = null" in handler
