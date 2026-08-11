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
    assert (
        "function validInitialization(result)"
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
    assert (
        "validInitialization("
        in handler
    )
    assert (
        "INITIALIZE_DISCOVERY_DELAY_MS"
        in handler
    )
    assert (
        "INITIALIZE_MAX_ATTEMPTS"
        in handler
    )

    validator_start = CONTROL.index(
        "function validInitialization(result)"
    )
    validator_end = CONTROL.index(
        "async function startPlanning()",
        validator_start,
    )
    validator = CONTROL[
        validator_start:validator_end
    ]

    required = (
        "global_localization_requested === true",
        "nomotion_updates_requested === 20",
        "stationary_required === true",
        "initial_pose_supplied === false",
        "pose_published === false",
        "path_computed === false",
        "path_executed === false",
        "navigation_goal_executed === false",
        "controller_enabled === false",
        "navigator_enabled === false",
        "motion_enabled === false",
    )

    for value in required:
        assert value in validator

    assert "STOP_ENDPOINT" in handler
    assert (
        handler.index("INITIALIZE_ENDPOINT")
        < handler.index("STOP_ENDPOINT")
    )

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

def test_planning_start_waits_for_amcl_discovery():
    assert (
        "const INITIALIZE_DISCOVERY_DELAY_MS = 10000"
        in CONTROL
    )
    assert (
        "await wait(\n"
        "                INITIALIZE_DISCOVERY_DELAY_MS"
        in CONTROL
    )
    assert (
        "Planning-only Nav2 started. Waiting for "
        in CONTROL
    )
    assert (
        "AMCL services to become available..."
        in CONTROL
    )


def test_planning_initializer_retries_are_bounded():
    assert (
        "const INITIALIZE_MAX_ATTEMPTS = 4"
        in CONTROL
    )
    assert (
        "attempt <= INITIALIZE_MAX_ATTEMPTS"
        in CONTROL
    )
    assert (
        "INITIALIZE_RETRY_DELAY_MS"
        in CONTROL
    )
    assert (
        "initializationComplete = true"
        in CONTROL
    )


def test_planning_stays_active_between_initializer_retries():
    retry_start = CONTROL.index(
        "for (\n"
        "                let attempt = 1;"
    )
    retry_end = CONTROL.index(
        "if (!initializationComplete)",
        retry_start,
    )
    retry = CONTROL[retry_start:retry_end]

    assert "INITIALIZE_ENDPOINT" in retry
    assert "INITIALIZE_RETRY_DELAY_MS" in retry
    assert "STOP_ENDPOINT" not in retry
    assert (
        "Planning remains safely active"
        in retry
    )


def test_initializer_validation_preserves_safety_contract():
    validator_start = CONTROL.index(
        "function validInitialization"
    )
    validator_end = CONTROL.index(
        "async function startPlanning",
        validator_start,
    )
    validator = CONTROL[
        validator_start:validator_end
    ]

    required = (
        "global_localization_requested === true",
        "nomotion_updates_requested === 20",
        "stationary_required === true",
        "initial_pose_supplied === false",
        "pose_published === false",
        "path_computed === false",
        "path_executed === false",
        "navigation_goal_executed === false",
        "controller_enabled === false",
        "navigator_enabled === false",
        "motion_enabled === false",
    )

    for value in required:
        assert value in validator
