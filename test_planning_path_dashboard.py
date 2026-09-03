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



def test_goal_selection_uses_map_metadata_and_tony2_payload():
    for marker in (
        "canvasToMap",
        "occupancyMap.origin",
        "occupancyMap.resolution",
        "occupancyMap.cells[index] !== 0",
    ):
        assert marker in CONTROL

    start = CONTROL.index(
        "async function computePath()"
    )

    end = CONTROL.index(
        "function selectGoal",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    compact = " ".join(
        handler.split()
    )

    for marker in (
        "goal_x: selectedGoal.x",
        "goal_y: selectedGoal.y",
        "goal_yaw: selectedGoal.yaw",
    ):
        assert marker in compact

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
    for marker in (
        "Click a free map location, then press GO.",
        "Robot Bridge verifies localization",
        "guarded goal within 0.50 m and 25 seconds",
        "STOP",
        "MAYDAY immediately cancels the active goal.",
    ):
        assert marker in HTML

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



def test_start_runs_tony2_global_initializer_sequence():
    assert (
        'const START_ENDPOINT ='
        in CONTROL
    )

    assert (
        '"/dashboard/navigation-start"'
        in CONTROL
    )

    assert (
        'const INITIALIZE_ENDPOINT ='
        in CONTROL
    )

    assert (
        '"/dashboard/navigation-initialize-localization"'
        in CONTROL
    )

    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    assert (
        handler.index("START_ENDPOINT")
        < handler.index("INITIALIZE_ENDPOINT")
    )

    assert "navigationWasStarted(" in handler
    assert "validInitialization(" in handler


def test_start_adopts_trusted_tony2_pose_without_refresh():
    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    compact = " ".join(
        handler.split()
    )

    assert (
        "initialization.final_pose"
        in compact
    )

    assert (
        "initialization.uncertainty"
        in compact
    )

    assert "latestPose = {" in handler
    assert 'frame_id: "map"' in handler

    assert "POSE_REFRESH_ENDPOINT" not in handler
    assert "LOCALIZATION_ENDPOINT" not in handler


def test_startup_global_localization_request_is_parameter_free():
    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    request_start = handler.index(
        "INITIALIZE_ENDPOINT"
    )

    request_end = handler.index(
        "validInitialization(",
        request_start,
    )

    request = handler[
        request_start:request_end
    ]

    assert 'body: "{}"' in request

    for marker in (
        "goal_x",
        "goal_y",
        "goal_yaw",
        "selectedGoal",
        "planner_id",
        "cmd_vel",
        "initial_pose",
        '"service"',
        '"topic"',
    ):
        assert marker not in request


def test_startup_waits_for_trusted_tony2_pose_before_ready():
    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    validated = handler.index(
        "validInitialization("
    )

    complete = handler.index(
        "if (!initializationComplete)"
    )

    final_pose = handler.index(
        "initialization.final_pose"
    )

    ready = handler.index(
        "planningReady = true"
    )

    assert (
        validated
        < complete
        < final_pose
        < ready
    )


def test_startup_ready_requires_trusted_global_localization():
    start = CONTROL.index(
        "function validInitialization"
    )

    end = CONTROL.index(
        "function validPoseRefresh",
        start,
    )

    validator = CONTROL[
        start:end
    ]

    compact = " ".join(
        validator.split()
    )

    required = (
        "initialization.trusted === true",
        (
            "initialization.localization_method "
            '=== "amcl_global"'
        ),
        (
            "initialization.search_scope "
            '=== "full_saved_map"'
        ),
        "initialization.seed_pose_used === false",
        (
            "initialization.global_localization_requested "
            "=== true"
        ),
        (
            "initialization.initial_pose_supplied "
            "=== false"
        ),
        (
            "initialization.nomotion_updates_requested "
            "=== 20"
        ),
        "navigation.state === \"READY\"",
        "navigation.planner_enabled === true",
        "navigation.transform_ready === true",
        (
            "navigation.motion_output_connected "
            "=== false"
        ),
        "navigation.goal_active === false",
    )

    for marker in required:
        assert marker in compact

def test_new_pose_requires_numeric_coordinates():
    start = CONTROL.index("function isNewFreshPose")
    end = CONTROL.index(
        "async function startPlanning",
        start,
    )
    validator = CONTROL[start:end]

    for marker in (
        "pose && pose.position",
        "position && position.x",
        "position && position.y",
        "pose && pose.yaw_radians",
        "Number.isFinite(poseX)",
        "Number.isFinite(poseY)",
        "Number.isFinite(poseYaw)",
    ):
        assert marker in validator



def test_startup_localization_failure_stops_tony2_navigation():
    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    catch_start = handler.index(
        "} catch (error) {"
    )

    failure = handler[
        catch_start:
    ]

    assert "if (navigationStarted)" in failure
    assert "STOP_ENDPOINT" in failure

    assert (
        failure.index(
            "if (navigationStarted)"
        )
        < failure.index(
            "STOP_ENDPOINT"
        )
    )


def test_initializer_failure_clears_planning_ui_state():
    start = CONTROL.index(
        "async function startPlanning()"
    )

    end = CONTROL.index(
        "async function performAction(",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    failure = handler[
        handler.index(
            "} catch (error) {"
        ):
    ]

    for marker in (
        "planningRunning = false",
        "planningReady = false",
        "latestPose = null",
        "latestPoseReceivedAt = null",
        "selectedGoal = null",
        "computedPath = null",
        "uncertaintyBlocked = true",
    ):
        assert marker in failure

    assert "if (navigationStarted)" in failure
    assert "STOP_ENDPOINT" in failure


def test_tony2_start_waits_for_amcl_discovery():
    assert (
        "const INITIALIZE_DISCOVERY_DELAY_MS = 2000"
        in CONTROL
    )

    assert (
        "await wait(\n"
        "                INITIALIZE_DISCOVERY_DELAY_MS"
        in CONTROL
    )

    assert (
        "Mayday must remain stationary while "
        in CONTROL
    )

    assert (
        "Tony2 AMCL searches the saved map..."
        in CONTROL
    )


def test_tony2_global_initializer_retries_are_bounded():
    assert (
        "const INITIALIZE_MAX_ATTEMPTS = 10"
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

    compact = " ".join(
        CONTROL.split()
    )

    assert (
        "initializationComplete = true"
        in compact
    )


def test_tony2_navigation_stays_active_between_localization_retries():
    retry_start = CONTROL.index(
        "for (\n"
        "                let attempt = 1;"
    )

    retry_end = CONTROL.index(
        "if (!initializationComplete)",
        retry_start,
    )

    retry = CONTROL[
        retry_start:retry_end
    ]

    assert "INITIALIZE_ENDPOINT" in retry
    assert "INITIALIZE_RETRY_DELAY_MS" in retry
    assert "STOP_ENDPOINT" not in retry

    assert (
        "Retrying stationary "
        in retry
    )

    assert (
        "global localization..."
        in retry
    )


def test_tony2_initializer_validation_preserves_safety_contract():
    start = CONTROL.index(
        "function validInitialization"
    )

    end = CONTROL.index(
        "function validPoseRefresh",
        start,
    )

    validator = CONTROL[
        start:end
    ]

    compact = " ".join(
        validator.split()
    )

    required = (
        "initialization.trusted === true",
        (
            "initialization.localization_method "
            '=== "amcl_global"'
        ),
        (
            "initialization.search_scope "
            '=== "full_saved_map"'
        ),
        (
            "initialization.seed_pose_used "
            "=== false"
        ),
        (
            "initialization.global_localization_requested "
            "=== true"
        ),
        (
            "initialization.nomotion_updates_requested "
            "=== 20"
        ),
        (
            "initialization.stationary_required "
            "=== true"
        ),
        (
            "initialization.initial_pose_supplied "
            "=== false"
        ),
        (
            "initialization.navigation_goal_executed "
            "=== false"
        ),
        (
            "initialization.motion_enabled "
            "=== false"
        ),
        (
            "navigation.motion_output_connected "
            "=== false"
        ),
        (
            "navigation.goal_active "
            "=== false"
        ),
    )

    for marker in required:
        assert marker in compact

def test_pose_refresh_proxy_is_fixed_and_parameter_free():
    start = SERVER.index(
        "def planning_refresh_localization"
    )
    end = SERVER.index(
        "def planning_compute_path",
        start,
    )
    proxy = SERVER[start:end]

    assert '"/planning/refresh-localization"' in proxy
    assert "payload={}" in proxy
    assert "timeout=20.0" in proxy

    for marker in (
        "request_payload",
        "goal_x",
        "goal_y",
        "goal_yaw",
        "initial_pose",
        '"service"',
        '"topic"',
        '"frame"',
        "planner_id",
        "cmd_vel",
    ):
        assert marker not in proxy


def test_pose_refresh_dashboard_route_is_post_only():
    get_start = SERVER.index("def do_GET(self):")
    post_start = SERVER.index("def do_POST(self):")
    get_source = SERVER[get_start:post_start]
    post_source = SERVER[post_start:]
    route = "/dashboard/planning-refresh-localization"

    assert route not in get_source
    assert route in post_source
    assert "self.planning_refresh_localization()" in post_source




def test_compute_requires_trusted_cached_tony2_pose():
    start = CONTROL.index(
        "async function computePath()"
    )

    end = CONTROL.index(
        "function selectGoal",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    guard_end = handler.index(
        "actionInFlight = true"
    )

    guard = handler[
        :guard_end
    ]

    for marker in (
        "|| !planningRunning",
        "|| !planningReady",
        "|| !selectedGoal",
        "|| !latestPose",
        "|| uncertaintyBlocked",
    ):
        assert marker in guard

    assert "POSE_REFRESH_ENDPOINT" not in handler


def test_compute_uses_tony2_path_without_pose_refresh():
    start = CONTROL.index(
        "async function computePath()"
    )

    end = CONTROL.index(
        "function selectGoal",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    assert "COMPUTE_ENDPOINT" in handler
    assert "POSE_REFRESH_ENDPOINT" not in handler
    assert "LOCALIZATION_ENDPOINT" not in handler

    assert (
        '"COMPUTE_PATH_TO_POSE_ONLY"'
        in handler
    )

    assert (
        "path.read_only !== true"
        in handler
    )

    assert (
        "path.executed !== false"
        in handler
    )

    assert (
        "path.motion_enabled !== false"
        in handler
    )

def test_pose_refresh_contract_preserves_safety():
    start = CONTROL.index("function validPoseRefresh")
    end = CONTROL.index("function isNewFreshPose", start)
    validator = CONTROL[start:end]

    required = (
        'action\n                === "PLANNING_POSE_REFRESHED"',
        "nomotion_updates_requested === 1",
        "global_localization_requested === false",
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


def test_refreshed_pose_must_be_new_map_frame_and_fresh():
    start = CONTROL.index("function isNewFreshPose")
    end = CONTROL.index("async function startPlanning", start)
    validator = CONTROL[start:end]

    required = (
        "payload.runtime_active === true",
        "telemetry.available === true",
        'pose.frame_id === "map"',
        "Number.isFinite(ageSeconds)",
        "ageSeconds < 3",
        "telemetry.received_at",
        "!== previousReceivedAt",
    )

    for value in required:
        assert value in validator



def test_compute_has_no_localization_retry_loop():
    start = CONTROL.index(
        "async function computePath()"
    )

    boundaries = []

    for marker in (
        "\n    function ",
        "\n    async function ",
    ):
        found = CONTROL.find(
            marker,
            start + 1,
        )

        if found >= 0:
            boundaries.append(
                found
            )

    assert boundaries

    handler = CONTROL[
        start:min(boundaries)
    ]

    assert "COMPUTE_ENDPOINT" in handler

    for obsolete in (
        "POSE_REFRESH_MAX_ATTEMPTS",
        "POSE_REFRESH_RETRY_DELAY_MS",
        "INITIALIZE_ENDPOINT",
        "POSE_REFRESH_ENDPOINT",
        "LOCALIZATION_ENDPOINT",
        "NAVIGATION_INITIALIZE_ENDPOINT",
        "NAVIGATION_GOAL_ENDPOINT",
    ):
        assert obsolete not in handler

def test_compute_request_accepts_only_selected_map_goal():
    start = CONTROL.index(
        "async function computePath()"
    )

    end = CONTROL.index(
        "function selectGoal",
        start,
    )

    handler = CONTROL[
        start:end
    ]

    request_start = handler.index(
        "body: JSON.stringify({"
    )

    request_end = handler.index(
        "}),",
        request_start,
    )

    request = handler[
        request_start:request_end
    ]

    compact = " ".join(
        request.split()
    )

    for marker in (
        "goal_x: selectedGoal.x",
        "goal_y: selectedGoal.y",
        "goal_yaw: selectedGoal.yaw",
    ):
        assert marker in compact

    for forbidden in (
        "service:",
        "topic:",
        "frame_id:",
        "planner_id:",
        "initial_pose",
        "cmd_vel",
        "linear_x",
        "angular_z",
    ):
        assert forbidden not in request

def test_pose_refresh_adds_no_motion_or_navigation_execution():
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


def test_normal_navigation_is_select_then_go():
    dashboard = Path("voice_relay/index.html").read_text(
        encoding="utf-8"
    )

    assert "Guarded Map Navigation" in dashboard
    assert "GO — max 0.50 m" in dashboard
    assert (
        "Click a free map location, then press GO."
        in dashboard
    )

    button_start = dashboard.index(
        "function updateButtons()"
    )
    button_end = dashboard.index(
        "function renderUncertainty",
        button_start,
    )
    buttons = dashboard[button_start:button_end]

    assert "|| !occupancyMap" in buttons
    assert "|| !selectedGoal" in buttons

    send_start = dashboard.index(
        "async function sendMayday()"
    )
    send_end = dashboard.index(
        "function selectGoal",
        send_start,
    )
    send = dashboard[send_start:send_end]

    for obsolete_guard in (
        "|| !planningRunning",
        "|| !computedPath",
        "pathLength > MAX_NAVIGATION_DISTANCE_METERS",
    ):
        assert obsolete_guard not in send

    assert send.index("NAVIGATION_START_ENDPOINT") < send.index(
        "NAVIGATION_INITIALIZE_ENDPOINT"
    )
    assert send.index("NAVIGATION_INITIALIZE_ENDPOINT") < send.index(
        "NAVIGATION_GOAL_ENDPOINT"
    )

    select_start = dashboard.index(
        "function selectGoal"
    )
    select = dashboard[
        select_start:select_start + 1400
    ]

    assert "!planningRunning" not in select
    assert "actionInFlight || navigationExecuting" in select


def test_direct_go_uses_short_bounded_initialization_waits():
    assert (
        "const INITIALIZE_DISCOVERY_DELAY_MS = 2000;"
        in CONTROL
    )
    assert (
        "const INITIALIZE_RETRY_DELAY_MS = 2000;"
        in CONTROL
    )
    assert (
        "const INITIALIZE_MAX_ATTEMPTS = 10;"
        in CONTROL
    )


def test_planning_buttons_are_visible_with_guarded_defaults():
    dashboard = Path(
        "voice_relay/index.html"
    ).read_text(
        encoding="utf-8"
    )

    expected = {
        "startPlanningButton": False,
        "stopPlanningButton": True,
        "computePlanningPathButton": True,
    }

    for button_id, should_be_disabled in expected.items():
        marker = (
            f'id="{button_id}"'
        )

        marker_index = dashboard.index(
            marker
        )

        start = dashboard.rfind(
            "<button",
            0,
            marker_index,
        )

        end = dashboard.index(
            ">",
            marker_index,
        )

        opening = dashboard[
            start:end
        ]

        assert " hidden" not in opening

        if should_be_disabled:
            assert " disabled" in opening
        else:
            assert " disabled" not in opening
