#!/usr/bin/env python3

"""
Static regression tests for normal fixed-map navigation routing.

The public dashboard continues to use /dashboard/navigation-*.

Those routes MUST execute Tony2NavigationRuntime, whose Nav2 graph
is isolated on ROS domain 43 / rmw_zenoh_cpp.

Mayday's hardware graph and Robot Bridge remain domain 42 /
rmw_fastrtps_cpp.

Normal dashboard navigation must never directly start Mayday's
domain-42 Nav2 runtime.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(
    encoding="utf-8"
)

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(
    encoding="utf-8"
)

SUPERVISOR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_supervisor.py"
).read_text(
    encoding="utf-8"
)

EGRESS = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_motion_egress.py"
).read_text(
    encoding="utf-8"
)


def method_source(
    name,
    next_name,
):
    start = SERVER.index(
        f"    def {name}("
    )

    end = SERVER.index(
        f"    def {next_name}(",
        start,
    )

    return SERVER[start:end]


def test_normal_navigation_status_uses_tony2_runtime():
    source = method_source(
        "navigation_control_status",
        "navigation_control_action",
    )

    assert (
        "self.mapping_navigation_status()"
        in source
    )

    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/status"'
        not in source
    )

    assert (
        'response["navigation"] = navigation'
        in source
    )


def test_normal_navigation_start_stop_use_tony2_runtime():
    source = method_source(
        "navigation_control_action",
        "navigation_initialize_localization",
    )

    assert (
        "self.mapping_navigation_control_action("
        in source
    )

    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/{action}"'
        not in source
    )

    assert (
        'response["navigation"] = navigation'
        in source
    )

    assert (
        'f"navigation_{action}"'
        in source
    )



def test_normal_navigation_requests_home_localization():
    source = method_source(
        "navigation_initialize_localization",
        "navigation_goal",
    )

    assert "runtime.initialize_home_localization()" in source
    assert "runtime.initialize_global_localization()" not in source
    assert "self.ensure_mayday_stationary()" in source




def test_normal_navigation_initialization_is_home_seeded():
    source = method_source(
        "navigation_initialize_localization",
        "navigation_goal",
    )

    compact = "".join(source.split())

    assert 'localization.get("seed_pose_used")isTrue' in compact
    assert 'localization.get("initial_pose_supplied")isTrue' in compact
    assert (
        'localization.get("global_localization_requested")'
        'isFalse'
        in compact
    )



def test_normal_navigation_goal_uses_tony2_runtime():
    source = method_source(
        "navigation_goal",
        "mapping_pose_status",
    )

    assert (
        "self.mapping_navigation_goal(payload)"
        in source
    )

    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/goal"'
        not in source
    )

    assert (
        'response["navigation"] = navigation'
        in source
    )

    assert (
        'response["action"] = "navigation_goal"'
        in source
    )


def test_public_navigation_routes_are_preserved():
    required = (
        "/dashboard/navigation-start",
        "/dashboard/navigation-stop",
        "/dashboard/navigation-initialize-localization",
        "/dashboard/navigation-goal",
    )

    for route in required:
        assert route in SERVER
        assert route in HTML


def test_dashboard_requires_tony2_isolation_on_start():
    start = HTML.index(
        "function navigationWasStarted(result)"
    )

    end = HTML.index(
        "function navigationInitializationSucceeded",
        start,
    )

    source = HTML[start:end]

    required = (
        'navigation.host === "Tony2"',
        '=== "tony2_guarded_navigation"',
        "navigation.fixed_map_required === true",
        '=== "zenoh_localhost"',
        "navigation.motion_output_connected",
        "=== false",
    )

    for marker in required:
        assert marker in source



def test_dashboard_requires_validated_home_before_go():
    start = HTML.index(
        "function navigationInitializationSucceeded(result)"
    )
    end = HTML.index("async function", start)

    compact = "".join(HTML[start:end].split())

    required = (
        "initialization.trusted===true",
        'initialization.localization_method==="amcl_seeded"',
        'initialization.search_scope==="known_home_pose"',
        "initialization.seed_pose_used===true",
        "initialization.global_localization_requested===false",
        "initialization.initial_pose_supplied===true",
        "initialization.stationary_required===true",
        "initialization.navigation_goal_executed===false",
        "initialization.motion_enabled===false",
        'navigation.state==="READY"',
        'navigation.host==="Tony2"',
        "navigation.action_server_ready===true",
        "navigation.transform_ready===true",
        "navigation.goal_submission_enabled===true",
        "navigation.motion_output_connected===false",
        'navigation.isolation_transport==="zenoh_localhost"',
    )

    for marker in required:
        assert marker in compact



def test_goal_payload_remains_fixed_and_numeric():
    required = (
        "goal_x: executionGoal.x",
        "goal_y: executionGoal.y",
        "goal_yaw: executionGoal.yaw",
    )

    for marker in required:
        assert marker in HTML

    assert (
        "const MAX_NAVIGATION_DISTANCE_METERS = 0.50"
        in HTML
    )


def test_navigation_goal_keeps_persistent_runtime_running():
    start = HTML.index(
        "async function sendMayday"
    )

    end = HTML.index(
        "\n    function selectGoal(",
        start,
    )

    source = HTML[
        start:
        end
    ]

    assert (
        "NAVIGATION_GOAL_ENDPOINT"
        in source
    )

    # A normal destination attempt must not tear down the
    # already-running persistent Nav2 runtime.
    assert (
        "NAVIGATION_STOP_ENDPOINT"
        not in source
    )

    assert (
        "NAVIGATION_START_ENDPOINT"
        not in source
    )

    assert (
        "NAVIGATION_INITIALIZE_ENDPOINT"
        not in source
    )

    assert (
        "navigation.goal_submission_enabled !== true"
        in source
    )

    assert (
        "navigation.motion_output_connected !== false"
        in source
    )

    assert (
        "Persistent navigation remains READY."
        in source
    )

    # Explicit operator STOP remains independently available.
    stop_start = HTML.index(
        "async function stopMayday"
    )

    stop_end = HTML.index(
        "\n    async function sendMayday",
        stop_start,
    )

    stop_source = HTML[
        stop_start:
        stop_end
    ]

    assert (
        "NAVIGATION_STOP_ENDPOINT"
        in stop_source
    )

def test_tony2_runtime_reports_fixed_map_and_zenoh():
    required = (
        '"host": "Tony2"',
        '"source":',
        '"tony2_guarded_navigation"',
        '"fixed_map_required": True',
        '"localization_mode":',
        '"amcl"',
        '"isolation_transport":',
        '"zenoh_localhost"',
    )

    for marker in required:
        assert marker in RUNTIME


def test_tony2_runtime_requires_ready_before_goal():
    start = RUNTIME.index(
        "    def submit_goal("
    )

    source = RUNTIME[start:]

    assert (
        'status.get("state") != "READY"'
        in source
    )

    assert (
        '"goal_submission_enabled"'
        in source
    )

    assert (
        '"motion_output_connected"'
        in source
    )

    assert (
        "self._begin_motion_lease()"
        in source
    )

    assert (
        "self._wait_for_motion_arm_ack("
        in source
    )


def test_supervisor_keeps_domain_42_ingress():
    required = (
        '"ROS_DOMAIN_ID=42"',
        '"ROS_LOCALHOST_ONLY=0"',
        '"rmw_fastrtps_cpp"',
    )

    for marker in required:
        assert marker in SUPERVISOR


def test_supervisor_keeps_domain_43_nav2():
    required = (
        '"ROS_DOMAIN_ID": "43"',
        '"ROS_DOMAIN_ID=43"',
        '"RMW_IMPLEMENTATION=rmw_zenoh_cpp"',
        "ZENOH_CONFIG_OVERRIDE",
        "127.0.0.1:7447",
    )

    for marker in required:
        assert marker in SUPERVISOR


def test_controller_output_is_not_physical_cmd_vel():
    assert (
        '"/tony2_nav_"'
        in SUPERVISOR
    )

    assert (
        '"cmd_vel_egress"'
        in SUPERVISOR
    )

    assert (
        'INPUT_TOPIC = "/tony2_nav_cmd_vel_egress"'
        in EGRESS
    )

    assert (
        "It never publishes ROS /cmd_vel directly."
        in EGRESS
    )


def test_egress_requires_transient_arm_lease():
    required = (
        "validate_arm_payload",
        "self._armed_token",
        "def _sync_arm_state(",
        "lease",
        "token",
    )

    for marker in required:
        assert marker in EGRESS


def test_normal_server_navigation_has_no_direct_robot_nav2_calls():
    start = SERVER.index(
        "    def navigation_control_status("
    )

    end = SERVER.index(
        "    def mapping_pose_status(",
        start,
    )

    source = SERVER[start:end]

    forbidden = (
        'f"{ROBOT_BRIDGE_URL}/navigation/status"',
        'f"{ROBOT_BRIDGE_URL}/navigation/{action}"',
        'f"{ROBOT_BRIDGE_URL}/navigation/goal"',
        '"/navigation/initialize-localization"',
    )

    for marker in forbidden:
        assert marker not in source

def test_go_submits_directly_through_persistent_navigation():
    from pathlib import Path

    html = Path(
        "voice_relay/index.html"
    ).read_text(
        encoding="utf-8"
    )

    start = html.index(
        "    async function sendMayday() {"
    )

    end = html.index(
        "\n    function selectGoal(",
        start,
    )

    block = html[
        start:
        end
    ]

    required = (
        "STATUS_ENDPOINT",
        "NAVIGATION_GOAL_ENDPOINT",
        'navigation.state !== "READY"',
        "navigation.running !== true",
        "navigation.owned !== true",
        'navigation.host !== "Tony2"',
        (
            'navigation.source\n'
            '                    !== '
            '"tony2_guarded_navigation"'
        ),
        "navigation.action_server_ready !== true",
        "navigation.transform_ready !== true",
        "navigation.motion_egress_ready !== true",
        "navigation.motion_egress_idle !== true",
        "navigation.goal_submission_enabled !== true",
        "navigation.goal_active !== false",
        "navigation.motion_output_connected !== false",
        "goal_x: executionGoal.x",
        "goal_y: executionGoal.y",
        "goal_yaw: executionGoal.yaw",
        'result.status\n                    !== '
        '"NAVIGATION_SUCCEEDED"',
        "result.executed !== true",
        "result.bounded !== true",
        "Persistent navigation remains READY.",
    )

    for marker in required:
        assert marker in block

    forbidden = (
        "STOP_ENDPOINT,",
        "NAVIGATION_START_ENDPOINT",
        "NAVIGATION_INITIALIZE_ENDPOINT",
        "navigationInitializationSucceeded(",
        "INITIALIZE_DISCOVERY_DELAY_MS",
        "INITIALIZE_RETRY_DELAY_MS",
        "INITIALIZE_MAX_ATTEMPTS",
        "NAVIGATION_STOP_ENDPOINT",
        "known_home_pose",
        "amcl_seeded",
        "planningRunning = false",
        "planningReady = false",
    )

    for marker in forbidden:
        assert marker not in block
