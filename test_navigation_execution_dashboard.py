from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = (
    ROOT / "voice_relay/server.py"
).read_text(encoding="utf-8")
HTML = (
    ROOT / "voice_relay/index.html"
).read_text(encoding="utf-8")


def method(name, next_name):
    start = SERVER.index(f"    def {name}(")
    end = SERVER.index(f"    def {next_name}(", start)
    return SERVER[start:end]


def test_navigation_status_proxy_is_fixed():
    source = method(
        "navigation_control_status",
        "navigation_control_action",
    )

    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/status"'
        in source
    )
    assert '"GET"' in source
    assert "timeout=5.0" in source


def test_navigation_actions_are_fixed():
    source = method(
        "navigation_control_action",
        "navigation_initialize_localization",
    )

    assert 'action not in ("start", "stop")' in source
    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/{action}"'
        in source
    )

    for marker in (
        "goal_x",
        "goal_y",
        "goal_yaw",
        "cmd_vel",
        "controller",
        "navigator",
        "launch",
        "map_yaml",
    ):
        assert marker not in source


def test_navigation_initializer_is_parameter_free():
    source = method(
        "navigation_initialize_localization",
        "navigation_goal",
    )

    assert (
        '"/navigation/initialize-localization"'
        in source
    )
    assert "payload={}" in source
    assert "timeout=75.0" in source

    for marker in (
        "request_payload",
        "goal_x",
        "goal_y",
        "goal_yaw",
        "initial_pose",
        "cmd_vel",
    ):
        assert marker not in source


def test_navigation_goal_accepts_only_numeric_goal():
    source = method(
        "navigation_goal",
        "localization_status",
    )

    for field in (
        '"goal_x"',
        '"goal_y"',
        '"goal_yaw"',
    ):
        assert field in source

    assert "set(payload) != required" in source
    assert "isinstance(value, bool)" in source
    assert "math.isfinite(value)" in source
    assert (
        'f"{ROBOT_BRIDGE_URL}/navigation/goal"'
        in source
    )
    assert "timeout=25.0" in source


def test_navigation_routes_have_fixed_methods():
    get_start = SERVER.index("    def do_GET(self):")
    post_start = SERVER.index("    def do_POST(self):")
    get_source = SERVER[get_start:post_start]
    post_source = SERVER[post_start:]

    assert (
        '"/dashboard/navigation-control"'
        in get_source
    )

    for route in (
        "/dashboard/navigation-start",
        "/dashboard/navigation-stop",
        "/dashboard/navigation-initialize-localization",
        "/dashboard/navigation-goal",
    ):
        assert f'"{route}"' in post_source
        assert f'"{route}"' not in get_source


def test_proxy_does_not_expand_navigation_limits():
    source = SERVER[
        SERVER.index("    def navigation_control_status("):
        SERVER.index("    def mapping_pose_status(")
    ]

    for marker in (
        "0.25",
        "15.0",
        "MAXIMUM_GOAL_DISTANCE",
        "MAXIMUM_EXECUTION",
        "retries",
        "recoveries",
        "behavior_tree",
    ):
        assert marker not in source


def test_send_mayday_controls_exist():
    for identifier in (
        "sendMaydayButton",
        "stopMaydayButton",
    ):
        assert HTML.count(f'id="{identifier}"') == 1

    assert "GO — max 0.50 m" in HTML
    assert "STOP MAYDAY" in HTML


def test_send_mayday_uses_fixed_dashboard_routes():
    for route in (
        "/dashboard/navigation-start",
        "/dashboard/navigation-stop",
        "/dashboard/navigation-initialize-localization",
        "/dashboard/navigation-goal",
    ):
        assert f'"{route}"' in HTML


def test_send_requires_preview_within_fixed_limit():
    assert (
        "const MAX_NAVIGATION_DISTANCE_METERS = 0.50"
        in HTML
    )
    assert "function sendMayday()" in HTML
    assert (
        "> MAX_NAVIGATION_DISTANCE_METERS"
        in HTML
    )
    assert "!selectedGoal" in HTML


def test_send_stops_planning_before_navigation():
    start = HTML.index("async function sendMayday()")
    end = HTML.index(
        "function selectGoal",
        start,
    )
    source = HTML[start:end]

    planning_stop = source.index("STOP_ENDPOINT")
    navigation_start = source.index(
        "NAVIGATION_START_ENDPOINT"
    )
    initialization = source.index(
        "NAVIGATION_INITIALIZE_ENDPOINT"
    )
    goal = source.index("NAVIGATION_GOAL_ENDPOINT")

    assert (
        planning_stop
        < navigation_start
        < initialization
        < goal
    )


def test_navigation_goal_payload_is_fixed():
    start = HTML.index("async function sendMayday()")
    end = HTML.index(
        "function selectGoal",
        start,
    )
    source = HTML[start:end]

    for marker in (
        "goal_x: executionGoal.x",
        "goal_y: executionGoal.y",
        "goal_yaw: executionGoal.yaw",
    ):
        assert marker in source

    for forbidden in (
        "service:",
        "topic:",
        "frame:",
        "planner_id:",
        "controller:",
        "navigator:",
        "cmd_vel:",
        "linear_x:",
        "angular_z:",
    ):
        assert forbidden not in source


def test_navigation_always_stops():
    start = HTML.index("async function sendMayday()")
    end = HTML.index(
        "function selectGoal",
        start,
    )
    source = HTML[start:end]

    finally_start = source.index("} finally {")
    finally_source = source[finally_start:]

    assert "NAVIGATION_STOP_ENDPOINT" in finally_source
    assert "navigationExecuting = false" in finally_source
    assert "planningRunning = false" in finally_source


def test_emergency_stop_remains_separate():
    assert "async function stopMayday()" in HTML
    assert (
        'emergencyStop.addEventListener(\n'
        '            "click",\n'
        '            stopMayday'
        in HTML
    )
    assert (
        "emergencyStop.disabled = (\n"
        "                !navigationExecuting"
        in HTML
    )
