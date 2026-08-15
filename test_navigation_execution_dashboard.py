from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = (
    ROOT / "voice_relay/server.py"
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
        SERVER.index("    def localization_status(")
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
