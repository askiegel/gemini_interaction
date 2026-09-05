from pathlib import Path


ROOT = Path(__file__).resolve().parent

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
)

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
)

PROOF = (
    ROOT
    / "voice_relay"
    / "startup_proof.py"
)


def test_startup_dashboard_routes_exist():
    server = SERVER.read_text(
        encoding="utf-8"
    )

    assert (
        '"/dashboard/startup-proof"'
        in server
    )

    assert (
        '"/dashboard/startup-prepare"'
        in server
    )

    assert "prove_ready" in server
    assert "prepare_session" in server


def test_startup_tab_has_decisive_controls():
    html = HTML.read_text(
        encoding="utf-8"
    )

    required = (
        "MAYDAY_STARTUP_TAB_BEGIN",
        "Mayday Startup Proof",
        "Prepare Session",
        "Prove Ready",
        "Copy Proof Log",
        "READY FOR NAVIGATION",
        "NOT READY",
        "Startup Proof Log",
        "/dashboard/startup-proof",
        "/dashboard/startup-prepare",
    )

    for value in required:
        assert value in html


def test_prepare_does_not_start_navigation():
    source = PROOF.read_text(
        encoding="utf-8"
    )

    forbidden = (
        '"/navigation/start"',
        '"/planning/start"',
        "navigation/goal",
        "NavigateToPose",
        "FollowPath",
        "create_publisher",
        "geometry_msgs.msg",
        "Twist(",
        "ros2 topic pub",
        "publish(cmd",
        "publish(command",
    )

    for value in forbidden:
        assert value not in source


def test_proof_requires_real_locomotion_chain():
    source = PROOF.read_text(
        encoding="utf-8"
    )

    required = (
        "quadruped_controller_node",
        "servo_interface",
        "/joint_group_effort_controller/",
        "/cmd_vel",
        "/scan",
        "motion_zero",
        "navigation_clean",
        "guarded_asset",
        "gait_threshold",
    )

    for value in required:
        assert value in source


def test_robot_bridge_boot_service_is_required():
    import ast

    source = PROOF.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    label = "Robot Bridge boot service"

    matches = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        constants = [
            child.value
            for child in ast.walk(node)
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            )
        ]

        if label not in constants:
            continue

        required = [
            keyword
            for keyword in node.keywords
            if keyword.arg == "required"
        ]

        matches.append(required)

    assert len(matches) == 1
    assert len(matches[0]) == 1

    value = matches[0][0].value

    assert isinstance(value, ast.Constant)
    assert value.value is True
