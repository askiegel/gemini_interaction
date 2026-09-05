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


def test_startup_live_progress_stream():
    import ast

    server = SERVER.read_text(
        encoding="utf-8"
    )

    html = HTML.read_text(
        encoding="utf-8"
    )

    proof = PROOF.read_text(
        encoding="utf-8"
    )

    assert (
        '"/dashboard/startup-proof-stream"'
        in server
    )

    assert (
        "application/x-ndjson"
        in server
    )

    assert (
        "progress_callback"
        in proof
    )

    assert (
        "STARTUP_CHECK_PLAN"
        in proof
    )

    required_ui = (
        "PROOF_STREAM_URL",
        "WAITING",
        "CHECKING",
        "response.body.getReader()",
        "TextDecoder",
        "completeProgressCheck",
        "renderProgressPlan",
    )

    for value in required_ui:
        assert value in html

    tree = ast.parse(proof)

    plan = None

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        names = [
            target.id
            for target in node.targets
            if isinstance(
                target,
                ast.Name,
            )
        ]

        if (
            "STARTUP_CHECK_PLAN"
            in names
        ):
            plan = ast.literal_eval(
                node.value
            )
            break

    assert plan is not None
    assert len(plan) == 15

    assert len(
        {
            item["id"]
            for item in plan
        }
    ) == 15

    assert all(
        item["required"]
        for item in plan
    )


def test_prepare_requires_independent_live_proof():
    server = SERVER.read_text(
        encoding="utf-8"
    )

    start = server.index(
        'if path == '
        '"/dashboard/startup-prepare":'
    )

    end = server.index(
        'if path in (',
        start,
    )

    block = server[
        start:end
    ]

    assert "prepare_session" in block
    assert "prove_ready" not in block
    assert '"proof_required"' in block
