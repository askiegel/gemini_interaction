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
    assert len(plan) == 19

    assert len(
        {
            item["id"]
            for item in plan
        }
    ) == len(plan)

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



def test_startup_uses_global_current_pose_localization():
    from pathlib import Path

    server = Path(
        "voice_relay/server.py"
    ).read_text(
        encoding="utf-8"
    )

    start = server.index(
        'if path == "/dashboard/startup-prepare":'
    )

    end = server.index(
        "\n            return",
        start,
    )

    block = server[
        start:
        end
    ]

    assert "runtime.start()" in block
    assert (
        "initialize_global_localization"
        in block
    )
    assert (
        "initialize_home_localization"
        not in block
    )
    assert (
        '"global_current_pose"'
        in block
    )
    assert (
        '"home_seed_used"'
        in block
    )
    assert (
        "send_goal"
        not in block
    )


def test_startup_proves_complete_boot_platform():
    from pathlib import Path

    proof = Path(
        "voice_relay/startup_proof.py"
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "tony2_cognitive_boot",
        "camera_vision_pipeline",
        "camera_relay_boot_service",
        "Persistent Nav2 / AMCL ready",
        "current_pose_localization",
        "Current physical pose localized",
        "mayday-vision-server.service",
        "mayday-vision-service.service",
        "mayday-cognitive-runtime.service",
        "mayday-voice-relay.service",
        "mayday-camera-relay.service",
    )

    for value in required:
        assert value in proof


def test_startup_global_localization_never_calls_home_initializer():
    from pathlib import Path

    server = Path(
        "voice_relay/server.py"
    ).read_text(
        encoding="utf-8"
    )

    start = server.index(
        'if path == "/dashboard/startup-prepare":'
    )

    end = server.index(
        "\n            return",
        start,
    )

    block = server[
        start:
        end
    ]

    assert (
        "initialize_home_localization"
        not in block
    )

    assert (
        "initialize_global_localization"
        in block
    )



def test_startup_requires_trusted_global_localization():
    from pathlib import Path

    server = Path(
        "voice_relay/server.py"
    ).read_text(
        encoding="utf-8"
    )

    start = server.index(
        'if path == "/dashboard/startup-prepare":'
    )

    end = server.index(
        "\n            return",
        start,
    )

    block = server[
        start:
        end
    ]

    required = (
        "prelocalization_ready",
        '"transform_ready"',
        "initialize_global_localization",
        '"OPERATOR_POSE_VALIDATED"',
        '"trusted"',
        '"covariance_tight"',
        '"alignment_good"',
        '"global_search_completed"',
        '"full_saved_map"',
        '"amcl_global"',
        '"seed_pose_used"',
        '"initial_pose_supplied"',
        '"navigation_goal_executed"',
        '"motion_enabled"',
        "set_startup_localization_evidence",
    )

    for value in required:
        assert value in block

    assert (
        "initialize_home_localization"
        not in block
    )

    assert (
        "send_goal"
        not in block
    )

    # The pre-localization phase must require NO transform.
    pre = block[
        block.index(
            "def prelocalization_ready"
        ):
        block.index(
            "def navigation_ready"
        )
    ]

    assert (
        '"transform_ready"'
        in pre
    )

    assert (
        "is False"
        in pre
    )


def test_startup_proof_uses_validated_localization_attestation():
    from pathlib import Path

    proof = Path(
        "voice_relay/startup_proof.py"
    ).read_text(
        encoding="utf-8"
    )

    required = (
        "_STARTUP_LOCALIZATION_EVIDENCE",
        "clear_startup_localization_evidence",
        "set_startup_localization_evidence",
        "get_startup_localization_evidence",
        '"OPERATOR_POSE_VALIDATED"',
        '"trusted"',
        '"covariance_tight"',
        '"alignment_good"',
        '"global_search_completed"',
        '"full_saved_map"',
        '"amcl_global"',
        '"seed_pose_used"',
        '"initial_pose_supplied"',
        '"navigation_goal_executed"',
        '"motion_enabled"',
    )

    for value in required:
        assert value in proof



def test_prelocalization_does_not_require_navigation_servers_active():
    from pathlib import Path

    server = Path(
        "voice_relay/server.py"
    ).read_text(
        encoding="utf-8"
    )

    prepare_start = server.index(
        'if path == "/dashboard/startup-prepare":'
    )

    prepare_end = server.index(
        "\n            return",
        prepare_start,
    )

    block = server[
        prepare_start:
        prepare_end
    ]

    pre_start = block.index(
        "def prelocalization_ready"
    )

    post_start = block.index(
        "def navigation_ready"
    )

    pre = block[
        pre_start:
        post_start
    ]

    post = block[
        post_start:
    ]

    # Before AMCL establishes map->odom, these lifecycle
    # nodes are allowed to remain inactive.
    for value in (
        '"planner_enabled"',
        '"controller_enabled"',
        '"navigator_enabled"',
    ):
        assert value not in pre

    # After localization they are mandatory.
    for value in (
        '"planner_enabled"',
        '"controller_enabled"',
        '"navigator_enabled"',
    ):
        assert value in post

    # The actual localization prerequisites remain enforced.
    for value in (
        '"map_server_enabled"',
        '"localization_enabled"',
        '"transform_ready"',
        '"goal_submission_enabled"',
        '"goal_active"',
        '"motion_output_connected"',
        '"motion_egress_ready"',
        '"motion_egress_idle"',
    ):
        assert value in pre

    assert "initialize_global_localization" in block
    assert "initialize_home_localization" not in block



def test_startup_uses_guarded_runtime_localization_attestation():
    from pathlib import Path

    server = Path(
        "voice_relay/server.py"
    ).read_text(
        encoding="utf-8"
    )

    proof = Path(
        "voice_relay/startup_proof.py"
    ).read_text(
        encoding="utf-8"
    )

    prepare_start = server.index(
        'if path == "/dashboard/startup-prepare":'
    )

    prepare_end = server.index(
        "\n            return",
        prepare_start,
    )

    prepare = server[
        prepare_start:
        prepare_end
    ]

    assert (
        "initialize_global_localization"
        in prepare
    )

    assert (
        "initialize_home_localization"
        not in prepare
    )

    assert (
        '"OPERATOR_POSE_VALIDATED"'
        in prepare
    )

    assert (
        '"trusted"'
        in prepare
    )

    assert (
        '"full_saved_map"'
        in prepare
    )

    assert (
        '"localization_attestation"'
        in prepare
    )

    # Robot Bridge cannot authoritatively observe the
    # isolated Tony2 AMCL runtime.
    assert (
        '"/telemetry/localization"'
        not in prepare
    )

    assert (
        '"/amcl_pose"'
        not in prepare
    )

    current_check = proof[
        proof.index(
            '"current_pose_localization"'
        ) - 9000:
        proof.index(
            '"current_pose_localization"'
        ) + 3000
    ]

    assert (
        "get_startup_localization_evidence"
        in current_check
    )

    assert (
        '"OPERATOR_POSE_VALIDATED"'
        in current_check
    )

    assert (
        '"trusted"'
        in current_check
    )

    assert (
        '"covariance_tight"'
        in current_check
    )

    assert (
        '"alignment_good"'
        in current_check
    )

    assert (
        '"global_search_completed"'
        in current_check
    )

    assert (
        '"/telemetry/localization"'
        not in current_check
    )

    assert (
        '"/amcl_pose"'
        not in current_check
    )



def test_hardware_node_proof_retries_transient_ros_discovery():
    from pathlib import Path

    proof = Path(
        "voice_relay/startup_proof.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "NODE_DISCOVERY_MAX_SAMPLES = 3"
        in proof
    )

    assert (
        "NODE_DISCOVERY_RETRY_SECONDS = 0.75"
        in proof
    )

    assert (
        "def _startup_hardware_nodes_with_retry("
        in proof
    )

    assert (
        "ros2 node list --no-daemon"
        in proof
    )

    assert (
        "FASTDDS_BUILTIN_TRANSPORTS=UDPv4"
        in proof
    )

    assert (
        "_startup_hardware_nodes_with_retry("
        in proof
    )

    expected_start = proof.index(
        "    expected_nodes = ("
    )

    hardware_end = proof.index(
        '        "bridge_process",',
        expected_start,
    )

    hardware = proof[
        expected_start:
        hardware_end
    ]

    # /servo_interface remains REQUIRED.
    assert (
        '"/servo_interface"'
        in hardware
    )

    # All original required nodes remain present.
    for node in (
        "/quadruped_controller_node",
        "/servo_interface",
        "/LD06",
        "/robot_state_publisher",
        "/state_estimation_node",
        "/base_to_footprint_ekf",
        "/footprint_to_odom_ekf",
    ):
        assert node in hardware

    # The check must still fail when a required node
    # is absent after all bounded samples.
    assert (
        "missing_nodes"
        in hardware
    )

    assert (
        "not missing_nodes"
        in hardware
    )

    # The implementation explicitly replaces each sample
    # instead of unioning partial graph observations.
    helper_start = proof.index(
        "def _startup_hardware_nodes_with_retry("
    )

    helper_end = proof.index(
        "def prove_ready(",
        helper_start,
    )

    helper = proof[
        helper_start:
        helper_end
    ]

    assert (
        "current_nodes = tuple("
        in helper
    )

    assert (
        "current_nodes.update"
        not in helper
    )

    assert (
        "|="
        not in helper
    )

def test_cmd_vel_proof_retries_transient_ros_discovery():
    from pathlib import Path

    proof = Path(
        "voice_relay/startup_proof.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "def _startup_cmd_vel_sample_complete("
        in proof
    )

    assert (
        "def _startup_cmd_vel_with_retry("
        in proof
    )

    helper_start = proof.index(
        "def _startup_cmd_vel_with_retry("
    )

    helper_end = proof.index(
        "def prove_ready(",
        helper_start,
    )

    helper = proof[
        helper_start:
        helper_end
    ]

    assert (
        "NODE_DISCOVERY_MAX_SAMPLES + 1"
        in helper
    )

    assert (
        "NODE_DISCOVERY_RETRY_SECONDS"
        in helper
    )

    assert (
        "ros2 topic info"
        in helper
    )

    assert (
        "/cmd_vel"
        in helper
    )

    assert (
        "--verbose"
        in helper
    )

    # Every successful retry replaces the entire
    # topic-info sample. Partial DDS observations
    # must never be accumulated across retries.
    assert (
        "current_cmdvel = ("
        in helper
    )

    assert (
        "completed.stdout.strip()"
        in helper
    )

    assert ".union(" not in helper
    assert "|=" not in helper

    cmdvel_start = proof.index(
        "        cmdvel,"
    )

    cmdvel_end = proof.index(
        "    quadruped = ros[",
        cmdvel_start,
    )

    cmdvel_check = proof[
        cmdvel_start:
        cmdvel_end
    ]

    assert (
        "_startup_cmd_vel_with_retry("
        in cmdvel_check
    )

    assert (
        "_startup_cmd_vel_sample_complete("
        in cmdvel_check
    )

    assert (
        "cmdvel_discovery_attempts"
        in cmdvel_check
    )

    assert (
        "discovery sample(s)"
        in cmdvel_check
    )
