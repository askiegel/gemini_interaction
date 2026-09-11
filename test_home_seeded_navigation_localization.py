from pathlib import Path


ROOT = Path(__file__).resolve().parent

HELPER = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_initial_pose.py"
).read_text(encoding="utf-8")

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(encoding="utf-8")

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(encoding="utf-8")

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(encoding="utf-8")


def test_seed_and_global_amcl_modes_exist():
    assert "SetInitialPose" in HELPER
    assert '"/set_initial_pose"' in HELPER
    assert '"/reinitialize_global_localization"' in HELPER
    assert '"--seed-pose"' in HELPER


def test_home_pose_is_fixed():
    assert "def initialize_home_localization(self):" in RUNTIME
    assert "-0.4499999441206735," in RUNTIME
    assert "0.20000006556511413," in RUNTIME
    assert "-0.9198951421531195," in RUNTIME
    assert "seed_pose=True" in RUNTIME


def test_global_recovery_is_preserved():
    assert "def initialize_global_localization(self):" in RUNTIME
    assert "seed_pose=False" in RUNTIME



def test_normal_server_uses_home_localization():
    start = SERVER.index(
        "    def navigation_initialize_localization"
    )
    end = SERVER.index(
        "    def navigation_goal",
        start,
    )

    section = SERVER[start:end]

    assert (
        "runtime.initialize_home_localization()"
        in section
    )

    assert (
        "runtime.initialize_global_localization()"
        not in section
    )


def test_no_contradictory_dashboard_contract():
    start = HTML.index(
        "function navigationInitializationSucceeded(result)"
    )
    end = HTML.index(
        "async function",
        start,
    )

    compact = "".join(
        HTML[start:end].split()
    )

    required = (
        'initialization.localization_method==="amcl_seeded"',
        'initialization.search_scope==="known_home_pose"',
        "initialization.seed_pose_used===true",
        "initialization.global_localization_requested===false",
        "initialization.initial_pose_supplied===true",
        "Number(initialization.nomotion_updates_requested)===40",
    )

    for marker in required:
        assert marker in compact

    assert (
        'initialization.localization_method==="amcl_global"'
        not in compact
    )

    assert (
        'initialization.search_scope==="full_saved_map"'
        not in compact
    )

def test_trust_thresholds_unchanged():
    required = (
        "SCAN_CONFIRMATION_SAMPLES = 5",
        "SCAN_CONFIRMATION_REQUIRED_PASSES = 3",
        "MAX_MEAN_ENDPOINT_ERROR_METERS = 0.18",
        "MIN_WITHIN_10CM_RATIO = 0.35",
        "MIN_KNOWN_RATIO = 0.55",
        "MIN_INSIDE_RATIO = 0.75",
        "covariance_tight",
        "alignment_good",
    )

    for marker in required:
        assert marker in HELPER

def test_seed_flag_precedes_ros_argument_tail():
    start = RUNTIME.index(
        "    def initialize_operator_pose"
    )

    section = RUNTIME[start:]

    seed = section.index(
        '"--seed-pose"'
    )

    ros_args = section.index(
        '"--ros-args"'
    )

    assert seed < ros_args


def home_evidence():
    return {
        "ok": True, "trusted": True, "frame_id": "map",
        "localization_method": "amcl_seeded",
        "search_scope": "known_home_pose",
        "seed_pose_used": True, "initial_pose_supplied": True,
        "global_localization_requested": False,
        "nomotion_updates_requested": 40,
        "stationary_required": True, "navigation_goal_executed": False,
        "motion_enabled": False,
        "diagnostic": {
            "covariance_tight": True, "alignment_good": True,
            "trusted": True, "global_search_completed": False,
            "seed_pose_applied": True,
        },
        "final_pose": {"x": -0.4499999441206735, "y": 0.20000006556511413, "yaw_rad": -0.9198951421531195},
        "uncertainty": {"sigma_x_m": 0.01, "sigma_y_m": 0.01, "sigma_yaw_rad": 0.01},
    }


def contract_expression(source, name):
    # Evaluate only a pure acceptance expression, never import the server/proof.
    import ast
    tree = ast.parse(source)
    matches = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name
                       for target in node.targets)]
    assert len(matches) == 1
    return compile(ast.Expression(matches[0]), "localization_contract", "eval")


def test_normal_contracts_require_independent_home_evidence():
    import ast
    from copy import deepcopy
    proof = (ROOT / "voice_relay" / "startup_proof.py").read_text()
    start = SERVER.index("    def navigation_initialize_localization")
    end = SERVER.index("    def navigation_compute_path", start)
    import textwrap
    handlers = [node for node in ast.walk(ast.parse(SERVER))
                if isinstance(node, ast.FunctionDef) and node.name == "do_POST"]
    assert len(handlers) == 1
    startup_loops = [node for node in ast.walk(handlers[0])
                     if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name)
                     and node.target.id == "localization_attempt"]
    assert len(startup_loops) == 1
    # The assignment before this loop initializes failure to False.
    # Only the retry-loop assignment evaluates Home acceptance evidence.
    startup_contract = ast.unparse(startup_loops[0])
    expressions = [
        contract_expression(textwrap.dedent(SERVER[start:end]), "trusted"),
        contract_expression(startup_contract, "localization_valid"),
        contract_expression(proof, "localization_result_ok"),
    ]

    def accepts(expression, localization):
        result = {"action": "OPERATOR_POSE_VALIDATED", "localization": localization}
        return eval(expression, {}, {
            "result": result, "evidence": result, "localization_result": result,
            "localization": localization, "diagnostic": localization["diagnostic"],
            "final_pose": localization["final_pose"],
            "uncertainty": localization["uncertainty"],
        })

    for expression in expressions:
        assert accepts(expression, home_evidence())
        for field, value in {
            "trusted": False, "nomotion_updates_requested": 20,
            "localization_method": "amcl_global", "search_scope": "full_saved_map",
            "seed_pose_used": False, "initial_pose_supplied": False,
            "global_localization_requested": True, "stationary_required": False,
            "navigation_goal_executed": True, "motion_enabled": True,
        }.items():
            invalid = deepcopy(home_evidence())
            invalid[field] = value
            assert not accepts(expression, invalid), field
        for field, value in {
            "covariance_tight": False, "alignment_good": False, "trusted": False,
            "global_search_completed": True, "seed_pose_applied": False,
        }.items():
            invalid = deepcopy(home_evidence())
            invalid["diagnostic"][field] = value
            assert not accepts(expression, invalid), field


def test_home_wrapper_validation_controls_navigation_gate(tmp_path):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from voice_relay.tony2_navigation_runtime import Tony2NavigationRuntime

    for trusted in (True, False):
        runtime = Tony2NavigationRuntime(runtime_dir=tmp_path)
        preinit = {
            "running": True, "state": "STARTING", "map_server_enabled": True,
            "localization_enabled": True, "transform_ready": False,
            "goal_submission_enabled": False, "goal_active": False,
            "motion_output_connected": False, "motion_egress_ready": True,
            "motion_egress_idle": True,
        }
        ready = {**preinit, "state": "READY", "transform_ready": True}
        payload = home_evidence()
        payload["trusted"] = trusted
        with patch.object(runtime, "status", side_effect=[preinit, ready]), patch(
            "voice_relay.tony2_navigation_runtime.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
        ) as run, patch.object(runtime, "stop", return_value={"navigation": {}}) as stop, patch.object(
            runtime, "initialize_global_localization"
        ) as global_init:
            result = runtime.initialize_home_localization()
        command = run.call_args.args[0]
        assert "--x=-0.4499999441206735" in command
        assert "--y=0.20000006556511413" in command
        assert "--yaw=-0.9198951421531195" in command
        assert "--seed-pose" in command
        assert runtime._localization_validated is trusted
        assert result["action"] == (
            "OPERATOR_POSE_VALIDATED" if trusted else "OPERATOR_POSE_REJECTED"
        )
        assert stop.call_count == (0 if trusted else 1)
        global_init.assert_not_called()
        with patch.object(runtime, "_runtime_pids", return_value={"supervisor": 1, "probe": 2, "goal": None}), patch.object(
            runtime, "_read_snapshot", return_value={
                "map_server_enabled": True, "localization_enabled": True,
                "planner_enabled": True, "controller_enabled": True,
                "navigator_enabled": True, "action_server_ready": True,
                "transform_ready": True,
            }
        ), patch.object(runtime, "mapping_status", return_value={
            "running": False, "cartographer": None, "occupancy_grid": None,
        }), patch.object(
            runtime, "_read_motion_egress_status", return_value={"running": True, "armed": False, "token": None}
        ):
            assert runtime.status()["goal_submission_enabled"] is trusted


def test_global_wrapper_remains_unseeded():
    from unittest.mock import patch
    from voice_relay.tony2_navigation_runtime import Tony2NavigationRuntime
    runtime = Tony2NavigationRuntime()
    with patch.object(runtime, "initialize_operator_pose") as initialize:
        runtime.initialize_global_localization()
    initialize.assert_called_once_with(0.0, 0.0, 0.0, seed_pose=False)
