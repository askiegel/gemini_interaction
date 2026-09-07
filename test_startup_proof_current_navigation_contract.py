import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PROOF_PATH = (
    ROOT
    / "voice_relay"
    / "startup_proof.py"
)

RUNTIME_PATH = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
)

PROOF = PROOF_PATH.read_text(
    encoding="utf-8"
)

RUNTIME = RUNTIME_PATH.read_text(
    encoding="utf-8"
)


def module_constant(
    source,
    name,
):
    tree = ast.parse(source)

    values = []

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            values.append(
                ast.literal_eval(node.value)
            )

    assert len(values) == 1

    return values[0]


def runtime_constant(
    name,
):
    tree = ast.parse(RUNTIME)

    classes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name
            == "Tony2NavigationRuntime"
        )
    ]

    assert len(classes) == 1

    values = []

    for node in classes[0].body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            values.append(
                ast.literal_eval(node.value)
            )

    assert len(values) == 1

    return values[0]


def localization_contract_source():
    start = PROOF.index(
        "localization_result_ok = ("
    )

    end = PROOF.index(
        "numeric_values = (",
        start,
    )

    return PROOF[start:end]


def test_startup_envelope_matches_navigation_runtime():
    assert (
        module_constant(
            PROOF,
            "EXPECTED_GOAL_DISTANCE",
        )
        == runtime_constant(
            "MAXIMUM_GOAL_DISTANCE_METERS"
        )
    )

    assert (
        module_constant(
            PROOF,
            "EXPECTED_TIMEOUT",
        )
        == runtime_constant(
            "EXECUTION_TIMEOUT_SECONDS"
        )
    )


def test_localization_attestation_does_not_depend_on_transient_pids():
    source = localization_contract_source()

    assert "evidence_pids" not in source
    assert "current_pids" not in source


def test_global_localization_safety_contract_remains_required():
    source = localization_contract_source()

    required = (
        '== "OPERATOR_POSE_VALIDATED"',
        '"trusted"',
        '== "map"',
        '"localization_method"',
        '== "amcl_global"',
        '"search_scope"',
        '== "full_saved_map"',
        '"seed_pose_used"',
        '"global_localization_requested"',
        '"initial_pose_supplied"',
        '"stationary_required"',
        '"navigation_goal_executed"',
        '"motion_enabled"',
    )

    for marker in required:
        assert marker in source


def test_startup_source_check_is_still_strict():
    # We intentionally preserve the source-cleanliness proof.
    # Development waypoint files were moved outside the repo
    # instead of weakening this requirement.
    assert (
        'and not local[\n'
        '                    "dirty"\n'
        '                ]'
        in PROOF
    )


def test_normal_navigation_localization_records_startup_evidence():
    server = (
        ROOT
        / "voice_relay"
        / "server.py"
    ).read_text(
        encoding="utf-8"
    )

    start = server.index(
        "    def navigation_initialize_localization(self):"
    )

    end = server.index(
        "\n    def ",
        start + 10,
    )

    method = server[start:end]

    evidence = method.index(
        "set_startup_localization_evidence("
    )

    success = method.index(
        '        return 200, {\n'
        '            "ok": True,'
    )

    assert evidence < success

    assert (
        "set_startup_localization_evidence(\n"
        "                result\n"
        "            )"
        in method
    )


def test_hardware_node_discovery_has_bounded_convergence_window():
    assert (
        module_constant(
            PROOF,
            "NODE_DISCOVERY_MAX_SAMPLES",
        )
        == 8
    )

    assert (
        module_constant(
            PROOF,
            "NODE_DISCOVERY_RETRY_SECONDS",
        )
        == 0.75
    )


def test_missing_localization_evidence_does_not_dereference_none():
    start = PROOF.index(
        "final_pose = ("
    )

    end = PROOF.index(
        "localization_result_ok = (",
        start,
    )

    section = PROOF[start:end]

    assert (
        'localization.get("final_pose")'
        in section
    )

    assert (
        'localization.get("uncertainty")'
        in section
    )

    assert section.count(
        "else {}"
    ) >= 2
