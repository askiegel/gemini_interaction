import ast
import os
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "voice_relay"
    / "tony2_navigation_initial_pose.py"
)

SOURCE = SOURCE_PATH.read_text(
    encoding="utf-8"
)


def load_profile_function():
    tree = ast.parse(SOURCE)

    functions = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            == "is_stationary_promoted_map"
        )
    ]

    assert len(functions) == 1

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "os": os,
    }

    exec(
        compile(
            module,
            str(SOURCE_PATH),
            "exec",
        ),
        namespace,
    )

    return namespace[
        "is_stationary_promoted_map"
    ]


def test_active_promoted_map_is_sparse_profile(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "HOME",
        str(tmp_path),
    )

    active = (
        tmp_path
        / ".local"
        / "share"
        / "mayday"
        / "persistent_map"
        / "active"
        / "mayday_supervised_route_03.yaml"
    )

    monkeypatch.setenv(
        "MAYDAY_FIXED_MAP_YAML",
        str(active),
    )

    assert load_profile_function()() is True


def test_bundled_map_keeps_dense_profile(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "HOME",
        str(tmp_path),
    )

    bundled = (
        tmp_path
        / "robot_services"
        / "cognitive"
        / "voice_relay"
        / "tony2_navigation_assets"
        / "mayday_supervised_route_03.yaml"
    )

    monkeypatch.setenv(
        "MAYDAY_FIXED_MAP_YAML",
        str(bundled),
    )

    assert load_profile_function()() is False


def test_known_ratio_threshold_is_not_lowered():
    assert "MIN_KNOWN_RATIO" in SOURCE

    assert (
        "or known_ratio\n"
        "                        >= MIN_KNOWN_RATIO"
        in SOURCE
    )


def test_geometric_gates_remain_unconditional():
    required = (
        "mean_endpoint_error",
        "MAX_MEAN_ENDPOINT_ERROR_METERS",
        "within_10cm_ratio",
        "MIN_WITHIN_10CM_RATIO",
        "inside_ratio",
        "MIN_INSIDE_RATIO",
    )

    for marker in required:
        assert marker in SOURCE


def test_sparse_profile_is_reported_diagnostically():
    compact = " ".join(
        SOURCE.split()
    )

    assert (
        '"stationary_sparse_map": '
        'stationary_sparse_map'
        in compact
    )

    assert (
        '"known_ratio_required": '
        'not stationary_sparse_map'
        in compact
    )
