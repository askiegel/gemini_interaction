import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ASSET = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_assets"
    / "mayday_guarded_navigation.yaml"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
)


def runtime_asset_hash():
    source = RUNTIME.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            continue

        if isinstance(node, ast.Assign):
            named = any(
                isinstance(target, ast.Name)
                and target.id == "ASSET_HASHES"
                for target in node.targets
            )
        else:
            named = (
                isinstance(node.target, ast.Name)
                and node.target.id == "ASSET_HASHES"
            )

        if not named:
            continue

        value = node.value

        assert isinstance(
            value,
            ast.Dict,
        )

        for key, item in zip(
            value.keys,
            value.values,
        ):
            if (
                isinstance(key, ast.Constant)
                and key.value
                == "mayday_guarded_navigation.yaml"
            ):
                return ast.literal_eval(
                    item
                )

    raise AssertionError(
        "guarded navigation asset hash missing"
    )


def test_guarded_navigation_asset_hash_matches():
    actual = hashlib.sha256(
        ASSET.read_bytes()
    ).hexdigest()

    assert runtime_asset_hash() == actual


def test_goal_acceptance_configuration():
    config = ASSET.read_text(
        encoding="utf-8"
    )

    assert "xy_goal_tolerance: 0.03" in config
    assert "yaw_goal_tolerance: 0.35" in config
    assert "movement_time_allowance: 12.0" in config
    assert "required_movement_radius: 0.03" in config
