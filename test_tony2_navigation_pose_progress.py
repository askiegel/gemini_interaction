from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

PARAMS = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_assets"
    / "mayday_guarded_navigation.yaml"
)


def parameters():
    data = yaml.safe_load(
        PARAMS.read_text(
            encoding="utf-8"
        )
    )

    return data[
        "controller_server"
    ]["ros__parameters"]


def test_rotation_counts_as_progress():
    progress = parameters()[
        "progress_checker"
    ]

    assert (
        progress["plugin"]
        == "nav2_controller::PoseProgressChecker"
    )

    assert (
        progress["required_movement_radius"]
        == 0.03
    )

    assert (
        progress["required_movement_angle"]
        == 0.10
    )

    assert (
        progress["movement_time_allowance"]
        == 12.0
    )


def test_fix_does_not_relax_goal_tolerance():
    goal = parameters()[
        "guarded_goal_checker"
    ]

    assert goal["xy_goal_tolerance"] == 0.03
    assert goal["yaw_goal_tolerance"] == 0.35
