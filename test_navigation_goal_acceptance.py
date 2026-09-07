from pathlib import Path


CONFIG = (
    Path(__file__).resolve().parent
    / "voice_relay"
    / "tony2_navigation_assets"
    / "mayday_guarded_navigation.yaml"
).read_text(
    encoding="utf-8"
)


def test_xy_goal_tolerance_remains_precise():
    assert (
        "xy_goal_tolerance: 0.03"
        in CONFIG
    )


def test_final_yaw_tolerance_accepts_quadruped_heading_noise():
    assert (
        "yaw_goal_tolerance: 0.35"
        in CONFIG
    )


def test_old_overly_strict_yaw_tolerance_removed():
    assert (
        "yaw_goal_tolerance: 0.15"
        not in CONFIG
    )


def test_progress_checker_remains_guarded():
    assert (
        "required_movement_radius: 0.03"
        in CONFIG
    )

    assert (
        "movement_time_allowance: 12.0"
        in CONFIG
    )
