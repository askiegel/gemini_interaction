#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(
    __file__
).resolve().parent

JS = (
    ROOT
    / "voice_relay"
    / "operator_console.js"
).read_text(
    encoding="utf-8"
)


def section(name):
    start = JS.index(
        f"    function {name}("
    )

    end = JS.index(
        "\n    function ",
        start + 10,
    )

    return JS[start:end]


def test_robot_local_review_geometry_rotates_ccw_90():
    renderer = section(
        "localToCanvas"
    )

    assert (
        "MAYDAY_REVIEW_ALL_CCW90"
        in renderer
    )

    # New screen offset is (-x, +y), which is
    # visual CCW 90 from the previous (-y, -x).
    assert (
        "frame.center\n"
        "                - point.x"
        in renderer
    )

    assert (
        "frame.center\n"
        "                + point.y"
        in renderer
    )


def test_persistent_reference_gets_same_additional_ccw_turn():
    renderer = section(
        "drawPersistentReference"
    )

    assert (
        "MAYDAY_REVIEW_PERSISTENT_ALL_CCW90"
        in renderer
    )

    assert (
        "context.rotate("
        in renderer
    )

    assert (
        "-Math.PI"
        in renderer
    )


def test_candidate_overlay_uses_common_robot_local_mapping():
    renderer = section(
        "drawCandidateOverlay"
    )

    assert (
        "localToCanvas("
        in renderer
    )


def test_live_lidar_uses_common_robot_local_mapping():
    renderer = section(
        "drawLiveOnly"
    )

    assert (
        "localToCanvas("
        in renderer
    )


def test_forward_grid_and_arrow_are_not_rotated():
    setup = section(
        "setupRobotCanvas"
    )

    robot = section(
        "drawRobot"
    )

    assert (
        '"FORWARD"'
        in setup
    )

    # Do not rotate the entire canvas; only convert geometry
    # coordinates through localToCanvas().
    assert (
        "context.rotate("
        not in setup
    )

    assert (
        "context.rotate("
        not in robot
    )


def test_no_navigation_or_map_mutation():
    renderer = section(
        "localToCanvas"
    )

    for forbidden in (
        "/dashboard/navigation-goal",
        "/dashboard/persistent-map/promote",
        "/cmd_vel",
        "NavigateToPose",
        "MotionArmLease",
    ):
        assert forbidden not in renderer
