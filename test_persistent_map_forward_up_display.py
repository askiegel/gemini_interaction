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

CSS = (
    ROOT
    / "voice_relay"
    / "operator_console.css"
).read_text(
    encoding="utf-8"
)

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)

MARKER = (
    "/* MAYDAY_FORWARD_UP_PERSISTENT_MAP_DISPLAY */"
)


def feature_source():
    assert MARKER in JS

    return JS.split(
        MARKER,
        1,
    )[1]


def test_all_map_space_canvases_rotate_together():
    for canvas in (
        "#mapCanvas",
        "#localizedLidarCanvas",
        "#localizationPoseCanvas",
        "#planningPathCanvas",
    ):
        assert canvas in CSS

    assert (
        "--mayday-map-display-rotation"
        in CSS
    )

    assert (
        "rotate("
        in CSS
    )


def test_baseline_map_x_is_forward_up():
    source = feature_source()

    assert (
        "const BASELINE_ROTATION_DEGREES ="
        in source
    )

    assert "-90.0" in source

    assert (
        "FORWARD · +X BASELINE"
        in source
    )


def test_localized_display_tracks_mayday_heading():
    source = feature_source()

    assert (
        'const LOCALIZATION_ENDPOINT ='
        in source
    )

    assert (
        '"/dashboard/localization"'
        in source
    )

    assert (
        "rotation =\n                    yaw - 90.0;"
        in source
    )

    assert (
        "FORWARD · MAYDAY-UP"
        in source
    )


def test_orientation_is_presentation_only():
    source = feature_source()

    forbidden = (
        "/dashboard/map-promote",
        "/dashboard/persistent-map/promote",
        "/dashboard/navigation-start",
        "/dashboard/navigation-goal",
        "/dashboard/mapping-start",
        "/cmd_vel",
        "MotionArmLease",
        "NavigateToPose",
    )

    for value in forbidden:
        assert value not in source


def test_ros_map_coordinates_are_not_modified():
    source = feature_source()

    assert (
        "OccupancyGrid data"
        in source
    )

    assert (
        "map origin"
        in source
    )

    assert (
        "navigation goals"
        in source
    )

    assert (
        "presentation only"
        in source.lower()
    )


def test_goal_click_is_inverse_rotated():
    source = feature_source()

    assert (
        "function inverseRotatePoint("
        in source
    )

    assert (
        "CSS applied R(angle)."
        in source
    )

    assert (
        "R(-angle)"
        in source
    )

    assert (
        "stopImmediatePropagation"
        in source
    )

    assert (
        "new MouseEvent("
        in source
    )


def test_synthetic_click_is_not_remapped_twice():
    source = feature_source()

    assert (
        "if (!event.isTrusted)"
        in source
    )

    assert (
        "canvas.dispatchEvent("
        in source
    )


def test_map_stage_is_square_to_prevent_rotation_crop():
    assert (
        "aspect-ratio: 1 / 1"
        in CSS
    )

    assert (
        "overflow: hidden"
        in CSS
    )


def test_forward_label_is_screen_aligned():
    assert (
        "#persistentMapForwardLabel"
        in CSS
    )

    source = feature_source()

    assert (
        "persistentMapForwardLabel"
        in source
    )


def test_html_marks_forward_up_feature():
    assert (
        "MAYDAY_FORWARD_UP_PERSISTENT_MAP_DISPLAY"
        in HTML
    )
