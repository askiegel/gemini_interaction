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


def test_box_verified_negative_x_baseline_is_forward_up():
    source = feature_source()

    assert (
        "const BASELINE_ROTATION_DEGREES ="
        in source
    )

    assert "90.0" in source

    assert (
        "FORWARD · -X BASELINE"
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




def test_candidate_preview_keeps_180_degree_correction():
    marker = (
        "/* MAYDAY_180_DEGREE_BASELINE_CORRECTION */"
    )

    assert marker in CSS

    correction = CSS.split(
        marker,
        1,
    )[1]

    assert (
        "#persistentMapCandidateCanvas"
        in correction
    )

    assert (
        "rotate(180deg)"
        in correction
    )

    # Review Large now owns the persistent-reference
    # orientation inside drawPersistentReference().
    assert (
        "#persistentMapReviewPersistentCanvas"
        not in correction
    )


def test_robot_local_candidate_live_overlay_is_not_double_rotated():
    marker = (
        "/* MAYDAY_180_DEGREE_BASELINE_CORRECTION */"
    )

    correction = CSS.split(
        marker,
        1,
    )[1]

    assert (
        "#persistentMapReviewOverlayCanvas"
        not in correction
    )

    assert (
        "#persistentMapReviewLiveCanvas"
        not in correction
    )


def test_localized_pose_still_controls_true_forward_up():
    source = feature_source()

    # Once AMCL provides a trusted/fresh map-frame yaw,
    # use the real robot heading rather than the temporary
    # unlocalized stationary-map baseline correction.
    assert (
        "rotation =\n"
        "                    yaw - 90.0;"
        in source
    )




def test_review_reference_uses_90_ccw_renderer_not_css():
    start = JS.index(
        "function drawPersistentReference("
    )

    end = JS.index(
        "\n    function ",
        start + 10,
    )

    renderer = JS[
        start:end
    ]

    assert (
        "MAYDAY_REVIEW_PERSISTENT_90_CCW"
        in renderer
    )

    assert (
        "-Math.PI / 2"
        in renderer
    )

    marker = (
        "/* MAYDAY_180_DEGREE_BASELINE_CORRECTION */"
    )

    assert marker in CSS

    correction = CSS.split(
        marker,
        1,
    )[1]

    assert (
        "#persistentMapReviewPersistentCanvas"
        not in correction
    )
