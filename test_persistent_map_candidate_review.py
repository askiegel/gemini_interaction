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


def review_source():
    marker = (
        "/* MAYDAY_PERSISTENT_MAP_CANDIDATE_REVIEW */"
    )

    assert marker in JS

    return JS.split(
        marker,
        1,
    )[1]


def test_review_uses_same_robot_centered_scale():
    source = review_source()

    assert (
        "const REVIEW_RADIUS_METERS = 4.0;"
        in source
    )

    def function_section(name):
        token = (
            f"function {name}("
        )

        start = source.index(
            token
        )

        end = source.find(
            "\n    function ",
            start + len(token),
        )

        if end < 0:
            end = len(source)

        return source[
            start:end
        ]

    setup = function_section(
        "setupRobotCanvas"
    )

    mapping = function_section(
        "localToCanvas"
    )

    candidate = function_section(
        "drawCandidateOverlay"
    )

    live = function_section(
        "drawLiveOnly"
    )

    assert (
        "REVIEW_RADIUS_METERS"
        in setup
    )

    assert (
        "MAYDAY_REVIEW_ALL_CCW90"
        in mapping
    )

    assert (
        "setupRobotCanvas("
        in candidate
    )

    assert (
        "localToCanvas("
        in candidate
    )

    assert (
        "setupRobotCanvas("
        in live
    )

    assert (
        "localToCanvas("
        in live
    )

def test_current_persistent_map_is_not_falsely_registered():
    source = review_source()

    assert (
        "Reference only"
        in source
    )

    assert (
        "not currently"
        in source
    )

    assert (
        "not registered"
        in source
        or "not currently"
        in source
    )


def test_review_overlays_live_lidar_on_candidate():
    source = review_source()

    assert (
        "function drawCandidateOverlay("
        in source
    )

    assert (
        "Cyan = candidate occupied geometry."
        in source
    )

    assert (
        "Orange = current Live LiDAR."
        in source
    )

    assert (
        "candidateOccupiedPoints("
        in source
    )


def test_structural_comparison_is_conservative():
    source = review_source()

    required = (
        "const MATCH_DISTANCE_METERS = 0.15;",
        "const MIN_MATCH_RATIO = 0.55;",
        "const MAX_MEAN_ERROR_METERS = 0.15;",
        "const MIN_INSIDE_RATIO = 0.80;",
        "const MIN_LIVE_RETURNS = 80;",
        "const MIN_CAPTURE_SCANS = 100;",
        "const MIN_STABLE_BINS = 100;",
    )

    for marker in required:
        assert marker in source


def test_promotion_requires_three_of_five_checks():
    source = review_source()

    assert (
        "const REVIEW_WINDOW = 5;"
        in source
    )

    assert (
        "const REQUIRED_PASSES = 3;"
        in source
    )

    assert (
        "comparisonHistory.length"
        in source
    )

    assert (
        "passCount"
        in source
    )


def test_use_new_map_is_disabled_until_review_passes():
    source = review_source()

    assert (
        "promote.disabled ="
        in source
    )

    assert (
        "!promotionAllowed"
        in source
    )

    assert (
        "stopImmediatePropagation"
        in source
    )

    assert (
        "Use New Map is disabled until"
        in source
    )


def test_large_review_contains_three_views():
    source = review_source()

    required = (
        "persistentMapReviewPersistentCanvas",
        "persistentMapReviewOverlayCanvas",
        "persistentMapReviewLiveCanvas",
        "Current Persistent Map",
        "Candidate + Live Overlay",
        "Live LiDAR",
        "Review Large",
    )

    for marker in required:
        assert marker in source


def test_refresh_controls_are_remounted_above_map_stage():
    source = review_source()

    assert (
        "function remountRefreshControls("
        in source
    )

    assert (
        'byId(\n                "mapCanvas"'
        in source
    )

    assert (
        'mapCanvas.closest(\n                ".map-stage"'
        in source
    )

    assert (
        "insertBefore("
        in source
    )


def test_review_is_read_only_except_existing_promotion_controls():
    source = review_source()

    forbidden = (
        "/dashboard/navigation-goal",
        "/cmd_vel",
        "MotionArmLease",
        "NavigateToPose",
    )

    for marker in forbidden:
        assert marker not in source


def test_large_review_css_exists():
    assert (
        "MAYDAY_PERSISTENT_MAP_CANDIDATE_REVIEW"
        in CSS
    )

    for marker in (
        "#persistentMapReviewModal",
        ".persistent-map-review-grid",
        ".persistent-map-review-canvas",
        "#persistentMapStructuralStatus",
    ):
        assert marker in CSS



def test_persistent_reference_is_rotated_90_ccw_in_renderer():
    source = review_source()

    start = source.index(
        "function drawPersistentReference("
    )

    end = source.index(
        "\n    function ",
        start + 10,
    )

    renderer = source[
        start:end
    ]

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

    assert (
        "context.save();"
        in renderer
    )

    assert (
        "context.restore();"
        in renderer
    )


def test_candidate_live_overlay_has_no_review_rotation():
    source = review_source()

    start = source.index(
        "function drawCandidateOverlay("
    )

    end = source.index(
        "\n    function ",
        start + 10,
    )

    renderer = source[
        start:end
    ]

    assert (
        "MAYDAY_REVIEW_PERSISTENT_ALL_CCW90"
        not in renderer
    )

    assert (
        "-Math.PI"
        not in renderer
    )
