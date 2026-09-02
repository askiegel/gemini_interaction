#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Tony Kiegel
#

from pathlib import Path


ROOT = Path(__file__).resolve().parent

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(encoding="utf-8")

CSS = (
    ROOT
    / "voice_relay"
    / "operator_console.css"
).read_text(encoding="utf-8")

JS = (
    ROOT
    / "voice_relay"
    / "operator_console.js"
).read_text(encoding="utf-8")

MARKER = "/* Dual environment comparison */"

FEATURE_JS = JS.split(
    MARKER,
    1,
)[1]


def test_dual_environment_reuses_existing_canvases():
    assert 'id="mapCanvas"' in HTML
    assert 'id="lidarCanvas"' in HTML
    assert 'id="localizationPoseCanvas"' in HTML

    assert (
        'byId(\n'
        '            "mapCanvas"\n'
        '        )'
        in FEATURE_JS
    )

    assert (
        'byId(\n'
        '            "lidarCanvas"\n'
        '        )'
        in FEATURE_JS
    )

    assert "workspace.appendChild(mapCard);" in FEATURE_JS
    assert "workspace.appendChild(lidarCard);" in FEATURE_JS

    # This feature rearranges the established canvases.
    # It must not create replacement canvases.
    assert 'createElement("canvas")' not in FEATURE_JS


def test_persistent_map_is_displayed_left_of_live_lidar():
    persistent = FEATURE_JS.index(
        "workspace.appendChild(mapCard);"
    )

    live = FEATURE_JS.index(
        "workspace.appendChild(lidarCard);"
    )

    assert persistent < live

    assert (
        '"Stationary / Persistent Map"'
        in FEATURE_JS
    )

    assert (
        '"Live LiDAR"'
        in FEATURE_JS
    )


def test_same_map_pose_updates_both_panels():
    assert (
        'const LOCALIZATION_ENDPOINT = '
        '"/dashboard/localization";'
        in FEATURE_JS
    )

    assert "function setBothPoseReadouts(" in FEATURE_JS
    assert '"persistentMaydayPose"' in FEATURE_JS
    assert '"liveMaydayPose"' in FEATURE_JS

    assert (
        "setBothPoseReadouts(\n"
        "                summary,\n"
        "                summary + "
        '" · LiDAR centered on robot",'
        in FEATURE_JS
    )


def test_live_lidar_preserves_robot_centered_semantics():
    assert (
        '"Mayday: local LiDAR origin · forward up"'
        in FEATURE_JS
    )

    assert (
        '"Raw LD06 geometry centered on Mayday"'
        in FEATURE_JS
    )

    # The existing renderer already draws the Mayday marker.
    assert (
        "drawRobot(context, centerX, centerY);"
        in JS
    )


def test_saved_map_keeps_existing_pose_overlay():
    assert 'id="localizedLidarCanvas"' in HTML
    assert 'id="localizationPoseCanvas"' in HTML
    assert 'id="planningPathCanvas"' in HTML

    assert (
        "dual-environment-persistent"
        in FEATURE_JS
    )


def test_dual_environment_layout_is_responsive():
    assert ".dual-environment-workspace {" in CSS

    assert (
        "grid-template-columns:\n"
        "        minmax(0, 1fr)\n"
        "        minmax(0, 1fr);"
        in CSS
    )

    assert "@media (max-width: 1200px)" in CSS

    assert (
        ".dual-environment-workspace "
        ".map-stage"
        in CSS
    )

    assert (
        ".dual-environment-workspace "
        ".lidar-stage"
        in CSS
    )


def test_feature_is_read_only():
    forbidden = (
        'method: "POST"',
        'method:"POST"',
        "/dashboard/navigation-goal",
        "/dashboard/mapping-navigation-goal",
        "/motion",
        "/stop",
        "cmd_vel",
        "NavigateToPose",
    )

    for value in forbidden:
        assert value not in FEATURE_JS


def test_change_detection_is_not_faked():
    assert (
        "Visualization only — change detector next"
        in FEATURE_JS
    )

    assert "structural_change_detected" not in FEATURE_JS
    assert "map_agreement_score" not in FEATURE_JS
