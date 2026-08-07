#!/usr/bin/env python3

"""Checks for the read-only localized LiDAR map overlay."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

HTML = (
    ROOT / 'voice_relay' / 'index.html'
).read_text(encoding='utf-8')

CSS = (
    ROOT / 'voice_relay' / 'operator_console.css'
).read_text(encoding='utf-8')

JS = (
    ROOT / 'voice_relay' / 'operator_console.js'
).read_text(encoding='utf-8')

START = JS.index(
    '/* Read-only localized LiDAR map overlay */'
)
END = JS.index(
    '/* Read-only saved occupancy-map visualization */'
)
OVERLAY = JS[START:END]


def test_overlay_canvas_is_unique():
    assert HTML.count(
        'id="localizedLidarCanvas"'
    ) == 1
    assert HTML.count(
        'id="localizedLidarOverlayStatus"'
    ) == 1


def test_overlay_is_transparent_and_noninteractive():
    assert '#localizedLidarCanvas {' in CSS
    assert 'pointer-events: none;' in CSS
    assert 'z-index: 2;' in CSS
    assert '#localizationPoseCanvas {' in CSS
    assert 'z-index: 3;' in CSS


def test_overlay_uses_existing_read_only_proxies():
    assert (
        '"/dashboard/map"'
        in OVERLAY
    )
    assert (
        '"/dashboard/lidar"'
        in OVERLAY
    )
    assert (
        '"/dashboard/localization"'
        in OVERLAY
    )
    assert 'method: "POST"' not in OVERLAY


def test_scan_is_transformed_into_map_frame():
    assert (
        'SCAN_TO_BASE_ROTATION_RADIANS'
        in OVERLAY
    )
    assert (
        'robotYaw'
        in OVERLAY
    )
    assert (
        'rawRange * Math.cos(mapBearing)'
        in OVERLAY
    )
    assert (
        'rawRange * Math.sin(mapBearing)'
        in OVERLAY
    )


def test_hardware_orientation_correction_is_retained():
    assert (
        'Math.PI / 2'
        in OVERLAY
    )
    assert (
        '+ SCAN_TO_BASE_ROTATION_RADIANS'
        in OVERLAY
    )


def test_overlay_requires_active_localization():
    assert (
        'localization.runtime_active !== true'
        in OVERLAY
    )
    assert (
        'localization.telemetry.available'
        in OVERLAY
    )
    assert (
        'localization.telemetry.pose'
        in OVERLAY
    )


def test_stale_scan_is_not_drawn():
    assert (
        'Number(lidar.telemetry.age_seconds)'
        in OVERLAY
    )
    assert (
        '> 1.0'
        in OVERLAY
    )
    assert '"Live scan is stale"' in OVERLAY


def test_overlay_does_not_modify_saved_map():
    forbidden = (
        'POST',
        'PUT',
        'PATCH',
        'DELETE',
        '/motion',
        '/stop',
        'cmd_vel',
        'navigation_goal',
        'cells[',
    )

    for marker in forbidden:
        assert marker not in OVERLAY


def test_legend_identifies_live_scan():
    assert (
        'Live localized LiDAR'
        in HTML
    )
    assert (
        '.map-legend-swatch.localized-lidar'
        in CSS
    )
