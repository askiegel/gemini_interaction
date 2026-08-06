#!/usr/bin/env python3
"""Offline tests for the read-only localization map overlay."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = (
    ROOT / 'voice_relay' / 'server.py'
).read_text(encoding='utf-8')
HTML = (
    ROOT / 'voice_relay' / 'index.html'
).read_text(encoding='utf-8')
CSS = (
    ROOT / 'voice_relay' / 'operator_console.css'
).read_text(encoding='utf-8')
JAVASCRIPT = (
    ROOT / 'voice_relay' / 'operator_console.js'
).read_text(encoding='utf-8')

MARKER = '/* Read-only localization pose overlay */'
OVERLAY_JAVASCRIPT = JAVASCRIPT.split(MARKER, 1)[1]


def test_localization_proxy_is_get_only():
    assert 'def localization_status(self):' in SERVER
    assert (
        'f"{ROBOT_BRIDGE_URL}/telemetry/localization"'
        in SERVER
    )
    assert (
        'if path == "/dashboard/localization":'
        in SERVER
    )
    assert '"GET"' in SERVER


def test_localization_canvas_is_layered_over_map():
    assert 'id="mapCanvas"' in HTML
    assert 'id="localizationPoseCanvas"' in HTML
    assert HTML.index('id="mapCanvas"') < HTML.index(
        'id="localizationPoseCanvas"'
    )
    assert '#localizationPoseCanvas' in CSS
    assert 'position: absolute;' in CSS
    assert 'pointer-events: none;' in CSS


def test_localization_status_is_visible():
    for identifier in (
        'localizationOverlayStatus',
        'localizationConnection',
        'localizationPose',
        'localizationHeading',
        'localizationPoseAge',
    ):
        assert f'id="{identifier}"' in HTML


def test_pose_uses_saved_map_geometry():
    assert (
        'Number(position.x) - Number(origin.x)'
        in OVERLAY_JAVASCRIPT
    )
    assert (
        'Number(position.y) - Number(origin.y)'
        in OVERLAY_JAVASCRIPT
    )
    assert (
        '/ resolution'
        in OVERLAY_JAVASCRIPT
    )
    assert (
        'sourceHeight - mapCellY'
        in OVERLAY_JAVASCRIPT
    )


def test_heading_uses_ros_map_yaw():
    assert 'context.rotate(-yaw);' in OVERLAY_JAVASCRIPT
    assert 'pose.yaw_radians' in OVERLAY_JAVASCRIPT
    assert 'pose.yaw_degrees' in OVERLAY_JAVASCRIPT
    assert '"Mayday"' in OVERLAY_JAVASCRIPT
    assert 'const pose = lastPose;' in OVERLAY_JAVASCRIPT
    assert 'drawRobot(pose);' in OVERLAY_JAVASCRIPT


def test_stopped_localization_clears_cached_pose():
    assert (
        'payload.runtime_active === false'
        in OVERLAY_JAVASCRIPT
    )
    assert 'clearPose();' in OVERLAY_JAVASCRIPT
    assert 'lastPose = null;' in OVERLAY_JAVASCRIPT
    assert '"Localization stopped"' in OVERLAY_JAVASCRIPT
    assert 'setText("localizationPose", "—");' in (
        OVERLAY_JAVASCRIPT
    )


def test_overlay_has_no_command_capability():
    forbidden = (
        '/motion',
        '/stop',
        '/cmd_vel',
        'navigate_to_pose',
        'compute_path',
        'initialpose',
        'linear_x',
        'angular_z',
    )

    lowered = OVERLAY_JAVASCRIPT.lower()

    for value in forbidden:
        assert value not in lowered

    assert 'pointer-events: none;' in CSS
    assert 'Visualization only.' in HTML
