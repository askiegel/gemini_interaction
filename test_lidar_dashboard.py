#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Tony Kiegel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import patch

from voice_relay.server import VoiceRelayHandler


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
SERVER = (
    ROOT / 'voice_relay' / 'server.py'
).read_text(encoding='utf-8')


def test_perception_page_contains_large_lidar_canvas():
    assert 'id="perceptionPage"' in HTML
    assert 'id="lidarCanvas"' in HTML
    assert 'id="lidarStatusPill"' in HTML
    assert 'id="lidarConnection"' in HTML
    assert 'id="lidarSamples"' in HTML
    assert 'Visualization only.' in HTML


def test_lidar_canvas_has_large_responsive_layout():
    assert '.lidar-layout' in CSS
    assert '.lidar-stage' in CSS
    assert '#lidarCanvas' in CSS
    assert 'min-height: 520px' in CSS
    assert 'height: 70vh' in CSS


def test_lidar_display_corrects_physical_orientation():
    assert (
        'const SCAN_DISPLAY_ROTATION_RADIANS = Math.PI / 2'
        in JS
    )
    assert (
        'Number(scan.angle_min)\n'
        '            + SCAN_DISPLAY_ROTATION_RADIANS'
        in JS
    )
    assert (
        'presentation-only'
        in JS
    )
    assert (
        'does not modify ROS sensor data'
        in JS
    )
    assert (
        'centerX\n'
        '                        - Math.sin(angle) * range * scale'
        in JS
    )


def test_lidar_javascript_is_read_only():
    assert 'const ENDPOINT = "/dashboard/lidar"' in JS
    assert 'const REFRESH_MS = 250' in JS
    assert 'function drawScan(scan)' in JS
    assert 'requestAnimationFrame' not in JS

    forbidden = (
        '"/motion"',
        '"/stop"',
        'cmd_vel',
        'navigation goal',
    )

    lidar_source = JS.split(
        '/* Live read-only LD06 visualization */',
        1,
    )[1]

    for value in forbidden:
        assert value not in lidar_source


def test_voice_relay_defines_lidar_proxy():
    assert 'def lidar_status(self):' in SERVER
    assert 'f"{ROBOT_BRIDGE_URL}/telemetry/lidar"' in SERVER
    assert 'if path == "/dashboard/lidar":' in SERVER


def test_lidar_proxy_preserves_robot_payload():
    handler = object.__new__(VoiceRelayHandler)
    robot_payload = {
        'ok': True,
        'telemetry': {
            'available': True,
            'age_seconds': 0.05,
            'scan': {
                'frame_id': 'lidar_link',
                'sample_count': 500,
                'valid_sample_count': 420,
                'ranges': [1.0, None, 2.0],
            },
        },
    }

    response = {
        'ok': True,
        'status_code': 200,
        'data': robot_payload,
        'error': None,
    }

    with patch(
        'voice_relay.server.request_json',
        return_value=response,
    ) as request:
        status_code, payload = handler.lidar_status()

    assert status_code == 200
    assert payload == robot_payload
    request.assert_called_once()


def test_lidar_proxy_reports_unavailable_robot():
    handler = object.__new__(VoiceRelayHandler)

    response = {
        'ok': False,
        'status_code': None,
        'data': None,
        'error': 'Robot unavailable',
    }

    with patch(
        'voice_relay.server.request_json',
        return_value=response,
    ):
        status_code, payload = handler.lidar_status()

    assert status_code == 503
    assert payload['ok'] is False
    assert payload['error'] == 'Robot unavailable'


def test_hidden_dashboard_pages_do_not_poll_in_background():
    required = (
        'function missionControlIsVisible()',
        'if (!missionControlIsVisible()) return;',
        'function diagnosticsIsVisible()',
        'if (!diagnosticsIsVisible()) return;',
        'function missionHistoryIsVisible()',
        'if (!missionHistoryIsVisible()) return;',
        'function worldModelIsVisible()',
        'if (!worldModelIsVisible()) return;',
        'function networkIsVisible()',
        'if (!networkIsVisible()) return;',
    )

    for marker in required:
        assert marker in JS


def test_inline_status_polling_is_mission_control_only():
    assert 'function missionStatusPageIsVisible()' in HTML
    assert 'if (!missionStatusPageIsVisible()) return;' in HTML


def test_localization_status_polling_has_backpressure():
    source = JS.split(
        '/* Minimal guarded localization buttons */',
        1,
    )[1].split(
        '"/dashboard/mapping-control"',
        1,
    )[0]

    assert 'let statusRequestInFlight = false;' in source
    assert (
        'if (busy || statusRequestInFlight) return;'
        in source
    )
    assert 'statusRequestInFlight = true;' in source
    assert 'finally {' in source
    assert 'statusRequestInFlight = false;' in source
