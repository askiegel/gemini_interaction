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
