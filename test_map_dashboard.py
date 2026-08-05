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


def test_perception_page_contains_large_map_canvas():
    assert 'id="perceptionPage"' in HTML
    assert 'id="mapCanvas"' in HTML
    assert 'id="mapStatusPill"' in HTML
    assert 'id="mapDimensions"' in HTML
    assert 'id="mapUnknownCells"' in HTML
    assert 'Saved Occupancy Map' in HTML
    assert 'Visualization only.' in HTML


def test_map_canvas_has_large_responsive_layout():
    assert '.map-layout' in CSS
    assert '.map-stage' in CSS
    assert '#mapCanvas' in CSS
    assert 'min-height: 620px' in CSS
    assert 'height: 76vh' in CSS


def test_map_javascript_preserves_occupancy_semantics():
    assert 'const MAP_ENDPOINT = "/dashboard/map"' in JS
    assert 'function createMapImage(occupancyMap)' in JS
    assert 'function drawMap(occupancyMap)' in JS
    assert 'value === 0' in JS
    assert 'value === 100' in JS
    assert 'height - 1 - mapY' in JS


def test_map_javascript_is_read_only():
    map_source = JS.split(
        '/* Read-only saved occupancy-map visualization */',
        1,
    )[1]

    forbidden = (
        '"/motion"',
        '"/stop"',
        'cmd_vel',
        'navigation goal',
        'initialpose',
        'POST',
        'PUT',
        'PATCH',
        'DELETE',
    )

    for value in forbidden:
        assert value not in map_source


def test_voice_relay_defines_map_proxy():
    assert 'def map_status(self):' in SERVER
    assert (
        'f"{ROBOT_BRIDGE_URL}/telemetry/map"'
        in SERVER
    )
    assert 'if path == "/dashboard/map":' in SERVER


def test_map_proxy_preserves_robot_payload():
    handler = object.__new__(VoiceRelayHandler)
    robot_payload = {
        'ok': True,
        'source': 'validated_saved_map',
        'telemetry': {
            'available': True,
            'map': {
                'frame_id': 'map',
                'width': 152,
                'height': 264,
                'cell_count': 40128,
                'cells': [-1, 0, 100],
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
        status_code, payload = handler.map_status()

    assert status_code == 200
    assert payload == robot_payload
    request.assert_called_once_with(
        'GET',
        (
            'http://minipupperv2.local:8090'
            '/telemetry/map'
        ),
        timeout=5.0,
    )


def test_map_proxy_reports_unavailable_robot():
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
        status_code, payload = handler.map_status()

    assert status_code == 503
    assert payload['ok'] is False
    assert payload['error'] == 'Robot unavailable'
