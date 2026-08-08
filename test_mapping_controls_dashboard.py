#!/usr/bin/env python3

"""Tests for guarded supervised mapping dashboard controls."""

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

START = JS.index(
    '/* Guarded supervised mapping controls */'
)
END = JS.index(
    '/* Read-only candidate map review */'
)
CONTROL = JS[START:END]


def test_mapping_controls_are_present_once():
    identifiers = (
        'mappingControlState',
        'mappingControlMessage',
        'mappingReadinessThreshold',
        'startMappingButton',
        'stopMappingButton',
        'saveCandidateButton',
    )

    for identifier in identifiers:
        assert HTML.count(f'id="{identifier}"') == 1

    assert 'Supervised Mapping' in HTML
    assert 'Stop Without Saving' in HTML
    assert 'Save Review Candidate' in HTML


def test_mapping_controls_have_guarded_styles():
    assert '.mapping-controls {' in CSS
    assert '.mapping-control-button.start' in CSS
    assert '.mapping-control-button.stop' in CSS
    assert '.mapping-control-button.save' in CSS
    assert '.mapping-control-state.running' in CSS
    assert '.mapping-control-state.error' in CSS


def test_browser_uses_only_fixed_mapping_routes():
    required = (
        '"/dashboard/mapping-control"',
        '"/dashboard/mapping-start"',
        '"/dashboard/mapping-stop"',
        '"/dashboard/mapping-save-candidate"',
    )

    for route in required:
        assert route in CONTROL

    forbidden = (
        'map_yaml',
        'launch_command',
        'initial_pose',
        'planner',
        'controller_server',
        'cmd_vel',
        'request_body',
    )

    for marker in forbidden:
        assert marker not in CONTROL


def test_save_requires_running_owned_mapping():
    assert 'mapping.running === true' in CONTROL
    assert 'mapping.owned === true' in CONTROL
    assert 'save.disabled = actionInFlight || !running' in CONTROL
    assert 'mapping.validated_map_mutable !== false' in CONTROL
    assert 'mapping.planning_enabled !== false' in CONTROL


def test_mapping_actions_are_post_only():
    assert 'method: "POST"' in CONTROL
    assert 'performAction(' in CONTROL
    assert 'window.confirm(' in CONTROL
    assert 'payload.candidate' in CONTROL
    assert 'Review candidate saved.' in CONTROL


def test_candidate_review_boundary_remains_read_only():
    review_start = JS.index(
        '/* Read-only candidate map review */'
    )
    review_end = JS.index(
        '/* Read-only localized LiDAR map overlay */'
    )
    review = JS[review_start:review_end]

    for marker in (
        'POST',
        'PUT',
        'PATCH',
        'DELETE',
        '/motion',
        '"/stop"',
        'cmd_vel',
        'navigation_goal',
    ):
        assert marker not in review


def test_voice_relay_defines_mapping_proxies():
    assert 'def mapping_control_status(self):' in SERVER
    assert 'def mapping_control_action(self, action):' in SERVER
    assert (
        'f"{ROBOT_BRIDGE_URL}/mapping/status"'
        in SERVER
    )
    assert (
        'f"{ROBOT_BRIDGE_URL}/mapping/{route}"'
        in SERVER
    )
    assert (
        'if path == "/dashboard/mapping-control":'
        in SERVER
    )
    assert (
        '"/dashboard/mapping-save-candidate"'
        in SERVER
    )


def test_mapping_status_proxy_preserves_payload():
    handler = object.__new__(VoiceRelayHandler)
    robot_payload = {
        'ok': True,
        'mapping': {
            'running': False,
            'owned': False,
            'state': 'STOPPED',
            'planning_enabled': False,
            'validated_map_mutable': False,
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
        status_code, payload = (
            handler.mapping_control_status()
        )

    assert status_code == 200
    assert payload == robot_payload
    request.assert_called_once_with(
        'GET',
        (
            'http://minipupperv2.local:8090'
            '/mapping/status'
        ),
        timeout=5.0,
    )


def test_mapping_action_uses_fixed_route_without_body():
    handler = object.__new__(VoiceRelayHandler)
    robot_payload = {
        'ok': True,
        'mapping': {
            'running': True,
            'owned': True,
            'state': 'RUNNING',
        },
    }
    response = {
        'ok': True,
        'status_code': 201,
        'data': robot_payload,
        'error': None,
    }

    with patch(
        'voice_relay.server.request_json',
        return_value=response,
    ) as request:
        status_code, payload = (
            handler.mapping_control_action('start')
        )

    assert status_code == 201
    assert payload == robot_payload
    request.assert_called_once_with(
        'POST',
        (
            'http://minipupperv2.local:8090'
            '/mapping/start'
        ),
        timeout=20.0,
    )


def test_mapping_action_rejects_unknown_action():
    handler = object.__new__(VoiceRelayHandler)

    with patch(
        'voice_relay.server.request_json'
    ) as request:
        status_code, payload = (
            handler.mapping_control_action('promote')
        )

    assert status_code == 400
    assert payload['ok'] is False
    request.assert_not_called()
