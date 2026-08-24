#!/usr/bin/env python3
"""Tests for the read-only live Cartographer dashboard map."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HTML = (ROOT / 'voice_relay/index.html').read_text(
    encoding='utf-8',
)
CSS = (
    ROOT / 'voice_relay/operator_console.css'
).read_text(encoding='utf-8')
JS = (
    ROOT / 'voice_relay/operator_console.js'
).read_text(encoding='utf-8')
SERVER = (
    ROOT / 'voice_relay/server.py'
).read_text(encoding='utf-8')

START = JS.index(
    '/* Read-only live Cartographer mapping map */'
)
END = JS.index(
    '/* Read-only candidate map review */'
)
LIVE = JS[START:END]


def load_server():
    spec = importlib.util.spec_from_file_location(
        'live_mapping_dashboard_server',
        ROOT / 'voice_relay/server.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_mapping_workspace_exists():
    for identifier in (
        'liveMappingCanvas',
        'liveMappingStatus',
        'liveMappingMessage',
        'liveMappingRuntime',
        'liveMappingDimensions',
        'liveMappingProbability',
        'liveMappingAge',
    ):
        assert f'id="{identifier}"' in HTML


def test_live_mapping_workspace_is_responsive():
    assert '.live-mapping-layout' in CSS
    assert '.live-mapping-stage' in CSS
    assert '#liveMappingCanvas' in CSS
    assert '@media (max-width: 950px)' in CSS
    assert '@media (max-width: 650px)' in CSS


def test_live_mapping_uses_get_only_proxy():
    assert (
        'const ENDPOINT = "/dashboard/mapping-map";'
        in LIVE
    )
    assert 'fetch(ENDPOINT' in LIVE
    assert 'method:' not in LIVE


def test_live_mapping_requires_runtime_safety():
    assert 'payload.read_only !== true' in LIVE
    assert 'payload.authoritative !== false' in LIVE
    assert 'payload.runtime_active !== true' in LIVE
    assert 'telemetry.available !== true' in LIVE
    assert 'telemetry.status !== "READY"' in LIVE
    assert (
        '"ros_occupancy_probabilities"'
        in LIVE
    )


def test_live_mapping_renders_probability_gradient():
    assert 'value >= 0 && value <= 100' in LIVE
    assert '248 - value * 2.2' in LIVE
    assert 'probability_cell_count' in LIVE
    assert 'imageSmoothingEnabled = false' in LIVE


def test_stopped_mapping_clears_presentation():
    assert 'function setStopped()' in LIVE
    assert 'liveMap = null;' in LIVE
    assert 'clearCanvas();' in LIVE
    assert '"MAPPING_STOPPED"' in LIVE
    assert 'setText("liveMappingRuntime", "Stopped")' in LIVE


def test_map_labels_do_not_claim_geographic_north():
    assert '"LIVE MAP +Y"' in LIVE
    assert '"CANDIDATE — MAP +Y"' in JS
    assert '"MAP +Y"' in JS
    assert '"NORTH / +Y"' not in JS
    assert '"CANDIDATE — NORTH / +Y"' not in JS


def test_voice_relay_defines_live_map_proxy(monkeypatch):
    server = load_server()
    calls = []

    def fake_request(method, url, timeout):
        calls.append((method, url, timeout))
        return {
            'ok': True,
            'status_code': 200,
            'data': {
                'ok': True,
                'runtime_active': True,
                'read_only': True,
                'authoritative': False,
                'telemetry': {
                    'available': True,
                    'status': 'READY',
                    'map': {'width': 10},
                },
            },
            'error': None,
        }

    monkeypatch.setattr(
        server,
        'request_json',
        fake_request,
    )

    handler = object.__new__(
        server.VoiceRelayHandler
    )
    status, payload = (
        handler.live_mapping_map_status()
    )

    assert status == 200
    assert payload['ok'] is True
    assert calls == [
        (
            'GET',
            (
                f'{server.ROBOT_BRIDGE_URL}'
                '/telemetry/mapping-map'
            ),
            10.0,
        )
    ]


def test_voice_relay_preserves_stopped_payload(monkeypatch):
    server = load_server()

    def fake_request(method, url, timeout):
        del method
        del url
        del timeout

        return {
            'ok': False,
            'status_code': 503,
            'data': {
                'ok': False,
                'runtime_active': False,
                'read_only': True,
                'authoritative': False,
                'telemetry': {
                    'available': False,
                    'status': 'MAPPING_STOPPED',
                    'map': None,
                },
            },
            'error': None,
        }

    monkeypatch.setattr(
        server,
        'request_json',
        fake_request,
    )

    handler = object.__new__(
        server.VoiceRelayHandler
    )
    status, payload = (
        handler.live_mapping_map_status()
    )

    assert status == 503
    assert payload['runtime_active'] is False
    assert payload['telemetry']['map'] is None


def test_live_mapping_remains_read_only():
    forbidden = (
        'method: "POST"',
        'method: "PUT"',
        'method: "PATCH"',
        'method: "DELETE"',
        '/mapping/start',
        '/mapping/stop',
        '/mapping/save-candidate',
        'promote',
        'cmd_vel',
        'navigation goal',
    )

    for marker in forbidden:
        assert marker not in LIVE

    assert (
        'if path == "/dashboard/mapping-map":'
        in SERVER
    )


def test_live_mapping_has_manual_refresh_control():
    assert 'id="liveMappingRefreshButton"' in HTML
    assert "Refresh Map" in HTML

    assert (
        'byId("liveMappingRefreshButton")'
        in LIVE
    )

    assert (
        'refreshButton.addEventListener('
        in LIVE
    )

    # The map side remains GET-only.
    assert 'method: "POST"' not in LIVE
