#!/usr/bin/env python3

"""Tests for read-only candidate map dashboard review."""

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
    '/* Read-only candidate map review */'
)
END = JS.index(
    '/* Read-only localized LiDAR map overlay */'
)
REVIEW = JS[START:END]


def test_candidate_review_workspace_exists():
    identifiers = (
        'candidateReviewCanvas',
        'candidateReviewStatus',
        'candidateInventoryList',
        'candidateReviewName',
        'candidateReviewClassification',
        'candidateDimensionDelta',
        'candidateOriginDelta',
        'candidateCellDelta',
    )

    for identifier in identifiers:
        assert HTML.count(f'id="{identifier}"') == 1

    assert 'Candidate Map Review' in HTML
    assert 'Review only.' in HTML


def test_candidate_review_is_large_and_responsive():
    assert '.candidate-review-layout' in CSS
    assert '.candidate-review-stage' in CSS
    assert '#candidateReviewCanvas' in CSS
    assert 'min-height: 540px' in CSS
    assert 'height: 68vh' in CSS
    assert '@media (max-width: 950px)' in CSS


def test_candidate_review_uses_get_only_proxy():
    assert (
        'const ENDPOINT = "/dashboard/map-candidates"'
        in REVIEW
    )
    assert 'fetch(ENDPOINT' in REVIEW
    assert 'cache: "no-store"' in REVIEW

    forbidden = (
        'POST',
        'PUT',
        'PATCH',
        'DELETE',
        '/motion',
        '"/stop"',
        'cmd_vel',
        'navigation_goal',
        'promote(',
        'remove(',
        'unlink(',
    )

    for marker in forbidden:
        assert marker not in REVIEW


def test_candidate_review_preserves_occupancy_semantics():
    assert 'function createMapImage(occupancyMap)' in REVIEW
    assert 'value === 0' in REVIEW
    assert 'value === 100' in REVIEW
    assert 'height - 1 - mapY' in REVIEW
    assert 'reviewMap = candidate.map' in REVIEW


def test_candidate_review_requires_safety_flags():
    assert 'telemetry.read_only !== true' in REVIEW
    assert 'telemetry.promotion_enabled !== false' in REVIEW
    assert '"REVIEW_READY"' in REVIEW
    assert 'candidate.review_ready === true' in REVIEW


def test_invalid_candidate_is_listed_but_not_rendered():
    assert 'renderInventory(candidates)' in REVIEW
    assert 'candidate.classification' in REVIEW
    assert 'candidate.map' in REVIEW
    assert (
        'const readyCandidates = candidates.filter('
        in REVIEW
    )
    assert 'candidateIsRenderable' in REVIEW


def test_voice_relay_defines_candidate_proxy():
    assert 'def candidate_map_status(self):' in SERVER
    assert (
        'f"{ROBOT_BRIDGE_URL}/telemetry/map-candidates"'
        in SERVER
    )
    assert (
        'if path == "/dashboard/map-candidates":'
        in SERVER
    )


def test_candidate_proxy_preserves_robot_payload():
    handler = object.__new__(VoiceRelayHandler)
    robot_payload = {
        'ok': True,
        'telemetry': {
            'available': True,
            'candidate_count': 2,
            'review_ready_count': 1,
            'invalid_count': 1,
            'read_only': True,
            'promotion_enabled': False,
            'candidates': [],
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
            handler.candidate_map_status()
        )

    assert status_code == 200
    assert payload == robot_payload
    request.assert_called_once_with(
        'GET',
        (
            'http://minipupperv2.local:8090'
            '/telemetry/map-candidates'
        ),
        timeout=10.0,
    )


def test_candidate_proxy_reports_unavailable_robot():
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
        status_code, payload = (
            handler.candidate_map_status()
        )

    assert status_code == 503
    assert payload['ok'] is False
    assert payload['error'] == 'Robot unavailable'

def test_review_ready_candidates_are_selectable():
    assert 'let selectedCandidateName = null;' in REVIEW
    assert 'let reviewCandidates = [];' in REVIEW
    assert 'function candidateIsRenderable(' in REVIEW
    assert 'function selectCandidate(candidate)' in REVIEW
    assert 'item.addEventListener(' in REVIEW
    assert '"click"' in REVIEW
    assert 'selectCandidate(candidate);' in REVIEW
    assert '"aria-pressed"' in REVIEW


def test_candidate_selection_survives_refresh():
    assert 'candidate.name' in REVIEW
    assert '=== selectedCandidateName' in REVIEW
    assert 'selectedCandidateName = ready.name;' in REVIEW
    assert (
        'readyCandidates.length - 1'
        in REVIEW
    )


def test_invalid_candidates_remain_noninteractive():
    assert (
        'const selectable ='
        in REVIEW
    )
    assert (
        'selectable ? "button" : "div"'
        in REVIEW
    )
    assert (
        'candidate.classification'
        in REVIEW
    )
    assert '"REVIEW_READY"' in REVIEW
    assert '&& candidate.map' in REVIEW


def test_candidate_selection_remains_read_only():
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
        assert marker not in REVIEW
