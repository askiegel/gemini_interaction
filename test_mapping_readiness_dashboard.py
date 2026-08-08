#!/usr/bin/env python3

"""Tests for live mapping-readiness dashboard feedback."""

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
    '/* Guarded supervised mapping controls */'
)
END = JS.index(
    '/* Read-only live Cartographer mapping map */'
)
CONTROL = JS[START:END]


def test_live_readiness_identifiers_exist_once():
    identifiers = (
        'mappingReadinessLive',
        'mappingReadinessStatus',
        'mappingSubmapProgressText',
        'mappingSubmapProgress',
        'mappingSubmapProgressBar',
        'mappingMatureProgressText',
        'mappingMatureProgress',
        'mappingMatureProgressBar',
        'mappingSubmapVersions',
    )

    for identifier in identifiers:
        assert HTML.count(f'id="{identifier}"') == 1


def test_readiness_progress_is_accessible():
    assert 'role="progressbar"' in HTML
    assert 'aria-label="Submap progress"' in HTML
    assert (
        'aria-label="Mature submap progress"'
        in HTML
    )
    assert 'aria-valuenow="0"' in HTML


def test_readiness_styles_distinguish_states():
    assert '.mapping-readiness-live {' in CSS
    assert (
        '.mapping-readiness-live-status.building'
        in CSS
    )
    assert (
        '.mapping-readiness-live-status.ready'
        in CSS
    )
    assert (
        '.mapping-readiness-live-status.error'
        in CSS
    )
    assert (
        '.mapping-readiness-progress-track'
        in CSS
    )


def test_controller_reads_live_readiness():
    required = (
        'mapping.readiness',
        'readiness.submap_count',
        'readiness.mature_submap_count',
        'readiness.minimum_submap_count',
        'readiness.minimum_mature_submap_count',
        'readiness.minimum_mature_version',
        'readiness.submaps',
    )

    for marker in required:
        assert marker in CONTROL


def test_controller_renders_every_submap_version():
    assert 'mappingSubmapVersions' in CONTROL
    assert '`#${submap.index} v${submap.version}`' in CONTROL
    assert '.join(" · ")' in CONTROL


def test_ready_state_requires_backend_confirmation():
    assert 'readiness.ready === true' in CONTROL
    assert (
        'readiness.status === "READY_TO_SAVE"'
        in CONTROL
    )
    assert 'latestReadinessReady' in CONTROL


def test_save_is_disabled_until_ready():
    assert (
        'save.disabled = actionInFlight || !running'
        in CONTROL
    )
    assert '!latestReadinessReady' in CONTROL
    assert 'Mapping is still building. Wait until live ' in CONTROL
    assert 'readiness reports Ready to save.' in CONTROL


def test_progress_values_are_bounded():
    assert 'function clampProgress(value)' in CONTROL
    assert 'Math.max(0, Math.min(1, number))' in CONTROL
    assert 'aria-valuenow' in CONTROL
    assert 'bar.style.width' in CONTROL


def test_stopped_mapping_clears_visible_progress():
    assert '"MAPPING_STOPPED"' in CONTROL
    assert 'label = "Mapping stopped"' in CONTROL
    assert 'state = "stopped"' in CONTROL
    assert '"No live submaps"' in CONTROL


def test_readiness_feature_adds_no_unsafe_actions():
    forbidden = (
        'promote-candidate',
        'controller_server',
        'bt_navigator',
        'cmd_vel',
        'navigation_goal',
        'map_yaml',
        'candidate_root',
    )

    for marker in forbidden:
        assert marker not in CONTROL
