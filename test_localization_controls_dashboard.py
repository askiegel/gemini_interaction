#!/usr/bin/env python3

"""Checks for minimal guarded localization buttons."""

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
JS = (
    ROOT / 'voice_relay' / 'operator_console.js'
).read_text(encoding='utf-8')


def test_three_guarded_proxy_routes_exist():
    assert '/dashboard/localization-control' in SERVER
    assert '/dashboard/localization-start' in SERVER
    assert '/dashboard/localization-stop' in SERVER


def test_buttons_are_unique():
    assert HTML.count(
        'id="startLocalizationButton"'
    ) == 1
    assert HTML.count(
        'id="stopLocalizationButton"'
    ) == 1


def test_actions_are_fixed():
    assert 'act(START, "Starting")' in JS
    assert 'act(STOP, "Stopping")' in JS
    assert 'method: "POST"' in JS


def test_unsafe_state_is_rejected():
    assert 'planning_enabled !== false' in JS
    assert 'control_enabled !== false' in JS


def test_external_runtime_is_not_stoppable():
    assert 'control.running && !control.owned' in JS
    assert 'setButtons(true, true)' in JS


def test_no_pose_synchronization_was_added():
    control = JS[
        JS.index(
            '/* Minimal guarded localization buttons */'
        ):
        JS.index(
            '/* Read-only saved occupancy-map visualization */'
        )
    ]

    assert 'localizationPoseCanvas' not in control
    assert 'CustomEvent' not in control
    assert 'controlGeneration' not in control
    assert 'requestGeneration' not in control


def test_no_navigation_or_motion_capability():
    control = JS[
        JS.index(
            '/* Minimal guarded localization buttons */'
        ):
        JS.index(
            '/* Read-only saved occupancy-map visualization */'
        )
    ]

    for marker in (
        '/cmd_vel',
        '/motion',
        'navigation_goal',
        'initial_pose',
        'planner_server',
        'controller_server',
    ):
        assert marker not in control


def test_button_styles_exist():
    assert '.localization-controls {' in CSS
    assert '.localization-control-button {' in CSS
