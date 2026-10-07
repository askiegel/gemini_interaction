"""Offline full-sweep, adaptive-duration and one-action strafe regressions."""
import math
import socket

import pytest

from local_motion_safety_envelope import evaluate_local_motion_safety
from marvin_local_obstacle_avoidance import (
    LOCAL_AVOIDANCE_STRAFE_SPEED_MPS, LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
    LOCAL_AVOIDANCE_STRAFE_MIN_SECONDS, MAX_LOCAL_AVOIDANCE_ACTIONS,
    TURN_SPEED, TURN_DURATION, safe_marvin_strafe_duration, select_marvin_escape_action,
)
from marvin_route_obstruction import evaluate_route_progress
from test_marvin_avoidance_planner_progress import scan, route
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_find_marvin_closed_loop import motions, run
from test_robot_bridge_lateral_client import client_with_transport


@pytest.fixture(autouse=True)
def no_live_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Duration tests must not access robot or services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def guard(state, sign, seconds):
    return evaluate_local_motion_safety(state, expected_session='test',
        linear_y=sign*.08, duration=seconds, lateral_swept_footprint=True)


def test_canonical_contract_preserves_all_other_primitive_limits():
    from behavior_manager import BehaviorManager
    assert LOCAL_AVOIDANCE_STRAFE_SPEED_MPS == .08
    assert LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS == 1.
    assert LOCAL_AVOIDANCE_STRAFE_SPEED_MPS * LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS == .08
    assert LOCAL_AVOIDANCE_STRAFE_MIN_SECONDS == .25
    assert MAX_LOCAL_AVOIDANCE_ACTIONS == 6
    assert (TURN_SPEED, TURN_DURATION) == (.25, .5)
    assert BehaviorManager.FIND_APPROACH_FORWARD_SPEED == .10
    assert BehaviorManager.FIND_APPROACH_FORWARD_SECONDS == .50


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('x,y', [(0., .51), (.44, .16), (-.44, .16)])
def test_full_eight_cm_sweep_detects_side_and_front_rear_corner_hazards(sign, x, y):
    state = scan([(x, sign*y)])
    assert guard(state, sign, .5)['permitted']  # Four cm does not reach the hazard.
    full = guard(state, sign, 1.)
    assert not full['permitted']
    assert full['reason'] == 'translation_protected_region_violated'
    assert full['protected_radius_m'] == .45
    duration, safe = safe_marvin_strafe_duration(state, expected_session='test', linear_y=sign*.08)
    assert .5 < duration < 1.
    assert safe['permitted'] and guard(state, sign, duration)['permitted']
    assert not guard(state, sign, min(1., duration+.001))['permitted']
    assert len(safe['required_sectors']) == 8


@pytest.mark.parametrize('sign', [1, -1])
def test_clear_sweep_accepts_full_duration_and_predicts_full_signed_translation(sign):
    state = scan([(.55, -sign*.1)])
    association = {'verified_marvin_distance_m': 1.2, 'target_bearing_degrees': 0.}
    result = select_marvin_escape_action(state, association, expected_session='test', allow_strafe=True)
    candidate = result['options']['STRAFE_LEFT' if sign == 1 else 'STRAFE_RIGHT']
    assert candidate['requested_duration'] == 1.
    assert candidate['nominal_lateral_displacement_m'] == .08
    assert candidate['predicted_route']['blocking_obstacle_y_m'] == pytest.approx(-sign*.18)
    assert candidate['predicted_route']['marvin_bearing_deg'] == pytest.approx(math.degrees(math.atan2(-sign*.08, 1.2)))


@pytest.mark.parametrize('sign', [1, -1])
def test_start_footprint_or_too_little_travel_rejects_primitive(sign):
    for y in [.44, .465]:
        duration, safe = safe_marvin_strafe_duration(scan([(0., sign*y)]), expected_session='test', linear_y=sign*.08)
        assert duration == 0. and not safe['permitted']


@pytest.mark.parametrize('fault', ['stale', 'session', 'coverage'])
def test_shortening_cannot_admit_invalid_sensor_evidence(fault):
    state = scan([(0., .51)])
    if fault == 'stale': state['age_at_receipt_seconds'] = .301
    elif fault == 'session': state['producer_session'] = 'other'
    else: state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    duration, safe = safe_marvin_strafe_duration(state, expected_session='test', linear_y=.08)
    assert duration == 0. and not safe['permitted']


def test_prediction_uses_shortened_path_instead_of_full_eight_cm():
    state = scan([(.55, -.1), (0., .51)])
    result = select_marvin_escape_action(state, {'verified_marvin_distance_m':1.2, 'target_bearing_degrees':0.}, expected_session='test', allow_strafe=True)
    option = result['options']['STRAFE_LEFT']; duration = option['requested_duration']
    assert result['action_type'] == 'STRAFE_LEFT'
    assert .5 < duration < .75
    assert option['nominal_lateral_displacement_m'] == pytest.approx(.08*duration)
    assert option['predicted_route']['blocking_obstacle_y_m'] == pytest.approx(-.1-.08*duration)


@pytest.mark.parametrize('y', [.08, -.08])
def test_client_transports_full_one_second_pure_strafe(monkeypatch, y):
    client, requests, _ = client_with_transport(monkeypatch)
    result = client.move_lateral(speed=y, seconds=1.)
    assert result['ok']
    assert requests[-1] == ('POST', '/motion', {'linear_x':0., 'linear_y':y, 'angular_z':0., 'duration':1.})


@pytest.mark.parametrize('shortened', [False, True])
def test_runtime_executes_one_selected_sweep_then_stops_refreshes_and_resumes_forward(tmp_path, monkeypatch, shortened):
    scene = LEFT_OPEN + ([(0., .51)] if shortened else [])
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.6),(0,.6),(0,.5)], [scene,None])
    runtime, _, _, events, _ = bundle
    import behavior_manager
    evaluate = behavior_manager.evaluate_local_motion_safety
    sweeps = []
    def traced(*args, **kwargs):
        if kwargs.get('lateral_swept_footprint'):
            sweeps.append((kwargs['linear_y'], kwargs['duration']))
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(behavior_manager, 'evaluate_local_motion_safety', traced)
    result = run(runtime)
    commands = motions(events)
    assert result['state'] == 'ARRIVED'
    assert len(commands) == 2 and commands[0][:2] == ('strafe', .08)
    duration = commands[0][2]
    assert (.5 < duration < .75) if shortened else duration == 1.
    assert commands[1] == ('forward', .10, .50)
    assert events[events.index(commands[0])+1] == 'stop'
    first = result['history'][0]['result']
    assert first['requested_duration'] == duration and first['bridge_stop_confirmed']
    assert sweeps and all(y == .08 and seconds == duration for y, seconds in sweeps)
    assert first['local_detour']['options']['STRAFE_LEFT']['requested_duration'] == duration
    safety = first['lateral_step']['lateral_safety']; assert safety['permitted'] and safety['protected_radius_m'] == .45
    assert len(safety['required_sectors']) == 8
    action = result['progress_diagnostics']['actions'][0]
    assert action['command']['duration'] == duration
    assert result['lidar_wait_history'][0]['snapshot']['acquisition_sequence'] > result['history'][0]['action_lidar_evidence'][1]
    assert int(result['history'][1]['source_frame_stamp_ns']) > int(result['history'][0]['source_frame_stamp_ns'])
    assert result['local_avoidance_actions'] == 1


def test_jit_reduced_safe_duration_vetoes_original_longer_command(tmp_path, monkeypatch):
    scenes = [LEFT_OPEN]
    bundle, _, client = strafe_runtime(tmp_path, monkeypatch, [(0,.6)], scenes)
    request = client._request
    def changed(method, path, payload=None):
        response = request(method, path, payload)
        if method == 'GET': scenes[0] = LEFT_OPEN + [(0., .51)]
        return response
    client._request = changed
    result = run(bundle[0])
    assert result['state'] == 'BLOCKED' and motions(bundle[3]) == []
    assert result['history'][0]['result']['source_stamp_consumed']


def test_ineffective_longer_strafe_still_cannot_repeat(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.6)]*3, [LEFT_OPEN])
    result = run(bundle[0])
    assert motions(bundle[3]) == [('strafe', .08, 1.)]
    assert result['reason'] == 'find_marvin_local_avoidance_no_progress'
    assert not result['local_avoidance_history'][-1]['selection']['meaningful_progress']
    assert 'STRAFE_LEFT' in result['local_avoidance_history'][-1]['selection']['ineffective_action_types']


def test_progress_threshold_is_not_lowered_for_longer_strafe():
    before = route(); after = route(overlap=.2969, nearest_overlap=.2469, distance=.58)
    assert not evaluate_route_progress(before, after)['meaningful_progress']
    after = route(overlap=.29, nearest_overlap=.24)
    assert evaluate_route_progress(before, after)['meaningful_progress']
