"""Offline frame geometry, ranking and material-progress regressions.

Synthetic ranking forecasts test the ranking contract independently. A real
pure in-place rotation cannot reduce collision occupancy on the same target
ray, and the production geometry tests explicitly enforce that invariant.
"""
import copy
import json
import math
from pathlib import Path
import socket
import time

import pytest

from local_motion_safety_envelope import build_local_motion_lidar_geometry
from marvin_local_obstacle_avoidance import (
    MAX_LOCAL_AVOIDANCE_ACTIONS, rank_marvin_escape_options, select_marvin_escape_action,
)
from marvin_route_obstruction import evaluate_marvin_route, evaluate_route_progress
from test_find_marvin_closed_loop import motions, run
from test_marvin_lateral_avoidance import LEFT_OPEN, strafe_runtime


@pytest.fixture(autouse=True)
def no_live_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Planner tests cannot access robot/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def scan(points):
    geometry = build_local_motion_lidar_geometry({
        'frame_id': 'lidar_link', 'angle_min': -math.pi,
        'angle_increment': math.tau / 80, 'range_min': .02,
        'range_max': 8., 'ranges': [2.] * 80})
    geometry['points'].extend({'x_m': x, 'y_m': y} for x, y in points)
    return dict(available=True, valid=True, reason='fresh', producer_session='test',
                acquisition_sequence=1, age_at_receipt_seconds=0.,
                sectors={'front': {'available': True, 'state': 'CLEAR'}},
                received_monotonic_seconds=time.monotonic(), local_motion_geometry=geometry)


def route(*, overlap=.30, nearest_overlap=.25, occupancy=10, distance=.55, blocked=True):
    return dict(valid=True, route_to_marvin_obstructed=blocked,
                corridor_overlap_m=overlap, blocking_obstacle_overlap_m=nearest_overlap,
                route_occupancy=occupancy, blocking_obstacle_distance_m=distance,
                blocking_obstacle_centerline_clearance_m=.45-nearest_overlap)


@pytest.mark.parametrize('changes,expected,reason', [
    ({'distance': .554}, False, 'no_material_route_improvement'),
    ({'distance': .70}, False, 'no_material_route_improvement'),
    ({'occupancy': 9}, True, 'blocking_return_count_reduced'),
    ({'overlap': .29}, True, 'corridor_overlap_improved'),
    ({'nearest_overlap': .24}, True, 'blocker_centerline_clearance_improved'),
    ({'overlap': .295, 'nearest_overlap': .245}, False, 'no_material_route_improvement'),
    ({'occupancy': 9, 'overlap': .32}, False, 'route_geometry_worsened'),
    ({'overlap': .28, 'nearest_overlap': .27}, False, 'route_geometry_worsened'),
    ({'blocked': False, 'occupancy': 0, 'overlap': 0., 'nearest_overlap': 0.}, True, 'route_cleared'),
])
def test_material_route_progress(changes, expected, reason):
    result = evaluate_route_progress(route(), route(**changes))
    assert result['meaningful_progress'] is expected
    assert result['meaningful_progress_reason'] == reason
    assert result['blocker_distance_change_m'] == pytest.approx(changes.get('distance', .55)-.55)


@pytest.mark.parametrize('bad', [None, {}, {'valid': True}, route(overlap=float('nan'))])
def test_invalid_progress_evidence_does_not_permit_repetition(bad):
    assert not evaluate_route_progress(route(), bad)['meaningful_progress']


def forecast(gain, *, heading=0., clearance=1.):
    before = route()
    after = route(overlap=.30-gain, nearest_overlap=.25-gain)
    return dict(permitted=True, hard_safety_permitted=True, predicted_route=after,
                route_progress=evaluate_route_progress(before, after),
                side_clearance_m=clearance, heading_error_deg=heading)


@pytest.mark.parametrize('strafe_gain,turn_gain,expected', [
    (.060, .014, 'STRAFE_LEFT'),
    (.014, .060, 'TURN_LEFT'),
    (.058, .060, 'STRAFE_LEFT'),
    (.060, .060, 'STRAFE_LEFT'),
])
def test_all_useful_primitives_compete_and_near_ties_prefer_strafe(strafe_gain, turn_gain, expected):
    options = {'STRAFE_LEFT': forecast(strafe_gain), 'TURN_LEFT': forecast(turn_gain)}
    assert rank_marvin_escape_options(options, list(options)) == expected
    assert all('ranking_score' in v for v in options.values())


@pytest.mark.parametrize('bearing', [-12., 0., 15.])
@pytest.mark.parametrize('turn', [-.125, .125])
def test_pure_turn_transforms_target_and_returns_without_world_motion(bearing, turn):
    lidar = scan([(.55, -.10), (.60, .10)])
    association = {'verified_marvin_distance_m': 1.25, 'target_bearing_degrees': bearing}
    before = evaluate_marvin_route(lidar, association, expected_session='test')
    after = evaluate_marvin_route(lidar, association, expected_session='test', heading_change=turn)
    assert after['route_occupancy'] == before['route_occupancy']
    assert after['corridor_overlap_m'] == pytest.approx(before['corridor_overlap_m'])
    assert after['blocking_obstacle_centerline_clearance_m'] == pytest.approx(before['blocking_obstacle_centerline_clearance_m'])
    assert after['blocking_obstacle_distance_m'] == pytest.approx(before['blocking_obstacle_distance_m'])
    x, y = before['blocking_obstacle_x_m'], before['blocking_obstacle_y_m']
    assert after['blocking_obstacle_x_m'] == pytest.approx(math.cos(turn)*x + math.sin(turn)*y)
    assert after['blocking_obstacle_y_m'] == pytest.approx(-math.sin(turn)*x + math.cos(turn)*y)
    assert after['marvin_bearing_deg'] == pytest.approx(bearing-math.degrees(turn))
    assert not evaluate_route_progress(before, after)['meaningful_progress']


@pytest.mark.parametrize('turn', [-.125, .125, .5, 1.0])
def test_pure_turn_cannot_lose_exact_boundary_return_to_roundoff(turn):
    state = scan([(.55, .45)])
    association = {'verified_marvin_distance_m': 1.2, 'target_bearing_degrees': 0.}
    before = evaluate_marvin_route(state, association, expected_session='test')
    after = evaluate_marvin_route(state, association, expected_session='test', heading_change=turn)
    assert after['route_occupancy'] == before['route_occupancy'] == 1
    assert not evaluate_route_progress(before, after)['meaningful_progress']


@pytest.mark.parametrize('dy', [-.04, .04])
def test_strafe_transforms_full_target_before_rebuilding_standoff(dy):
    lidar = scan([(.55, -.10)])
    association = {'verified_marvin_distance_m': 1.2, 'target_bearing_degrees': 0.}
    result = evaluate_marvin_route(lidar, association, expected_session='test', translation_y=dy)
    assert result['blocking_obstacle_x_m'] == pytest.approx(.55)
    assert result['blocking_obstacle_y_m'] == pytest.approx(-.10-dy)
    assert result['heading_error_after_deg'] == pytest.approx(math.degrees(math.atan2(-dy, 1.2)))
    assert result['route_lookahead_m'] == pytest.approx(math.hypot(1.2, dy)-.50)
    assert result['route_width_m'] == .90


def test_safe_turn_does_not_claim_artificial_route_improvement():
    state = scan([(.55, -.1)])
    result = select_marvin_escape_action(state, {'verified_marvin_distance_m': 1.2,
        'target_bearing_degrees': 0.}, expected_session='test', allow_strafe=True)
    assert result['action_type'] == 'STRAFE_LEFT'
    for kind in ('TURN_LEFT', 'TURN_RIGHT'):
        assert result['options'][kind]['hard_safety_permitted']
        assert not result['options'][kind]['improves_route']


def test_ineffective_strafe_reconsiders_all_candidates_without_repeating():
    state = scan([(.55, -.1)])
    association = {'verified_marvin_distance_m': 1.2, 'target_bearing_degrees': 0.}
    first = select_marvin_escape_action(state, association, expected_session='test', allow_strafe=True)
    second = select_marvin_escape_action(state, association, expected_session='test', allow_strafe=True,
                                        previous_selection=first)
    assert second['action_type'] is None
    assert second['reason'] == 'find_marvin_local_avoidance_no_progress'
    assert second['ineffective_action_types'] == ['STRAFE_LEFT']
    assert set(second['options']) == {'STRAFE_LEFT', 'STRAFE_RIGHT', 'TURN_LEFT', 'TURN_RIGHT'}
    assert second['meaningful_progress_reason'] == 'no_material_route_improvement'


def test_missing_side_clearance_cannot_raise_or_select_an_action():
    state = scan([(.55, -.1)])
    state['local_motion_geometry']['sectors']['left'].pop('minimum_distance_from_base_m')
    state['local_motion_geometry']['sectors']['right'].pop('minimum_distance_from_base_m')
    result = select_marvin_escape_action(state, {'verified_marvin_distance_m': 1.2,
        'target_bearing_degrees': 0.}, expected_session='test', allow_strafe=True)
    assert result['action_type'] is None


def test_alternative_turn_is_rankable_after_ineffective_strafe():
    # Ranking contract only: production pure-turn invariance is tested above.
    options = {'STRAFE_LEFT': forecast(.014), 'TURN_LEFT': forecast(.020)}
    eligible = [k for k in options if k not in {'STRAFE_LEFT'}]
    assert rank_marvin_escape_options(options, eligible) == 'TURN_LEFT'


def test_live_geometry_replay_blocks_marginal_repeat_and_keeps_terminal_obstruction():
    fixture = json.loads((Path(__file__).parent / 'test_fixtures' /
        'marvin_4f9791c5_avoidance_geometry.json').read_text())
    selections = []
    for row in fixture['scans']:
        state = scan([])
        state['local_motion_geometry'] = copy.deepcopy(row['local_motion_geometry'])
        state['acquisition_sequence'] = row['acquisition_sequence']
        selected = select_marvin_escape_action(state, row['association'], expected_session='test', allow_strafe=True)
        assert selected['route']['route_to_marvin_obstructed']
        assert selected['route']['route_occupancy'] == row['expected_observed_route']['route_occupancy']
        assert all(not selected['options'][k]['improves_route'] for k in ('TURN_LEFT', 'TURN_RIGHT'))
        selections.append(selected)
    assert selections[0]['action_type'] == 'STRAFE_LEFT'
    row = fixture['scans'][1]
    state = scan([]); state['local_motion_geometry'] = copy.deepcopy(row['local_motion_geometry'])
    second = select_marvin_escape_action(state, row['association'], expected_session='test', allow_strafe=True,
                                        previous_selection=selections[0])
    assert second['action_type'] is None
    assert second['meaningful_progress_reason'] == 'no_material_route_improvement'
    assert second['reason'] == 'find_marvin_local_avoidance_no_progress'
    before, after = fixture['scans'][2]['expected_observed_route'], fixture['actual_progress_history'][2]['route']
    assert evaluate_route_progress(before, after)['meaningful_progress_reason'] == 'route_geometry_worsened'


def test_full_mission_marginal_route_change_hits_phase_stagnation(tmp_path, monkeypatch):
    second = [(x+.004, y) if x>0 else (x, y) for x,y in LEFT_OPEN]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.6)]*8, [LEFT_OPEN, second])
    result = run(bundle[0])
    assert result['state'] == 'BLOCKED'
    assert result['reason'] == 'find_marvin_clear_side_stagnation_exhausted'
    assert result['blocked_wait_recheck_count'] == 0
    assert motions(bundle[3]) == [('strafe', .08, 1.)]*4
    assert result['local_avoidance_actions'] == 4
    assert MAX_LOCAL_AVOIDANCE_ACTIONS == result['max_local_avoidance_actions'] == 6
    row = result['local_avoidance_history'][0]
    assert not row['meaningful_progress']
    assert row['meaningful_progress_reason'] == 'no_material_route_improvement'
    assert row['actual_route_occupancy'] is not None
    assert row['predicted_max_overlap_m'] is not None
    assert row['actual_blocker_centerline_clearance_m'] is not None
    assert result['lidar_wait_history'][0]['snapshot']['acquisition_sequence'] > result['history'][0]['action_lidar_evidence'][1]
    assert result['final_observation']['source_frame_stamp_ns'] > result['history'][0]['source_frame_stamp_ns']
    assert bundle[3][-1] == 'stop'
    diagnostic = result['progress_diagnostics']['actions'][0]
    assert diagnostic['post_action_avoidance']['meaningful_progress_reason'] == row['meaningful_progress_reason']
    assert diagnostic['predicted_max_overlap_m'] == row['predicted_max_overlap_m']


def test_post_strafe_evidence_is_not_overwritten_by_subsequent_alignment(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.6),(55,.6),(0,.5)], [LEFT_OPEN, None])
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED'
    assert motions(bundle[3]) == [('strafe', .08, 1.), ('turn', 'RIGHT', .25, .5)]
    row = result['local_avoidance_history'][0]
    # The first post-strafe camera is the one authorizing alignment, not the
    # later post-alignment ARRIVED observation.
    assert row['post_action_target_association']['acquisition_sequence'] == result['history'][1]['observation']['arrival']['acquisition_sequence']
    assert row['post_action_target_association']['acquisition_sequence'] < result['final_observation']['arrival']['acquisition_sequence']
    assert row['meaningful_progress_reason'] == evaluate_route_progress(
        result['history'][0]['result']['local_detour']['route'],
        result['history'][1]['observation']['arrival']['route'])['meaningful_progress_reason']
