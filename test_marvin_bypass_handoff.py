"""Offline shared-selector priority, retained production scans and real guards.

Sensors and transport are fakes; selections are never injected. Recorded scans
end at the first counterfactual bypass. Synthetic continuation is labelled.
"""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from marvin_local_obstacle_avoidance import select_marvin_escape_action, MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import evaluate_route_progress
from runtime import _marvin_proof_selection_history
from test_find_marvin_closed_loop import run, motions
from test_marvin_lateral_avoidance import strafe_runtime
from test_marvin_local_bypass import scene_plan, OPEN_LEFT
from test_marvin_live_proof_continuation import initial, arm, step, complete


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Handoff tests cannot access robot/network/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def live_rows():
    return json.loads((Path(__file__).parent / 'test_fixtures/marvin_396e9ec4_bypass_handoff.json').read_text())['avoidance_rows']


def select_row(row, previous=None, remaining=6):
    state = copy.deepcopy(row['lidar'])
    state['received_monotonic_seconds'] = time.monotonic()
    state['age_at_receipt_seconds'] = 0.
    return select_marvin_escape_action(state, row['association'],
        expected_session=state['producer_session'], allow_strafe=True,
        previous_selection=previous, remaining_avoidance_actions=remaining)


def stopped_outcome(selection, row):
    """Build historical evidence from recorded measured geometry, not commands."""
    selection = copy.deepcopy(selection)
    after = row['recorded_post_action']['target_association']
    measured = evaluate_route_progress(selection['route'], after['route'])
    assert measured['meaningful_progress'] == row['recorded_post_action']['meaningful_progress']
    selection['first_post_action_strafe_progress'] = dict(progress=measured,
        producer_session=after['producer_session'],
        action_acquisition_sequence=selection['acquisition_sequence'],
        acquisition_sequence=after['acquisition_sequence'])
    return selection


def handoff_case():
    rows = live_rows()
    first = select_row(rows[0])
    return rows[1], stopped_outcome(first, rows[0])


def test_recorded_first_action_stays_strafe_and_cannot_bypass_before_separation():
    row = live_rows()[0]
    selected = select_row(row)
    assert selected['action_type'] == 'STRAFE_LEFT'
    assert selected['local_bypass']['local_bypass_reason'] == 'bypass_lateral_clearance_not_established'
    assert selected['local_bypass']['bypass_corridor_occupancy'] is None
    assert not selected['bypass_handoff']['selected']


def test_fresh_clear_corridor_alone_keeps_first_useful_strafe():
    row, _ = handoff_case()
    selected = select_row(row)
    assert selected['local_bypass']['bypass_forward_permitted']
    assert selected['action_type'] == 'STRAFE_LEFT'
    assert selected['bypass_handoff']['reason'] == 'same_side_lateral_history_required'


def test_measured_progress_hands_off_while_strafe_is_still_useful():
    row, previous = handoff_case()
    selected = select_row(row, previous, 5)
    assert selected['action_type'] == 'BYPASS_FORWARD'
    assert selected['reason'] == 'find_marvin_local_bypass_handoff_selected'
    d = selected['bypass_handoff']
    assert d['selected'] and d['eligible'] and d['measured_lateral_progress']
    assert d['predicted_strafe_overlap_improvement_m'] == pytest.approx(.0457368928621766)
    assert d['predicted_strafe_centerline_improvement_m'] > .01
    assert d['predicted_bypass_longitudinal_improvement_m'] == pytest.approx(.05)
    assert d['bypass_corridor_occupancy'] == d['bypass_corridor_overlap_m'] == 0
    assert d['current_side_clearance_m'] == pytest.approx(.99841771268834)
    assert d['current_avoidance_count'] == 1 and d['remaining_avoidance_actions'] == 5
    assert selected['options']['STRAFE_LEFT']['permitted'] and selected['options']['STRAFE_LEFT']['improves_route']
    assert selected['options']['BYPASS_FORWARD']['route_progress']['meaningful_progress']


@pytest.mark.parametrize('fault', ['missing', 'failed', 'count_only_later_alignment', 'wrong_session',
    'future_outcome', 'same_action_scan', 'bool_sequence', 'turn_history', 'wrong_direction',
    'suppressed', 'recovery_used', 'recovery_selected', 'stationary_recovery', 'malformed', 'malformed_progress'])
def test_handoff_requires_first_measured_same_side_outcome_and_no_suppression(fault):
    row, old = handoff_case()
    if fault == 'missing': old.pop('first_post_action_strafe_progress')
    if fault == 'malformed': old['first_post_action_strafe_progress'] = 'invalid'
    if fault == 'malformed_progress': old['first_post_action_strafe_progress']['progress'] = 'invalid'
    if fault in {'failed', 'count_only_later_alignment'}:
        old['first_post_action_strafe_progress']['progress']['meaningful_progress'] = False
    if fault == 'wrong_session': old['first_post_action_strafe_progress']['producer_session'] = 'other'
    if fault == 'future_outcome': old['first_post_action_strafe_progress']['acquisition_sequence'] = 99999
    if fault == 'same_action_scan': old['first_post_action_strafe_progress']['acquisition_sequence'] = old['acquisition_sequence']
    if fault == 'bool_sequence': old['first_post_action_strafe_progress']['action_acquisition_sequence'] = True
    if fault == 'turn_history': old['action_type'] = 'TURN_LEFT'
    if fault == 'wrong_direction': old['direction'] = 'RIGHT'
    if fault == 'suppressed': old['ineffective_action_types'] = ['BYPASS_FORWARD']
    if fault == 'recovery_used': old['post_bypass_lateral_recovery_used'] = True
    if fault == 'recovery_selected': old['post_bypass_lateral_recovery_selected'] = True
    if fault == 'stationary_recovery': old['stationary_lateral_reconsidered'] = True
    before = copy.deepcopy(old)
    selected = select_row(row, old, 5)
    assert selected['action_type'] != 'BYPASS_FORWARD'
    assert not selected['bypass_handoff']['selected']
    assert old == before


@pytest.mark.parametrize('fault', ['stale', 'session', 'rear_coverage', 'endpoint_hazard', 'depth', 'margin'])
def test_established_history_cannot_override_fresh_bypass_safety(fault):
    row, old = handoff_case()
    state = copy.deepcopy(row['lidar']); association = copy.deepcopy(row['association'])
    state['received_monotonic_seconds'] = time.monotonic()
    if fault == 'stale': state['received_monotonic_seconds'] -= 1.
    if fault == 'session': state['producer_session'] = 'wrong'
    if fault == 'rear_coverage': state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'endpoint_hazard': state['local_motion_geometry']['points'].append({'x_m': .17, 'y_m': .40})
    if fault == 'depth': association['verified_marvin_distance_m'] = None
    if fault == 'margin': association['verified_marvin_conservative_distance_m'] = .52
    selected = select_marvin_escape_action(state, association, expected_session=row['lidar']['producer_session'],
        allow_strafe=True, previous_selection=old, remaining_avoidance_actions=5)
    assert selected['action_type'] != 'BYPASS_FORWARD'


def measured_scene(points=OPEN_LEFT, association=None):
    state, current = scene_plan(points, association=association)
    old = copy.deepcopy(current)
    old['route']['corridor_overlap_m'] += .02
    old['route']['blocking_obstacle_overlap_m'] += .02
    old['first_post_action_strafe_progress'] = dict(
        progress=evaluate_route_progress(old['route'], current['route']),
        producer_session='test', action_acquisition_sequence=1, acquisition_sequence=2)
    state['acquisition_sequence'] = 3
    return state, old


def test_strongly_superior_strafe_keeps_priority():
    association = dict(verified_marvin_distance_m=3., verified_marvin_conservative_distance_m=2.9,
                       target_bearing_degrees=0.)
    state, old = measured_scene(association=association)
    selected = select_marvin_escape_action(state, association, expected_session='test',
        allow_strafe=True, previous_selection=old, remaining_avoidance_actions=5)
    assert selected['action_type'] == 'STRAFE_LEFT'
    d = selected['bypass_handoff']
    assert d['eligible'] and not d['selected']
    assert d['predicted_strafe_overlap_improvement_m'] > .053
    assert d['reason'] == 'strafe_gain_exceeds_bypass_passage'


def test_route_clearing_strafe_keeps_priority():
    points = [(.65, -.44), (0., 1.2), (0., -.65)]
    state, old = measured_scene(points)
    from test_marvin_local_bypass import ASSOCIATION
    selected = select_marvin_escape_action(state, ASSOCIATION, expected_session='test',
        allow_strafe=True, previous_selection=old, remaining_avoidance_actions=5)
    assert selected['action_type'] == 'STRAFE_LEFT'
    assert selected['bypass_handoff']['reason'] == 'strafe_clears_direct_route'


@pytest.mark.parametrize('passage', [0., .009, None, float('nan')])
def test_clear_corridor_and_measured_history_still_require_predicted_passage(passage):
    from marvin_local_obstacle_avoidance import _bypass_handoff
    row, old = handoff_case()
    selected = select_row(row, old, 5)
    # Restore the still-useful lateral competitor, without injecting a planner
    # result: this exercises the numerical comparison's invalid forecast gate.
    comparison = dict(selected, action_type='STRAFE_LEFT', direction='LEFT')
    forecast = dict(selected['options']['BYPASS_FORWARD']['route_progress'],
                    bypass_longitudinal_progress_m=passage)
    d = _bypass_handoff(comparison, old, selected['local_bypass'], forecast,
        expected_session=row['lidar']['producer_session'], remaining_avoidance_actions=5)
    assert not d['selected'] and d['reason'] == 'fresh_bypass_passage_required'


def test_right_side_handoff_is_symmetric_and_side_memory_is_stable():
    from test_marvin_local_bypass import ASSOCIATION
    state, old = measured_scene([(x, -y) for x, y in OPEN_LEFT])
    assert old['action_type'] == 'STRAFE_RIGHT'
    state['local_motion_geometry']['sectors']['left']['minimum_distance_from_base_m'] = 4.
    selected = select_marvin_escape_action(state, ASSOCIATION, expected_session='test',
        allow_strafe=True, previous_selection=old, remaining_avoidance_actions=5)
    assert selected['action_type'] == 'BYPASS_FORWARD' and selected['direction'] == 'RIGHT'
    assert not selected['bypass_side_change_allowed']


@pytest.mark.parametrize('gain,preferred', [(.0529, True), (.053, True), (.0531, False)])
def test_existing_three_mm_tolerance_bounds_forward_priority(gain, preferred):
    from marvin_local_obstacle_avoidance import _bypass_handoff
    row, old = handoff_case()
    selected = select_row(row, old, 5)
    comparison = copy.deepcopy(selected)
    comparison.update(action_type='STRAFE_LEFT', direction='LEFT')
    comparison['options']['STRAFE_LEFT']['route_progress'].update(
        corridor_overlap_reduction_m=gain, centerline_clearance_improvement_m=gain)
    forecast = dict(selected['options']['BYPASS_FORWARD']['route_progress'],
                    bypass_longitudinal_progress_m=.05)
    d = _bypass_handoff(comparison, old, selected['local_bypass'], forecast,
        expected_session=row['lidar']['producer_session'], remaining_avoidance_actions=5)
    assert d['selected'] is preferred


def test_better_right_clearance_cannot_reverse_established_left_handoff():
    row, old = handoff_case()
    for sector in ('front_right', 'right', 'rear_right'):
        row['lidar']['local_motion_geometry']['sectors'][sector]['minimum_distance_from_base_m'] = 5.
    selected = select_row(row, old, 5)
    assert selected['action_type'] == 'BYPASS_FORWARD' and selected['direction'] == 'LEFT'
    assert not selected['bypass_side_change_allowed']


@pytest.mark.parametrize('remaining', [0, -1, True, 1.5])
def test_handoff_cannot_overrun_six_action_budget(remaining):
    row, old = handoff_case()
    selected = select_row(row, old, remaining)
    assert selected['action_type'] is None
    assert selected['reason'] == 'find_marvin_local_avoidance_exhausted'
    assert MAX_LOCAL_AVOIDANCE_ACTIONS == 6


def test_live_replay_naturally_hands_off_before_exhaustion_without_injected_selection():
    rows = live_rows(); old = None; baseline = []
    # Without the newly recorded first-outcome evidence, the previous useful-
    # strafe priority remains. Every primitive here comes from the real selector.
    for count, row in enumerate(rows):
        old = select_row(row, old, 6-count)
        baseline.append(old['action_type'])
    assert baseline == ['STRAFE_LEFT'] * 6
    first = select_row(rows[0])
    second = select_row(rows[1], stopped_outcome(first, rows[0]), 5)
    assert [first['action_type'], second['action_type']] == ['STRAFE_LEFT', 'BYPASS_FORWARD']
    assert rows[1]['physical_action'] == 2
    assert second['bypass_handoff']['current_avoidance_count'] == 1
    assert second['local_bypass']['bypass_target_x_m'] == .15
    assert second['local_bypass']['bypass_target_y_m'] == 0.
    assert second['local_bypass']['protected_radius_m'] == .45


def priority_bundle(tmp_path, monkeypatch, *, clear=True, specs=None):
    # Synthetic continuation: measured scans are independently specified; no
    # command-distance integration manufactures bypass progress.
    scenes = [[(.65, -.10)], [(.65, -.18)], None if clear else [(.65, -.18)]]
    if specs is None:
        specs = ([(0, 1.1)] * 2 + [(0, round(v/100, 2)) for v in range(105, 49, -5)]
                 if clear else [(0, 1.1)] * 20)
    bundle, flags, client = strafe_runtime(tmp_path, monkeypatch, specs, scenes)
    read = bundle[0].world_model.get_lidar_obstacles
    def acquire(**kwargs):
        bundle[4][0] += 1000
        return read(**kwargs)
    bundle[0].world_model.get_lidar_obstacles = acquire
    return bundle, flags, client


def assert_sensor_contracts(result, runtime):
    stamps = [h['source_frame_stamp_ns'] for h in result['history'] if h.get('motion_executed')]
    assert len(stamps) == len(set(stamps))
    assert all(type(stamp) is int and stamp in runtime._marvin_alignment_consumed_source_frame_stamps for stamp in stamps)
    for h in result['history']:
        if not h.get('motion_executed'): continue
        assert h['result']['stop_result']['ok']
        assert any(w['snapshot']['acquisition_sequence'] > h['action_lidar_evidence'][1]
                   for w in result['lidar_wait_history'] if w.get('ok'))
    assert result['stop_result']['ok'] and not result['bridge_after_stop']['motion']['streaming']


def test_normal_mission_shared_selector_handoff_then_direct_forward_and_arrival(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch)
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED'
    assert motions(bundle[3])[:3] == [('strafe', .08, 1.), ('forward', .1, .5), ('forward', .1, .5)]
    assert result['local_avoidance_actions'] == 2 and result['local_bypass_actions'] == 1
    assert result['completed_forward_actions'] > 0
    jit = result['history'][1]['result']['local_detour']
    assert jit['accepted'] and jit['action_type'] == 'BYPASS_FORWARD'
    assert jit['bypass_handoff']['selected']
    assert jit['acquisition_sequence'] > result['local_avoidance_history'][1]['selection']['acquisition_sequence']
    diagnostics = result['progress_diagnostics']
    assert diagnostics['action_summary'][1]['pre_action_avoidance']['bypass_handoff']['selected']
    assert diagnostics['actions'][1]['local_avoidance_selection']['bypass_handoff']['selected']
    assert_sensor_contracts(result, bundle[0])
    for i, a in enumerate(result['progress_diagnostics']['actions']):
        assert a['first_post_action_lidar'] and a['first_new_post_action_camera']
        if i < 2: assert a['next_associated_camera']['source_frame_stamp_ns'] > a['authorizing_camera']['source_frame_stamp_ns']


def test_proof_same_selector_retains_outcome_and_hands_off_then_direct_forward(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch, specs=[(0, 1.1)] * 12)
    r = bundle[0]
    first = initial(bundle); complete(bundle, first, 0)
    old = copy.deepcopy(r._marvin_live_proof_continuation.previous_selection)
    assert old['first_post_action_strafe_progress']['progress']['meaningful_progress']
    assert 'options' not in old and 'local_bypass' not in old
    assert arm(r)['ok']; second = step(r); complete(bundle, second, 1)
    action = second['controller_result']['history'][0]['result']
    assert action['local_detour']['action_type'] == 'BYPASS_FORWARD'
    assert action['local_detour']['bypass_handoff']['selected']
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions'] == 2
    assert arm(r)['ok']; third = step(r); complete(bundle, third, 2)
    assert third['controller_result']['history'][0]['state'] == 'ADVANCING'
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions'] == 2


def test_alignment_preserves_first_outcome_and_does_not_spend_avoidance_budget(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch, clear=False)
    r, behavior = bundle[:2]
    complete(bundle, initial(bundle), 0)
    old = copy.deepcopy(r._marvin_live_proof_continuation.previous_selection)
    behavior.specs = iter([(100, 1.1), (0, 1.1)])
    assert arm(r)['ok']; aligned = step(r); complete(bundle, aligned, 1)
    c = r._marvin_live_proof_continuation
    assert aligned['controller_result']['history'][0]['state'] == 'ALIGNING'
    assert c.avoidance['local_avoidance_actions'] == 1
    assert c.previous_selection == old


def test_alignment_cannot_turn_failed_strafe_outcome_into_handoff_credit(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch, clear=False)
    r, behavior, _, events, _ = bundle
    read = r.world_model.get_lidar_obstacles
    def scene(**kwargs):
        scan = read(**kwargs)
        if len(motions(events)) == 1:
            # First strafe realizes no lateral progress. Improved geometry is
            # supplied only after alignment, not attributed to that strafe.
            for p in scan['local_motion_geometry']['points']:
                if p['x_m'] == .65 and p['y_m'] == -.18: p['y_m'] = -.10
        return scan
    r.world_model.get_lidar_obstacles = scene
    complete(bundle, initial(bundle), 0)
    frozen = copy.deepcopy(r._marvin_live_proof_continuation.previous_selection['first_post_action_strafe_progress'])
    assert not frozen['progress']['meaningful_progress']
    behavior.specs = iter([(100, 1.1), (0, 1.1)])
    assert arm(r)['ok']; complete(bundle, step(r), 1)
    assert r._marvin_live_proof_continuation.previous_selection['first_post_action_strafe_progress'] == frozen
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions'] == 1
    behavior.specs = iter([(0, 1.1)] * 6)
    assert arm(r)['ok']; result = step(r); complete(bundle, result, 2)
    selected = result['controller_result']['history'][0]['result']['local_detour']
    assert selected['action_type'] == 'STRAFE_LEFT'
    assert selected['local_bypass']['bypass_forward_permitted']
    assert selected['bypass_handoff']['reason'] == 'measured_lateral_progress_required'


def test_failed_bypass_still_gets_one_recovery_then_stationary_wait(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch, clear=False)
    result = run(bundle[0])
    assert result['reason'] == 'find_marvin_blocked_wait_exhausted'
    assert result['blocked_wait_reason'] == 'find_marvin_local_avoidance_no_progress'
    assert result['local_avoidance_actions'] == 5 and result['local_bypass_actions'] == 3
    assert motions(bundle[3]) == [('strafe', .08, 1.)] + [('forward', .1, .5)]*3 + [('strafe', .08, 1.)]
    assert result['blocked_wait_recheck_count'] == 12
    assert MAX_LOCAL_AVOIDANCE_ACTIONS == 6
    assert_sensor_contracts(result, bundle[0])


def test_handoff_final_jit_veto_waits_without_transport_or_budget_spend(tmp_path, monkeypatch):
    bundle, _, _ = priority_bundle(tmp_path, monkeypatch, clear=False)
    runtime, behavior = bundle[:2]
    original = behavior.execute_single_marvin_approach_step
    read = runtime.world_model.get_lidar_obstacles
    hazard = [False]
    def guarded_scan(**kwargs):
        scan = read(**kwargs)
        if hazard[0]:
            scan['local_motion_geometry']['points'].append({'x_m': .17, 'y_m': .40})
        return scan
    runtime.world_model.get_lidar_obstacles = guarded_scan
    def veto_after_runtime_selection(**kwargs):
        if kwargs.get('local_selection_validator'):
            hazard[0] = True
        return original(**kwargs)
    behavior.execute_single_marvin_approach_step = veto_after_runtime_selection
    result = run(runtime)
    assert result['reason'] == 'find_marvin_blocked_wait_exhausted'
    assert result['blocked_wait_reason'] == 'marvin_local_bypass_jit_veto'
    assert result['local_avoidance_actions'] == 1 and result['local_bypass_actions'] == 0
    assert motions(bundle[3]) == [('strafe', .08, 1.)]
    veto = result['history'][1]['result']
    assert veto['source_stamp_consumed'] and not veto['motion_executed']
    assert veto['pre_transport_jit_veto']['transport_attempted'] is False
    assert veto['source_frame_stamp_ns'] in runtime._marvin_alignment_consumed_source_frame_stamps
    assert result['blocked_wait_recheck_count'] == 12


def test_repeated_planning_never_accumulates_handoff_or_mutates_history():
    row, old = handoff_case(); before = copy.deepcopy(old)
    results = [select_row(row, old, 5) for _ in range(6)]
    assert all(r['action_type'] == 'BYPASS_FORWARD' for r in results)
    assert old == before
    # A physically attempted bypass with no measured passage is suppressed,
    # rather than repeatedly gaining credit from advisory reads.
    previous = results[0]
    failed = select_row(row, previous, 4)
    assert failed['action_type'] != 'BYPASS_FORWARD'
    assert 'BYPASS_FORWARD' in failed['ineffective_action_types']


def test_history_preserves_only_measured_outcome_not_cached_handoff_permission():
    row, old = handoff_case()
    selected = select_row(row, old, 5)
    retained = _marvin_proof_selection_history(old)
    assert retained['first_post_action_strafe_progress'] == old['first_post_action_strafe_progress']
    assert not {'local_bypass', 'options', 'bypass_handoff'} & retained.keys()
    assert selected['bypass_handoff']['selected']
