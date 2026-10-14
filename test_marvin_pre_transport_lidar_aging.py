"""Retained metadata replay and synthetic currentness continuations, offline only."""
import copy
import json
import math
import socket
from pathlib import Path

import pytest

from lidar_perception import read_lidar_state
from marvin_blocked_wait import pre_transport_jit_veto_evidence
from marvin_detour_watchdog import Passage, MATERIAL_GEOMETRY_CHANGE_M
from marvin_obstacle_phases import plan_phase_action
from test_marvin_avoidance_planner_progress import scan
from test_marvin_local_bypass import ASSOCIATION, OPEN_LEFT, mission_bundle
from test_find_marvin_closed_loop import motions, run
from test_marvin_lidar_currentness_recovery import recovery_bundle
from local_motion_safety_envelope import _distance_to_segment, LOCAL_LIDAR_PROTECTED_RADIUS_M

FIXTURE = json.loads((Path(__file__).parent / 'test_fixtures/marvin_0302257b_currentness.json').read_text())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Offline investigation cannot access robots or start services')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr('subprocess.Popen', denied)


def test_retained_switch_has_equal_reported_geometry_and_expired_bypass_evidence():
    plan, jit = FIXTURE['planning_selection'], FIXTURE['jit_selection']
    assert plan['route'] == jit['route']
    assert (plan['action_type'], jit['action_type']) == ('BYPASS_FORWARD', 'STRAFE_LEFT')
    assert jit['local_bypass']['local_bypass_reason'] == 'stale'
    sample = FIXTURE['jit_snapshot_metadata']
    assert 'points' not in sample['local_motion_geometry']  # Not a full raw-scan replay.
    state = read_lidar_state(sample, expected_session=sample['producer_session'],
                             now=FIXTURE['terminal_monotonic_seconds'])
    assert state['reason'] == 'stale' and state['effective_age_seconds'] > .3


def test_retained_nearest_return_matches_capsule_overlap_without_inventing_missing_scan():
    route = FIXTURE['initial_route']
    theta = math.radians(route['marvin_bearing_deg'])
    horizon = route['route_lookahead_m']
    gap = _distance_to_segment(route['blocking_obstacle_x_m'], route['blocking_obstacle_y_m'],
                               horizon * math.cos(theta), horizon * math.sin(theta))
    assert gap == pytest.approx(route['blocking_obstacle_centerline_clearance_m'])
    assert LOCAL_LIDAR_PROTECTED_RADIUS_M - gap == pytest.approx(route['blocking_obstacle_overlap_m'])
    assert gap < LOCAL_LIDAR_PROTECTED_RADIUS_M == .45
    assert route['route_occupancy'] == 20  # A reported count, not 20 reconstructed points.


@pytest.mark.parametrize('index', [0, 1])
def test_retained_completed_strafe_uses_center_separation_not_centerline_clearance(index):
    row = FIXTURE['strafe_outcomes'][index]
    before = Passage.from_selection(row['before_selection'])
    after = Passage.from_native(row['after_association'], 'LEFT')
    gain = after.center_separation_m - before.center_separation_m
    assert gain > MATERIAL_GEOMETRY_CHANGE_M == .02
    assert row['reported_progress']['progress']['centerline_clearance_improvement_m'] < .02
    if index == 1:
        assert after.pass_safe and not before.pass_safe


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_clock_crossing_inside_phase_planner_is_currentness_not_changed_geometry(monkeypatch, side):
    # This is a synthetic complete scan. The retained raw scan is unavailable.
    points = OPEN_LEFT if side == 'LEFT' else [(x, -y) for x, y in OPEN_LEFT]
    sample = scan(points)
    now = [sample['received_monotonic_seconds']]
    monkeypatch.setattr('lidar_perception.time.monotonic', lambda: now[0])
    import marvin_obstacle_phases as phases
    native_bypass = phases.plan_local_bypass
    def expires(*args, **kwargs):
        now[0] += .301
        return native_bypass(*args, **kwargs)
    monkeypatch.setattr(phases, 'plan_local_bypass', expires)
    result = plan_phase_action(sample, ASSOCIATION, expected_session='test',
                               allow_strafe=True, committed_side=side)
    assert result['route']['valid']
    assert result['action_type'] == 'STRAFE_' + side
    assert result['local_bypass']['local_bypass_reason'] == 'stale'
    assert not read_lidar_state(sample, expected_session='test')['valid']


def refusal_bundle(tmp_path, monkeypatch, consumed=False):
    b, state, _ = recovery_bundle(tmp_path, monkeypatch)
    rt = b[0]
    b[4][0] += 328_185_000  # Consistent native receipt-clock evidence, not just a stale label.
    stamp = 1791676153628433231
    sample = state['sample']
    association = dict(ok=True, producer_session=sample['producer_session'],
                       acquisition_sequence=sample['acquisition_sequence'])
    result = dict(ok=False, motion_executed=False, actions_executed=0,
        execution_authorized=False, reason='marvin_local_bypass_jit_veto',
        source_frame_stamp_ns=stamp, source_stamp_consumed=consumed,
        local_detour={'local_bypass': {'local_bypass_reason': 'stale'}},
        pre_transport_jit_veto=pre_transport_jit_veto_evidence(sample, association))
    if consumed:
        rt._marvin_alignment_consumed_source_frame_stamps.add(stamp)
    def classify():
        return rt._marvin_pre_transport_stale_lidar(result, stamp=stamp,
            expected_session='offline-v2-session', planning_sequence=41)
    return b, state, result, classify


@pytest.mark.parametrize('consumed', [False, True])
def test_certified_zero_transport_stale_scan_is_recoverable_without_reusing_authority(tmp_path, monkeypatch, consumed):
    b, state, result, classify = refusal_bundle(tmp_path, monkeypatch, consumed)
    sample = classify()
    assert sample['reason'] == 'stale'
    assert sample['acquisition_sequence'] == 42
    assert not motions(b[3])
    assert (result['source_frame_stamp_ns'] in b[0]._marvin_alignment_consumed_source_frame_stamps) is consumed


@pytest.mark.parametrize('fault', ['transport', 'uncertain', 'wrong_stamp', 'float_stamp',
    'consumption', 'association_sequence', 'association_session', 'sequence_regression',
    'missing_points', 'nan_point', 'stop_failed', 'missing_certificate', 'not_stale',
    'malformed_selection', 'malformed_bypass', 'invalid_geometry', 'invalid_receipt',
    'invalid_age_at_receipt', 'stale_label_on_current_scan'])
def test_uncertified_refusal_cannot_enter_currentness_recovery(tmp_path, monkeypatch, fault):
    b, _, result, classify = refusal_bundle(tmp_path, monkeypatch)
    cert = result['pre_transport_jit_veto']
    if fault == 'transport': cert['transport_attempted'] = True
    if fault == 'uncertain': result['delivery_uncertain'] = True
    if fault == 'wrong_stamp': result['source_frame_stamp_ns'] += 1
    if fault == 'float_stamp': result['source_frame_stamp_ns'] = float(result['source_frame_stamp_ns'])
    if fault == 'consumption': result['source_stamp_consumed'] = True
    if fault == 'association_sequence': cert['target_association']['acquisition_sequence'] += 1
    if fault == 'association_session': cert['target_association']['producer_session'] = 'other'
    if fault == 'sequence_regression': cert['lidar_snapshot']['acquisition_sequence'] = 40
    if fault == 'missing_points': cert['lidar_snapshot']['local_motion_geometry'].pop('points')
    if fault == 'nan_point': cert['lidar_snapshot']['local_motion_geometry']['points'][0]['x_m'] = math.nan
    if fault == 'stop_failed': result['stop_result'] = {'ok': False}
    if fault == 'missing_certificate': result.pop('pre_transport_jit_veto')
    if fault == 'not_stale': result['local_detour']['local_bypass']['local_bypass_reason'] = 'local_bypass_corridor_blocked'
    if fault == 'malformed_selection': result['local_detour'] = 'invalid'
    if fault == 'malformed_bypass': result['local_detour']['local_bypass'] = 'invalid'
    if fault == 'invalid_geometry': cert['lidar_snapshot']['local_motion_geometry']['valid'] = False
    if fault == 'invalid_receipt': cert['lidar_snapshot']['received_monotonic_seconds'] = math.nan
    if fault == 'invalid_age_at_receipt': cert['lidar_snapshot']['age_at_receipt_seconds'] = math.nan
    if fault == 'stale_label_on_current_scan': cert['lidar_snapshot']['received_monotonic_seconds'] += .328185
    assert classify() is None
    assert not motions(b[3])


def aging_mission(tmp_path, monkeypatch, fault=None):
    from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
    b, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0, .6)] * 8 + [(0, .5)], [LEFT_OPEN])
    rt, behavior, robot, events, clock = b
    native_select = rt._select_marvin_detour_for_context
    native_action = behavior.execute_guarded_marvin_lateral_step
    state = {'inside_jit': False, 'expired': False, 'waiting': False,
             'source_stamp': None, 'motions_at_recovery': None}
    def action(**kwargs):
        state['inside_jit'] = True
        try:
            result = native_action(**kwargs)
            if result.get('pre_transport_jit_veto'):
                state['source_stamp'] = behavior._marvin_v2_tracker_episode['last_tracker_source_frame_stamp_ns']
            return result
        finally:
            state['inside_jit'] = False
    behavior.execute_guarded_marvin_lateral_step = action
    def expire(*args, **kwargs):
        if state['inside_jit'] and not state['expired']:
            state['expired'] = True
            clock[0] += 301_000_000
            state['fault_applied'] = fault
            if fault == 'timeout': behavior.freeze_lidar = True
            if fault == 'session': rt.lidar_worker.session = 'changed'
            if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
            if fault == 'worker': rt.lidar_worker.running = False
            if fault == 'deadline': clock[0] += 120_000_000_000
            if fault == 'stop': robot.stop = lambda: {'ok': False}
        return native_select(*args, **kwargs)
    monkeypatch.setattr(rt, '_select_marvin_detour_for_context', expire)
    recovery = rt._recover_marvin_stationary_lidar
    def recover(**kwargs):
        state['waiting'] = True
        state['motions_at_recovery'] = list(motions(events))
        if fault == 'stop': robot.stop = lambda: {'ok': False}
        if fault == 'session': rt.lidar_worker.session = 'changed'
        if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
        if fault == 'worker': rt.lidar_worker.running = False
        if fault == 'deadline': clock[0] += 120_000_000_000
        try: return recovery(**kwargs)
        finally: state['waiting'] = False
    rt._recover_marvin_stationary_lidar = recover
    read = rt.world_model.get_lidar_obstacles
    def scan_read(**kwargs):
        sample = read(**kwargs)
        if state['waiting']:
            assert motions(events) == state['motions_at_recovery']
            assert robot.status()['motion']['linear_x'] == robot.status()['motion']['linear_y'] == 0
            if fault == 'timeout':
                sample.update(available=False, valid=False, reason='stale')
        return sample
    rt.world_model.get_lidar_obstacles = scan_read
    behavior.reacquire_find_marvin_v2 = lambda **k: behavior.observe_find_marvin_v2()
    return b, state


def test_stale_preliminary_bypass_retires_stamp_then_reacquires_and_replans(tmp_path, monkeypatch):
    b, state = aging_mission(tmp_path, monkeypatch)
    result = run(b[0])
    # The metric-bypass stale certificate remains covered above. The actual
    # selected lateral primitive has no such bypass certificate: aging here
    # must stop conservatively rather than fabricate a recovery authorization.
    assert result['state'] == 'BLOCKED' and result['reason'] == 'marvin_local_detour_jit_veto'
    assert state['expired'] and not state['waiting']
    assert not result['stationary_lidar_recovery_history'] and not motions(b[3])
    refused = result['history'][0]
    assert not refused['motion_executed'] and refused['result']['actions_executed'] == 0
    assert refused['result']['pre_transport_jit_veto']['transport_attempted'] is False
    assert refused['result']['lateral_step']['lateral_result'] is None
    assert result['local_avoidance_actions'] == result['local_bypass_actions'] == 0
    assert result['stop_result']['ok']
    before = list(b[3])
    replay = b[0]._execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=refused['source_frame_stamp_ns'], dispatch_guard=lambda: True)
    assert not replay['motion_executed'] and list(b[3]) == before


@pytest.mark.parametrize('fault,reason', [('timeout', 'find_marvin_new_lidar_evidence_timeout'),
    ('session', 'find_marvin_lidar_producer_session_changed'),
    ('stop', 'find_marvin_stop_or_bridge_failed'),
    ('owner', 'find_marvin_mission_preempted'),
    ('worker', 'find_marvin_lidar_worker_stopped'),
    ('deadline', 'find_marvin_detour_watchdog_exhausted')])
def test_nonrecoverable_preliminary_aging_remains_terminal(tmp_path, monkeypatch, fault, reason):
    b, state = aging_mission(tmp_path, monkeypatch, fault)
    result = run(b[0])
    assert state['expired'] and state['fault_applied'] == fault
    expected = reason if fault in {'stop', 'deadline'} else 'marvin_local_detour_jit_veto'
    assert result['reason'] == expected
    assert result['local_avoidance_actions'] == result['local_bypass_actions'] == 0
    assert not motions(b[3]) and not result['stationary_lidar_recovery_history']
    if fault == 'stop':
        assert not result['stop_result']['ok']
    else:
        assert result['stop_result']['ok']
