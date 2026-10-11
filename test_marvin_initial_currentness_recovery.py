"""Initial negative dispatch provenance and bounded continuations; offline only."""
import copy
import json
import math
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lidar_perception import LidarPerceptionWorker, MAXIMUM_EFFECTIVE_AGE_SECONDS
from runtime import CognitiveRuntime
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_marvin_lidar_currentness_recovery import recovery_bundle
from test_marvin_local_bypass import mission_bundle
from test_marvin_lateral_avoidance import strafe_runtime

FIXTURE = json.loads((Path(__file__).parent / 'test_fixtures/marvin_21fcdada_initial_currentness.json').read_text())
AGE = FIXTURE['failed_snapshot']['effective_age_seconds']


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('Offline recovery tests cannot access robots or start services')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr('subprocess.Popen', denied)


def initial_refusal(tmp_path, monkeypatch, *, retained=False):
    b, state, _ = recovery_bundle(tmp_path, monkeypatch)
    rt = b[0]
    if retained:
        state['sample'] = copy.deepcopy(FIXTURE['failed_snapshot'])
        rt.lidar_worker.session = state['sample']['producer_session']
        now = state['sample']['received_monotonic_seconds'] + AGE - state['sample']['age_at_receipt_seconds']
        monkeypatch.setattr('runtime.time.monotonic', lambda: now)
    else:
        state['sample'].update(available=False, valid=False, reason='stale',
                              age_at_receipt_seconds=AGE, effective_age_seconds=AGE)
    stamp = FIXTURE['source_frame_stamp_ns']
    b[1].execute_single_marvin_approach_step = lambda **k: pytest.fail('Primitive must not be called at initial refusal')
    result = rt._execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=stamp, dispatch_guard=lambda: True)
    classify = lambda: rt._marvin_initial_approach_stale_lidar(result, stamp=stamp,
        expected_session=rt.lidar_worker.session)
    return b, state, result, classify


def test_retained_306_824ms_initial_boundary_is_unattempted_not_a_JIT_veto(tmp_path, monkeypatch):
    b, _, result, classify = initial_refusal(tmp_path, monkeypatch, retained=True)
    assert result['reason'] == 'marvin_approach_lidar_not_current'
    assert result['initial_currentness_refusal']['acquisition_sequence'] == 16195
    assert result['transport_attempted'] is False and result['delivery_resolved'] is True
    assert result['source_stamp_consumed'] is False and result['approach_result'] is None
    assert 'pre_transport_jit_veto' not in result
    sample = classify()
    assert sample['effective_age_seconds'] == pytest.approx(.30682405101833865)
    assert sample['producer_session'] == FIXTURE['failed_snapshot']['producer_session']
    assert FIXTURE['source_frame_stamp_ns'] not in b[0]._marvin_alignment_consumed_source_frame_stamps
    assert not motions(b[3])
    cert = result['initial_currentness_refusal']
    assert len(json.dumps(cert, allow_nan=False)) < 700
    assert 'points' not in json.dumps(cert) and 'ranges' not in json.dumps(cert)


@pytest.mark.parametrize('fault', ['missing_certificate', 'wrong_stage', 'transport', 'uncertain',
    'consumed', 'consumption_mismatch', 'wrong_stamp', 'float_stamp', 'wrong_certificate_stamp',
    'approach_called', 'transport_result', 'motion', 'actions', 'bool_actions', 'authorized',
    'unresolved', 'stop_failed', 'stop_present', 'wrong_reason', 'certificate_transport',
    'certificate_unresolved', 'certificate_consumed', 'sequence', 'bool_sequence', 'missing_sequence',
    'session', 'certificate_session', 'certificate_sequence', 'worker', 'not_stale',
    'certificate_reason', 'receipt_nan', 'receipt_future', 'age_nan', 'age_negative', 'age_overflow', 'label_only'])
def test_uncertified_initial_refusal_cannot_recover(tmp_path, monkeypatch, fault):
    b, _, r, classify = initial_refusal(tmp_path, monkeypatch)
    c, sample = r['initial_currentness_refusal'], r['lidar']
    if fault == 'missing_certificate': r.pop('initial_currentness_refusal')
    if fault == 'wrong_stage': c['stage'] = 'CLIENT_FINAL_GUARD'
    if fault == 'transport': r['transport_attempted'] = True
    if fault == 'uncertain': r['delivery_uncertain'] = True
    if fault == 'consumed': b[0]._marvin_alignment_consumed_source_frame_stamps.add(FIXTURE['source_frame_stamp_ns'])
    if fault == 'consumption_mismatch': r['source_stamp_consumed'] = True
    if fault == 'wrong_stamp': r['source_frame_stamp_ns'] += 1
    if fault == 'float_stamp': r['source_frame_stamp_ns'] = float(r['source_frame_stamp_ns'])
    if fault == 'wrong_certificate_stamp': c['source_frame_stamp_ns'] += 1
    if fault == 'approach_called': r['approach_result'] = {'ok': False}
    if fault == 'transport_result': r['transport_result'] = {'ok': False}
    if fault == 'motion': r['motion_executed'] = True
    if fault == 'actions': r['actions_executed'] = 1
    if fault == 'bool_actions': r['actions_executed'] = False
    if fault == 'authorized': r['execution_authorized'] = True
    if fault == 'unresolved': r['delivery_resolved'] = False
    if fault == 'stop_failed': r['stop_result'] = {'ok': False}
    if fault == 'stop_present': r['stop_result'] = {'ok': True}  # This exact stage precedes STOP.
    if fault == 'wrong_reason': r['reason'] = 'motion_dispatch_preempted'
    if fault == 'certificate_transport': c['transport_attempted'] = True
    if fault == 'certificate_unresolved': c['delivery_resolved'] = False
    if fault == 'certificate_consumed': c['source_stamp_consumed'] = True
    if fault == 'sequence': sample['acquisition_sequence'] = 0
    if fault == 'bool_sequence': sample['acquisition_sequence'] = True
    if fault == 'missing_sequence': sample.pop('acquisition_sequence')
    if fault == 'session': sample['producer_session'] = 'other'
    if fault == 'certificate_session': c['producer_session'] = 'other'
    if fault == 'certificate_sequence': c['acquisition_sequence'] += 1
    if fault == 'worker': b[0].lidar_worker.running = False
    if fault == 'not_stale': sample['reason'] = 'invalid_geometry'
    if fault == 'certificate_reason': c['lidar_reason'] = 'invalid_geometry'
    if fault == 'receipt_nan': sample['received_monotonic_seconds'] = math.nan
    if fault == 'receipt_future': sample['received_monotonic_seconds'] = time.monotonic() + 1
    if fault == 'age_nan': sample['age_at_receipt_seconds'] = math.nan
    if fault == 'age_negative': sample['age_at_receipt_seconds'] = -1
    if fault == 'age_overflow':
        sample.update(age_at_receipt_seconds=1.7e308, received_monotonic_seconds=-1.7e308)
    if fault == 'label_only': sample['age_at_receipt_seconds'] = 0
    assert classify() is None
    assert not motions(b[3])


@pytest.mark.parametrize('sample', [None, [], 'invalid'])
def test_malformed_initial_snapshot_is_negative_only(tmp_path, monkeypatch, sample):
    b, _, _ = recovery_bundle(tmp_path, monkeypatch)
    b[0].world_model.get_lidar_obstacles = lambda **k: sample
    r = b[0]._execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=FIXTURE['source_frame_stamp_ns'], dispatch_guard=lambda: True)
    assert r['transport_attempted'] is False
    assert b[0]._marvin_initial_approach_stale_lidar(r, stamp=FIXTURE['source_frame_stamp_ns'],
        expected_session=b[0].lidar_worker.session) is None
    assert not motions(b[3])


def initial_mission(tmp_path, monkeypatch, *, side='LEFT', direct=False, fault=None):
    if direct:
        b = make_runtime(tmp_path, monkeypatch, [(0, .6), (0, .6), (0, .5)])
        status = b[2].status
        b[2].status = lambda: dict(status(), motion=dict(status()['motion'], linear_y=0.))
        b[0].forward_interlock = SimpleNamespace(status=lambda: {'active_forward': False, 'pending_forward': False})
    elif side == 'LEFT':
        b, _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=1)
    else:
        from test_marvin_local_bypass import OPEN_LEFT, pursuit_specs
        scenes = [[(x, -y) for x, y in OPEN_LEFT]] * 2 + [None]
        b, _, _ = strafe_runtime(tmp_path, monkeypatch, pursuit_specs([(0, 1.1)] * 2), scenes)
        reader = b[0].world_model.get_lidar_obstacles
        def timestamped(**kwargs):
            b[4][0] += 1000
            return reader(**kwargs)
        b[0].world_model.get_lidar_obstacles = timestamped
    rt, behavior, robot, events, clock = b
    state = {'inside': False, 'expired': False, 'waiting': False, 'refusals': [], 'at_wait': None,
             'waiting_reads': [], 'wait_calls': 0, 'replacement_results': []}
    action = rt._execute_single_marvin_approach
    def invoke(**kwargs):
        state['inside'] = True
        try:
            result = action(**kwargs)
            if result.get('initial_currentness_refusal'):
                state['refusals'].append(copy.deepcopy(result))
            elif state['expired']:
                state['replacement_results'].append(copy.deepcopy(result))
            return result
        finally:
            state['inside'] = False
    rt._execute_single_marvin_approach = invoke
    read = rt.world_model.get_lidar_obstacles
    def read_scan(**kwargs):
        sample = read(**kwargs)
        if state['inside'] and not state['expired']:
            state['expired'] = True
            sample.update(available=False, valid=False, reason='stale', age_at_receipt_seconds=AGE,
                effective_age_seconds=AGE)
            state['failed'] = copy.deepcopy(sample)
        if state['waiting']:
            assert motions(events) == state['at_wait']
            assert robot.status()['motion']['linear_x'] == robot.status()['motion']['linear_y'] == 0
            if fault in {'timeout', 'equal', 'frozen', 'regressed', 'geometry', 'coverage'}:
                if fault in {'timeout', 'equal', 'regressed'}:
                    sample = copy.deepcopy(state['failed'])
                if fault == 'equal': sample.update(valid=True, available=True, reason='fresh',
                    age_at_receipt_seconds=0., effective_age_seconds=0., received_monotonic_seconds=time.monotonic())
                if fault == 'regressed': sample['acquisition_sequence'] -= 1
                if fault == 'frozen': sample.update(valid=False, available=False, reason='stale',
                    age_at_receipt_seconds=AGE, effective_age_seconds=AGE)
                if fault == 'geometry': sample['local_motion_geometry']['valid'] = False
                if fault == 'coverage': sample['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
            state['waiting_reads'].append(copy.deepcopy(sample))
        return sample
    rt.world_model.get_lidar_obstacles = read_scan
    recover = rt._recover_marvin_stationary_lidar
    def wait(**kwargs):
        state['waiting'] = True
        state['wait_calls'] += 1
        state['at_wait'] = list(motions(events))
        if fault == 'stop': robot.stop = lambda: {'ok': False}
        if fault == 'bridge':
            status = robot.status
            robot.status = lambda: dict(status(), motion=dict(status()['motion'], linear_x=.1))
        if fault == 'session': rt.lidar_worker.session = 'changed'
        if fault == 'worker': rt.lidar_worker.running = False
        if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
        if fault == 'deadline': clock[0] += 120_000_000_000
        if fault in {'active', 'pending'}:
            rt.forward_interlock.status = lambda: {'active_forward': fault == 'active', 'pending_forward': fault == 'pending'}
        try:
            return recover(**kwargs)
        finally:
            state['waiting'] = False
    rt._recover_marvin_stationary_lidar = wait
    def reacquire(**kwargs):
        value = behavior.observe_find_marvin_v2()
        preview = value['preview_result']; tracker = preview['opencv_tracker']
        if fault == 'identity': preview['identity_confirmed'] = False
        if fault == 'quality': tracker['quality'] = .799999
        if fault == 'unmatched': tracker['matched'] = False
        if fault in {'duplicate', 'older'}:
            stamp = state['refusals'][0]['source_frame_stamp_ns']
            preview['identity_source_frame_stamp_ns'] = stamp - (fault == 'older')
            preview['source_frame_stamp_ns'] = tracker['source_frame_stamp_ns'] = stamp
        return value
    behavior.reacquire_find_marvin_v2 = reacquire
    return b, state


@pytest.mark.parametrize('mode', ['LEFT', 'RIGHT', 'DIRECT'])
def test_initial_stale_pause_retires_authority_and_requires_independent_replacement(tmp_path, monkeypatch, mode):
    b, state = initial_mission(tmp_path, monkeypatch, side=mode, direct=mode == 'DIRECT')
    result = run(b[0])
    assert result['state'] == 'ARRIVED', result['reason']
    r, = result['stationary_lidar_recovery_history']
    stamp = state['refusals'][0]['source_frame_stamp_ns']
    assert r['entry_point'] == 'initial_approach_currentness'
    assert r['phase'] == ('DIRECT' if mode == 'DIRECT' else 'PASS_OBSTACLE')
    assert r['committed_side'] == (None if mode == 'DIRECT' else mode)
    assert r['stop_ok'] and r['bridge_zero'] and r['new_sequence'] > r['selected_sequence_floor']
    assert r['retired_source_stamp_ns'] == stamp and not r['source_stamp_transport_consumed']
    assert stamp not in b[0]._marvin_alignment_consumed_source_frame_stamps
    assert r['avoidance_count'] == r['watchdog']['total_detour_actions'] == (0 if mode == 'DIRECT' else 1)
    assert r['bypass_count'] == 0
    refused = next(h for h in result['history'] if h['source_frame_stamp_ns'] == stamp)
    assert not refused['motion_executed'] and refused['result']['actions_executed'] == 0
    assert refused['result']['lidar_currentness_failure'] == 'STALE_AT_INITIAL_APPROACH'
    replacement = next(h for h in result['history'] if h['motion_executed'] and h['source_frame_stamp_ns'] > stamp)
    obs = replacement['observation']
    assert obs['identity_source'] == 'gemini_marvin_candidate_selection'
    assert obs['source_frame_stamp_ns'] > obs['identity_source_frame_stamp_ns'] > stamp
    assert obs['opencv_tracker']['quality'] >= .8
    assert replacement['action_lidar_evidence'][1] > r['selected_sequence_floor']
    assert replacement['result']['source_stamp_consumed']
    if mode != 'DIRECT':
        assert replacement['result']['local_detour']['accepted']
        assert result['local_avoidance_actions'] == 2 and result['local_bypass_actions'] == 1
        assert result['local_avoidance_history'][1]['actual_route_occupancy'] is None
    assert result['consecutive_lidar_currentness_failures'] == 0
    before = list(events := b[3])
    replay = b[0]._execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=stamp, dispatch_guard=lambda: True)
    assert not replay['motion_executed'] and events == before
    assert stamp not in b[0]._marvin_alignment_consumed_source_frame_stamps


@pytest.mark.parametrize('fault,reason', [('timeout', 'new_lidar_evidence_timeout'),
    ('equal', 'new_lidar_evidence_timeout'), ('frozen', 'new_lidar_evidence_timeout'),
    ('regressed', 'acquisition_sequence_invalid'), ('geometry', 'new_lidar_evidence_timeout'),
    ('coverage', 'new_lidar_evidence_timeout'), ('session', 'producer_session_changed'),
    ('worker', 'worker_stopped'), ('owner', 'mission_preempted'), ('deadline', 'detour_watchdog_exhausted'),
    ('bridge', 'stop_or_bridge_failed'), ('active', 'interlock_not_stopped'), ('pending', 'interlock_not_stopped'),
    ('stop', 'stop_or_bridge_failed'), ('identity', 'fresh_identity_required'), ('quality', 'fresh_identity_required'),
    ('unmatched', 'fresh_identity_required'), ('duplicate', 'fresh_identity_required'), ('older', 'fresh_identity_required')])
def test_initial_recovery_failure_cannot_authorize_replacement(tmp_path, monkeypatch, fault, reason):
    b, state = initial_mission(tmp_path, monkeypatch, fault=fault)
    result = run(b[0])
    assert result['reason'].endswith(reason), result['reason']
    assert state['wait_calls'] == 1
    assert len(motions(b[3])) == result['local_avoidance_actions'] == 1
    assert result['local_bypass_actions'] == 0 and not state['replacement_results']
    assert state['refusals'][0]['source_frame_stamp_ns'] not in b[0]._marvin_alignment_consumed_source_frame_stamps
    if fault in {'timeout', 'equal', 'frozen', 'geometry', 'coverage'}:
        r = result['stationary_lidar_recovery_history'][0]
        assert r['wait_elapsed_seconds'] <= .600001
    if fault == 'bridge':
        assert result['stationary_lidar_recovery_history'][0]['final_recovery_result'].endswith('bridge_not_stopped')
    if fault == 'deadline':
        assert time.monotonic() - result['detour_context']['watchdog']['detour_started_monotonic'] >= 120


def test_motion_and_recovery_limits_unchanged():
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == .3
    assert CognitiveRuntime.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS == .6
    assert CognitiveRuntime.MAX_CONSECUTIVE_MARVIN_LIDAR_INTERRUPTION_FAILURES == 3
    assert CognitiveRuntime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.


def test_exact_retained_scan_wait_releases_only_new_complete_same_session_scan(tmp_path, monkeypatch):
    b, state, result, classify = initial_refusal(tmp_path, monkeypatch, retained=True)
    failing = classify()
    now = [time.monotonic()]
    monkeypatch.setattr('runtime.time.monotonic', lambda: now[0])
    sleeps = []
    def publish(seconds):
        assert not motions(b[3])
        now[0] += seconds
        sleeps.append(seconds)
        if len(sleeps) == 2:
            fresh = copy.deepcopy(failing)
            fresh.update(acquisition_sequence=16196, reason='fresh', available=True, valid=True,
                age_at_receipt_seconds=.1, received_monotonic_seconds=now[0], effective_age_seconds=.1)
            fresh['source']['stamp_seconds'] += .1  # Synthetic new scan, not a historical continuation.
            state['sample'] = fresh
    monkeypatch.setattr('runtime.time.sleep', publish)
    recovery = b[0]._recover_marvin_stationary_lidar(snapshot=failing,
        expected_session=failing['producer_session'], sequence_floors=[('last_action_jit', failing['producer_session'], 16183)],
        execution_guard=lambda: True, entry_point='initial_approach_currentness', delivery_resolved=True)
    assert recovery['ok'] and recovery['new_sequence'] == 16196
    assert recovery['selected_sequence_floor'] == 16195
    assert recovery['wait_elapsed_seconds'] == pytest.approx(.1)
    assert recovery['stop_ok'] and recovery['bridge_zero']
    assert not motions(b[3]) and not b[0]._marvin_alignment_consumed_source_frame_stamps


def test_frozen_source_cannot_release_wait_even_when_worker_poll_sequences_advance(tmp_path, monkeypatch):
    from world_model import WorldModel
    b, _, _ = recovery_bundle(tmp_path, monkeypatch)
    rt, _, _, events, clock = b
    raw = {'ok': True, 'telemetry': {'available': True, 'age_seconds': 0., 'scan': {
        'frame_id': 'lidar_link', 'stamp_seconds': 900., 'angle_min': -math.pi,
        'angle_increment': 2 * math.pi / 360, 'range_min': .1, 'range_max': 10., 'ranges': [2.] * 360}}}
    worker = LidarPerceptionWorker(rt.world_model, fetch=lambda: copy.deepcopy(raw),
        monotonic=lambda: time.monotonic(), wall_clock=lambda: time.time())
    first = worker.run_once()
    assert first['valid']
    clock[0] += 306_824_051
    frozen = worker.run_once()
    assert not frozen['valid'] and frozen['reason'] == 'stale'
    rt.lidar_worker = SimpleNamespace(running=True, session=worker.session)
    rt.world_model.get_lidar_obstacles = lambda **k: WorldModel.get_lidar_obstacles(rt.world_model, **k)
    def poll(seconds):
        assert not motions(events)
        clock[0] += max(1, round(seconds * 1e9))
        worker.run_once()
    monkeypatch.setattr('runtime.time.sleep', poll)
    result = rt._recover_marvin_stationary_lidar(snapshot=frozen, expected_session=worker.session,
        sequence_floors=[('prior_scan', worker.session, first['acquisition_sequence'])],
        execution_guard=lambda: True, entry_point='initial_approach_currentness', delivery_resolved=True)
    assert result['final_recovery_result'] == 'find_marvin_new_lidar_evidence_timeout'
    assert worker.sequence > frozen['acquisition_sequence']
    assert worker._stamp == 900 and worker._age >= AGE
    assert not motions(events)


def test_initial_recovery_shares_existing_three_unresolved_failure_bound(tmp_path, monkeypatch):
    b, state = initial_mission(tmp_path, monkeypatch, direct=True)
    rt = b[0]
    planning = rt._active_localization_lidar_is_current
    def unresolved(**kwargs):
        session, sample = planning(**kwargs)
        if state['expired']:
            sample.update(available=False, valid=False, reason='stale', effective_age_seconds=AGE)
        return session, sample
    rt._active_localization_lidar_is_current = unresolved
    result = run(rt)
    assert result['reason'] == 'find_marvin_lidar_recovery_exhausted'
    assert result['consecutive_lidar_currentness_failures'] == 3
    assert state['wait_calls'] == 2  # Third refusal exhausts, no third wait/fourth episode.
    assert len(result['stationary_lidar_recovery_history']) == 3
    assert result['stationary_lidar_recovery_history'][0]['entry_point'] == 'initial_approach_currentness'
    assert not motions(b[3])
    assert not rt._marvin_alignment_consumed_source_frame_stamps


def test_retired_stamp_is_rejected_during_replacement_before_fresh_motion(tmp_path, monkeypatch):
    b, state = initial_mission(tmp_path, monkeypatch)
    execute = b[0]._execute_single_marvin_approach
    attempted_old = []
    def replacement(**kwargs):
        if state['refusals'] and not attempted_old:
            old = state['refusals'][0]['source_frame_stamp_ns']
            assert kwargs['source_frame_stamp_ns'] > old
            before = list(b[3])
            refused = execute(**dict(kwargs, source_frame_stamp_ns=old))
            assert refused['reason'] == 'marvin_approach_observation_not_authorized'
            assert not refused['motion_executed'] and b[3] == before
            assert old not in b[0]._marvin_alignment_consumed_source_frame_stamps
            attempted_old.append(old)
        return execute(**kwargs)
    b[0]._execute_single_marvin_approach = replacement
    result = run(b[0])
    assert result['state'] == 'ARRIVED'
    assert attempted_old and result['local_avoidance_actions'] == 2
