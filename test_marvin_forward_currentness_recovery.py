"""Offline primitive-boundary freshness recovery, with no network or hardware."""
import copy
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_find_marvin_closed_loop import make_runtime, motions, run

FIXTURE = json.loads((Path(__file__).parent / 'test_fixtures/marvin_69555d95_forward_currentness.json').read_text())
AGE = FIXTURE['failed_action_lidar_evidence']['effective_age_seconds']


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('Offline tests must not access robots or start services')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr('subprocess.Popen', denied)


def mission_bundle(tmp_path, monkeypatch, fault=None):
    # Retained alignment/error sequence; independent synthetic camera arrival
    # oracle from the existing lifecycle fixture, never live arrival calibration.
    b = make_runtime(tmp_path, monkeypatch,
        [(69.5, .8), (69.5, .8), (18.5, .8)] + [(0, .8)] * 6 + [(0, .5)])
    rt, behavior, robot, events, clock = b
    status = robot.status
    robot.status = lambda: dict(status(), motion=dict(status()['motion'], linear_y=0.))
    rt.forward_interlock = SimpleNamespace(status=lambda: {'active_forward': False, 'pending_forward': False})
    state = {'primitive': False, 'expired': False, 'waiting': False,
             'refusals': [], 'waits': [], 'waiting_reads': []}
    mission = rt._execute_normal_marvin_find_mission
    def execute_mission(*args, **kwargs):
        state['mission_result'] = mission(*args, **kwargs)
        return state['mission_result']
    rt._execute_normal_marvin_find_mission = execute_mission
    primitive = behavior.execute_single_marvin_approach_step
    def invoke(**kwargs):
        state['primitive'] = True
        try:
            return primitive(**kwargs)
        finally:
            state['primitive'] = False
    behavior.execute_single_marvin_approach_step = invoke
    read = rt.world_model.get_lidar_obstacles
    def scan(**kwargs):
        sample = read(**kwargs)
        seq = sample['acquisition_sequence']
        sample['bridge_scan_identity'] = {'producer_session': 'synthetic-native-session',
            'acquisition_sequence': seq, 'source_stamp_ns': 1_000_000_000 + seq * 100_000_000}
        if state['primitive'] and not state['expired']:
            state['expired'] = True
            sample.update(available=False, valid=False, reason='stale',
                age_at_receipt_seconds=AGE, effective_age_seconds=AGE)
            state['failed'] = copy.deepcopy(sample)
        if state['waiting']:
            assert motions(events) == state['at_wait']  # No movement while acquiring evidence.
            state['waiting_reads'].append(copy.deepcopy(sample))
            if fault in {'cached', 'cached_then_fresh'} and (fault == 'cached' or len(state['waiting_reads']) <= 3):
                sample['bridge_scan_identity'] = copy.deepcopy(state['failed']['bridge_scan_identity'])
            if fault == 'frozen':
                sample.update(available=False, valid=False, reason='stale',
                    age_at_receipt_seconds=AGE, effective_age_seconds=AGE)
            if fault == 'unavailable':
                sample.update(available=False, valid=False, reason='sensor_unavailable', acquisition_sequence=None)
            if fault == 'geometry': sample['local_motion_geometry']['valid'] = False
            if fault == 'coverage': sample['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
            if fault == 'native_session': sample['bridge_scan_identity']['producer_session'] = 'changed'
            if fault == 'native_regression': sample['bridge_scan_identity']['acquisition_sequence'] = 0
            if fault == 'native_partial': sample['bridge_scan_identity']['source_stamp_ns'] = state['failed']['bridge_scan_identity']['source_stamp_ns']
            if fault == 'missing_native': sample.pop('bridge_scan_identity')
            if fault == 'worker_session': sample['producer_session'] = 'changed'
        return sample
    rt.world_model.get_lidar_obstacles = scan
    execute = rt._execute_single_marvin_approach
    def action(**kwargs):
        result = execute(**kwargs)
        if result.get('reason') == 'marvin_single_approach_lidar_not_trusted':
            state['refusals'].append(copy.deepcopy(result))
        return result
    rt._execute_single_marvin_approach = action
    recover = rt._recover_marvin_stationary_lidar
    def recovery(**kwargs):
        state['waiting'] = True
        state['at_wait'] = list(motions(events))
        if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
        if fault == 'cancel': rt._control_generation += 1
        if fault == 'stop': robot.stop = lambda: {'ok': False}
        if fault == 'obstacle': behavior.unsafe_forward = True
        try:
            value = recover(**kwargs)
            state['waits'].append(copy.deepcopy(value))
            return value
        finally:
            state['waiting'] = False
    rt._recover_marvin_stationary_lidar = recovery
    def reacquire(**kwargs):
        value = behavior.observe_find_marvin_v2()
        preview = value['preview_result']
        if fault == 'identity': preview['identity_confirmed'] = False
        if fault == 'unmatched': preview['opencv_tracker']['matched'] = False
        if fault == 'quality': preview['opencv_tracker']['quality'] = .79
        if fault == 'duplicate_camera':
            old = state['refusals'][0]['source_frame_stamp_ns']
            preview['source_frame_stamp_ns'] = preview['identity_source_frame_stamp_ns'] = old
            preview['opencv_tracker']['source_frame_stamp_ns'] = old
        if fault == 'post_camera_owner': rt._marvin_motion_owner_is_current = lambda: False
        if fault == 'post_camera_cancel': rt._control_generation += 1
        if fault == 'post_camera_native_session':
            previous = rt.world_model.get_lidar_obstacles
            def changed(**kwargs):
                sample = previous(**kwargs)
                sample['bridge_scan_identity']['producer_session'] = 'changed'
                return sample
            rt.world_model.get_lidar_obstacles = changed
        return value
    behavior.reacquire_find_marvin_v2 = reacquire
    return b, state


@pytest.mark.parametrize('fault', [None, 'cached_then_fresh'])
def test_retained_302ms_primitive_refusal_recovers_without_replaying_alignment(tmp_path, monkeypatch, fault):
    b, state = mission_bundle(tmp_path, monkeypatch, fault)
    result = run(b[0])
    assert result['state'] == 'ARRIVED', result['reason']
    assert motions(b[3])[:2] == [('turn', 'RIGHT', .25, .5)] * 2
    assert all(m[0] == 'forward' and m[1:] == (.1, .5) for m in motions(b[3])[2:])
    assert result['actions_executed'] == len(motions(b[3]))
    refusal, = state['refusals']
    assert refusal['actions_executed'] == 0 and refusal['source_stamp_consumed'] is True
    assert refusal['approach_result']['forward_safety']['reason'] == 'stale'
    assert refusal['source_frame_stamp_ns'] in b[0]._marvin_alignment_consumed_source_frame_stamps
    recovery, = result['stationary_lidar_recovery_history']
    assert recovery['entry_point'] == 'pre_dispatch_forward_currentness'
    assert recovery['new_sequence'] > recovery['selected_sequence_floor']
    assert recovery['new_native_identity']['source_stamp_ns'] > recovery['previous_native_identity']['source_stamp_ns']
    assert recovery['new_camera_source_stamp_ns'] > refusal['source_frame_stamp_ns']
    assert recovery['wait_elapsed_seconds'] <= b[0].MARVIN_NEW_LIDAR_TIMEOUT_SECONDS
    if fault == 'cached_then_fresh': assert recovery['wait_poll_count'] >= 4
    assert b[1].identity_sources[3] == 'gemini_marvin_candidate_selection'
    assert b[0].get_status()['runtime_state'] == 'IDLE'


@pytest.mark.parametrize('fault', ['cached', 'frozen', 'unavailable', 'geometry', 'coverage',
    'native_session', 'native_regression', 'native_partial', 'missing_native', 'worker_session',
    'owner', 'cancel', 'stop', 'obstacle', 'identity', 'unmatched', 'quality', 'duplicate_camera',
    'post_camera_owner', 'post_camera_cancel', 'post_camera_native_session'])
def test_unproven_replacement_stays_stopped_and_preserves_alignment(tmp_path, monkeypatch, fault):
    b, state = mission_bundle(tmp_path, monkeypatch, fault)
    result = run(b[0])
    if fault in {'cancel', 'post_camera_cancel'}:
        # Runtime deliberately suppresses publishing an old-generation result.
        assert result is None
        result = state['mission_result']
        assert result['state'] == 'STOPPED'
    assert result['state'] != 'ARRIVED' and not result.get('safe_approach_complete')
    assert motions(b[3]) == [('turn', 'RIGHT', .25, .5)] * 2
    assert result['actions_executed'] == 2
    assert len(state['refusals']) == 1
    assert result['reason'] != 'marvin_single_approach_lidar_not_trusted', result['reason']
    if fault in {'cached', 'frozen', 'unavailable', 'geometry', 'coverage'}:
        wait, = state['waits']
        assert wait['final_recovery_result'] == 'find_marvin_new_lidar_evidence_timeout'
        assert wait['wait_elapsed_seconds'] <= b[0].MARVIN_NEW_LIDAR_TIMEOUT_SECONDS + 1e-8
    if fault == 'obstacle':
        assert result['blocked_wait_history'], result
        assert 'stale' not in result['reason']


@pytest.mark.parametrize('fault', [None, 'transport', 'unresolved', 'uncertain', 'outer_uncertain',
    'moved', 'actions', 'bool_actions', 'unconsumed', 'consumption_mismatch', 'wrong_stamp', 'float_stamp',
    'stop', 'interrupted', 'completed', 'forward_result', 'transport_result', 'obstacle', 'fresh',
    'label_only', 'age_nan', 'age_negative', 'receipt_future', 'sequence', 'session', 'worker'])
def test_primitive_negative_provenance_never_hides_delivery_or_unknown_faults(tmp_path, monkeypatch, fault):
    b, state = mission_bundle(tmp_path, monkeypatch)
    run(b[0])
    rt = b[0]
    value = copy.deepcopy(state['refusals'][0])
    approach = value['approach_result']; sample = approach['lidar']
    stamp = value['source_frame_stamp_ns']
    if fault == 'transport': approach['transport_attempted'] = True
    if fault == 'unresolved': approach['delivery_resolved'] = False
    if fault == 'uncertain': approach['delivery_uncertain'] = True
    if fault == 'outer_uncertain': value['delivery_uncertain'] = True
    if fault == 'moved': value['motion_executed'] = True
    if fault == 'actions': value['actions_executed'] = 1
    if fault == 'bool_actions': value['actions_executed'] = False
    if fault == 'unconsumed': value['source_stamp_consumed'] = False
    if fault == 'consumption_mismatch': rt._marvin_alignment_consumed_source_frame_stamps.clear()
    if fault == 'wrong_stamp': value['source_frame_stamp_ns'] += 1
    if fault == 'float_stamp': value['source_frame_stamp_ns'] = float(stamp)
    if fault == 'stop': value['stop_result'] = {'ok': False}
    if fault == 'interrupted': value['interrupted'] = True
    if fault == 'completed': value['full_step_completed'] = True
    if fault == 'forward_result': approach['forward_result'] = {'ok': False}
    if fault == 'transport_result': approach['transport_result'] = {'ok': False}
    if fault == 'obstacle': approach['forward_safety']['reason'] = 'protected_envelope_occupied'
    if fault == 'fresh': sample['reason'] = 'fresh'
    # Rebase only for explicit timing faults: the completed fake mission clock
    # has otherwise advanced; no stale label can substitute for measured aging.
    if fault == 'label_only': sample.update(received_monotonic_seconds=time.monotonic(), age_at_receipt_seconds=0.)
    if fault == 'age_nan': sample['age_at_receipt_seconds'] = float('nan')
    if fault == 'age_negative': sample['age_at_receipt_seconds'] = -1
    if fault == 'receipt_future': sample['received_monotonic_seconds'] = time.monotonic() + 1
    if fault == 'sequence': sample['acquisition_sequence'] = 0
    if fault == 'session': sample['producer_session'] = 'changed'
    if fault == 'worker': rt.lidar_worker.running = False
    before = list(b[3])
    found = rt._marvin_primitive_approach_stale_lidar(value, stamp=stamp,
        expected_session=rt.lidar_worker.session, planning_sequence=state['failed']['acquisition_sequence'] - 1)
    assert (found is not None) is (fault is None)
    assert b[3] == before  # Classification itself never STOPs, polls or dispatches.
