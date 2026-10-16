"""Offline late-planning aging replay; fake sensing/transport, real mission gates."""
import ast
import copy
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import pytest
import runtime as runtime_module
from test_find_marvin_closed_loop import make_runtime, motions, run

RETAINED = json.loads((Path(__file__).parent / 'test_fixtures' /
    'marvin_9cfb81ca_late_clearance.json').read_text())
ENTRY = 'late_pre_dispatch_clearance'
AGE = RETAINED['last_wait_snapshot']['effective_age_seconds']


def episode(tmp_path, monkeypatch, *, fault=None, full_prefix=False, baseline=False):
    prefix = [(row['error_px'], .8) for row in RETAINED['completed_prefix']] if full_prefix else [(0, .8)]
    bundle = make_runtime(tmp_path, monkeypatch, prefix + [(12.5, .8)] * 60 + [(0, .5)])
    rt, behavior, robot, events, clock = bundle
    # End after exactly one newly authorized forward step, using only the
    # established synthetic camera-arrival oracle, never live calibration.
    original_specs = behavior.specs
    def specs():
        for index, spec in enumerate(original_specs):
            if len(motions(events)) > len(prefix):
                yield (0, .5)
            else:
                yield spec
    behavior.specs = specs()
    status = robot.status
    robot.status = lambda: dict(status(), motion=dict(status()['motion'], linear_y=0.))
    rt.forward_interlock = SimpleNamespace(status=lambda: {'active_forward': False, 'pending_forward': False})
    state = {'injected': False, 'waiting': False, 'waits': [], 'reads': [], 'reacquired': [], 'prefix': len(prefix)}
    execute = rt._execute_normal_marvin_find_mission
    def mission(*args, **kwargs):
        state['result'] = execute(*args, **kwargs)
        return state['result']
    rt._execute_normal_marvin_find_mission = mission
    read = rt.world_model.get_lidar_obstacles
    def scan(**kwargs):
        sample = read(**kwargs)
        seq = sample['acquisition_sequence']
        sample['bridge_scan_identity'] = {'producer_session': 'offline-native-session',
            'acquisition_sequence': seq, 'source_stamp_ns': 1_000_000_000 + seq * 100_000_000}
        if state['waiting']:
            assert motions(events) == state['at_wait']
            if fault in {'cached', 'cached_then_fresh'} and (fault == 'cached' or len(state['reads']) < 3):
                sample['bridge_scan_identity'] = copy.deepcopy(state['failed']['bridge_scan_identity'])
            if fault == 'native_stamp_only': sample['bridge_scan_identity']['acquisition_sequence'] = state['failed']['bridge_scan_identity']['acquisition_sequence']
            if fault == 'cached_worker': sample['acquisition_sequence'] = state['failed']['acquisition_sequence']
            if fault == 'partial_native': sample['bridge_scan_identity']['source_stamp_ns'] = state['failed']['bridge_scan_identity']['source_stamp_ns']
            if fault == 'regressed_native': sample['bridge_scan_identity']['acquisition_sequence'] = 1
            if fault == 'native_session': sample['bridge_scan_identity']['producer_session'] = 'changed'
            if fault == 'missing_native': sample.pop('bridge_scan_identity')
            if fault == 'frozen': sample.update(available=False, valid=False, reason='stale')
            if fault == 'invalid_geometry': sample['local_motion_geometry']['valid'] = False
            if fault == 'missing_coverage': sample['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
            state['reads'].append(copy.deepcopy(sample))
        return sample
    rt.world_model.get_lidar_obstacles = scan
    planning = rt._active_localization_lidar_is_current
    def plan(**kwargs):
        session, sample = planning(**kwargs)
        state['planning'] = sample
        return session, sample
    rt._active_localization_lidar_is_current = plan
    clearance = rt._marvin_v2_motion_clearance
    def late(tracker, lidar=None, **kwargs):
        inject = (lidar is state.get('planning') and lidar is not None
                  and len(motions(events)) >= len(prefix)
                  and (not state['injected'] or fault == 'exhaustion'))
        if inject:
            state['injected'] = True
            lidar.update(available=True, valid=True, reason='fresh', effective_age_seconds=AGE,
                age_at_receipt_seconds=AGE, received_monotonic_seconds=time.monotonic())
            state['fresh'] = copy.deepcopy(lidar)
            # A real residence-time crossing after planning accepts ~248 ms.
            clock[0] += round((.300001 - AGE) * 1e9)
            if fault == 'missing_native_floor': lidar.pop('bridge_scan_identity')
            state['failed'] = copy.deepcopy(lidar)
        value = clearance(tracker, lidar, **kwargs)
        if inject:
            state['late_result'] = copy.deepcopy(value)
        return value
    rt._marvin_v2_motion_clearance = late
    recover = rt._recover_marvin_stationary_lidar
    def recovery(**kwargs):
        state['waiting'] = True
        state['at_wait'] = list(motions(events))
        state['retired_stamp'] = behavior.stamps[-1]
        if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
        if fault == 'cancel': rt._control_generation += 1
        if fault == 'stop': robot.stop = lambda: {'ok': False}
        if fault == 'obstacle': behavior.unsafe_forward = True
        if fault == 'active_controller': robot.status = lambda: dict(status(), motion={'linear_x': .1, 'linear_y': 0., 'angular_z': 0., 'streaming': True})
        try:
            value = recover(**kwargs)
            state['waits'].append(copy.deepcopy(value))
            return value
        finally:
            state['waiting'] = False
    rt._recover_marvin_stationary_lidar = recovery
    observe = behavior.observe_find_marvin_v2
    def reacquire(**kwargs):
        if state['injected']:
            state['geometry_before_reacquisition'] = copy.deepcopy(rt._marvin_alignment_geometry_history)
        behavior._clear_marvin_v2_tracker_episode()
        value = observe()
        preview = value['preview_result']; tracker = preview.get('opencv_tracker') or {}
        if state['injected']:
            floor = kwargs['minimum_source_frame_stamp_ns']
            if fault == 'identity': preview['identity_confirmed'] = False
            if fault == 'lost': tracker['matched'] = False
            if fault == 'quality': tracker['quality'] = .79
            if fault == 'old_camera': preview['source_frame_stamp_ns'] = tracker['source_frame_stamp_ns'] = floor
            if fault == 'old_semantic': preview['identity_source_frame_stamp_ns'] = floor
            if fault == 'stale_camera': preview['received_monotonic_seconds'] = tracker['received_monotonic_seconds'] = time.monotonic() - 1.01
            if fault == 'frame_mismatch': tracker['source_frame_stamp_ns'] -= 1
            if fault == 'identity_switch': preview['identity_source'] = 'marvin_session_continuity'
            if fault == 'operator_stop': rt.submit_intent({'intent': 'STOP'})
            if fault == 'post_camera_owner': rt._marvin_motion_owner_is_current = lambda: False
            if fault == 'post_camera_native':
                previous = rt.world_model.get_lidar_obstacles
                def changed(**kwargs):
                    sample = previous(**kwargs)
                    sample['bridge_scan_identity']['producer_session'] = 'changed'
                    return sample
                rt.world_model.get_lidar_obstacles = changed
            state['reacquired'].append(copy.deepcopy(preview))
        return value
    behavior.reacquire_find_marvin_v2 = reacquire
    if fault == 'final_jit':
        action = behavior.execute_single_marvin_approach_step
        def jit(**kwargs):
            if state['injected']: behavior.invalid_lidar = True
            return action(**kwargs)
        behavior.execute_single_marvin_approach_step = jit
    if baseline:
        path = os.environ.get('MAYDAY_LATE_LIDAR_BASELINE_FILE')
        if not path: pytest.skip('retained baseline source required')
        cls = next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'CognitiveRuntime')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_normal_marvin_find_mission_locked')
        namespace = dict(vars(runtime_module))
        exec(compile(ast.Module(body=[method], type_ignores=[]), path, 'exec'), namespace)
        rt._execute_normal_marvin_find_mission_locked = MethodType(namespace[method.name], rt)
    return bundle, state


@pytest.mark.parametrize('baseline', [True, False])
@pytest.mark.parametrize('full_prefix', [False, True])
def test_retained_late_expiry_preserves_completed_prefix(tmp_path, monkeypatch, baseline, full_prefix, record_property):
    b, state = episode(tmp_path, monkeypatch, baseline=baseline, full_prefix=full_prefix)
    result = run(b[0])
    assert state['injected']
    assert runtime_module._marvin_complete_lidar_is_current(state['fresh'], b[0].lidar_worker.session)
    if baseline:
        assert result['reason'] == 'find_marvin_lidar_not_current'
        assert not state['waits']
        assert result['actions_executed'] == len(motions(b[3])) == state['prefix']
    else:
        assert result['state'] == 'ARRIVED', result['reason']
        recovery, = result['stationary_lidar_recovery_history']
        assert recovery['entry_point'] == ENTRY
        assert recovery['stop_ok'] and recovery['bridge_zero']
        assert recovery['new_sequence'] > recovery['selected_sequence_floor']
        assert recovery['new_native_identity']['acquisition_sequence'] > recovery['previous_native_identity']['acquisition_sequence']
        assert recovery['new_native_identity']['source_stamp_ns'] > recovery['previous_native_identity']['source_stamp_ns']
        assert recovery['new_camera_source_stamp_ns'] > recovery['retired_source_stamp_ns']
        assert recovery['retired_source_stamp_ns'] not in b[0]._marvin_alignment_consumed_source_frame_stamps
        assert result['actions_executed'] == len(motions(b[3])) == state['prefix'] + 1
        replacement = result['history'][state['prefix']]
        assert replacement['result']['full_step_completed']
        assert replacement['observation']['identity_source'] == 'gemini_marvin_candidate_selection'
        assert state['geometry_before_reacquisition'] == []
        assert replacement['observation']['geometry_continuity']['reason'] == 'not_strict_alignment_geometry'
        obs = replacement['observation']
        assert obs['source_frame_stamp_ns'] > obs['identity_source_frame_stamp_ns'] > recovery['retired_source_stamp_ns']
        assert all(h['result']['full_step_completed'] for h in result['history'][:state['prefix']])
    assert result['completed_forward_actions'] == (8 if full_prefix else 1) + (not baseline)
    assert result['interrupted_forward_attempts'] == 0
    if full_prefix:
        assert [m[0] for m in motions(b[3])[:12]] == ['turn','turn','forward','turn','turn'] + ['forward'] * 7
    assert b[0].get_status()['runtime_state'] == 'IDLE'
    record_property('late_clearance_replay', json.dumps({'baseline': baseline, 'full_prefix': full_prefix,
        'reason': result['reason'], 'actions': result['actions_executed'],
        'forward_completed': result['completed_forward_actions'], 'recoveries': state['waits']}))


def test_cached_then_advancing_scan_can_recover(tmp_path, monkeypatch):
    b, state = episode(tmp_path, monkeypatch, fault='cached_then_fresh')
    result = run(b[0])
    assert result['state'] == 'ARRIVED', result['reason']
    recovery, = state['waits']
    assert recovery['wait_poll_count'] >= 4
    assert recovery['wait_elapsed_seconds'] <= b[0].MARVIN_NEW_LIDAR_TIMEOUT_SECONDS


@pytest.mark.parametrize('fault', ['cached', 'partial_native', 'regressed_native', 'native_session',
    'missing_native', 'missing_native_floor', 'native_stamp_only', 'cached_worker', 'frozen', 'invalid_geometry', 'missing_coverage', 'owner', 'cancel', 'stop',
    'obstacle', 'active_controller', 'identity', 'lost', 'quality', 'old_camera', 'old_semantic',
    'stale_camera', 'frame_mismatch', 'identity_switch', 'operator_stop', 'post_camera_owner',
    'post_camera_native', 'final_jit', 'exhaustion'])
def test_recovery_never_authorizes_unsafe_or_uncertain_motion(tmp_path, monkeypatch, fault):
    b, state = episode(tmp_path, monkeypatch, fault=fault)
    result = run(b[0])
    if fault == 'cancel':
        assert result is None
        result = state['result']
    if fault == 'operator_stop':
        assert result['behavior'] == 'STOP'
        assert len(motions(b[3])) == 1
        return
    assert result['state'] != 'ARRIVED', result
    assert state['waits']
    assert result['actions_executed'] == len(motions(b[3])) == 1
    assert result['completed_forward_actions'] == 1
    assert result['interrupted_forward_attempts'] == 0
    assert not result.get('safe_approach_complete')
    if fault in {'cached', 'cached_worker', 'frozen', 'invalid_geometry', 'missing_coverage'}:
        assert state['waits'][-1]['final_recovery_result'] == 'find_marvin_new_lidar_evidence_timeout'
        assert state['waits'][-1]['wait_elapsed_seconds'] <= b[0].MARVIN_NEW_LIDAR_TIMEOUT_SECONDS + 1e-8
    if fault == 'exhaustion': assert result['reason'] == 'find_marvin_lidar_recovery_exhausted'


@pytest.mark.parametrize('fault', [None, 'fresh', 'at_limit', 'label_only', 'unavailable',
    'invalid_geometry', 'missing_geometry', 'coverage', 'wrong_frame', 'missing_points',
    'invalid_point', 'obstacle', 'rear_obstacle', 'wrong_session', 'sequence', 'invalid_age',
    'future_receipt', 'missing_receipt', 'missing_age'])
def test_actual_retained_scan_is_recoverable_only_when_expired_and_otherwise_complete(
        tmp_path, monkeypatch, fault):
    b = make_runtime(tmp_path, monkeypatch, [(12.5, .8)])
    rt, _, _, events, clock = b
    sample = copy.deepcopy(RETAINED['last_wait_snapshot'])
    sample['producer_session'] = rt.lidar_worker.session
    # Preserve measured receipt-age + residence, rebased to the fake host clock.
    sample['received_monotonic_seconds'] = time.monotonic() - (AGE - sample['age_at_receipt_seconds'])
    if fault not in {'fresh', 'at_limit'}: clock[0] += round((.300001 - AGE) * 1e9)
    if fault == 'at_limit':
        monkeypatch.setattr(runtime_module.time, 'monotonic', lambda: .30)
        sample.update(received_monotonic_seconds=0., age_at_receipt_seconds=0.)
    if fault == 'label_only': sample['reason'] = 'stale'
    if fault == 'unavailable': sample.update(available=False, valid=False)
    if fault == 'invalid_geometry': sample['local_motion_geometry']['valid'] = False
    if fault == 'missing_geometry': sample.pop('local_motion_geometry')
    if fault == 'coverage': sample['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'wrong_frame': sample['local_motion_geometry']['frame_id'] = 'unknown'
    if fault == 'missing_points': sample['local_motion_geometry'].pop('points')
    if fault == 'invalid_point': sample['local_motion_geometry']['points'].append({'x_m': float('nan'), 'y_m': 1.})
    if fault in {'obstacle', 'rear_obstacle'}:
        sample['local_motion_geometry']['points'].append({'x_m': -.2 if fault == 'rear_obstacle' else .35, 'y_m': .1})
    if fault == 'wrong_session': sample['producer_session'] = 'changed'
    if fault == 'sequence': sample['acquisition_sequence'] = -1
    if fault == 'invalid_age': sample['age_at_receipt_seconds'] = float('nan')
    if fault == 'future_receipt': sample['received_monotonic_seconds'] = time.monotonic() + 1
    if fault == 'missing_receipt': sample.pop('received_monotonic_seconds')
    if fault == 'missing_age': sample.pop('age_at_receipt_seconds')
    before = copy.deepcopy(sample)
    result = rt._marvin_v2_motion_clearance({'bbox': {}}, sample)
    recovered = result.get('expired_planning_snapshot')
    assert (recovered is not None) is (fault is None)
    if recovered:
        assert result['ok'] is False and result['arrived_at_marvin'] is False
        assert recovered['effective_age_seconds'] > .3
        assert recovered['available'] is recovered['valid'] is False
        assert recovered['bridge_scan_identity'] == sample['bridge_scan_identity']
        assert recovered['received_monotonic_seconds'] == sample['received_monotonic_seconds']
    assert repr(sample) == repr(before)  # No restamping, flag reset or evidence mutation.
    assert not events
