"""Offline stationary currentness recovery; producer/robot/provider are doubles."""
import copy
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import CognitiveRuntime, _marvin_complete_lidar_is_current
from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_marvin_local_bypass import mission_bundle, pursuit_specs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError("No live transport/services permitted")
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr("subprocess.Popen", denied)


def recovery_bundle(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .6)] * 20 + [(0, .5)])
    rt, behavior, robot, events, clock = bundle
    read_status = robot.status
    def status():
        result = read_status()
        result['motion']['linear_y'] = 0.
        return result
    robot.status = status
    interlock = {'active_forward': False, 'pending_forward': False}
    rt.forward_interlock = SimpleNamespace(status=lambda: dict(interlock))
    rt._marvin_motion_owner_is_current = lambda: True
    behavior.freeze_lidar = True
    behavior.sequence = 42
    fresh = behavior.lidar()
    sample = copy.deepcopy(fresh)
    sample.update(reason='stale', effective_age_seconds=.328185, valid=False)
    state = {'sample': sample, 'sleeps': 0, 'publish': True, 'on_sleep': None, 'guard': True}
    rt.world_model.get_lidar_obstacles = lambda **k: copy.deepcopy(state['sample'])
    def sleep(seconds):
        assert not motions(events)
        assert robot.status()['motion'] == {'linear_x': 0., 'linear_y': 0., 'angular_z': 0., 'streaming': False}
        clock[0] += max(1, round(seconds * 1e9))
        state['sleeps'] += 1
        if state['on_sleep']:
            state['on_sleep']()
        if state['publish'] and state['sleeps'] == 2:
            state['sample'] = copy.deepcopy(fresh)
            state['sample']['acquisition_sequence'] = 43
    monkeypatch.setattr('runtime.time.sleep', sleep)
    return bundle, state, interlock


def recover(bundle, state, *, floors=None):
    rt = bundle[0]
    return rt._recover_marvin_stationary_lidar(snapshot=state['sample'],
        expected_session='offline-v2-session',
        sequence_floors=floors or [('last_action_jit', 'offline-v2-session', 41)],
        execution_guard=lambda: state['guard'], entry_point='planning_reassessment', delivery_resolved=True)


@pytest.mark.parametrize('failure', ['stale', 'unavailable', 'geometry', 'coverage', 'invalid'])
def test_transient_currentness_waits_for_complete_new_scan(tmp_path, monkeypatch, failure):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    if failure == 'unavailable': state['sample']['available'] = False
    if failure == 'geometry': state['sample']['local_motion_geometry']['valid'] = False
    if failure == 'coverage': state['sample']['local_motion_geometry']['sectors']['left']['available'] = False
    if failure == 'invalid': state['sample']['valid'] = False
    result = recover(bundle, state)
    assert result['ok'] and result['new_sequence'] == 43
    assert result['selected_sequence_floor'] == 42
    assert result['sequence_floor_source'] == 'failing_snapshot'
    assert result['wait_poll_count'] == 3 and result['wait_elapsed_seconds'] == pytest.approx(.1)
    assert result['stop_ok'] and result['bridge_zero']
    assert result['motion_executed'] is result['source_stamp_consumed'] is False
    assert not motions(bundle[3]) and not bundle[0]._marvin_alignment_consumed_source_frame_stamps
    assert 'points' not in json.dumps(result) and len(json.dumps(result)) < 1600


@pytest.mark.parametrize('fault,reason', [
    ('bridge_x', 'bridge_not_stopped'), ('bridge_y', 'bridge_not_stopped'),
    ('bridge_yaw', 'bridge_not_stopped'), ('streaming', 'bridge_not_stopped'),
    ('ros', 'bridge_not_stopped'), ('not_ready', 'bridge_not_stopped'),
    ('active', 'interlock_not_stopped'), ('pending', 'interlock_not_stopped'),
    ('unknown_interlock', 'interlock_unknown'), ('stop', 'stop_unconfirmed'),
    ('owner', 'mission_preempted'), ('generation', 'mission_preempted'),
    ('worker', 'worker_stopped'), ('session', 'producer_session_changed'),
    ('regression', 'acquisition_sequence_invalid'), ('bool_sequence', 'acquisition_sequence_invalid'),
    ('missing_sequence', 'acquisition_sequence_invalid'), ('missing_session', 'evidence_invalid'),
])
def test_uncertified_stationary_or_producer_state_never_recovers(tmp_path, monkeypatch, fault, reason):
    bundle, state, interlock = recovery_bundle(tmp_path, monkeypatch)
    rt, _, robot, events, _ = bundle
    status = robot.status
    if fault.startswith('bridge_') or fault in {'streaming', 'ros', 'not_ready'}:
        def bad_status():
            result = status()
            if fault.startswith('bridge_'):
                result['motion'][{'bridge_x': 'linear_x', 'bridge_y': 'linear_y', 'bridge_yaw': 'angular_z'}[fault]] = .1
            if fault == 'streaming': result['motion']['streaming'] = True
            if fault == 'ros': result['ros_ready'] = False
            if fault == 'not_ready': result['status'] = 'BUSY'
            return result
        robot.status = bad_status
    if fault == 'active': interlock['active_forward'] = True
    if fault == 'pending': interlock['pending_forward'] = True
    if fault == 'unknown_interlock': rt.forward_interlock = None
    if fault == 'stop': robot.stop = lambda: {'ok': False}
    if fault == 'owner': rt._marvin_motion_owner_is_current = lambda: False
    if fault == 'generation': state['guard'] = False
    if fault == 'worker': rt.lidar_worker.running = False
    if fault == 'session': rt.lidar_worker.session = 'other'
    if fault == 'regression': state['sample']['acquisition_sequence'] = 40
    if fault == 'bool_sequence': state['sample']['acquisition_sequence'] = True
    if fault == 'missing_sequence': state['sample']['acquisition_sequence'] = None
    if fault == 'missing_session': state['sample'].pop('producer_session')
    result = recover(bundle, state)
    assert not result['ok'] and result['final_recovery_result'].endswith(reason)
    assert state['sleeps'] == 0 and not motions(events)


@pytest.mark.parametrize('fault', ['stale', 'geometry', 'coverage', 'same_sequence', 'over_age', 'invalid', 'unavailable'])
def test_bad_intermediate_new_scan_cannot_release_wait(tmp_path, monkeypatch, fault):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    state['publish'] = False
    first = copy.deepcopy(state['sample'])
    first.update(reason='fresh', valid=True, available=True, effective_age_seconds=0., acquisition_sequence=43)
    if fault == 'stale': first.update(reason='stale', effective_age_seconds=.31)
    if fault == 'over_age': first['effective_age_seconds'] = .300001
    if fault == 'geometry': first['local_motion_geometry']['valid'] = False
    if fault == 'coverage': first['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'same_sequence': first['acquisition_sequence'] = 42
    if fault == 'invalid': first['valid'] = False
    if fault == 'unavailable': first['available'] = False
    def advance():
        state['sample'] = copy.deepcopy(first)
        if state['sleeps'] == 4:
            state['sample'].update(acquisition_sequence=44, reason='fresh', valid=True, available=True, effective_age_seconds=0.)
            state['sample']['local_motion_geometry'] = bundle[1].lidar()['local_motion_geometry']
    state['on_sleep'] = advance
    result = recover(bundle, state)
    assert result['ok'] and result['new_sequence'] == 44
    assert result['wait_elapsed_seconds'] == pytest.approx(.2)


@pytest.mark.parametrize('fault,reason', [('session', 'producer_session_changed'),
    ('worker', 'worker_stopped'), ('owner', 'mission_preempted'),
    ('interlock', 'interlock_not_stopped'), ('bridge', 'bridge_not_stopped'),
    ('regression', 'acquisition_sequence_invalid')])
def test_loss_during_wait_is_terminal(tmp_path, monkeypatch, fault, reason):
    bundle, state, interlock = recovery_bundle(tmp_path, monkeypatch)
    state['publish'] = False
    def lose():
        if fault == 'session': bundle[0].lidar_worker.session = 'new'
        if fault == 'worker': bundle[0].lidar_worker.running = False
        if fault == 'owner': state['guard'] = False
        if fault == 'interlock': interlock['pending_forward'] = True
        if fault == 'bridge': bundle[2].ready = False
        if fault == 'regression': state['sample']['acquisition_sequence'] = 41
    state['on_sleep'] = lose
    result = recover(bundle, state)
    assert not result['ok'] and result['final_recovery_result'].endswith(reason)
    assert state['sleeps'] == 1 and not motions(bundle[3])


def test_timeout_is_bounded_and_stopped(tmp_path, monkeypatch):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    state['publish'] = False
    result = recover(bundle, state)
    assert not result['ok'] and result['final_recovery_result'] == 'find_marvin_new_lidar_evidence_timeout'
    assert result['wait_elapsed_seconds'] == pytest.approx(CognitiveRuntime.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS)
    assert result['wait_poll_count'] <= 13 and not motions(bundle[3])


def test_unavailable_missing_sequence_uses_prior_floor(tmp_path, monkeypatch):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    state['sample'].update(available=False, acquisition_sequence=None)
    result = recover(bundle, state)
    assert result['ok'] and result['selected_sequence_floor'] == 41
    assert result['sequence_floor_source'] == 'last_action_jit'


def inject_planning_outage(bundle, monkeypatch, *, kind='stale', repeated=False, after_actions=0):
    rt, behavior, robot, events, clock = bundle
    status = robot.status
    def full_status():
        value = status(); value['motion']['linear_y'] = 0.
        return value
    robot.status = full_status
    if rt.forward_interlock is None:
        rt.forward_interlock = SimpleNamespace(status=lambda: {'active_forward': False, 'pending_forward': False})
    behavior.reacquire_find_marvin_v2 = lambda **k: behavior.observe_find_marvin_v2()
    original = rt._active_localization_lidar_is_current
    read = rt.world_model.get_lidar_obstacles
    state = {'entered': 0, 'waiting': False, 'sleeps': 0, 'retired': [], 'dispatches': []}
    def planning(**kwargs):
        session, sample = original(**kwargs)
        if len(motions(events)) >= after_actions and (state['entered'] == 0 or repeated):
            state['entered'] += 1; state['waiting'] = True; state['sleeps'] = 0
            state['retired'].append(behavior.stamps[-1])
            state['dispatches'].append(len(motions(events)))
            sample = copy.deepcopy(sample)
            if kind == 'stale': sample.update(reason='stale', effective_age_seconds=.328185)
            if kind == 'geometry': sample['local_motion_geometry']['valid'] = False
            if kind == 'coverage': sample['local_motion_geometry']['sectors']['left']['available'] = False
        return session, sample
    rt._active_localization_lidar_is_current = planning
    def telemetry(**kwargs):
        sample = read(**kwargs)
        if state['waiting']:
            sample.update(reason='stale', effective_age_seconds=.328185)
        return sample
    rt.world_model.get_lidar_obstacles = telemetry
    original_sleep = time.sleep
    def sleep(seconds):
        if state['waiting']:
            assert len(motions(events)) == state['dispatches'][-1]
            assert robot.status()['motion']['streaming'] is False
            clock[0] += round(seconds * 1e9)
            state['sleeps'] += 1
            if kind != 'timeout' and state['sleeps'] == 2: state['waiting'] = False
        else: original_sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep', sleep)
    return state


@pytest.mark.parametrize('kind', ['stale', 'geometry', 'coverage'])
def test_planning_recovery_retires_visual_authority_then_new_strict_cycle_dispatches(tmp_path, monkeypatch, kind):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .6), (0, .6), (0, .5)])
    state = inject_planning_outage(bundle, monkeypatch, kind=kind)
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED', result['reason']
    recovery, = result['stationary_lidar_recovery_history']
    old = state['retired'][0]
    assert old == recovery['retired_source_stamp_ns']
    assert not recovery['source_stamp_transport_consumed']
    assert old not in bundle[0]._marvin_alignment_consumed_source_frame_stamps
    assert recovery['new_camera_source_stamp_ns'] > old
    assert len(motions(bundle[3])) == 1
    assert result['history'][0]['source_frame_stamp_ns'] == recovery['new_camera_source_stamp_ns']
    assert result['history'][0]['action_lidar_evidence'][1] > recovery['selected_sequence_floor']
    assert bundle[1].identity_sources[:2] == ['gemini_marvin_candidate_selection'] * 2
    assert result['consecutive_lidar_interruptions'] == 0


def test_pass_side_watchdog_and_counts_preserved_across_stationary_wait(tmp_path, monkeypatch):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=3)
    bundle[1].specs = iter(pursuit_specs([(0., 1.1)] * 5))
    state = inject_planning_outage(bundle, monkeypatch, after_actions=2)
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED', result['reason']
    recovery, = result['stationary_lidar_recovery_history']
    assert recovery['new_camera_source_stamp_ns'] > state['retired'][0]
    assert recovery['phase'] == 'PASS_OBSTACLE' and recovery['committed_side'] == 'LEFT'
    assert recovery['avoidance_count'] == 2 and recovery['phase_action_count'] == 1
    assert recovery['watchdog']['total_detour_actions'] == 2
    assert recovery['watchdog']['detour_started_monotonic'] is not None
    assert result['local_avoidance_actions'] == 4
    assert result['completed_strafe_actions'] == 1 and result['completed_bypass_forward_actions'] == 3
    assert state['retired'][0] not in bundle[0]._marvin_alignment_consumed_source_frame_stamps
    assert [r['phase'] for r in result['navigation_phase_history']] == ['CLEAR_SIDE','PASS_OBSTACLE','REJOIN','DIRECT','ARRIVED']
    assert len(motions(bundle[3])) == result['actions_executed']


def test_three_consecutive_planning_failures_exhaust_without_fourth_wait(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .6)] * 8)
    state = inject_planning_outage(bundle, monkeypatch, repeated=True)
    result = run(bundle[0])
    assert result['reason'] == 'find_marvin_lidar_recovery_exhausted'
    assert state['entered'] == result['consecutive_lidar_currentness_failures'] == 3
    assert len(result['stationary_lidar_recovery_history']) == 3 and not motions(bundle[3])
    assert [r.get('wait_poll_count', 0) for r in result['stationary_lidar_recovery_history']] == [3,3,0]


def test_retained_live_prefix_preserves_unknown_terminal_predicate():
    fixture = json.loads(Path('test_fixtures/marvin_729099dd_lidar_currentness.json').read_text())
    assert fixture['historical_terminal_class'] == 'LIDAR_CURRENTNESS_FAILURE_EXACT_PREDICATE_UNKNOWN'
    assert len(fixture['physical_prefix']) == 12
    interrupted = fixture['physical_prefix'][4]
    assert interrupted['transport_occurred'] and interrupted['interrupted']
    assert interrupted['invalidating_age_seconds'] > MAXIMUM_EFFECTIVE_AGE_SECONDS
    assert fixture['avoidance_counts'] == [1,2,3,4,5,6,7,8,8,9,10,11]
    assert fixture['first_after_six']['planning_sequence'] == 9271
    assert fixture['first_after_six']['newer_stopped_sequence'] == 9281
    assert fixture['terminal_failing_snapshot'] is None


def test_thresholds_and_wait_bound_unchanged():
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == .3
    assert CognitiveRuntime.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS == .6
    assert CognitiveRuntime.MAX_CONSECUTIVE_MARVIN_LIDAR_INTERRUPTION_FAILURES == 3
    assert CognitiveRuntime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0


@pytest.mark.parametrize('fault', ['duplicate', 'older', 'low_quality', 'unmatched', 'identity_false'])
def test_reacquisition_cannot_reuse_stamp_or_relax_tracker(tmp_path, monkeypatch, fault):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .6)] * 4)
    state = inject_planning_outage(bundle, monkeypatch)
    behavior = bundle[1]
    acquire = behavior.reacquire_find_marvin_v2
    def bad(**kwargs):
        value = acquire(**kwargs)
        preview = value['preview_result']; tracker = preview['opencv_tracker']
        floor = kwargs['minimum_source_frame_stamp_ns']
        if fault in {'duplicate', 'older'}:
            preview['identity_source_frame_stamp_ns'] = floor - int(fault == 'older')
            preview['source_frame_stamp_ns'] = tracker['source_frame_stamp_ns'] = floor
        if fault == 'low_quality': tracker['quality'] = .799999
        if fault == 'unmatched': tracker['matched'] = False
        if fault == 'identity_false': preview['identity_confirmed'] = False
        return value
    behavior.reacquire_find_marvin_v2 = bad
    result = run(bundle[0])
    assert result['reason'] == 'find_marvin_lidar_recovery_fresh_identity_required'
    assert not motions(bundle[3])
    assert state['retired'][0] not in bundle[0]._marvin_alignment_consumed_source_frame_stamps
    assert result['completed_forward_actions'] == 0


def test_watchdog_time_includes_stationary_wait_and_does_not_restart(tmp_path, monkeypatch):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=3)
    state = inject_planning_outage(bundle, monkeypatch, after_actions=2)
    sleep = time.sleep
    def late(seconds):
        if state['waiting']:
            bundle[4][0] += 121_000_000_000
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep', late)
    result = run(bundle[0])
    assert result['reason'] == 'find_marvin_detour_watchdog_exhausted'
    assert len(motions(bundle[3])) == result['local_avoidance_actions'] == 2
    assert result['detour_context']['committed_side'] == 'LEFT'
    assert result['detour_context']['phase'] == 'PASS_OBSTACLE'
    assert time.monotonic() - result['detour_context']['watchdog']['detour_started_monotonic'] >= 120


def test_delivery_uncertainty_cannot_enter_stationary_wait(tmp_path, monkeypatch):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    result = bundle[0]._recover_marvin_stationary_lidar(snapshot=state['sample'],
        expected_session='offline-v2-session', sequence_floors=[],
        execution_guard=lambda: True, entry_point='planning_reassessment', delivery_resolved=False)
    assert result['final_recovery_result'] == 'find_marvin_action_delivery_uncertain'
    assert state['sleeps'] == 0 and not motions(bundle[3])


def test_increasing_then_regressed_scan_during_wait_is_terminal(tmp_path, monkeypatch):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    state['publish'] = False
    def publication():
        state['sample']['acquisition_sequence'] = 45 if state['sleeps'] == 1 else 44
    state['on_sleep'] = publication
    result = recover(bundle, state)
    assert result['final_recovery_result'] == 'find_marvin_lidar_acquisition_sequence_invalid'
    assert state['sleeps'] == 2 and not motions(bundle[3])


def test_observation_lidar_failure_even_if_planning_read_recovers_requires_new_camera(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .6), (0, .6), (0, .5)])
    rt, behavior, robot, events, _ = bundle
    status = robot.status
    robot.status = lambda: dict(status(), motion=dict(status()['motion'], linear_y=0.))
    rt.forward_interlock = SimpleNamespace(status=lambda: {'active_forward': False, 'pending_forward': False})
    behavior.reacquire_find_marvin_v2 = lambda **k: behavior.observe_find_marvin_v2()
    read = behavior.lidar; calls = [0]
    def telemetry(**k):
        value = read(**k); calls[0] += 1
        if calls[0] == 1:
            value.update(valid=False, reason='stale', effective_age_seconds=.31)
        return value
    rt.world_model.get_lidar_obstacles = telemetry
    result = run(rt)
    assert result['state'] == 'ARRIVED'
    recovery, = result['stationary_lidar_recovery_history']
    assert recovery['entry_point'] == 'observation_route_reassessment'
    assert recovery['new_camera_source_stamp_ns'] > recovery['retired_source_stamp_ns']
    assert recovery['retired_source_stamp_ns'] not in rt._marvin_alignment_consumed_source_frame_stamps
    assert len(motions(events)) == 1


def test_retained_prefix_replays_physical_and_traversal_watchdog_accounting():
    from dataclasses import replace
    from marvin_detour_watchdog import DetourWatchdog
    fixture = json.loads(Path('test_fixtures/marvin_729099dd_lidar_currentness.json').read_text())
    watchdog = DetourWatchdog().enter('CLEAR_SIDE', 100.)
    for row in fixture['physical_prefix']:
        watchdog = watchdog.enter(row['phase'], 100. + row['index'])
        watchdog = watchdog.physical_action()
        if row['completed'] and row['primitive'] != 'TURN_LEFT':
            watchdog = watchdog.completed_traversal({})
        if row['primitive'] != 'TURN_LEFT':
            assert row['stamp_consumed'] and row['bridge_zero_confirmed']
        else:
            assert row['completed']  # The action ledger omits these two alignment fields.
    assert watchdog.total_detour_actions == 12 and watchdog.phase_actions == 9
    assert fixture['transported_recovery_wait']['new_sequence'] > 9196
    assert fixture['physical_prefix'][5]['source_stamp'] != fixture['physical_prefix'][4]['source_stamp']
    # The historical mission's measured progress counters are retained, not inferred.
    native = fixture['native_terminal_context']['watchdog']
    watchdog = replace(watchdog, phase_stagnation_count=native['phase_stagnation_count'])
    assert watchdog.exhaustion(100. + fixture['detour_elapsed_seconds']) is None
    assert fixture['terminal_failing_snapshot'] is None  # No fabricated predicate.


def test_failure_certificate_remains_bounded_with_malformed_optional_fields(tmp_path, monkeypatch):
    bundle, state, _ = recovery_bundle(tmp_path, monkeypatch)
    state['sample']['sectors'] = [0] * 10000
    state['sample']['reason'] = 'x' * 100000
    result = recover(bundle, state)
    assert result['ok']
    assert len(result['reason']) == 128 and result['front_state'] is None
    assert len(json.dumps(result)) < 1800
