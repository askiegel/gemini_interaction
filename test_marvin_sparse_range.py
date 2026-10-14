"""Offline retained-scan replays and stopped recovery; no live transport."""
import copy
import gzip
import json
import math
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import pytest

from marvin_lidar_standoff import evaluate_marvin_lidar_standoff
from marvin_target_range_association import MarvinTargetRangeAssociation
from local_motion_safety_envelope import evaluate_local_motion_safety
from test_find_marvin_closed_loop import make_runtime, delayed_runtime, motions, run
from test_marvin_continuous_lidar_recovery import stationary_status

FIXTURE = json.loads(gzip.decompress((Path(__file__).parent /
    'test_fixtures/marvin_930a9281_sparse_range.json.gz').read_bytes()))
CAMERA = FIXTURE['calibration']
IDENTITY = FIXTURE['refusal']['identity']


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Sparse recovery tests cannot access robots/services')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def view(name='refusal'):
    v = copy.deepcopy(FIXTURE[name])
    v['lidar'].update(received_monotonic_seconds=time.monotonic(),
                      age_at_receipt_seconds=0., effective_age_seconds=0.)
    v['tracker']['received_monotonic_seconds'] = time.monotonic()
    return v


def newer(v, index=1):
    v = copy.deepcopy(v)
    s, t = v['lidar'], v['tracker']
    s['acquisition_sequence'] += index
    s['bridge_scan_identity']['acquisition_sequence'] += index
    s['bridge_scan_identity']['source_stamp_ns'] += index * 100_000_000
    t['source_frame_stamp_ns'] += index * 100_000_000
    s['received_monotonic_seconds'] = t['received_monotonic_seconds'] = time.monotonic()
    return v


def candidate(v):
    return evaluate_marvin_lidar_standoff(v['tracker'], v['lidar'], CAMERA,
                                        expected_session=v['lidar']['producer_session'])


def seeded():
    gate = MarvinTargetRangeAssociation()
    # Action diagnostics omit native metadata. Post-action retained waiter
    # scans establish the producer only; never invent native sequences/stamps.
    native_producers = {r['native']['producer_session']
                        for r in FIXTURE['native_post_action_scan_identities']}
    assert native_producers == {FIXTURE['refusal']['lidar']['bridge_scan_identity']['producer_session']}
    for row in FIXTURE['preceding_actions']:
        v = copy.deepcopy(row)
        v['lidar']['received_monotonic_seconds'] = time.monotonic()
        v['lidar']['bridge_scan_identity'] = {'producer_session': next(iter(native_producers))}
        c = candidate(v)
        result = gate.evaluate(c, v['lidar'], v['tracker'],
            expected_session=v['lidar']['producer_session'], identity=IDENTITY, commit=True)
        assert result['target_range_association_trusted']
        assert not result['arrived_at_marvin']
        gate.record_forward_bound(.1, .5)
    return gate


def evaluate(gate, v, commit=False):
    return gate.evaluate(candidate(v), v['lidar'], v['tracker'],
        expected_session=v['lidar']['producer_session'], identity=v['identity'], commit=commit)


def begin(gate, v):
    return gate.begin_sparse_recovery(candidate(v), v['lidar'], v['tracker'], identity=v['identity'])


def corroborate(gate, v):
    return gate.corroborate_sparse(candidate(v), v['lidar'], v['tracker'], identity=v['identity'])


def test_exact_refusal_and_four_preceding_actions_replay():
    gate, v = seeded(), view()
    c = candidate(v)
    assert c['point_count'] == 11 and c['candidate_surface_point_count'] == 4
    assert c['measured_distance_m'] == pytest.approx(1.4072977952889822)
    assert evaluate(gate, v)['reason'] == 'marvin_target_range_returns_ambiguous'
    assert begin(gate, v)
    assert not corroborate(gate, newer(v))
    assert not evaluate(gate, newer(v))['target_range_association_trusted']
    latest = newer(v, 2)
    assert corroborate(gate, latest)
    result = evaluate(gate, latest, commit=True)
    assert result['target_range_association_trusted'] and result['sparse_surface_corroborated']
    assert result['arrived_at_marvin'] is False
    assert gate.anchor['measured_distance_m'] == pytest.approx(1.354330815261963)
    # Planning/JIT rereads do not count observations; current scan is checked.
    jit = newer(latest)
    jit['tracker'] = latest['tracker']
    assert evaluate(gate, jit)['target_range_association_trusted']
    gate.record_forward_bound(.1, .5)
    assert not evaluate(gate, jit)['target_range_association_trusted']


def test_retained_two_scans_one_camera_are_not_independent_pairs():
    gate, prior, final = seeded(), view('prior_sparse_scan_same_camera'), view()
    assert begin(gate, prior)
    assert prior['tracker']['source_frame_stamp_ns'] == final['tracker']['source_frame_stamp_ns']
    assert not corroborate(gate, final)
    assert not evaluate(gate, final)['target_range_association_trusted']


def test_three_adjacent_rays_still_require_two_new_stopped_pairs():
    gate, v = seeded(), view()
    points = v['lidar']['local_motion_geometry']['points']
    first = min((p for p in points if -1.7<p['robot_bearing_deg']<.6
                 and 1.3<math.hypot(p['x_m'],p['y_m'])<1.5), key=lambda p:p['robot_bearing_deg'])
    points.remove(first)
    assert candidate(v)['candidate_surface_point_count']==3
    assert begin(gate,v) and not corroborate(gate,newer(v))
    latest=newer(v,2)
    assert corroborate(gate,latest)
    assert evaluate(gate,latest)['sparse_surface_corroborated']
    assert not evaluate(gate,latest)['arrived_at_marvin']


def test_october6_all_forward_observations_and_arrival_measurements_replay():
    assert len(FIXTURE['october6_forward_actions']) == 24
    assert len(FIXTURE['october6_action_states']) == 43
    for row in FIXTURE['october6_forward_actions'] + [FIXTURE['october6_arrival']]:
        c = candidate(row)
        for field in ('measured_distance_m', 'target_distance_m', 'target_bearing_degrees'):
            assert c[field] == pytest.approx(row['arrival'][field])
        assert c['point_count'] == row['arrival']['point_count']
        assert c['candidate_surface_point_count'] >= 5
    c = candidate(FIXTURE['october6_arrival'])
    assert c['candidate_at_standoff'] and c['target_distance_m'] == pytest.approx(.4903512722816691)
    # Archived October 6 does not contain the later association gate's anchor;
    # verify its unchanged dense branch with the previous trusted recorded depth.
    v = copy.deepcopy(FIXTURE['october6_arrival'])
    v['lidar']['received_monotonic_seconds'] = time.monotonic()
    gate = MarvinTargetRangeAssociation()
    gate.anchor = dict(measured_distance_m=c['measured_distance_m'],
        target_distance_m=c['target_distance_m'], producer_session=v['lidar']['producer_session'],
        translation_bound_m=0., identity='oct6')
    assert evaluate(gate, v)['arrived_at_marvin']
    assert gate.sparse_confirmation is None


@pytest.mark.parametrize('fault', ['identity', 'session', 'native_session', 'stale',
    'invalid', 'camera_duplicate', 'sequence_duplicate', 'native_duplicate',
    'native_stamp_duplicate', 'regressed', 'inconsistent_range', 'competing',
    'wall', 'isolated', 'disjoint', 'unrelated_foreground', 'tracker', 'boundary'])
def test_bad_sparse_evidence_never_corroborates(fault):
    gate, v = seeded(), view()
    assert begin(gate, v)
    assert not corroborate(gate, newer(v))
    bad = newer(v, 2)
    s, t = bad['lidar'], bad['tracker']
    if fault == 'identity': bad['identity'] = 'another-Marvin-episode'
    if fault == 'session': s['producer_session'] = 'new-worker'
    if fault == 'native_session': s['bridge_scan_identity']['producer_session'] = 'new-native'
    if fault == 'stale': s.update(reason='stale', valid=False, effective_age_seconds=.31)
    if fault == 'invalid': s['local_motion_geometry']['valid'] = False
    if fault == 'camera_duplicate': t['source_frame_stamp_ns'] -= 100_000_000
    if fault == 'sequence_duplicate': s['acquisition_sequence'] -= 1
    if fault == 'native_duplicate': s['bridge_scan_identity']['acquisition_sequence'] -= 1
    if fault == 'native_stamp_duplicate': s['bridge_scan_identity']['source_stamp_ns'] -= 100_000_000
    if fault == 'regressed': s['bridge_scan_identity']['source_stamp_ns'] = 0
    if fault == 'tracker': t['matched'] = False
    points = s['local_motion_geometry']['points']
    near = [p for p in points if 1.3 < math.hypot(p['x_m'], p['y_m']) < 1.5
            and -1.7 < p['robot_bearing_deg'] < .6]
    if fault in {'inconsistent_range', 'unrelated_foreground', 'boundary'}:
        target = 1.8 if fault == 'inconsistent_range' else .8 if fault == 'unrelated_foreground' else .59
        for p in near:
            factor = target/math.hypot(p['x_m'], p['y_m'])
            p['x_m'] *= factor; p['y_m'] *= factor
    if fault == 'competing':
        points.append(dict(x_m=1.55, y_m=-.03, distance_m=1.55, robot_bearing_deg=-1.))
    if fault == 'wall':
        points.append(dict(x_m=1.41, y_m=-.28, distance_m=1.44, robot_bearing_deg=-11.))
    if fault == 'isolated':
        for p in near[1:]: points.remove(p)
    if fault == 'disjoint': near[-1]['robot_bearing_deg'] += 3
    assert not corroborate(gate, bad)
    assert not evaluate(gate, bad)['target_range_association_trusted']
    assert not evaluate(gate, bad)['arrived_at_marvin']


def test_no_anchor_and_four_return_arrival_cannot_use_fallback():
    v = view()
    assert not begin(MarvinTargetRangeAssociation(), v)
    gate = seeded()
    for p in v['lidar']['local_motion_geometry']['points']:
        if 1.3 < math.hypot(p['x_m'], p['y_m']) < 1.5:
            p['x_m'] *= .4; p['y_m'] *= .4
    assert candidate(v)['candidate_at_standoff']
    assert not begin(gate, v)
    assert not evaluate(gate, v)['arrived_at_marvin']


def test_native_producer_change_before_stopped_recovery_rejects_old_anchor():
    gate, v = seeded(), view()
    v['lidar']['bridge_scan_identity']['producer_session'] = 'replacement-sensor'
    assert not begin(gate, v)
    assert not evaluate(gate, v)['target_range_association_trusted']


def test_competing_surface_inside_full_bbox_outside_center_band_rejected():
    gate, v = seeded(), view()
    # Projects to x=415, within the full bbox but outside the central band.
    x = 1.53
    y = -(415-CAMERA['cx_pixels']) * (x-CAMERA['x_m']) / CAMERA['fx_pixels']
    v['lidar']['local_motion_geometry']['points'].append(dict(x_m=x,y_m=y,
        distance_m=math.hypot(x,y),robot_bearing_deg=math.degrees(math.atan2(y,x))))
    assert candidate(v)['sparse_surface']['competing_ranges_m']
    assert not begin(gate, v)


def test_competing_surface_uses_existing_cumulative_range_allowance():
    gate, v = seeded(), view()
    gate.record_forward_bound(.1, 2.)
    # A second plausible surface after further bounded pursuit, farther than
    # twice the cluster tolerance, still makes the association ambiguous.
    v['lidar']['local_motion_geometry']['points'].append(dict(x_m=1.65,y_m=-.02,
        distance_m=1.65,robot_bearing_deg=-.7))
    assert not begin(gate,v)


def test_new_competing_surface_at_jit_revokes_sparse_confirmation():
    gate, v = seeded(), view()
    assert begin(gate,v) and not corroborate(gate,newer(v))
    latest=newer(v,2)
    assert corroborate(gate,latest)
    jit=newer(latest)
    jit['tracker']=latest['tracker']
    jit['lidar']['local_motion_geometry']['points'].append(dict(x_m=1.55,y_m=-.03,
        distance_m=1.55,robot_bearing_deg=-1.))
    assert not evaluate(gate,jit)['target_range_association_trusted']
    assert gate.sparse_confirmation is None


def test_corroborated_sparse_surface_at_arrival_boundary_still_cannot_arrive():
    gate, v = seeded(), view()
    assert begin(gate,v) and not corroborate(gate,newer(v))
    latest = newer(v,2)
    assert corroborate(gate,latest)
    c = candidate(latest)
    # Even a calibration-boundary change after corroboration cannot reuse it
    # to authorize arrival. The dense arrival policy must establish its own proof.
    c.update(candidate_at_standoff=True,target_distance_m=.50)
    result = gate.evaluate(c,latest['lidar'],latest['tracker'],
        expected_session=latest['lidar']['producer_session'],identity=IDENTITY)
    assert not result['arrived_at_marvin'] and not result['target_range_association_trusted']


def recovery_runtime(tmp_path, monkeypatch, fault=None):
    r, behavior, robot, events, clock = make_runtime(tmp_path, monkeypatch, [])
    r._marvin_target_range_association = seeded()
    r.lidar_worker.session = FIXTURE['refusal']['lidar']['producer_session']
    r.forward_interlock = SimpleNamespace(status=lambda: dict(active_forward=False,
        pending_forward=False, stop_pending=False, last_stop_error=None))
    r._marvin_motion_owner_is_current = lambda: True
    locked = object()
    behavior._marvin_v2_tracker_episode = dict(episode_id=IDENTITY, marvin_tracker=locked)
    state = {'reads': 0, 'observations': 0, 'wait_motion_counts': [], 'guard': True}
    original_status = robot.status
    def status():
        s = original_status();s['motion']['linear_y'] = 0.
        s['guarded_controller'] = stationary_status()
        if fault == 'owner' and state['observations']:
            s['guarded_controller']['sequence'] += 1
        if fault == 'stop_pending': r.forward_interlock.status = lambda: dict(active_forward=False,
            pending_forward=False, stop_pending=True, last_stop_error=None)
        return s
    robot.status = status
    if fault == 'stop_failed': robot.stop = lambda: {'ok': False}
    def scan(**kwargs):
        state['reads'] += 1
        v = newer(view(), state['reads'])
        if fault == 'frozen': v['lidar']['acquisition_sequence'] = FIXTURE['refusal']['lidar']['acquisition_sequence']
        return v['lidar']
    r.world_model.get_lidar_obstacles = scan
    def observation(**kwargs):
        state['observations'] += 1
        state['wait_motion_counts'].append(len(motions(events)))
        clock[0] += 100_000_000
        v = newer(view(), state['observations'])
        v['tracker']['received_monotonic_seconds'] = time.monotonic()
        s = scan()
        if fault == 'duplicate_camera': v['tracker']['source_frame_stamp_ns'] = FIXTURE['refusal']['tracker']['source_frame_stamp_ns']
        if fault == 'identity': behavior._marvin_v2_tracker_episode['marvin_tracker'] = object()
        if fault == 'STOP': state['guard'] = False
        arrival = r._marvin_v2_lidar_arrival(v['tracker'], s,
            corroborate_sparse=kwargs['stopped_sparse_range'])
        decision = 'FORWARD' if arrival['target_range_association_trusted'] else 'BLOCKED'
        if decision == 'FORWARD': r._marvin_alignment_observation = {'source_frame_stamp_ns':v['tracker']['source_frame_stamp_ns']}
        return dict(ok=True, identity_confirmed=True, identity_source='marvin_locked_tracker_continuity',
            source_frame_stamp_ns=v['tracker']['source_frame_stamp_ns'], opencv_tracker=v['tracker'],
            received_monotonic_seconds=time.monotonic(), arrival=arrival,
            controller={'decision':decision})
    r._observe_find_marvin_v2 = observation
    return r, events, state


def test_recovery_is_stopped_bounded_and_returns_to_ordinary_planning(tmp_path, monkeypatch):
    r, events, state = recovery_runtime(tmp_path, monkeypatch)
    v = view()
    evidence, latest = r._recover_marvin_sparse_range(
        observation={'identity_confirmed': True, 'opencv_tracker':v['tracker']},
        lidar=v['lidar'], expected_session=r.lidar_worker.session, execution_guard=lambda:state['guard'])
    assert evidence['ok'] and len(evidence['observations']) == 2
    assert state['wait_motion_counts'] == [0, 0] and not motions(events)
    assert events == ['stop']
    assert latest['controller']['decision'] == 'FORWARD'
    # Inject an obstacle after corroboration: the ordinary JIT safety gate still vetoes.
    s = r.world_model.get_lidar_obstacles()
    s['local_motion_geometry']['points'].append(dict(x_m=.4,y_m=0.,distance_m=.4,robot_bearing_deg=0.))
    guard = evaluate_local_motion_safety(s, expected_session=r.lidar_worker.session, linear_x=.1,duration=.5)
    assert not guard['permitted'] and guard['protected_radius_m'] == .45


@pytest.mark.parametrize('fault', ['stop_failed','owner','stop_pending','frozen','identity',
                                  'duplicate_camera','STOP'])
def test_recovery_never_moves_on_failed_stationary_evidence(tmp_path, monkeypatch, fault):
    r, events, state = recovery_runtime(tmp_path, monkeypatch, fault)
    v = view()
    evidence, _ = r._recover_marvin_sparse_range(
        observation={'identity_confirmed':True,'opencv_tracker':v['tracker']},lidar=v['lidar'],
        expected_session=r.lidar_worker.session,execution_guard=lambda:state['guard'])
    assert not evidence['ok'] and not motions(events)
    assert r._marvin_target_range_association.sparse_confirmation is None


@pytest.mark.parametrize('jit_intrusion', [False, True])
def test_full_mission_retained_refusal_recovers_through_real_identity_planning_and_jit(
        tmp_path, monkeypatch, jit_intrusion):
    # Real Runtime/BehaviorManager strict identity/continuity, association,
    # planner and final-JIT code; only scene acquisition and robot are fake.
    r, b, robot, events, clock = delayed_runtime(tmp_path, monkeypatch, [(0,1.3)] * 10)
    rows = copy.deepcopy(FIXTURE['preceding_actions'] + [FIXTURE['refusal']] * 3)
    state = {'stage': -1, 'reads': 0, 'inside_fifth': False, 'recovering': False,
             'observations_while_stopped': [], 'recoveries': []}
    r.lidar_worker.session = FIXTURE['refusal']['lidar']['producer_session']
    r.forward_interlock = SimpleNamespace(status=lambda: dict(active_forward=False,
        pending_forward=False, stop_pending=False, last_stop_error=None))
    original_status = robot.status
    def status():
        s = original_status();s['motion']['linear_y'] = 0.
        s['guarded_controller'] = stationary_status()
        return s
    robot.status = status
    def current_box():
        box = rows[max(0,state['stage'])]['tracker']['bbox']
        # Production tracker seed padding stays 20% per side; invert it to
        # supply a semantic proposal whose seeded box is the recorded box.
        dx=(box['x2']-box['x1'])/1.4*.2;dy=(box['y2']-box['y1'])/1.4*.2
        return dict(x1=box['x1']+dx,x2=box['x2']-dx,y1=box['y1']+dy,y2=box['y2']-dy)
    b.current_box = current_box
    observer = b.observe_find_marvin_v2
    def observe():
        state['stage'] += 1
        if state['recovering']:
            state['observations_while_stopped'].append(len(motions(events)))
        return observer()
    b.observe_find_marvin_v2 = observe
    def scan(**kwargs):
        state['reads'] += 1;clock[0] += 1_000_000
        s=copy.deepcopy(rows[max(0,state['stage'])]['lidar'])
        s.update(acquisition_sequence=7000+state['reads'],received_monotonic_seconds=time.monotonic(),
                 age_at_receipt_seconds=0.,effective_age_seconds=0.)
        s['bridge_scan_identity']=dict(producer_session='retained-native',
            acquisition_sequence=8000+state['reads'],source_stamp_ns=clock[0])
        if jit_intrusion and state['inside_fifth']:
            s['local_motion_geometry']['points'].append(dict(x_m=.4,y_m=0.,
                distance_m=.4,robot_bearing_deg=0.))
        return s
    r.world_model.get_lidar_obstacles = scan
    recovery = r._recover_marvin_sparse_range
    def recover(**kwargs):
        state['recovering']=True
        try:
            result = recovery(**kwargs)
            state['recoveries'].append(copy.deepcopy(result[0]))
            return result
        finally:state['recovering']=False
    r._recover_marvin_sparse_range=recover
    execute=r._execute_single_marvin_approach
    def approach(**kwargs):
        state['inside_fifth']=len(motions(events))==4
        try:return execute(**kwargs)
        finally:state['inside_fifth']=False
    r._execute_single_marvin_approach=approach
    def after_motion():
        if len(motions(events))==5:
            r.submit_intent({'intent':'STOP','speech':'Offline operator stop'})
    robot.on_motion=after_motion
    result=run(r)
    assert len(motions(events)) == (4 if jit_intrusion else 5), result
    assert state['observations_while_stopped']==[4,4]
    assert state['recoveries'][0]['ok']
    assert state['recoveries'][0]['motion_executed'] is False
    assert len(b.created_trackers)==1  # Continued strict identity; no reseeding.
    assert result.get('arrived_at_marvin') is not True
    assert r._marvin_alignment_observation is None
