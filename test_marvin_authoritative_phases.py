"""Authoritative phases with real native guards and entirely offline transport."""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from marvin_obstacle_phases import DetourContext, Frontier, Phase, plan_phase_action, phase_from_admission
from marvin_local_bypass import plan_local_bypass
from marvin_local_obstacle_avoidance import MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import evaluate_route_progress
from test_marvin_avoidance_planner_progress import scan
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_marvin_local_bypass import ASSOCIATION, OPEN_LEFT, mission_bundle, pursuit_specs
from test_find_marvin_closed_loop import make_runtime, motions, run


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Authoritative validation cannot access robots or services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def plan(points, side='LEFT', **kwargs):
    state = scan(points)
    return state, plan_phase_action(state, ASSOCIATION, expected_session='test',
        allow_strafe=True, committed_side=side, **kwargs)


def test_direct_clear_uses_existing_forward_and_arrival(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0,.6),(0,.5)])
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED' and result['navigation_phase'] == 'ARRIVED'
    assert motions(bundle[3]) == [('forward',.1,.5)]
    assert result['local_avoidance_actions'] == 0


def test_obstacle_enters_clear_side_and_same_side_strafe():
    _, result = plan(LEFT_OPEN, side=None, entering_detour=True)
    assert result['phase'] == 'CLEAR_SIDE' and result['action_type'] == 'STRAFE_LEFT'
    assert result['committed_side'] == 'LEFT'


def test_established_passage_enters_pass_and_bypass():
    _, result = plan(OPEN_LEFT)
    assert result['phase'] == 'PASS_OBSTACLE' and result['action_type'] == 'BYPASS_FORWARD'
    b = result['local_bypass']
    assert b['bypass_forward_permitted'] and b['bypass_corridor_occupancy'] == 0
    assert b['bypass_corridor_overlap_m'] == 0 and b['protected_radius_m'] == .45
    assert b['acquisition_sequence'] == result['acquisition_sequence']


def test_initial_side_establishment_then_immediate_pass_without_handoff_credit(tmp_path, monkeypatch):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=2)
    result = run(bundle[0])
    phases = [r['phase'] for r in result['navigation_phase_history']]
    assert phases == ['CLEAR_SIDE','PASS_OBSTACLE','REJOIN','DIRECT','ARRIVED']
    assert [r['selection']['action_type'] for r in result['local_avoidance_history']] == [
        'STRAFE_LEFT','BYPASS_FORWARD','BYPASS_FORWARD']
    assert not any('bypass_episode' in r['selection'] for r in result['local_avoidance_history'])


def test_neutral_route_does_not_blacklist_pass_or_claim_progress(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,1.1)]*9, [OPEN_LEFT])
    result = run(bundle[0])
    assert result['local_bypass_actions'] == 5
    assert result['local_avoidance_actions'] == len(motions(bundle[3])) == 6
    assert result['reason'] == 'find_marvin_local_avoidance_exhausted'
    for row in result['local_avoidance_history'][2:]:
        assert row['selection']['phase'] == 'PASS_OBSTACLE'
        assert row['selection']['meaningful_progress'] is False


@pytest.mark.parametrize('points', [
    [(.65,-.14),(0,1.2),(0,-.65)],
    OPEN_LEFT + [(.46,-.12)],
])
def test_lost_clearance_or_blocked_corridor_repairs_same_side(points):
    _, result = plan(points)
    assert result['local_bypass']['bypass_forward_permitted'] is False
    assert result['phase'] == 'CLEAR_SIDE' and result['action_type'] == 'STRAFE_LEFT'


def test_neither_safe_pass_nor_repair_enters_wait():
    _, result = plan(OPEN_LEFT + [(0,.3)])
    assert result['phase'] == 'BLOCKED_WAIT' and result['action_type'] is None


def test_committed_side_does_not_oscillate_or_switch_to_safe_other_side():
    _, result = plan([(.65,-.14),(0,.48),(0,-1.2)])
    assert result['committed_side'] == 'LEFT'
    assert result['action_type'] != 'STRAFE_RIGHT'


@pytest.mark.parametrize('fault', ['stale','session','coverage','invalid'])
def test_sensor_failure_cannot_propose_pass(fault):
    state = scan(OPEN_LEFT)
    if fault == 'stale':state['received_monotonic_seconds'] -= .301
    if fault == 'session':state['producer_session'] = 'other'
    if fault == 'coverage':state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'invalid':state['local_motion_geometry']['valid'] = False
    result = plan_phase_action(state, ASSOCIATION, expected_session='test',
        allow_strafe=True, committed_side='LEFT')
    assert result['action_type'] is None
    assert result['phase'] is None  # Unhealthy evidence is not healthy BLOCKED_WAIT.


@pytest.mark.parametrize('point', [(.18,0),(.40,.2),(-.1,.40),(.17,-.40)])
def test_protected_capsule_045_unchanged(point):
    _, result = plan(OPEN_LEFT + [point])
    assert not result['local_bypass']['bypass_forward_permitted']


@pytest.mark.parametrize('remaining', [0,-1,True])
def test_cap_external_to_phase_rejects_seventh_action(remaining):
    _, result = plan(OPEN_LEFT, remaining_avoidance_actions=remaining)
    assert result['action_type'] is None
    assert result['reason'] == 'find_marvin_local_avoidance_exhausted'


def test_context_completion_and_alignment_do_not_reset_side_or_count_progress():
    first = Frontier('s',10,1791504411407983987)
    ctx = DetourContext().enter(Phase.CLEAR_SIDE, first, side='LEFT')
    ctx = ctx.completed(first, traversal=True)
    second = Frontier('s',11,first.source_frame_stamp_ns+1)
    after_turn = ctx.completed(second, traversal=False)
    assert after_turn.committed_side == 'LEFT' and after_turn.phase_action_count == 1
    assert after_turn.phase == Phase.CLEAR_SIDE
    assert after_turn.last_completion_evidence == second


@pytest.mark.parametrize('stamp', [1791504411407984000.0,True,0])
def test_frontier_rejects_inexact_stamp(stamp):
    with pytest.raises(ValueError):Frontier('s',1,stamp)


def test_retired_veto_stamp_cannot_complete_or_authorize_repair():
    e = Frontier('s',10,1791504411407983987)
    ctx = DetourContext().enter(Phase.PASS_OBSTACLE,e,side='LEFT').retire(e)
    with pytest.raises(ValueError):ctx.completed(e,traversal=True)
    assert ctx.retired_source_stamp_ns == e.source_frame_stamp_ns


def test_context_session_change_fails_closed():
    ctx = DetourContext().enter(Phase.CLEAR_SIDE,Frontier('s',1,1),side='LEFT')
    with pytest.raises(ValueError):ctx.enter(Phase.PASS_OBSTACLE,Frontier('other',2,2))


def test_route_clear_discards_side_and_rejoins_direct():
    _, result = plan([])
    assert result['phase'] == 'REJOIN' and result['action_type'] is None
    e = Frontier('s',2,2)
    ctx = DetourContext(phase=Phase.PASS_OBSTACLE,committed_side='LEFT').enter(Phase.REJOIN,e)
    assert ctx.committed_side is None
    assert ctx.enter(Phase.DIRECT,e).phase == Phase.DIRECT


def test_authoritative_center_metric_not_minimum_metric():
    # Two independent blockers: nearest center passes .15; farther blocking
    # return has lower minimum-side separation. Native center gate unchanged.
    _, result = plan([(.62,-.155437),(.70,-.141939),(0,1.2),(0,-.65)])
    assert result['blocker_center_separation_m'] == pytest.approx(.155437)
    assert result['minimum_side_separation_m'] == pytest.approx(.141939)
    assert result['phase'] == 'PASS_OBSTACLE'


def test_alignment_keeps_committed_side_and_requires_new_geometry(tmp_path, monkeypatch):
    specs = pursuit_specs([(0,1.1),(100,1.1),(0,1.1)])
    b, _, _ = strafe_runtime(tmp_path,monkeypatch,specs,[OPEN_LEFT,OPEN_LEFT,OPEN_LEFT,None])
    result = run(b[0])
    assert [r['state'] for r in result['history']][:3] == ['AVOIDING','ALIGNING','AVOIDING']
    assert result['local_avoidance_history'][1]['selection']['direction'] == 'LEFT'
    assert result['local_avoidance_actions'] == sum(r['state']=='AVOIDING' and r['motion_executed'] for r in result['history'])
    a, turn, bypass = result['history'][:3]
    assert bypass['action_lidar_evidence'][1] > turn['action_lidar_evidence'][1]
    assert turn['source_frame_stamp_ns'] > a['source_frame_stamp_ns']


def veto_to_repair(tmp_path, monkeypatch, *, repeat=False):
    b, _, _ = strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0,1.1)]*4),
        [OPEN_LEFT,OPEN_LEFT,OPEN_LEFT,None])
    r, behavior, robot, events, _ = b
    read = r.world_model.get_lidar_obstacles
    flags = {'jit':False,'vetoed':False}
    def lidar(**kwargs):
        value = read(**kwargs)
        if flags['jit']:
            for point in value['local_motion_geometry']['points']:
                if .60 < point['x_m'] < .70:point['y_m'] = -.14
        return value
    r.world_model.get_lidar_obstacles = lidar
    execute = behavior.execute_single_marvin_approach_step
    def bypass(**kwargs):
        if kwargs.get('local_selection_validator') and not flags['vetoed']:
            flags['jit'] = True
            value = execute(**kwargs)
            flags['vetoed'] = True
            if repeat:behavior.repeat_camera = True
            return value
        return execute(**kwargs)
    behavior.execute_single_marvin_approach_step = bypass
    # Restore the scene after the same-side repair, allowing ordinary pursuit.
    robot.on_motion = lambda: flags.update(jit=False) if len(motions(events)) >= 2 else None
    return b, flags


def test_jit_bypass_veto_replans_repair_after_new_strict_observation(tmp_path, monkeypatch):
    b, flags = veto_to_repair(tmp_path,monkeypatch)
    result = run(b[0])
    vetoes = [r for r in result['local_avoidance_history'] if r.get('vetoed_before_transport')]
    assert flags['vetoed'] and len(vetoes) == 1
    assert vetoes[0]['phase_after_veto'] == 'CLEAR_SIDE'
    assert result['blocked_wait_recheck_count'] == 0
    rows = result['history']
    veto_index = next(i for i,r in enumerate(rows) if r['result'].get('pre_transport_jit_veto'))
    veto, repair = rows[veto_index:veto_index+2]
    assert not veto['motion_executed'] and repair['result']['action_type'] == 'STRAFE_LEFT'
    assert repair['source_frame_stamp_ns'] > veto['source_frame_stamp_ns']
    assert repair['action_lidar_evidence'][1] > vetoes[0]['jit_veto_lidar_sequence']
    assert veto['source_frame_stamp_ns'] in b[0]._marvin_alignment_consumed_source_frame_stamps
    assert result['state'] == 'ARRIVED'


def test_veto_replayed_camera_cannot_dispatch_repair(tmp_path,monkeypatch):
    b, _ = veto_to_repair(tmp_path,monkeypatch,repeat=True)
    result = run(b[0])
    assert result['state'] == 'REVERIFY_REQUIRED'
    assert motions(b[3]) == [('strafe',.08,1.)]


def test_disabled_shadow_not_required_or_consulted(tmp_path,monkeypatch):
    monkeypatch.setenv('MARVIN_NAVIGATION_SHADOW_ENABLED','false')
    b, _, _ = mission_bundle(tmp_path,monkeypatch)
    assert b[0]._marvin_navigation_shadow is None
    result = run(b[0])
    assert result['state'] == 'ARRIVED'
    assert any(r['phase']=='PASS_OBSTACLE' for r in result['navigation_phase_history'])


def test_repeated_strafe_native_fixture_enters_pass_before_six():
    fixture = json.loads((Path(__file__).parent/'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    side = 'LEFT'; phases = []
    for row in fixture['cycles']:
        state = copy.deepcopy(row['lidar']);state['received_monotonic_seconds'] = time.monotonic()
        state['age_at_receipt_seconds'] = 0.
        result = plan_phase_action(state,row['association'],expected_session=state['producer_session'],
            allow_strafe=True,committed_side=side)
        phases.append(result['phase'])
    assert 'PASS_OBSTACLE' in phases[:6]


def test_2033f0c3_native_worsening_bypass_retains_safe_pass():
    f=json.loads((Path(__file__).parent/'test_fixtures/marvin_2033f0c3_bypass_continuation.json').read_text())
    phases=[]
    for prefix in ['pre_bypass','post_bypass']:
        state=copy.deepcopy(f[prefix+'_lidar']);state['received_monotonic_seconds']=time.monotonic()
        state['age_at_receipt_seconds']=0.
        phases.append(plan_phase_action(state,f[prefix+'_association'],expected_session=state['producer_session'],
            allow_strafe=True,committed_side='LEFT')['phase'])
    assert phases == ['PASS_OBSTACLE','PASS_OBSTACLE']
    assert f['recorded_bypass_progress']['bypass_longitudinal_progress_m'] < 0


def test_031fe9af_retained_veto_supports_repair_but_not_raw_cloud_replay():
    f=json.loads((Path(__file__).parent/'test_fixtures/marvin_authoritative_retained_cases.json').read_text())['mission_031fe9af']
    jit=f['jit_selection']
    assert -jit['route']['blocking_obstacle_y_m'] < .15
    assert jit['options']['STRAFE_LEFT']['hard_safety_permitted'] is True
    assert jit['options']['STRAFE_LEFT']['permitted'] is True
    assert 'not retained' in f['origin']['geometry_provenance']
    assert phase_from_admission(route_obstructed=jit['route']['route_to_marvin_obstructed'],
        pass_safe=jit['local_bypass']['bypass_forward_permitted'],
        repair_safe=jit['options']['STRAFE_LEFT']['permitted']) == Phase.CLEAR_SIDE


def test_29949cca_retained_center_gate_stays_pass_despite_minimum_crossing():
    f=json.loads((Path(__file__).parent/'test_fixtures/marvin_authoritative_retained_cases.json').read_text())['mission_29949cca']
    physical=[r['primitive'] for r in f['actions'] if r['physical_dispatch_confirmed']]
    assert physical == ['STRAFE_LEFT','TURN_RIGHT','BYPASS_FORWARD']
    post=f['actions'][2]['post_action']
    assert post['minimum_side_separation_m'] < .15 < post['blocker_center_lateral_separation_m']
    c=next(c for c in f['geometry'] if c['key']['sequence']==5985)
    left=next(s for s in c['sides'] if s['side']=='LEFT')
    assert left['pass_occupancy']==0 and left['pass_overlap_m']==0 and left['protected_capsule_clear']
    assert left['lateral_feasible'] is None  # Missing native history remains missing.
    pass_safe = (left['lateral_separation_m'] >= .15 and left['protected_capsule_clear'] is True
        and left['pass_occupancy'] == 0 and left['pass_overlap_m'] == 0.
        and left['target_recomputed_sequence'] == c['key']['sequence'])
    assert phase_from_admission(route_obstructed=True, pass_safe=pass_safe,
        repair_safe=left['lateral_feasible']) == Phase.PASS_OBSTACLE


@pytest.mark.parametrize('pass_safe,repair_safe', [(None,None),(False,None),(None,False)])
def test_missing_historical_admission_does_not_invent_healthy_wait(pass_safe,repair_safe):
    assert phase_from_admission(route_obstructed=True,pass_safe=pass_safe,repair_safe=repair_safe) is None


@pytest.mark.parametrize('fault', ['stop','bridge','session','delivery'])
def test_authoritative_mission_failures_stop_without_followup(tmp_path,monkeypatch,fault):
    b, _, _ = mission_bundle(tmp_path,monkeypatch)
    r, behavior, robot, events, _ = b
    if fault == 'stop':
        original=robot.stop
        robot.stop=lambda: {'ok':False} if motions(events) else original()
    elif fault == 'bridge':
        robot.on_motion=lambda:setattr(robot,'ready',False)
    elif fault == 'session':
        robot.on_motion=lambda:setattr(r.lidar_worker,'session','changed')
    else:
        original=r._execute_single_marvin_strafe
        def uncertain(**kwargs):
            value=original(**kwargs);value['delivery_uncertain']=True
            return value
        r._execute_single_marvin_strafe=uncertain
    result=run(r)
    assert result['state']=='BLOCKED' and not result['arrived_at_marvin']
    assert len(motions(events)) <= 1
    assert events[-1]=='stop' or fault=='stop'


def test_clear_side_jit_veto_switches_to_pass_after_fresh_observation(tmp_path,monkeypatch):
    scenes=[[ (.65,-.14)],[(.65,-.14)],None]
    b, _, _ = strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0,1.1)]*3),scenes)
    r, behavior, robot, events, _ = b
    original=behavior.execute_guarded_marvin_lateral_step
    flags={'changed':False}
    def strafe(**kwargs):
        if len(motions(events))==1 and not flags['changed']:
            scenes[1]=[(.65,-.25)];flags['changed']=True
        return original(**kwargs)
    behavior.execute_guarded_marvin_lateral_step=strafe
    result=run(r)
    vetoes=[x for x in result['local_avoidance_history'] if x.get('vetoed_before_transport')]
    assert len(vetoes)==1 and vetoes[0]['phase_after_veto']=='PASS_OBSTACLE'
    assert result['state']=='ARRIVED' and result['local_avoidance_actions']==2
    assert motions(events)[:2]==[('strafe',.08,1.),('forward',.1,.5)]


def test_blocked_wait_wakes_for_independently_safe_pass_only(tmp_path,monkeypatch):
    from test_marvin_blocked_wait import wait_bundle
    b,f,d,history,association,_ = wait_bundle(tmp_path,monkeypatch,change_after=.5)
    r, _, robot, events, _ = b
    old=robot.status
    robot.status=lambda:dict(old(),motion=dict(old()['motion'],linear_y=0.),motion_capabilities={'linear_y':True})
    f['changed']=False
    ctx=DetourContext(phase=Phase.BLOCKED_WAIT,committed_side='LEFT')
    entry=r.world_model.get_lidar_obstacles()
    result=r._wait_for_marvin_blocked_route(expected_session=r.lidar_worker.session,
        lidar=entry,association=association,execution_guard=lambda:True,
        diagnostics=d,history=history,detour_context=ctx)
    assert result['ok'] and result['reason']=='find_marvin_blocked_wait_phase_action_available'
    assert not motions(events) and r._marvin_alignment_observation is None
