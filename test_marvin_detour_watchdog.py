"""Offline watchdog regressions. Retained native history is never future authority."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import socket
import time

import pytest

from marvin_detour_watchdog import (
    DetourWatchdog, Passage, DETOUR_MAX_SECONDS, CLEAR_SIDE_MAX_UNPROVEN_ACTIONS,
    CLEAR_SIDE_MAX_ACTIONS, PASS_MAX_UNPROVEN_ACTIONS, PASS_MAX_ACTIONS,
    DETOUR_EMERGENCY_PHYSICAL_ACTIONS, MAX_EQUIVALENT_VETOES, MAX_PHASE_TRANSITIONS,
)
from marvin_obstacle_phases import DetourContext, Frontier, Phase, plan_phase_action
from test_marvin_local_bypass import mission_bundle, OPEN_LEFT, ASSOCIATION, pursuit_specs
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_find_marvin_closed_loop import run, motions, make_runtime
from test_marvin_authoritative_phases import veto_to_repair
from test_marvin_avoidance_planner_progress import scan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Watchdog validation must remain offline')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr('subprocess.Popen', denied)


def facts(seq=1, x=.65, sep=.25, *, safe=True, session='s', extent=None):
    return {'producer_session':session, 'acquisition_sequence':seq,
        'route':{'valid':True, 'route_to_marvin_obstructed':True,
            'blocking_obstacle_x_m':x, 'blocking_obstacle_y_m':-sep,
            'blocking_obstacle_longitudinal_extent_m':x+.02 if extent is None else extent},
        'local_bypass_candidates':{'LEFT':{'bypass_forward_permitted':safe}}}


def selection(seq=1, x=.65, sep=.25, *, safe=True):
    a=facts(seq,x,sep,safe=safe)
    return dict(a, committed_side='LEFT', local_bypass=a['local_bypass_candidates']['LEFT'])


def action(w, n, *, progress=False):
    before=.8 if not progress else .8-.025*n
    return w.physical_action().completed_traversal(selection(n*2+1,before)).reassess(
        facts(n*2+2,before-(.025 if progress else 0)),side='LEFT')


def continued_bundle(tmp_path,monkeypatch,steps):
    # Synthetic continuation: independently safe corridor at every action;
    # route clear after the chosen number. No commanded-distance simulation.
    b,flags,client=strafe_runtime(tmp_path,monkeypatch,
        pursuit_specs([(0,1.1)]*(steps+1)),[OPEN_LEFT]*(steps+1)+[None])
    read=b[0].world_model.get_lidar_obstacles
    def timestamped(**kwargs):
        b[4][0]+=1000
        return read(**kwargs)
    b[0].world_model.get_lidar_obstacles=timestamped
    return b,flags,client


def test_provisional_time_bound_allows_retained_mission():
    assert DETOUR_MAX_SECONDS == 120 and DETOUR_MAX_SECONDS > 34.78
    w=DetourWatchdog().enter('PASS_OBSTACLE',0)
    assert w.exhaustion(34.78) is None
    assert w.exhaustion(120)=='find_marvin_detour_watchdog_exhausted'


def test_total_timer_never_resets_on_phase_change_or_rejoin():
    w=DetourWatchdog().enter('CLEAR_SIDE',10)
    for phase,now in [('PASS_OBSTACLE',20),('CLEAR_SIDE',40),('BLOCKED_WAIT',60),('REJOIN',100)]:
        w=w.enter(phase,now)
        assert w.detour_started_monotonic==10 and w.phase_started_monotonic==now
    assert w.exhaustion(130)=='find_marvin_detour_watchdog_exhausted'


def test_later_obstacle_does_not_reset_mission_watchdog():
    w=DetourWatchdog().enter('CLEAR_SIDE',10).enter('REJOIN',20).enter('DIRECT',21)
    assert w.exhaustion(150) is None  # Normal direct pursuit remains unchanged.
    w=w.enter('CLEAR_SIDE',150)
    assert w.detour_started_monotonic==10
    assert w.exhaustion(150)=='find_marvin_detour_watchdog_exhausted'


def test_alignment_counts_emergency_actions_and_time_but_not_progress():
    w=DetourWatchdog().enter('CLEAR_SIDE',1).physical_action()
    assert w.total_detour_actions==1 and w.phase_actions==w.phase_stagnation_count==0
    assert w.exhaustion(121)=='find_marvin_detour_watchdog_exhausted'


@pytest.mark.parametrize('phase,limit,reason',[
    ('CLEAR_SIDE',CLEAR_SIDE_MAX_UNPROVEN_ACTIONS,'find_marvin_clear_side_stagnation_exhausted'),
    ('PASS_OBSTACLE',PASS_MAX_UNPROVEN_ACTIONS,'find_marvin_pass_stagnation_exhausted')])
def test_unproven_safe_phase_actions_are_bounded(phase,limit,reason):
    w=DetourWatchdog().enter(phase,0)
    for i in range(limit):
        assert w.exhaustion(i) is None
        w=action(w,i)
    assert w.exhaustion(limit)==reason


def test_fresh_outcome_is_allowed_before_last_action_stagnation_decision():
    w=replace(DetourWatchdog().enter('PASS_OBSTACLE',0),phase_stagnation_count=11)
    w=w.physical_action().completed_traversal(selection(1))
    assert w.pending_reassessment and w.exhaustion(1) is None
    w=w.reassess(facts(2,x=.625),side='LEFT')
    assert w.phase_stagnation_count==0 and w.exhaustion(2) is None


@pytest.mark.parametrize('fault',['missing','same_scan','session','side','lateral_change','cluster_change'])
def test_unreliable_or_noncomparable_geometry_never_credits_pass_progress(fault):
    w=DetourWatchdog().enter('PASS_OBSTACLE',0).physical_action().completed_traversal(selection())
    post=facts(2,x=.60);side='LEFT'
    if fault=='missing':post={}
    if fault=='same_scan':post['acquisition_sequence']=1
    if fault=='session':post['producer_session']='other'
    if fault=='side':side='RIGHT'
    if fault=='lateral_change':post['route']['blocking_obstacle_y_m']=-.4
    if fault=='cluster_change':post['route']['blocking_obstacle_longitudinal_extent_m']=.9
    w=w.reassess(post,side=side)
    assert w.phase_stagnation_count==1 and not w.pending_reassessment


def test_measured_pass_advancement_uses_obstacle_geometry_not_command_distance():
    w=DetourWatchdog().enter('PASS_OBSTACLE',0).completed_traversal(selection())
    w=w.reassess(facts(2,x=.60),side='LEFT')
    assert w.phase_stagnation_count==0 and w.last_progress=='measured_obstacle_relative_advancement'
    w=w.completed_traversal(selection(3,x=.60)).reassess(facts(4,x=.60),side='LEFT')
    assert w.phase_stagnation_count==1


def test_clear_side_progress_is_lateral_passage_feasibility():
    w=DetourWatchdog().enter('CLEAR_SIDE',0).completed_traversal(selection(sep=.14,safe=False))
    w=w.reassess(facts(2,sep=.18,safe=True),side='LEFT')
    assert w.phase_stagnation_count==0 and w.last_progress=='measured_passage_feasibility_improved'


@pytest.mark.parametrize('phase,limit,reason',[
    ('CLEAR_SIDE',CLEAR_SIDE_MAX_ACTIONS,'find_marvin_clear_side_attempts_exhausted'),
    ('PASS_OBSTACLE',PASS_MAX_ACTIONS,'find_marvin_pass_attempts_exhausted')])
def test_even_reported_progress_cannot_make_phase_attempts_unlimited(phase,limit,reason):
    w=DetourWatchdog().enter(phase,0)
    for i in range(limit):
        w=w.physical_action().completed_traversal(selection(i*2+1,sep=.2))
        w=w.reassess(facts(i*2+2,sep=.25 if phase=='CLEAR_SIDE' else .2,x=.625),side='LEFT')
    assert w.phase_stagnation_count==0
    assert w.exhaustion(1)==reason


def test_phase_entry_resets_only_local_stagnation():
    w=action(DetourWatchdog().enter('CLEAR_SIDE',0),0)
    w=w.enter('PASS_OBSTACLE',3)
    assert w.phase_actions==w.phase_stagnation_count==0
    assert w.total_detour_actions==1 and w.detour_started_monotonic==0


def test_equivalent_vetoes_do_not_count_physical_actions_and_are_bounded():
    w=DetourWatchdog().enter('PASS_OBSTACLE',0)
    for i in range(MAX_EQUIVALENT_VETOES):w=w.veto('BYPASS_FORWARD','LEFT','pass_unsafe')
    assert w.total_detour_actions==w.phase_actions==0
    assert w.exhaustion(1)=='find_marvin_equivalent_jit_veto_exhausted'


def test_alternating_vetoes_and_phase_cycles_are_bounded_without_time_progress():
    w=DetourWatchdog().enter('CLEAR_SIDE',0)
    for i in range(MAX_PHASE_TRANSITIONS):
        w=w.enter('PASS_OBSTACLE' if i%2==0 else 'CLEAR_SIDE',0)
        w=w.veto('BYPASS_FORWARD' if i%2==0 else 'STRAFE_LEFT','LEFT',str(i))
    assert w.total_detour_actions==0 and w.detour_started_monotonic==0
    assert w.exhaustion(0)=='find_marvin_detour_antispin_exhausted'


def test_emergency_ceiling_counts_alignment_and_survives_frozen_clock():
    w=DetourWatchdog().enter('CLEAR_SIDE',0)
    for i in range(DETOUR_EMERGENCY_PHYSICAL_ACTIONS):
        assert w.exhaustion(0) is None
        w=w.physical_action()
    assert w.exhaustion(0)=='find_marvin_detour_emergency_action_ceiling_exhausted'
    assert w.total_detour_actions==64


@pytest.mark.parametrize('clock',[float('nan'),float('inf'),-1])
def test_invalid_clock_fails_closed(clock):
    assert DetourWatchdog().enter('PASS_OBSTACLE',0).exhaustion(clock)=='find_marvin_detour_clock_invalid'


def test_watchdog_context_does_not_replace_exact_frontiers():
    e=Frontier('s',1,1791641209874566495)
    ctx=DetourContext().enter(Phase.CLEAR_SIDE,e,side='LEFT',now=10)
    assert ctx.record()['watchdog']['limits']['detour_seconds']==120
    ctx=ctx.retire(e)
    with pytest.raises(ValueError):ctx.completed(e,traversal=True)
    with pytest.raises(ValueError):Frontier('s',2,float(e.source_frame_stamp_ns))


def test_native_d68ede76_six_boundary_proposes_pass_without_historical_authority():
    f=json.loads((Path(__file__).parent/'test_fixtures/marvin_d68ede76_watchdog.json').read_text())
    state=copy.deepcopy(f['final_native_lidar']);state['received_monotonic_seconds']=time.monotonic();state['age_at_receipt_seconds']=0.
    legacy=plan_phase_action(state,f['final_association'],expected_session=state['producer_session'],
        allow_strafe=True,committed_side='LEFT',remaining_avoidance_actions=0)
    assert legacy['reason']=='find_marvin_local_avoidance_exhausted'
    new=plan_phase_action(state,f['final_association'],expected_session=state['producer_session'],allow_strafe=True,committed_side='LEFT')
    assert f['avoidance_count']==6 and new['phase']=='PASS_OBSTACLE' and new['action_type']=='BYPASS_FORWARD'
    assert new['blocker_center_separation_m']==pytest.approx(.21324750599334322)
    w=DetourWatchdog().enter('CLEAR_SIDE',0)
    for i,a in enumerate(f['actions']):
        w=w.enter(a['phase'],i).physical_action()
        if a['planning_selection']:w=w.completed_traversal(a['jit_selection']).reassess(a['post_target_association'] or {},side='LEFT')
    assert w.exhaustion(f['elapsed_seconds_at_old_boundary']) is None
    assert w.total_detour_actions==12 and w.phase_actions==5
    assert f['historical_limitations']  # No future action/geometry is fabricated.


@pytest.mark.parametrize('steps',[6,9])
def test_fresh_seventh_and_several_more_bypasses_rejoin_and_arrive(tmp_path,monkeypatch,steps):
    b,_,_=continued_bundle(tmp_path,monkeypatch,steps)
    result=run(b[0])
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==1+steps and result['local_bypass_actions']==steps
    assert result['legacy_avoidance_counter_only']
    assert [x['phase'] for x in result['navigation_phase_history']][-3:]==['REJOIN','DIRECT','ARRIVED']
    assert result['completed_forward_actions']>0
    rows=[x for x in result['history'] if x['state']=='AVOIDING']
    assert len({x['source_frame_stamp_ns'] for x in rows})==len(rows)
    for prev,next_ in zip(rows,rows[1:]):
        assert next_['source_frame_stamp_ns']>prev['source_frame_stamp_ns']
        assert next_['action_lidar_evidence'][1]>prev['action_lidar_evidence'][1]
        assert next_['result']['source_stamp_consumed'] and next_['result']['bridge_stop_confirmed']
        assert next_['result']['local_detour']['accepted']
    assert all(x['result']['approach_result']['forward_safety']['permitted'] for x in rows if x['result']['action_type']=='BYPASS_FORWARD')
    assert b[3][-1]=='stop'


@pytest.mark.parametrize('fault',['repeat_camera','freeze_lidar','unsafe','delivery'])
def test_no_grandfathered_seventh_action_or_safety_bypass(tmp_path,monkeypatch,fault):
    b,_,_=continued_bundle(tmp_path,monkeypatch,7);r,behavior,robot,events,_=b
    def after():
        if len(motions(events))==6:
            if fault=='repeat_camera':behavior.repeat_camera=True
            elif fault=='freeze_lidar':behavior.freeze_lidar=True
            elif fault=='unsafe':behavior.unsafe_forward=True
    robot.on_motion=after
    if fault=='delivery':
        old=r._execute_single_marvin_approach
        def uncertain(**kwargs):
            value=old(**kwargs)
            if len(motions(events))==6:value['delivery_uncertain']=True
            return value
        r._execute_single_marvin_approach=uncertain
    result=run(r)
    assert len(motions(events))==6
    assert not result['arrived_at_marvin'] and result['stop_result']['ok']
    assert result['reason']!='find_marvin_local_avoidance_exhausted'
    if fault=='delivery':assert result['reason']=='find_marvin_action_delivery_uncertain'


def test_constant_safe_passage_stops_by_watchdog_and_bridge_zero(tmp_path,monkeypatch):
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*40,[OPEN_LEFT])
    result=run(b[0]);w=result['detour_context']['watchdog']
    assert result['reason']=='find_marvin_pass_stagnation_exhausted'
    assert result['local_avoidance_actions']==1+PASS_MAX_UNPROVEN_ACTIONS
    assert w['phase_stagnation_count']==PASS_MAX_UNPROVEN_ACTIONS
    assert result['mission_outcome']=='safe_incomplete' and result['bridge_after_stop']['status']=='READY'
    assert result['stop_result']['ok'] and b[3][-1]=='stop'


def test_unproductive_clear_side_stops_by_phase_watchdog(tmp_path,monkeypatch):
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)]*20,[LEFT_OPEN])
    result=run(b[0])
    assert result['reason']=='find_marvin_clear_side_stagnation_exhausted'
    assert result['local_avoidance_actions']==CLEAR_SIDE_MAX_UNPROVEN_ACTIONS
    assert motions(b[3])==[('strafe',.08,1.)]*CLEAR_SIDE_MAX_UNPROVEN_ACTIONS and result['stop_result']['ok']


def test_deadline_in_camera_reacquisition_does_not_dispatch_stale_frame(tmp_path,monkeypatch):
    b,_,_=continued_bundle(tmp_path,monkeypatch,7);r,behavior,robot,events,clock=b
    behavior.on_observe=lambda:clock.__setitem__(0,clock[0]+121_000_000_000) if motions(events) else None
    result=run(r)
    assert result['reason']=='find_marvin_detour_watchdog_exhausted'
    assert len(motions(events))==1 and result['stop_result']['ok']


def test_deadline_during_jit_rejects_dispatch_and_stops(tmp_path,monkeypatch):
    b,_,_=continued_bundle(tmp_path,monkeypatch,7);r,behavior,robot,events,clock=b
    old=behavior.execute_single_marvin_approach_step
    def delayed(**kwargs):
        if kwargs.get('local_selection_validator'):clock[0]+=121_000_000_000
        return old(**kwargs)
    behavior.execute_single_marvin_approach_step=delayed
    result=run(r)
    assert result['reason']=='find_marvin_detour_watchdog_exhausted'
    assert len(motions(events))==1 and result['stop_result']['ok']


def test_alignment_does_not_reset_phase_watchdog_side_or_time(tmp_path,monkeypatch):
    specs=pursuit_specs([(0,1.1),(100,1.1),(0,1.1)])
    b,_,_=strafe_runtime(tmp_path,monkeypatch,specs,[OPEN_LEFT,OPEN_LEFT,OPEN_LEFT,None])
    result=run(b[0]);w=result['detour_context']['watchdog']
    assert result['state']=='ARRIVED'
    assert result['local_avoidance_actions']==2 and w['total_detour_actions']==3
    assert [h['state'] for h in result['history']][:3]==['AVOIDING','ALIGNING','AVOIDING']
    assert result['local_avoidance_history'][1]['selection']['committed_side']=='LEFT'


def test_jit_veto_returns_to_repair_without_action_slot_or_old_stamp(tmp_path,monkeypatch):
    b,_=veto_to_repair(tmp_path,monkeypatch)
    result=run(b[0]);w=result['detour_context']['watchdog']
    assert result['state']=='ARRIVED'
    assert w['total_veto_count']==1 and w['total_detour_actions']==result['local_avoidance_actions']
    rows=result['history'];i=next(i for i,x in enumerate(rows) if x['result'].get('pre_transport_jit_veto'))
    assert not rows[i]['motion_executed'] and rows[i+1]['source_frame_stamp_ns']>rows[i]['source_frame_stamp_ns']
    assert rows[i+1]['result']['action_type']=='STRAFE_LEFT'


def test_safe_pass_becomes_unsafe_and_repairs_then_rejoins(tmp_path,monkeypatch):
    specs=pursuit_specs([(0,1.1)]*9)
    b,_,_=strafe_runtime(tmp_path,monkeypatch,specs,[OPEN_LEFT]*7+[[ (.65,-.14),(0,1.2),(0,-.65)],None])
    result=run(b[0])
    assert result['state']=='ARRIVED',result['reason']
    phases=[x['phase'] for x in result['navigation_phase_history']]
    assert phases[:3]==['CLEAR_SIDE','PASS_OBSTACLE','CLEAR_SIDE']
    assert result['local_avoidance_actions']==8


def test_no_safe_action_still_uses_existing_blocked_wait(tmp_path,monkeypatch):
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*3,[OPEN_LEFT+[(0,.3)]])
    result=run(b[0])
    assert result['reason']=='find_marvin_blocked_wait_exhausted' and result['blocked_wait_recheck_count']==12
    assert not motions(b[3])


def test_ordinary_arrival_is_unchanged_and_does_not_start_watchdog(tmp_path,monkeypatch):
    b=make_runtime(tmp_path,monkeypatch,[(0,.6),(0,.5)])
    result=run(b[0])
    assert result['state']=='ARRIVED' and result['local_avoidance_actions']==0
    assert result['detour_context']['watchdog']['detour_started_monotonic'] is None


def test_045_protected_capsule_and_015_center_admission_still_apply():
    for points in [OPEN_LEFT+[(.40,.2)],[ (.65,-.149),(0,1.2),(0,-.65)]]:
        state=scan(points)
        result=plan_phase_action(state,ASSOCIATION,expected_session='test',allow_strafe=True,committed_side='LEFT')
        assert result['action_type']!='BYPASS_FORWARD'
    state=scan([(.62,-.155437),(.70,-.141939),(0,1.2),(0,-.65)])
    result=plan_phase_action(state,ASSOCIATION,expected_session='test',allow_strafe=True,committed_side='LEFT')
    assert result['action_type']=='BYPASS_FORWARD' and result['minimum_side_separation_m']<.15


def test_runtime_equivalent_zero_transport_veto_spin_is_bounded(tmp_path,monkeypatch):
    b,_,_=continued_bundle(tmp_path,monkeypatch,20);r,behavior,robot,events,_=b
    flags={'jit':False};read=r.world_model.get_lidar_obstacles
    def lidar(**kwargs):
        value=read(**kwargs)
        if flags['jit']:
            for p in value['local_motion_geometry']['points']:
                if .60<p['x_m']<.70:p['y_m']=-.14
        return value
    r.world_model.get_lidar_obstacles=lidar
    original=behavior.execute_single_marvin_approach_step
    def veto(**kwargs):
        if kwargs.get('local_selection_validator'):flags['jit']=True
        return original(**kwargs)
    behavior.execute_single_marvin_approach_step=veto
    behavior.on_observe=lambda:flags.update(jit=False)
    result=run(r);w=result['detour_context']['watchdog']
    assert result['reason']=='find_marvin_equivalent_jit_veto_exhausted'
    assert result['local_avoidance_actions']==w['total_detour_actions']==1
    assert w['total_veto_count']==MAX_EQUIVALENT_VETOES
    assert motions(events)==[('strafe',.08,1.)] and result['stop_result']['ok']
    vetoes=[x for x in result['history'] if x['result'].get('pre_transport_jit_veto')]
    assert len(vetoes)==6 and len({x['source_frame_stamp_ns'] for x in vetoes})==6
    assert all(x['source_frame_stamp_ns'] in r._marvin_alignment_consumed_source_frame_stamps for x in vetoes)


def test_runtime_fast_phase_cycles_hit_total_antispin_bound(tmp_path,monkeypatch):
    repair=[(.65,-.14),(0,1.2),(0,-.65)]
    scenes=[OPEN_LEFT]+[OPEN_LEFT if i%2==0 else repair for i in range(40)]
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*50,scenes)
    result=run(b[0]);w=result['detour_context']['watchdog']
    assert result['reason']=='find_marvin_detour_antispin_exhausted'
    assert w['phase_transitions']==MAX_PHASE_TRANSITIONS
    assert w['total_detour_actions']<DETOUR_EMERGENCY_PHYSICAL_ACTIONS
    assert result['mission_outcome']=='safe_incomplete' and result['stop_result']['ok']


def test_runtime_emergency_backstop_includes_alignment(tmp_path,monkeypatch):
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]+[(100,1.1)]*100,[OPEN_LEFT])
    result=run(b[0]);w=result['detour_context']['watchdog']
    assert result['reason']=='find_marvin_detour_emergency_action_ceiling_exhausted'
    assert w['total_detour_actions']==DETOUR_EMERGENCY_PHYSICAL_ACTIONS
    assert result['local_avoidance_actions']==1 and len(motions(b[3]))==64
    assert w['phase_actions']==1 and result['stop_result']['ok']


def test_blocked_wait_also_remains_inside_total_detour_deadline(tmp_path,monkeypatch):
    b,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*10,[OPEN_LEFT,OPEN_LEFT+[(0,.3)]])
    r,behavior,robot,events,clock=b;wait=r._wait_for_marvin_blocked_route
    def delayed(**kwargs):
        clock[0]+=119_000_000_000
        return wait(**kwargs)
    r._wait_for_marvin_blocked_route=delayed
    result=run(r)
    assert result['reason']=='find_marvin_detour_watchdog_exhausted'
    assert result['local_avoidance_actions']==1 and result['stop_result']['ok']


@pytest.mark.parametrize('fault',['stop','bridge'])
def test_stop_and_bridge_failures_still_fail_closed_after_six(tmp_path,monkeypatch,fault):
    b,_,_=continued_bundle(tmp_path,monkeypatch,9);r,behavior,robot,events,_=b
    original=robot.stop
    if fault=='stop':robot.stop=lambda:{'ok':False} if len(motions(events))>=6 else original()
    else:robot.on_motion=lambda:setattr(robot,'ready',False) if len(motions(events))>=6 else None
    result=run(r)
    assert len(motions(events))==6
    assert not result['arrived_at_marvin'] and result['mission_outcome']=='safe_failure'
    assert result['reason']=='find_marvin_stop_or_bridge_failed'


def test_action_window_must_fit_deadline_without_granting_admission():
    w=DetourWatchdog().enter('PASS_OBSTACLE',0)
    assert w.action_fits_deadline(119,.5)
    assert not w.action_fits_deadline(119.75,.5)
    assert w.total_detour_actions==0


def test_jit_cannot_start_action_that_would_cross_detour_deadline(tmp_path,monkeypatch):
    b,_,_=continued_bundle(tmp_path,monkeypatch,7);r,behavior,robot,events,clock=b
    old=behavior.execute_single_marvin_approach_step
    def delayed(**kwargs):
        if kwargs.get('local_selection_validator'):clock[0]+=118_600_000_000
        return old(**kwargs)
    behavior.execute_single_marvin_approach_step=delayed
    result=run(r)
    assert result['reason']=='find_marvin_detour_watchdog_exhausted'
    assert motions(events)==[('strafe',.08,1.)] and result['stop_result']['ok']
