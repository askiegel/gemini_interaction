"""Offline real executors: pre-transport vetoes cannot create motion authority."""
import copy
import json
import math
from pathlib import Path
import socket
import time

import pytest

from marvin_blocked_wait import (
    PRE_TRANSPORT_JIT_WAIT_REASONS, explicit_pre_transport_jit_veto,
    pre_transport_jit_veto_evidence,
)
from test_find_marvin_closed_loop import run, motions
from test_marvin_blocked_wait import wait_bundle
from test_marvin_lateral_avoidance import strafe_runtime
from test_marvin_local_bypass import OPEN_LEFT


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No live network/services in JIT-veto tests')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def evidence_bundle(tmp_path, monkeypatch):
    b, flags, diagnostics, history, association, execute = wait_bundle(tmp_path, monkeypatch)
    r, _, robot, _, _ = b
    status = robot.status
    robot.status = lambda: dict(status(), motion=dict(status()['motion'], linear_y=0.))
    sample = r.world_model.get_lidar_obstacles()
    association = dict(association, producer_session=r.lidar_worker.session,
        acquisition_sequence=sample['acquisition_sequence'], ok=True)
    stamp = 1791504411407983987
    r._marvin_alignment_consumed_source_frame_stamps.add(stamp)
    result = {'ok':False,'reason':'marvin_local_detour_jit_veto','motion_executed':False,
        'actions_executed':0,'source_frame_stamp_ns':stamp,'source_stamp_consumed':True,
        'stop_result':{'ok':True},'pre_transport_jit_veto':pre_transport_jit_veto_evidence(sample, association)}
    def admit(value=None, guard=lambda:True):
        return r._marvin_jit_veto_wait_evidence(result if value is None else value,
            stamp=stamp, expected_session=r.lidar_worker.session,
            planning_sequence=sample['acquisition_sequence']-1, execution_guard=guard)
    return b, result, admit


def test_explicit_negative_dispatch_evidence_requires_every_field(tmp_path, monkeypatch):
    b, result, admit = evidence_bundle(tmp_path, monkeypatch)
    admitted = admit()
    assert admitted and admitted['snapshot']['acquisition_sequence'] == result['pre_transport_jit_veto']['lidar_snapshot']['acquisition_sequence']
    assert not motions(b[3]) and b[3][-1] == 'stop'
    assert b[0]._marvin_alignment_consumed_source_frame_stamps == {result['source_frame_stamp_ns']}
    assert PRE_TRANSPORT_JIT_WAIT_REASONS == {'marvin_local_detour_jit_veto','marvin_local_bypass_jit_veto'}


@pytest.mark.parametrize('field,value', [
    ('motion_executed',True), ('motion_executed',None), ('actions_executed',1),
    ('actions_executed',False), ('transport_attempted',True), ('delivery_uncertain',True),
    ('confirmed_forwarded',True), ('forwarded',True), ('physical_dispatch_confirmed',True),
    ('transport_result',{'ok':False}), ('lateral_result',{}), ('turn_result',{'ok':True}),
    ('forward_result',{'transport_attempted':False}), ('error','unexpected exception'),
    ('transport_error','unknown delivery'), ('interrupted',True),
    ('reason','marvin_lateral_transport_failed'), ('reason','marvin_lateral_step_exception'),
    ('reason','marvin_motion_observation_stale_or_preempted'), ('reason',[]),
    ('pre_transport_jit_veto',None), ('source_frame_stamp_ns',1791504411407984000.0),
    ('source_stamp_consumed',False), ('stop_result',{'ok':False}),
    ('full_step_completed',True), ('physical_dispatch_count',1), ('dispatch_opportunities',1),
])
def test_contradictory_or_uncertain_results_cannot_wait(tmp_path, monkeypatch, field, value):
    b, result, admit = evidence_bundle(tmp_path, monkeypatch)
    result[field] = value
    assert admit() is None
    assert not motions(b[3])


@pytest.mark.parametrize('fault', ['nested_transport','nested_uncertain','certificate_phase',
    'certificate_attempted','session','stale','sequence','association_sequence','coverage',
    'geometry','route_clear','owner','stop','bridge','ros','streaming','linear_y',
    'missing_linear_y','active_forward','pending_forward','producer_stopped'])
def test_evidence_health_ownership_stop_and_stream_gates_fail_closed(tmp_path, monkeypatch, fault):
    b, result, admit = evidence_bundle(tmp_path, monkeypatch)
    r, _, robot, _, _ = b
    e=result['pre_transport_jit_veto'];s=e['lidar_snapshot'];a=e['target_association']
    guard=lambda:True
    if fault=='nested_transport':result['lateral_step']={'lateral_result':{'transport_attempted':True}}
    if fault=='nested_uncertain':result['approach_result']={'delivery_uncertain':True}
    if fault=='certificate_phase':e['phase']='after_transport_call'
    if fault=='certificate_attempted':e['transport_attempted']=True
    if fault=='session':s['producer_session']='other'
    if fault=='stale':s['received_monotonic_seconds']-=.301
    if fault=='sequence':s['acquisition_sequence']=-1
    if fault=='association_sequence':a['acquisition_sequence']+=1
    if fault=='coverage':s['local_motion_geometry']['sectors']['rear']['valid_sample_count']=0
    if fault=='geometry':s['local_motion_geometry']['valid']=False
    if fault=='route_clear':s['local_motion_geometry']['points']=[]
    if fault=='owner':guard=lambda:False
    if fault=='stop':robot.stop=lambda:{'ok':False}
    if fault in {'bridge','ros','streaming','linear_y','missing_linear_y'}:
        old=robot.status
        def status():
            v=old()
            if fault=='bridge':v['status']='ERROR'
            if fault=='ros':v['ros_ready']=False
            if fault=='streaming':v['motion']['streaming']=True
            if fault=='linear_y':v['motion']['linear_y']=.08
            if fault=='missing_linear_y':v['motion'].pop('linear_y')
            return v
        robot.status=status
    if fault in {'active_forward','pending_forward'}:
        r.forward_interlock.status=lambda:{'active_forward':fault=='active_forward','pending_forward':fault=='pending_forward'}
    if fault=='producer_stopped':r.lidar_worker.running=False
    assert admit(guard=guard) is None
    assert not motions(b[3])


def veto_mission(tmp_path, monkeypatch, *, clear_after=2, mutate=None, resume_scene=None):
    b, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,1.1)]*4+[(0,1.05)], [OPEN_LEFT])
    r, behavior, robot, events, clock = b
    f={'jit':False,'waiting':False,'removed':False,'checks':0,'entry':None,'resume':0,'veto':None}
    read=r.world_model.get_lidar_obstacles
    def lidar(**kw):
        s=read(**kw)
        if f['removed']:
            s['local_motion_geometry']['points']=s['local_motion_geometry']['points'][:-len(OPEN_LEFT)]
            s['local_motion_geometry']['points'].extend({'x_m':x,'y_m':y} for x,y in (resume_scene or []))
        elif f['jit']:
            s['local_motion_geometry']['points'].append({'x_m':0.,'y_m':.3})
        return s
    r.world_model.get_lidar_obstacles=lidar
    execute=behavior.execute_guarded_marvin_lateral_step
    def guarded(**kw):
        f['jit']=True
        value=execute(**kw)
        f['veto']=copy.deepcopy(value)
        if mutate:mutate(value, r)
        return value
    behavior.execute_guarded_marvin_lateral_step=guarded
    publish=r._publish_behavior_tracking
    def telemetry(d):
        if d.get('state')=='BLOCKED_WAIT':
            f.update(waiting=True,checks=d['blocked_wait_recheck_count'])
            if f['entry'] is None:f['entry']=copy.deepcopy(d)
        publish(d)
    r._publish_behavior_tracking=telemetry
    sleep=time.sleep
    def tick(seconds):
        if f['waiting']:
            assert not motions(events) and robot.status()['motion']['linear_y']==0
            if clear_after is not None and f['checks']>=clear_after-1:f['removed']=True
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep',tick)
    def semantic(*,minimum_source_frame_stamp_ns):
        f['waiting']=False;f['resume']+=1
        events.append('jit_wait_semantic')
        behavior._clear_marvin_v2_tracker_episode()
        value=behavior.observe_find_marvin_v2()
        assert value['preview_result']['identity_source_frame_stamp_ns']>minimum_source_frame_stamp_ns
        return value
    behavior.reacquire_find_marvin_v2=semantic
    behavior.on_observe=lambda:setattr(r,'_control_generation',r._control_generation+1) if f['removed'] and motions(events) else None
    captured={};normal=r._execute_normal_marvin_find_mission
    def capture(*args,**kw):
        captured['result']=normal(*args,**kw)
        return captured['result']
    r._execute_normal_marvin_find_mission=capture
    return b,f,captured


def test_real_pre_transport_veto_waits_then_resumes_forward_with_new_exact_stamp(tmp_path, monkeypatch):
    b,f,c=veto_mission(tmp_path, monkeypatch)
    b[1].source_offset_ns=1791504411407983987-b[4][0]
    run(b[0]);result=c['result']
    assert f['entry'] is not None and f['resume']==1
    assert result['state']=='STOPPED' and result['stop_result']['ok']
    assert result['blocked_wait_reason']=='marvin_local_detour_jit_veto'
    veto=result['history'][0];action=veto['result']
    assert action['reason']=='marvin_local_detour_jit_veto' and action['source_stamp_consumed']
    assert not action['motion_executed'] and action['actions_executed']==0
    assert f['veto']['lateral_result'] is None
    sequence=action['pre_transport_jit_veto']['lidar_snapshot']['acquisition_sequence']
    assert f['entry']['blocked_wait_initial_lidar_sequence']==sequence
    assert sequence>veto['observation']['local_avoidance_selection']['acquisition_sequence']
    assert result['local_avoidance_actions']==result['local_bypass_actions']==0
    assert motions(b[3])==[('forward',.1,.5)]
    assert result['blocked_wait_recheck_count']==2
    assert [row['camera_recheck_performed'] for row in result['blocked_wait_history']]==[False,True]
    forward=result['history'][1]
    assert forward['state']=='ADVANCING' and forward['result']['full_step_completed']
    old=veto['source_frame_stamp_ns'];new=forward['source_frame_stamp_ns']
    assert type(old) is type(new) is int
    assert old<forward['observation']['identity_source_frame_stamp_ns']<new
    assert b[0]._marvin_alignment_consumed_source_frame_stamps=={old,new}
    assert result['local_avoidance_history'][0]['actual_route_occupancy'] is None
    assert result['local_avoidance_history'][0]['vetoed_before_transport']
    assert not result['local_avoidance_history'][0]['dispatched']
    assert result['final_observation']['source_frame_stamp_ns']>new
    assert all(not row['motion_executed'] and not row['source_stamp_consumed'] for row in result['blocked_wait_history'])
    assert result['max_local_avoidance_actions']==6


def test_unchanged_veto_geometry_exhausts_without_gemini_or_budget(tmp_path, monkeypatch):
    b,f,c=veto_mission(tmp_path, monkeypatch, clear_after=None)
    run(b[0]);result=c['result']
    assert result['reason']=='find_marvin_blocked_wait_exhausted' and result['mission_outcome']=='safe_incomplete'
    assert result['blocked_wait_recheck_count']==12 and f['resume']==0
    assert not motions(b[3])
    assert len(b[0]._marvin_alignment_consumed_source_frame_stamps)==1
    assert result['local_avoidance_actions']==result['local_bypass_actions']==0
    assert all(not h['camera_recheck_performed'] for h in result['blocked_wait_history'])


@pytest.mark.parametrize('fault',['uncertain','transport','exception','missing_certificate'])
def test_real_veto_with_uncertain_or_missing_contract_remains_terminal(tmp_path,monkeypatch,fault):
    def mutate(value,r):
        if fault=='uncertain':value['delivery_uncertain']=True
        if fault=='transport':value['lateral_result']={'transport_attempted':True,'transport_result':{'ok':False}}
        if fault=='exception':value['error']='unresolved exception'
        if fault=='missing_certificate':value.pop('pre_transport_jit_veto')
    b,f,c=veto_mission(tmp_path,monkeypatch,mutate=mutate)
    run(b[0]);result=c['result']
    assert f['entry'] is None and result['blocked_wait_recheck_count']==0
    assert result['reason']=='marvin_local_detour_jit_veto'
    assert not motions(b[3]) and result['stop_result']['ok']


def test_proof_harness_veto_does_not_automatically_wait(tmp_path,monkeypatch):
    b,f,c=veto_mission(tmp_path,monkeypatch)
    b[0]._set_runtime_state('IDLE')
    result=b[0].execute_find_marvin_live_proof_step(max_physical_actions=1)
    assert f['entry'] is None and not motions(b[3])
    assert 'controller_result' in result, result
    controller=result['controller_result']
    assert controller['reason']=='marvin_local_detour_jit_veto'
    assert 'blocked_wait_history' not in controller
    assert result['proof_state']=='FAILED_LOCKED'


def test_bypass_allowlist_needs_same_explicit_negative_dispatch_contract(tmp_path,monkeypatch):
    b,result,admit=evidence_bundle(tmp_path,monkeypatch)
    result['reason']='marvin_local_bypass_jit_veto'
    assert admit() is not None
    result['approach_result']={'forward_result':{'transport_attempted':True}}
    assert admit() is None


def test_live_three_action_sequence_jit_5958_waits_then_direct_forward(tmp_path, monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_jit_veto_blocked_wait.json').read_text())
    baseline=fixture['blocked_baseline']
    # Match recorded occupancy/range/overlap using calibrated synthetic returns.
    # Subsequent geometry deliberately crosses the unchanged material-progress
    # threshold between planning and JIT. The real selector and executors decide.
    y=-.089247
    x=math.sqrt(baseline['blocker_range_m']**2-y*y)
    scenes=[[(x,y)]*baseline['occupancy']+[(0.,1.05),(0.,-.64)],
            [(.60,-.12)]*20+[(0.,1.05),(0.,-.64)]]
    b,_,_=strafe_runtime(tmp_path,monkeypatch,
        [(e,1.44) for e in fixture['camera_horizontal_errors']],scenes)
    r,behavior,robot,events,clock=b
    flags={'veto':False,'waiting':False,'removed':False,'checks':0,'entry':None,'semantic':0,'reads':0,'wait_sequence':5958}
    read=r.world_model.get_lidar_obstacles
    def lidar(**kw):
        value=read(**kw)
        if flags['removed']:
            value['local_motion_geometry']['points']=value['local_motion_geometry']['points'][:-len(scenes[-1])]
        elif flags['veto']:
            for p in value['local_motion_geometry']['points'][-len(scenes[-1]):-2]:
                p.update(y_m=-.10,distance_m=math.hypot(.60,.10),robot_bearing_deg=math.degrees(math.atan2(-.10,.60)))
            value['local_motion_geometry']['points'].append({'x_m':0.,'y_m':.3})
        if len(motions(events))>=3:
            if flags['waiting'] or flags['removed']:
                flags['wait_sequence']+=1
                value['acquisition_sequence']=flags['wait_sequence']
            elif flags['veto']:
                value['acquisition_sequence']=5958
            else:
                flags['reads']+=1
                value['acquisition_sequence']=min(5957,5953+flags['reads'])
        return value
    r.world_model.get_lidar_obstacles=lidar
    guarded=behavior.execute_guarded_marvin_lateral_step
    def execute(**kw):
        if len(motions(events))==3:flags['veto']=True
        return guarded(**kw)
    behavior.execute_guarded_marvin_lateral_step=execute
    publish=r._publish_behavior_tracking
    def telemetry(d):
        if d.get('state')=='BLOCKED_WAIT':
            flags.update(waiting=True,checks=d['blocked_wait_recheck_count'])
            if flags['entry'] is None:flags['entry']=copy.deepcopy(d)
        publish(d)
    r._publish_behavior_tracking=telemetry
    sleep=time.sleep
    def tick(seconds):
        if flags['waiting']:
            assert len(motions(events))==3 and robot.status()['motion']['linear_y']==0
            if flags['checks']>=1:flags['removed']=True
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep',tick)
    def reacquire(*,minimum_source_frame_stamp_ns):
        flags['semantic']+=1;flags['waiting']=False
        behavior._clear_marvin_v2_tracker_episode()
        value=behavior.observe_find_marvin_v2()
        assert value['preview_result']['identity_source_frame_stamp_ns']>minimum_source_frame_stamp_ns
        return value
    behavior.reacquire_find_marvin_v2=reacquire
    behavior.on_observe=lambda:setattr(r,'_control_generation',r._control_generation+1) if len(motions(events))==4 else None
    capture={};normal=r._execute_normal_marvin_find_mission
    def record(*args,**kw):
        capture['result']=normal(*args,**kw)
        return capture['result']
    r._execute_normal_marvin_find_mission=record
    behavior.source_offset_ns=1791504407964761583-clock[0]
    run(r);result=capture['result']
    assert motions(events)==[('strafe',.08,1.),('turn','RIGHT',.25,.5),('turn','RIGHT',.25,.5),('forward',.1,.5)],result['reason']
    route=result['history'][0]['observation']['arrival']['route']
    assert route['route_occupancy']==21
    assert route['corridor_overlap_m']==pytest.approx(baseline['overlap_m'],abs=.001)
    assert route['blocking_obstacle_distance_m']==pytest.approx(baseline['blocker_range_m'])
    veto=result['history'][3]
    assert veto['observation']['local_avoidance_selection']['action_type']=='STRAFE_LEFT'
    assert veto['observation']['local_avoidance_selection']['acquisition_sequence']==5957
    assert veto['result']['reason']=='marvin_local_detour_jit_veto'
    assert veto['result']['source_stamp_consumed'] and not veto['result']['motion_executed']
    assert veto['result']['lateral_step']['lateral_result'] is None
    assert flags['entry']['blocked_wait_initial_lidar_sequence']==5958
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==0
    assert result['actions_executed']==4 and result['max_local_avoidance_actions']==6
    assert [h['camera_recheck_performed'] for h in result['blocked_wait_history']]==[False,True]
    assert flags['semantic']==1
    forward=result['history'][4]
    assert forward['state']=='ADVANCING' and forward['result']['full_step_completed']
    assert veto['source_frame_stamp_ns']<forward['observation']['identity_source_frame_stamp_ns']<forward['source_frame_stamp_ns']
    assert len(r._marvin_alignment_consumed_source_frame_stamps)==5 # Four motions + veto.
    assert result['stop_result']['ok'] and result['final_observation']['source_frame_stamp_ns']>forward['source_frame_stamp_ns']
    assert result['local_avoidance_history'][1]['actual_route_occupancy'] is None
    replay={'fixture_origin':fixture['origin'],'primitives':fixture['completed_primitives']+['FORWARD'],
        'planning_sequence':5957,'wait_entry_sequence':5958,'rechecks':result['blocked_wait_history'],
        'veto_consumed_stamp':str(veto['source_frame_stamp_ns']),
        'new_semantic_stamp':str(forward['observation']['identity_source_frame_stamp_ns']),
        'new_action_stamp':str(forward['source_frame_stamp_ns']),
        'avoidance_count':1,'bypass_count':0,'zero_veto_transport':True,
        'ordinary_forward':True,'stop_confirmed':True}
    (tmp_path/'live-case-replay-summary.json').write_text(json.dumps(replay,indent=2))


def test_real_bypass_outer_guard_veto_enters_existing_wait_without_bypass_slot(tmp_path, monkeypatch):
    from test_marvin_local_bypass import mission_bundle
    b,_,_=mission_bundle(tmp_path,monkeypatch)
    r,behavior,robot,events,_=b
    primitive=behavior.execute_single_marvin_approach_step
    read=r.world_model.get_lidar_obstacles
    def changed_at_guard(**kwargs):
        if kwargs.get('local_selection_validator'):
            calls=[0]
            def scan(**kw):
                value=read(**kw);calls[0]+=1
                if calls[0]>=2:value['local_motion_geometry']['points'].append({'x_m':.40,'y_m':0.})
                return value
            r.world_model.get_lidar_obstacles=scan
        return primitive(**kwargs)
    behavior.execute_single_marvin_approach_step=changed_at_guard
    result=run(r)
    assert result['reason']=='find_marvin_blocked_wait_exhausted',result['reason']
    assert result['blocked_wait_reason']=='marvin_local_bypass_jit_veto'
    assert motions(events)==[('strafe',.08,1.)]
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==0
    veto=result['history'][1]['result']
    assert veto['source_stamp_consumed'] and veto['pre_transport_jit_veto']['transport_attempted'] is False
    assert result['blocked_wait_initial_lidar_sequence']==veto['pre_transport_jit_veto']['lidar_snapshot']['acquisition_sequence']
    assert len(r._marvin_alignment_consumed_source_frame_stamps)==2
    assert result['previous_action_type']=='STRAFE_LEFT'


def test_client_guard_failure_after_motion_method_invocation_is_not_certified(tmp_path,monkeypatch):
    scenes=[OPEN_LEFT]
    b,_,client=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*3,scenes)
    request=client._request
    def changed(method,path,payload=None):
        value=request(method,path,payload)
        if method=='GET':scenes[0]=OPEN_LEFT+[(0.,.3)]
        return value
    client._request=changed
    result=run(b[0])
    assert result['state']=='BLOCKED' and result['blocked_wait_recheck_count']==0
    assert not motions(b[3])
    assert result['history'][0]['result']['source_stamp_consumed']
    assert 'pre_transport_jit_veto' not in result['history'][0]['result']


@pytest.mark.parametrize('fault',['session_after_stop','stream_after_stop','malformed_sector','malformed_points'])
def test_post_stop_admission_is_rechecked_and_malformed_geometry_fails_closed(tmp_path,monkeypatch,fault):
    b,result,admit=evidence_bundle(tmp_path,monkeypatch)
    r,_,robot,_,_=b
    stop=robot.stop
    def stopped():
        value=stop()
        if fault=='session_after_stop':r.lidar_worker.session='new'
        if fault=='stream_after_stop':r.forward_interlock.status=lambda:{'active_forward':True,'pending_forward':False}
        return value
    robot.stop=stopped
    if fault=='malformed_sector':result['pre_transport_jit_veto']['lidar_snapshot']['local_motion_geometry']['sectors']['rear']=1
    if fault=='malformed_points':result['pre_transport_jit_veto']['lidar_snapshot']['local_motion_geometry']['points']=[1]
    assert admit() is None
    assert not motions(b[3])


def test_changed_but_obstructed_geometry_returns_to_shared_avoidance_selector(tmp_path,monkeypatch):
    b,f,c=veto_mission(tmp_path,monkeypatch,resume_scene=[(.62,-.30),(0.,1.2),(0.,-.65)])
    run(b[0]);result=c['result']
    assert f['resume']==1 and result['blocked_wait_resume_reason']=='find_marvin_blocked_wait_phase_action_available'
    assert motions(b[3])==[('forward',.1,.5)]
    assert result['history'][1]['state']=='AVOIDING'
    assert result['history'][1]['result']['full_step_completed']
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==1
    assert result['stop_result']['ok'] and result['final_observation']['source_frame_stamp_ns']>result['history'][1]['source_frame_stamp_ns']


def test_resume_cannot_reuse_consumed_veto_frame_as_semantic_authority(tmp_path,monkeypatch):
    b,f,c=veto_mission(tmp_path,monkeypatch)
    def replay(*,minimum_source_frame_stamp_ns):
        b[1]._clear_marvin_v2_tracker_episode()
        value=b[1].observe_find_marvin_v2()
        value['preview_result']['identity_source_frame_stamp_ns']=minimum_source_frame_stamp_ns
        return value
    b[1].reacquire_find_marvin_v2=replay
    run(b[0]);result=c['result']
    assert not motions(b[3]) and result['stop_result']['ok']
    assert len(b[0]._marvin_alignment_consumed_source_frame_stamps)==1
    assert result['local_avoidance_actions']==result['local_bypass_actions']==0
    assert result['reason']=='find_marvin_blocked_wait_fresh_identity_required'


def test_bypass_interlock_failure_is_not_a_deterministic_geometry_certificate(tmp_path,monkeypatch):
    from test_marvin_local_bypass import mission_bundle
    b,_,_=mission_bundle(tmp_path,monkeypatch)
    r,behavior,robot,events,_=b
    primitive=behavior.execute_single_marvin_approach_step
    def blocked_interlock(**kwargs):
        if kwargs.get('local_selection_validator'):
            robot.forward_interlock.refresh=lambda:(False,'unresolved_interlock_failure')
        return primitive(**kwargs)
    behavior.execute_single_marvin_approach_step=blocked_interlock
    result=run(r)
    assert result['reason']=='marvin_local_bypass_jit_veto'
    assert result['blocked_wait_recheck_count']==0
    assert motions(events)==[('strafe',.08,1.)]
    assert 'pre_transport_jit_veto' not in result['history'][1]['result']
    assert result['stop_result']['ok'] and result['local_bypass_actions']==0
