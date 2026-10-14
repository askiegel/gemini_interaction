"""Offline real final transport boundary; retained history is never authority."""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from robot_bridge.client import RobotBridgeClient
from test_find_marvin_closed_loop import run, motions, make_runtime
from test_marvin_local_bypass import mission_bundle, OPEN_LEFT, pursuit_specs
from test_marvin_lateral_avoidance import strafe_runtime
from test_marvin_jit_veto_blocked_wait import evidence_bundle


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Dispatch repair tests must remain offline')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr('subprocess.Popen', denied)


def native_rejection(tmp_path, monkeypatch, reason='SOURCE_OBSERVATION_STALE'):
    b, result, _ = evidence_bundle(tmp_path, monkeypatch)
    r=b[0];stamp=result['source_frame_stamp_ns'];s=result['pre_transport_jit_veto']['lidar_snapshot']
    cert=dict(version=1,transport_boundary='before_POST_motion',reason=reason,
        transport_attempted=False,delivery_uncertain=False,source_stamp=stamp,
        source_receipt_monotonic_seconds=100.,source_age_seconds=1.01,source_fresh=False,
        control_generation_current=True,motion_owner_current=True,mission_context_current=True,
        watchdog_not_exhausted=True,action_fits_deadline=True,
        producer_session=s['producer_session'],acquisition_sequence=s['acquisition_sequence'],
        local_selection_accepted=reason!='LOCAL_SELECTION_REJECTED',
        local_safety_permitted=reason!='LOCAL_MOTION_SAFETY_REJECTED',
        interlock_permitted=reason!='FORWARD_INTERLOCK_REJECTED')
    forward=dict(ok=False,forwarded=False,error='motion_dispatch_preempted',
        transport_attempted=False,delivery_uncertain=False,dispatch_guard_rejection=cert)
    result.update(dispatch_guard_rejection=cert,bridge_stop_confirmed=True,
        reason='marvin_dispatch_guard_'+reason.lower(),approach_result={'forward_result':forward})
    return b,result


@pytest.mark.parametrize('fault',[None,'attempted','uncertain','forwarded','moved','stop','bridge',
    'unconsumed','float_stamp','different_stamp','nested_uncertain','nested_transport','version'])
def test_negative_certificate_cannot_override_contradictory_transport(tmp_path,monkeypatch,fault):
    b,v=native_rejection(tmp_path,monkeypatch);f=v['approach_result']['forward_result']
    if fault=='attempted':f['transport_attempted']=True
    if fault=='uncertain':f['delivery_uncertain']=True
    if fault=='forwarded':f['forwarded']=True
    if fault=='moved':v['motion_executed']=True
    if fault=='stop':v['stop_result']={'ok':False}
    if fault=='bridge':v['bridge_stop_confirmed']=False
    if fault=='unconsumed':b[0]._marvin_alignment_consumed_source_frame_stamps.clear()
    if fault=='float_stamp':v['dispatch_guard_rejection']['source_stamp']=float(v['source_frame_stamp_ns'])
    if fault=='different_stamp':v['dispatch_guard_rejection']['source_stamp']+=1
    if fault=='nested_uncertain':v['other']={'delivery_uncertain':True}
    if fault=='nested_transport':f['transport_result']={'ok':False}
    if fault=='version':v['dispatch_guard_rejection']['version']=2
    assert bool(b[0]._marvin_certified_dispatch_rejection(v,v['source_frame_stamp_ns'])) is (fault is None)


@pytest.mark.parametrize('reason',['LOCAL_SELECTION_REJECTED','LOCAL_MOTION_SAFETY_REJECTED'])
def test_final_geometry_refusal_uses_existing_native_JIT_validation(tmp_path,monkeypatch,reason):
    b,v=native_rejection(tmp_path,monkeypatch,reason);r=b[0]
    s=v['pre_transport_jit_veto']['lidar_snapshot']
    admitted=r._marvin_jit_veto_wait_evidence(v,stamp=v['source_frame_stamp_ns'],
        expected_session=s['producer_session'],planning_sequence=s['acquisition_sequence']-1,
        execution_guard=lambda:True)
    assert admitted and admitted['snapshot']['acquisition_sequence']==s['acquisition_sequence']
    assert not motions(b[3])


def client_with_stub_interlock():
    from types import SimpleNamespace
    c=RobotBridgeClient(base_url='http://offline.invalid')
    c.forward_interlock=SimpleNamespace(begin_positive_dispatch=lambda **k:1,
        finalize_positive_dispatch=lambda *a:True,stop_active=lambda:None)
    calls=[];c._request=lambda *a,**k:calls.append(a) or {'ok':True}
    return c,calls


@pytest.mark.parametrize('reason',['SOURCE_OBSERVATION_STALE','CONTROL_GENERATION_CHANGED',
    'MOTION_OWNER_LOST','MISSION_CONTEXT_LOST','DETOUR_DEADLINE_EXHAUSTED',
    'ACTION_NO_LONGER_FITS_DEADLINE','LOCAL_SELECTION_REJECTED','LOCAL_MOTION_SAFETY_REJECTED',
    'FORWARD_INTERLOCK_REJECTED','UNKNOWN_FINAL_GUARD_FAILURE'])
def test_client_retains_exact_final_reason_and_never_POSTs(reason):
    c,calls=client_with_stub_interlock()
    def guard():return False
    guard.decision={'reason':reason,'guard_stage':'test','source_stamp':1791658875641965778}
    result=c.move_forward(speed=.1,seconds=.5,dispatch_guard=guard)
    assert result['error']=='motion_dispatch_preempted' and not calls
    assert result['dispatch_guard_rejection']['reason']==reason
    assert result['transport_attempted'] is result['delivery_uncertain'] is False
    assert result['dispatch_guard_rejection']['source_stamp']==1791658875641965778


def test_certificate_is_bounded_serializable_and_excludes_sensor_payloads():
    c,calls=client_with_stub_interlock()
    def guard():return False
    guard.decision={'reason':'SOURCE_OBSERVATION_STALE','source_stamp':1791658875641965778,
        'local_selection_reason':'x'*10000,'source_age_seconds':float('inf'),
        'image':b'raw','points':[{}]*10000,'unexpected':{'payload':'raw'}}
    cert=c.move_forward(speed=.1,seconds=.5,dispatch_guard=guard)['dispatch_guard_rejection']
    encoded=json.dumps(cert,allow_nan=False)
    assert len(encoded)<1600 and len(cert['local_selection_reason'])==160
    assert not {'image','points','unexpected','source_age_seconds'} & cert.keys()


def test_plain_boolean_guard_has_unknown_certificate_not_invented_reason():
    c,calls=client_with_stub_interlock()
    result=c.move_forward(speed=.1,seconds=.5,dispatch_guard=lambda:False)
    assert result['dispatch_guard_rejection']['reason']=='UNKNOWN_FINAL_GUARD_FAILURE'
    assert not calls


def final_boundary_mission(tmp_path,monkeypatch,fault='stale',repeat=False):
    from test_marvin_lateral_avoidance import LEFT_OPEN
    b,flags,c=strafe_runtime(tmp_path,monkeypatch,[(0,.6)]*4+[(0,.5)],[LEFT_OPEN,None])
    r,behavior,robot,events,clock=b
    old=c._request
    def request(method,path,payload=None):
        if path=='/motion' and payload['linear_x']>0:
            events.append(('forward',payload['linear_x'],payload['duration']))
            clock[0]+=round(payload['duration']*1e9)
            return dict(ok=True,action='motion',mode='bounded',automatic_stop=True,
                returned_immediately=False,**payload)
        return old(method,path,payload)
    c._request=request
    ordinary_forward=robot.move_forward
    robot.move_forward=lambda **kw: c.move_forward(**kw) if kw.get('dispatch_guard') is not None else ordinary_forward(**kw)
    before_motion=c.motion;flags['refusals']=[];flags['boundary_checks']=0
    read=r.world_model.get_lidar_obstacles
    def lidar(**kwargs):
        value=read(**kwargs)
        if flags.get('repair'):
            # No metric bypass is reachable. Force a real lateral-to-turn
            # selection change while preserving fresh complete scan geometry.
            value['local_motion_geometry']['points'].extend([
                {'x_m':0.,'y_m':.465},{'x_m':0.,'y_m':-.465}])
        return value
    r.world_model.get_lidar_obstacles=lidar
    robot.on_motion=lambda:flags.update(repair=False)
    def boundary(**kw):
        if kw.get('linear_y',0)!=0 and (not flags['refusals'] or repeat):
            flags['boundary_checks']+=1
            if fault=='stale':clock[0]+=1_000_000_001
            if fault=='generation':r._control_generation+=1
            if fault=='owner':monkeypatch.setattr(r,'_marvin_motion_owner_is_current',lambda:False)
            if fault in {'cancel','context'}:r.mission_manager.complete_active_mission()
            if fault=='context':monkeypatch.setattr(r,'_marvin_motion_owner_is_current',lambda:True)
            if fault=='deadline':clock[0]+=120_001_000_000
            if fault=='fits':clock[0]+=119_500_000_000
            if fault=='repair':flags['repair']=True
            safety_original=None
            if fault=='safety':
                import behavior_manager
                safety_original=behavior_manager.evaluate_local_motion_safety
                def unsafe(*args,**kwargs):return dict(safety_original(*args,**kwargs),permitted=False,reason='controlled_safety_refusal')
                behavior_manager.evaluate_local_motion_safety=unsafe
            if fault=='unknown':
                original_guard=kw['dispatch_guard']
                def unknown():
                    original_guard()
                    unknown.decision=dict(original_guard.decision,reason='UNKNOWN_FINAL_GUARD_FAILURE')
                    return False
                kw['dispatch_guard']=unknown
            if fault=='interlock_stale':
                c.forward_interlock.refresh=lambda:(False,'stale_lidar')
            if fault=='interlock_unknown':c.forward_interlock.refresh=lambda:(False,'reader_exception')
            if fault=='selection':
                oldselect=r._select_marvin_detour_for_context
                def select(*args,**kwargs):return dict(oldselect(*args,**kwargs),accepted=False,action_type='NONE')
                monkeypatch.setattr(r,'_select_marvin_detour_for_context',select)
            value=before_motion(**kw)
            if safety_original is not None:behavior_manager.evaluate_local_motion_safety=safety_original
            if value.get('dispatch_guard_rejection'):flags['refusals'].append(copy.deepcopy(value))
            if fault=='interlock_stale':
                del c.forward_interlock.refresh
            return value
        return before_motion(**kw)
    c.motion=boundary
    def semantic(*,minimum_source_frame_stamp_ns):
        if flags['refusals']:events.append('fresh_semantic_after_refusal')
        behavior._clear_marvin_v2_tracker_episode()
        return behavior.observe_find_marvin_v2()
    behavior.reacquire_find_marvin_v2=semantic
    normal=r._execute_normal_marvin_find_mission
    def capture(*args,**kwargs):
        flags['mission_result']=normal(*args,**kwargs)
        return flags['mission_result']
    r._execute_normal_marvin_find_mission=capture
    return b,flags,c


def test_stale_boundary_new_stamp_replans_and_dispatches_independently(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch)
    result=run(b[0]);refused=next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))
    cert=refused['result']['dispatch_guard_rejection'];stamp=refused['source_frame_stamp_ns']
    assert cert['reason']=='SOURCE_OBSERVATION_STALE' and cert['source_age_seconds']>1
    assert refused['result']['source_stamp_consumed'] and refused['result']['bridge_stop_confirmed']
    assert not refused['motion_executed'] and refused['result']['actions_executed']==0
    assert cert['local_selection_accepted'] and cert['local_safety_permitted'] and cert['interlock_permitted']
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==0
    later=result['history'][result['history'].index(refused)+1]
    assert later['source_frame_stamp_ns']>stamp
    assert later['observation']['identity_source_frame_stamp_ns']>stamp
    assert later['observation']['opencv_tracker']['quality']>=.8
    assert later['action_lidar_evidence'][1]>refused['action_lidar_evidence'][1]
    assert later['result']['local_detour']['accepted'] and later['result']['source_stamp_consumed']
    assert stamp in b[0]._marvin_alignment_consumed_source_frame_stamps
    assert all(h['source_frame_stamp_ns']!=stamp for h in result['history'] if h['motion_executed'])
    assert len(flags['refusals'])==1 and 'fresh_semantic_after_refusal' in b[3]
    assert all(x['committed_side']=='LEFT' for x in result['navigation_phase_history'] if x['phase']=='CLEAR_SIDE')
    assert result['detour_context']['watchdog']['total_veto_count']==1
    assert result['detour_context']['watchdog']['total_detour_actions']==1
    assert b[0].MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS==1.0


@pytest.mark.parametrize('fault,reason',[('generation','CONTROL_GENERATION_CHANGED'),
    ('owner','MOTION_OWNER_LOST'),('cancel','MOTION_OWNER_LOST'),('context','MISSION_CONTEXT_LOST'),
    ('fits','ACTION_NO_LONGER_FITS_DEADLINE'),('deadline','DETOUR_DEADLINE_EXHAUSTED'),
    ('interlock_unknown','FORWARD_INTERLOCK_REJECTED')])
def test_preemption_deadline_or_unknown_interlock_never_recovers(tmp_path,monkeypatch,fault,reason):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,fault)
    run(b[0]);result=flags['mission_result'];cert=flags['refusals'][0]['dispatch_guard_rejection']
    assert cert['reason']==reason and not motions(b[3])
    assert result['state'] in {'STOPPED','BLOCKED'}
    assert 'fresh_semantic_after_refusal' not in b[3]
    if fault in {'deadline','fits'}:assert result['reason']=='find_marvin_detour_watchdog_exhausted'


def test_interlock_stale_zero_transport_reverifies_with_new_observation(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,'interlock_stale')
    result=run(b[0])
    assert flags['refusals'][0]['dispatch_guard_rejection']['interlock_reason']=='stale_lidar'
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==0


def test_repeated_stale_final_refusals_are_bounded_without_physical_progress(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,repeat=True)
    # Provide enough independently fresh observations to reach existing anti-spin bound.
    b[1].specs=iter(pursuit_specs([(0,1.1)]*40))
    result=run(b[0])
    assert result['state']=='BLOCKED'
    assert result['reason'] in {'find_marvin_equivalent_jit_veto_exhausted','find_marvin_dispatch_reverify_exhausted'}
    assert not motions(b[3]) and result['local_avoidance_actions']==0
    assert len(flags['refusals'])<=7
    assert result['detour_context']['watchdog']['phase_actions']==0


def test_retained_prefix_is_historical_unknown_not_fabricated_staleness():
    f=json.loads((Path(__file__).parent/'test_fixtures/marvin_dced894d_dispatch_guard.json').read_text())
    assert f['mission_id']=='mission-dced894d'
    assert f['phase_sequence']==['DIRECT','CLEAR_SIDE','PASS_OBSTACLE']
    assert f['physical_prefix']==['TURN_RIGHT','STRAFE_LEFT','BYPASS_FORWARD']+['TURN_RIGHT']*3
    assert f['failure']['classification']=='UNKNOWN_FINAL_GUARD_FAILURE'
    assert f['failure']['source_stamp']==1791658875641965778
    assert f['failure']['source_age_at_command_completion_seconds']<1
    assert f['failure']['selection_accepted'] and f['failure']['local_safety_permitted']
    assert f['failure']['transport_attempted'] is False


def test_real_final_geometry_rejection_replans_same_side_repair(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,'repair')
    result=run(b[0]);refused=next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))
    assert refused['result']['dispatch_guard_rejection']['reason']=='LOCAL_SELECTION_REJECTED'
    assert not refused['motion_executed'] and refused['result']['source_stamp_consumed']
    index=result['history'].index(refused);later=result['history'][index+1]
    assert later['result']['local_detour']['action_type']=='TURN_LEFT'
    assert later['source_frame_stamp_ns']>refused['source_frame_stamp_ns']
    assert later['observation']['identity_source_frame_stamp_ns']>refused['source_frame_stamp_ns']
    assert any(h['phase']=='CLEAR_SIDE' for h in result['navigation_phase_history'])
    assert not any(h['phase']=='PASS_OBSTACLE' for h in result['navigation_phase_history'])
    assert result['state']=='ARRIVED',result['reason']


@pytest.mark.parametrize('fault',['old','older','quality','absent'])
def test_failed_reverification_never_reuses_stale_authority(tmp_path,monkeypatch,fault):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch)
    original=b[1].reacquire_find_marvin_v2
    def semantic(**kwargs):
        if not flags['refusals']:
            return original(**kwargs)
        if fault=='quality':
            value=original(**kwargs)
            value['preview_result']['opencv_tracker']['quality']=.79999
            value['preview_result']['opencv_tracker']['matched']=False
            return value
        if fault=='absent':b[1].specs=iter(['absent'])
        if fault in {'old','older'}:
            value=original(**kwargs)
            value['preview_result']['source_frame_stamp_ns']=kwargs['minimum_source_frame_stamp_ns']-(fault=='older')
            value['preview_result']['identity_source_frame_stamp_ns']=kwargs['minimum_source_frame_stamp_ns']-1
            return value
        return original(**kwargs)
    b[1].reacquire_find_marvin_v2=semantic
    result=run(b[0])
    assert result['state'] in {'BLOCKED','REVERIFY_REQUIRED'}
    assert not motions(b[3]) and len(flags['refusals'])==1
    assert result['local_avoidance_actions']==0


@pytest.mark.parametrize('age,expected',[(.999999,True),(1.,True),(1.000000001,False)])
def test_one_second_source_freshness_is_exactly_preserved(tmp_path,monkeypatch,age,expected):
    b=make_runtime(tmp_path,monkeypatch,[]);now=time.monotonic()
    assert b[0]._marvin_motion_stamp_is_fresh(1791658875641965778,now-age) is expected
    assert not b[0]._marvin_motion_stamp_is_fresh(float(1791658875641965778),now-age)


def test_old_consumed_stamp_cannot_reenter_executor(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch);result=run(b[0])
    old=next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))['source_frame_stamp_ns']
    r=b[0];events_before=list(b[3])
    rejected=r._execute_single_marvin_approach(linear_speed=.1,duration=.5,
        source_frame_stamp_ns=old,dispatch_guard=lambda:True)
    assert not rejected['motion_executed'] and rejected['actions_executed']==0
    assert b[3]==events_before and old in r._marvin_alignment_consumed_source_frame_stamps


def test_final_primitive_safety_refusal_rechecks_phase_without_motion_credit(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,'safety');result=run(b[0])
    refused=next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))
    assert refused['result']['dispatch_guard_rejection']['reason']=='LOCAL_MOTION_SAFETY_REJECTED'
    assert refused['result']['pre_transport_jit_veto']['transport_attempted'] is False
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==1 and result['local_bypass_actions']==0
    assert refused['result']['motion_executed'] is False


def test_unknown_final_refusal_stays_fail_closed(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch,'unknown');result=run(b[0])
    assert result['state']=='BLOCKED'
    assert result['reason']=='marvin_dispatch_guard_unknown_final_guard_failure'
    assert not motions(b[3]) and result['local_avoidance_actions']==0
    assert 'fresh_semantic_after_refusal' not in b[3]


@pytest.mark.parametrize('age',[None,float('nan'),0.,1.])
def test_stale_label_without_stale_measurement_cannot_recover(tmp_path,monkeypatch,age):
    b,v=native_rejection(tmp_path,monkeypatch)
    v['dispatch_guard_rejection']['source_age_seconds']=age
    assert b[0]._marvin_certified_dispatch_rejection(v,v['source_frame_stamp_ns']) is None


def test_source_age_is_measured_at_freshness_predicate_not_guard_entry(tmp_path,monkeypatch):
    r,_,_,_,clock=make_runtime(tmp_path,monkeypatch,[])
    stamp=1791658875641965778;receipt=time.monotonic()
    r._marvin_alignment_observation={'source_frame_stamp_ns':stamp,'received_monotonic_seconds':receipt,
        'opencv_tracker':{'source_frame_stamp_ns':stamp,'received_monotonic_seconds':receipt}}
    monkeypatch.setattr(r,'_marvin_motion_owner_is_current',lambda:True)
    def action_guard():
        clock[0]+=1_000_000_001
        action_guard.decision=dict(mission_context_current=True,watchdog_not_exhausted=True,
            action_fits_deadline=True,reason=None)
        return True
    def callback(**kwargs):
        g=kwargs['dispatch_guard'];assert g() is False
        return g.decision
    decision=r._dispatch_marvin_observation_action(callback,source_frame_stamp_ns=stamp,
        mission_execution_guard=action_guard)
    assert decision['reason']=='SOURCE_OBSERVATION_STALE'
    assert decision['source_age_seconds']>1.


def test_count_nested_guards_through_real_final_client_boundary(tmp_path,monkeypatch):
    b,flags,c=final_boundary_mission(tmp_path,monkeypatch)
    dispatch=b[0]._dispatch_marvin_observation_action;counts=[]
    def counted_dispatch(callback,**kwargs):
        counts.append({'stamp':kwargs['source_frame_stamp_ns'],'count':0})
        row=counts[-1]
        def counted_executor(**arguments):
            original_guard=arguments['dispatch_guard']
            def counted_guard():
                row['count']+=1
                admitted=original_guard()
                counted_guard.decision=original_guard.decision
                return admitted
            arguments['dispatch_guard']=counted_guard
            return callback(**arguments)
        return dispatch(counted_executor,**kwargs)
    monkeypatch.setattr(b[0],'_dispatch_marvin_observation_action',counted_dispatch)
    result=run(b[0])
    refused=next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))
    refused_count=next(x['count'] for x in counts if x['stamp']==refused['source_frame_stamp_ns'])
    admitted_count=next(x['count'] for x in counts if any(h['source_frame_stamp_ns']==x['stamp']
        and h['result'].get('action_type')=='STRAFE_LEFT' and h['motion_executed'] for h in result['history']))
    # The selected primitive is lateral, whose nesting differs from the retired
    # bypass. Both must repeat the lease guard through the real client boundary;
    # rejection must prevent the remaining admission/transport checks.
    assert refused_count >= 3 and admitted_count > refused_count
    assert len(flags['refusals'])==1 and len(motions(b[3]))==1
