"""Real final client boundary; all camera, LiDAR and transport are offline fixtures."""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from behavior_manager import BehaviorManager
from robot_bridge.client import RobotBridgeClient
from marvin_obstacle_phases import plan_phase_action
from test_find_marvin_closed_loop import run, motions
from test_marvin_local_bypass import OPEN_LEFT, ASSOCIATION, pursuit_specs
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_marvin_dispatch_guard_repair import final_boundary_mission, native_rejection


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Final-JIT tests cannot access robot/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


REPAIR = [(.65, -.14), (0., 1.2), (0., -.65)]
FIXTURE = Path(__file__).parent / 'test_fixtures/marvin_105f2dfe_final_jit_action_switch.json'


def lateral_switch_mission(tmp_path, monkeypatch, *, fault='switch', direction='LEFT'):
    mirror = 1 if direction == 'LEFT' else -1
    def side(points):return [(x, mirror*y) for x,y in points]
    scenes = [side(LEFT_OPEN), None]
    b, flags, client = strafe_runtime(tmp_path, monkeypatch,
        pursuit_specs([(0., 1.1)]*4), scenes)
    r, behavior, robot, events, clock = b
    flags.update(final=False, refusals=[], evaluations=[], semantic_calls=0, semantic_stamps=[], phase_samples=[])
    read = r.world_model.get_lidar_obstacles
    def lidar(**kw):
        value = read(**kw)
        if flags['final'] and fault == 'switch':
            value['local_motion_geometry']['points'].extend([
                {'x_m':0.,'y_m':.465},{'x_m':0.,'y_m':-.465}])
        return value
    r.world_model.get_lidar_obstacles = lidar
    request = client._request
    def transport(method, path, payload=None):
        if path == '/motion' and payload['linear_x'] > 0:
            events.append(('forward', payload['linear_x'], payload['duration']))
            clock[0] += round(payload['duration']*1e9)
            return dict(ok=True, action='motion', mode='bounded', automatic_stop=True,
                returned_immediately=False, **payload)
        return request(method, path, payload)
    client._request = transport
    original_forward = robot.move_forward
    robot.move_forward = lambda **kw: client.move_forward(**kw) if kw.get('dispatch_guard') else original_forward(**kw)
    original_select = r._select_marvin_detour_for_context
    def select(*args, **kw):
        value = original_select(*args, **kw)
        if flags['final'] and fault == 'absent':return {'accepted':False}
        if flags['final'] and fault == 'malformed':return None
        return value
    monkeypatch.setattr(r, '_select_marvin_detour_for_context', select)
    original_motion = client.motion
    def boundary(**kw):
        triggered = kw.get('linear_y', 0) != 0 and not motions(events) and not flags['refusals']
        if not triggered:return original_motion(**kw)
        flags['final'] = True
        before = list(motions(events))
        if fault == 'stale':clock[0] += 1_000_000_001
        if fault == 'generation':r._control_generation += 1
        if fault == 'owner':monkeypatch.setattr(r, '_marvin_motion_owner_is_current', lambda:False)
        if fault == 'context':
            r.mission_manager.complete_active_mission()
            monkeypatch.setattr(r, '_marvin_motion_owner_is_current', lambda:True)
        if fault == 'deadline':clock[0] += 120_001_000_000
        safety = None
        if fault == 'safety':
            import behavior_manager
            safety = behavior_manager.evaluate_local_motion_safety
            monkeypatch.setattr(behavior_manager, 'evaluate_local_motion_safety',
                lambda *a, **k: dict(safety(*a, **k), permitted=False, reason='controlled_lateral_safety_veto'))
        refresh = None
        if fault.startswith('interlock_'):
            refresh = client.forward_interlock.refresh
            reason = 'stale_lidar' if fault == 'interlock_stale' else 'reader_exception'
            client.forward_interlock.refresh = lambda:(False, reason)
        guard = kw['dispatch_guard']
        def counted():
            admitted = guard()
            counted.decision = getattr(guard, 'decision', {})
            flags['evaluations'].append(copy.deepcopy(counted.decision))
            if fault == 'unknown':
                counted.decision = dict(counted.decision, reason='UNKNOWN_FINAL_GUARD_FAILURE')
                return False
            return admitted
        kw['dispatch_guard'] = counted
        value = original_motion(**kw)
        if safety is not None:monkeypatch.setattr('behavior_manager.evaluate_local_motion_safety', safety)
        if refresh is not None:client.forward_interlock.refresh = refresh
        assert motions(events) == before  # The selected replacement is never dispatched here.
        if value.get('dispatch_guard_rejection'):flags['refusals'].append(copy.deepcopy(value))
        return value
    client.motion = boundary
    normal = behavior.execute_guarded_marvin_lateral_step
    def guarded(**kw):
        validator = kw['selection_validator']
        def capture(sample):
            value = validator(sample)
            flags.setdefault('local_evaluations', []).append({
                'sequence':sample['acquisition_sequence'], 'selection':copy.deepcopy(value),
                'at_client_boundary':flags['final']})
            return value
        kw['selection_validator'] = capture
        return normal(**kw)
    behavior.execute_guarded_marvin_lateral_step = guarded
    def semantic(*, minimum_source_frame_stamp_ns):
        if flags['refusals']:
            flags['semantic_calls'] += 1
            events.append('replacement_semantic')
        behavior._clear_marvin_v2_tracker_episode()
        value=behavior.observe_find_marvin_v2()
        if flags['refusals']:
            preview=value['preview_result']
            assert preview['source_frame_stamp_ns']>preview['identity_source_frame_stamp_ns']>minimum_source_frame_stamp_ns
            flags['semantic_stamps'].append(preview['source_frame_stamp_ns'])
        return value
    behavior.reacquire_find_marvin_v2 = semantic
    normal_mission = r._execute_normal_marvin_find_mission
    def capture_mission(*args, **kw):
        flags['mission_result'] = normal_mission(*args, **kw)
        return flags['mission_result']
    r._execute_normal_marvin_find_mission = capture_mission
    return b, flags, client


def refused_row(result):return next(h for h in result['history'] if h['result'].get('dispatch_guard_rejection'))


def test_nested_lateral_switch_certifies_exact_latest_scan_and_replans(tmp_path, monkeypatch):
    b, f, _ = lateral_switch_mission(tmp_path, monkeypatch)
    result = run(b[0]);row = refused_row(result);v = row['result'];cert = v['dispatch_guard_rejection']
    assert result['state'] == 'ARRIVED', result['reason']
    assert cert['reason'] == 'LOCAL_SELECTION_REJECTED'
    # The existing comparison classifier only names bypass/strafe switches.
    # A real lateral-to-turn change still vetoes without inventing that class.
    assert cert['selection_rejection_class'] is None
    assert cert['requested_action_type'] == 'STRAFE_LEFT' and cert['final_selected_action_type'] == 'TURN_LEFT'
    assert cert['requested_direction'] == cert['final_selected_direction'] == 'LEFT'
    assert cert['guard_stage'] == 'lateral_dispatch'
    assert cert['final_jit_lidar_sequence'] > cert['planning_lidar_sequence']
    assert cert['acquisition_sequence'] == cert['final_jit_lidar_sequence']
    assert cert['planning_lidar_sequence'] == result['local_avoidance_history'][0]['selection']['acquisition_sequence']
    assert cert['local_selection_accepted'] is False and cert['local_safety_permitted'] is False
    assert cert['transport_attempted'] is False and cert['delivery_uncertain'] is False
    assert cert['source_fresh'] is True and cert['source_age_seconds'] <= 1.
    assert all(cert[k] is True for k in ('control_generation_current','motion_owner_current',
        'mission_context_current','watchdog_not_exhausted','action_fits_deadline'))
    assert v['pre_transport_jit_veto']['lidar_snapshot']['acquisition_sequence'] == cert['final_jit_lidar_sequence']
    assert v['pre_transport_jit_veto']['target_association']['acquisition_sequence'] == cert['final_jit_lidar_sequence']
    assert v['reason'] == 'marvin_dispatch_guard_local_selection_rejected'
    assert v['source_stamp_consumed'] and v['bridge_stop_confirmed'] and v['stop_result']['ok']
    assert v['actions_executed'] == 0 and not row['motion_executed']
    assert result['local_avoidance_actions'] == 1 and result['local_bypass_actions'] == 0
    assert motions(b[3])[0] == ('turn','LEFT',.25,.5)
    assert all(m==('forward',.1,.5) for m in motions(b[3])[1:])
    transitions = result['navigation_phase_history']
    assert any(h['phase']=='CLEAR_SIDE' for h in transitions)
    assert not any(h['phase']=='PASS_OBSTACLE' for h in transitions)
    assert all(h['committed_side']=='LEFT' for h in transitions if h['phase']=='CLEAR_SIDE')
    assert result['detour_context']['watchdog']['total_veto_count'] == 1
    assert result['detour_context']['watchdog']['total_detour_actions'] == 1
    assert result['local_avoidance_history'][0]['vetoed_before_transport']
    assert result['local_avoidance_history'][0]['dispatched'] is False
    # Exactly the preliminary and client-nested local evaluations; nested evidence replaces the first.
    ev = f['local_evaluations']
    tail = next(i for i,e in enumerate(ev) if e['at_client_boundary'])
    assert ev[tail-1]['selection']['accepted'] is True
    assert ev[tail-1]['selection']['action_type'] == 'STRAFE_LEFT'
    assert ev[tail]['sequence'] > ev[tail-1]['sequence']
    assert ev[tail]['selection']['action_type'] == 'TURN_LEFT'
    assert len(f['refusals']) == len(f['evaluations']) == 1
    assert f['semantic_calls']==len(set(f['semantic_stamps']))>=1


@pytest.mark.parametrize('direction', ['LEFT','RIGHT'])
def test_new_strict_stamp_independently_authorizes_replacement_no_old_retry(tmp_path,monkeypatch,direction):
    b,f,_ = lateral_switch_mission(tmp_path,monkeypatch,direction=direction)
    result=run(b[0]);row=refused_row(result);old=row['source_frame_stamp_ns']
    later=result['history'][result['history'].index(row)+1]
    assert later['result']['local_detour']['action_type']=='TURN_'+direction and later['motion_executed']
    observation=later['observation']
    assert type(old) is int and type(later['source_frame_stamp_ns']) is int
    assert old < observation['identity_source_frame_stamp_ns'] < later['source_frame_stamp_ns']
    assert observation['opencv_tracker']['quality'] >= observation['opencv_tracker']['threshold'] == .8
    assert later['action_lidar_evidence'][1] > row['result']['dispatch_guard_rejection']['final_jit_lidar_sequence']
    assert later['result']['turn_result']['permitted'] and later['result']['turn_result']['confirmed_forwarded']
    assert later['result']['stop_result']['ok']
    assert (later['result']['angular_speed'],later['result']['duration'])==(.25,.5)
    assert old in b[0]._marvin_alignment_consumed_source_frame_stamps
    assert not any(h['motion_executed'] and h['source_frame_stamp_ns']==old for h in result['history'])
    before=list(b[3])
    denied=b[0]._execute_single_marvin_approach(linear_speed=.1,duration=.5,
        source_frame_stamp_ns=old,dispatch_guard=lambda:True)
    assert not denied['motion_executed'] and denied['actions_executed']==0 and b[3]==before


def test_reverse_switch_uses_same_native_veto_and_clear_side_policy(tmp_path,monkeypatch):
    b,f,_=final_boundary_mission(tmp_path,monkeypatch,'repair')
    result=run(b[0]);row=refused_row(result);cert=row['result']['dispatch_guard_rejection']
    assert BehaviorManager._marvin_selection_rejection_class(
        dict(accepted=False,action_type='STRAFE_LEFT',direction='LEFT'),'BYPASS_FORWARD','LEFT')=='LOCAL_SELECTION_CHANGED_AT_FINAL_JIT'
    assert cert['requested_action_type']=='STRAFE_LEFT' and cert['final_selected_action_type']=='TURN_LEFT'
    assert cert['selection_rejection_class'] is None
    assert cert['requested_direction']==cert['final_selected_direction']=='LEFT'
    assert cert['final_jit_lidar_sequence']>cert['planning_lidar_sequence']
    assert row['result']['pre_transport_jit_veto']['transport_attempted'] is False
    assert any(h['phase']=='CLEAR_SIDE' for h in result['navigation_phase_history'])
    later=result['history'][result['history'].index(row)+1]
    assert later['result']['local_detour']['action_type']=='TURN_LEFT' and later['motion_executed']
    assert row['source_frame_stamp_ns'] < later['observation']['identity_source_frame_stamp_ns'] < later['source_frame_stamp_ns']
    assert result['state']=='ARRIVED' and result['local_avoidance_actions']==1


@pytest.mark.parametrize('fault,reason',[
    ('safety','LOCAL_MOTION_SAFETY_REJECTED'),('stale','SOURCE_OBSERVATION_STALE'),
    ('interlock_stale','FORWARD_INTERLOCK_REJECTED'),('interlock_unknown','FORWARD_INTERLOCK_REJECTED'),
    ('generation','CONTROL_GENERATION_CHANGED'),('owner','MOTION_OWNER_LOST'),
    ('context','MISSION_CONTEXT_LOST'),('deadline','DETOUR_DEADLINE_EXHAUSTED'),
    ('unknown','UNKNOWN_FINAL_GUARD_FAILURE'),('malformed','UNKNOWN_FINAL_GUARD_FAILURE'),
    ('absent','UNKNOWN_FINAL_GUARD_FAILURE')])
def test_other_lateral_final_predicates_are_distinct_and_fail_closed_when_uncertified(tmp_path,monkeypatch,fault,reason):
    b,f,_=lateral_switch_mission(tmp_path,monkeypatch,fault=fault)
    run(b[0]);result=f['mission_result'];cert=f['refusals'][0]['dispatch_guard_rejection']
    assert cert['reason']==reason
    assert cert['transport_attempted'] is False and cert['delivery_uncertain'] is False
    if fault in {'stale','interlock_stale','safety'}:
        assert result['state']=='ARRIVED',result['reason']
        assert f['semantic_calls']==len(set(f['semantic_stamps']))>=1
        assert result['local_avoidance_actions']==1
    else:
        assert result['state'] in {'STOPPED','BLOCKED'}
        assert f['semantic_calls']==0 and not motions(b[3])
        if fault=='deadline':assert result['reason']=='find_marvin_detour_watchdog_exhausted'
    if fault not in {'switch'}:assert cert.get('selection_rejection_class') is None


@pytest.mark.parametrize('field,value',[
    ('transport_attempted',True),('delivery_uncertain',True),('forwarded',True),('ok',True),
    ('version',2),('source_stamp',1791669298519885955.0),('source_stamp',1791669298519885954),
    ('transport_boundary','after_POST_motion'),('local_selection_accepted',True)])
def test_lateral_negative_provenance_cannot_hide_transport_or_invalid_certificate(tmp_path,monkeypatch,field,value):
    b,v=native_rejection(tmp_path,monkeypatch,'LOCAL_SELECTION_REJECTED')
    v['dispatch_guard_rejection']['local_selection_accepted']=False
    v['lateral_step']={'lateral_result':v.pop('approach_result')['forward_result']}
    assert b[0]._marvin_certified_dispatch_rejection(v,v['source_frame_stamp_ns'])
    if field in {'transport_attempted','delivery_uncertain','forwarded','ok'}:v['lateral_step']['lateral_result'][field]=value
    else:v['dispatch_guard_rejection'][field]=value
    assert b[0]._marvin_certified_dispatch_rejection(v,v['source_frame_stamp_ns']) is None


@pytest.mark.parametrize('selection',[{}, {'accepted':False}, {'accepted':False,'action_type':None},
    {'accepted':False,'action_type':'BYPASS_FORWARD','direction':None},
    {'accepted':False,'action_type':'TELEVISION','direction':'LEFT'}])
def test_missing_or_invalid_selection_never_claims_action_switch(selection):
    assert BehaviorManager._marvin_selection_rejection_class(selection,'STRAFE_LEFT','LEFT') is None


def test_certificate_schema_remains_bounded_and_serializable():
    guard=lambda:False
    guard.decision=dict(requested_action_type='STRAFE_LEFT',requested_direction='LEFT',
        final_selected_action_type='BYPASS_FORWARD',final_selected_direction='LEFT',
        planning_lidar_sequence=4093,final_jit_lidar_sequence=4094,
        selection_rejection_class='LOCAL_SELECTION_CHANGED_AT_FINAL_JIT',
        reason='LOCAL_SELECTION_REJECTED',source_stamp=1791669298519885955,
        raw_scan=[0]*10000,prompt='secret',image='secret')
    cert=RobotBridgeClient._pre_transport_guard_rejection(guard)
    assert not {'raw_scan','prompt','image'} & cert.keys()
    assert len(json.dumps(cert))<2500
    assert json.loads(json.dumps(cert))['source_stamp']==1791669298519885955
    assert cert['final_selected_action_type']=='BYPASS_FORWARD' and cert['transport_attempted'] is False


def test_retained_prefix_replay_does_not_relabel_historical_unknown():
    f=json.loads(FIXTURE.read_text())
    assert f['mission_id']=='mission-105f2dfe'
    assert f['phase_sequence']==['DIRECT','CLEAR_SIDE','PASS_OBSTACLE','CLEAR_SIDE']
    assert f['physical_prefix']==['STRAFE_LEFT','BYPASS_FORWARD','TURN_RIGHT','TURN_RIGHT']+['BYPASS_FORWARD']*8
    assert f['avoidance_counts']==[1,2,2,2,3,4,5,6,7,8,9,10]
    event=f['final_refusal']
    assert event['historical_runtime_certificate']['reason']=='UNKNOWN_FINAL_GUARD_FAILURE'
    assert event['requested_action_type']=='STRAFE_LEFT' and event['final_selected_action_type']=='BYPASS_FORWARD'
    assert event['source_stamp']==1791669298519885955 and event['source_stamp_consumed']
    assert event['planning_sequence']==4093 and event['final_jit_sequence']==4094
    assert event['local_safety_permitted'] and not event['local_selection_accepted']
    assert event['transport_attempted'] is False
    classified=BehaviorManager._marvin_selection_rejection_class(dict(accepted=False,
        action_type=event['final_selected_action_type'],direction='LEFT'),'STRAFE_LEFT','LEFT')
    assert classified=='LOCAL_SELECTION_CHANGED_AT_FINAL_JIT'
    # Classification of the retained comparison is offline, not a replacement historical certificate.
    assert event['historical_runtime_certificate']['reason']=='UNKNOWN_FINAL_GUARD_FAILURE'


@pytest.mark.parametrize('fault',['quality','old_stamp','older_stamp','float_stamp'])
def test_replacement_identity_cannot_dispatch_without_new_strict_authority(tmp_path,monkeypatch,fault):
    b,f,_=lateral_switch_mission(tmp_path,monkeypatch)
    semantic=b[1].reacquire_find_marvin_v2
    def invalid(*,minimum_source_frame_stamp_ns):
        v=semantic(minimum_source_frame_stamp_ns=minimum_source_frame_stamp_ns)
        if not f['refusals']:
            return v
        o=v['preview_result'];t=o['opencv_tracker']
        if fault=='quality':t.update(quality=.799999,matched=False)
        else:
            stamp=minimum_source_frame_stamp_ns-(fault=='older_stamp')
            if fault=='float_stamp':stamp=float(stamp+1000)
            o.update(source_frame_stamp_ns=stamp,identity_source_frame_stamp_ns=minimum_source_frame_stamp_ns-1)
            t['source_frame_stamp_ns']=stamp
        return v
    b[1].reacquire_find_marvin_v2=invalid
    result=run(b[0]);row=refused_row(result)
    assert result['state'] in {'BLOCKED','REVERIFY_REQUIRED'}
    assert not motions(b[3]) and result['local_avoidance_actions']==0
    assert row['result']['actions_executed']==0 and row['result']['source_stamp_consumed']
    assert row['source_frame_stamp_ns'] in b[0]._marvin_alignment_consumed_source_frame_stamps


@pytest.mark.parametrize('fault',['stop','bridge','interlock','session','duplicate_transport'])
def test_lateral_certification_does_not_bypass_native_stationary_admission(tmp_path,monkeypatch,fault):
    b,v=native_rejection(tmp_path,monkeypatch,'LOCAL_SELECTION_REJECTED')
    v['dispatch_guard_rejection']['local_selection_accepted']=False
    forward=v.pop('approach_result')['forward_result'];v['lateral_step']={'lateral_result':forward}
    r,_,robot,_,_=b;e=v['pre_transport_jit_veto'];s=e['lidar_snapshot']
    if fault=='stop':v['stop_result']={'ok':False}
    if fault=='bridge':robot.status=lambda:{'ok':True,'status':'READY','ros_ready':True,
        'motion':dict(linear_x=0,linear_y=.08,angular_z=0,streaming=False)}
    if fault=='interlock':r.forward_interlock.status=lambda:{'active_forward':True,'pending_forward':False}
    if fault=='session':s['producer_session']='other'
    if fault=='duplicate_transport':v['approach_result']={'forward_result':copy.deepcopy(forward)}
    admitted=r._marvin_jit_veto_wait_evidence(v,stamp=v['source_frame_stamp_ns'],
        expected_session=r.lidar_worker.session,planning_sequence=s['acquisition_sequence']-1,
        execution_guard=lambda:True)
    assert admitted is None and not motions(b[3])


@pytest.mark.parametrize('requested,selected,direction',[
    ('STRAFE_LEFT','BYPASS_FORWARD','LEFT'),('STRAFE_RIGHT','BYPASS_FORWARD','RIGHT'),
    ('BYPASS_FORWARD','STRAFE_LEFT','LEFT'),('BYPASS_FORWARD','STRAFE_RIGHT','RIGHT')])
def test_bidirectional_switch_subclassification_requires_explicit_valid_selection(requested,selected,direction):
    assert BehaviorManager._marvin_selection_rejection_class(
        dict(accepted=False,action_type=selected,direction=direction),requested,direction
    )=='LOCAL_SELECTION_CHANGED_AT_FINAL_JIT'
    assert BehaviorManager._marvin_selection_rejection_class(
        dict(accepted=True,action_type=selected,direction=direction),requested,direction
    ) is None


@pytest.mark.parametrize('sequence',[True,0,1.0,9999])
def test_final_certificate_scan_must_match_exact_veto_evidence(tmp_path,monkeypatch,sequence):
    b,v=native_rejection(tmp_path,monkeypatch,'LOCAL_SELECTION_REJECTED')
    v['dispatch_guard_rejection']['local_selection_accepted']=False
    v['dispatch_guard_rejection']['final_jit_lidar_sequence']=sequence
    v['lateral_step']={'lateral_result':v.pop('approach_result')['forward_result']}
    e=v['pre_transport_jit_veto'];s=e['lidar_snapshot']
    assert b[0]._marvin_jit_veto_wait_evidence(v,stamp=v['source_frame_stamp_ns'],
        expected_session=b[0].lidar_worker.session,planning_sequence=s['acquisition_sequence']-1,
        execution_guard=lambda:True) is None
