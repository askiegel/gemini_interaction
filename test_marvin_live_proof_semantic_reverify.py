"""Offline semantic misses preserve sealed history, never movement authority."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import socket
import threading
import time

import pytest

from runtime import _MarvinLiveProofContinuation
from test_find_marvin_closed_loop import motions, run
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_marvin_live_proof_continuation import arm, complete, step
from test_marvin_live_proof_tracker_reset import proof_runtime, forward, begin


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Semantic reverify tests must not access network/robot/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def avoidance_bundle(tmp_path, monkeypatch):
    shifted = [(x,y-.04) if x>0 else (x,y) for x,y in LEFT_OPEN]
    return strafe_runtime(tmp_path, monkeypatch, [(0,.8)]*16,
        [LEFT_OPEN, shifted, None], factory=proof_runtime)[0]


def miss(bundle):
    r,b,_,events,_=bundle
    c=r._marvin_live_proof_continuation
    digest=r._marvin_live_proof_checkpoint_digest
    prior=asdict(c); consumed=set(r._marvin_alignment_consumed_source_frame_stamps)
    calls=b.semantic_calls; count=len(motions(events)); created=len(b.created_trackers)
    assert arm(r)['ok']; before=list(events)
    assert not arm(r)['ok'] and events==before
    b.confirm_identity=False
    result=step(r)
    assert result['controller_result']['state']=='REVERIFY_REQUIRED'
    assert result['reason']=='marvin_identity_not_confirmed'
    assert result['proof_state']=='REVERIFY_DISARMED' and result['continuation_available']
    assert result['motion_executed'] is False
    assert r._marvin_live_proof_continuation is c and asdict(c)==prior
    assert r._marvin_live_proof_checkpoint_digest==digest
    assert b.semantic_calls==calls+1 and len(b.created_trackers)==created
    assert len(motions(events))==count
    assert r._marvin_alignment_consumed_source_frame_stamps==consumed
    assert b._marvin_v2_tracker_episode is None and r._marvin_alignment_observation is None
    proof=result['controller_result']['proof']
    assert proof['dispatch_opportunities']==proof['physical_dispatches_or_uncertain']==0
    assert proof['source_stamps']==[] and proof['post_action_evidence'] is None
    stamp=result['reverify_camera_floor_source_frame_stamp_ns']
    assert type(stamp) is int and stamp==b.identity_frames[-1] and stamp>c.camera_floor_stamp
    assert stamp not in consumed
    assert r._marvin_live_proof_checkpoint_valid()
    assert result['controller_result']['stop_result']['ok']
    assert not step(r)['execution_authorized'] and b.semantic_calls==calls+1
    return result


def test_two_misses_preserve_checkpoint_then_fresh_action_resumes_progress(tmp_path, monkeypatch):
    bundle=avoidance_bundle(tmp_path,monkeypatch); r,b,_,events,_=bundle
    begin(bundle); c=r._marvin_live_proof_continuation; saved=asdict(c)
    consumed=set(r._marvin_alignment_consumed_source_frame_stamps)
    first=miss(bundle); floor1=first['reverify_camera_floor_source_frame_stamp_ns']
    second=miss(bundle); floor2=second['reverify_camera_floor_source_frame_stamp_ns']
    assert floor2>floor1 and b.proof_source_floors[-1]==floor1
    assert asdict(r._marvin_live_proof_continuation)==saved
    assert r._marvin_target_range_association.__dict__==c.range_association
    assert c.avoidance['local_avoidance_actions']==1 and c.avoidance['local_bypass_actions']==0
    assert c.avoidance['last_detour_direction']=='LEFT' and c.previous_selection['action_type']=='STRAFE_LEFT'
    assert c.avoidance_history[0]['actual_route_progress']['meaningful_progress']
    b.confirm_identity=True; b.visual_shift=110
    assert arm(r)['ok']; second_action=step(r); complete(bundle,second_action,1)
    row=second_action['controller_result']['local_avoidance_history'][1]
    assert row['previous_action_type']=='STRAFE_LEFT'
    assert row['previous_clearances']==c.previous_clearances
    assert row['selection']['action_type']=='STRAFE_LEFT' and row['selection']['progress_improved'] is True
    obs=second_action['controller_result']['history'][0]['observation']
    semantic=obs['identity_source_frame_stamp_ns']; action=obs['source_frame_stamp_ns']
    assert floor2<semantic<action and b.proof_source_floors[-1]==floor2
    assert obs['strict_tracker_episode']['initialized_this_observation']
    assert obs['strict_tracker_episode']['continued_existing_tracker'] is False
    assert r._marvin_alignment_consumed_source_frame_stamps==consumed|{action}
    assert floor1 not in r._marvin_alignment_consumed_source_frame_stamps and floor2 not in r._marvin_alignment_consumed_source_frame_stamps
    new=r._marvin_live_proof_continuation
    assert new.avoidance['local_avoidance_actions']==2 and new.avoidance['last_detour_direction']=='LEFT'
    assert new.avoidance_history[0]==c.avoidance_history[0]
    assert new.completion['state']=='PROOF_COMPLETE' and r._marvin_live_proof_reverify is None
    assert len(motions(events))==2


@pytest.mark.parametrize('field', list(_MarvinLiveProofContinuation.__dataclass_fields__))
def test_miss_preserves_each_authoritative_checkpoint_field(tmp_path, monkeypatch, field):
    bundle=avoidance_bundle(tmp_path,monkeypatch); begin(bundle)
    r=bundle[0]; old=copy.deepcopy(getattr(r._marvin_live_proof_continuation,field))
    miss(bundle)
    assert getattr(r._marvin_live_proof_continuation,field)==old


def test_rearm_from_reverify_observes_nothing_and_retains_floors_without_credits(tmp_path, monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,_=bundle; begin(bundle); miss(bundle)
    c=r._marvin_live_proof_continuation; retry=r._marvin_live_proof_reverify
    before=list(events); calls=b.semantic_calls
    result=arm(r)
    assert result['ok'] and result['proof_state']=='ARMED' and result['motion_executed'] is False
    assert result['reverify_camera_floor_source_frame_stamp_ns']==retry.camera_floor_stamp
    assert events==before and b.semantic_calls==calls
    assert r._marvin_live_proof_continuation is c and r._marvin_live_proof_reverify is retry
    assert not arm(r)['ok'] and events==before


@pytest.mark.parametrize('fault', ['session','stop_failure','bridge_not_ready','bridge_ros','bridge_motion',
    'lidar_invalid','lidar_regression','consumed_stamp','dispatch_opportunity','uncertain_delivery',
    'action_history','post_action_evidence','jit_authority','action_lidar_changed','checkpoint_mutated',
    'ownership','exception','tracker_failure','mid_attempt_lidar_regression','unknown_reason','float_stamp','bool_stamp','old_stamp','missing_stamp'])
def test_unsafe_or_non_allowlisted_failure_locks(tmp_path,monkeypatch,fault):
    bundle=forward(tmp_path,monkeypatch); r,b,robot,events,_=bundle; begin(bundle)
    c=r._marvin_live_proof_continuation; assert arm(r)['ok']; b.confirm_identity=False
    original=r._observe_find_marvin_v2
    def observe(**kwargs):
        if fault=='exception': raise RuntimeError('offline perception exception')
        obs=original(**kwargs)
        if fault=='session': r.lidar_worker.session='changed'
        if fault=='stop_failure': robot.stop=lambda:{'ok':False}
        if fault.startswith('bridge_'):
            status=robot.status
            def altered():
                value=copy.deepcopy(status())
                if fault=='bridge_not_ready': value['status']='ERROR'
                if fault=='bridge_ros': value['ros_ready']=False
                if fault=='bridge_motion': value['motion']['linear_x']=.1
                return value
            robot.status=altered
        if fault in {'lidar_invalid','lidar_regression'}:
            r._active_localization_lidar_is_current=lambda:(r.lidar_worker.session,
                None if fault=='lidar_invalid' else {'acquisition_sequence':c.post_lidar_sequence-1})
        if fault=='mid_attempt_lidar_regression':
            r._active_localization_lidar_is_current=lambda:(r.lidar_worker.session, {'acquisition_sequence':c.post_lidar_sequence})
        if fault=='consumed_stamp': r._marvin_alignment_consumed_source_frame_stamps.add(obs['identity_source_frame_stamp_ns'])
        if fault=='dispatch_opportunity': r._marvin_live_proof_owner.dispatch_opportunities=1
        if fault=='action_lidar_changed': r._marvin_last_action_lidar_evidence=(c.producer_session,c.post_lidar_sequence+1)
        if fault=='checkpoint_mutated': c.avoidance['local_avoidance_actions']=3
        if fault=='ownership': r._invalidate_marvin_live_proof('ownership_conflict')
        if fault=='tracker_failure': obs['perception_reason']='marvin_v2_tracker_unmatched'
        if fault=='unknown_reason': obs['perception_reason']='arbitrary_perception_error'
        if fault=='float_stamp': obs['identity_source_frame_stamp_ns']=float(c.camera_floor_stamp+1)
        if fault=='bool_stamp': obs['identity_source_frame_stamp_ns']=True
        if fault=='old_stamp': obs['identity_source_frame_stamp_ns']=c.camera_floor_stamp
        if fault=='missing_stamp': obs['identity_source_frame_stamp_ns']=None
        return obs
    r._observe_find_marvin_v2=observe
    loop=r._execute_normal_marvin_find_mission_locked
    def altered_result(*args,**kwargs):
        result=loop(*args,**kwargs)
        if fault=='uncertain_delivery': result['proof']['physical_dispatches_or_uncertain']=1
        if fault=='action_history': result['history'].append({'motion_executed':False})
        if fault=='post_action_evidence': result['proof']['post_action_evidence']={}
        if fault=='jit_authority': r._marvin_alignment_observation={'old_authority':True}
        return result
    r._execute_normal_marvin_find_mission_locked=altered_result
    result=step(r)
    assert result['proof_state']=='FAILED_LOCKED' and not result['continuation_available']
    assert not arm(r)['ok'] and not step(r)['execution_authorized']
    assert len(motions(events))==1


@pytest.mark.parametrize('fault',['checkpoint','retry_floor','retry_digest','session','regression','invalid_lidar'])
def test_reverify_rearm_rejects_tampering_and_sensor_discontinuity(tmp_path,monkeypatch,fault):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,_=bundle; begin(bundle); miss(bundle)
    if fault=='checkpoint': r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']=3
    if fault=='retry_floor': r._marvin_live_proof_reverify=replace(r._marvin_live_proof_reverify,camera_floor_stamp=1)
    if fault=='retry_digest': r._marvin_live_proof_reverify_digest='bad'
    if fault=='session': r.lidar_worker.session='new'
    if fault in {'regression','invalid_lidar'}:
        r._active_localization_lidar_is_current=lambda:(r.lidar_worker.session,
            None if fault=='invalid_lidar' else {'acquisition_sequence':r._marvin_live_proof_reverify.lidar_sequence-1})
    assert not arm(r)['ok'] and r._marvin_live_proof_state=='FAILED_LOCKED'
    assert len(motions(events))==1


def test_next_attempt_rejects_same_failed_semantic_stamp(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,_=bundle; begin(bundle); result=miss(bundle)
    floor=result['reverify_camera_floor_source_frame_stamp_ns']; assert arm(r)['ok']
    b.confirm_identity=True; observe=r._observe_find_marvin_v2
    def stale(**kwargs):
        obs=observe(**kwargs); obs['identity_source_frame_stamp_ns']=floor; return obs
    r._observe_find_marvin_v2=stale
    result=step(r)
    assert result['proof_state']=='FAILED_LOCKED' and len(motions(events))==1
    assert floor not in r._marvin_alignment_consumed_source_frame_stamps


def test_no_checkpoint_initial_miss_stays_failed_locked(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,_=bundle
    b.confirm_identity=False; r._set_runtime_state('IDLE'); result=step(r)
    assert result['proof_state']=='FAILED_LOCKED' and not result['continuation_available']
    # Initial proof retains its existing bounded search policy, never creates
    # a recoverable checkpoint from an absent target.
    assert len(motions(events))<=1 and not arm(r)['ok']


def test_explicit_current_frame_without_proposal_is_recoverable(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,clock=bundle; begin(bundle); assert arm(r)['ok']
    def absent(**kwargs):
        clock[0]+=10_000_000
        return [],'not_found',{'latest_source_frame_stamp_ns':clock[0]}
    b._confirm_marvin_proposal_candidates_with_status=absent
    result=step(r)
    assert result['reason']=='Marvin was not found in the current camera frame.'
    assert result['proof_state']=='REVERIFY_DISARMED' and result['continuation_available']
    assert result['reverify_camera_floor_source_frame_stamp_ns']==clock[0] and len(motions(events))==1


def test_restart_has_no_recoverable_checkpoint(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); begin(bundle); miss(bundle)
    new=forward(tmp_path/'new',monkeypatch)[0]
    assert new._marvin_live_proof_state=='UNINITIALIZED'
    assert new._marvin_live_proof_continuation is new._marvin_live_proof_reverify is None
    assert not arm(new)['ok']


def test_concurrent_reverify_rearm_and_step_share_one_permission(tmp_path,monkeypatch):
    bundle=avoidance_bundle(tmp_path,monkeypatch); r,b,_,events,_=bundle; begin(bundle); miss(bundle)
    barrier=threading.Barrier(3); results=[]
    def rearm(): barrier.wait(); results.append(arm(r))
    threads=[threading.Thread(target=rearm) for _ in range(2)]
    for t in threads:t.start()
    barrier.wait()
    for t in threads:t.join(3); assert not t.is_alive()
    assert sum(x['ok'] for x in results)==1
    entered=threading.Event(); release=threading.Event(); original=b.on_semantic
    def hold(): entered.set(); assert release.wait(3)
    b.on_semantic=hold; results=[]
    t=threading.Thread(target=lambda:results.append(step(r))); t.start(); assert entered.wait(3)
    try:
        assert not step(r)['execution_authorized'] and not arm(r)['ok']
    finally:release.set();t.join(3)
    assert not t.is_alive() and results[0]['proof_state']=='REVERIFY_DISARMED'
    assert len(motions(events))==1


def test_normal_continuous_mission_remains_one_semantic_episode(tmp_path,monkeypatch):
    bundle=proof_runtime(tmp_path,monkeypatch,[(0,.6),(0,.58),(0,.5)])
    result=run(bundle[0]); b=bundle[1]
    assert result['state']=='ARRIVED' and len(motions(bundle[3]))==2
    assert b.semantic_calls==1 and len(b.created_trackers)==1
    assert b.MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU==.70
    assert 'proof' not in result and bundle[0]._marvin_live_proof_reverify is None


@pytest.mark.parametrize('failure',['interrupted','delivery_uncertain','stop_failure','post_camera','jit'])
def test_later_authorized_action_failure_still_invalidates_checkpoint(tmp_path,monkeypatch,failure):
    bundle=avoidance_bundle(tmp_path,monkeypatch); r,b,robot,events,_=bundle
    begin(bundle); miss(bundle); b.confirm_identity=True; assert arm(r)['ok']
    if failure=='stop_failure': robot.on_motion=lambda:setattr(robot,'stop',lambda:{'ok':False})
    if failure=='post_camera': robot.on_motion=lambda:setattr(b,'refresh_mode','repeat_action')
    execute=r._execute_single_marvin_strafe
    def fault(**kwargs):
        result=execute(**kwargs)
        if failure=='interrupted': result['full_step_completed']=False; result['interrupted']=True
        if failure=='delivery_uncertain': result['delivery_uncertain']=True
        if failure=='jit': result['ok']=False; result['full_step_completed']=False; result['reason']='jit_transport_ambiguous'
        return result
    r._execute_single_marvin_strafe=fault
    result=step(r)
    assert result['proof_state']=='FAILED_LOCKED' and not result['continuation_available']
    assert not arm(r)['ok']
    assert len(motions(events))<=2


def test_completion_stamp_cannot_authorize_motion_after_miss(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,_,events,_=bundle; begin(bundle)
    c=r._marvin_live_proof_continuation; miss(bundle); assert arm(r)['ok']; b.confirm_identity=True
    original=r._observe_find_marvin_v2
    def replay(**kwargs):
        observation=original(**kwargs)
        observation['source_frame_stamp_ns']=c.previous_stamp
        return observation
    r._observe_find_marvin_v2=replay
    result=step(r)
    assert result['proof_state']=='FAILED_LOCKED' and len(motions(events))==1
    assert result['controller_result']['proof']['source_stamps']==[]


def test_exact_reverify_floor_has_decimal_safe_api_representation(tmp_path,monkeypatch):
    from runtime_api import _precision_safe_source_frame_stamps
    bundle=forward(tmp_path,monkeypatch); begin(bundle); result=miss(bundle)
    stamp=result['reverify_camera_floor_source_frame_stamp_ns']
    encoded=_precision_safe_source_frame_stamps(result)
    assert type(stamp) is int and stamp>2**53
    assert encoded['reverify_camera_floor_source_frame_stamp_ns']==str(stamp)
    assert int(encoded['reverify_camera_floor_source_frame_stamp_ns'])==stamp


def test_bridge_command_during_semantic_miss_prevents_recovery(tmp_path,monkeypatch):
    bundle=forward(tmp_path,monkeypatch); r,b,robot,_,_=bundle; begin(bundle); assert arm(r)['ok']
    original=robot.status; command=['old']
    robot.status=lambda:dict(original(),motion=dict(original()['motion'],last_command_at=command[0]))
    b.confirm_identity=False; b.on_semantic=lambda:command.__setitem__(0,'new')
    assert step(r)['proof_state']=='FAILED_LOCKED' and not arm(r)['ok']


def test_saved_a6c4d675_replay_survives_miss_reaches_bypass_direct_and_arrival(tmp_path,monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.38)]+[(40,1.38)]*24,[None],factory=proof_runtime)
    r,b,_,events,clock=bundle; original=r.world_model.get_lidar_obstacles; clear=[False]
    def scan(**kwargs):
        count=len(motions(events))
        if count==0 or clear[0]:
            clock[0]+=1000;return original(**kwargs)
        state=copy.deepcopy(fixture['cycles'][min(count-1,5)]['lidar'])
        b.sequence+=1;clock[0]+=1000
        state.update(producer_session=r.lidar_worker.session,acquisition_sequence=b.sequence,
            received_monotonic_seconds=time.monotonic(),age_at_receipt_seconds=0.,effective_age_seconds=0.,reason='fresh')
        if count>=7:
            for point in state['local_motion_geometry']['points']:
                point['x_m']-=.05
                point['distance_m']=math.hypot(point['x_m'],point['y_m'])
                point['robot_bearing_deg']=math.degrees(math.atan2(point['y_m'],point['x_m']))
        return state
    r.world_model.get_lidar_obstacles=scan
    results=[begin(bundle)]; miss_summary=None
    for count in range(1,7):
        if count==2:
            c=r._marvin_live_proof_continuation
            assert c.previous_selection['action_type']=='STRAFE_LEFT'
            saved=asdict(c); negative=miss(bundle)
            assert asdict(r._marvin_live_proof_continuation)==saved
            miss_summary={'terminal':negative['controller_result']['state'],
                'proof_state':negative['proof_state'],'floor':negative['reverify_camera_floor_source_frame_stamp_ns'],
                'previous_selection':c.previous_selection,'progress':c.avoidance_history[-1]['actual_route_progress']}
            b.confirm_identity=True
        assert arm(r)['ok']; b.visual_shift=110*(count%2)
        result=step(r); complete(bundle,result,count); results.append(result)
    selected=[x['controller_result']['history'][0]['result'].get('action_type') or
        x['controller_result']['history'][0]['state'] for x in results]
    assert selected==['ADVANCING']+['STRAFE_LEFT']*5+['BYPASS_FORWARD']
    c=r._marvin_live_proof_continuation
    assert c.avoidance['local_avoidance_actions']==6 and c.avoidance['local_bypass_actions']==1
    bypass=results[-1]['controller_result']['history'][0]['result']
    target=bypass['local_detour']['local_bypass']
    assert bypass['direction']=='LEFT' and target['bypass_target_x_m']<=.15 and target['protected_radius_m']==.45
    assert target['bypass_corridor_occupancy']==0 and target['bypass_corridor_overlap_m']==0
    assert bypass['approach_result']['forward_safety']['permitted']
    assert bypass['local_detour']['acquisition_sequence']>results[-1]['controller_result']['avoidance_planning_lidar_sequence']
    assert motions(events)[-1]==('forward',.1,.5)
    assert not step(r)['execution_authorized'] and len(motions(events))==7
    clear[0]=True; b.specs=iter([(0,1.38),(0,1.33)])
    assert arm(r)['ok']; direct=step(r); complete(bundle,direct,7)
    assert direct['controller_result']['history'][0]['state']=='ADVANCING'
    assert r._marvin_live_proof_continuation.previous_selection is None
    assert r._marvin_live_proof_continuation.avoidance['local_bypass_target_x_m'] is None
    direct_count=8
    for cm in range(128,57,-5):
        b.specs=iter([(0,cm/100),(0,(cm-5)/100)])
        assert arm(r)['ok']; result=step(r); complete(bundle,result,direct_count)
        assert result['controller_result']['history'][0]['state']=='ADVANCING'
        assert result['controller_result']['local_avoidance_actions']==6
        direct_count+=1
    b.specs=iter([(0,.5)])
    assert arm(r)['ok']; arrival=step(r)
    assert arrival['controller_result']['state']=='ARRIVED' and not arrival['motion_executed']
    assert arrival['proof_state']=='ARRIVED_DISARMED' and not arrival['continuation_available']
    assert len(motions(events))==direct_count and not arm(r)['ok']
    print('Semantic-miss replay:',json.dumps({'actions':selected+['ADVANCING','ARRIVED'],
        'semantic_miss':miss_summary,'avoidance_budget':6,'bypass_count':1,
        'speed':.1,'duration':.5,'protected_radius':.45,'motion_count':direct_count,
        'selection_injected':False,'new_tracker_per_successful_authorization':True},sort_keys=True))


def test_semantic_miss_preserves_nonzero_bypass_count_and_frozen_bypass_progress(tmp_path,monkeypatch):
    from test_marvin_local_bypass import OPEN_LEFT
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*14,[OPEN_LEFT],factory=proof_runtime)
    r=bundle[0];begin(bundle);assert arm(r)['ok'];complete(bundle,step(r),1)
    c=r._marvin_live_proof_continuation
    assert c.previous_selection['action_type']=='BYPASS_FORWARD'
    assert c.avoidance['local_bypass_actions']==1 and c.avoidance['local_avoidance_actions']==2
    frozen=copy.deepcopy(c.previous_selection['first_post_action_bypass_progress'])
    miss(bundle)
    assert r._marvin_live_proof_continuation is c
    assert c.previous_selection['first_post_action_bypass_progress']==frozen
    assert c.avoidance['local_bypass_actions']==1 and c.avoidance['local_avoidance_actions']==2
