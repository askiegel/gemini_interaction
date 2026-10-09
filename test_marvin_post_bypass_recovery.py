"""Offline geometry, production-loop and explicit-proof recovery regressions."""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from marvin_local_obstacle_avoidance import select_marvin_escape_action, MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import evaluate_route_progress
from runtime import _marvin_proof_selection_history
from test_find_marvin_closed_loop import motions, run
from test_marvin_lateral_avoidance import strafe_runtime
from test_marvin_local_bypass import OPEN_LEFT, scene_plan, bypass_plan
from test_marvin_live_proof_continuation import initial, arm, step, complete


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Recovery tests cannot access robot/network/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def live_geometry():
    f = json.loads((Path(__file__).parent / 'test_fixtures/marvin_post_bypass_recovery_geometry.json').read_text())
    state = f['lidar']
    state.update(received_monotonic_seconds=time.monotonic(), age_at_receipt_seconds=0., effective_age_seconds=0.)
    return f, state, f['previous_selection']


def select(state, association, old, **kwargs):
    return select_marvin_escape_action(state, association, expected_session=state['producer_session'],
        allow_strafe=kwargs.pop('allow_strafe', True), previous_selection=old, **kwargs)


def recovered_plan():
    f, state, old = live_geometry()
    result = select(state, f['association'], old, remaining_avoidance_actions=3)
    return f, state, result


def test_live_geometry_naturally_recovers_only_left_and_retains_failed_bypass():
    f, state, old = live_geometry(); before = copy.deepcopy(old)
    result = select(state, f['association'], old, remaining_avoidance_actions=3)
    assert result['action_type'] == 'STRAFE_LEFT'
    assert result['direction'] == 'LEFT'
    assert result['reason'] == 'find_marvin_post_bypass_lateral_recovery_selected'
    assert result['post_bypass_lateral_recovery_used'] is True
    assert result['post_bypass_lateral_recovery_selected'] is True
    assert result['ineffective_action_types'] == ['BYPASS_FORWARD']
    assert result['actual_route_progress'] == old['first_post_action_bypass_progress']
    assert not result['bypass_side_change_allowed'] and old == before
    assert 'BYPASS_FORWARD' not in result['options']
    assert result['route'] == f['expected_route']
    predicted = result['options']['STRAFE_LEFT']['route_progress']
    assert predicted['corridor_overlap_reduction_m'] == pytest.approx(.04164157700938681)
    assert predicted['centerline_clearance_improvement_m'] == pytest.approx(.04297997941182463)


@pytest.mark.parametrize('fault', ['stale', 'invalid', 'session', 'coverage', 'unsafe', 'no_lateral',
    'tiny_prediction', 'used', 'malformed_used', 'no_budget', 'malformed_budget', 'missing_side'])
def test_recovery_fails_closed_without_all_fresh_admission_conditions(fault):
    f, state, old = live_geometry(); kw = {}
    if fault == 'stale': state['received_monotonic_seconds'] -= 1.
    if fault == 'invalid': state['valid'] = False
    if fault == 'session': state['producer_session'] = 'wrong'
    if fault == 'coverage': state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'unsafe': state['local_motion_geometry']['points'].append({'x_m':0., 'y_m':.40})
    if fault == 'no_lateral': kw['allow_strafe'] = False
    if fault == 'tiny_prediction': kw['strafe_duration_limit'] = .1
    if fault == 'used': old['post_bypass_lateral_recovery_used'] = True
    if fault == 'malformed_used': old['post_bypass_lateral_recovery_used'] = 1
    if fault == 'no_budget': kw['remaining_avoidance_actions'] = 0
    if fault == 'malformed_budget': kw['remaining_avoidance_actions'] = True
    if fault == 'missing_side': old['direction'] = None
    result = select_marvin_escape_action(state, f['association'], expected_session=f['lidar'].get('producer_session') if fault != 'session' else 'original',
        allow_strafe=kw.pop('allow_strafe', True), previous_selection=old, **kw)
    assert result['action_type'] is None
    assert not result.get('post_bypass_lateral_recovery_selected', False)


@pytest.mark.parametrize('used', [True, 1, 'false'])
def test_used_marker_cannot_be_bypassed_by_removing_lateral_from_ineffective_set(used):
    f, state, old = live_geometry(); old['ineffective_action_types'] = ['BYPASS_FORWARD']
    old['post_bypass_lateral_recovery_used'] = used
    result = select(state, f['association'], old)
    assert result['action_type'] is None
    assert 'BYPASS_FORWARD' in result['ineffective_action_types']


def test_failed_recovery_freezes_outcome_and_cannot_repeat_even_after_alignment_geometry_changes():
    f, state, recovery = recovered_plan()
    recovery['first_post_action_lateral_recovery_progress'] = evaluate_route_progress(recovery['route'], recovery['route'])
    frozen = copy.deepcopy(recovery['first_post_action_lateral_recovery_progress'])
    # Fresh geometry might make the old comparison look better after alignment;
    # it cannot overwrite the first stopped outcome of the recovery.
    for point in state['local_motion_geometry']['points']:
        if point['x_m'] > .4: point['y_m'] -= .08
    for _ in range(3):
        state['acquisition_sequence'] += 1
        result = select(state, f['association'], recovery)
        assert result['action_type'] is None
        assert set(result['ineffective_action_types']) >= {'BYPASS_FORWARD','STRAFE_LEFT'}
        assert result['actual_route_progress'] == frozen
        assert result['post_bypass_lateral_recovery_used'] is True
    assert recovery['first_post_action_lateral_recovery_progress'] == frozen


def test_successful_recovery_opens_new_epoch_without_automatically_restoring_bypass():
    f, state, recovery = recovered_plan()
    improved = copy.deepcopy(recovery['route']); improved['corridor_overlap_m'] -= .02; improved['blocking_obstacle_overlap_m'] -= .02
    recovery['first_post_action_lateral_recovery_progress'] = evaluate_route_progress(recovery['route'], improved)
    result = select(state, f['association'], recovery)
    assert result['action_type'] == 'STRAFE_LEFT'
    assert result['progress_improved'] is True
    assert result['ineffective_action_types'] == ['BYPASS_FORWARD']
    assert result['post_bypass_lateral_recovery_used'] is False
    assert not result.get('post_bypass_lateral_recovery_selected')
    assert 'BYPASS_FORWARD' not in result['options']
    # Only a later ordinary physical action with its own progress can reopen
    # the ordinary epoch rules, never the recovery's prediction alone.
    result['route'] = dict(result['route'], corridor_overlap_m=result['route']['corridor_overlap_m']+.02,
                           blocking_obstacle_overlap_m=result['route']['blocking_obstacle_overlap_m']+.02)
    later = select(state, f['association'], result)
    assert 'BYPASS_FORWARD' not in later.get('ineffective_action_types', [])


def test_selective_reconsideration_keeps_unrelated_ineffective_memory():
    f, state, old = live_geometry(); old['ineffective_action_types'] += ['STRAFE_RIGHT','TURN_LEFT']
    result = select(state, f['association'], old)
    assert result['action_type'] == 'STRAFE_LEFT'
    assert set(result['ineffective_action_types']) == {'BYPASS_FORWARD','STRAFE_RIGHT','TURN_LEFT'}


@pytest.mark.parametrize('side', ['LEFT','RIGHT'])
def test_recovery_is_symmetric_but_never_changes_established_side(side):
    _, old = bypass_plan()
    if side == 'RIGHT':
        points = [(x,-y) for x,y in OPEN_LEFT]
        _, first = scene_plan(points); _, old = scene_plan(points, previous=first)
    points = OPEN_LEFT if side == 'LEFT' else [(x,-y) for x,y in OPEN_LEFT]
    _, result = scene_plan(points, previous=old)
    assert result['action_type'] == 'STRAFE_'+side and result['direction'] == side
    assert not result['bypass_side_change_allowed']
    assert 'BYPASS_FORWARD' in result['ineffective_action_types']


def test_repeated_planning_does_not_mutate_history_or_accumulate_recovery_credits():
    f,state,old = live_geometry(); before = copy.deepcopy(old)
    a = select(state,f['association'],old); b = select(state,f['association'],old)
    assert a['action_type'] == b['action_type'] == 'STRAFE_LEFT' and old == before
    a['first_post_action_lateral_recovery_progress'] = evaluate_route_progress(a['route'],a['route'])
    assert select(state,f['association'],a)['action_type'] is None


SECOND = [(.65,-.29),(0.,1.2),(0.,-.65)]

def recovery_bundle(tmp_path, monkeypatch, *, success=False):
    # Two measured lateral actions are needed to establish the 0.15 m side
    # separation. Once established, the new priority legitimately hands off.
    # A newly observed point occupies the forward bypass capsule after step
    # one, but leaves lateral motion safe. This ends continuation and exercises
    # the existing one-use recovery rather than a still-clear bypass episode.
    blocked = SECOND + [(.58,0.)]
    scenes = [[(.65,-.10),(0.,1.2),(0.,-.65)],
              [(.65,-.14),(0.,1.2),(0.,-.65)], SECOND, blocked,
              [(.65,-.34),(0.,1.2),(0.,-.65),(.58,-.08)] if success else blocked, None]
    bundle, flags, client = strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*40,scenes)
    r,_,_,_,clock = bundle; read = r.world_model.get_lidar_obstacles
    def acquire(**kw):
        clock[0] += 1000
        return read(**kw)
    r.world_model.get_lidar_obstacles = acquire
    return bundle,flags,client


def checkpoint_after_bypass(bundle):
    results = [initial(bundle)];complete(bundle,results[0],0)
    for count in (1,2):
        assert arm(bundle[0])['ok'];results.append(step(bundle[0]));complete(bundle,results[-1],count)
    r=bundle[0];c=r._marvin_live_proof_continuation
    assert c.avoidance['local_avoidance_actions']==3 and c.avoidance['local_bypass_actions']==1
    assert c.previous_selection['action_type']=='BYPASS_FORWARD'
    assert c.previous_selection['first_post_action_bypass_progress']['meaningful_progress'] is False
    return results


@pytest.mark.parametrize('success',[False,True])
def test_proof_recovery_uses_real_shared_selector_and_preserves_budget_stop_and_exact_sources(tmp_path,monkeypatch,success):
    bundle,_,_=recovery_bundle(tmp_path,monkeypatch,success=success);r,b,robot,events,_=bundle
    results=checkpoint_after_bypass(bundle);prior=r._marvin_live_proof_continuation
    assert arm(r)['ok'];result=step(r);complete(bundle,result,3);c=r._marvin_live_proof_continuation
    assert c.avoidance['local_avoidance_actions']==4 and c.avoidance['local_bypass_actions']==1
    assert c.previous_selection['post_bypass_lateral_recovery_used'] is True
    assert c.previous_selection['post_bypass_lateral_recovery_selected'] is True
    progress=c.previous_selection['first_post_action_lateral_recovery_progress'];assert progress['meaningful_progress'] is success
    assert c.avoidance_history[2]['actual_route_progress']==prior.avoidance_history[2]['actual_route_progress']
    assert motions(events)==[('strafe',.08,1.),('strafe',.08,1.),('forward',.1,.5),('strafe',.08,1.)]
    row=result['controller_result']['history'][0];a=row['result']
    assert a['source_stamp_consumed'] and a['full_step_completed'] and a['bridge_stop_confirmed']
    assert row['source_frame_stamp_ns']>prior.camera_floor_stamp and type(row['source_frame_stamp_ns']) is int
    assert a['local_detour']['accepted'] and a['local_detour']['post_bypass_lateral_recovery_used']
    assert a['lateral_step']['lateral_safety']['protected_radius_m']==.45
    assert len(a['lateral_step']['lateral_safety']['required_sectors'])==8
    assert not step(r)['execution_authorized'] and len(motions(events))==4
    assert arm(r)['ok'];later=step(r)
    if not success:
        assert later['controller_result']['state']=='BLOCKED'
        assert later['controller_result']['proof']['dispatch_opportunities']==0
        assert len(motions(events))==4
        assert {'STRAFE_LEFT','BYPASS_FORWARD'} <= set(later['controller_result']['local_avoidance_history'][-1]['selection']['ineffective_action_types'])
    else:
        complete(bundle,later,4)
        assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']==5
        assert r._marvin_live_proof_continuation.avoidance['local_bypass_actions']==1
    assert robot.status()['motion']['streaming'] is False


def test_production_continuous_loop_uses_one_recovery_and_stops_without_loop(tmp_path,monkeypatch):
    bundle,_,_=recovery_bundle(tmp_path,monkeypatch);result=run(bundle[0])
    assert result['state']=='BLOCKED'
    assert result['local_avoidance_actions']==4 and result['local_bypass_actions']==1
    assert len(motions(bundle[3]))==4
    assert result['reason']=='find_marvin_blocked_wait_exhausted'
    assert result['blocked_wait_reason']=='find_marvin_local_avoidance_no_progress'
    assert result['local_avoidance_history'][-2]['selection']['post_bypass_lateral_recovery_selected']


@pytest.mark.parametrize('phase',['bypass','recovery'])
def test_alignment_never_resets_recovery_marker_or_frozen_outcome(tmp_path,monkeypatch,phase):
    bundle,_,_=recovery_bundle(tmp_path,monkeypatch);r,b,_,events,_=bundle
    checkpoint_after_bypass(bundle)
    if phase=='recovery':
        assert arm(r)['ok'];complete(bundle,step(r),3)
    prior=copy.deepcopy(r._marvin_live_proof_continuation.previous_selection);count=len(motions(events))
    b.specs=iter([(100,1.1),(0,1.1)])
    assert arm(r)['ok'];aligned=step(r);complete(bundle,aligned,count)
    assert aligned['controller_result']['history'][0]['state']=='ALIGNING'
    assert r._marvin_live_proof_continuation.previous_selection==prior
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']==(3 if phase=='bypass' else 4)


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_no_remaining_avoidance_budget_can_enable_recovery(limit):
    f,state,old=live_geometry();assert select(state,f['association'],old,remaining_avoidance_actions=limit)['action_type'] is None


def test_six_action_runtime_budget_remains_authoritative(tmp_path,monkeypatch):
    scenes=[OPEN_LEFT,OPEN_LEFT]+[[(.65-.011*i,-.25),(0.,1.2),(0.,-.65)] for i in range(1,8)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*20,scenes)
    result=run(bundle[0])
    assert result['reason']=='find_marvin_local_avoidance_exhausted'
    assert result['local_avoidance_actions']==MAX_LOCAL_AVOIDANCE_ACTIONS==6
    assert len(motions(bundle[3]))==6


def test_proof_history_retains_markers_and_frozen_progress_but_no_motion_permission():
    f,state,result=recovered_plan();result['first_post_action_lateral_recovery_progress']=evaluate_route_progress(result['route'],result['route'])
    retained=_marvin_proof_selection_history(result)
    assert retained['post_bypass_lateral_recovery_used'] is True
    assert retained['first_post_action_lateral_recovery_progress']==result['first_post_action_lateral_recovery_progress']
    assert 'options' not in retained and 'local_bypass' not in retained


@pytest.mark.parametrize('fault',['unsafe','budget'])
def test_recovery_reacquires_jit_geometry_and_budget_before_dispatch(tmp_path,monkeypatch,fault):
    bundle,_,_=recovery_bundle(tmp_path,monkeypatch);r=bundle[0];checkpoint_after_bypass(bundle)
    execute=r._execute_single_marvin_strafe
    def veto(**kw):
        assert kw['local_detour_context']['remaining_avoidance_actions']==3
        if fault=='budget':kw['local_detour_context']['remaining_avoidance_actions']=0
        else:
            read=r.world_model.get_lidar_obstacles
            def unsafe(**args):
                sample=read(**args);sample['local_motion_geometry']['points'].append({'x_m':0.,'y_m':.40})
                return sample
            r.world_model.get_lidar_obstacles=unsafe
        return execute(**kw)
    r._execute_single_marvin_strafe=veto
    assert arm(r)['ok'];result=step(r)
    assert result['controller_result']['state']=='BLOCKED'
    assert result['controller_result']['history'][0]['result']['reason']=='marvin_local_detour_jit_veto'
    assert len(motions(bundle[3]))==3 and result['controller_result']['local_avoidance_actions']==3


def test_saved_a6c4d675_extended_replay_continues_then_direct_forward_and_arrives(tmp_path,monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.38)]+[(40,1.38)]*30,[None])
    r,b,robot,events,clock=bundle;read=r.world_model.get_lidar_obstacles;clear=[False]
    # Replay saved scans, followed by an explicitly synthetic clear-route
    # continuation. These counterfactual scenes do not imply measured travel.
    def acquire(**kw):
        count=len(motions(events));clock[0]+=1000
        if count==0 or clear[0]:return read(**kw)
        idx=0 if count==1 else 4 if count==2 else 5
        sample=copy.deepcopy(fixture['cycles'][idx]['lidar']);b.sequence+=1
        sample.update(producer_session=r.lidar_worker.session,acquisition_sequence=b.sequence,
            received_monotonic_seconds=time.monotonic(),age_at_receipt_seconds=0.,effective_age_seconds=0.,reason='fresh')
        return sample
    r.world_model.get_lidar_obstacles=acquire
    results=[initial(bundle)];complete(bundle,results[0],0)
    for count in range(1,4):
        assert arm(r)['ok'];results.append(step(r));complete(bundle,results[-1],count)
    selected=[x['controller_result']['history'][0]['result'].get('action_type') or x['controller_result']['history'][0]['state'] for x in results]
    assert selected==['ADVANCING','STRAFE_LEFT','BYPASS_FORWARD','BYPASS_FORWARD']
    assert results[3]['controller_result']['history'][0]['result']['local_detour']['bypass_continuation']
    assert r._marvin_live_proof_continuation.previous_selection['bypass_episode']['step']==2
    assert results[2]['controller_result']['local_avoidance_history'][-1]['actual_route_progress']['meaningful_progress'] is False
    assert results[3]['controller_result']['local_avoidance_actions']==3
    clear[0]=True;b.specs=iter([(0,1.38),(0,1.33)])
    assert arm(r)['ok'];direct=step(r);complete(bundle,direct,4)
    assert direct['controller_result']['history'][0]['state']=='ADVANCING'
    assert r._marvin_live_proof_continuation.previous_selection is None
    assert not direct['controller_result']['local_bypass_active']
    for cm in range(133,50,-5):
        b.specs=iter([(0,cm/100),(0,max(.5,(cm-5)/100))])
        assert arm(r)['ok'];forward=step(r);complete(bundle,forward,len(motions(events))-1)
    b.specs=iter([(0,.5)])
    assert arm(r)['ok'];arrived=step(r)
    assert arrived['controller_result']['state']=='ARRIVED' and not arrived['motion_executed']
    assert arrived['proof_state']=='ARRIVED_DISARMED' and not arrived['continuation_available']
    assert robot.status()['motion']['streaming'] is False
    print('Bounded bypass continuation replay:',json.dumps({'saved_fixture':fixture['mission_id'],
        'selected':selected+['FORWARD','ARRIVED'],'avoidance_actions':3,'bypass_actions':2,
        'failed_bypass_progress':results[2]['controller_result']['local_avoidance_history'][-1]['actual_route_progress'],
        'continuation_reason':results[3]['controller_result']['history'][0]['result']['local_detour']['reason']}))
