"""Offline bounded episodes: recorded production geometry and independent fakes."""
import copy
import json
from pathlib import Path
import socket
import time

import pytest

from marvin_bypass_episode import (
    MAX_BYPASS_EPISODE_STEPS, bypass_completion, bypass_continuation,
    end_bypass_episode, start_bypass_episode,
)
from marvin_local_bypass import evaluate_avoidance_progress
from marvin_local_obstacle_avoidance import select_marvin_escape_action, MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import evaluate_marvin_route
from runtime import _marvin_proof_selection_history
from test_find_marvin_closed_loop import run, motions
from test_marvin_local_bypass import ASSOCIATION, OPEN_LEFT, bypass_plan, scene_plan
from test_marvin_lateral_avoidance import strafe_runtime
from test_marvin_live_proof_continuation import initial, arm, step, complete
from test_marvin_bypass_handoff import assert_sensor_contracts


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Continuation tests cannot access robot/network/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def fresh(state):
    state = copy.deepcopy(state)
    state.update(received_monotonic_seconds=time.monotonic(), age_at_receipt_seconds=0., effective_age_seconds=0.)
    return state


def select(state, previous=None, remaining=6, association=ASSOCIATION, session=None):
    return select_marvin_escape_action(state, association,
        expected_session=session or state['producer_session'], allow_strafe=True,
        previous_selection=previous, remaining_avoidance_actions=remaining)


def completed_result(selection):
    assert selection['action_type'] == 'BYPASS_FORWARD'
    return dict(ok=True, execution_authorized=True, motion_executed=True,
        full_step_completed=True, source_stamp_consumed=True, interrupted=False,
        source_frame_stamp_ns=1000+selection['acquisition_sequence'], action_type='BYPASS_FORWARD',
        local_detour=selection, bridge_stop_confirmed=True, stop_result={'ok': True},
        approach_result={'forward_result': dict(ok=True, executed=True,
            automatic_stop=True, returned_immediately=False, linear_x=.1, linear_y=0.,
            angular_z=0., duration=.5)})


def zero_bridge():
    return dict(ok=True, status='READY', ros_ready=True,
        motion=dict(linear_x=0., linear_y=0., angular_z=0., streaming=False))


def stopped(selection, state, *, association=ASSOCIATION, result=None, bridge=None):
    """Historical outcomes from independently specified scans; no integrated travel."""
    old = copy.deepcopy(selection)
    evidence = (old['producer_session'], old['acquisition_sequence'])
    old['bypass_step_completion'] = bypass_completion(
        result or completed_result(old), bridge or zero_bridge(), evidence)
    route = evaluate_marvin_route(state, association, expected_session=state['producer_session'])
    progress = evaluate_avoidance_progress(old, route, old['local_bypass'])
    old['first_post_action_bypass_progress'] = progress
    old['bypass_step_outcome'] = dict(progress=copy.deepcopy(progress),
        producer_session=state['producer_session'],
        action_acquisition_sequence=old['acquisition_sequence'], acquisition_sequence=state['acquisition_sequence'])
    return old


def neutral_case():
    state, first = bypass_plan()
    state['acquisition_sequence'] = first['acquisition_sequence']+1
    old = stopped(first, state)
    assert not old['first_post_action_bypass_progress']['meaningful_progress']
    return state, old


def live_case():
    f = json.loads((Path(__file__).parent/'test_fixtures/marvin_2033f0c3_bypass_continuation.json').read_text())
    before, after = fresh(f['pre_bypass_lidar']), fresh(f['post_bypass_lidar'])
    first = select(before, f['previous_strafe_selection'], 3, f['pre_bypass_association'])
    assert first['action_type'] == 'BYPASS_FORWARD' and first['bypass_handoff']['selected']
    result = copy.deepcopy(f['recorded_bypass_result'])
    result['source_frame_stamp_ns'] = int(result['source_frame_stamp_ns'])
    result['local_detour'] = first
    old = stopped(first, after, association=f['post_bypass_association'], result=result)
    progress = {k:v for k,v in f['recorded_bypass_progress'].items() if k not in {
        'route','target_association','progress_improved','actual_route_occupancy',
        'actual_max_overlap_m','actual_blocker_centerline_clearance_m'}}
    # Exact recorded first outcome owns attribution, not a later alignment.
    old['first_post_action_bypass_progress'] = progress
    old['bypass_step_outcome']['progress'] = copy.deepcopy(progress)
    return f, before, after, old


def test_recorded_production_first_worsening_bypass_naturally_continues():
    f, _, after, old = live_case()
    assert old['bypass_step_completion']['confirmed']
    assert old['first_post_action_bypass_progress']['bypass_longitudinal_progress_m'] == pytest.approx(-.006878655228392039)
    second = select(after, old, 2, f['post_bypass_association'])
    assert second['action_type'] == 'BYPASS_FORWARD'
    assert second['bypass_continuation'] and second['bypass_episode_step'] == 2
    assert second['progress_improved'] is False and second['meaningful_progress'] is False
    assert second['actual_route_progress'] == old['first_post_action_bypass_progress']
    assert 'BYPASS_FORWARD' not in second['ineffective_action_types']
    assert second['local_bypass']['bypass_corridor_occupancy'] == 0
    assert second['local_bypass']['bypass_corridor_overlap_m'] == 0.
    assert second['local_bypass']['acquisition_sequence'] > old['bypass_step_completion']['acquisition_sequence']


def test_neutral_first_step_continues_without_claiming_measured_success():
    state, old = neutral_case()
    second = select(state, old, 4)
    assert second['action_type'] == 'BYPASS_FORWARD' and second['bypass_continuation']
    assert second['bypass_episode_step'] == 2 and second['bypass_episode_max_steps'] == 3
    assert not second['meaningful_progress'] and not second['progress_improved']
    assert second['options']['BYPASS_FORWARD']['route_progress']['meaningful_progress']  # Prediction only.


def test_genuine_measured_passage_keeps_existing_success_path():
    _, first = bypass_plan()
    state, _ = scene_plan([(.60,-.25),(0.,1.2),(0.,-.65)])
    state['acquisition_sequence'] = first['acquisition_sequence']+1
    old = stopped(first, state)
    second = select(state, old)
    assert second['action_type'] == 'BYPASS_FORWARD'
    assert second['meaningful_progress'] and not second['bypass_continuation']
    assert second['meaningful_progress_reason'] == 'bypass_longitudinal_passage_improved'
    assert second['bypass_episode_step'] == 1  # Genuine progress ends the old epoch.


@pytest.mark.parametrize('fault', ['stale','invalid','session','coverage','hazard','separation','reversed_geometry'])
def test_fresh_geometry_failures_reject_continuation(fault):
    state, old = neutral_case()
    if fault == 'stale': state['received_monotonic_seconds'] -= 1.
    if fault == 'invalid': state['valid'] = False
    if fault == 'session': state['producer_session'] = 'other'
    if fault == 'coverage': state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'hazard': state['local_motion_geometry']['points'].append({'x_m':.17,'y_m':.40})
    if fault in ['separation','reversed_geometry']:
        for point in state['local_motion_geometry']['points']:
            if point['x_m'] > .4: point['y_m'] = -.10 if fault == 'separation' else .25
    result = select(state, old, session=old['producer_session'])
    assert result['action_type'] != 'BYPASS_FORWARD' and not result.get('bypass_continuation')


@pytest.mark.parametrize('field,value', [
    ('bypass_corridor_occupancy',1),('bypass_corridor_occupancy',False),
    ('bypass_corridor_overlap_m',.001),('route_to_bypass_obstructed',True),
    ('protected_radius_m',.44),('bypass_target_x_m',None),
])
def test_corridor_certificate_cannot_override_individual_gates(field, value):
    state, old = neutral_case();bypass=copy.deepcopy(select(state, old)['local_bypass'])
    bypass[field]=value
    if field == 'bypass_target_x_m':
        # Target rebuilding is tested through the real selector below; malformed
        # targets must fail closed even if a caller supplies a permission bit.
        bypass['bypass_distance_m']=None
    d=bypass_continuation(old,bypass,old['first_post_action_bypass_progress'],
        expected_session=state['producer_session'],current_sequence=state['acquisition_sequence'])
    assert not d['eligible']


@pytest.mark.parametrize('fault', ['missing_completion','missing_outcome','missing_frozen','future_outcome',
    'same_sequence','wrong_completion_session','wrong_outcome_session','wrong_side','wrong_step',
    'malformed_step','bound','inactive','suppression','recovery','stationary_recovery'])
def test_history_and_bounds_fail_closed(fault):
    state,old=neutral_case()
    if fault=='missing_completion':old.pop('bypass_step_completion')
    if fault=='missing_outcome':old.pop('bypass_step_outcome')
    if fault=='missing_frozen':old.pop('first_post_action_bypass_progress')
    if fault=='future_outcome':old['bypass_step_outcome']['acquisition_sequence']=999999
    if fault=='same_sequence':old['bypass_step_outcome']['acquisition_sequence']=old['bypass_step_completion']['acquisition_sequence']
    if fault=='wrong_completion_session':old['bypass_step_completion']['producer_session']='wrong'
    if fault=='wrong_outcome_session':old['bypass_step_outcome']['producer_session']='wrong'
    if fault=='wrong_side':old['bypass_episode']['side']='RIGHT'
    if fault=='wrong_step':old['bypass_step_completion']['step']=2
    if fault=='malformed_step':old['bypass_episode']['step']=True
    if fault=='bound':old['bypass_episode']['step']=3
    if fault=='inactive':old=end_bypass_episode(old,'jit_veto')
    if fault=='suppression':old['ineffective_action_types'].append('BYPASS_FORWARD')
    if fault=='recovery':old['post_bypass_lateral_recovery_used']=True
    if fault=='stationary_recovery':old['stationary_lateral_reconsidered']=True
    result=select(state,old)
    assert result['action_type']!='BYPASS_FORWARD' and not result.get('bypass_continuation')


@pytest.mark.parametrize('fault', ['uncertain','transport_failed','unconfirmed_transport','interrupted',
    'partial','stop_failed','bridge_failed','ros_failed','linear_x','linear_y','yaw','streaming',
    'missing_stamp_consumption','wrong_action_sequence','overspeed','overduration'])
def test_transport_STOP_and_bridge_failures_cannot_certify_next_step(fault):
    state,old=neutral_case();result=completed_result(old);bridge=zero_bridge();evidence=(old['producer_session'],old['acquisition_sequence'])
    transport=result['approach_result']['forward_result']
    if fault=='uncertain':transport['delivery_uncertain']=True
    if fault=='transport_failed':transport['ok']=False
    if fault=='unconfirmed_transport':transport.pop('executed')
    if fault=='interrupted':result['interrupted']=True
    if fault=='partial':result['full_step_completed']=False
    if fault=='stop_failed':result['stop_result']['ok']=False
    if fault=='bridge_failed':bridge['ok']=False
    if fault=='ros_failed':bridge['ros_ready']=False
    if fault in ['linear_x','linear_y']:bridge['motion'][fault]=.01
    if fault=='yaw':bridge['motion']['angular_z']=.01
    if fault=='streaming':bridge['motion']['streaming']=True
    if fault=='missing_stamp_consumption':result['source_stamp_consumed']=False
    if fault=='wrong_action_sequence':evidence=(old['producer_session'],999)
    if fault=='overspeed':transport['linear_x']=.11
    if fault=='overduration':transport['duration']=.51
    old['bypass_step_completion']=bypass_completion(result,bridge,evidence)
    assert old['bypass_step_completion']['confirmed'] is False
    assert select(state,old)['action_type']!='BYPASS_FORWARD'


def test_recomputed_target_ignores_old_permission_and_every_read_is_advisory():
    state,old=neutral_case();old['local_bypass'].update(bypass_target_x_m=999.,bypass_target_y_m=9.)
    before=copy.deepcopy(old)
    for _ in range(8):
        selected=select(state,old)
        assert selected['bypass_episode_step']==2
        assert selected['local_bypass']['bypass_target_x_m']==.15
        assert selected['local_bypass']['bypass_target_y_m']==0.
        assert selected['local_bypass']['acquisition_sequence']==state['acquisition_sequence']
    assert old==before
    # A planning result without a new completion certificate cannot authorize #3.
    assert select(state,selected)['action_type']!='BYPASS_FORWARD'


@pytest.mark.parametrize('horizon,maximum',[(.15,3),(.10,2),(.05,1)])
def test_shorter_initial_horizon_reduces_hard_physical_attempt_bound(horizon,maximum):
    _,first=bypass_plan();bypass=dict(first['local_bypass'],bypass_distance_m=horizon)
    assert start_bypass_episode(bypass,'test')['max_steps']==maximum
    assert MAX_BYPASS_EPISODE_STEPS==3


def test_three_no_progress_steps_then_authoritative_suppression_and_only_one_recovery():
    state,old=neutral_case()
    for remaining,number in [(4,2),(3,3)]:
        selected=select(state,old,remaining)
        assert selected['action_type']=='BYPASS_FORWARD' and selected['bypass_episode_step']==number
        assert selected['meaningful_progress'] is False
        state['acquisition_sequence']+=1
        old=stopped(selected,state)
    fourth=select(state,old,2)
    assert fourth['action_type']=='STRAFE_LEFT' and fourth['post_bypass_lateral_recovery_selected']
    assert 'BYPASS_FORWARD' in fourth['ineffective_action_types']
    assert fourth['bypass_episode_reason']=='bypass_episode_bound_reached'
    again=select(state,fourth,1)
    assert again['action_type'] is None and 'BYPASS_FORWARD' in again['ineffective_action_types']


@pytest.mark.parametrize('remaining',[0,-1,True,1.5])
def test_global_six_action_bound_precedes_episode_credit(remaining):
    state,old=neutral_case();result=select(state,old,remaining)
    assert result['action_type'] is None and result['reason']=='find_marvin_local_avoidance_exhausted'
    assert MAX_LOCAL_AVOIDANCE_ACTIONS==6
    assert result['bypass_episode']['active'] is False
    assert 'BYPASS_FORWARD' in result['ineffective_action_types']


def episode_bundle(tmp_path,monkeypatch,*,clear_after=None,unsafe=False,three_strafes=False):
    initial_scenes=[[(.65,-.10)],[(.65,-.125)],[(.65,-.14)],[(.65,-.18)]] if three_strafes else [[(.65,-.10)],[(.65,-.18)]]
    scenes=initial_scenes+[[(.65,-.18)]]*3
    initial_strafes=3 if three_strafes else 1
    if clear_after is not None:
        scenes=initial_scenes+[[(.65,-.18)]]*(clear_after-1)+[None]
    if unsafe:scenes=initial_scenes+[[(.65,-.18),(.17,.40)]]
    count=initial_strafes+(clear_after or 4)
    specs=([(0,1.1)]*count+[(0,round(x/100,2)) for x in range(105,49,-5)]
        if clear_after else [(0,1.1)]*40)
    bundle,flags,client=strafe_runtime(tmp_path,monkeypatch,specs,scenes)
    read=bundle[0].world_model.get_lidar_obstacles
    def acquire(**kwargs):
        bundle[4][0]+=1000
        return read(**kwargs)
    bundle[0].world_model.get_lidar_obstacles=acquire
    return bundle,flags,client


@pytest.mark.parametrize('bypasses',[2,3])
def test_production_continuation_then_route_clear_resumes_ordinary_forward(tmp_path,monkeypatch,bypasses):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch,clear_after=bypasses)
    result=run(bundle[0])
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==1+bypasses and result['local_bypass_actions']==bypasses
    assert result['completed_forward_actions']>0 and not result['local_bypass_active']
    rows=[h for h in result['history'] if h['result'].get('action_type')=='BYPASS_FORWARD']
    assert all(h['result']['local_detour']['phase']=='PASS_OBSTACLE' for h in rows)
    assert all('bypass_episode_step' not in h['result']['local_detour'] for h in rows)
    assert result['local_avoidance_history'][2]['selection']['meaningful_progress'] is False
    ordinary=[h for h in result['history'] if h['state']=='ADVANCING']
    assert ordinary and all(h['observation']['arrival']['route_to_marvin_obstructed'] is False for h in ordinary)
    assert all(h['result'].get('action_type')!='BYPASS_FORWARD' for h in ordinary)
    assert_sensor_contracts(result,bundle[0])


def test_production_neutral_pass_remains_safe_until_phase_stagnation(tmp_path,monkeypatch):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch)
    result=run(bundle[0])
    assert result['reason']=='find_marvin_pass_stagnation_exhausted'
    assert result['local_bypass_actions']==12 and result['local_avoidance_actions']==13
    assert motions(bundle[3])==[('strafe',.08,1.)]+[('forward',.1,.5)]*12
    assert result['blocked_wait_recheck_count']==0
    assert all(r['selection']['phase']=='PASS_OBSTACLE' for r in result['local_avoidance_history'][1:])
    assert_sensor_contracts(result,bundle[0])


def test_production_unsafe_corridor_dispatches_no_second_bypass(tmp_path,monkeypatch):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch,unsafe=True)
    result=run(bundle[0])
    assert result['local_bypass_actions']==1 and result['local_avoidance_actions']==2
    assert motions(bundle[3])==[('strafe',.08,1.),('forward',.1,.5)]
    assert result['state']=='BLOCKED' and result['stop_result']['ok']


def test_production_three_strafes_then_pass_is_bounded_by_phase_stagnation(tmp_path,monkeypatch):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch,three_strafes=True)
    result=run(bundle[0])
    assert result['reason']=='find_marvin_pass_stagnation_exhausted'
    assert result['local_avoidance_actions']==15 and result['local_bypass_actions']==12
    assert motions(bundle[3])==[('strafe',.08,1.)]*3+[('forward',.1,.5)]*12
    assert_sensor_contracts(result,bundle[0])


def test_proof_checkpoints_preserve_episode_and_alignment_never_resets_credit(tmp_path,monkeypatch):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch);r,behavior=bundle[:2]
    complete(bundle,initial(bundle),0)
    assert arm(r)['ok'];complete(bundle,step(r),1)
    prior=copy.deepcopy(r._marvin_live_proof_continuation.previous_selection)
    assert r._marvin_live_proof_continuation.detour_context.phase.value=='PASS_OBSTACLE'
    assert r._marvin_live_proof_continuation.detour_context.phase_action_count==1
    assert r._marvin_live_proof_continuation.detour_context.committed_side=='LEFT'
    assert prior['first_post_action_bypass_progress']['meaningful_progress'] is False
    assert not {'options','local_bypass','bypass_handoff'} & prior.keys()
    behavior.specs=iter([(100,1.1),(0,1.1)])
    assert arm(r)['ok'];aligned=step(r);complete(bundle,aligned,2)
    assert aligned['controller_result']['history'][0]['state']=='ALIGNING'
    assert r._marvin_live_proof_continuation.previous_selection==prior
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']==2
    behavior.specs=iter([(0,1.1)]*12)
    assert arm(r)['ok'];second=step(r);complete(bundle,second,3)
    chosen=second['controller_result']['history'][0]['result']['local_detour']
    assert chosen['action_type']=='BYPASS_FORWARD' and chosen['phase']=='PASS_OBSTACLE'
    assert r._marvin_live_proof_continuation.detour_context.phase_action_count==2
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']==3
    assert arm(r)['ok'];third=step(r);complete(bundle,third,4)
    assert r._marvin_live_proof_continuation.detour_context.phase_action_count==3
    assert r._marvin_live_proof_continuation.avoidance['local_avoidance_actions']==4


def test_second_step_final_JIT_veto_waits_without_transport_or_extra_budget(tmp_path,monkeypatch):
    bundle,_,_=episode_bundle(tmp_path,monkeypatch);runtime,behavior=bundle[:2]
    original=behavior.execute_single_marvin_approach_step;read=runtime.world_model.get_lidar_obstacles;hazard=[False];attempt=[0]
    def scan(**kwargs):
        state=read(**kwargs)
        if hazard[0]:state['local_motion_geometry']['points'].append({'x_m':.17,'y_m':.40})
        return state
    def approach(**kwargs):
        attempt[0]+=1
        if attempt[0]==2:hazard[0]=True
        return original(**kwargs)
    runtime.world_model.get_lidar_obstacles=scan;behavior.execute_single_marvin_approach_step=approach
    result=run(runtime)
    assert result['reason']=='find_marvin_blocked_wait_exhausted'
    assert result['blocked_wait_reason']=='marvin_local_bypass_jit_veto'
    assert result['local_avoidance_actions']==2 and result['local_bypass_actions']==1
    assert motions(bundle[3])==[('strafe',.08,1.),('forward',.1,.5)]
    veto=result['history'][2]['result']
    assert veto['source_stamp_consumed'] and not veto['motion_executed']
    assert veto['pre_transport_jit_veto']['transport_attempted'] is False
    assert result['blocked_wait_recheck_count']==12


def test_episode_checkpoint_contains_history_not_fresh_permission():
    _,old=neutral_case();retained=_marvin_proof_selection_history(old)
    assert retained['bypass_episode']==old['bypass_episode']
    assert retained['bypass_step_completion']==old['bypass_step_completion']
    assert retained['bypass_step_outcome']==old['bypass_step_outcome']
    assert not {'options','local_bypass','bypass_episode_diagnostics'} & retained.keys()


@pytest.mark.parametrize('reason', ['fresh_corridor_unsafe', 'jit_veto', 'stop_failure',
    'delivery_uncertain', 'producer_session_changed', 'avoidance_budget_exhausted'])
def test_closed_episode_cannot_reopen_on_identical_clear_geometry(reason):
    state, old = neutral_case()
    ended = end_bypass_episode(old, reason)
    selected = select(state, ended)
    assert selected['action_type'] != 'BYPASS_FORWARD'
    assert 'BYPASS_FORWARD' in selected['ineffective_action_types']
    assert selected['bypass_episode']['active'] is False


def test_episode_side_change_cannot_use_legacy_reversal_escape():
    state, old = neutral_case()
    opposite, _ = scene_plan([(.65,.25),(0.,.65),(0.,1.2),(0.,-1.2)])
    opposite['acquisition_sequence'] = state['acquisition_sequence'] + 1
    selected = select(opposite, old)
    assert not selected['bypass_side_change_allowed']
    assert selected['action_type'] not in {'BYPASS_FORWARD', 'STRAFE_RIGHT'}
    assert selected['bypass_episode']['active'] is False


def test_route_clear_explicitly_ends_episode_in_shared_selector():
    state, old = neutral_case()
    clear, _ = scene_plan([])
    clear['acquisition_sequence'] = state['acquisition_sequence'] + 1
    selected = select(clear, old)
    assert not selected['route']['route_to_marvin_obstructed']
    assert selected['bypass_episode_active'] is False
    assert selected['bypass_episode_reason'] == 'direct_route_clear'


def test_noninteger_action_outcome_stamp_fails_closed():
    state, old = neutral_case()
    old['bypass_step_outcome']['action_acquisition_sequence'] = float(old['acquisition_sequence'])
    assert select(state, old)['action_type'] != 'BYPASS_FORWARD'
