"""Offline local bypass geometry and complete guarded mission handoffs."""
import copy
import json
import math
from pathlib import Path
import socket
import time

import pytest

from marvin_local_bypass import (
    plan_local_bypass, evaluate_avoidance_progress, LOCAL_BYPASS_HORIZON_M,
    BYPASS_FORWARD_SPEED_MPS, BYPASS_FORWARD_SECONDS,
)
from marvin_local_obstacle_avoidance import select_marvin_escape_action, MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import evaluate_marvin_route
from test_marvin_avoidance_planner_progress import scan
from test_marvin_lateral_avoidance import strafe_runtime
from test_find_marvin_closed_loop import run, motions


@pytest.fixture(autouse=True)
def no_live_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Bypass validation must not access robot/services')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


ASSOCIATION = {'verified_marvin_distance_m': 1.2, 'verified_marvin_conservative_distance_m': 1.1, 'target_bearing_degrees': 0.}
OPEN_LEFT = [(.65, -.25), (0., 1.2), (0., -.65)]


def scene_plan(points=OPEN_LEFT, *, previous=None, association=None):
    state = scan(points)
    names = ('front','front_left','left','rear_left','rear','rear_right','right','front_right')
    for point in state['local_motion_geometry']['points']:
        x,y=point['x_m'],point['y_m']
        sector=state['local_motion_geometry']['sectors'][names[int((math.degrees(math.atan2(y,x))+22.5)%360//45)]]
        sector['minimum_distance_from_base_m']=min(sector['minimum_distance_from_base_m'],math.hypot(x,y))
    return state, select_marvin_escape_action(state, association or ASSOCIATION,
        expected_session='test', allow_strafe=True, previous_selection=previous)


def bypass_plan():
    _, first = scene_plan()
    state, second = scene_plan(previous=first)
    assert first['action_type'] == 'STRAFE_LEFT'
    assert second['action_type'] == 'BYPASS_FORWARD'
    return state, second


def test_useful_lateral_progress_retains_strafe_then_no_progress_selects_bypass():
    state, plan = bypass_plan()
    assert plan['ineffective_action_types'] == ['STRAFE_LEFT']
    assert plan['route']['route_to_marvin_obstructed']
    bypass = plan['local_bypass']
    assert bypass['local_bypass_side'] == 'LEFT'
    assert bypass['bypass_forward_permitted'] and not bypass['route_to_bypass_obstructed']
    assert bypass['bypass_target_x_m'] == LOCAL_BYPASS_HORIZON_M == .15
    assert bypass['bypass_target_y_m'] == bypass['bypass_bearing_deg'] == 0
    assert bypass['protected_radius_m'] == .45
    assert plan['options']['BYPASS_FORWARD']['requested_duration'] == .50
    assert plan['options']['BYPASS_FORWARD']['route_progress']['meaningful_progress']
    assert BYPASS_FORWARD_SPEED_MPS == .10 and BYPASS_FORWARD_SECONDS == .50


@pytest.mark.parametrize('point', [(.18, 0.), (.40, .2), (-.1, .40), (.17, -.40)])
def test_full_bypass_capsule_includes_start_endpoint_and_side_corners(point):
    state, plan = bypass_plan()
    state['local_motion_geometry']['points'].append({'x_m': point[0], 'y_m': point[1]})
    bypass = plan_local_bypass(state, ASSOCIATION, plan['route'], expected_session='test', side='LEFT')
    assert not bypass['bypass_forward_permitted']
    assert bypass['route_to_bypass_obstructed'] and bypass['bypass_corridor_occupancy'] > 0


@pytest.mark.parametrize('fault', ['stale', 'invalid', 'session', 'coverage', 'nan', 'unverified', 'standoff', 'sequence'])
def test_sensor_identity_depth_and_coverage_failures_cannot_create_bypass(fault):
    state, plan = bypass_plan()
    association = dict(ASSOCIATION)
    if fault == 'stale': state['received_monotonic_seconds'] -= .301
    if fault == 'invalid': state['valid'] = False
    if fault == 'session': state['producer_session'] = 'other'
    if fault == 'coverage': state['local_motion_geometry']['sectors']['rear']['valid_sample_count'] = 0
    if fault == 'nan': state['local_motion_geometry']['points'].append({'x_m': float('nan'), 'y_m': 0.})
    if fault == 'unverified': association['verified_marvin_distance_m'] = None
    if fault == 'standoff': association['verified_marvin_distance_m'] = .49
    if fault == 'sequence': state['acquisition_sequence'] = True
    assert not plan_local_bypass(state, association, plan['route'], expected_session='test', side='LEFT')['bypass_forward_permitted']


def test_bypass_never_heads_through_blocker_or_invents_initial_side_clearance():
    for point in ((.55, 0.), (.55, -.10), (.48, -.17)):
        state = scan([point])
        route = evaluate_marvin_route(state, ASSOCIATION, expected_session='test')
        bypass = plan_local_bypass(state, ASSOCIATION, route, expected_session='test', side='LEFT')
        assert not bypass['bypass_forward_permitted']


def test_measured_longitudinal_progress_allows_one_more_fresh_bypass_not_radial_change():
    _, old = bypass_plan()
    state, next_plan = scene_plan([(.60, -.25), (0., 1.2), (0., -.65)], previous=old)
    assert next_plan['action_type'] == 'BYPASS_FORWARD'
    assert next_plan['meaningful_progress_reason'] == 'bypass_longitudinal_passage_improved'
    assert next_plan['actual_route_progress']['bypass_longitudinal_progress_m'] == pytest.approx(.05)
    _, stationary = scene_plan(previous=old)
    assert stationary['action_type'] == 'STRAFE_LEFT'
    assert stationary['post_bypass_lateral_recovery_selected']
    assert 'BYPASS_FORWARD' in stationary['ineffective_action_types']
    _, failed_recovery = scene_plan(previous=stationary)
    assert failed_recovery['action_type'] is None
    assert {'BYPASS_FORWARD', 'STRAFE_LEFT'} <= set(failed_recovery['ineffective_action_types'])
    route = dict(old['route'], blocking_obstacle_distance_m=1.0)
    assert not evaluate_avoidance_progress(old, route, old['local_bypass'])['meaningful_progress']


def test_bypass_side_memory_does_not_flip_for_clearance_flicker():
    _, old = bypass_plan()
    state = scan([(.6, -.25)])
    state['local_motion_geometry']['sectors']['right']['minimum_distance_from_base_m'] = 3.
    state['local_motion_geometry']['sectors']['left']['minimum_distance_from_base_m'] = 1.
    plan = select_marvin_escape_action(state, ASSOCIATION, expected_session='test', allow_strafe=True, previous_selection=old)
    assert plan['action_type'] == 'BYPASS_FORWARD' and plan['direction'] == 'LEFT'


def test_clear_direct_route_abandons_local_target():
    _, old = bypass_plan()
    _, plan = scene_plan([(2., -.5)], previous=old)
    assert not plan['route']['route_to_marvin_obstructed']
    assert plan['action_type'] is None and 'local_bypass' not in plan


def pursuit_specs(prefix):
    return prefix + [(0., round(x / 100, 2)) for x in range(105, 49, -5)]


def mission_bundle(tmp_path, monkeypatch, *, bypass_steps=1):
    scenes = [OPEN_LEFT, OPEN_LEFT]
    for step in range(1, bypass_steps):
        scenes.append([(.65-.05*step, -.25), (0., 1.2), (0., -.65)])
    scenes.append(None)
    specs = pursuit_specs([(0., 1.1)] * (bypass_steps+1))
    bundle, flags, client = strafe_runtime(tmp_path, monkeypatch, specs, scenes)
    runtime, _, _, _, clock = bundle
    read = runtime.world_model.get_lidar_obstacles
    def timestamped_read(**kwargs):
        clock[0] += 1000  # Independent mocked producer receipt after STOP, not equal to STOP.
        return read(**kwargs)
    runtime.world_model.get_lidar_obstacles = timestamped_read
    return bundle, flags, client


@pytest.mark.parametrize('steps', [1, 2, 3])
def test_guarded_bypass_forward_while_marvin_route_blocked_then_resume_pursuit(tmp_path, monkeypatch, steps):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch, bypass_steps=steps)
    runtime, behavior, robot, events, _ = bundle
    result = run(runtime)
    assert result['state'] == 'ARRIVED', result['reason']
    assert result['arrived_at_marvin']
    assert result['local_avoidance_actions'] == 1 + steps
    assert result['local_bypass_actions'] == result['completed_bypass_forward_actions'] == steps
    assert result['local_bypass_active'] is False
    assert result['local_bypass_target_x_m'] is None
    assert motions(events)[0] == ('strafe', .08, 1.)
    history = [row for row in result['history'] if not row.get('decision_only')]
    stamps = [row['source_frame_stamp_ns'] for row in history]
    assert len(stamps) == len(set(stamps))
    bypass_rows = [row for row in history if row['result'].get('action_type') == 'BYPASS_FORWARD']
    for row in bypass_rows:
        assert row['state'] == 'AVOIDING'
        assert row['observation']['arrival']['route_to_marvin_obstructed']
        assert row['result']['source_stamp_consumed'] and row['result']['bridge_stop_confirmed']
        assert row['result']['approach_result']['forward_safety']['permitted']
        assert row['result']['local_detour']['accepted']
        assert row['result']['local_detour']['local_bypass']['bypass_forward_permitted']
    for row in history:
        assert row['source_frame_stamp_ns'] in runtime._marvin_alignment_consumed_source_frame_stamps
        sequence = row['action_lidar_evidence'][1]
        assert any(wait['snapshot']['acquisition_sequence'] > sequence for wait in result['lidar_wait_history'])
    diagnostics = [row for row in result['progress_diagnostics']['actions'] if row['action_type'] == 'BYPASS_FORWARD']
    assert len(diagnostics) == steps
    for row in diagnostics:
        assert row['type'] == 'bypass_forward'
        assert row['command']['linear_x'] == .1
        assert row['command']['linear_y'] == row['command']['angular_z'] == 0
        assert row['command']['duration'] == .5
        assert row['first_post_action_lidar'] and row['first_new_post_action_camera']
    assert all(action[1:] == (.1, .5) for action in motions(events)[1:])
    assert robot.status()['motion'] == {'linear_x': 0., 'linear_y': 0., 'angular_z': 0., 'streaming': False}


def test_neutral_bypass_retains_pass_until_six_action_guard(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0.,1.1)]*16, [OPEN_LEFT])
    result = run(bundle[0])
    assert motions(bundle[3]) == [('strafe', .08, 1.)] + [('forward', .1, .5)]*5
    assert result['state'] == 'BLOCKED' and result['reason'] == 'find_marvin_local_avoidance_exhausted'
    assert result['blocked_wait_recheck_count'] == 0
    assert result['local_avoidance_actions'] == 6
    assert result['local_bypass_actions'] == 5
    assert result['stop_result']['ok']


def test_bypass_actions_share_unchanged_six_action_budget(tmp_path, monkeypatch):
    scenes = [OPEN_LEFT, OPEN_LEFT] + [[(.65-.011*i, -.25), (0.,1.2), (0.,-.65)] for i in range(1,8)]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0.,1.1)]*12, scenes)
    result = run(bundle[0])
    assert result['reason'] == 'find_marvin_local_avoidance_exhausted'
    assert MAX_LOCAL_AVOIDANCE_ACTIONS == result['local_avoidance_actions'] == 6
    assert result['local_bypass_actions'] == 5
    assert len(motions(bundle[3])) == 6


@pytest.mark.parametrize('phase', ['dispatch', 'post'])
def test_stop_during_bypass_preempts_without_second_motion(tmp_path, monkeypatch, phase):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch)
    runtime, _, robot, events, _ = bundle
    if phase == 'post':
        robot.on_motion = lambda: runtime.submit_intent({'intent': 'STOP', 'speech': 'Stop.'}) if len(motions(events)) == 2 else None
    else:
        original = runtime.behavior_manager.execute_single_marvin_approach_step
        def stop_before_dispatch(**kwargs):
            if kwargs.get('local_selection_validator'):
                runtime.submit_intent({'intent': 'STOP', 'speech': 'Stop.'})
            return original(**kwargs)
        runtime.behavior_manager.execute_single_marvin_approach_step = stop_before_dispatch
    result = run(runtime)
    assert result['state'] == 'STOPPED'
    assert len(motions(events)) == (2 if phase == 'post' else 1)
    assert robot.status()['motion']['streaming'] is False


def test_duplicate_camera_after_bypass_cannot_authorize_second_action(tmp_path, monkeypatch):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch)
    runtime, behavior, robot, events, _ = bundle
    robot.on_motion = lambda: setattr(behavior, 'repeat_camera', True) if len(motions(events)) == 2 else None
    result = run(runtime)
    assert result['state'] == 'REVERIFY_REQUIRED'
    assert len(motions(events)) == 2


def test_clear_path_never_enters_bypass(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.6),(0,.5)], [None])
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED'
    assert result['local_bypass_actions'] == result['local_avoidance_actions'] == 0
    assert result['local_avoidance_history'] == []
    assert motions(bundle[3]) == [('forward', .1, .5)]


def test_retained_a6c4d675_geometry_transitions_to_bypass_without_forcing_route_clear():
    fixture = json.loads((Path(__file__).parent / 'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    assert fixture['mission_id'] == 'mission-a6c4d675'
    for row in fixture['cycles']:
        state = copy.deepcopy(row['lidar'])
        state.update(received_monotonic_seconds=time.monotonic(), age_at_receipt_seconds=0.)
        plan = select_marvin_escape_action(state, row['association'], expected_session=state['producer_session'],
            allow_strafe=True, previous_selection=row['previous_selection'])
        assert plan['route']['route_to_marvin_obstructed']
        if row['cycle'] < 6:
            assert plan['action_type'] == 'STRAFE_LEFT'
        else:
            assert plan['action_type'] == 'BYPASS_FORWARD'
            assert not plan['local_bypass']['route_to_bypass_obstructed']
            assert plan['local_bypass']['bypass_corridor_occupancy'] == 0
            assert plan['route']['route_occupancy'] == 20
            assert plan['route']['corridor_overlap_m'] == pytest.approx(.2733190035)
            assert plan['options']['BYPASS_FORWARD']['predicted_route']['route_to_marvin_obstructed']
            # Five strafes plus this bypass consume six actions. Reaching Marvin
            # is not demonstrated by this saved terminal geometry alone.
            assert MAX_LOCAL_AVOIDANCE_ACTIONS == 6


@pytest.mark.parametrize('direction', ['LEFT', 'RIGHT'])
def test_bypass_can_be_established_on_either_open_side(direction):
    sign = 1 if direction == 'LEFT' else -1
    points = [(x, y * sign) for x, y in OPEN_LEFT]
    _, first = scene_plan(points)
    _, plan = scene_plan(points, previous=first)
    assert plan['action_type'] == 'BYPASS_FORWARD' and plan['direction'] == direction


def test_post_bypass_marvin_shift_uses_ordinary_alignment(tmp_path, monkeypatch):
    specs = pursuit_specs([(0.,1.1),(0.,1.1),(100.,1.1)])
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, specs, [OPEN_LEFT,OPEN_LEFT,None])
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED'
    assert motions(bundle[3])[:3] == [('strafe',.08,1.),('forward',.1,.5),('turn','RIGHT',.25,.5)]
    assert result['history'][2]['state'] == 'ALIGNING'
    assert result['local_avoidance_actions'] == 2


@pytest.mark.parametrize('still_blocked', [False, True])
def test_tracker_loss_after_bypass_uses_existing_gemini_recovery(tmp_path, monkeypatch, still_blocked):
    from test_find_marvin_reacquisition import recovery_runtime
    def factory(path, mp, specs):
        return recovery_runtime(path, mp, specs, failures=(6,))
    specs = pursuit_specs([(0.,1.1)]*5)
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, specs,
        [OPEN_LEFT,OPEN_LEFT,[(.60,-.25),(0.,1.2),(0.,-.65)] if still_blocked else None,None], factory=factory)
    result = run(bundle[0])
    assert result['state'] == 'ARRIVED', result['reason']
    assert result['reacquisition_attempts'] >= 1
    assert result['completed_bypass_forward_actions'] == (2 if still_blocked else 1)
    assert 'gemini' in bundle[3] and 'reacquire' in bundle[3]
    assert not result['local_bypass_active']


def test_fresh_jit_geometry_can_veto_previously_clear_bypass(tmp_path, monkeypatch):
    bundle, _, _ = mission_bundle(tmp_path, monkeypatch)
    runtime, _, _, events, _ = bundle
    primitive = runtime.behavior_manager.execute_single_marvin_approach_step
    read = runtime.world_model.get_lidar_obstacles
    def unsafe_after_selection(**kwargs):
        if kwargs.get('local_selection_validator'):
            def unsafe(**kw):
                state = read(**kw)
                state['local_motion_geometry']['points'].append({'x_m':.40,'y_m':0.})
                return state
            runtime.world_model.get_lidar_obstacles = unsafe
        return primitive(**kwargs)
    runtime.behavior_manager.execute_single_marvin_approach_step = unsafe_after_selection
    result = run(runtime)
    assert result['state'] == 'BLOCKED'
    assert motions(events) == [('strafe', .08, 1.)]
    assert result['history'][-1]['result']['source_stamp_consumed']
    assert not result['history'][-1]['result']['full_step_completed']


@pytest.mark.parametrize('recover', [True, False])
def test_lidar_expiry_during_bypass_stops_and_uses_existing_bounded_recovery(tmp_path, monkeypatch, recover):
    bundle, flags, client = mission_bundle(tmp_path, monkeypatch)
    runtime, behavior, robot, events, clock = bundle
    flags['publish'] = recover
    lateral_transport = client._request
    def transport(method, path, payload=None):
        if not payload or payload['linear_x'] == 0:
            return lateral_transport(method,path,payload)
        assert payload['linear_x'] == .1 and payload['angular_z'] == 0 and payload.get('linear_y',0) == 0
        events.append(('forward',payload['linear_x'],payload['duration']))
        clock[0] += 120_000_000
        behavior.freeze_lidar = True
        behavior.sequence += 1
        flags['outage'] = True
        assert client.forward_interlock.refresh() == (False,'stale')
        assert events[-1] == 'stop'
        return dict(ok=True,action='motion',mode='bounded',automatic_stop=True,returned_immediately=False,**payload)
    client._request = transport
    def forward(*, speed, seconds, dispatch_guard=None):
        return client.move_forward(speed=speed,seconds=seconds,dispatch_guard=dispatch_guard)
    robot.move_forward = forward
    # Restore the ordinary offline forward transport after the interrupted bypass;
    # new perception makes the direct route clear, never resume the old command.
    original_forward = type(robot).move_forward
    flags['on_sleep'] = lambda: setattr(robot,'move_forward',lambda **kw: original_forward(robot,**kw)) if flags['sleeps'] == 2 else None
    result = run(runtime)
    interrupted = result['history'][1]['result']
    assert interrupted['interrupted'] and interrupted['source_stamp_consumed']
    assert not interrupted['full_step_completed']
    assert result['interrupted_bypass_forward_attempts'] == 1
    assert result['completed_bypass_forward_actions'] == 0
    assert result['local_avoidance_actions'] == 2  # Dispatched interruption consumes the shared budget.
    assert result['lidar_recovery_history']
    if recover:
        assert result['state'] == 'ARRIVED', result['reason']
        assert result['history'][2]['source_frame_stamp_ns'] > result['history'][1]['source_frame_stamp_ns']
    else:
        assert result['state'] == 'BLOCKED' and result['reason'] == 'find_marvin_new_lidar_evidence_timeout'
        assert len(motions(events)) == 2
    assert robot.status()['motion']['streaming'] is False


@pytest.mark.parametrize('fault', ['stale', 'invalid', 'session', 'camera', 'identity'])
def test_current_sensor_or_identity_fault_never_dispatches_bypass(tmp_path, monkeypatch, fault):
    bundle, _, _ = mission_bundle(tmp_path,monkeypatch)
    runtime, _, _, events, clock = bundle
    primitive = runtime.behavior_manager.execute_single_marvin_approach_step
    read = runtime.world_model.get_lidar_obstacles
    def fault_after_admission(**kwargs):
        if kwargs.get('local_selection_validator'):
            if fault == 'camera':
                clock[0] += 1_000_000_001
            elif fault == 'identity':
                runtime._marvin_alignment_observation = None
            else:
                def faulty(**kw):
                    state = read(**kw)
                    if fault == 'stale': state['received_monotonic_seconds'] -= .301
                    if fault == 'invalid': state['valid'] = False
                    if fault == 'session': state['producer_session'] = 'wrong'
                    return state
                runtime.world_model.get_lidar_obstacles = faulty
        return primitive(**kwargs)
    runtime.behavior_manager.execute_single_marvin_approach_step = fault_after_admission
    result = run(runtime)
    assert result['state'] in {'BLOCKED','STOPPED'}
    assert motions(events) == [('strafe',.08,1.)]
    assert result['local_bypass_actions'] == 0
    assert result['history'][-1]['result']['source_stamp_consumed']


def test_forward_client_propagates_final_guard_without_altering_old_schema(monkeypatch):
    from test_robot_bridge_lateral_client import client_with_transport
    client,requests,interlock = client_with_transport(monkeypatch)
    calls=[]
    result=client.move_forward(speed=.1,seconds=.5,dispatch_guard=lambda: calls.append('guard') or True)
    assert result['ok'] and calls==['guard']
    assert requests == [('POST','/motion',{'linear_x':.1,'angular_z':0.,'duration':.5})]
    requests.clear()
    result=client.move_forward(speed=.1,seconds=.5,dispatch_guard=lambda: False)
    assert not result['ok'] and requests==[] and not interlock.pending


def test_world_model_reports_ephemeral_bypass_without_changing_mission_target(tmp_path,monkeypatch):
    bundle,_,_=mission_bundle(tmp_path,monkeypatch)
    runtime,_,_,_,_=bundle
    published=[]
    publish=runtime._publish_behavior_tracking
    def record(state):
        publish(state)
        published.append(dict(runtime.tracking_state))
    runtime._publish_behavior_tracking=record
    result=run(runtime)
    states=[s for s in published if s.get('local_bypass_active')]
    assert states
    assert all(s['target_label']=='marvin' and s['local_bypass_side']=='LEFT' for s in states)
    assert all(s['bypass_distance_m']==.15 and s['bypass_bearing_deg']==0 for s in states)
    assert all(s['route_to_marvin_obstructed'] and not s['route_to_bypass_obstructed'] for s in states)
    assert result['local_bypass_target_x_m'] is None
    assert result['target']=='marvin'


@pytest.mark.parametrize('conservative', [None, .49, .53])
def test_bypass_requires_verified_conservative_standoff_margin(conservative):
    state, plan = bypass_plan()
    association = dict(ASSOCIATION, verified_marvin_conservative_distance_m=conservative)
    bypass = plan_local_bypass(state,association,plan['route'],expected_session='test',side='LEFT')
    assert not bypass['bypass_forward_permitted']
    assert bypass['local_bypass_reason']=='bypass_verified_standoff_margin_insufficient'


def test_bypass_progress_is_not_a_changed_selected_nearest_return_alone():
    _,old=bypass_plan()
    route=dict(old['route'],blocking_obstacle_x_m=.2,blocking_obstacle_distance_m=.7)
    # The full blocking cloud's longitudinal extent remains stationary.
    assert not evaluate_avoidance_progress(old,route,old['local_bypass'])['meaningful_progress']


def test_side_change_requires_old_bypass_unusable_and_new_measurably_better_side():
    _,old=bypass_plan()
    # Legacy history has no pending bounded episode; retain the existing
    # independently measured reversal policy for that state.
    old.pop('bypass_episode')
    points=[(.65,.25),(0.,.65),(0.,1.2),(0.,-1.2)]
    _,changed=scene_plan(points,previous=old)
    assert changed['bypass_side_change_allowed']
    assert changed['direction']=='RIGHT' and changed['action_type']=='STRAFE_RIGHT'
    # Neither current safety nor a clearly changed side alone can manufacture
    # the required measured alternative-clearance improvement.
    old['right_clearance_m']=1.2
    _,blocked=scene_plan(points,previous=old)
    assert not blocked['bypass_side_change_allowed']
    assert blocked['action_type'] is None


def test_later_alignment_cannot_manufacture_progress_for_ineffective_bypass():
    state,old=bypass_plan()
    unchanged=evaluate_avoidance_progress(old,old['route'],old['local_bypass'])
    assert not unchanged['meaningful_progress']
    old['first_post_action_bypass_progress']=unchanged
    # A LEFT in-place turn alone reduces longitudinal base-frame x even though
    # both the obstacle and SAME Marvin ray remain fixed in the world.
    rotated=evaluate_marvin_route(state,ASSOCIATION,expected_session='test',heading_change=.125)
    assert old['route']['blocking_obstacle_longitudinal_extent_m']-rotated['blocking_obstacle_longitudinal_extent_m']>.01
    assert rotated['corridor_overlap_m']==pytest.approx(old['route']['corridor_overlap_m'])
    progress=evaluate_avoidance_progress(old,rotated,old['local_bypass'])
    assert progress==unchanged and not progress['meaningful_progress']


def test_successful_bypass_actual_outcome_is_retained_before_alignment(tmp_path,monkeypatch):
    bundle,_,_=mission_bundle(tmp_path,monkeypatch,bypass_steps=2)
    result=run(bundle[0])
    assert result['state']=='ARRIVED'
    row=result['history'][1]
    first=row['result']['local_detour']['first_post_action_bypass_progress']
    assert first['meaningful_progress']
    assert first['meaningful_progress_reason']=='bypass_longitudinal_passage_improved'
    assert first['bypass_longitudinal_progress_m']==pytest.approx(.05)
    assert result['local_avoidance_history'][1]['actual_route_progress']==first


def test_useful_bypass_can_transition_back_to_fresh_strafe_when_next_corridor_blocks(tmp_path,monkeypatch):
    def scene(x):return [(x,-.35),(0.,1.2),(0.,-.65)]
    scenes=[scene(.55),scene(.55),scene(.50),scene(.45),scene(.40),None]
    specs=pursuit_specs([(0.,1.1)]*5)
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,specs,scenes)
    result=run(bundle[0])
    assert result['state']=='ARRIVED',result['reason']
    assert [row['action_type'] for row in result['local_avoidance_history']] == [
        'STRAFE_LEFT','BYPASS_FORWARD','BYPASS_FORWARD','BYPASS_FORWARD','STRAFE_LEFT']
    assert result['local_avoidance_actions']==5 and result['local_bypass_actions']==3
    third_bypass=result['local_avoidance_history'][3]
    assert third_bypass['meaningful_progress']
    assert third_bypass['meaningful_progress_reason']=='bypass_longitudinal_passage_improved'
    assert third_bypass['post_action_bypass']['route_to_bypass_obstructed']
    assert not third_bypass['post_action_bypass']['bypass_forward_permitted']
    next_selection=result['local_avoidance_history'][4]['selection']
    assert not next_selection['local_bypass']['bypass_forward_permitted']
    assert next_selection['options']['STRAFE_LEFT']['hard_safety_permitted']
    assert next_selection['options']['STRAFE_LEFT']['improves_route']
    assert motions(bundle[3])[:5]==[
        ('strafe',.08,1.),('forward',.1,.5),('forward',.1,.5),('forward',.1,.5),('strafe',.08,1.)]
    assert all(row['result']['source_stamp_consumed'] for row in result['history'])
    assert len({row['source_frame_stamp_ns'] for row in result['history']})==len(result['history'])


def test_current_open_bypass_side_is_geometric_not_selected_by_unrelated_rear_clearance():
    points=OPEN_LEFT+[(-.08,.445)]
    _,plan=scene_plan(points)
    assert plan['left_clearance_m']<plan['right_clearance_m']
    assert plan['action_type']=='BYPASS_FORWARD' and plan['direction']=='LEFT'
    assert not plan['options']['STRAFE_LEFT']['hard_safety_permitted']
    assert not plan['options']['STRAFE_RIGHT']['improves_route']
    assert plan['local_bypass']['bypass_forward_permitted']
    assert plan['local_bypass']['bypass_corridor_occupancy']==0


def test_already_established_side_clearance_can_start_bypass_without_forcing_a_strafe(tmp_path,monkeypatch):
    points=OPEN_LEFT+[(-.08,.445)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,pursuit_specs([(0.,1.1)]),[points,None])
    result=run(bundle[0])
    assert result['state']=='ARRIVED',result['reason']
    assert result['local_avoidance_actions']==result['local_bypass_actions']==1
    assert result['completed_strafe_actions']==0
    assert result['history'][0]['result']['action_type']=='BYPASS_FORWARD'
    assert motions(bundle[3])[0]==('forward',.1,.5)
    assert result['history'][0]['result']['source_stamp_consumed']
    assert result['history'][0]['result']['bridge_stop_confirmed']
