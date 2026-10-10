"""Offline production client/interlock, runtime ownership and sensor handoffs."""
import math
import socket
import time
import pytest
from robot_bridge.client import RobotBridgeClient
from robot_bridge.forward_interlock import ForwardMotionInterlock
from local_motion_safety_envelope import evaluate_local_motion_safety
from marvin_local_obstacle_avoidance import select_marvin_escape_action
from marvin_route_obstruction import evaluate_marvin_route, route_progress
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_find_marvin_reacquisition import recovery_runtime


@pytest.fixture(autouse=True)
def no_robot_access(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No live access in lateral tests')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)


def strafe_runtime(tmp_path, monkeypatch, specs, scenes, factory=make_runtime, interruptions=()):
    bundle = factory(tmp_path, monkeypatch, specs)
    runtime, behavior, robot, events, clock = bundle
    original = runtime.world_model.get_lidar_obstacles
    flags = {'outage': False, 'sleeps': 0, 'publish': True, 'on_sleep': None, 'on_dispatch': None}
    def telemetry(**kwargs):
        scan = original(**kwargs)
        points = scan['local_motion_geometry']['points']
        scene = scenes[min(len(motions(events)), len(scenes)-1)]
        if scene:
            for x, y in scene:
                points.append({'x_m': x, 'y_m': y, 'distance_m': math.hypot(x,y),
                    'robot_bearing_deg': math.degrees(math.atan2(y,x))})
        # Rebuild sector minima while preserving independently supplied full scan coverage.
        for sector in scan['local_motion_geometry']['sectors'].values():
            sector['minimum_distance_from_base_m'] = 1.5
        names = ('front','front_left','left','rear_left','rear','rear_right','right','front_right')
        for p in points:
            i = int((math.degrees(math.atan2(p['y_m'],p['x_m'])) + 22.5) % 360 // 45)
            sector = scan['local_motion_geometry']['sectors'][names[i]]
            sector['minimum_distance_from_base_m'] = min(sector['minimum_distance_from_base_m'],math.hypot(p['x_m'],p['y_m']))
        scan['sectors'] = {'front': {'available': True,'state': 'CLEAR'}}
        if flags['outage']:
            scan.update(available=False,valid=False,reason='stale',effective_age_seconds=.342)
        return scan
    runtime.world_model.get_lidar_obstacles = telemetry
    status = robot.status
    robot.status = lambda: dict(status(), motion=dict(status()['motion'],linear_y=0.),
        motion_capabilities={'linear_y':True,'max_linear_y':.1})
    client = RobotBridgeClient(base_url='http://offline.invalid')
    client.configure_forward_interlock(ForwardMotionInterlock(telemetry,
        expected_session=runtime.lidar_worker.session, stop_callback=robot.stop))
    robot.forward_interlock = client.forward_interlock
    plan = iter(interruptions)
    def request(method, path, payload=None):
        if method == 'GET' and path == '/status':return robot.status()
        assert path == '/motion' and method == 'POST'
        assert payload['linear_x'] == payload['angular_z'] == 0
        assert abs(payload['linear_y']) == .08 and payload['duration'] <= 1.0
        events.append(('strafe',payload['linear_y'],payload['duration']))
        interrupted = next(plan, False)
        clock[0] += 120_000_000 if interrupted else int(payload['duration'] * 1_000_000_000)
        if interrupted:
            flags['outage'] = True
            behavior.freeze_lidar = True
            behavior.sequence += 1
            client.forward_interlock.refresh()
        if flags['on_dispatch']:flags['on_dispatch']()
        if robot.on_motion:robot.on_motion()
        return dict(ok=True,action='motion',mode='bounded',automatic_stop=True,returned_immediately=False,**payload)
    client._request = request
    robot.move_lateral = client.move_lateral
    original_sleep = time.sleep
    def sleep(seconds):
        original_sleep(seconds)
        if flags['outage']:
            assert robot.status()['motion']['linear_y'] == 0
            events.append('stopped_lateral_wait');flags['sleeps'] += 1
            if flags['on_sleep']:flags['on_sleep']()
            if flags['publish'] and flags['sleeps'] == 2:
                flags['outage'] = False;behavior.freeze_lidar = False;behavior.sequence += 1
    monkeypatch.setattr('runtime.time.sleep',sleep)
    return bundle, flags, client


LEFT_OPEN = [(.55,0.),(.551,.003),(.551,-.003),(0.,1.2),(0.,-.48)]
RIGHT_OPEN = [(.55,0.),(.551,.003),(.551,-.003),(0.,.48),(0.,-1.2)]


@pytest.mark.parametrize('scene,y',[(LEFT_OPEN,.08),(RIGHT_OPEN,-.08)])
def test_live_055_foreground_routes_to_one_guarded_strafe_then_forward(tmp_path,monkeypatch,scene,y):
    bundle,_,_ = strafe_runtime(tmp_path,monkeypatch,[(0,.60),(0,.60),(0,.5)],[scene,None])
    runtime,behavior,robot,events,_ = bundle
    result = run(runtime)
    assert result['state']=='ARRIVED' and result['arrived_at_marvin']
    first = result['history'][0]
    o = first['observation'];a = o['arrival']
    assert a['candidate_target_return_distance_m']==pytest.approx(.55)
    assert not a['target_range_association_trusted'] and not a['arrived_at_marvin']
    assert not a['direct_path_blocked'] and a['route_to_marvin_obstructed']
    assert motions(events)==[('strafe',y,1.),('forward',.1,.5)]
    assert result['local_avoidance_actions']==1
    selection = result['local_avoidance_history'][0]['selection']
    assert set(selection['options'])=={'STRAFE_LEFT','STRAFE_RIGHT','TURN_LEFT','TURN_RIGHT'}
    assert selection['action_type']==('STRAFE_LEFT' if y>0 else 'STRAFE_RIGHT')
    assert first['result']['source_stamp_consumed'] and first['result']['full_step_completed']
    assert first['source_frame_stamp_ns']<result['history'][1]['source_frame_stamp_ns']
    assert result['lidar_wait_history'][0]['snapshot']['acquisition_sequence']>first['action_lidar_evidence'][1]
    assert events[events.index(('strafe',y,1.))+1]=='stop'
    assert robot.status()['motion']=={'linear_x':0.,'linear_y':0.,'angular_z':0.,'streaming':False}
    diagnostic = result['progress_diagnostics']['actions'][0]
    assert diagnostic['command']['linear_y']==y
    assert diagnostic['type']=='detour_strafe' and result['completed_strafe_actions']==1
    assert result['progress_diagnostics']['action_summary'][0]['nominal_lateral_displacement_m']==pytest.approx(.08)
    assert diagnostic['post_action_avoidance']['progress_improved']


def test_two_strafes_require_new_evidence_and_observed_progress(tmp_path,monkeypatch):
    second = [(x,y-.04) if x>0 else (x,y) for x,y in LEFT_OPEN]
    bundle,_,_ = strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*3+[(0,.5)],[LEFT_OPEN,second,None])
    result=run(bundle[0]);events=bundle[3]
    assert result['state']=='ARRIVED'
    assert motions(events)==[('strafe',.08,1.),('strafe',.08,1.),('forward',.1,.5)]
    assert result['local_avoidance_actions']==2
    assert result['local_avoidance_history'][1]['selection']['progress_improved']
    assert len(set(x['source_frame_stamp_ns'] for x in result['history']))==3


def test_no_progress_cannot_repeat_strafe_indefinitely(tmp_path,monkeypatch):
    bundle,_,_ = strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*9,[LEFT_OPEN])
    result=run(bundle[0]);actions=[e for e in motions(bundle[3]) if e[0]=='strafe']
    assert len(actions)==6
    assert result['reason']=='find_marvin_local_avoidance_exhausted'
    assert result['state']=='BLOCKED' and result['local_avoidance_actions']==6


@pytest.mark.parametrize('point',[(0,.44),(.31,.34),(-.31,.34),(.1,.478)])
def test_side_and_front_rear_endpoint_corner_hazards_block_lateral(tmp_path,monkeypatch,point):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8)],[None])
    state=bundle[0].world_model.get_lidar_obstacles()
    state['local_motion_geometry']['points'].append({'x_m':point[0],'y_m':point[1]})
    safe=evaluate_local_motion_safety(state,expected_session=bundle[0].lidar_worker.session,linear_y=.08,duration=.5,lateral_swept_footprint=True)
    assert not safe['permitted'] and safe['protected_radius_m']==.45


def test_strafe_unsafe_turn_safe_without_real_route_gain_fails_closed(tmp_path,monkeypatch):
    scene=LEFT_OPEN+[(-.02,.465),(0.,-.465)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*4,[scene])
    result=run(bundle[0]);selection=result['local_avoidance_history'][0]['selection']
    assert not selection['options']['STRAFE_LEFT']['permitted']
    assert selection['options']['TURN_LEFT']['hard_safety_permitted']
    assert not selection['options']['TURN_LEFT']['improves_route']
    assert selection['action_type'] is None
    assert result['reason']=='find_marvin_blocked_wait_exhausted'
    assert result['blocked_wait_reason']=='find_marvin_no_safe_local_detour'
    assert motions(bundle[3])==[]


def test_six_action_budget_bounds_repeated_strafes(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*8,[LEFT_OPEN])
    # Force improving, individually safe repeated plans to reach the existing budget.
    original=select_marvin_escape_action
    def progressing(*a,**k):
        k['previous_selection']=None
        return original(*a,**k)
    monkeypatch.setattr('runtime.select_marvin_escape_action',progressing)
    result=run(bundle[0]);assert result['reason']=='find_marvin_local_avoidance_exhausted'
    assert result['local_avoidance_actions']==len(motions(bundle[3]))==6


def test_stale_during_strafe_stops_recovers_new_stamp_and_arrives(tmp_path,monkeypatch):
    bundle,flags,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60),(0,.60),(0,.5)],[LEFT_OPEN,None],interruptions=(True,))
    result=run(bundle[0]);first=result['history'][0]['result']
    assert result['state']=='ARRIVED'
    assert first['interrupted'] and not first['full_step_completed'] and first['source_stamp_consumed']
    assert result['local_avoidance_actions']==1
    assert len(result['lidar_recovery_history'])==1
    assert flags['sleeps']==2
    assert result['history'][1]['source_frame_stamp_ns']>result['history'][0]['source_frame_stamp_ns']
    assert result['completed_forward_actions']==1
    assert result['progress_diagnostics']['actions'][0]['interrupted']
    assert result['interrupted_strafe_attempts']==1 and result['completed_strafe_actions']==0
    assert result['progress_diagnostics']['action_summary'][0]['nominal_lateral_displacement_m'] is None


def test_lateral_sensor_outage_is_bounded(tmp_path,monkeypatch):
    bundle,flags,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)],[LEFT_OPEN],interruptions=(True,))
    flags['publish']=False
    result=run(bundle[0]);assert result['state']=='BLOCKED'
    assert result['reason']=='find_marvin_new_lidar_evidence_timeout'
    assert len(motions(bundle[3]))==1 and flags['sleeps']<=13


@pytest.mark.parametrize('phase',['dispatch','wait'])
def test_stop_preempts_strafe_or_recovery(tmp_path,monkeypatch,phase):
    bundle,flags,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*2,[LEFT_OPEN],interruptions=(True,) if phase=='wait' else ())
    stop=lambda:bundle[0].submit_intent({'intent':'STOP','speech':'Stop.'})
    flags['on_dispatch' if phase=='dispatch' else 'on_sleep']=stop
    result=run(bundle[0]);assert result['behavior']=='STOP'
    assert len(motions(bundle[3]))==1 and bundle[3][-1]=='stop'


def test_duplicate_camera_cannot_authorize_second_strafe(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*2,[LEFT_OPEN])
    bundle[2].on_motion=lambda:setattr(bundle[1],'repeat_camera',True)
    result=run(bundle[0]);assert len(motions(bundle[3]))==1
    assert result['state']=='REVERIFY_REQUIRED'


def test_tracker_loss_after_strafe_uses_gemini(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.60)]*3+[(0,.5)],[LEFT_OPEN,None],factory=recovery_runtime)
    result=run(bundle[0]);assert result['state']=='ARRIVED'
    assert result['reacquisition_history'][0]['succeeded']
    assert bundle[1].semantic_calls==2 and result['local_avoidance_actions']==1


def test_clear_path_and_trusted_standoff_do_not_select_avoidance(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6),(0,.5)],[None])
    def forbidden(*a,**k):pytest.fail('Clear pursuit must be unchanged')
    monkeypatch.setattr('runtime.select_marvin_escape_action',forbidden)
    result=run(bundle[0]);assert result['state']=='ARRIVED'
    assert motions(bundle[3])==[('forward',.10,.50)]
    assert result['final_observation']['arrival']['target_standoff_m']==.5


def test_safe_strafe_then_unsafe_repair_waits_without_injected_detour_turn(tmp_path,monkeypatch):
    scenes=[LEFT_OPEN,LEFT_OPEN+[(-.02,.465),(0.,-.465)]]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)]*9,scenes)
    result=run(bundle[0])
    assert result['reason']=='find_marvin_blocked_wait_exhausted'
    assert motions(bundle[3])==[('strafe',.08,1.)]
    assert result['local_avoidance_actions']==1
    assert result['stop_result']['ok']


def test_all_four_candidates_share_exact_scan_and_strafe_tie_prefers_left(tmp_path,monkeypatch):
    scene=LEFT_OPEN[:3]+[(0.,1.2),(0.,-1.2)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[scene])
    runtime=bundle[0];lidar=runtime.world_model.get_lidar_obstacles()
    association={'target_bearing_degrees':0.,'verified_marvin_distance_m':None}
    result=select_marvin_escape_action(lidar,association,expected_session=runtime.lidar_worker.session,allow_strafe=True)
    assert result['action_type']=='STRAFE_LEFT'
    assert len(result['options'])==4 and result['acquisition_sequence']==lidar['acquisition_sequence']
    assert all(option['hard_safety_permitted'] for option in result['options'].values())


def test_right_corridor_improvement_wins_when_both_strafes_safe(tmp_path,monkeypatch):
    scene=[(.55,.07),(.551,.073),(.551,.067),(0.,1.2),(0.,-1.2)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[scene])
    runtime=bundle[0];lidar=runtime.world_model.get_lidar_obstacles()
    result=select_marvin_escape_action(lidar,{'target_bearing_degrees':0.,'verified_marvin_distance_m':None},
        expected_session=runtime.lidar_worker.session,allow_strafe=True)
    assert result['options']['STRAFE_LEFT']['hard_safety_permitted']
    assert result['options']['STRAFE_RIGHT']['hard_safety_permitted']
    assert result['action_type']=='STRAFE_RIGHT'


def test_stale_or_changed_producer_never_proposes_candidates(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[LEFT_OPEN])
    runtime=bundle[0]
    for changes in ({'age_at_receipt_seconds':.301},{'producer_session':'other'}):
        lidar=runtime.world_model.get_lidar_obstacles();lidar.update(changes)
        result=select_marvin_escape_action(lidar,{'target_bearing_degrees':0.},
            expected_session=runtime.lidar_worker.session,allow_strafe=True)
        assert result['action_type'] is None and result['options']=={}


def test_still_useful_left_cannot_reverse_merely_for_right_clearance(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[LEFT_OPEN])
    runtime=bundle[0];association={'target_bearing_degrees':0.}
    before=runtime.world_model.get_lidar_obstacles()
    old=select_marvin_escape_action(before,association,expected_session=runtime.lidar_worker.session,allow_strafe=True)
    assert old['action_type']=='STRAFE_LEFT'
    after=runtime.world_model.get_lidar_obstacles()
    for p in after['local_motion_geometry']['points']:
        if p['x_m']>0:p['y_m']-=.04
    for n in ('right','front_right','rear_right'):
        after['local_motion_geometry']['sectors'][n]['minimum_distance_from_base_m']=1.5
    new=select_marvin_escape_action(after,association,expected_session=runtime.lidar_worker.session,
        allow_strafe=True,previous_selection=old)
    assert new['progress_improved'] and new['direction']=='LEFT'


def test_full_lateral_guard_blocks_start_circle_on_either_side(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8)],[None])
    runtime=bundle[0]
    for y in (.44,-.44):
        state=runtime.world_model.get_lidar_obstacles()
        state['local_motion_geometry']['points'].append({'x_m':0.,'y_m':y})
        result=evaluate_local_motion_safety(state,expected_session=runtime.lidar_worker.session,
            linear_y=.08,duration=.5,lateral_swept_footprint=True)
        assert not result['permitted'] and result['protected_radius_m']==.45


def test_full_lateral_guard_requires_complete_start_footprint_coverage(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8)],[None])
    runtime=bundle[0];state=runtime.world_model.get_lidar_obstacles()
    state['local_motion_geometry']['sectors']['right']['valid_sample_count']=0
    result=evaluate_local_motion_safety(state,expected_session=runtime.lidar_worker.session,
        linear_y=.08,duration=.5,lateral_swept_footprint=True)
    assert not result['permitted'] and result['reason']=='insufficient_lidar_samples'


def test_guarded_lateral_rejects_bad_acknowledgement_and_confirms_stop(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[LEFT_OPEN])
    robot=bundle[2]
    # Old/malformed acknowledgements cannot claim a completed lateral step.
    robot.move_lateral=lambda **kwargs: {'ok':True,'mode':'bounded','automatic_stop':True}
    result=run(bundle[0])
    assert result['state']=='BLOCKED' and result['local_avoidance_actions']==0
    assert not result['history'][0]['result']['full_step_completed']
    assert result['history'][0]['result']['source_stamp_consumed']
    assert bundle[3][-1]=='stop'


def test_missing_post_stop_lateral_state_cannot_release_recovery(tmp_path,monkeypatch):
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[LEFT_OPEN],interruptions=(True,))
    robot=bundle[2];status=robot.status
    def lose_lateral_telemetry():
        response=status()
        if motions(bundle[3]): response['motion'].pop('linear_y')
        return response
    robot.status=lose_lateral_telemetry
    result=run(bundle[0])
    assert result['state']=='BLOCKED' and len(motions(bundle[3]))==1
    assert not result['history'][0]['result']['bridge_stop_confirmed']
    assert result['reason']=='find_marvin_lidar_recovery_stop_unconfirmed'


def test_jit_geometry_changed_during_capability_check_vetoes_transport(tmp_path,monkeypatch):
    scenes=[LEFT_OPEN]
    bundle,_,client=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],scenes)
    request=client._request
    def obstruct_during_status(method,path,payload=None):
        response=request(method,path,payload)
        if method=='GET': scenes[0]=LEFT_OPEN+[(0.,.44)]
        return response
    client._request=obstruct_during_status
    result=run(bundle[0])
    assert result['state']=='BLOCKED' and motions(bundle[3])==[]
    assert result['history'][0]['result']['source_stamp_consumed']
    assert not result['history'][0]['result']['full_step_completed']


@pytest.mark.parametrize('change',['STOP','camera_expired'])
def test_stop_or_camera_expiration_during_capability_get_preempts_dispatch(tmp_path,monkeypatch,change):
    bundle,_,client=strafe_runtime(tmp_path,monkeypatch,[(0,.6)],[LEFT_OPEN])
    request=client._request
    def interrupt_get(method,path,payload=None):
        response=request(method,path,payload)
        if method=='GET':
            if change=='STOP': bundle[0].submit_intent({'intent':'STOP','speech':'Stop.'})
            else: bundle[4][0]+=1_000_000_001
        return response
    client._request=interrupt_get
    result=run(bundle[0])
    assert motions(bundle[3])==[] and bundle[3][-1]=='stop'
    assert result['behavior']=='STOP' if change=='STOP' else result['state']=='BLOCKED'
