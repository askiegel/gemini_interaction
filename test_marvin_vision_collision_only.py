"""Vision-led mission replays with fake robot clients; no live calibration claims."""
import ast
import copy
import gzip
import json
import math
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import pytest

from marvin_arrival_policy import evaluate_marvin_pursuit_arrival
from marvin_route_obstruction import evaluate_marvin_route
from test_find_marvin_closed_loop import make_runtime, motions, run

FIXTURE = json.loads(gzip.decompress((Path(__file__).parent /
    'test_fixtures/marvin_930a9281_sparse_range.json.gz').read_bytes()))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline vision pursuit cannot access robots/services')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr('marvin_target_range_association.MarvinTargetRangeAssociation.evaluate', forbidden)
    monkeypatch.setattr('marvin_lidar_standoff.evaluate_marvin_lidar_standoff', forbidden)


def retained(tmp_path, monkeypatch):
    r, b, robot, events, clock = make_runtime(tmp_path, monkeypatch, [], calibrated_arrival=False)
    v = copy.deepcopy(FIXTURE['refusal'])
    v['lidar'].update(received_monotonic_seconds=time.monotonic(),
                      effective_age_seconds=0.,age_at_receipt_seconds=0.)
    r.lidar_worker = SimpleNamespace(session=v['lidar']['producer_session'],running=True)
    r.marvin_camera_model = copy.deepcopy(FIXTURE['calibration'])
    return r,v


@pytest.mark.parametrize('support',[0,1,3,4,5])
def test_exact_four_return_case_and_other_support_counts_do_not_gate_pursuit(tmp_path,monkeypatch,support):
    r,v = retained(tmp_path,monkeypatch)
    points=v['lidar']['local_motion_geometry']['points']
    near=[p for p in points if 1.3<math.hypot(p['x_m'],p['y_m'])<1.5
          and -1.7<p['robot_bearing_deg']<.6]
    assert len(near)==4
    for p in near:points.remove(p)
    points.extend(near[:support])
    if support==5:points.append(dict(x_m=1.42,y_m=.03,distance_m=1.42,robot_bearing_deg=1.2))
    result=r._marvin_v2_motion_clearance(v['tracker'],v['lidar'])
    assert result['ok'] and result['authority']=='visual_tracking'
    assert not result['direct_path_blocked'] and not result['route_to_marvin_obstructed']
    assert result['arrived_at_marvin'] is False
    assert 'target_distance_m' not in result and 'candidate_surface_point_count' not in result
    assert 'target_range_association_trusted' not in result
    assert result['association_forward_probe']['protected_radius_m']==.45


def test_camera_bearing_ignores_surface_range_and_camera_sensor_translation(tmp_path,monkeypatch):
    r,v=retained(tmp_path,monkeypatch)
    before=r._marvin_v2_motion_clearance(v['tracker'],v['lidar'])
    for p in v['lidar']['local_motion_geometry']['points']:
        if p['x_m']>1.:p['x_m']*=2.;p['y_m']*=2.
    r.marvin_camera_model.update(x_m=999.,y_m=-999.,range_uncertainty_m=999.)
    after=r._marvin_v2_motion_clearance(v['tracker'],v['lidar'])
    assert after['target_bearing_degrees']==before['target_bearing_degrees']
    assert after['route']['route_depth_source']=='bounded_command_envelope'
    assert after['route']['route_lookahead_m']==.05


@pytest.mark.parametrize('fault',['intrusion','rear_intrusion','stale','coverage','session','sequence',
                                  'advertised_age','cached_receipt','boolean_sequence'])
def test_independent_collision_gates_still_fail_closed(tmp_path,monkeypatch,fault):
    r,v=retained(tmp_path,monkeypatch);s=v['lidar']
    if fault=='intrusion':s['local_motion_geometry']['points'].append(dict(x_m=.4,y_m=0.,distance_m=.4,robot_bearing_deg=0.))
    if fault=='rear_intrusion':s['local_motion_geometry']['points'].append(dict(x_m=-.4,y_m=0.,distance_m=.4,robot_bearing_deg=180.))
    if fault=='stale':s.update(valid=False,reason='stale',effective_age_seconds=.31)
    if fault=='coverage':s['local_motion_geometry']['sectors']['front']['valid_sample_count']=4
    if fault=='session':s['producer_session']='another-worker'
    if fault=='sequence':r._marvin_last_action_lidar_evidence=(s['producer_session'],s['acquisition_sequence'])
    if fault=='advertised_age':s['effective_age_seconds']=.300001
    if fault=='cached_receipt':s['received_monotonic_seconds']-=.300001
    if fault=='boolean_sequence':s['acquisition_sequence']=True
    result=r._marvin_v2_motion_clearance(v['tracker'],s)
    assert not result['ok'] or result['direct_path_blocked'] or result['route_to_marvin_obstructed']
    assert not result['arrived_at_marvin']
    if fault in {'intrusion','rear_intrusion'}:assert result['route_to_marvin_obstructed']


@pytest.mark.parametrize('fault',['resolution','focal_length','yaw','principal_point','bbox','vertical_box'])
def test_camera_bearing_calibration_and_box_geometry_remain_required(tmp_path,monkeypatch,fault):
    r,v=retained(tmp_path,monkeypatch)
    if fault=='resolution':r.marvin_camera_model['image_width']=1280
    if fault=='focal_length':r.marvin_camera_model['fx_pixels']=0.
    if fault=='yaw':r.marvin_camera_model['yaw_degrees']=float('nan')
    if fault=='principal_point':r.marvin_camera_model['cx_pixels']=-1.
    if fault=='bbox':v['tracker']['bbox']['x2']=641.
    if fault=='vertical_box':v['tracker']['bbox']['y1']=-1.
    result=r._marvin_v2_motion_clearance(v['tracker'],v['lidar'])
    assert not result['ok'] and result['reason']=='marvin_camera_bearing_unavailable'
    assert not result['arrived_at_marvin']


def test_legacy_visual_threshold_is_not_promoted_to_arrival():
    preview=dict(ok=True,preview=True,authoritative=False,target='marvin',
        target_found=True,source='marvin_local_tracker',identity_confirmed=True,
        motion_authorized_marvin_candidate=True,image_width=640,image_height=480,
        source_timestamp='2026-10-14T01:12:19+00:00',
        bbox=dict(x1=224,y1=17,x2=446,y2=290))
    result=evaluate_marvin_pursuit_arrival(preview,now=preview['source_timestamp'])
    assert result['legacy_visual_threshold_met']
    assert not result['arrived_at_marvin'] and not result['calibration_verified']
    assert result['reason']=='marvin_visual_arrival_calibration_unverified'


def test_vision_pursuit_uses_full_bounded_step_without_lidar_distance_taper(tmp_path,monkeypatch):
    r,b,robot,events,clock=make_runtime(tmp_path,monkeypatch,[(0,.51),'lost'],calibrated_arrival=False)
    result=run(r)
    assert motions(events)==[('forward',.1,.5)]
    assert not result['arrived_at_marvin']
    assert result['state']=='REVERIFY_REQUIRED'


def blocked_scene(tmp_path,monkeypatch,*,clear,obstacle_x=.4):
    r,b,robot,events,clock=make_runtime(tmp_path,monkeypatch,
        [(0,1.4),(0,1.4),'lost'],calibrated_arrival=False)
    state={'blocked':True,'wait_frames':[],'resumes':0}
    scan=b.lidar
    def lidar(**kwargs):
        s=scan(**kwargs)
        if state['blocked']:
            s['local_motion_geometry']['points'].append(dict(x_m=obstacle_x,y_m=0.,distance_m=.4,
                robot_bearing_deg=0. if obstacle_x>0 else 180.))
        return s
    r.world_model.get_lidar_obstacles=lidar
    publish=r._publish_behavior_tracking
    def tracking(value):
        if value.get('state')=='BLOCKED_WAIT':
            state['wait_frames'].append(len(b.stamps))
            assert not motions(events)
        return publish(value)
    r._publish_behavior_tracking=tracking
    sleep=time.sleep
    def tick(seconds):
        if clear and state['wait_frames']:state['blocked']=False
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep',tick)
    def reacquire(*,minimum_source_frame_stamp_ns):
        assert not state['blocked']
        state['resumes']+=1
        b._clear_marvin_v2_tracker_episode()
        observation=b.observe_find_marvin_v2()
        assert observation['preview_result']['source_frame_stamp_ns']>minimum_source_frame_stamp_ns
        return observation
    b.reacquire_find_marvin_v2=reacquire
    return r,b,events,state


def test_collision_stop_wait_clearance_and_fresh_camera_precede_replacement(tmp_path,monkeypatch):
    r,b,events,state=blocked_scene(tmp_path,monkeypatch,clear=True)
    result=run(r)
    assert state['wait_frames'] and state['resumes']>=1
    assert events.index('stop')<events.index(('forward',.1,.5))
    assert motions(events)==[('forward',.1,.5)]
    assert result['blocked_wait_recheck_count']>=1
    assert result['state']=='REVERIFY_REQUIRED' and not result['arrived_at_marvin']


@pytest.mark.parametrize('obstacle_x',[.4,-.4])
def test_uncalibrated_arrival_stops_safely_at_collision_boundary(tmp_path,monkeypatch,obstacle_x):
    r,b,events,state=blocked_scene(tmp_path,monkeypatch,clear=False,obstacle_x=obstacle_x)
    result=run(r)
    assert state['wait_frames'] and not state['resumes'] and not motions(events)
    assert result['state']=='BLOCKED' and not result['arrived_at_marvin']
    assert result['reason']=='marvin_visual_arrival_unverified_at_lidar_safety_boundary'
    assert result['stop_result']['ok'] and events[-1]=='stop'


def test_stop_preempts_existing_collision_wait(tmp_path,monkeypatch):
    r,b,events,state=blocked_scene(tmp_path,monkeypatch,clear=False)
    publish=r._publish_behavior_tracking
    def tracking(value):
        publish(value)
        if value.get('state')=='BLOCKED_WAIT':r.submit_intent({'intent':'STOP','speech':'Offline stop'})
    r._publish_behavior_tracking=tracking
    result=run(r)
    assert result['behavior']=='STOP' and not motions(events)
    assert r.mission_manager.get_active_mission() is None and not r.mission_manager.get_queue()


@pytest.mark.parametrize('distance',[.4,1.4])
def test_mocked_calibrated_visual_arrival_is_independent_of_lidar_range(tmp_path,monkeypatch,distance):
    # Positive control-flow oracle only; no physical calibration is implied.
    r,b,robot,events,clock=make_runtime(tmp_path,monkeypatch,[(0,distance)])
    monkeypatch.setattr('runtime.evaluate_marvin_pursuit_arrival',lambda preview:
        dict(ok=True,calibration_verified=True,arrived_at_marvin=True,reason='offline_visual_oracle'))
    result=run(r)
    assert result['state']=='ARRIVED' and not motions(events)
    assert result['final_observation']['arrival']['authority']=='visual_tracking'


def test_active_runtime_has_no_target_range_or_sparse_decision_calls():
    source=(Path(__file__).parent/'runtime.py').read_text()
    calls={n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert ('_marvin_v2_'+'lidar_arrival') not in calls and '_recover_marvin_sparse_range' not in calls
    assert 'record_forward_bound' not in calls and 'corroborate_sparse' not in calls
    assert 'marvin_target_range_returns_ambiguous' not in source


@pytest.mark.parametrize('action',['forward','forward_rear','alignment'])
def test_collision_appearing_at_final_jit_stops_waits_and_requires_new_frame(tmp_path,monkeypatch,action):
    alignment=action=='alignment'
    spec=(120,1.4) if alignment else (0,1.4)
    r,b,robot,events,clock=make_runtime(tmp_path,monkeypatch,[spec,spec,'lost'],calibrated_arrival=False)
    status=robot.status
    robot.status=lambda:dict(status(),motion=dict(status()['motion'],linear_y=0.))
    r.forward_interlock=SimpleNamespace(status=lambda:dict(active_forward=False,
        pending_forward=False,stop_pending=False,last_stop_error=None))
    state={'inject':False,'wait_seen':False,'clear':False}
    method='_execute_single_marvin_alignment' if alignment else '_execute_single_marvin_approach'
    execute=getattr(r,method)
    def at_jit(**kwargs):
        state['inject']=True
        return execute(**kwargs)
    monkeypatch.setattr(r,method,at_jit)
    scan=b.lidar
    def lidar(**kwargs):
        s=scan(**kwargs)
        if state['inject'] and not state['clear']:
            x,y=(-.20,-.30) if alignment else (-.4,0.) if action=='forward_rear' else (.4,0.)
            s['local_motion_geometry']['points'].append(dict(x_m=x,y_m=y,
                distance_m=math.hypot(x,y),robot_bearing_deg=math.degrees(math.atan2(y,x))))
        return s
    r.world_model.get_lidar_obstacles=lidar
    publish=r._publish_behavior_tracking
    def tracking(value):
        if value.get('state')=='BLOCKED_WAIT':
            assert not motions(events)
            state['wait_seen']=True
        publish(value)
    r._publish_behavior_tracking=tracking
    sleep=time.sleep
    def tick(seconds):
        if state['wait_seen']:state['clear']=True
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep',tick)
    def reacquire(*,minimum_source_frame_stamp_ns):
        b._clear_marvin_v2_tracker_episode()
        return b.observe_find_marvin_v2()
    b.reacquire_find_marvin_v2=reacquire
    result=run(r)
    assert state['wait_seen'], {k:result.get(k) for k in ('state','reason','history')}
    assert result['blocked_wait_recheck_count']>=1
    assert len(motions(events))==1 and motions(events)[0][0]==('turn' if alignment else 'forward')
    assert len(set(b.stamps))==len(b.stamps)
    assert result['history'][0]['motion_executed'] is False
    assert result['history'][0]['result']['pre_transport_jit_veto']['transport_attempted'] is False
    assert not result['arrived_at_marvin']
