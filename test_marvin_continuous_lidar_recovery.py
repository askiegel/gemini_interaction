"""Offline bounded recovery: real parser/runtime/mission, synthetic sensing."""
import copy
import socket
import time

import pytest

from robot_bridge.http_response import server_error, certified_active_lidar_interruption
from runtime import CognitiveRuntime
from marvin_blocked_wait import blocked_wait_diagnostics
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_lidar_perception import setup as worker_setup


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("No robot/network/process access")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr("subprocess.Popen", refuse)


def response(axes=(.1, 0., 0.), duration=.5, reason="active_lidar_intrusion"):
    identity = dict(controller_session="controller-1", bridge_epoch="bridge-1")
    accepted = dict(identity, evaluated_ns=1_010_000_000, last_operation=1,
        last_request_generation=4, last_request_sequence=1,
        last_request_issued_ns=1_000_000_000, last_request_accepted=True)
    stop = dict(identity, evaluated_ns=1_130_000_000, last_operation=3,
        last_request_generation=4, last_request_sequence=3,
        last_request_issued_ns=1_120_000_000, last_request_accepted=True,
        generation=4, sequence=3, active=False, gait_zero=True,
        linear_x=0., linear_y=0., angular_z=0.)
    floor = dict(producer_session="native-1", acquisition_sequence=10, source_stamp_ns=100)
    gate = dict(floor, permitted=False, reason=reason, effective_age_seconds=.02)
    if reason in {"active_lidar_unavailable", "active_lidar_cache_busy"}:
        gate = dict(permitted=False, reason=reason)
    return dict(identity, ok=False, state="interrupted", transport_attempted=True,
        controller_gait_zero=True, automatic_stop=True, physical_stop_verified=False,
        completed_progress_credit=False, reason=reason, stop_reason="scoped_gait_zero",
        generation=4, start_ns=1_000_000_000, deadline_ns=1_000_000_000+int(duration*1e9),
        linear_x=axes[0], linear_y=axes[1], angular_z=axes[2], duration=duration,
        controller_acceptance=accepted, stop_certificate=stop,
        last_verified_lidar=floor, lidar_safety=gate)


def certificate(**kwargs):
    parsed = server_error(503, response(**kwargs), endpoint="/motion")
    assert parsed["evidence_valid"]
    return parsed, certified_active_lidar_interruption(parsed)


@pytest.mark.parametrize("reason", ["active_lidar_intrusion", "active_lidar_stale",
    "active_lidar_unavailable", "active_lidar_cache_busy", "active_lidar_coverage_missing"])
def test_bounded_exact_certificate_preserved(reason):
    body = response(reason=reason)
    body["lidar_safety"]["raw_scan"] = [1]*1000
    body["secret"] = "excluded"
    parsed = server_error(503, body, endpoint="/motion")
    cert = certified_active_lidar_interruption(parsed)
    assert cert and cert["scan_floor"]["acquisition_sequence"] == 10
    assert not parsed["completed_progress_credit"] and not parsed["retry_permitted"]
    assert "raw_scan" not in parsed["lidar_safety"] and "secret" not in parsed


@pytest.mark.parametrize("fault", ["uncertain", "no_transport", "completed", "no_stop",
    "wrong_stop_owner", "active", "gait", "velocity", "unacknowledged", "stop_seq",
    "no_acceptance", "no_floor", "source_session", "source_regression", "source_fault",
    "unknown_reason", "boolean_sequence", "bad_duration", "emergency_stop"])
def test_bad_certificates_cannot_recover(fault):
    body = response()
    if fault == "uncertain": body["delivery_uncertain"] = True
    elif fault == "no_transport": body["transport_attempted"] = False
    elif fault == "completed": body["completed_progress_credit"] = True
    elif fault == "no_stop": body.pop("stop_certificate")
    elif fault == "wrong_stop_owner": body["stop_certificate"]["bridge_epoch"] = "new-owner"
    elif fault == "active": body["stop_certificate"]["active"] = True
    elif fault == "gait": body["stop_certificate"]["gait_zero"] = False
    elif fault == "velocity": body["stop_certificate"]["linear_x"] = .1
    elif fault == "unacknowledged": body["stop_certificate"]["last_request_accepted"] = False
    elif fault == "stop_seq": body["stop_certificate"]["sequence"] = 1
    elif fault == "no_acceptance": body.pop("controller_acceptance")
    elif fault == "no_floor": body.pop("last_verified_lidar")
    elif fault == "source_session": body["lidar_safety"]["producer_session"] = "other"
    elif fault == "source_regression": body["lidar_safety"]["acquisition_sequence"] = 9
    elif fault == "source_fault": body["lidar_safety"]["source_fault"] = "active_lidar_source_regressed"
    elif fault == "unknown_reason": body["reason"] = body["lidar_safety"]["reason"] = "active_lidar_unknown"
    elif fault == "boolean_sequence": body["lidar_safety"]["acquisition_sequence"] = True
    elif fault == "bad_duration": body["duration"] = 2.5
    elif fault == "emergency_stop": body["stop_certificate"]["last_operation"] = 4
    assert certified_active_lidar_interruption(server_error(503, body, endpoint="/motion")) is None


def stationary_status():
    return dict(available=True, protocol_version=1, controller_session="controller-1",
        bridge_epoch="bridge-1", generation=4, sequence=3, active=False, gait_zero=True,
        command_pending=False, admission_quarantined=False, effective_age_seconds=.01,
        last_operation=3,last_request_generation=4,last_request_sequence=3,
        last_request_issued_ns=1_120_000_000,last_request_accepted=True,
        linear_x=0., linear_y=0., angular_z=0.)


def prepare(bundle, monkeypatch, *, clear_after=1., frozen=False, fault=None):
    r, behavior, robot, events, clock = bundle
    flags = dict(waiting=False, began=None, checks=0, fault=fault, observed=[])
    def reacquire(**kw):
        # Match production reacquisition: retire the interrupted tracker episode
        # before the new Gemini identity, rather than reuse locked continuity.
        behavior._clear_marvin_v2_tracker_episode()
        return behavior.observe_find_marvin_v2()
    behavior.reacquire_find_marvin_v2 = reacquire
    flags["generation"] = 4
    status = robot.status
    def current_status():
        value = status()
        value["motion"]["linear_y"] = 0.
        value["guarded_controller"] = stationary_status()
        value["guarded_controller"]["generation"] = flags["generation"]
        if flags["fault"] in {"owner", "quarantine", "nonzero", "controller_stale"}:
            k, v = {"owner":("generation",5), "quarantine":("admission_quarantined",True),
                    "nonzero":("linear_y",.1), "controller_stale":("effective_age_seconds",.25)}[flags["fault"]]
            value["guarded_controller"][k] = v
        return value
    robot.status = current_status
    read = r.world_model.get_lidar_obstacles
    def lidar(**kw):
        state = read(**kw)
        flags["checks"] += 1
        seq = 10 if frozen else 10+flags["checks"]
        state["bridge_scan_identity"] = dict(producer_session="native-1",
            acquisition_sequence=seq, source_stamp_ns=90+seq)
        if flags["waiting"]:
            flags["observed"].append(seq)
            blocked = clear_after is None or time.monotonic()-flags["began"] < clear_after
            if blocked:
                state["local_motion_geometry"]["points"].append(dict(x_m=.3,y_m=0.))
            if flags["fault"] == "session": state["bridge_scan_identity"]["producer_session"] = "changed"
            if flags["fault"] == "regression": state["bridge_scan_identity"]["acquisition_sequence"] = 9
            if flags["fault"] == "stale": state.update(reason="stale", effective_age_seconds=.31)
            if flags["fault"] == "coverage":
                next(iter(state["local_motion_geometry"]["sectors"].values()))["valid_sample_count"] = 0
        return state
    r.world_model.get_lidar_obstacles = lidar
    return flags


def wait_case(tmp_path, monkeypatch, **kwargs):
    axes = kwargs.pop("axes", (.1,0.,0.))
    bundle = make_runtime(tmp_path, monkeypatch, [])
    flags = prepare(bundle, monkeypatch, **kwargs)
    r, _, _, _, _ = bundle
    _, cert = certificate(axes=axes)
    entry = r.world_model.get_lidar_obstacles()
    association = dict(verified_marvin_distance_m=.8, verified_marvin_conservative_distance_m=.7,
        target_bearing_degrees=0., route=dict(valid=True, route_to_marvin_obstructed=False))
    flags.update(waiting=True, began=time.monotonic())
    diagnostics, history = blocked_wait_diagnostics(), []
    def execute(guard=lambda:True):
        return r._wait_for_marvin_blocked_route(expected_session=r.lidar_worker.session,
            lidar=entry, association=association, execution_guard=guard,
            diagnostics=diagnostics, history=history, interruption=cert)
    return bundle, flags, diagnostics, history, execute


def test_obstacle_clears_only_after_full_envelope_and_new_native_scan(tmp_path, monkeypatch):
    bundle, flags, d, history, execute = wait_case(tmp_path, monkeypatch)
    outcome = execute()
    assert outcome["ok"] and outcome["reason"] == "find_marvin_protected_envelope_cleared"
    assert d["blocked_wait_recheck_count"] == 2
    assert [h["protected_envelope_clear"] for h in history] == [False, True]
    assert history[-1]["bridge_scan_identity"]["acquisition_sequence"] > 10
    assert not motions(bundle[3])


@pytest.mark.parametrize("fault", ["session", "regression", "stale", "coverage", "owner",
    "quarantine", "nonzero", "controller_stale"])
def test_sensing_stop_or_ownership_fault_cannot_release_wait(tmp_path, monkeypatch, fault):
    bundle, _, _, _, execute = wait_case(tmp_path, monkeypatch, clear_after=0., fault=fault)
    assert not execute()["ok"]
    assert not motions(bundle[3])


@pytest.mark.parametrize("frozen", [False, True])
def test_persistent_or_equal_source_exhausts_existing_wait(tmp_path, monkeypatch, frozen):
    bundle, _, d, _, execute = wait_case(tmp_path, monkeypatch,
        clear_after=0. if frozen else None, frozen=frozen)
    result = execute()
    assert not result["ok"] and result["reason"] == "find_marvin_blocked_wait_exhausted"
    assert d["blocked_wait_recheck_count"] == 12 and d["blocked_wait_total_seconds"] <= 30.
    assert not motions(bundle[3])


@pytest.mark.parametrize("axes", [(.1,0.,0.),(-.1,0.,0.),(0.,.08,0.),(0.,-.08,0.),(0.,0.,.25),(0.,0.,-.25)])
@pytest.mark.parametrize("gap", [.45, .450001])
def test_full_envelope_boundary_all_directions(axes, gap):
    _, cert = certificate(axes=axes)
    point = dict(x_m=0.,y_m=gap) if axes[0] else dict(x_m=gap,y_m=0.)
    assert CognitiveRuntime._marvin_interrupted_envelope_clear(
        dict(local_motion_geometry=dict(points=[point])), cert) is (gap > .45)


def test_worker_retains_native_identity_without_renaming_poll_identity(worker_setup):
    world, worker, clock, source = worker_setup
    source["telemetry"].update(producer_session="native-1", acquisition_sequence=500, source_stamp_ns=123)
    first, second = worker.run_once(), worker.run_once()
    assert second["acquisition_sequence"] == first["acquisition_sequence"]+1
    assert second["producer_session"] == worker.session != "native-1"
    assert first["bridge_scan_identity"] == second["bridge_scan_identity"]


@pytest.mark.parametrize("reason", ["active_lidar_intrusion", "active_lidar_stale", "active_lidar_unavailable"])
def test_normal_mission_reconfirms_new_identity_before_replacement(tmp_path, monkeypatch, reason):
    # Interrupted approach, stationary boundary view, independent replacement
    # view, then the existing synthetic calibrated-arrival oracle.
    bundle = make_runtime(tmp_path, monkeypatch, [(0,.60),(0,.60),(0,.60),(0,.50)])
    r, b, robot, events, clock = bundle
    flags = prepare(bundle, monkeypatch)
    old = robot.move_forward
    def forward(*, speed, seconds, dispatch_guard=None):
        if not flags["waiting"]:
            assert dispatch_guard is None or dispatch_guard()
            events.append(("forward",speed,seconds))
            clock[0] += 120_000_000
            flags.update(waiting=True, began=time.monotonic())
            parsed, _ = certificate(duration=seconds, reason=reason)
            return parsed
        return old(speed=speed, seconds=seconds, dispatch_guard=dispatch_guard)
    robot.move_forward = forward
    outcome = run(r)
    assert outcome["state"] == "ARRIVED", outcome["reason"]
    assert outcome["interrupted_forward_attempts"] == 1 and outcome["completed_forward_actions"] == 1
    first, replacement = outcome["history"]
    assert not first["result"]["full_step_completed"] and first["result"]["source_stamp_consumed"]
    assert first["source_frame_stamp_ns"] < replacement["observation"]["identity_source_frame_stamp_ns"] < replacement["source_frame_stamp_ns"]
    boundary, = outcome["collision_boundary_observations"]
    assert boundary["camera_recheck_performed"] and boundary["marvin_confirmed_forward"]
    assert not boundary["safe_approach_complete"] and not boundary["motion_authorized"]
    assert (first["source_frame_stamp_ns"] < boundary["observation"]["source_frame_stamp_ns"]
        < replacement["observation"]["identity_source_frame_stamp_ns"])
    assert b.identity_sources[:2] == ["gemini_marvin_candidate_selection"]*2
    assert len(motions(events)) == 2 and r.mission_manager.get_active_mission() is None
    assert outcome["lidar_recovery_history"][0]["transported_interruption"]
    duplicate = r.execute_single_marvin_approach(linear_speed=.1, duration=.5,
        source_frame_stamp_ns=first["source_frame_stamp_ns"])
    assert duplicate["reason"] == "marvin_approach_observation_already_consumed"


def test_cancellation_during_wait_never_dispatches(tmp_path, monkeypatch):
    bundle, _, _, _, execute = wait_case(tmp_path, monkeypatch)
    start = time.monotonic()
    result = execute(lambda: time.monotonic()-start < .2)
    assert not result["ok"] and result["reason"] == "find_marvin_mission_preempted"
    assert not motions(bundle[3])


def test_failed_stop_cannot_enter_wait(tmp_path, monkeypatch):
    bundle, _, _, _, execute = wait_case(tmp_path, monkeypatch)
    bundle[2].stop = lambda:dict(ok=False)
    assert execute()["reason"] == "find_marvin_blocked_wait_stop_failed"


@pytest.mark.parametrize("side", ["LEFT","RIGHT"])
def test_transported_strafe_counts_once_preserves_side_without_progress(tmp_path,monkeypatch,side):
    from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN, RIGHT_OPEN
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.6)]*3+[(0,.5)],
        [LEFT_OPEN if side=="LEFT" else RIGHT_OPEN,None,None])
    r,b,robot,events,clock=bundle
    flags=prepare(bundle,monkeypatch)
    def lateral(*,speed,seconds,dispatch_guard=None):
        assert dispatch_guard is None or dispatch_guard()
        assert not flags["waiting"]
        events.append(("strafe",speed,seconds));clock[0]+=120_000_000
        flags.update(waiting=True,began=time.monotonic())
        parsed,_=certificate(axes=(0.,speed,0.),duration=seconds)
        return parsed
    robot.move_lateral=lateral
    outcome=run(r)
    assert outcome["state"]=="ARRIVED",outcome["reason"]
    assert outcome["local_avoidance_actions"]==1
    assert outcome["completed_strafe_actions"]==0 and outcome["interrupted_strafe_attempts"]==1
    recovery=outcome["lidar_recovery_history"][0]
    assert recovery["committed_side"]==side and recovery["phase"]=="CLEAR_SIDE"
    assert not recovery["completed_progress_credit"]
    assert len(motions(events))==2


@pytest.mark.parametrize("fault", ["identity","repeat_camera","tracker","stop_pending","stop_error","cancel"])
def test_mission_recovery_faults_never_replace_movement(tmp_path,monkeypatch,fault):
    specs=[(0,.6)]+(["invalid"]*5 if fault=="identity" else [(0,.6),(0,.5)])
    bundle=make_runtime(tmp_path,monkeypatch,specs)
    r,b,robot,events,clock=bundle
    flags=prepare(bundle,monkeypatch)
    def forward(*,speed,seconds,dispatch_guard=None):
        events.append(("forward",speed,seconds));clock[0]+=120_000_000
        flags.update(waiting=True,began=time.monotonic())
        if fault=="repeat_camera":b.repeat_camera=True
        if fault=="tracker":
            original=b.observe_find_marvin_v2
            def bad_tracker():
                evidence=original()
                evidence["preview_result"]["opencv_tracker"]["quality"]=.79
                return evidence
            b.observe_find_marvin_v2=bad_tracker
        if fault=="stop_pending":r.forward_interlock._stops_inflight=1
        if fault=="stop_error":r.forward_interlock._last_stop_error="uncertain"
        if fault=="cancel":r.submit_intent({"intent":"STOP","speech":"cancel"})
        return certificate(duration=seconds)[0]
    robot.move_forward=forward
    outcome=run(r)
    assert not outcome.get("arrived_at_marvin")
    assert len(motions(events))==1


def test_interlock_wrapping_never_hides_failed_stop():
    parsed,cert=certificate()
    wrapped=dict(parsed,bounded_forward_invalidated=True,transport_result=parsed,
        interlock_stop_succeeded=False)
    assert CognitiveRuntime._marvin_bridge_lidar_interruption(wrapped) is None
    wrapped["interlock_stop_succeeded"]=True
    assert CognitiveRuntime._marvin_bridge_lidar_interruption(wrapped)==cert


@pytest.mark.parametrize("fault", ["equal_sequence_new_stamp","new_sequence_equal_stamp","radius"])
def test_inconsistent_source_or_footprint_cannot_certify_recovery(fault):
    body=response()
    if fault=="equal_sequence_new_stamp":body["lidar_safety"]["source_stamp_ns"]+=1
    elif fault=="new_sequence_equal_stamp":body["lidar_safety"]["acquisition_sequence"]+=1
    else:body["lidar_safety"]["protected_radius_m"] = .44
    assert certified_active_lidar_interruption(server_error(503,body,endpoint="/motion")) is None


def test_aligned_target_not_required_to_wait_for_rotation_clearance(tmp_path,monkeypatch):
    bundle, _, _, _, execute=wait_case(tmp_path,monkeypatch,clear_after=0.,axes=(0.,0.,.25))
    original=bundle[0]._wait_for_marvin_blocked_route
    # A turn can be admitted by strict visual alignment before target range
    # association. Unknown historical route is not a movement permission.
    def unknown_route(**kw):
        kw['association']=dict(route={'valid':False})
        return original(**kw)
    monkeypatch.setattr(bundle[0],'_wait_for_marvin_blocked_route',unknown_route)
    assert execute()['ok']
    assert not motions(bundle[3])


def test_cancellation_after_wait_reacquisition_never_dispatches(tmp_path,monkeypatch):
    bundle=make_runtime(tmp_path,monkeypatch,[(0,.6),(0,.6),(0,.5)])
    r,b,robot,events,clock=bundle
    flags=prepare(bundle,monkeypatch,clear_after=0.)
    def forward(*,speed,seconds,dispatch_guard=None):
        assert not flags['waiting']
        events.append(('forward',speed,seconds));clock[0]+=120_000_000
        flags.update(waiting=True,began=time.monotonic())
        return certificate(duration=seconds)[0]
    robot.move_forward=forward
    b.on_observe=lambda:r.submit_intent({'intent':'STOP'}) if flags['waiting'] else None
    result=run(r)
    assert not result.get('arrived_at_marvin') and len(motions(events))==1
    assert r.mission_manager.get_active_mission() is None


def test_third_consecutive_transported_interruption_exhausts_without_fourth_command(tmp_path,monkeypatch):
    bundle=make_runtime(tmp_path,monkeypatch,[(0,.6)]*5)
    r,b,robot,events,clock=bundle
    flags=prepare(bundle,monkeypatch,clear_after=0.)
    dispatched=[]
    def forward(*,speed,seconds,dispatch_guard=None):
        assert dispatch_guard is None or dispatch_guard()
        events.append(('forward',speed,seconds));clock[0]+=120_000_000
        flags.update(waiting=True,began=time.monotonic(),generation=4+len(dispatched))
        body=response(duration=seconds)
        generation=flags['generation']
        body['generation']=generation
        for key in ('controller_acceptance','stop_certificate'):
            body[key]['last_request_generation']=generation
        body['stop_certificate']['generation']=generation
        floor=dict(producer_session='native-1',acquisition_sequence=10+flags['checks'],
                   source_stamp_ns=100+flags['checks'])
        body['lidar_safety'].update(floor);body['last_verified_lidar']=floor
        dispatched.append(generation)
        return server_error(503,body,endpoint='/motion')
    robot.move_forward=forward
    outcome=run(r)
    assert outcome['reason']=='find_marvin_lidar_recovery_exhausted'
    assert len(motions(events))==3 and len(set(dispatched))==3
    assert outcome['interrupted_forward_attempts']==3 and outcome['completed_forward_actions']==0
    assert len(outcome['lidar_recovery_history'])==2  # third failure terminates before wait


@pytest.mark.parametrize('fault',['delivery','missing_native_floor','controller_restart'])
def test_nonrecoverable_bridge_outcomes_end_without_replacement(tmp_path,monkeypatch,fault):
    bundle=make_runtime(tmp_path,monkeypatch,[(0,.6),(0,.6)])
    r,b,robot,events,clock=bundle
    flags=prepare(bundle,monkeypatch)
    def forward(*,speed,seconds,dispatch_guard=None):
        events.append(('forward',speed,seconds));clock[0]+=120_000_000
        flags.update(waiting=True,began=time.monotonic())
        body=response(duration=seconds)
        if fault=='delivery':body['delivery_uncertain']=True
        if fault=='missing_native_floor':body.pop('last_verified_lidar')
        if fault=='controller_restart':
            status=robot.status
            def restarted():
                value=status();value['guarded_controller']['controller_session']='restarted'
                return value
            robot.status=restarted
        return server_error(503,body,endpoint='/motion')
    robot.move_forward=forward
    result=run(r)
    assert not result.get('arrived_at_marvin') and len(motions(events))==1
    assert not result['lidar_recovery_history']


def test_new_emergency_zero_during_wait_is_terminal_not_clearance(tmp_path,monkeypatch):
    bundle,flags,_,_,execute=wait_case(tmp_path,monkeypatch,clear_after=0.)
    status=bundle[2].status
    initial=time.monotonic()
    def emergency_seen():
        value=status()
        if time.monotonic()>initial+.1:
            value['guarded_controller']['last_operation']=4
            value['guarded_controller']['last_request_issued_ns']+=1
        return value
    bundle[2].status=emergency_seen
    result=execute()
    assert not result['ok'] and result['reason']=='find_marvin_interruption_owner_changed'
    assert not motions(bundle[3])


@pytest.mark.parametrize('fault',[None,'operator','missing_geometry','failed_stop','deadline','transport_uncertain'])
def test_simultaneous_local_turn_geometry_stop_needs_both_certificates(tmp_path,monkeypatch,fault):
    from guarded_turn_policy import validate_guarded_turn, ROTATIONAL_SWEPT_FOOTPRINT
    bundle=make_runtime(tmp_path,monkeypatch,[])
    state=bundle[1].lidar()
    state['sectors']={'front':dict(available=True,state='CLEAR')}
    state['local_motion_geometry']['points'].append(dict(x_m=.3,y_m=0.,distance_m=.3,robot_bearing_deg=0.))
    validation=validate_guarded_turn('LEFT',.25,.5,state,expected_session=bundle[0].lidar_worker.session,
                                    target_directed=True,safety_mode=ROTATIONAL_SWEPT_FOOTPRINT)
    assert validation['reason']=='rotational_protected_region_violated'
    parsed,_=certificate(axes=(0.,0.,.25))
    result=dict(generation_invalidated=True,producer_session=bundle[0].lidar_worker.session,
        monitor_reason='rotational_protected_region_violated',monitor_validation=validation,
        transport_began=True,transport_returned=True,transport_error=None,delivery_uncertain=False,
        stop_fallback_result={'ok':True},stop_fallback_error=None,last_stop_error=None,
        stop_events=[{'source':'monitor','result':{'ok':True},'error':None}],transport_result=parsed)
    if fault=='operator':result['stop_events'][0]['source']='operator'
    if fault=='missing_geometry':result['monitor_validation']=None
    if fault=='failed_stop':result['stop_fallback_result']['ok']=False
    if fault=='deadline':result['monitor_reason']='deadline_stop_failed'
    if fault=='transport_uncertain':result['delivery_uncertain']=True
    assert (CognitiveRuntime._marvin_bridge_lidar_interruption(result) is not None) is (fault is None)
