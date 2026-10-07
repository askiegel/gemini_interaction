"""Offline proof boundary: production loop/safety with mocked sensors/transport."""
import json
import copy
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler, _precision_safe_source_frame_stamps
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_find_marvin_local_obstacle_avoidance import avoidance_runtime
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_marvin_local_bypass import OPEN_LEFT
from marvin_local_obstacle_avoidance import select_marvin_escape_action
from marvin_target_range_association import MarvinTargetRangeAssociation


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Proof tests must not access robot/network/services")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


def proof(runtime):
    runtime._set_runtime_state("IDLE")
    return runtime.execute_find_marvin_live_proof_step(max_physical_actions=1)


def assert_complete(runtime, result, events, expected_motion):
    assert motions(events) == [expected_motion]
    assert result["actions_executed"] == 1 and result["motion_executed"] is True
    c = result["controller_result"]
    assert c["state"] == "PROOF_COMPLETE", c["reason"]
    p = c["proof"]
    assert p["action_complete"] and p["physical_dispatches_or_uncertain"] == 1
    assert p["dispatch_opportunities"] == 1
    row = c["history"][0]
    evidence = p["post_action_evidence"]
    assert evidence["lidar_wait"]["ok"]
    assert evidence["lidar_wait"]["snapshot"]["acquisition_sequence"] > row["action_lidar_evidence"][1]
    assert evidence["observation"]["source_frame_stamp_ns"] > row["source_frame_stamp_ns"]
    assert p["source_stamps"] == [{"source_frame_stamp_ns": row["source_frame_stamp_ns"], "consumed": True}]
    assert c["stop_result"]["ok"] and c["bridge_after_stop"]["status"] == "READY"
    assert runtime.mission_manager.get_active_mission() is None
    assert runtime.mission_manager.get_history() == [] and runtime.mission_manager.get_queue() == []
    assert runtime.get_status()["runtime_state"] == "IDLE"
    assert runtime._marvin_live_proof_owner is None


def test_clear_route_one_forward_then_new_evidence_without_action_two(tmp_path, monkeypatch):
    r, b, robot, events, clock = make_runtime(tmp_path, monkeypatch, [(0,.8),(0,.75),(0,.7)])
    result = proof(r)
    assert_complete(r, result, events, ("forward",.1,.5))
    assert len(b.stamps) == 2
    assert result["controller_result"]["local_avoidance_actions"] == 0


@pytest.mark.parametrize("scene,expected", [(LEFT_OPEN,("strafe",.08,1.)),
    ([ (x,-y) for x,y in LEFT_OPEN ],("strafe",-.08,1.))])
def test_obstructed_route_one_strafe(tmp_path, monkeypatch, scene, expected):
    bundle, _, _ = strafe_runtime(tmp_path,monkeypatch,[(0,.8),(0,.8),(0,.8)],[scene,None])
    assert_complete(bundle[0],proof(bundle[0]),bundle[3],expected)


def test_obstructed_route_one_existing_guarded_turn(tmp_path,monkeypatch):
    r,_,_,events,_=avoidance_runtime(tmp_path,monkeypatch,[(0,.8),(0,.8),(0,.8)],[(1.2,.48),None])
    result=proof(r)
    assert_complete(r,result,events,("turn","LEFT",.25,.5))
    assert result["controller_result"]["history"][0]["result"]["action"] == "single_marvin_local_detour_turn"


def established_bypass(tmp_path,monkeypatch):
    # Existing regression geometry: lateral translation is unsafe at a rear
    # corner while the established LEFT forward corridor is independently clear.
    return strafe_runtime(tmp_path,monkeypatch,[(0,1.1),(0,1.1),(0,1.1)],
        [OPEN_LEFT+[(-.08,.452)],None])


def test_established_left_bypass_uses_existing_guarded_forward(tmp_path,monkeypatch):
    bundle,_,_=established_bypass(tmp_path,monkeypatch)
    r,_,_,events,_=bundle
    executor=r.behavior_manager.execute_single_marvin_approach_step
    r.behavior_manager.execute_single_marvin_approach_step=Mock(wraps=executor)
    result=proof(r)
    assert_complete(r,result,events,("forward",.1,.5))
    c=result["controller_result"]; row=c["history"][0]
    assert row["result"]["action_type"] == "BYPASS_FORWARD"
    assert c["local_avoidance_actions"] == c["completed_bypass_forward_actions"] == 1
    r.behavior_manager.execute_single_marvin_approach_step.assert_called_once()
    kw=r.behavior_manager.execute_single_marvin_approach_step.call_args.kwargs
    assert kw["linear_speed"]==.1 and kw["duration"]==.5
    assert callable(kw["local_selection_validator"])
    assert row["result"]["local_detour"]["local_bypass"]["protected_radius_m"] == .45


@pytest.mark.parametrize("fault",["stale","session","coverage","capsule"])
def test_jit_bypass_fault_zero_motion_consumed_stamp_not_reusable(tmp_path,monkeypatch,fault):
    bundle,_,_=established_bypass(tmp_path,monkeypatch)
    r,_,_,events,_=bundle
    execute=r.behavior_manager.execute_single_marvin_approach_step
    read=r.world_model.get_lidar_obstacles
    def veto(**kw):
        def faulty(**opts):
            scan=read(**opts)
            if fault=="stale": scan["received_monotonic_seconds"]-=.301
            if fault=="session": scan["producer_session"]="wrong"
            if fault=="coverage": scan["local_motion_geometry"]["sectors"]["rear"]["valid_sample_count"]=0
            if fault=="capsule": scan["local_motion_geometry"]["points"].append({"x_m":.4,"y_m":.2})
            return scan
        r.world_model.get_lidar_obstacles=faulty
        return execute(**kw)
    r.behavior_manager.execute_single_marvin_approach_step=veto
    result=proof(r); c=result["controller_result"]
    assert motions(events)==[] and result["actions_executed"]==0
    assert not c["proof"]["action_complete"]
    assert c["proof"]["source_stamps"][0]["consumed"]
    assert c["proof"]["physical_dispatches_or_uncertain"]==0
    assert r.execute_find_marvin_live_proof_step(max_physical_actions=1)["reason"]=="marvin_live_proof_already_consumed"


@pytest.mark.parametrize("fault",["stale","session","invalid"])
def test_initial_lidar_failure_zero_motion(tmp_path,monkeypatch,fault):
    r,_,_,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8)])
    read=r.world_model.get_lidar_obstacles
    def bad(**kw):
        scan=read(**kw)
        if fault=="stale": scan.update(valid=False,reason="stale",effective_age_seconds=.31)
        if fault=="session": scan["producer_session"]="wrong"
        if fault=="invalid": scan["local_motion_geometry"]["valid"]=False
        return scan
    r.world_model.get_lidar_obstacles=bad
    result=proof(r)
    assert not result["controller_result"]["proof"]["action_complete"] and motions(events)==[]


@pytest.mark.parametrize("fault",["duplicate_camera","no_new_lidar","stop","stop_unconfirmed"])
def test_post_action_requirements_fail_without_second_action(tmp_path,monkeypatch,fault):
    r,b,robot,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75),(0,.7)])
    def after():
        if fault=="duplicate_camera": b.repeat_camera=True
        if fault=="no_new_lidar": b.freeze_lidar=True
        if fault=="stop": r.submit_intent({"intent":"STOP"})
        if fault=="stop_unconfirmed": robot.ready=False
    robot.on_motion=after
    result=proof(r);c=result["controller_result"]
    assert motions(events)==[("forward",.1,.5)] and not c["proof"]["action_complete"]
    assert c["state"] != "PROOF_COMPLETE"
    if fault=="duplicate_camera": assert c["reason"]=="find_marvin_new_camera_frame_required"
    if fault=="no_new_lidar": assert c["reason"]=="find_marvin_new_lidar_evidence_timeout"
    if fault=="stop": assert c["state"]=="STOPPED" and r.get_status()["runtime_state"]=="STOPPED"


def test_exclusive_owner_never_registers_or_overlaps_missions(tmp_path,monkeypatch):
    r,b,robot,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75)])
    observed=[]
    def inspect():
        assert r.mission_manager.get_active_mission() is None
        assert r.run_once() is None
        with pytest.raises(ValueError,match="marvin_live_proof_owns_runtime"):
            r.submit_intent({"intent":"FIND_OBJECT","target":"marvin"})
        thread=threading.Thread(target=lambda: observed.append(r._marvin_motion_owner_is_current()))
        thread.start();thread.join(2)
        assert not thread.is_alive()
    b.on_observe=inspect
    result=proof(r)
    assert result["actions_executed"]==1 and observed==[False,False]


def test_busy_mission_refuses_proof_without_consuming_guard(tmp_path,monkeypatch):
    r,_,_,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8)])
    r.submit_intent({"intent":"FIND_OBJECT","target":"marvin"})
    result=r.execute_find_marvin_live_proof_step(max_physical_actions=1)
    assert result["reason"]=="marvin_live_proof_runtime_not_idle"
    assert not r._marvin_live_proof_consumed and motions(events)==[]


def test_normal_mission_keeps_multiple_actions_and_six_budget(tmp_path,monkeypatch):
    r,_,_,events,_=make_runtime(tmp_path,monkeypatch,[(0,.60),(0,.59),(0,.5)])
    result=run(r)
    assert result["state"]=="ARRIVED" and len(motions(events))==2
    assert result["max_local_avoidance_actions"]==6 and "proof" not in result
    assert not r._marvin_live_proof_consumed


def test_exact_nanosecond_consumption_and_one_shot(tmp_path,monkeypatch):
    r,b,_,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75)])
    b.source_offset_ns=9876543210123456789
    result=proof(r); c=result["controller_result"]
    stamp=c["history"][0]["source_frame_stamp_ns"]
    assert type(stamp) is int and stamp>2**53
    assert stamp in r._marvin_alignment_consumed_source_frame_stamps
    payload=_precision_safe_source_frame_stamps(result)
    assert payload["controller_result"]["proof"]["source_stamps"][0]["source_frame_stamp_ns"]==str(stamp)
    assert int(json.loads(json.dumps(payload))["controller_result"]["history"][0]["source_frame_stamp_ns"])==stamp
    second=r.execute_find_marvin_live_proof_step(max_physical_actions=1)
    assert second["reason"]=="marvin_live_proof_already_consumed" and len(motions(events))==1


def test_retained_a6c4d675_established_side_through_proof_loop(tmp_path,monkeypatch):
    fixture=json.loads((Path(__file__).parent/"test_fixtures/marvin_a6c4d675_bypass_geometry.json").read_text())
    row=fixture["cycles"][-1]
    assert row["cycle"]==6 and fixture["mission_id"]=="mission-a6c4d675"
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(40,1.38),(40,1.38),(40,1.38)],[None])
    r,b,robot,events,clock=bundle
    # Replay the retained continuation's historical identity/range and prior
    # selection, not five extra physical actions. The public API exposes none
    # of these inputs. Current geometry, selector and every JIT guard remain real.
    historical=copy.deepcopy(row["previous_selection"])
    def selector(scan,association,**kw):
        assert scan["producer_session"]==r.lidar_worker.session
        return select_marvin_escape_action(scan,association,
            **dict(kw,previous_selection=historical))
    monkeypatch.setattr("runtime.select_marvin_escape_action",selector)
    def association_state():
        state=MarvinTargetRangeAssociation()
        state.anchor={"measured_distance_m":row["association"]["verified_marvin_distance_m"],
            "target_distance_m":row["association"]["verified_marvin_conservative_distance_m"],
            "producer_session":r.lidar_worker.session,"translation_bound_m":0.}
        state.initial_anchor=dict(state.anchor)
        return state
    monkeypatch.setattr("runtime.MarvinTargetRangeAssociation",association_state)
    def scan(**kw):
        b.sequence+=1
        clock[0]+=1000
        state=copy.deepcopy(row["lidar"])
        state.update(producer_session=r.lidar_worker.session,acquisition_sequence=b.sequence,
            received_monotonic_seconds=time.monotonic(),age_at_receipt_seconds=0.,
            effective_age_seconds=0.,reason="fresh")
        return state
    r.world_model.get_lidar_obstacles=scan
    result=proof(r)
    assert_complete(r,result,events,("forward",.1,.5))
    selection=result["controller_result"]["history"][0]["result"]["local_detour"]
    assert selection["action_type"]=="BYPASS_FORWARD" and selection["direction"]=="LEFT"
    assert selection["route"]["route_to_marvin_obstructed"]
    assert selection["local_bypass"]["bypass_corridor_occupancy"]==0


@pytest.mark.parametrize("kind",["forward","strafe","turn"])
def test_transport_uncertainty_consumes_opportunity_and_stamp_without_retry(tmp_path,monkeypatch,kind):
    if kind=="strafe":
        bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8),(0,.8)],[LEFT_OPEN])
    else:
        bundle=make_runtime(tmp_path,monkeypatch,[(100 if kind=="turn" else 0,.8),(0,.8)])
    r,b,robot,events,_=bundle
    def failed(*args,**kwargs):
        events.append(("uncertain",kind))
        raise TimeoutError("mocked delivery cannot be confirmed")
    if kind=="strafe": robot.move_lateral=failed
    elif kind=="forward": robot.move_forward=failed
    else: b._execute_target_directed_turn=failed
    result=proof(r);c=result["controller_result"]
    assert motions(events)==[("uncertain",kind)]
    assert c["proof"]["dispatch_opportunities"]==1
    assert c["proof"]["physical_dispatches_or_uncertain"]==1
    assert c["proof"]["source_stamps"][0]["consumed"]
    assert not c["proof"]["action_complete"] and result["actions_executed"]==0
    assert c["stop_result"]["ok"] and c["bridge_after_stop"]["status"]=="READY"
    assert r.execute_find_marvin_live_proof_step(max_physical_actions=1)["reason"]=="marvin_live_proof_already_consumed"


def test_nested_second_dispatch_is_refused_by_owner_before_transport(tmp_path,monkeypatch):
    r,b,robot,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75)])
    second=[]
    def after_first():
        if motions(events):
            second.append(r.execute_single_marvin_alignment(direction="LEFT",
                angular_speed=.25,duration=.5,source_frame_stamp_ns=b.stamps[-1]))
    b.on_observe=after_first
    result=proof(r)
    assert_complete(r,result,events,("forward",.1,.5))
    assert second[0]["reason"]=="marvin_live_proof_action_limit_reached"
    assert second[0]["execution_authorized"] is False


def test_stale_lateral_interruption_recovers_evidence_without_resuming(tmp_path,monkeypatch):
    bundle,flags,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8),(0,.8),(0,.8)],
        [LEFT_OPEN,None],interruptions=(True,))
    r,_,_,events,_=bundle
    result=proof(r); c=result["controller_result"]
    assert motions(events)==[("strafe",.08,1.)]
    assert c["state"]=="PROOF_INTERRUPTED" and not c["proof"]["action_complete"]
    assert c["proof"]["physical_dispatches_or_uncertain"]==1
    assert c["proof"]["source_stamps"][0]["consumed"]
    assert c["history"][0]["result"]["interrupted"]
    evidence=c["proof"]["post_action_evidence"]
    assert evidence["lidar_wait"]["snapshot"]["acquisition_sequence"]>c["history"][0]["action_lidar_evidence"][1]
    assert evidence["observation"]["source_frame_stamp_ns"]>c["history"][0]["source_frame_stamp_ns"]
    assert c["bridge_after_stop"]["status"]=="READY"


@pytest.mark.parametrize("failure",["malformed_action","action_exception","post_observe_exception"])
def test_unexpected_failure_retains_consumed_stamp_and_uncertain_accounting(tmp_path,monkeypatch,failure):
    r,b,robot,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75)])
    action=r.execute_single_marvin_approach
    def failed_action(**kw):
        action(**kw)
        if failure=="action_exception": raise RuntimeError("mocked result lost after dispatch")
        return None
    if failure=="post_observe_exception":
        observe=r.observe_find_marvin_v2
        def failed_observe():
            if motions(events): raise RuntimeError("mocked post-action camera exception")
            return observe()
        r.observe_find_marvin_v2=failed_observe
    else:
        r.execute_single_marvin_approach=failed_action
    result=proof(r); c=result["controller_result"]
    assert motions(events)==[("forward",.1,.5)]
    assert not c["proof"]["action_complete"]
    assert c["proof"]["physical_dispatches_or_uncertain"]==1
    assert c["proof"]["source_stamps"]==[{"source_frame_stamp_ns":b.stamps[0],"consumed":True}]
    assert c["stop_result"]["ok"] and c["bridge_after_stop"]["status"]=="READY"
    assert r._marvin_live_proof_owner is None and r._behavior_execution_generation is None
    assert r.execute_find_marvin_live_proof_step(max_physical_actions=1)["reason"]=="marvin_live_proof_already_consumed"


@pytest.mark.parametrize("fault",["quality","identity"])
def test_invalid_post_action_tracker_cannot_claim_completion(tmp_path,monkeypatch,fault):
    r,b,robot,events,_=make_runtime(tmp_path,monkeypatch,[(0,.8),(0,.75)])
    observe=r.observe_find_marvin_v2
    def current():
        result=observe()
        if motions(events):
            if fault=="quality": result["opencv_tracker"]["quality"]=.786
            if fault=="identity": result["identity_confirmed"]=False
        return result
    r.observe_find_marvin_v2=current
    result=proof(r)
    assert motions(events)==[("forward",.1,.5)]
    assert not result["controller_result"]["proof"]["action_complete"]


@pytest.mark.parametrize("body",[{}, {"max_physical_actions":0}, {"max_physical_actions":2},
    {"max_physical_actions":True}, {"max_physical_actions":1.0}, {"max_physical_actions":"1"},
    {"max_physical_actions":1,"direction":"LEFT"}, {"max_physical_actions":1,"speed":.1}])
def test_api_rejects_unbounded_or_caller_selected_requests(body):
    method=Mock(return_value={"ok":True})
    handler=object.__new__(RuntimeAPIHandler)
    handler.path="/find-marvin/live-proof-step"
    handler.server=SimpleNamespace(runtime=SimpleNamespace(execute_find_marvin_live_proof_step=method))
    handler.require_json_request=lambda:body
    responses=[];handler.send_json=lambda code,payload:responses.append((code,payload))
    handler.do_POST()
    assert responses[-1][0]==400 and not method.called


def test_api_accepts_only_the_exact_one_action_contract():
    method=Mock(return_value={"ok":True})
    h=object.__new__(RuntimeAPIHandler);h.path="/find-marvin/live-proof-step"
    h.server=SimpleNamespace(runtime=SimpleNamespace(execute_find_marvin_live_proof_step=method))
    h.require_json_request=lambda:{"max_physical_actions":1}
    responses=[];h.send_json=lambda code,payload:responses.append((code,payload))
    h.do_POST();method.assert_called_once_with(max_physical_actions=1)
    assert responses==[(200,{"ok":True})]
