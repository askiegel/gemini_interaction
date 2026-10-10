"""Offline stationary re-sensing: real mission/planner, fake producer/transport."""
import copy
import json
import math
from pathlib import Path
import socket
import time

import pytest

from marvin_blocked_wait import (
    BLOCKED_WAIT_REASONS, MAX_RECHECKS, MAX_STATIONARY_SECONDS,
    blocked_wait_diagnostics, material_route_change, stationary_geometry_epoch,
)
from marvin_route_obstruction import evaluate_marvin_route
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_marvin_avoidance_planner_progress import scan
from test_marvin_lateral_avoidance import LEFT_OPEN, strafe_runtime
from test_marvin_live_proof_continuation import initial
from tracking_state import build_tracking_state


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Blocked-wait tests cannot access robots/network/services")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


ASSOCIATION = {"verified_marvin_distance_m": 1.526799235556347,
               "verified_marvin_conservative_distance_m": 1.4267992355563468,
               "target_bearing_degrees": -4.438659612522205}


def live_points():
    # Reconstruct a 20-return obstruction matching the live route metrics;
    # these are synthetic calibrated points, not claimed raw sensor returns.
    x = .65
    theta = math.radians(ASSOCIATION["target_bearing_degrees"])
    return [(.587055028777684, -.17303581079864946)] + [
        (x + i*.00001, (-.11135881275215963 + math.sin(theta) * (x + i*.00001)) / math.cos(theta))
        for i in range(19)]


def wait_bundle(tmp_path, monkeypatch, *, clear_after=None, change_after=None):
    bundle = make_runtime(tmp_path, monkeypatch, [])
    runtime, behavior, robot, events, clock = bundle
    flags = {"checks": 0, "clear": False, "changed": False, "freeze": False,
             "fault": None, "on_sleep": None, "waits": [], "publishes": []}
    read = behavior.lidar
    def lidar(**kwargs):
        if flags["freeze"]:
            behavior.freeze_lidar = True
        state = scan([] if flags["clear"] else live_points())
        state.update(producer_session=runtime.lidar_worker.session,
                     acquisition_sequence=read()["acquisition_sequence"], effective_age_seconds=0.)
        if flags["changed"]:
            for point in state["local_motion_geometry"]["points"][-20:]:
                point["y_m"] -= .03
        if flags["fault"]:
            flags["fault"](state)
        return state
    runtime.world_model.get_lidar_obstacles = lidar
    entry = lidar()
    association = dict(ASSOCIATION)
    association["route"] = evaluate_marvin_route(entry, association,
        expected_session=runtime.lidar_worker.session)
    diagnostics = blocked_wait_diagnostics()
    diagnostics["blocked_wait_reason"] = "find_marvin_local_avoidance_no_progress"
    history = []
    sleep = time.sleep
    def tick(seconds):
        flags["waits"].append(seconds)
        assert not motions(events)
        assert robot.status()["motion"]["streaming"] is False
        sleep(seconds)
        if flags["on_sleep"]:
            flags["on_sleep"]()
        if clear_after is not None and time.monotonic() >= 100.0 + clear_after:
            flags["clear"] = True
        if change_after is not None and time.monotonic() >= 100.0 + change_after:
            flags["changed"] = True
    monkeypatch.setattr("runtime.time.sleep", tick)
    monkeypatch.setattr(runtime, "_publish_behavior_tracking", lambda d: flags["publishes"].append(copy.deepcopy(d)))
    def execute(guard=lambda: True):
        return runtime._wait_for_marvin_blocked_route(expected_session=runtime.lidar_worker.session,
            lidar=entry, association=association, execution_guard=guard,
            diagnostics=diagnostics, history=history)
    return bundle, flags, diagnostics, history, association, execute


def mission_bundle(tmp_path, monkeypatch, *, clear_on_check=3, specs=None):
    specs = specs or [(0,.60),(0,.60),(0,.60),(0,.50)]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, specs, [LEFT_OPEN])
    r, behavior, robot, events, clock = bundle
    flags = {"waiting": False, "cleared": False, "publishes": [], "checks": 0,
             "on_wait": None, "on_resume": None, "resume_count": 0, "geometry_shift": 0.,
             "waiting_motion_count": None}
    read = r.world_model.get_lidar_obstacles
    def lidar(**kwargs):
        scan = read(**kwargs)
        if flags["cleared"]:
            # Remove only the synthetic foreground/side scene; retain target and scan coverage.
            scan["local_motion_geometry"]["points"] = scan["local_motion_geometry"]["points"][:-len(LEFT_OPEN)]
        elif flags["geometry_shift"]:
            for point in scan["local_motion_geometry"]["points"][-len(LEFT_OPEN):]:
                if point["x_m"] > .4:
                    point["y_m"] += flags["geometry_shift"]
        elif motions(events):
            # Phase WAIT requires an actual unsafe repair/pass corridor, not
            # the obsolete no-route-progress suppression of a safe strafe.
            scan["local_motion_geometry"]["points"].append({"x_m":0.,"y_m":.3})
        return scan
    r.world_model.get_lidar_obstacles = lidar
    publish = r._publish_behavior_tracking
    def telemetry(d):
        flags["publishes"].append(copy.deepcopy(d))
        if d.get("state") == "BLOCKED_WAIT":
            flags["waiting"] = True
            flags["checks"] = d["blocked_wait_recheck_count"]
            if flags["waiting_motion_count"] is None:
                flags["waiting_motion_count"] = len(motions(events))
        publish(d)
    r._publish_behavior_tracking = telemetry
    sleep = time.sleep
    def tick(seconds):
        if flags["waiting"]:
            assert len(motions(events)) == flags["waiting_motion_count"]
            assert robot.status()["motion"]["linear_y"] == 0
            if flags["on_wait"]:
                flags["on_wait"]()
            if clear_on_check is not None and flags["checks"] >= clear_on_check - 1:
                flags["cleared"] = True
        sleep(seconds)
    monkeypatch.setattr("runtime.time.sleep", tick)
    def reacquire(*, minimum_source_frame_stamp_ns):
        flags["waiting"] = False
        flags["resume_count"] += 1
        events.append("blocked_semantic_reacquire")
        if flags["on_resume"]:
            flags["on_resume"]()
        behavior._clear_marvin_v2_tracker_episode()
        evidence = behavior.observe_find_marvin_v2()
        if evidence["preview_result"].get("identity_confirmed"):
            evidence["preview_result"]["strict_tracker_episode"] = {
                "initialized_this_observation": True, "continued_existing_tracker": False}
        return evidence
    behavior.reacquire_find_marvin_v2 = reacquire
    return bundle, flags


def test_live_blocked_clear_geometry_replay_uses_new_scan_without_motion(tmp_path, monkeypatch):
    bundle, flags, d, history, association, execute = wait_bundle(tmp_path, monkeypatch, clear_after=3)
    result = execute()
    assert result["ok"] and result["reason"] == "find_marvin_blocked_wait_route_cleared"
    assert association["route"]["route_occupancy"] == 20
    assert association["route"]["corridor_overlap_m"] == pytest.approx(.33864118724784037)
    assert association["route"]["blocking_obstacle_distance_m"] == pytest.approx(.612025325155678)
    assert result["route"]["route_occupancy"] == 0 and result["route"]["corridor_overlap_m"] == 0
    assert [h["lidar_sequence"] for h in history] == sorted(set(h["lidar_sequence"] for h in history))
    assert history[0]["lidar_sequence"] > d["blocked_wait_initial_lidar_sequence"]
    assert not motions(bundle[3]) and "observe" not in bundle[3]


def test_normal_mission_avoid_wait_forward_then_arrives(tmp_path, monkeypatch):
    bundle, f = mission_bundle(tmp_path, monkeypatch)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert motions(bundle[3]) == [("strafe",.08,1.),("forward",.1,.5)]
    assert result["local_avoidance_actions"] == 1 and result["local_bypass_actions"] == 0
    assert result["blocked_wait_recheck_count"] == 3
    assert len([e for e in bundle[3] if e == "observe"]) == 4
    assert f["resume_count"] == 1
    assert [h["camera_recheck_performed"] for h in result["blocked_wait_history"]] == [False,False,True]
    assert any(d["state"] == "BLOCKED_WAIT" for d in f["publishes"])
    assert all(not row["motion_executed"] and not row["source_stamp_consumed"] for row in result["blocked_wait_history"])
    assert len(bundle[0]._marvin_alignment_consumed_source_frame_stamps) == 2
    assert not result["local_bypass_active"] and result["local_bypass_target_x_m"] is None
    assert result["blocked_wait_resume_reason"] == "find_marvin_blocked_wait_route_cleared"
    forward = [row for row in result["history"] if row["motion_executed"]][-1]
    assert forward["state"] == "ADVANCING" and forward["result"]["full_step_completed"]
    assert forward["observation"]["arrival"]["target_range_association_trusted"]
    assert forward["result"]["stop_result"]["ok"]
    assert any(h["snapshot"]["acquisition_sequence"] > forward["action_lidar_evidence"][1]
               for h in result["lidar_wait_history"])


def test_unchanged_obstruction_exhausts_without_camera_or_motion(tmp_path, monkeypatch):
    b, f, d, history, _, execute = wait_bundle(tmp_path, monkeypatch)
    result = execute()
    assert result["reason"] == "find_marvin_blocked_wait_exhausted"
    assert d["blocked_wait_recheck_count"] == MAX_RECHECKS
    assert d["blocked_wait_total_seconds"] <= MAX_STATIONARY_SECONDS
    assert not motions(b[3]) and b[3].count("observe") == 0
    assert not d["blocked_wait_camera_recheck_performed"]
    intervals = [row["blocked_wait_interval_seconds"] for row in f["publishes"]]
    assert .5 in intervals and 1.0 in intervals and 2.0 in intervals and max(intervals) == 2
    assert not d["blocked_wait_active"]


@pytest.mark.parametrize("fault,reason", [
    ("session","find_marvin_lidar_producer_session_changed"),
    ("stale","find_marvin_lidar_not_current"),
    ("invalid","find_marvin_lidar_not_current"),
    ("geometry","find_marvin_lidar_not_current"),
    ("coverage","find_marvin_blocked_wait_geometry_invalid"),
    ("sequence","find_marvin_lidar_acquisition_sequence_invalid"),
    ("freeze","find_marvin_new_lidar_evidence_timeout"),
])
def test_bad_lidar_fails_closed_before_camera(tmp_path, monkeypatch, fault, reason):
    b, f, d, history, _, execute = wait_bundle(tmp_path, monkeypatch)
    if fault == "freeze": f["freeze"] = True
    else:
        def corrupt(s):
            if fault == "session": s["producer_session"] = "other"
            if fault == "stale": s.update(valid=False,reason="stale")
            if fault == "invalid": s["valid"] = False
            if fault == "geometry": s["local_motion_geometry"]["valid"] = False
            if fault == "coverage": s["local_motion_geometry"]["sectors"]["rear"]["valid_sample_count"] = 0
            if fault == "sequence": s["acquisition_sequence"] = True
        f["fault"] = corrupt
    assert execute()["reason"] == reason
    assert not motions(b[3]) and "observe" not in b[3]


@pytest.mark.parametrize("fault", ["stop_false","stop_exception","bridge_not_ready","bridge_ros","bridge_motion","command_changed"])
def test_stop_bridge_and_command_failures_cannot_resume(tmp_path, monkeypatch, fault):
    b, f, d, history, _, execute = wait_bundle(tmp_path, monkeypatch, clear_after=.5)
    robot=b[2];status=robot.status
    if fault == "stop_false": robot.stop=lambda:{"ok":False}
    if fault == "stop_exception": robot.stop=lambda:(_ for _ in ()).throw(RuntimeError("stop"))
    def bad_status():
        s=status()
        if fault == "bridge_not_ready": s["status"]="STARTING"
        if fault == "bridge_ros": s["ros_ready"]=False
        if fault == "bridge_motion": s["motion"]["linear_x"]=.1
        if fault == "command_changed": s["motion"]["last_command_at"]="new" if f["waits"] else "old"
        return s
    robot.status=bad_status
    assert not execute()["ok"]
    assert not motions(b[3])


def test_preemption_interrupts_backoff_without_camera(tmp_path, monkeypatch):
    b, f, d, history, _, execute = wait_bundle(tmp_path, monkeypatch)
    assert execute(lambda: len(f["waits"]) < 2)["reason"] == "find_marvin_mission_preempted"
    assert not motions(b[3]) and d["blocked_wait_total_seconds"] < .5


def test_cumulative_duration_and_checks_cannot_reset_on_episode_reentry(tmp_path, monkeypatch):
    b,f,d,history,_,execute=wait_bundle(tmp_path,monkeypatch)
    b[0].MARVIN_BLOCKED_WAIT_MAX_SECONDS=.7
    assert execute()["reason"]=="find_marvin_blocked_wait_exhausted"
    assert d["blocked_wait_total_seconds"]==pytest.approx(.7)
    assert execute()["reason"]=="find_marvin_blocked_wait_exhausted"
    assert d["blocked_wait_recheck_count"]==1


def test_proof_harness_still_returns_immediately_on_no_progress(tmp_path, monkeypatch):
    bundle,flags=mission_bundle(tmp_path,monkeypatch,clear_on_check=None)
    first=initial(bundle)
    assert first["controller_result"]["state"]=="PROOF_COMPLETE"
    assert bundle[0].rearm_find_marvin_live_proof(rearm=True)["ok"]
    second=bundle[0].execute_find_marvin_live_proof_step(max_physical_actions=1)
    assert second["controller_result"]["state"]=="BLOCKED"
    assert second["reason"]=="find_marvin_no_safe_local_detour"
    assert len(motions(bundle[3]))==1
    assert not any(d["state"]=="BLOCKED_WAIT" for d in flags["publishes"])


@pytest.mark.parametrize("key,value,changed", [
    ("corridor_overlap_m",.338,False),("corridor_overlap_m",.32,True),
    ("blocking_obstacle_x_m",.59,False),("blocking_obstacle_x_m",.63,True),
    ("route_occupancy",19,False),("valid",False,False),
    ("route_to_marvin_obstructed",False,True),
])
def test_material_geometry_trigger_ignores_noise(key,value,changed):
    before=dict(valid=True,route_to_marvin_obstructed=True,corridor_overlap_m=.338641,
        blocking_obstacle_x_m=.587055,blocking_obstacle_y_m=-.173036,route_occupancy=20)
    assert material_route_change(before,dict(before,**{key:value})) is changed


def test_external_geometry_epoch_selectively_preserves_failed_bypass_and_side():
    before=dict(valid=True,route_to_marvin_obstructed=True,corridor_overlap_m=.33,
        blocking_obstacle_overlap_m=.3,route_occupancy=20,blocking_obstacle_x_m=.6)
    old=dict(action_type="BYPASS_FORWARD",direction="LEFT",route=before,
        ineffective_action_types=["BYPASS_FORWARD","STRAFE_LEFT","STRAFE_RIGHT"],
        post_bypass_lateral_recovery_used=True,
        first_post_action_bypass_progress={"meaningful_progress":False,"meaningful_progress_reason":"no_material_route_improvement"})
    after=dict(before,corridor_overlap_m=.30,blocking_obstacle_overlap_m=.27)
    next_state=stationary_geometry_epoch(old,before,after)
    assert next_state["ineffective_action_types"]==old["ineffective_action_types"]
    assert next_state["direction"]=="LEFT" and next_state["post_bypass_lateral_recovery_used"]
    assert next_state["stationary_geometry_epoch"]["entry_route"]==before
    assert next_state["first_post_action_bypass_progress"]==old["first_post_action_bypass_progress"]
    assert old["post_bypass_lateral_recovery_used"] is True and "STRAFE_LEFT" in old["ineffective_action_types"]
    assert stationary_geometry_epoch(old,before,before) is old


def test_blocked_wait_telemetry_survives_tracking_state():
    d=blocked_wait_diagnostics();d.update(blocked_wait_active=True,blocked_wait_recheck_count=2)
    t=build_tracking_state({"behavior":"FIND_OBJECT","target":"marvin","state":"BLOCKED_WAIT",**d})
    assert t["active"] and t["state"]=="BLOCKED_WAIT"
    assert all(t[k]==v for k,v in d.items())


@pytest.mark.parametrize("resume", ["arrived","identity_loss","replayed_stamp"])
def test_resume_arrival_and_fresh_identity_failures(tmp_path,monkeypatch,resume):
    specs=[(0,.6),(0,.6)] + ([(0,.5)] if resume=="arrived" else ["lost"]*4 if resume=="identity_loss" else [(0,.6)])
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=1,specs=specs)
    if resume=="replayed_stamp": f["on_resume"]=lambda:setattr(bundle[1],"repeat_camera",True)
    result=run(bundle[0])
    assert len(motions(bundle[3]))==1
    if resume=="arrived": assert result["state"]=="ARRIVED"
    else: assert result["state"]=="REVERIFY_REQUIRED"
    assert result["max_local_avoidance_actions"]==6
    assert result["local_avoidance_actions"]==1 and result["local_bypass_actions"]==0


def test_changed_obstacle_resumes_real_selector_on_established_side(tmp_path,monkeypatch):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=None,
        specs=[(0,.6),(0,.6),(0,.6),(0,.5)])
    f["on_wait"]=lambda:f.update(geometry_shift=-.04)
    bundle[2].on_motion=lambda:f.update(cleared=True) if len(motions(bundle[3]))==2 else None
    result=run(bundle[0])
    assert result["state"]=="ARRIVED"
    assert motions(bundle[3])==[("strafe",.08,1.),("strafe",.08,1.)]
    assert result["local_avoidance_actions"]==2 and result["local_bypass_actions"]==0
    assert result["blocked_wait_history"][0]["geometry_changed"]
    assert result["blocked_wait_history"][0]["route"]["route_to_marvin_obstructed"]
    assert result["local_avoidance_history"][-1]["selection"]["phase"] == "CLEAR_SIDE"
    assert result["last_detour_direction"]=="LEFT"


def test_stationary_epoch_must_still_pass_current_prediction_and_jit_geometry():
    from test_marvin_post_bypass_recovery import live_geometry, select
    f,state,old=live_geometry()
    old["post_bypass_lateral_recovery_used"]=True
    entry=copy.deepcopy(f["expected_route"])
    entry["corridor_overlap_m"]+=.03
    entry["blocking_obstacle_overlap_m"]+=.03
    assert select(state,f["association"],old)["action_type"] is None
    reconsidered=stationary_geometry_epoch(old,entry,f["expected_route"])
    useful=select(state,f["association"],reconsidered)
    assert useful["action_type"]=="STRAFE_LEFT" and useful["post_bypass_lateral_recovery_used"] is True
    assert "BYPASS_FORWARD" in useful["ineffective_action_types"]
    assert useful["actual_route_progress"]==old["first_post_action_bypass_progress"]
    # A geometry epoch is not a permission if the current JIT scene regresses.
    reconsidered["stationary_geometry_epoch"]["entry_route"]=copy.deepcopy(f["expected_route"])
    assert select(state,f["association"],reconsidered)["action_type"] is None


@pytest.mark.parametrize("reason", sorted(BLOCKED_WAIT_REASONS) + [
    "find_marvin_local_avoidance_exhausted", "marvin_lateral_transport_failed",
    "marvin_alignment_turn_exception", "invalid_lidar_geometry"])
def test_only_safe_pre_dispatch_dead_ends_enter_wait(tmp_path,monkeypatch,reason):
    from marvin_obstacle_phases import plan_phase_action
    bundle,flags=mission_bundle(tmp_path,monkeypatch,clear_on_check=None,specs=[(0,.6)])
    def reject(*a,**kw):
        # A decision-only failure is never a primitive injection or dispatch.
        result=plan_phase_action(*a,**kw)
        return dict(result,direction=None,action_type=None,reason=reason)
    monkeypatch.setattr("runtime.plan_phase_action",reject)
    result=run(bundle[0])
    assert not motions(bundle[3])
    if reason in BLOCKED_WAIT_REASONS:
        assert result["reason"]=="find_marvin_blocked_wait_exhausted"
        assert result["blocked_wait_reason"]==reason and result["blocked_wait_recheck_count"]==12
    else:
        assert result["reason"]==reason and result["blocked_wait_recheck_count"]==0


def test_runtime_time_backstop_is_safe_incomplete_and_preserves_budget(tmp_path,monkeypatch):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=None)
    bundle[0].MARVIN_BLOCKED_WAIT_MAX_SECONDS=1.1
    result=run(bundle[0])
    assert result["reason"]=="find_marvin_blocked_wait_exhausted"
    assert result["mission_outcome"]=="safe_incomplete"
    assert result["blocked_wait_total_seconds"]==pytest.approx(1.1)
    assert result["actions_executed"]==result["local_avoidance_actions"]==1
    assert len(bundle[0]._marvin_alignment_consumed_source_frame_stamps)==1
    assert f["resume_count"]==0 and bundle[3].count("observe")==2


def test_new_exact_camera_stamp_and_new_lidar_required_after_wait(tmp_path,monkeypatch):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=1)
    bundle[1].source_offset_ns=1791501698626664209-bundle[4][0]
    result=run(bundle[0]);moving=[row for row in result["history"] if row["motion_executed"]]
    decision_only=[row for row in result["history"] if row.get("decision_only")]
    old_stamp=decision_only[-1]["source_frame_stamp_ns"];new=moving[-1]["source_frame_stamp_ns"]
    assert type(old_stamp) is type(new) is int and new>old_stamp>moving[0]["source_frame_stamp_ns"]
    assert old_stamp not in bundle[0]._marvin_alignment_consumed_source_frame_stamps
    assert moving[-1]["action_lidar_evidence"][1]>result["blocked_wait_history"][-1]["lidar_sequence"]
    assert moving[-1]["result"]["source_stamp_consumed"] and moving[-1]["result"]["stop_result"]["ok"]


def test_materially_unchanged_resume_cannot_erase_ineffective_memory():
    from test_marvin_post_bypass_recovery import live_geometry
    f,state,old=live_geometry()
    assert stationary_geometry_epoch(old,f["expected_route"],f["expected_route"]) is old
    worse=copy.deepcopy(f["expected_route"]);worse["corridor_overlap_m"]+=.03
    assert stationary_geometry_epoch(old,f["expected_route"],worse) is old


def test_changed_scan_alone_is_not_identity_or_motion_authority(tmp_path,monkeypatch):
    b,f,d,history,association,execute=wait_bundle(tmp_path,monkeypatch,change_after=.5)
    result=execute()
    assert result["ok"] and result["reason"]=="find_marvin_blocked_wait_geometry_changed"
    assert result["route"]["route_to_marvin_obstructed"]
    assert not motions(b[3]) and not b[0]._marvin_alignment_consumed_source_frame_stamps
    assert b[0]._marvin_alignment_observation is None and not d["blocked_wait_camera_recheck_performed"]


def test_retained_route_with_explicit_unsafe_corridor_waits_then_forward(tmp_path,monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_blocked_clear_evidence.json').read_text())
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,
        [(48,1.4267992355563468)]*12+[(48,1.3767992355563468)], [live_points()])
    r,behavior,robot,events,clock=bundle
    flags={"waiting":False,"removed":False,"checks":0,"wait_motion_count":None}
    read=r.world_model.get_lidar_obstacles
    def lidar(**kw):
        state=read(**kw)
        # Calibrate the synthetic Marvin surface to the live +48 px bearing;
        # the generic Perception fixture centers its returns at y=0.
        for point in state['local_motion_geometry']['points'][-33:-20]:
            point['y_m'] -= (point['x_m'] - .081299) * (368.-321.6103934690693) / 597.6149561338204
        if flags["removed"]:
            state['local_motion_geometry']['points']=state['local_motion_geometry']['points'][:-20]
        elif motions(events):
            # Explicit synthetic rear hazard blocks native repair/pass safety
            # while preserving the recorded forward route summary. This is
            # not claimed as a historical sensor return.
            state['local_motion_geometry']['points'].append({'x_m':-.1,'y_m':.3})
        return state
    r.world_model.get_lidar_obstacles=lidar
    publish=r._publish_behavior_tracking
    def telemetry(d):
        if d.get('state')=='BLOCKED_WAIT':
            flags.update(waiting=True,checks=d['blocked_wait_recheck_count'])
            if flags['wait_motion_count'] is None: flags['wait_motion_count']=len(motions(events))
        publish(d)
    r._publish_behavior_tracking=telemetry
    sleep=time.sleep
    def tick(seconds):
        if flags['waiting']:
            assert len(motions(events))==flags['wait_motion_count']
            if flags['checks']>=1: flags['removed']=True
        sleep(seconds)
    monkeypatch.setattr('runtime.time.sleep',tick)
    def semantic(*,minimum_source_frame_stamp_ns):
        flags['waiting']=False
        behavior._clear_marvin_v2_tracker_episode()
        evidence=behavior.observe_find_marvin_v2()
        assert evidence['preview_result']['identity_source_frame_stamp_ns']>minimum_source_frame_stamp_ns
        return evidence
    behavior.reacquire_find_marvin_v2=semantic
    # End this offline replay after the first resumed forward and its newer
    # observation, rather than inventing an immediate 0.9 m jump to standoff.
    behavior.on_observe=lambda:setattr(r,'_control_generation',r._control_generation+1) if (
        flags['removed'] and flags['wait_motion_count'] is not None
        and len(motions(events))>flags['wait_motion_count']) else None
    captured={}
    execute=r._execute_normal_marvin_find_mission
    def capture(*a,**kw):
        captured['result']=execute(*a,**kw)
        return captured['result']
    r._execute_normal_marvin_find_mission=capture
    run(r)  # The executor correctly discards reporting from a preempted generation.
    result=captured['result']
    assert result.get('state')=='STOPPED', (result.get('reason'),motions(events))
    assert result['blocked_wait_recheck_count']==2
    first=result['blocked_wait_history'][0]['route'];last=result['blocked_wait_history'][-1]['semantic_route']
    assert first['route_occupancy']==fixture['blocked']['arrival']['route']['route_occupancy']==20
    assert first['corridor_overlap_m']==pytest.approx(.338641,abs=.001)
    assert first['blocking_obstacle_distance_m']==pytest.approx(.612025,abs=.001)
    assert not last['route_to_marvin_obstructed'] and last['route_occupancy']==0
    assert len(motions(events))==flags['wait_motion_count']+1
    assert motions(events)[-1]==('forward',.1,.5)
    assert all(not h['motion_executed'] and not h['source_stamp_consumed'] for h in result['blocked_wait_history'])
    assert [h['camera_recheck_performed'] for h in result['blocked_wait_history']]==[False,True]
    forward=[h for h in result['history'] if h['motion_executed']][-1]
    assert forward['state']=='ADVANCING' and forward['result']['full_step_completed']
    assert result['stop_result']['ok'] and result['final_observation']['identity_confirmed']
    assert result['final_observation']['source_frame_stamp_ns']>forward['source_frame_stamp_ns']
    replay={"entry":"BLOCKED_WAIT","entry_motion_count":flags['wait_motion_count'],
        "stationary_rechecks":result['blocked_wait_recheck_count'],
        "entry_route":first,"cleared_route":last,"motions":motions(events),
        "resumed_primitive":"FORWARD","resumed_command":{"linear_x":.1,"duration":.5},
        "camera_rechecks":[h['camera_recheck_performed'] for h in result['blocked_wait_history']],
        "wait_motion_count":0,"post_action_camera_stamp":result['final_observation']['source_frame_stamp_ns'],
        "action_stamp":forward['source_frame_stamp_ns'],"stop_confirmed":result['stop_result']['ok'],
        "terminal_state":result['state'],"terminal_note":"Offline test preempts after the first resumed FORWARD and its newer camera evidence."}
    (tmp_path/'blocked-clear-replay-summary.json').write_text(json.dumps(replay,indent=2)+'\n')


@pytest.mark.parametrize("fault", ["camera_exception","session_changed","lidar_invalid"])
def test_resume_revalidates_health_and_does_not_retry_arbitrary_exception(tmp_path,monkeypatch,fault):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=1)
    def fail():
        if fault=='camera_exception': raise RuntimeError('camera transport')
        if fault=='session_changed': bundle[0].lidar_worker.session='other-producer'
        if fault=='lidar_invalid': bundle[1].invalid_lidar=True
    f['on_resume']=fail
    result=run(bundle[0])
    assert result['state'] in {'BLOCKED','REVERIFY_REQUIRED'}
    assert len(motions(bundle[3]))==1 and f['resume_count']==1
    assert len(bundle[0]._marvin_alignment_consumed_source_frame_stamps)==1
    assert result['stop_result']['ok']


def test_live_status_projection_exposes_stationary_wait_diagnostics(tmp_path,monkeypatch):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=None)
    seen=[]
    f['on_wait']=lambda:seen.append(bundle[0].get_status_summary()['tracking'])
    result=run(bundle[0])
    assert result['reason']=='find_marvin_blocked_wait_exhausted'
    assert seen and all(d['state']=='BLOCKED_WAIT' and d['blocked_wait_active'] for d in seen)
    assert all(d['blocked_wait_reason']=='find_marvin_no_safe_local_detour' for d in seen)
    assert all(k in seen[-1] for k in blocked_wait_diagnostics())


def test_production_stop_failure_returns_safe_failure_without_waiting_or_retry(tmp_path,monkeypatch):
    bundle,f=mission_bundle(tmp_path,monkeypatch,clear_on_check=None)
    stops=bundle[2].stop
    def fail_after_action():
        if len(motions(bundle[3]))==1 and bundle[3].count('observe')>=2: return {'ok':False}
        return stops()
    bundle[2].stop=fail_after_action
    result=run(bundle[0])
    assert result['state']=='BLOCKED' and result['mission_outcome']=='safe_failure'
    assert len(motions(bundle[3]))==1 and result['blocked_wait_recheck_count']==0
