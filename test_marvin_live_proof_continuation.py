"""Offline re-arm/continuation tests using the real loop, selectors and guards."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime import _marvin_proof_checkpoint_digest
from runtime_api import RuntimeAPIHandler, _precision_safe_source_frame_stamps
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN, RIGHT_OPEN
from test_marvin_local_bypass import OPEN_LEFT


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Continuation tests must not access robot/network/services")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


def initial(bundle):
    behavior = bundle[1]
    if "reacquire_find_marvin_v2" not in behavior.__dict__:
        # The offline Perception fixture supplies controlled frames rather
        # than a camera client. Model the existing stopped recovery contract.
        def reacquire(*, minimum_source_frame_stamp_ns):
            behavior._clear_marvin_v2_tracker_episode()
            evidence = behavior.observe_find_marvin_v2()
            preview = evidence["preview_result"]
            if preview.get("identity_confirmed") is True:
                preview["strict_tracker_episode"] = {
                    "initialized_this_observation": True,
                    "continued_existing_tracker": False,
                }
            return evidence
        behavior.reacquire_find_marvin_v2 = reacquire
    bundle[0]._set_runtime_state("IDLE")
    return step(bundle[0])


def step(runtime):
    return runtime.execute_find_marvin_live_proof_step(max_physical_actions=1)


def arm(runtime):
    return runtime.rearm_find_marvin_live_proof(rearm=True)


def forward_bundle(tmp_path, monkeypatch):
    return make_runtime(tmp_path, monkeypatch,
        [(0, .80), (0, .75), (0, .75), (0, .70), (0, .70), (0, .65)])


def complete(bundle, result, before_count):
    runtime, behavior, robot, events, _ = bundle
    c = result["controller_result"]
    assert c["state"] == "PROOF_COMPLETE", c["reason"]
    assert result["proof_state"] == "COMPLETE_DISARMED"
    assert result["continuation_available"]
    assert result["actions_executed"] == 1
    assert len(motions(events)) == before_count + 1
    assert len(c["history"]) == 1
    p = c["proof"]
    assert p["max_physical_actions"] == p["dispatch_opportunities"] == p["physical_dispatches_or_uncertain"] == 1
    assert p["action_complete"]
    row = c["history"][0]
    assert row["result"].get("source_stamp_consumed",
        row["source_frame_stamp_ns"] in runtime._marvin_alignment_consumed_source_frame_stamps) is True
    assert row["result"].get("full_step_completed", row["result"].get("ok")) is True
    assert p["source_stamps"] == [{"source_frame_stamp_ns": row["source_frame_stamp_ns"], "consumed": True}]
    evidence = p["post_action_evidence"]
    assert evidence["lidar_wait"]["ok"]
    scan = evidence["lidar_wait"]["snapshot"]
    assert scan["producer_session"] == row["action_lidar_evidence"][0]
    assert scan["acquisition_sequence"] > row["action_lidar_evidence"][1]
    assert evidence["observation"]["source_frame_stamp_ns"] > row["source_frame_stamp_ns"]
    tracker = evidence["observation"]["opencv_tracker"]
    assert tracker["matched"] and tracker["quality"] >= max(.8, tracker["threshold"])
    assert c["stop_result"]["ok"] and c["bridge_after_stop"]["status"] == "READY"
    assert robot.status()["motion"]["streaming"] is False
    assert runtime._marvin_live_proof_owner is None
    assert runtime.get_status()["runtime_state"] == "IDLE"
    assert runtime.mission_manager.get_active_mission() is None
    assert runtime.mission_manager.get_queue() == runtime.mission_manager.get_history() == []
    assert runtime._marvin_live_proof_checkpoint_valid()


def test_initial_disarmed_second_refused_valid_rearm_one_more(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch)
    r, b, _, events, _ = bundle
    first = initial(bundle); complete(bundle, first, 0)
    checkpoint = r._marvin_live_proof_continuation
    assert checkpoint.action_lidar_evidence == tuple(first["controller_result"]["history"][0]["action_lidar_evidence"])
    assert step(r)["reason"] == "marvin_live_proof_already_consumed"
    assert len(motions(events)) == 1
    consumed = set(r._marvin_alignment_consumed_source_frame_stamps)
    assert arm(r)["ok"] and r._marvin_live_proof_state == "ARMED"
    assert r._marvin_alignment_consumed_source_frame_stamps == consumed
    assert not arm(r)["ok"]  # No accumulation of credits.
    second = step(r); complete(bundle, second, 1)
    assert second["controller_result"]["history"][0]["source_frame_stamp_ns"] > checkpoint.camera_floor_stamp
    assert second["controller_result"]["proof"]["cumulative_actions_completed"] == 2
    assert second["controller_result"]["history"][0]["action_lidar_evidence"][1] > checkpoint.post_lidar_sequence
    assert not step(r)["execution_authorized"] and len(motions(events)) == 2


def test_arm_consumed_before_camera_and_dispatch(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r, b, robot, _, _ = bundle
    complete(bundle, initial(bundle), 0); assert arm(r)["ok"]
    seen = []
    def inspect():
        seen.append(r._marvin_live_proof_state)
        assert r._marvin_live_proof_state == "EXECUTING"
        assert not arm(r)["ok"] and not step(r)["execution_authorized"]
    b.on_observe = inspect; robot.on_motion = inspect
    complete(bundle, step(r), 1)
    assert seen == ["EXECUTING"] * 3


def test_two_concurrent_steps_and_rearm_cannot_share_arm(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r, b, _, events, _ = bundle
    complete(bundle, initial(bundle), 0); assert arm(r)["ok"]
    entered, release = threading.Event(), threading.Event(); results = []
    def hold():
        entered.set(); assert release.wait(3)
    b.on_observe = hold
    thread = threading.Thread(target=lambda: results.append(step(r)))
    thread.start(); assert entered.wait(3)
    try:
        assert step(r)["reason"] == "marvin_live_proof_owner_busy"
        assert arm(r)["reason"] == "marvin_live_proof_owner_busy"
    finally:
        release.set(); thread.join(3)
    assert not thread.is_alive(); complete(bundle, results[0], 1)
    assert len(motions(events)) == 2


def test_two_concurrent_rearms_only_one_credit(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r = bundle[0]
    complete(bundle, initial(bundle), 0)
    barrier = threading.Barrier(3); results = []
    def rearm():
        barrier.wait(); results.append(arm(r))
    threads = [threading.Thread(target=rearm) for _ in range(2)]
    for thread in threads: thread.start()
    barrier.wait()
    for thread in threads: thread.join(3); assert not thread.is_alive()
    assert sum(result["ok"] is True for result in results) == 1
    complete(bundle, step(r), 1)
    assert not step(r)["execution_authorized"]


@pytest.mark.parametrize("fault", ["interrupted", "delivery_uncertain", "stop_unconfirmed",
    "stop_failure", "missing_lidar", "duplicate_camera", "tracker_quality", "identity", "exception"])
def test_failed_proof_remains_locked(tmp_path, monkeypatch, fault):
    if fault == "interrupted":
        bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0, .8)] * 5,
            [LEFT_OPEN, None], interruptions=(True,))
    else:
        bundle = forward_bundle(tmp_path, monkeypatch)
    r, b, robot, events, _ = bundle
    if fault in {"delivery_uncertain", "exception"}:
        def failed(**kwargs):
            events.append(("uncertain",)); raise TimeoutError("mock delivery uncertainty")
        robot.move_forward = failed
    observe = r.observe_find_marvin_v2
    def current():
        result = observe()
        if motions(events):
            if fault == "tracker_quality": result["opencv_tracker"]["quality"] = .79
            if fault == "identity": result["identity_confirmed"] = False
        return result
    r.observe_find_marvin_v2 = current
    def after():
        if fault == "stop_unconfirmed": robot.ready = False
        if fault == "stop_failure": robot.stop = lambda: {"ok": False}
        if fault == "missing_lidar": b.freeze_lidar = True
        if fault == "duplicate_camera": b.repeat_camera = True
    robot.on_motion = after
    result = initial(bundle)
    assert not result["controller_result"]["proof"]["action_complete"]
    assert r._marvin_live_proof_state == "FAILED_LOCKED"
    assert r._marvin_live_proof_continuation is None
    assert not arm(r)["ok"] and not step(r)["execution_authorized"]
    assert len(motions(events)) == 1


@pytest.mark.parametrize("fault", ["not_running", "not_idle", "active_mission", "queue",
    "proof_owner", "behavior_owner", "physical_owner", "bridge_not_ready", "bridge_x",
    "bridge_y", "bridge_turn", "bridge_streaming", "bridge_ros", "bridge_exception"])
def test_rearm_admission_requires_idle_stopped_runtime(tmp_path, monkeypatch, fault):
    bundle = forward_bundle(tmp_path, monkeypatch); r, b, robot, events, _ = bundle
    complete(bundle, initial(bundle), 0)
    if fault == "not_running": r.running = False
    if fault == "not_idle": r._set_runtime_state("STOPPED")
    if fault == "active_mission": r.mission_manager.handle_intent({"intent": "FIND_OBJECT", "target": "marvin"})
    if fault == "queue": r.mission_manager.mission_queue.append(SimpleNamespace(to_dict=lambda:{}))
    if fault == "proof_owner": r._marvin_live_proof_owner = object()
    if fault == "behavior_owner": r._behavior_execution_generation = r._control_generation
    if fault == "physical_owner": r._physical_action_lock.acquire()
    old = robot.status
    def status():
        if fault == "bridge_exception": raise TimeoutError("mock status failure")
        s = copy.deepcopy(old()); m = s["motion"]
        if fault == "bridge_not_ready": s["status"] = "ERROR"
        if fault == "bridge_ros": s["ros_ready"] = False
        if fault == "bridge_x": m["linear_x"] = .1
        if fault == "bridge_y": m["linear_y"] = .1
        if fault == "bridge_turn": m["angular_z"] = .1
        if fault == "bridge_streaming": m["streaming"] = True
        return s
    robot.status = status
    try: assert not arm(r)["ok"]
    finally:
        if fault == "physical_owner": r._physical_action_lock.release()
    assert len(motions(events)) == 1


@pytest.mark.parametrize("fault", ["missing", "malformed", "digest", "stop", "terminal",
    "lidar", "lidar_order", "camera_order", "tracker", "count", "uncertain", "stamp_float",
    "lost_tracker", "source_identity", "consumption", "generation", "session", "fresh_lidar_session"])
def test_corrupt_or_discontinuous_checkpoint_fails_closed(tmp_path, monkeypatch, fault):
    bundle = forward_bundle(tmp_path, monkeypatch); r, b, _, events, _ = bundle
    complete(bundle, initial(bundle), 0)
    c = r._marvin_live_proof_continuation
    if fault == "missing": r._marvin_live_proof_continuation = None
    elif fault == "malformed": r._marvin_live_proof_continuation = {"state": "PROOF_COMPLETE"}
    elif fault == "digest": c.avoidance["local_avoidance_actions"] = 5
    elif fault == "lost_tracker": b._clear_marvin_v2_tracker_episode()
    elif fault == "source_identity": b._marvin_v2_tracker_episode["identity_source_frame_stamp_ns"] += 1
    elif fault == "consumption": r._marvin_alignment_consumed_source_frame_stamps.clear()
    elif fault == "generation": r._control_generation += 1
    elif fault == "session": r.lidar_worker.session = "new-producer"
    elif fault == "fresh_lidar_session":
        read = r.world_model.get_lidar_obstacles
        r.world_model.get_lidar_obstacles = lambda **kw: dict(read(**kw), producer_session="other")
    else:
        changes = {}; cert = dict(c.completion)
        if fault == "stop": cert["stop_confirmed"] = False
        if fault == "terminal": cert["state"] = "PROOF_INTERRUPTED"
        if fault == "lidar": cert["lidar_valid"] = False
        if fault == "lidar_order": changes["post_lidar_sequence"] = c.action_lidar_evidence[1]
        if fault == "camera_order": changes["camera_floor_stamp"] = c.previous_stamp
        if fault == "tracker": cert["tracker_quality"] = .79
        if fault == "count": cert["dispatch_opportunities"] = 2
        if fault == "uncertain": cert["delivery_uncertain"] = True
        if fault == "stamp_float": changes["previous_stamp"] = float(c.previous_stamp)
        changes["completion"] = cert
        c = replace(c, **changes); r._marvin_live_proof_continuation = c
        r._marvin_live_proof_checkpoint_digest = _marvin_proof_checkpoint_digest(c)
    assert not arm(r)["ok"]
    assert r._marvin_live_proof_state == "FAILED_LOCKED" and r._marvin_live_proof_continuation is None
    assert len(motions(events)) == 1


@pytest.mark.parametrize("fault", ["duplicate_post_camera", "consumed_camera", "session", "identity", "old_tracker"])
def test_rearmed_step_requires_new_sources_and_never_replays_consumed_stamp(tmp_path, monkeypatch, fault):
    bundle = forward_bundle(tmp_path, monkeypatch); r, b, _, events, _ = bundle
    complete(bundle, initial(bundle), 0); c = r._marvin_live_proof_continuation
    assert arm(r)["ok"]
    if fault == "session": r.lidar_worker.session = "changed-after-arm"
    else:
        observe = r._observe_find_marvin_v2
        def invalid(**kwargs):
            result = observe(**kwargs)
            if fault in {"duplicate_post_camera", "consumed_camera"}:
                stamp = c.camera_floor_stamp if fault == "duplicate_post_camera" else c.previous_stamp
                result["source_frame_stamp_ns"] = result["opencv_tracker"]["source_frame_stamp_ns"] = stamp
            elif fault == "identity":
                result["identity_source"] = "marvin_locked_tracker_continuity"
                result["identity_source_frame_stamp_ns"] = c.identity_source_frame_stamp_ns + 1
            else:
                result["strict_tracker_episode"] = {
                    "continued_existing_tracker": True, "initialized_this_observation": False}
            return result
        r._observe_find_marvin_v2 = invalid
    result = step(r)
    assert not result["motion_executed"] and len(motions(events)) == 1
    assert r._marvin_live_proof_state == "FAILED_LOCKED"
    assert not arm(r)["ok"]


def test_restart_is_process_local_and_shutdown_invalidates(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r = bundle[0]
    complete(bundle, initial(bundle), 0)
    r.stop(); assert r._marvin_live_proof_continuation is None
    assert r._marvin_live_proof_state == "FAILED_LOCKED"
    path = tmp_path / "restart"; path.mkdir()
    other = forward_bundle(path, monkeypatch)
    assert other[0]._marvin_live_proof_state == "UNINITIALIZED"
    assert other[0]._marvin_live_proof_continuation is None
    assert not arm(other[0])["ok"]
    complete(other, initial(other), 0)


@pytest.mark.parametrize("armed", [False, True])
def test_normal_mission_submission_invalidates_without_changing_mission_loop(tmp_path, monkeypatch, armed):
    bundle = make_runtime(tmp_path, monkeypatch, [(0,.8),(0,.75),(0,.6),(0,.55),(0,.5)])
    r = bundle[0]; complete(bundle, initial(bundle), 0)
    if armed: assert arm(r)["ok"]
    result = run(r)
    assert result["state"] == "ARRIVED" and result["actions_executed"] == 2
    assert result["max_local_avoidance_actions"] == 6 and "proof" not in result
    assert r._marvin_live_proof_continuation is None
    assert not arm(r)["ok"] and not step(r)["execution_authorized"]


def test_unrelated_physical_owner_invalidates(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r = bundle[0]
    complete(bundle, initial(bundle), 0); assert arm(r)["ok"]
    # Exercise the existing local physical owner, stopping before sensor dispatch.
    r._local_reactive_navigation_goal_active = lambda: None
    r.run_local_reactive_step()
    assert r._marvin_live_proof_state == "FAILED_LOCKED" and r._marvin_live_proof_continuation is None
    assert not arm(r)["ok"]


def test_checkpoint_does_not_retain_dispatch_authority_or_alias_response(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,1.1)]*6, [OPEN_LEFT])
    r = bundle[0]; first = initial(bundle); complete(bundle, first, 0)
    c = r._marvin_live_proof_continuation
    data = asdict(c)
    def inspect(v):
        if isinstance(v, dict):
            for k,x in v.items():
                assert k not in {"observation", "opencv_tracker", "options", "forward_safety", "lateral_safety", "motion_token", "guarded_result"}
                if k.endswith("permitted"): assert x is not True
                inspect(x)
        elif isinstance(v, (list,tuple)):
            for x in v: inspect(x)
    inspect(data)
    assert r._marvin_alignment_observation is None
    first["controller_result"]["local_avoidance_history"][0]["selection"]["direction"] = "RIGHT"
    first["controller_result"]["proof"]["source_stamps"].clear()
    assert c.previous_selection["direction"] == "LEFT" and r._marvin_live_proof_checkpoint_valid()


def test_strafe_side_counts_clearances_and_measured_progress_survive(tmp_path, monkeypatch):
    second = [(x,y-.04) if x>0 else (x,y) for x,y in LEFT_OPEN]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.8)]*6, [LEFT_OPEN,second,None])
    r = bundle[0]; first = initial(bundle); complete(bundle, first, 0)
    c = r._marvin_live_proof_continuation
    assert c.previous_selection["action_type"] == c.avoidance["previous_action_type"] == "STRAFE_LEFT"
    assert c.avoidance["last_detour_direction"] == c.previous_selection["direction"] == "LEFT"
    assert c.avoidance["local_avoidance_actions"] == 1
    assert c.previous_clearances == {"LEFT":c.previous_selection["left_clearance_m"],"RIGHT":c.previous_selection["right_clearance_m"]}
    first_progress = copy.deepcopy(c.avoidance_history[0]["actual_route_progress"])
    assert first_progress["meaningful_progress"]
    assert arm(r)["ok"]; second_result = step(r); complete(bundle, second_result, 1)
    assert motions(bundle[3]) == [("strafe",.08,1.),("strafe",.08,1.)]
    c = r._marvin_live_proof_continuation
    assert c.avoidance["local_avoidance_actions"] == 2
    assert c.avoidance_history[0]["actual_route_progress"] == first_progress
    assert second_result["controller_result"]["local_avoidance_history"][1]["previous_action_type"] == "STRAFE_LEFT"


def test_no_progress_and_opposite_side_cannot_repeat_across_boundaries(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,.8)]*8, [LEFT_OPEN])
    r = bundle[0]; complete(bundle, initial(bundle), 0); assert arm(r)["ok"]
    result = step(r)
    assert result["controller_result"]["state"] == "BLOCKED"
    assert len(motions(bundle[3])) == 1
    selection = result["controller_result"]["local_avoidance_history"][-1]["selection"]
    assert "STRAFE_LEFT" in selection["ineffective_action_types"]
    assert selection["action_type"] is None


def test_side_ranking_flicker_cannot_undo_measured_left_progress(tmp_path, monkeypatch):
    newer=[(x,y-.04) if x>0 else (x,1.2 if y>0 else -1.8) for x,y in LEFT_OPEN]
    bundle, _, _=strafe_runtime(tmp_path,monkeypatch,[(0,.8)]*6,[LEFT_OPEN,newer,None])
    r=bundle[0];complete(bundle,initial(bundle),0);assert arm(r)['ok']
    result=step(r);complete(bundle,result,1)
    selection=result['controller_result']['history'][0]['result']['local_detour']
    assert selection['progress_improved']
    assert selection['right_clearance_m']>selection['left_clearance_m']
    assert selection['options']['STRAFE_RIGHT']['reverses_previous_direction']
    assert selection['options']['STRAFE_RIGHT']['undoes_previous_progress']
    assert selection['direction']=='LEFT' and len(motions(bundle[3]))==2


@pytest.mark.parametrize('fault',['stale','session','coverage','capsule'])
def test_every_rearmed_bypass_rechecks_jit_without_using_saved_permission(tmp_path,monkeypatch,fault):
    bundle, _, _=strafe_runtime(tmp_path,monkeypatch,[(0,1.1)]*6,[OPEN_LEFT])
    r=bundle[0];complete(bundle,initial(bundle),0);assert arm(r)['ok']
    execute=r.behavior_manager.execute_single_marvin_approach_step
    read=r.world_model.get_lidar_obstacles
    def guarded(**kw):
        def faulty(**opts):
            scan=read(**opts)
            if fault=='stale':scan['received_monotonic_seconds']-=.301
            if fault=='session':scan['producer_session']='wrong'
            if fault=='coverage':scan['local_motion_geometry']['sectors']['rear']['valid_sample_count']=0
            if fault=='capsule':scan['local_motion_geometry']['points'].append({'x_m':.4,'y_m':.2})
            return scan
        r.world_model.get_lidar_obstacles=faulty
        return execute(**kw)
    r.behavior_manager.execute_single_marvin_approach_step=guarded
    result=step(r);c=result['controller_result']
    assert len(motions(bundle[3]))==1 and not result['motion_executed']
    assert c['history'][0]['observation']['local_avoidance_action']=='BYPASS_FORWARD'
    assert c['proof']['source_stamps'][0]['consumed']
    assert r._marvin_live_proof_state=='FAILED_LOCKED' and not arm(r)['ok']


def test_bypass_and_direct_forward_each_require_separate_arm_then_arrival(tmp_path, monkeypatch):
    specs = [(0,1.1)]*4 + [(0,round((x-delta)/100,2))
        for x in range(105,54,-5) for delta in (0,5)] + [(0,.5)]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, specs, [OPEN_LEFT,OPEN_LEFT,None])
    r = bundle[0]; complete(bundle, initial(bundle), 0)
    assert motions(bundle[3]) == [("strafe",.08,1.)]
    assert arm(r)["ok"]; bypass = step(r); complete(bundle, bypass, 1)
    row = bypass["controller_result"]["history"][0]; action = row["result"]
    assert action["action_type"] == "BYPASS_FORWARD"
    assert action["direction"] == "LEFT"
    b = action["local_detour"]["local_bypass"]
    assert b["bypass_target_x_m"] == .15 and b["bypass_target_y_m"] == 0
    assert b["protected_radius_m"] == .45 and b["bypass_forward_permitted"]
    assert action["approach_result"]["forward_safety"]["permitted"]
    assert b['producer_session']==r.lidar_worker.session
    assert action['local_detour']['acquisition_sequence']>bypass['controller_result']['avoidance_planning_lidar_sequence']
    assert all(sector['valid_sample_count']>=3 for sector in
        bypass['controller_result']['proof']['post_action_evidence']['current_lidar']['local_motion_geometry']['sectors'].values())
    assert motions(bundle[3]) == [("strafe",.08,1.),("forward",.1,.5)]
    assert not step(r)["execution_authorized"]
    assert arm(r)["ok"]; direct = step(r); complete(bundle, direct, 2)
    assert direct["controller_result"]["history"][0]["state"] == "ADVANCING"
    assert motions(bundle[3])[-1] == ("forward",.1,.5)
    c = r._marvin_live_proof_continuation
    assert c.previous_selection is None and not c.avoidance["local_bypass_active"]
    assert c.avoidance["local_bypass_target_x_m"] is None
    assert c.avoidance["local_avoidance_actions"] == 2 and c.avoidance["local_bypass_actions"] == 1
    for count in range(3,13):
        assert arm(r)["ok"]; complete(bundle, step(r), count)
        assert motions(bundle[3])[-1] == ("forward",.1,.5)
    assert arm(r)["ok"]; arrived = step(r)
    assert arrived["controller_result"]["state"] == "ARRIVED"
    assert arrived["actions_executed"] == 0 and len(motions(bundle[3])) == 13
    assert arrived["proof_state"] == "ARRIVED_DISARMED" and not arrived["continuation_available"]
    assert not arm(r)["ok"]


def test_first_bypass_reassessment_survives_later_alignment(tmp_path, monkeypatch):
    shifted = [(.60,-.25),(0.,1.2),(0.,-.65)]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch,
        [(0,1.1)]*4+[(100,1.1)]+[(0,1.1)]*6, [OPEN_LEFT,OPEN_LEFT,OPEN_LEFT,shifted])
    r = bundle[0]; complete(bundle, initial(bundle), 0)
    assert arm(r)["ok"]; complete(bundle, step(r), 1)
    frozen = copy.deepcopy(r._marvin_live_proof_continuation.previous_selection["first_post_action_bypass_progress"])
    assert not frozen["meaningful_progress"]
    assert arm(r)["ok"]; complete(bundle, step(r), 2)
    assert motions(bundle[3])[-1][0] == "turn"
    assert r._marvin_live_proof_continuation.previous_selection["first_post_action_bypass_progress"] == frozen
    assert arm(r)["ok"]; result = step(r)
    complete(bundle, result, 3)
    continuation = result["controller_result"]["history"][0]["result"]
    assert continuation["action_type"] == "BYPASS_FORWARD"
    assert continuation["local_detour"]["actual_route_progress"] == frozen
    assert 'BYPASS_FORWARD' not in continuation["local_detour"]["ineffective_action_types"]
    assert r._marvin_live_proof_continuation.avoidance["local_avoidance_actions"] == 3


def test_six_action_avoidance_budget_is_cumulative(tmp_path, monkeypatch):
    scenes = [OPEN_LEFT,OPEN_LEFT] + [[(.65-.011*i,-.25),(0.,1.2),(0.,-.65)] for i in range(1,8)]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0,1.1)]*16, scenes)
    r = bundle[0]; complete(bundle, initial(bundle), 0)
    for count in range(1,6):
        assert arm(r)["ok"]; result = step(r); complete(bundle, result, count)
        assert r._marvin_live_proof_continuation.avoidance["local_avoidance_actions"] == count+1
    assert arm(r)["ok"]; result = step(r)
    assert result["controller_result"]["reason"] == "find_marvin_local_avoidance_exhausted"
    assert result["controller_result"]["local_avoidance_actions"] == 6
    assert result["controller_result"]["max_local_avoidance_actions"] == 6
    assert len(motions(bundle[3])) == 6


def test_exact_stamps_survive_rearm_history_and_decimal_serialization(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r,b,_,_,_ = bundle
    b.source_offset_ns = 9876543210123456789
    first = initial(bundle); complete(bundle, first, 0)
    stamp = first["controller_result"]["history"][0]["source_frame_stamp_ns"]
    assert type(stamp) is int and stamp > 2**53
    assert arm(r)["ok"]; second = step(r); complete(bundle, second, 1)
    c = r._marvin_live_proof_continuation
    assert c.action_history[0]["source_frame_stamp_ns"] == stamp
    assert stamp in r._marvin_alignment_consumed_source_frame_stamps
    payload = _precision_safe_source_frame_stamps(second)
    value = payload["controller_result"]["history"][0]["source_frame_stamp_ns"]
    assert type(value) is str and int(value) == second["controller_result"]["history"][0]["source_frame_stamp_ns"]
    assert int(value) > stamp


def test_retained_a6c4d675_replay_selects_bypass_without_injected_selection(tmp_path, monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    bundle, _, _ = strafe_runtime(tmp_path,monkeypatch,
        [(0,1.38)] + [(40,1.38)]*16, [None])
    r,b,_,events,clock=bundle
    original=r.world_model.get_lidar_obstacles;clear=[False]
    def scan(**kwargs):
        # Acquire a bounded, isolated target using ordinary fresh simulated
        # perception first. No range anchor or planner selection is injected.
        count=len(motions(events))
        if count==0 or clear[0]:
            clock[0]+=1000
            return original(**kwargs)
        row=fixture['cycles'][min(count-1,5)]
        state=copy.deepcopy(row['lidar'])
        b.sequence+=1;clock[0]+=1000
        state.update(producer_session=r.lidar_worker.session,acquisition_sequence=b.sequence,
            received_monotonic_seconds=time.monotonic(),age_at_receipt_seconds=0.,effective_age_seconds=0.,reason='fresh')
        if count>=7:
            # Fresh simulated longitudinal passage after the selected bypass.
            for point in state['local_motion_geometry']['points']:
                point['x_m']-=.05
                point['distance_m']=math.hypot(point['x_m'],point['y_m'])
                point['robot_bearing_deg']=math.degrees(math.atan2(point['y_m'],point['x_m']))
        return state
    r.world_model.get_lidar_obstacles=scan
    results=[initial(bundle)];complete(bundle,results[0],0)
    for count in range(1,7):
        assert arm(r)['ok']
        results.append(step(r));complete(bundle,results[-1],count)
        if results[-1]['controller_result']['history'][0]['result'].get('action_type') == 'BYPASS_FORWARD':
            break  # Recorded strafe scans cannot prove counterfactual bypass outcomes.
    selected=[result['controller_result']['history'][0]['result'].get('action_type') or
        result['controller_result']['history'][0]['state'] for result in results]
    assert selected==['ADVANCING','STRAFE_LEFT','STRAFE_LEFT','BYPASS_FORWARD']
    assert fixture['mission_id']=='mission-a6c4d675'
    c=r._marvin_live_proof_continuation
    assert c.avoidance['local_avoidance_actions']==3 and c.avoidance['local_bypass_actions']==1
    bypass=results[-1]['controller_result']['history'][0]['result']
    target=bypass['local_detour']['local_bypass']
    assert bypass['direction']=='LEFT' and target['bypass_target_x_m']<=.15
    assert target['protected_radius_m']==.45
    assert motions(events)[-1]==('forward',.1,.5)
    assert not step(r)['execution_authorized'] and len(motions(events))==4
    clear[0]=True;b.specs=iter([(0,1.38),(0,1.33)])
    assert arm(r)['ok'];direct=step(r);complete(bundle,direct,4)
    assert direct['controller_result']['history'][0]['state']=='ADVANCING'
    assert r._marvin_live_proof_continuation.previous_selection is None
    assert direct['controller_result']['local_avoidance_actions']==3
    assert motions(events)[-1]==('forward',.1,.5)
    print('a6c4d675 replay:',selected+['ADVANCING'])


@pytest.mark.parametrize("body", [{}, {"rearm":False}, {"rearm":1}, {"rearm":1.0},
    {"rearm":"true"}, {"rearm":None}, {"rearm":True,"max_physical_actions":1},
    {"rearm":True,"direction":"LEFT"}, {"rearm":True,"speed":.1},
    {"rearm":True,"duration":.5}, {"rearm":True,"source_frame_stamp_ns":1791415725535715481},
    {"rearm":True,"side":"LEFT"}, {"rearm":True,"selection":"BYPASS_FORWARD"}])
def test_rearm_api_contract_is_strict(body):
    method = Mock(return_value={"ok":True})
    h=object.__new__(RuntimeAPIHandler);h.path="/find-marvin/live-proof-rearm"
    h.server=SimpleNamespace(runtime=SimpleNamespace(rearm_find_marvin_live_proof=method))
    h.require_json_request=lambda:body;responses=[]
    h.send_json=lambda code,payload:responses.append((code,payload))
    h.do_POST();assert responses[-1][0]==400 and not method.called


@pytest.mark.parametrize("ok,code", [(True,200),(False,409)])
def test_rearm_api_accepts_only_true_and_never_executes_step(ok, code):
    method = Mock(return_value={"ok":ok});step_method=Mock()
    h=object.__new__(RuntimeAPIHandler);h.path="/find-marvin/live-proof-rearm"
    h.server=SimpleNamespace(runtime=SimpleNamespace(rearm_find_marvin_live_proof=method,
        execute_find_marvin_live_proof_step=step_method))
    h.require_json_request=lambda:{"rearm":True};responses=[]
    h.send_json=lambda code,payload:responses.append((code,payload))
    h.do_POST();method.assert_called_once_with(rearm=True)
    assert responses==[(code,{"ok":ok})] and not step_method.called


def test_rearm_without_prior_proof_cannot_grant_credit(tmp_path, monkeypatch):
    bundle = forward_bundle(tmp_path, monkeypatch); r=bundle[0];r._set_runtime_state("IDLE")
    assert not arm(r)["ok"] and r._marvin_live_proof_state == "UNINITIALIZED"
    complete(bundle, step(r), 0)


def test_delivery_uncertainty_cannot_claim_complete_or_rearm(tmp_path,monkeypatch):
    bundle=forward_bundle(tmp_path,monkeypatch);r=bundle[0]
    original=r.execute_single_marvin_approach
    def uncertain(**kw):
        result=original(**kw)
        result['approach_result']['forward_result']['delivery_uncertain']=True
        return result
    r.execute_single_marvin_approach=uncertain
    result=initial(bundle)
    assert result['controller_result']['state']=='PROOF_INTERRUPTED'
    assert result['proof_state']=='FAILED_LOCKED' and not arm(r)['ok']
    assert len(motions(bundle[3]))==1
