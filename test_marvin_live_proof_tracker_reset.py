"""Offline proof boundaries with real semantic/tracker acquisition and planning."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import pytest

from behavior_manager import BehaviorManager
from test_find_marvin_closed_loop import delayed_runtime, motions, run
from test_marvin_lateral_avoidance import strafe_runtime, LEFT_OPEN
from test_marvin_local_bypass import OPEN_LEFT
from test_marvin_live_proof_continuation import arm, complete, step


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Proof boundary tests must not access robot/network/services")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


def proof_runtime(tmp_path, monkeypatch, specs):
    bundle = delayed_runtime(tmp_path, monkeypatch, specs)
    r, b, _, events, _ = bundle
    reacquire = b.reacquire_find_marvin_v2
    b.proof_source_floors = []
    b.visual_shift = 0
    box = b.current_box

    def shifted_box():
        value = box()
        return dict(value, y1=value["y1"] + b.visual_shift,
                    y2=value["y2"] + b.visual_shift)

    b.current_box = shifted_box

    def fresh_semantic(**kwargs):
        # Advance only simulated scene/transport; execute the production
        # stopped reacquisition, Gemini, tracker init and action-frame gates.
        b.proof_source_floors.append(kwargs["minimum_source_frame_stamp_ns"])
        assert b._marvin_v2_tracker_episode is None
        assert r._marvin_live_proof_state == "EXECUTING"
        b.current_spec = next(b.specs)
        if b.current_spec != "absent":
            b.distance = b.current_spec[1]
        events.append("observe")
        return reacquire(**kwargs)

    b.reacquire_find_marvin_v2 = fresh_semantic
    return bundle


def begin(bundle):
    bundle[0]._set_runtime_state("IDLE")
    result = step(bundle[0])
    complete(bundle, result, 0)
    return result


def forward(tmp_path, monkeypatch):
    return proof_runtime(tmp_path, monkeypatch,
                         [(0, .8), (0, .75), (0, .75), (0, .7)])


def test_live_geometry_reproduces_production_iou_veto_without_lowering_threshold():
    b = object.__new__(BehaviorManager)
    import threading
    b._marvin_v2_tracker_episode_lock = threading.RLock()
    old = {"x1": 341, "y1": 198, "x2": 440, "y2": 315}
    raw = {"x1": 323., "y1": 206., "x2": 395., "y2": 313.}
    seed = b._expand_marvin_tracker_seed_bbox(raw, 640, 480)
    iou = b._target_bbox_iou({"bbox": old}, {"bbox": seed})
    assert iou == pytest.approx(.49913077071663126)
    assert b.MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU == .70
    b._marvin_v2_tracker_episode = {"marvin_tracker": object(), "tracker_bbox": old}
    b._acquire_marvin_tracker_observation_from_candidate = lambda *a, **kw: pytest.fail("Must veto before tracker initialization")
    result = b._acquire_strict_v2_tracker_observation_from_candidate(
        {"bbox": raw, "image_width": 640, "image_height": 480}, {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=1791418345824574940,
        execution_guard=None, frame=SimpleNamespace(received_at="offline"))
    assert result["reason"] == "marvin_v2_semantic_tracker_association_failed"
    assert result["strict_tracker_episode"]["association_threshold"] == .70


@pytest.mark.parametrize("pause_seconds", [0, 60, 3600])
def test_low_iou_resume_initializes_new_tracker_and_completes_one_action(tmp_path, monkeypatch, pause_seconds):
    bundle = forward(tmp_path, monkeypatch); r, b, _, events, _ = bundle
    begin(bundle); c = r._marvin_live_proof_continuation
    old = b._marvin_v2_tracker_episode
    bundle[4][0] += pause_seconds * 1_000_000_000
    assert arm(r)["ok"] and b._marvin_v2_tracker_episode is None
    assert r._marvin_live_proof_continuation is c
    b.visual_shift = 110
    result = step(r); complete(bundle, result, 1)
    row = result["controller_result"]["history"][0]; obs = row["observation"]
    iou = b._target_bbox_iou({"bbox": old["tracker_bbox"]}, obs["opencv_tracker"])
    assert iou < .70
    assert obs["strict_tracker_episode"]["initialized_this_observation"] is True
    assert obs["strict_tracker_episode"]["continued_existing_tracker"] is False
    assert obs["identity_source"] == "gemini_marvin_candidate_selection"
    assert c.camera_floor_stamp < obs["identity_source_frame_stamp_ns"] < obs["source_frame_stamp_ns"]
    assert b.proof_source_floors == [c.camera_floor_stamp]
    assert b.created_trackers[0] is not b.created_trackers[1]
    post = result["controller_result"]["proof"]["post_action_evidence"]["observation"]
    assert post["post_action_tracker_continuity"] is True
    assert b._marvin_v2_tracker_episode["marvin_tracker"] is b.created_trackers[1]
    assert b.semantic_calls == 2 and len(motions(events)) == 2
    assert not step(r)["execution_authorized"]


@pytest.mark.parametrize("field", [
    "avoidance", "previous_selection", "previous_clearances", "avoidance_history",
    "avoidance_lidar_refresh_history", "action_history", "last_action_state",
    "previous_stamp", "camera_floor_stamp", "action_finished_monotonic_seconds",
    "action_lidar_evidence", "post_lidar_sequence", "identity_source_frame_stamp_ns",
    "acquired", "range_association", "completion",
])
def test_rearm_preserves_checkpoint_fields_without_observation_or_motion(tmp_path, monkeypatch, field):
    shifted = [(x, y - .04) if x > 0 else (x, y) for x, y in LEFT_OPEN]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0, .8)] * 4,
                                   [LEFT_OPEN, shifted], factory=proof_runtime)
    r, b, _, events, _ = bundle; begin(bundle)
    c = r._marvin_live_proof_continuation; saved = copy.deepcopy(asdict(c))
    old_events = list(events); old_semantics = b.semantic_calls
    old_consumed = set(r._marvin_alignment_consumed_source_frame_stamps)
    old_digest = r._marvin_live_proof_checkpoint_digest
    r._marvin_alignment_observation = {"old": True}
    r._marvin_alignment_consensus = [{"old": True}]
    r._marvin_alignment_geometry_history = [{"old": True}]
    assert c.previous_selection["action_type"] == "STRAFE_LEFT"
    assert c.avoidance["last_detour_direction"] == "LEFT"
    assert c.avoidance["local_avoidance_actions"] == 1
    assert arm(r)["ok"]
    assert r._marvin_live_proof_continuation is c
    assert asdict(c)[field] == saved[field]
    assert r._marvin_live_proof_checkpoint_digest == old_digest
    assert b._marvin_v2_tracker_episode is None
    assert r._marvin_alignment_observation is None
    assert r._marvin_alignment_consensus == r._marvin_alignment_geometry_history == []
    assert r._marvin_alignment_consumed_source_frame_stamps == old_consumed
    assert events == old_events and b.semantic_calls == old_semantics
    assert r._marvin_live_proof_checkpoint_valid()


def test_low_iou_strafe_resume_uses_prior_measured_progress_and_side(tmp_path, monkeypatch):
    shifted = [(x, y - .04) if x > 0 else (x, y) for x, y in LEFT_OPEN]
    bundle, _, _ = strafe_runtime(tmp_path, monkeypatch, [(0, .8)] * 6,
                                   [LEFT_OPEN, shifted, None], factory=proof_runtime)
    r, b, _, events, _ = bundle; begin(bundle)
    c = r._marvin_live_proof_continuation; old = b._marvin_v2_tracker_episode
    progress = copy.deepcopy(c.avoidance_history[0]["actual_route_progress"])
    assert progress["meaningful_progress"]
    assert arm(r)["ok"]; b.visual_shift = 110
    result = step(r); complete(bundle, result, 1)
    row = result["controller_result"]["local_avoidance_history"][1]
    assert row["previous_action_type"] == "STRAFE_LEFT"
    assert row["previous_clearances"] == c.previous_clearances
    assert row["selection"]["action_type"] == "STRAFE_LEFT"
    assert row["selection"]["progress_improved"] is True
    assert r._marvin_live_proof_continuation.avoidance["local_avoidance_actions"] == 2
    assert r._marvin_live_proof_continuation.avoidance["last_detour_direction"] == "LEFT"
    assert r._marvin_live_proof_continuation.avoidance_history[0]["actual_route_progress"] == progress
    obs = result["controller_result"]["history"][0]["observation"]
    assert b._target_bbox_iou({"bbox": old["tracker_bbox"]}, obs["opencv_tracker"]) < .70
    assert len(motions(events)) == 2


@pytest.mark.parametrize("fault", ["gemini_negative", "tracker_quality", "tracker_unmatched",
    "tracker_initialization", "cached_action", "camera_unavailable", "old_identity_source",
    "semantic_floor", "semantic_float", "action_equals_identity", "old_tracker", "exception"])
def test_fresh_visual_failure_locks_without_old_tracker_fallback(tmp_path, monkeypatch, fault):
    bundle = forward(tmp_path, monkeypatch); r, b, _, events, _ = bundle
    begin(bundle); c = r._marvin_live_proof_continuation
    old = b.created_trackers[0]; consumed = set(r._marvin_alignment_consumed_source_frame_stamps)
    assert arm(r)["ok"]
    if fault == "gemini_negative": b.confirm_identity = False
    if fault == "cached_action": b.refresh_mode = "cached_semantic"
    if fault == "camera_unavailable": b.refresh_mode = "unavailable"
    if fault == "tracker_initialization":
        b.marvin_local_tracker_factory = lambda *a: (_ for _ in ()).throw(ValueError("offline tracker init failed"))
    if fault == "tracker_quality":
        create = b.marvin_local_tracker_factory
        def low(*args):
            tracker = create(*args); update = tracker.update
            def poor(frame):
                box = update(frame); tracker.last_quality = .79; return box
            tracker.update = poor
            return tracker
        b.marvin_local_tracker_factory = low
    original = r._observe_find_marvin_v2
    def bad(**kwargs):
        if fault == "exception": raise ValueError("offline proof exception")
        obs = original(**kwargs)
        if kwargs.get("reacquisition_source_floor") is not None:
            if fault == "tracker_unmatched": obs["opencv_tracker"]["matched"] = False
            if fault == "old_identity_source": obs["identity_source"] = "marvin_locked_tracker_continuity"
            if fault == "semantic_floor": obs["identity_source_frame_stamp_ns"] = c.camera_floor_stamp
            if fault == "semantic_float": obs["identity_source_frame_stamp_ns"] = float(c.camera_floor_stamp + 1)
            if fault == "action_equals_identity": obs["source_frame_stamp_ns"] = obs["identity_source_frame_stamp_ns"]
            if fault == "old_tracker": obs["strict_tracker_episode"] = {"initialized_this_observation":False, "continued_existing_tracker":True}
        return obs
    r._observe_find_marvin_v2 = bad
    result = step(r)
    assert not result["motion_executed"] and len(motions(events)) == 1
    assert r._marvin_alignment_consumed_source_frame_stamps == consumed
    assert result["proof_state"] == "FAILED_LOCKED" and not result["continuation_available"]
    assert b._marvin_v2_tracker_episode is None and r._marvin_alignment_observation is None
    assert not arm(r)["ok"] and not step(r)["execution_authorized"]
    assert b.created_trackers.count(old) == 1
    assert events[-1] == "stop"


@pytest.mark.parametrize("fault", ["raise", "no_clear", "missing"])
def test_visual_reset_failure_never_grants_arm(tmp_path, monkeypatch, fault):
    bundle = forward(tmp_path, monkeypatch); r, b, _, events, _ = bundle; begin(bundle)
    if fault == "raise": b._clear_marvin_v2_tracker_episode = lambda: (_ for _ in ()).throw(ValueError("reset failed"))
    if fault == "no_clear": b._clear_marvin_v2_tracker_episode = lambda: None
    if fault == "missing": b._clear_marvin_v2_tracker_episode = None
    result = arm(r)
    assert result["reason"] == "marvin_live_proof_visual_reset_failed" and not result["ok"]
    assert r._marvin_live_proof_state == "FAILED_LOCKED"
    assert not step(r)["execution_authorized"] and len(motions(events)) == 1


def test_rejected_rearm_and_admission_do_not_clear_production_visual_state(tmp_path, monkeypatch):
    bundle = forward(tmp_path, monkeypatch); r, b, _, _, _ = bundle; begin(bundle)
    episode = b._marvin_v2_tracker_episode; c = r._marvin_live_proof_continuation
    r._physical_action_lock.acquire()
    try:
        assert not arm(r)["ok"]
        assert b._marvin_v2_tracker_episode is episode and r._marvin_live_proof_continuation is c
    finally: r._physical_action_lock.release()
    assert arm(r)["ok"]
    # A different observer cannot silently supply an episode after the reset.
    b._marvin_v2_tracker_episode = episode
    assert step(r)["reason"] == "marvin_live_proof_continuation_invalid"
    assert b._marvin_v2_tracker_episode is episode


def test_normal_mission_keeps_one_tracker_and_continuous_action_loop(tmp_path, monkeypatch):
    bundle = delayed_runtime(tmp_path, monkeypatch, [(0,.60),(0,.58),(0,.5)])
    r,b,_,events,_ = bundle; result = run(r)
    assert result["state"] == "ARRIVED" and len(motions(events)) == 2
    assert b.semantic_calls == 1 and len(b.created_trackers) == 1
    assert "proof" not in result and result["max_local_avoidance_actions"] == 6
    assert b.MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU == .70


def test_post_action_tracker_stays_locked_despite_low_static_iou(tmp_path, monkeypatch):
    bundle = forward(tmp_path, monkeypatch); r,b,robot,_,_ = bundle
    robot.on_motion = lambda: setattr(b, "visual_shift", 110)
    result = begin(bundle)
    action = result["controller_result"]["history"][0]["observation"]
    post = result["controller_result"]["proof"]["post_action_evidence"]["observation"]
    assert b._target_bbox_iou(action["opencv_tracker"], post["opencv_tracker"]) < .70
    assert post["post_action_tracker_continuity"] is True
    assert post["post_action_tracker_diagnostics"]["pre_to_post_iou_used_for_admission"] is False
    assert b.semantic_calls == 1 and len(b.created_trackers) == 1
    assert r._marvin_live_proof_continuation is not None


def test_exact_semantic_and_action_floors_and_consumption_survive_new_episode(tmp_path, monkeypatch):
    bundle=forward(tmp_path,monkeypatch);r,b,_,events,_=bundle
    b.source_offset_ns=9876543210123456789
    first=begin(bundle);c=r._marvin_live_proof_continuation
    stamp=first["controller_result"]["history"][0]["source_frame_stamp_ns"]
    assert type(stamp) is int and stamp>2**53
    assert arm(r)["ok"];second=step(r);complete(bundle,second,1)
    obs=second["controller_result"]["history"][0]["observation"]
    assert type(obs["identity_source_frame_stamp_ns"]) is type(obs["source_frame_stamp_ns"]) is int
    assert stamp<c.camera_floor_stamp<obs["identity_source_frame_stamp_ns"]<obs["source_frame_stamp_ns"]
    assert {stamp,obs["source_frame_stamp_ns"]}<=r._marvin_alignment_consumed_source_frame_stamps
    assert r._marvin_live_proof_continuation.action_history[0]["source_frame_stamp_ns"]==stamp
    assert b.proof_source_floors==[c.camera_floor_stamp]
    assert not step(r)["execution_authorized"] and len(motions(events))==2


@pytest.mark.parametrize("progress", [False, True])
def test_progress_and_oscillation_gates_still_apply_after_new_semantics(tmp_path, monkeypatch, progress):
    newer=[(x,y-.04) if x>0 else (x,1.2 if y>0 else -1.8) for x,y in LEFT_OPEN]
    scenes=[LEFT_OPEN,newer,None] if progress else [LEFT_OPEN]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,.8)]*6,scenes,factory=proof_runtime)
    r,b,_,events,_=bundle;begin(bundle);assert arm(r)["ok"];b.visual_shift=110
    result=step(r);selection=result["controller_result"]["local_avoidance_history"][-1]["selection"]
    if progress:
        complete(bundle,result,1)
        assert selection["direction"]=="LEFT"
        assert selection["options"]["STRAFE_RIGHT"]["undoes_previous_progress"]
    else:
        assert result["controller_result"]["state"]=="BLOCKED"
        assert "STRAFE_LEFT" in selection["ineffective_action_types"]
        assert selection["action_type"] is None and len(motions(events))==1
        assert result["proof_state"]=="FAILED_LOCKED"


def test_frozen_first_bypass_outcome_survives_visual_reset_and_alignment(tmp_path, monkeypatch):
    shifted = [(.60,-.25),(0.,1.2),(0.,-.65)]
    bundle,_,_ = strafe_runtime(tmp_path,monkeypatch,
        [(0,1.1)]*4+[(100,1.1),(0,1.1),(0,1.1)],
        [OPEN_LEFT,OPEN_LEFT,OPEN_LEFT,shifted],factory=proof_runtime)
    r,b,_,events,_=bundle;begin(bundle)
    assert arm(r)["ok"];complete(bundle,step(r),1)
    c=r._marvin_live_proof_continuation
    frozen=copy.deepcopy(c.previous_selection["first_post_action_bypass_progress"])
    assert not frozen["meaningful_progress"]
    assert c.avoidance["local_bypass_actions"]==1
    assert arm(r)["ok"] and b._marvin_v2_tracker_episode is None
    assert c.previous_selection["first_post_action_bypass_progress"]==frozen
    b.visual_shift=110;complete(bundle,step(r),2)
    assert motions(events)[-1][0]=="turn"
    assert r._marvin_live_proof_continuation.previous_selection["first_post_action_bypass_progress"]==frozen
    assert arm(r)["ok"];result=step(r)
    assert result["controller_result"]["state"]=="BLOCKED" and len(motions(events))==3


def test_bypass_direct_forward_arrival_with_fresh_semantics_per_arm(tmp_path, monkeypatch):
    specs=[(0,1.1)]*4+[(0,round((x-delta)/100,2)) for x in range(105,54,-5) for delta in (0,5)]+[(0,.5)]
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,specs,[OPEN_LEFT,OPEN_LEFT,None],factory=proof_runtime)
    r,b,_,events,_=bundle;begin(bundle)
    assert motions(events)==[("strafe",.08,1.)]
    assert arm(r)["ok"];b.visual_shift=110;bypass=step(r);complete(bundle,bypass,1)
    action=bypass["controller_result"]["history"][0]["result"]
    assert action["action_type"]=="BYPASS_FORWARD" and action["direction"]=="LEFT"
    target=action["local_detour"]["local_bypass"]
    assert target["bypass_target_x_m"]<=.15 and target["protected_radius_m"]==.45
    assert motions(events)[-1]==("forward",.1,.5)
    for count in range(2,13):
        assert arm(r)["ok"];b.visual_shift=110*(count%2)
        result=step(r);complete(bundle,result,count)
        assert result["controller_result"]["history"][0]["state"]=="ADVANCING"
        c=r._marvin_live_proof_continuation
        assert c.previous_selection is None and c.avoidance["local_bypass_target_x_m"] is None
        assert c.avoidance["local_avoidance_actions"]==2 and c.avoidance["local_bypass_actions"]==1
    assert arm(r)["ok"];arrived=step(r)
    assert arrived["controller_result"]["state"]=="ARRIVED" and not arrived["motion_executed"]
    assert arrived["proof_state"]=="ARRIVED_DISARMED" and not arrived["continuation_available"]
    assert len(motions(events))==13 and b.semantic_calls==14
    assert not arm(r)["ok"]


def test_a6c4d675_replay_resets_trackers_and_keeps_cumulative_six_action_budget(tmp_path, monkeypatch):
    fixture=json.loads((Path(__file__).parent/'test_fixtures/marvin_a6c4d675_bypass_geometry.json').read_text())
    bundle,_,_=strafe_runtime(tmp_path,monkeypatch,[(0,1.38)]+[(40,1.38)]*16,[None],factory=proof_runtime)
    r,b,_,events,clock=bundle;original=r.world_model.get_lidar_obstacles;clear=[False]
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
    results=[begin(bundle)]
    for count in range(1,7):
        old=b._marvin_v2_tracker_episode
        assert arm(r)['ok'] and b._marvin_v2_tracker_episode is None
        b.visual_shift=110*(count%2)
        result=step(r);complete(bundle,result,count);results.append(result)
        obs=result['controller_result']['history'][0]['observation']
        assert obs['strict_tracker_episode']['initialized_this_observation']
        assert b._target_bbox_iou({'bbox':old['tracker_bbox']},obs['opencv_tracker'])<.70
    selected=[x['controller_result']['history'][0]['result'].get('action_type') or
              x['controller_result']['history'][0]['state'] for x in results]
    assert selected==['ADVANCING']+['STRAFE_LEFT']*5+['BYPASS_FORWARD']
    c=r._marvin_live_proof_continuation
    assert c.avoidance['local_avoidance_actions']==6 and c.avoidance['local_bypass_actions']==1
    bypass=results[-1]['controller_result']['history'][0]['result']
    target=bypass['local_detour']['local_bypass']
    assert bypass['direction']=='LEFT' and target['bypass_target_x_m']<=.15 and target['protected_radius_m']==.45
    assert bypass['approach_result']['forward_safety']['permitted']
    assert bypass['local_detour']['acquisition_sequence']>results[-1]['controller_result']['avoidance_planning_lidar_sequence']
    assert motions(events)[-1]==('forward',.1,.5)
    assert not step(r)['execution_authorized'] and len(motions(events))==7
    clear[0]=True;b.specs=iter([(0,1.38),(0,1.33)])
    assert arm(r)['ok'];direct=step(r);complete(bundle,direct,7)
    assert direct['controller_result']['history'][0]['state']=='ADVANCING'
    assert direct['controller_result']['local_avoidance_actions']==6
    assert r._marvin_live_proof_continuation.previous_selection is None
    assert b.semantic_calls==8 and len(b.created_trackers)==8
    print('Fresh-episode a6c4d675 replay:',selected+['ADVANCING'])
