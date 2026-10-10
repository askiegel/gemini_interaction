"""Offline missions: controlled sensors/transport, production safety/ownership."""
import math

import pytest

from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
from marvin_local_obstacle_avoidance import select_marvin_detour
from marvin_obstacle_phases import plan_phase_action
from test_marvin_lateral_avoidance import strafe_runtime
from test_find_marvin_closed_loop import make_runtime, motions, run, Worker
from test_find_marvin_reacquisition import recovery_runtime


def obstacle_geometry(runtime, behavior, events, modes):
    """Keep scan coverage coherent with added base-frame obstacle points."""
    original = behavior.lidar

    def read(**kwargs):
        state = original(**kwargs)
        mode = modes[min(len(motions(events)), len(modes) - 1)]
        if mode is None:
            return state
        left, right = mode
        additions = [(0.485, 0.10), (0, left), (0, -right)]
        points = state["local_motion_geometry"]["points"]
        for x, y in additions:
            points.append({"x_m": x, "y_m": y, "distance_m": math.hypot(x, y),
                           "robot_bearing_deg": math.degrees(math.atan2(y, x))})
        sectors = state["local_motion_geometry"]["sectors"]
        for sector in sectors.values():
            sector.update(valid_sample_count=0, available=False, minimum_distance_from_base_m=None)
        for point in points:
            bearing = math.degrees(math.atan2(point["y_m"], point["x_m"]))
            name = ("front" if -22.5 <= bearing < 22.5 else
                    "front_left" if 22.5 <= bearing < 67.5 else
                    "left" if 67.5 <= bearing < 112.5 else
                    "rear_left" if 112.5 <= bearing < 157.5 else
                    "front_right" if -67.5 <= bearing < -22.5 else
                    "right" if -112.5 <= bearing < -67.5 else
                    "rear_right" if -157.5 <= bearing < -112.5 else "rear")
            sector = sectors[name]
            distance = math.hypot(point["x_m"], point["y_m"])
            sector.update(valid_sample_count=sector["valid_sample_count"] + 1,
                          available=True, minimum_distance_from_base_m=min(
                              sector["minimum_distance_from_base_m"] or float("inf"), distance))
        state["sectors"] = {name: {"available": sector["available"], "state": "CLEAR",
            "robust_clearance_m": sector["minimum_distance_from_base_m"],
            "minimum_clearance_m": sector["minimum_distance_from_base_m"]}
            for name, sector in sectors.items()}
        return state

    runtime.world_model.get_lidar_obstacles = read
    if getattr(behavior.robot, 'forward_interlock', None) is not None:
        behavior.robot.forward_interlock.reader = lambda **kwargs: runtime.world_model.get_lidar_obstacles(**kwargs)
    return read


def avoidance_runtime(tmp_path, monkeypatch, specs, modes):
    scenes = [None if mode is None else [(0.485,0.),(0.,mode[0]),(0.,-mode[1])] for mode in modes]
    bundle, _, _ = strafe_runtime(tmp_path,monkeypatch,specs,scenes)
    return bundle


def test_clear_path_never_calls_detour_selection(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .60), (0, .5)])
    def forbidden(*args, **kwargs):
        pytest.fail("Clear pursuit must not invoke avoidance")
    monkeypatch.setattr("runtime.select_marvin_detour", forbidden)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert motions(events) == [("forward", .10, .50)]
    assert result["local_avoidance_actions"] == 0 and not result["local_avoidance_active"]


@pytest.mark.parametrize("sides,direction", [
    ((1.2, .48), "LEFT"), ((.48, 1.2), "RIGHT"),
    ((1.3, .8), "LEFT"), ((.8, 1.3), "RIGHT"),
    ((.9, .9), "LEFT"), ((.9, .905), "LEFT"),
])
def test_one_safe_bounded_strafe_then_new_evidence_and_ordinary_approach(
        tmp_path, monkeypatch, sides, direction):
    runtime, behavior, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .58), (0, .5)], [sides, None])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert motions(events) == [("strafe", .08 if direction == "LEFT" else -.08, 1.), ("forward", .10, .50)]
    assert [row["state"] for row in result["history"]] == ["AVOIDING", "ADVANCING"]
    assert result["local_avoidance_actions"] == 1 and not result["local_avoidance_active"]
    assert result["last_detour_improved_direct_path"] is True
    assert len(set(behavior.stamps)) == 3
    assert len(runtime._marvin_alignment_consumed_source_frame_stamps) == 2
    for row, wait in zip(result["history"], result["lidar_wait_history"]):
        assert wait["snapshot"]["acquisition_sequence"] > row["action_lidar_evidence"][1]
        index = events.index(("strafe", .08 if direction == "LEFT" else -.08, 1.)) if row["state"] == "AVOIDING" else events.index(("forward", .10, .50))
        assert events[index + 1] == "stop"
    detour = result["history"][0]["result"]
    assert detour["action"] == "single_marvin_local_strafe"
    assert detour["local_detour"]["direction"] == direction
    assert result["progress_diagnostics"]["actions"][0]["type"] == "detour_strafe"
    assert result["final_observation"]["arrival"]["target_distance_m"] <= .50
    assert robot.status()["motion"] == {"linear_x": 0, "linear_y":0, "angular_z": 0, "streaming": False}


@pytest.mark.parametrize("sides", [(.46, .46), (.40, 1.2), (1.2, .40)])
def test_no_escape_or_unsafe_rotational_circle_never_moves(tmp_path, monkeypatch, sides):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch, [(0, .8)], [sides])
    result = run(runtime)
    assert result["state"] == "BLOCKED" and motions(events) == []
    assert result["local_avoidance_actions"] == 0


def test_off_center_after_detour_uses_existing_alignment(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (120, .60), (0, .58), (0, .5)], [(1.2, .48), None])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [row["state"] for row in result["history"]] == ["AVOIDING", "ALIGNING", "ADVANCING"]
    assert motions(events)[:2] == [("strafe", .08, 1.), ("turn", "RIGHT", .25, .50)]
    assert result["local_avoidance_actions"] == 1


def test_tracker_loss_after_detour_uses_real_semantic_recovery(tmp_path, monkeypatch):
    bundle, _, _ = strafe_runtime(tmp_path,monkeypatch,
        [(0,.60),(0,.59),(0,.58),(0,.5)], [[(.485,0.),(0,1.2),(0,-.48)],None],factory=recovery_runtime)
    runtime, behavior, _, events, _ = bundle
    result = run(runtime)
    assert result["state"] == "ARRIVED" and result["reacquisition_attempts"] == 1
    assert result["reacquisition_history"][0]["succeeded"]
    assert result["history"][0]["state"] == "AVOIDING"
    assert result["local_avoidance_actions"] == 1
    assert behavior.semantic_calls == 2
    assert len(behavior.created_trackers) == 2
    assert events.index("reacquire") > events.index(("strafe", .08, 1.))


@pytest.mark.parametrize("fault", ["stale", "invalid", "session", "coverage"])
def test_sensor_failures_do_not_enter_avoidance(tmp_path, monkeypatch, fault):
    runtime, behavior, _, events, _ = avoidance_runtime(tmp_path, monkeypatch, [(0, .8)], [(1.2, .48)])
    original = runtime.world_model.get_lidar_obstacles
    def read(**kwargs):
        state = original(**kwargs)
        if fault == "stale":
            state.update(valid=False, reason="stale", effective_age_seconds=.300001)
        elif fault == "invalid":
            state.update(valid=False, reason="invalid_geometry")
        elif fault == "session":
            state["producer_session"] = "different-producer"
        else:
            state["local_motion_geometry"]["sectors"]["front"]["valid_sample_count"] = 0
        return state
    runtime.world_model.get_lidar_obstacles = read
    monkeypatch.setattr("runtime.select_marvin_detour", lambda *a, **k: pytest.fail("Invalid evidence must not enter avoidance"))
    result = run(runtime)
    assert result["state"] == "BLOCKED" and motions(events) == []
    assert not result["local_avoidance_active"]


def test_duplicate_camera_stamp_cannot_dispatch_second_detour(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .8)], [(1.2, .48)])
    robot.on_motion = lambda: setattr(behavior, "repeat_camera", True)
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert len(motions(events)) == 1 and result["local_avoidance_actions"] == 1


@pytest.mark.parametrize("phase", ["selection", "jit", "motion", "post_lidar"])
def test_stop_preempts_detour_and_no_followup_motion(tmp_path, monkeypatch, phase):
    runtime, behavior, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .7), (0, .5)], [(1.2, .48), None])
    stop = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    if phase == "selection":
        original = plan_phase_action
        def stop_select(*args, **kwargs):
            result = original(*args, **kwargs)
            stop()
            return result
        monkeypatch.setattr("runtime.plan_phase_action", stop_select)
    elif phase == "jit":
        guarded = behavior.execute_guarded_marvin_lateral_step
        def stop_jit(**kwargs):
            stop()
            return guarded(**kwargs)
        behavior.execute_guarded_marvin_lateral_step = stop_jit
    elif phase == "motion":
        robot.on_motion = stop
    else:
        original = runtime._wait_for_new_marvin_lidar_evidence
        def stop_wait(**kwargs):
            if motions(events):
                stop()  # This case specifically tests the physical post-action wait.
            return original(**kwargs)
        runtime._wait_for_new_marvin_lidar_evidence = stop_wait
    result = run(runtime)
    assert result["behavior"] == "STOP" and runtime.get_status()["runtime_state"] == "STOPPED"
    assert len(motions(events)) == (1 if phase in {"motion", "post_lidar"} else 0)
    assert events[-1] == "stop"


def test_six_detours_exhaust_mission_budget(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch, [(0, .8)] * 7, [(1.2, .48)])
    result = run(runtime)
    assert result["state"] == "BLOCKED" and result["reason"] == "find_marvin_local_avoidance_exhausted"
    assert result["local_avoidance_actions"] == 6 and len(motions(events)) == 6
    assert all(row["state"] == "AVOIDING" for row in result["history"])


def test_ranking_noise_does_not_produce_left_right_ping_pong(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60)] * 4 + [(0, .5)]*2, [(1.3, .8), (.8, 1.3), (1.3, .8), (.8, 1.3), None])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [motion[1] for motion in motions(events)] == [.08] * 4
    assert all(row["last_detour_improved_direct_path"] is False for row in result["local_avoidance_history"][1:])
    assert result["local_avoidance_history"][1]["previous_clearances"] is not None


def test_no_improvement_cannot_reverse_when_previous_side_becomes_blocked(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .8)], [(1.3, .8), (.46, .8)])
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "find_marvin_blocked_wait_exhausted"
    assert len(motions(events)) == 1


def test_reversal_requires_new_geometry_evidence(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .60), (0, .5)], [(1.3, .8), (.46, 1.3)])
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert motions(events)==[("strafe",.08,1.)]  # Commitment does not flip to an unrelated hemisphere.


def test_jit_scene_change_cannot_use_advisory_selection_to_authorize_motion(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = avoidance_runtime(tmp_path, monkeypatch, [(0, .8)], [(1.2, .48)])
    original = runtime._dispatch_marvin_observation_action
    def dispatch(*args, **kwargs):
        behavior.unsafe_forward = True  # Put a fresh return inside the hard circle at JIT.
        return original(*args, **kwargs)
    runtime._dispatch_marvin_observation_action = dispatch
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["history"][0]["result"]["reason"] == "marvin_local_detour_jit_veto"
    assert motions(events) == []


def test_avoidance_diagnostics_exposed_without_becoming_motion_authority(tmp_path, monkeypatch):
    runtime, _, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .5)]*2, [(1.2, .48), None])
    seen = []
    def capture():
        seen.append(runtime.get_status_summary()["tracking"])
    robot.on_motion = capture
    result = run(runtime)
    assert seen[0]["local_avoidance_active"] and seen[0]["direct_path_blocked"]
    assert seen[0]["left_clearance_m"] > seen[0]["right_clearance_m"]
    assert result["final_observation"]["arrival"]["hard_safety_envelope_m"] == .45
    assert result["final_observation"]["arrival"]["target_standoff_m"] == .50
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == .30
    assert runtime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0


def test_uncertain_dispatched_strafe_counts_once_and_terminates(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0,.8)], [(1.2,.48)])
    original = behavior.execute_guarded_marvin_lateral_step
    def uncertain(**kwargs):
        result = original(**kwargs)
        value=dict(result,ok=False,motion_executed=False)
        # Native client invalidation can preserve a confirmed bounded receipt
        # while marking the enclosing dispatch uncertain. It still uses one slot.
        receipt=dict(value['lateral_result'])
        value['lateral_result'].update(delivery_uncertain=True,
            transport_attempted=True, transport_result=receipt)
        return value
    behavior.execute_guarded_marvin_lateral_step=uncertain
    result=run(runtime)
    assert result['state']=='BLOCKED'
    assert len(motions(events))==result['local_avoidance_actions']==1
    assert result['local_avoidance_history'][0]['physical_dispatch_confirmed']
    assert not result['local_avoidance_history'][0]['motion_executed']
    assert events[-1]=='stop'
