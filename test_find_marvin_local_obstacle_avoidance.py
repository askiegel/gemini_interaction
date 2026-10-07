"""Offline missions: controlled sensors/transport, production safety/ownership."""
import math

import pytest

from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
from marvin_local_obstacle_avoidance import select_marvin_detour
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
    return read


def avoidance_runtime(tmp_path, monkeypatch, specs, modes):
    bundle = make_runtime(tmp_path, monkeypatch, specs)
    runtime, behavior, _, events, _ = bundle
    obstacle_geometry(runtime, behavior, events, modes)
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
def test_one_safe_bounded_turn_then_new_evidence_and_ordinary_approach(
        tmp_path, monkeypatch, sides, direction):
    runtime, behavior, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .58), (0, .5)], [sides, None])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert motions(events) == [("turn", direction, .25, .50), ("forward", .10, .50)]
    assert [row["state"] for row in result["history"]] == ["AVOIDING", "ADVANCING"]
    assert result["local_avoidance_actions"] == 1 and not result["local_avoidance_active"]
    assert result["last_detour_improved_direct_path"] is True
    assert len(set(behavior.stamps)) == 3
    assert len(runtime._marvin_alignment_consumed_source_frame_stamps) == 2
    for row, wait in zip(result["history"], result["lidar_wait_history"]):
        assert wait["snapshot"]["acquisition_sequence"] > row["action_lidar_evidence"][1]
        index = events.index(("turn", direction, .25, .50)) if row["state"] == "AVOIDING" else events.index(("forward", .10, .50))
        assert events[index + 1] == "stop"
    detour = result["history"][0]["result"]
    assert detour["action"] == "single_marvin_local_detour_turn"
    assert detour["local_detour"]["direction"] == direction
    assert result["progress_diagnostics"]["actions"][0]["type"] == "detour_turn"
    assert result["final_observation"]["arrival"]["target_distance_m"] <= .50
    assert robot.status()["motion"] == {"linear_x": 0, "angular_z": 0, "streaming": False}


@pytest.mark.parametrize("sides", [(.48, .48), (.40, 1.2), (1.2, .40)])
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
    assert motions(events)[:2] == [("turn", "LEFT", .25, .50), ("turn", "RIGHT", .25, .50)]
    assert result["local_avoidance_actions"] == 1


def test_tracker_loss_after_detour_uses_real_semantic_recovery(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .59), (0, .58), (0, .5)])
    obstacle_geometry(runtime, behavior, events, [(1.2, .48), None])
    result = run(runtime)
    assert result["state"] == "ARRIVED" and result["reacquisition_attempts"] == 1
    assert result["reacquisition_history"][0]["succeeded"]
    assert result["history"][0]["state"] == "AVOIDING"
    assert result["local_avoidance_actions"] == 1
    assert behavior.semantic_calls == 2
    assert len(behavior.created_trackers) == 2
    assert events.index("reacquire") > events.index(("turn", "LEFT", .25, .50))


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
        original = select_marvin_detour
        def stop_select(*args, **kwargs):
            result = original(*args, **kwargs)
            stop()
            return result
        monkeypatch.setattr("runtime.select_marvin_detour", stop_select)
    elif phase == "jit":
        behavior.after_guard_check = stop
    elif phase == "motion":
        robot.on_motion = stop
    else:
        original = runtime._wait_for_new_marvin_lidar_evidence
        def stop_wait(**kwargs):
            stop()
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
        [(0, .60)] * 4 + [(0, .5)], [(1.3, .8), (.8, 1.3), (1.3, .8), (.8, 1.3)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [motion[1] for motion in motions(events)] == ["LEFT"] * 4
    assert all(row["last_detour_improved_direct_path"] is False for row in result["local_avoidance_history"][1:])
    assert result["local_avoidance_history"][1]["previous_clearances"] is not None


def test_no_improvement_cannot_reverse_when_previous_side_becomes_blocked(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .8)], [(1.3, .8), (.48, .8)])
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "find_marvin_local_avoidance_oscillation_blocked"
    assert len(motions(events)) == 1


def test_reversal_requires_new_geometry_evidence(tmp_path, monkeypatch):
    runtime, _, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .60), (0, .5)], [(1.3, .8), (.48, 1.3)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [motion[1] for motion in motions(events)] == ["LEFT", "RIGHT"]


def test_jit_scene_change_cannot_use_advisory_selection_to_authorize_motion(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = avoidance_runtime(tmp_path, monkeypatch, [(0, .8)], [(1.2, .48)])
    original = runtime._dispatch_marvin_observation_action
    def dispatch(*args, **kwargs):
        behavior.unsafe_forward = True  # Put a fresh return inside the hard circle at JIT.
        return original(*args, **kwargs)
    runtime._dispatch_marvin_observation_action = dispatch
    result = run(runtime)
    assert result["state"] == "BLOCKED" and result["reason"] == "marvin_local_detour_jit_veto"
    assert motions(events) == []


def test_avoidance_diagnostics_exposed_without_becoming_motion_authority(tmp_path, monkeypatch):
    runtime, _, robot, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .5)], [(1.2, .48)])
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


def test_interrupted_dispatched_turn_counts_once_and_terminates(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = avoidance_runtime(tmp_path, monkeypatch,
        [(0, .8)], [(1.2, .48)])
    original = behavior._execute_target_directed_turn
    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        return dict(result, ok=False, permitted=False, reason="stale")
    behavior._execute_target_directed_turn = interrupted
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert len(motions(events)) == result["local_avoidance_actions"] == 1
    assert result["local_avoidance_history"][0]["physical_dispatch_confirmed"]
    assert not result["local_avoidance_history"][0]["motion_executed"]
    assert events[-1] == "stop"
