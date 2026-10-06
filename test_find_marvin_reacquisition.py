"""Offline recovery missions using production runtime and identity/tracker path."""
from types import SimpleNamespace

import pytest

from behavior_manager import BehaviorManager
from test_find_marvin_closed_loop import delayed_runtime, run, motions


def recovery_runtime(tmp_path, monkeypatch, specs, *, failures=(3,), mode="no_bbox"):
    runtime, behavior, robot, events, clock = delayed_runtime(tmp_path, monkeypatch, specs)
    update_count = [0]
    factory = behavior.marvin_local_tracker_factory

    def tracker(frame, seed):
        local = factory(frame, seed)
        update = local.update

        def failing_update(fresh):
            box = update(fresh)
            update_count[0] += 1
            if any(start <= update_count[0] < start + behavior.MARVIN_LOCAL_TRACKER_MAX_FRAMES
                   for start in failures):
                if mode == "invalid_bbox":
                    return dict(x1=-1, y1=0, x2=20, y2=30)
                local.last_quality = .60
                return None
            return box

        local.update = failing_update
        return local

    behavior.marvin_local_tracker_factory = tracker
    reacquire = behavior.reacquire_find_marvin_v2

    def next_reacquisition(**kwargs):
        events.append("reacquire")
        assert events[-2] == "stop"
        assert robot.status()["motion"]["linear_x"] == 0
        assert behavior._marvin_v2_tracker_episode is None
        behavior.current_spec = next(behavior.specs)
        if behavior.current_spec != "absent":
            behavior.distance = behavior.current_spec[1]
        return reacquire(**kwargs)

    behavior.reacquire_find_marvin_v2 = next_reacquisition
    select = behavior.semantic_vision.select_marvin_candidate

    def semantic(frame, candidates):
        events.append("gemini")
        count = len(motions(events))
        result = select(frame, candidates)
        assert len(motions(events)) == count
        assert robot.status()["motion"]["linear_x"] == 0
        return result

    behavior.semantic_vision.select_marvin_candidate = semantic
    return runtime, behavior, robot, events, clock


@pytest.mark.parametrize("error", [0, 120])
@pytest.mark.parametrize("mode", ["no_bbox", "invalid_bbox"])
def test_tracker_loss_reacquires_new_episode_and_resumes_until_arrival(tmp_path, monkeypatch, error, mode):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(error, .8), (0, .76), (0, .7), (0, .5)], mode=mode)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["actions_executed"] == 2
    assert len(motions(events)) == 2
    assert motions(events)[0][0] == ("turn" if error else "forward")
    assert motions(events)[1][0] == "forward"
    assert behavior.semantic_calls == 2
    assert len(behavior.created_trackers) == 2
    assert behavior.created_trackers[0] is not behavior.created_trackers[1]
    assert result["reacquisition_attempts"] == 1
    first, resumed = result["history"]
    recovery = result["reacquisition_history"][0]
    fresh = recovery["observation"]
    assert fresh is resumed["observation"]
    assert fresh["identity_source"] == "gemini_marvin_candidate_selection"
    assert first["observation"]["marvin_tracking_episode"]["episode_id"] != fresh["marvin_tracking_episode"]["episode_id"]
    assert recovery["source_floor"] < fresh["identity_source_frame_stamp_ns"] < fresh["source_frame_stamp_ns"]
    assert fresh["received_monotonic_seconds"] > behavior.identity_receipts[1] + 2.5
    assert fresh["opencv_tracker"]["source_frame_stamp_ns"] == resumed["source_frame_stamp_ns"]
    assert fresh["arrival"]["hard_safety_envelope_m"] == .45
    assert fresh["arrival"]["target_standoff_m"] == .50
    assert runtime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0
    diagnostic = recovery["loss_observation"]["post_action_tracker_diagnostics"]
    assert diagnostic["fresh_frames_attempted"] == diagnostic["frames_attempted"] == 3
    sample = diagnostic["frames"][0]
    assert sample["source_frame_stamp_ns"] > first["source_frame_stamp_ns"]
    assert sample["received_monotonic_seconds"] > diagnostic["action_stopped_monotonic_seconds"]
    assert diagnostic["pre_action_tracker_bbox"] == first["observation"]["opencv_tracker"]["bbox"]
    assert sample["image_width"] == 640 and sample["image_height"] == 480
    assert sample["opencv_tracker"]["threshold"] == .8
    assert sample["update_returned_no_bbox"] == (mode == "no_bbox")
    assert sample["bbox_invalid"] == (mode == "invalid_bbox")
    assert diagnostic["stop_to_refresh_seconds"] >= 0
    assert diagnostic["pre_to_post_iou_used_for_admission"] is False
    assert events[-1] == "stop"
    assert runtime.mission_manager.active_mission is None


def test_negative_reacquisition_is_bounded_without_search_or_motion(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), "absent", "absent", "absent"])
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_semantic_reacquisition_exhausted"
    assert result["reacquisition_attempts"] == result["consecutive_reacquisition_failures"] == 3
    assert result["max_consecutive_reacquisition_failures"] == 3
    assert behavior.semantic_calls == 4
    assert len(behavior.created_trackers) == 1
    assert len(motions(events)) == 1
    assert result["search_turns"] == 0
    assert behavior._marvin_v2_tracker_episode is None
    assert runtime._marvin_alignment_observation is None
    assert events[-1] == "stop"


@pytest.mark.parametrize("during", ["semantic", "fresh_frame"])
def test_stop_during_reacquisition_preempts_without_followup_motion(tmp_path, monkeypatch, during):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7)])
    stop = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    if during == "semantic":
        behavior.on_semantic = lambda: stop() if behavior.semantic_calls == 2 else None
    else:
        behavior.on_refresh = lambda: stop() if behavior.semantic_calls == 2 else None
    result = run(runtime)
    assert result["behavior"] == "STOP"
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert len(motions(events)) == 1
    assert behavior._marvin_v2_tracker_episode is None
    assert runtime._marvin_alignment_observation is None


def test_retry_can_succeed_after_semantic_negative(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), "absent", (0, .7), (0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 2
    assert behavior.semantic_calls == 3
    assert len(motions(events)) == 2
    old, new = result["reacquisition_history"]
    assert new["observation"]["identity_source_frame_stamp_ns"] > new["source_floor"] >= old["observation"]["identity_source_frame_stamp_ns"]


@pytest.mark.parametrize("mode", ["cached_semantic", "cached_during_gemini", "unavailable"])
def test_reacquisition_cannot_move_without_post_gemini_fresh_frame(tmp_path, monkeypatch, mode):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7), (0, .7), (0, .7)])
    behavior.on_semantic = lambda: setattr(behavior, "refresh_mode", mode) if behavior.semantic_calls >= 2 else None
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reacquisition_attempts"] == 3
    assert len(motions(events)) == 1
    assert behavior._marvin_v2_tracker_episode is None
    assert runtime._marvin_alignment_observation is None


def test_three_successful_reacquisitions_do_not_exhaust_recovery(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .75), (0, .71), (0, .7), (0, .66), (0, .65),
         (0, .61), (0, .6), (0, .5)], failures=(3, 8, 13, 18))
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 4
    assert result["consecutive_reacquisition_failures"] == 0
    assert behavior.semantic_calls == 5
    assert len(motions(events)) == 5
    assert events[-1] == "stop"


@pytest.mark.parametrize("freeze", ["invalid_lidar", "unsafe_forward"])
def test_reacquired_identity_still_requires_jit_lidar_safety(tmp_path, monkeypatch, freeze):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7)])
    behavior.on_semantic = lambda: setattr(behavior, freeze, True) if behavior.semantic_calls == 2 else None
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert len(motions(events)) == 1
    assert events[-1] == "stop"


def test_recovery_preserves_consumed_action_stamps(tmp_path, monkeypatch):
    runtime, _, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(120, .8), (0, .76), (120, .7), (0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    stamps = [row["source_frame_stamp_ns"] for row in result["history"]]
    assert len(set(stamps)) == 2
    assert all(stamp in runtime._marvin_alignment_consumed_source_frame_stamps for stamp in stamps)
    for stamp in stamps:
        retry = runtime.execute_single_marvin_alignment(direction="LEFT", angular_speed=.25,
                                                        duration=.50, source_frame_stamp_ns=stamp)
        assert retry["motion_executed"] is False
    assert len(motions(events)) == 2


@pytest.mark.parametrize("which", ["identity", "action"])
def test_reacquisition_rejects_reused_identity_or_action_stamp(tmp_path, monkeypatch, which):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7), (0, .7), (0, .7)])
    observe = runtime._observe_find_marvin_v2

    def replay(**kwargs):
        result = observe(**kwargs)
        floor = kwargs.get("reacquisition_source_floor")
        if floor is not None:
            key = "identity_source_frame_stamp_ns" if which == "identity" else "source_frame_stamp_ns"
            result[key] = floor
        return result

    monkeypatch.setattr(runtime, "_observe_find_marvin_v2", replay)
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reacquisition_attempts"] == 3
    assert len(motions(events)) == 1
    assert behavior._marvin_v2_tracker_episode is None


def test_reacquired_action_frame_expires_after_one_second_at_dispatch(tmp_path, monkeypatch):
    runtime, behavior, _, events, clock = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7)])
    approach = behavior.execute_single_marvin_approach_step

    def expired(**kwargs):
        if behavior.semantic_calls == 2:
            clock[0] += 1_000_000_001
        return approach(**kwargs)

    monkeypatch.setattr(behavior, "execute_single_marvin_approach_step", expired)
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert len(motions(events)) == 1


@pytest.mark.parametrize("problem", ["cached_receipt", "geometry", "camera_error"])
def test_continuity_diagnostics_explain_frame_and_geometry_failures(monkeypatch, problem):
    import threading
    clock = [10.0]
    monkeypatch.setattr("behavior_manager.time.monotonic", lambda: clock[0])
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    box = dict(x1=100, y1=100, x2=200, y2=200)
    invalid = dict(x1=-1, y1=100, x2=200, y2=200)
    tracker = SimpleNamespace(update=lambda _frame: invalid if problem == "geometry" else box,
                              last_quality=.95, MIN_MATCH_QUALITY=.8)
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": tracker, "tracker_bbox": box,
        "last_tracker_source_frame_stamp_ns": 100,
        "last_tracker_received_at": "2026-10-05T00:00:00+00:00",
        "last_tracker_received_monotonic_seconds": 9.9,
        "identity_source": "gemini_marvin_candidate_selection",
        "identity_source_frame_stamp_ns": 90,
    }
    count = [0]

    def fetch():
        if problem == "camera_error":
            raise OSError("camera unavailable")
        count[0] += 1
        clock[0] += .01
        return SimpleNamespace(width=640, height=480, source_frame_stamp_ns=100+count[0],
            received_at="2026-10-05T00:00:00+00:00" if problem == "cached_receipt"
                        else f"2026-10-05T00:00:0{count[0]}+00:00",
            received_monotonic_seconds=clock[0])

    manager.semantic_vision = SimpleNamespace(fetch_frame=fetch)
    assert manager.mark_strict_v2_action_dispatched(100, "forward")
    result = manager._continue_strict_v2_tracker_after_action()
    assert result["reason"] == "post_action_tracker_continuity_lost"
    diagnostics = result["post_action_tracker_diagnostics"]
    expected = {"cached_receipt": "local_receipt_timestamp_not_newer",
                "geometry": "invalid_bbox", "camera_error": "camera unavailable"}
    assert diagnostics["failure_reason"] == expected[problem]
    assert diagnostics["frames_attempted"] == (0 if problem == "camera_error" else 3)
    if problem == "geometry":
        assert all(frame["bbox_invalid"] for frame in diagnostics["frames"])
    assert manager._marvin_v2_tracker_episode is None


def test_cached_post_action_frame_waits_without_semantic_reacquisition(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .5)], failures=())
    robot.on_motion = lambda: setattr(behavior, "refresh_mode", "cached_once")
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 0
    assert behavior.semantic_calls == 1
    assert len(behavior.created_trackers) == 1
    diagnostic = result["history"][1]["observation"]["post_action_tracker_diagnostics"]
    assert diagnostic["failure_reason"] is None
    assert diagnostic["cached_frame_count"] == 1
    assert diagnostic["frames"][0]["source_frame_stamp_ns"] < diagnostic["pre_action_source_frame_stamp_ns"]
    assert len(motions(events)) == 2


def test_permanently_cached_camera_bounds_recovery_without_semantic_or_motion(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .7), (0, .7), (0, .7)], failures=())
    robot.on_motion = lambda: setattr(behavior, "refresh_mode", "repeat_action")
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_post_action_camera_new_frame_timeout"
    assert result["reacquisition_attempts"] == 0
    assert behavior.semantic_calls == 1
    assert len(behavior.created_trackers) == 1
    diagnostic = result["final_observation"]["post_action_tracker_diagnostics"]
    assert diagnostic["frames"][0]["camera_returned_cached_frame"] is True
    assert len(motions(events)) == 1


@pytest.mark.parametrize("malformed", [dict(x1=1), dict(x1="bad", y1=0, x2=20, y2=30)])
def test_malformed_tracker_box_does_not_escape_recovery_diagnostics(tmp_path, monkeypatch, malformed):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .5)], failures=())
    factory = behavior.marvin_local_tracker_factory
    count = [0]

    def tracker(frame, seed):
        local = factory(frame, seed)
        update = local.update

        def invalid_once(fresh):
            result = update(fresh)
            count[0] += 1
            return malformed if count[0] == 3 else result

        local.update = invalid_once
        return local

    behavior.marvin_local_tracker_factory = tracker
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 0
    diagnostic = result["history"][1]["observation"]["post_action_tracker_diagnostics"]
    assert diagnostic["failure_reason"] is None
    assert diagnostic["fresh_frames_attempted"] == 2
    assert diagnostic["frames"][0]["bbox_invalid"] is True
    assert len(motions(events)) == 2
