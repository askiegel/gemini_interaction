"""Offline full missions for cached frames, marginal matches and recovery bounds."""
import pytest

from marvin_local_tracker import MarvinLocalTracker
from test_find_marvin_closed_loop import delayed_runtime, motions, run
from test_find_marvin_reacquisition import recovery_runtime


def camera_schedule(bundle, tokens):
    runtime, behavior, robot, events, _ = bundle
    original = behavior.semantic_vision.fetch_frame
    schedule = iter(tokens)
    seen = []

    def fetch():
        episode = behavior._marvin_v2_tracker_episode or {}
        pending = episode.get("post_action_pending") is True
        frame = original()
        if pending:
            token = next(schedule, "fresh")
            if token == "action":
                frame.source_frame_stamp_ns = episode["post_action_source_frame_stamp_ns"]
            elif token == "last":
                frame.source_frame_stamp_ns = seen[-1]
            seen.append(frame.source_frame_stamp_ns)
            assert robot.status()["motion"] == {
                "linear_x": 0.0, "angular_z": 0.0, "streaming": False}
            assert events[-1] == "observe"
        return frame

    behavior.semantic_vision.fetch_frame = fetch
    return seen


def post_action_qualities(behavior, qualities):
    factory = behavior.marvin_local_tracker_factory
    values = iter(qualities)

    def factory_with_quality(frame, box):
        local = factory(frame, box)
        update = local.update

        def scheduled(fresh):
            bbox = update(fresh)
            if (behavior._marvin_v2_tracker_episode or {}).get("post_action_pending") is True:
                local.last_quality = next(values, .95)
                if local.last_quality < .80:
                    local.last_reason = "below_threshold"
                    return None
            return bbox

        local.update = scheduled
        return local

    behavior.marvin_local_tracker_factory = factory_with_quality


@pytest.mark.parametrize("cached_polls", [1, 4, 7])
def test_cached_action_stamp_waits_without_tracker_evaluation_or_semantic_budget(
    tmp_path, monkeypatch, cached_polls,
):
    bundle = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    runtime, behavior, _, events, _ = bundle
    seen = camera_schedule(bundle, ["action"] * cached_polls + ["fresh"])
    # Allow the same production-bounded behavior with enough simulated time.
    behavior.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS = .30
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == len(result["tracker_loss_history"]) == 0
    assert behavior.semantic_calls == len(behavior.created_trackers) == 1
    action_stamp = result["history"][0]["source_frame_stamp_ns"]
    assert seen[:cached_polls] == [action_stamp] * cached_polls
    assert behavior.updated_frames.count(action_stamp) == 1  # Its original admission only.
    assert seen[-1] > action_stamp
    diag = result["final_observation"]["post_action_tracker_diagnostics"]
    assert diag["fresh_frames_attempted"] == 1
    assert diag["cached_frame_count"] == cached_polls
    assert all(frame["tracker_evaluated"] is False for frame in diag["frames"][:-1])
    assert len(motions(events)) == 1


def test_marginal_new_frame_then_cached_copies_then_independent_good_frame_continues(
    tmp_path, monkeypatch,
):
    bundle = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    runtime, behavior, _, events, _ = bundle
    seen = camera_schedule(bundle, ["fresh", "last", "last", "fresh"])
    post_action_qualities(behavior, [.786365, .91])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    diag = result["final_observation"]["post_action_tracker_diagnostics"]
    assert seen[0] == seen[1] == seen[2] < seen[3]
    assert diag["fresh_frames_attempted"] == 2 and diag["cached_frame_count"] == 2
    bad, _, _, good = diag["frames"]
    assert bad["opencv_tracker"]["quality"] == .786365
    assert bad["opencv_tracker"]["matched"] is False
    assert good["opencv_tracker"]["quality"] == .91
    assert good["opencv_tracker"]["matched"] is True
    assert result["final_observation"]["source_frame_stamp_ns"] == seen[-1]
    assert result["reacquisition_attempts"] == 0 and behavior.semantic_calls == 1
    assert MarvinLocalTracker.MIN_MATCH_QUALITY == .80
    assert len(motions(events)) == 1


def test_three_fresh_below_threshold_frames_enter_semantic_reacquisition(tmp_path, monkeypatch):
    bundle = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .70), (0, .5)], failures=())
    runtime, behavior, _, events, _ = bundle
    post_action_qualities(behavior, [.786365, .79, .799999])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 1 and behavior.semantic_calls == 2
    loss, = result["tracker_loss_history"]
    diag = loss["post_action_tracker_diagnostics"]
    assert diag["fresh_frames_attempted"] == diag["maximum_fresh_frames"] == 3
    assert diag["failure_reason"] == "below_threshold"
    stamps = [frame["source_frame_stamp_ns"] for frame in diag["frames"]]
    assert stamps == sorted(set(stamps))
    assert all(frame["opencv_tracker"]["quality"] < .80 for frame in diag["frames"])
    assert len(motions(events)) == 2


def test_one_bad_fresh_frame_then_frozen_camera_cannot_be_accepted(tmp_path, monkeypatch):
    bundle = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .70), (0, .5)], failures=())
    runtime, behavior, _, _, _ = bundle
    camera_schedule(bundle, ["fresh"] + ["last"] * 10)
    post_action_qualities(behavior, [.786365])
    result = run(runtime)
    # The only genuine observation failed quality. Timeout cannot turn it
    # into continuity; the mission must reacquire identity while stopped.
    assert result["reacquisition_attempts"] >= 1
    loss = result["tracker_loss_history"][0]
    diag = loss["post_action_tracker_diagnostics"]
    assert diag["fresh_frames_attempted"] == 1
    assert diag["camera_new_frame_timeout"] is True
    assert diag["failure_reason"] == "below_threshold"


def test_success_resets_failure_counter_across_multiple_recovery_episodes(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), "absent", "absent", (0, .75),
         (0, .71), "absent", "absent", (0, .70), (0, .5)], failures=(3, 8))
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 6
    assert result["consecutive_reacquisition_failures"] == 0
    assert [row["consecutive_failures_after"] for row in result["reacquisition_history"]] == [1, 2, 0, 1, 2, 0]
    assert len(motions(events)) == 3
    assert behavior.semantic_calls == 7


def test_repeated_successful_recovery_still_has_a_mission_backstop(tmp_path, monkeypatch):
    runtime, _, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .75), (0, .71), (0, .70), (0, .66)],
        failures=(3, 8, 13))
    runtime.MAX_MARVIN_REACQUISITION_EPISODES = 2
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_recovery_backstop_exhausted"
    assert result["reacquisition_attempts"] == 2
    assert result["consecutive_reacquisition_failures"] == 0
    assert len(motions(events)) == 3
    assert events[-1] == "stop"


def test_stop_during_cached_camera_poll_preempts_without_semantic_or_additional_motion(tmp_path, monkeypatch):
    bundle = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .76)])
    runtime, behavior, _, events, _ = bundle
    camera_schedule(bundle, ["action"] * 10)
    sleep = __import__("time").sleep

    def stop_in_poll(seconds):
        episode = behavior._marvin_v2_tracker_episode or {}
        if episode.get("post_action_pending") is True:
            runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
        sleep(seconds)

    monkeypatch.setattr("behavior_manager.time.sleep", stop_in_poll)
    result = run(runtime)
    assert result["behavior"] == "STOP"
    assert len(motions(events)) == 1
    assert behavior.semantic_calls == 1
    assert events[-1] == "stop"


def test_initial_semantic_tracker_still_requires_two_independent_support_frames(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [(0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED" and motions(events) == []
    assert len(behavior.updated_frames) == behavior.MARVIN_LOCAL_TRACKER_MIN_SUPPORT == 2
    assert runtime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0
    assert runtime.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS == .60
    assert result["final_observation"]["arrival"]["hard_safety_envelope_m"] == .45
    assert result["final_observation"]["arrival"]["target_standoff_m"] == .50


def test_only_a_frame_independently_meeting_exact_threshold_can_release_refresh(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    post_action_qualities(behavior, [.786365, .799999, .80])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    diag = result["final_observation"]["post_action_tracker_diagnostics"]
    assert diag["fresh_frames_attempted"] == 3
    assert [frame["opencv_tracker"]["matched"] for frame in diag["frames"]] == [False, False, True]
    assert result["final_observation"]["opencv_tracker"]["quality"] == .80
    assert result["reacquisition_attempts"] == 0
    assert len(motions(events)) == 1


def test_good_tracker_result_after_refresh_deadline_is_not_admitted(tmp_path, monkeypatch):
    runtime, behavior, _, events, clock = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .70), (0, .5)], failures=())
    factory = behavior.marvin_local_tracker_factory

    def slow_factory(frame, box):
        local = factory(frame, box)
        update = local.update

        def slow_update(fresh):
            result = update(fresh)
            if ((behavior._marvin_v2_tracker_episode or {}).get("post_action_pending") is True
                    and local is behavior.created_trackers[0]):
                clock[0] += 150_000_000
            return result

        local.update = slow_update
        return local

    behavior.marvin_local_tracker_factory = slow_factory
    result = run(runtime)
    assert result["tracker_loss_history"][0]["post_action_tracker_diagnostics"]["failure_reason"] == "post_action_tracker_refresh_timeout"
    assert result["state"] == "ARRIVED"
    assert result["reacquisition_attempts"] == 1
    assert len(motions(events)) == 2
