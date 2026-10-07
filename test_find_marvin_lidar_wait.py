"""Stopped LiDAR generation waits using offline robot and producer doubles."""
import time

import pytest

from lidar_perception import read_lidar_state
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_find_marvin_reacquisition import recovery_runtime


def controlled_producer(bundle, monkeypatch, *, publish=True, before_stop=False, real_freshness=False):
    """Publish only on explicit simulated producer ticks, never on a read."""
    runtime, behavior, robot, events, clock = bundle
    behavior.freeze_lidar = True
    behavior.sequence = 42
    read = behavior.lidar
    state = {"waiting": False, "sleeps": 0, "reads": [], "published": [],
             "fault": None, "on_read": None, "on_sleep": None}
    receipts = {}

    def acquisition():
        behavior.sequence += 1  # The test producer, not the runtime, owns this.
        state["published"].append(behavior.sequence)
        state["waiting"] = False
        events.append("lidar_published")

    def motion_completed():
        state["waiting"] = True
        state["sleeps"] = 0
        if before_stop:
            acquisition()

    def telemetry(**kwargs):
        snapshot = read(**kwargs)
        if real_freshness:
            receipt = receipts.setdefault(snapshot["acquisition_sequence"], time.monotonic())
            snapshot["received_monotonic_seconds"] = receipt
            snapshot = read_lidar_state(snapshot, expected_session=runtime.lidar_worker.session)
        state["reads"].append((snapshot["acquisition_sequence"], len(motions(events))))
        if state["on_read"]:
            state["on_read"]()
        if state["published"] and state["fault"]:
            state["fault"](snapshot)
        return snapshot

    def sleep(seconds):
        assert robot.status()["motion"] == {
            "linear_x": 0.0, "angular_z": 0.0, "streaming": False}
        clock[0] += max(1, round(seconds * 1e9))
        if state["waiting"]:
            # LiDAR must advance before any next-cycle observation/semantics.
            assert events[-1] in {"stop", "lidar_wait_sleep"}
            events.append("lidar_wait_sleep")
            state["sleeps"] += 1
            if state["on_sleep"]:
                state["on_sleep"]()
            if publish and state["sleeps"] == 2:
                acquisition()

    robot.on_motion = motion_completed
    runtime.world_model.get_lidar_obstacles = telemetry
    monkeypatch.setattr("runtime.time.sleep", sleep)
    return state


@pytest.mark.parametrize("initial", ["absent", (120, .60), (0, .60)])
def test_same_generation_after_stop_waits_then_first_new_scan_continues(
    tmp_path, monkeypatch, initial,
):
    bundle = make_runtime(tmp_path, monkeypatch, [initial, (0, .5)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert len(motions(events)) == 1
    assert result["history"][0]["action_lidar_evidence"] == ("offline-v2-session", 42)
    wait, = result["lidar_wait_history"]
    assert wait["previous_acquisition_sequence"] == 42
    assert wait["snapshot"]["acquisition_sequence"] == 43
    assert wait["poll_count"] == 3
    assert wait["wait_elapsed_seconds"] == pytest.approx(.10)
    assert producer["published"] == [43]  # No N+2 required, even for arrival.
    assert all(seq == 43 for seq, count in producer["reads"][-3:])
    assert events.index("lidar_published") < len(events) - 1 - events[::-1].index("observe")


@pytest.mark.parametrize("initial", [(120, .60), (0, .60)])
def test_scan_published_during_completed_action_is_not_rebaselined(
    tmp_path, monkeypatch, initial,
):
    bundle = make_runtime(tmp_path, monkeypatch, [initial, (0, .5)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch, before_stop=True)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["history"][0]["action_lidar_evidence"][1] == 42
    assert result["lidar_wait_history"][0]["snapshot"]["acquisition_sequence"] == 43
    assert result["lidar_wait_history"][0]["poll_count"] == 1
    assert "lidar_wait_sleep" not in events
    assert producer["published"] == [43]


def test_frozen_generation_times_out_stopped_before_camera_or_additional_action(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime, _, robot, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch, publish=False, real_freshness=True)
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "find_marvin_new_lidar_evidence_timeout"
    assert len(motions(events)) == events.count("observe") == 1
    assert producer["published"] == []
    wait, = result["lidar_wait_history"]
    assert wait["wait_elapsed_seconds"] == pytest.approx(.60, abs=.000001)
    assert wait["poll_count"] <= 13
    assert events[-1] == "stop"
    assert robot.status()["motion"]["streaming"] is False


def test_previous_scan_aged_out_during_action_still_waits_for_fresh_n_plus_one(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .60), (0, .5)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch, real_freshness=True)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert producer["published"] == [43]
    assert result["lidar_wait_history"][0]["snapshot"]["valid"] is True
    assert len(motions(events)) == 1


@pytest.mark.parametrize("fault,reason", [
    ("stale", "find_marvin_lidar_not_current"),
    ("invalid", "find_marvin_lidar_not_current"),
    ("unavailable", "find_marvin_lidar_not_current"),
    ("geometry", "find_marvin_lidar_not_current"),
    ("session", "find_marvin_lidar_producer_session_changed"),
    ("bool_sequence", "find_marvin_lidar_acquisition_sequence_invalid"),
    ("regressed_sequence", "find_marvin_lidar_acquisition_sequence_invalid"),
])
def test_new_but_untrusted_scan_blocks_without_waiting_out_timeout(tmp_path, monkeypatch, fault, reason):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)

    def corrupt(snapshot):
        if fault == "stale":
            snapshot["received_monotonic_seconds"] -= .300001
            snapshot.update(read_lidar_state(snapshot, expected_session=runtime.lidar_worker.session))
        elif fault in {"invalid", "unavailable"}:
            snapshot["valid" if fault == "invalid" else "available"] = False
        elif fault == "geometry":
            snapshot["local_motion_geometry"]["valid"] = False
        elif fault == "session":
            snapshot["producer_session"] = "different-producer"
        elif fault == "bool_sequence":
            snapshot["acquisition_sequence"] = True
        else:
            snapshot["acquisition_sequence"] = 41

    producer["fault"] = corrupt
    result = run(runtime)
    assert result["state"] == "BLOCKED" and result["reason"] == reason
    assert len(motions(events)) == events.count("observe") == 1
    assert result["lidar_wait_history"][0]["wait_elapsed_seconds"] == pytest.approx(.10)
    assert events[-1] == "stop"


@pytest.mark.parametrize("during", ["poll", "sleep"])
def test_stop_during_wait_preempts_immediately(tmp_path, monkeypatch, during):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch, publish=False)
    stop = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    if during == "poll":
        producer["on_read"] = lambda: stop() if producer["waiting"] else None
    else:
        producer["on_sleep"] = stop
    result = run(runtime)
    assert result["behavior"] == "STOP"
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert len(motions(events)) == events.count("observe") == 1
    assert producer["sleeps"] == (1 if during == "sleep" else 0)
    assert events[-1] == "stop"


def test_worker_session_change_while_waiting_is_blocked(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch, publish=False)
    producer["on_sleep"] = lambda: setattr(runtime.lidar_worker, "session", "replacement")
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "find_marvin_lidar_producer_session_changed"
    assert len(motions(events)) == 1


def test_new_valid_scan_with_unsafe_geometry_still_hits_forward_safety_veto(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .75)])
    runtime, behavior, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    producer["on_sleep"] = lambda: setattr(behavior, "unsafe_forward", True)
    result = run(runtime)
    assert result["lidar_wait_history"][0]["ok"] is True
    assert result["state"] == "BLOCKED"
    assert len(motions(events)) == 1
    assert result["history"][-1]["result"]["motion_executed"] is False
    assert result["history"][-1]["result"]["approach_result"]["forward_safety"]["permitted"] is False


def test_each_forward_step_consumes_distinct_generations_and_arrives_at_standoff(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .60), (0, .59), (0, .58), (0, .50)])
    runtime, _, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [row["action_lidar_evidence"][1] for row in result["history"]] == [42, 43, 44]
    assert [row["snapshot"]["acquisition_sequence"] for row in result["lidar_wait_history"]] == [43, 44, 45]
    assert producer["published"] == [43, 44, 45]
    assert len(motions(events)) == 3
    assert all(row["observation"]["arrival"]["hard_safety_envelope_m"] == .45 for row in result["history"])
    assert all(row["observation"]["arrival"]["target_standoff_m"] == .50 for row in result["history"])


def test_tracker_recovery_alignment_wait_and_forward_continue_in_same_mission(tmp_path, monkeypatch):
    bundle = recovery_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .59), (120, .58), (0, .58), (0, .50)])
    runtime, behavior, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert [row["state"] for row in result["history"]] == ["ADVANCING", "ALIGNING", "ADVANCING"]
    assert result["reacquisition_attempts"] == len(result["tracker_loss_history"]) == 1
    assert behavior.semantic_calls == len(behavior.created_trackers) == 2
    assert producer["published"] == [43, 44, 45]
    recovery = result["reacquisition_history"][0]["observation"]
    assert recovery["identity_source_frame_stamp_ns"] < recovery["source_frame_stamp_ns"]
    assert result["history"][1]["observation"] is recovery
    assert result["lidar_wait_history"][1]["previous_acquisition_sequence"] == 43
    assert result["lidar_wait_history"][1]["snapshot"]["acquisition_sequence"] == 44
    assert runtime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0
    stamps = [row["source_frame_stamp_ns"] for row in result["history"]]
    assert len(set(stamps)) == 3
    for stamp in stamps:
        retry = runtime.execute_single_marvin_approach(
            linear_speed=.10, duration=.50, source_frame_stamp_ns=stamp)
        assert retry["motion_executed"] is False
    assert len(motions(events)) == 3


def test_actual_turn_jit_generation_is_retained_instead_of_runtime_prefetch(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(120, .8), (0, .5)])
    runtime, behavior, _, _, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    turn = behavior._execute_target_directed_turn

    def jit_turn(*args, **kwargs):
        behavior.sequence = 43  # A publication between runtime and JIT reads.
        return turn(*args, **kwargs)

    behavior._execute_target_directed_turn = jit_turn
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["history"][0]["action_lidar_evidence"][1] == 43
    assert result["lidar_wait_history"][0]["previous_acquisition_sequence"] == 43
    assert producer["published"] == [44]


def test_production_turn_preserves_jit_evidence_when_monitor_sees_later_scans():
    from test_guarded_turn_execution import SESSION, FakeRobot, SequenceWorldModel, manager, snapshot
    first, later = snapshot(), snapshot()
    first["acquisition_sequence"] = 42
    later["acquisition_sequence"] = 43
    behavior = manager(first, FakeRobot())
    behavior.world_model = SequenceWorldModel([first, later])
    result = behavior.execute_guarded_turn(
        "LEFT", .25, .10, expected_lidar_session=SESSION, now=10.0)
    assert result["ok"] is True and result["confirmed_forwarded"] is True
    assert len(behavior.world_model.calls) >= 2
    assert result["action_lidar_evidence"] == {
        "producer_session": SESSION, "acquisition_sequence": 42}


def test_lidar_wait_does_not_age_the_next_camera_action_observation(tmp_path, monkeypatch):
    bundle = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    runtime, behavior, _, events, _ = bundle
    producer = controlled_producer(bundle, monkeypatch)
    producer["on_sleep"] = lambda: setattr(behavior, "stale_camera", True)
    result = run(runtime)
    assert result["lidar_wait_history"][0]["ok"] is True
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "marvin_motion_observation_stale"
    assert len(motions(events)) == 1


def test_freezing_after_reacquisition_can_consume_one_new_scan_but_never_reuse_it(tmp_path, monkeypatch):
    # A valid N+1 already exists when semantics run. Freezing after that
    # permits it once, then the following action must await N+2 or time out.
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .76), (0, .70)])
    behavior.on_semantic = lambda: setattr(behavior, "freeze_lidar", True) if behavior.semantic_calls == 2 else None
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "find_marvin_new_lidar_evidence_timeout"
    assert result["reacquisition_attempts"] == 1
    first, resumed = result["history"]
    assert first["action_lidar_evidence"][1] < resumed["action_lidar_evidence"][1]
    assert len(motions(events)) == 2
    assert result["lidar_wait_history"][-1]["ok"] is False
