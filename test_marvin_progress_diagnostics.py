"""Offline retention tests. All motion/ROS/HTTP transports are fakes."""
import math
import socket
import time

import pytest

from marvin_progress_diagnostics import MarvinProgressDiagnostics
from test_find_marvin_closed_loop import make_runtime, motions, run
from test_find_marvin_reacquisition import recovery_runtime


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live transport forbidden in diagnostic tests")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.delenv("MARVIN_ODOM_DIAGNOSTICS_ENABLED", raising=False)


def odom(receipt, stamp, *, x=0., y=0., yaw=0., age=0., publisher="existing-ekf"):
    return {"stamp_ns": stamp, "received_monotonic_seconds": receipt,
            "source_age_seconds": age, "frame_id": "odom", "child_frame_id": "base_footprint",
            "x": x, "y": y, "yaw": yaw, "linear_x": 0., "linear_y": 0., "angular_z": 0.,
            "publishers": [publisher]}


def observation(stamp=100, *, distance=.9, sequence=10):
    selected = {"x_m": distance, "y_m": .002, "distance_m": distance, "robot_bearing_deg": .1}
    return {"source_frame_stamp_ns": stamp, "received_monotonic_seconds": 1.,
            "identity_source": "marvin_locked_tracker_continuity",
            "opencv_tracker": {"quality": .91, "bbox": {"x1": 270, "x2": 370, "y1": 100, "y2": 300},
                               "image_width": 640, "image_height": 480},
            "arrival": {"ok": True, "measured_distance_m": distance, "target_distance_m": distance-.1,
                        "producer_session": "session", "acquisition_sequence": sequence,
                        "bearing_interval_degrees": [-5., 5.], "point_count": 11,
                        "selected_return": selected}}


def prepare(diagnostic, *, start=1., stop=1.5, sequence=10):
    observed = observation(sequence=sequence)
    diagnostic.prepare_action("ADVANCING", observed)
    diagnostic.command_event("start", {"start_monotonic_seconds": start, "linear_x": .10,
                                       "angular_z": 0., "duration": .5})
    diagnostic.command_event("complete", {"completion_monotonic_seconds": stop,
                                          "bridge_acknowledgement": {"ok": True, "automatic_stop": True,
                                                                     "returned_immediately": False}})
    diagnostic.action_result({"motion_executed": True, "approach_result": {
        "target_standoff": observed["arrival"],
        "action_lidar_evidence": {"producer_session": "session", "acquisition_sequence": sequence}}}, stop)


def scan(sequence, receipt, **overrides):
    return dict({"available": True, "valid": True, "effective_age_seconds": .02,
                 "producer_session": "session", "acquisition_sequence": sequence,
                 "received_monotonic_seconds": receipt, "source": {"stamp_seconds": 123.5},
                 "local_motion_geometry": {"valid": True, "frame_id": "lidar_link",
                                           "points": [{"x_m": .86, "y_m": .02}]}}, **overrides)


def test_pre_and_first_post_odom_are_correlated_and_deltas_are_diagnostic_only():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    diagnostic.record_odom(odom(.99, 10, yaw=math.pi-.01))
    diagnostic.record_odom(odom(1.2, 11, x=.015))  # During action, never post-STOP.
    prepare(diagnostic)
    diagnostic.record_odom(odom(1.51, 12, x=.03, y=.04, yaw=-math.pi+.02))
    diagnostic.record_odom(odom(1.52, 13, x=.1))
    report = diagnostic.snapshot()
    row = report["actions"][0]
    assert row["pre_odom"]["stamp_ns"] == 10 and row["post_odom"]["stamp_ns"] == 12
    assert row["odom_delta"] == pytest.approx({"x": .03, "y": .04, "planar_translation_m": .05, "yaw_radians": .03})
    assert report["action_summary"][0]["nominal_displacement_m"] == pytest.approx(.05)
    assert report["diagnostic_only"] is True
    assert report["odometry_source"]["independent_translation_verified"] is False


@pytest.mark.parametrize("bad", [
    odom(1.51, 10),  # Duplicate source stamp.
    odom(1.51, 11, age=.02),  # Received after STOP, measured before STOP.
    odom(1.51, 11, publisher="replacement"),
    dict(odom(1.51, 11), frame_id="map"),
    odom(1.51, 11, x=float("nan")),
    odom(1.51, 11, age=-1),
    dict(odom(1.51, 11), publishers=[]),
    dict(odom(1.51, 11), publishers=["first", "second"]),
])
def test_invalid_or_unrelated_post_odom_cannot_supply_progress(bad):
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    diagnostic.record_odom(odom(.99, 10))
    prepare(diagnostic)
    diagnostic.record_odom(bad)
    assert diagnostic.snapshot()["actions"][0]["post_odom"] is None


def test_missing_post_odom_cannot_be_assigned_from_the_next_action():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    diagnostic.record_odom(odom(.99, 10))
    prepare(diagnostic)
    prepare(diagnostic, start=1.6, stop=2.1, sequence=11)
    diagnostic.record_odom(odom(2.11, 12, x=.08))
    rows = diagnostic.snapshot()["actions"]
    assert rows[0]["post_odom"] is None
    assert rows[1]["post_odom"]["stamp_ns"] == 12


def test_first_new_raw_lidar_is_retained_before_target_reassociation():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    prepare(diagnostic)
    diagnostic.record_lidar(scan(10, 1.51))
    diagnostic.record_lidar(scan(11, 1.49))  # Before STOP.
    diagnostic.record_lidar(scan(12, 1.51, valid=False))
    diagnostic.record_lidar(scan(13, 1.52, producer_session="replacement"))
    diagnostic.record_lidar(scan(14, 1.53, effective_age_seconds=.31))
    diagnostic.record_lidar(scan(14, 1.53, local_motion_geometry={"valid": False, "points": []}))
    first = scan(15, 1.54)
    diagnostic.record_lidar(first)
    diagnostic.record_lidar(scan(16, 1.55))
    row = diagnostic.snapshot()["actions"][0]
    assert row["first_post_action_lidar"]["acquisition_sequence"] == 15
    assert row["first_post_action_lidar"]["local_motion_geometry"]["points"] == first["local_motion_geometry"]["points"]
    assert row["first_post_action_lidar"]["source"] == first["source"]
    assert row["first_post_action_lidar"]["target_associated"] is False
    assert row["next_target_association"] is None
    diagnostic.observe(observation(101, distance=.88, sequence=20), reacquisition=True)
    summary = diagnostic.snapshot()["action_summary"][0]
    assert summary["first_post_action_lidar_sequence"] == 15
    assert summary["target_range_delta_m"] == pytest.approx(-.02)
    assert summary["jit_selected_return"]["x_m"] == .9
    assert summary["next_selected_return"]["x_m"] == .88
    assert summary["reacquisition_used"] is True


def test_scan_delivered_during_bridge_response_is_not_lost():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    diagnostic.prepare_action("ADVANCING", observation())
    diagnostic.command_event("start", {"start_monotonic_seconds": 1., "linear_x": .10, "duration": .5})
    diagnostic.record_lidar(scan(11, 1.52))  # Callback wins the action-result race.
    diagnostic.command_event("complete", {"completion_monotonic_seconds": 1.5,
                                          "bridge_acknowledgement": {"ok": True, "automatic_stop": True,
                                                                     "returned_immediately": False}})
    diagnostic.action_result({"motion_executed": True, "approach_result": {"target_standoff": observation()["arrival"]}}, 1.6)
    assert diagnostic.snapshot()["actions"][0]["first_post_action_lidar"]["acquisition_sequence"] == 11


def test_samples_and_exported_reports_do_not_alias_control_observations():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    observed = observation()
    diagnostic.observe(observed)
    diagnostic.prepare_action("ADVANCING", observed)
    observed["opencv_tracker"]["bbox"]["x1"] = -100
    report = diagnostic.snapshot()
    report["actions"][0]["authorizing_camera"]["bbox"]["x1"] = -200
    assert diagnostic.snapshot()["actions"][0]["authorizing_camera"]["bbox"]["x1"] == 270


def test_retention_is_bounded_and_creating_diagnostics_does_no_io():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.start()  # Disabled unless explicitly configured.
    assert diagnostic._thread is None
    diagnostic.begin("one")
    for _ in range(270):
        diagnostic.prepare_action("ADVANCING", observation())
    assert len(diagnostic.snapshot()["actions"]) == diagnostic.MAX_ACTIONS
    assert diagnostic.snapshot()["actions_dropped"] == 14


@pytest.mark.parametrize("flag", ["unsafe_forward", "invalid_lidar", "stale_camera"])
def test_favorable_diagnostics_cannot_override_real_admission(tmp_path, monkeypatch, flag):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    setattr(behavior, flag, True)
    runtime.marvin_progress_diagnostics.record_odom(odom(time.monotonic(), 1, x=10))
    result = run(runtime)
    assert result["arrived_at_marvin"] is False
    assert motions(events) == []


@pytest.mark.parametrize("break_diagnostics", [False, True])
def test_missing_or_broken_diagnostics_do_not_change_motion_or_standoff(tmp_path, monkeypatch, break_diagnostics):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .75), (0, .5)])
    if break_diagnostics:
        def broken(*args, **kwargs):
            raise RuntimeError("diagnostic storage failed")
        for method in ("begin", "prepare_action", "command_event", "action_result", "observe", "terminal", "snapshot"):
            monkeypatch.setattr(runtime.marvin_progress_diagnostics, method, broken)
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert motions(events) == [("forward", .10, .5)] * 2
    assert result["final_observation"]["arrival"]["target_standoff_m"] == .5
    assert result["final_observation"]["arrival"]["hard_safety_envelope_m"] == .45
    if not break_diagnostics:
        summary = result["progress_diagnostics"]["action_summary"]
        assert [row["nominal_displacement_m"] for row in summary] == pytest.approx([.05, .05])
        assert all(row["odom_translation_m"] is None for row in summary)
        assert summary[0]["target_range_delta_m"] == pytest.approx(-.05)
        assert summary[0]["jit_selected_return"]["x_m"] == pytest.approx(.9)


@pytest.mark.parametrize("terminal", ["ARRIVED", "BLOCKED", "REVERIFY_REQUIRED", "STOPPED"])
def test_last_available_observation_survives_every_terminal_state(terminal):
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    good = observation()
    diagnostic.observe(good)
    diagnostic.observe({"ok": False, "perception_reason": "tracker_lost"})
    diagnostic.terminal(terminal, "test_reason")
    report = diagnostic.snapshot()["terminal"]
    assert report["state"] == terminal
    assert report["last_target_association"] == good["arrival"]
    assert report["last_camera"]["bbox"] == good["opencv_tracker"]["bbox"]
    assert report["last_camera"]["tracker_quality"] == .91


def test_stop_retains_mission_diagnostics_without_overwriting_operator_stop(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    robot.on_motion = lambda: runtime.submit_intent({"intent": "STOP"})
    result = run(runtime)
    assert result["behavior"] == "STOP" and result["state"] == "STOPPED"
    diagnostic = runtime.get_status()["marvin_progress_diagnostics"]
    assert diagnostic["terminal"]["state"] == "STOPPED"
    assert diagnostic["terminal"]["last_camera"]["source_frame_stamp_ns"] == behavior.stamps[0]
    assert len(motions(events)) == 1


def test_recovery_retains_first_new_tracker_and_correct_semantic_stamps(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .75), (0, .75), (0, .5)])
    # This fixture replaces BehaviorManager after runtime construction.
    behavior.marvin_command_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("command_event", *args)
    behavior.marvin_perception_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("perception_event", *args)
    result = run(runtime)
    assert result["state"] == "ARRIVED" and result["reacquisition_attempts"] == 1
    diagnostic = result["progress_diagnostics"]
    first = diagnostic["actions"][0]
    assert first["reacquisition_used"] is True
    assert len(first["semantic_reacquisitions"]) == 1
    semantic = first["semantic_reacquisitions"][0]
    recovery = result["reacquisition_history"][0]["observation"]
    assert semantic["source_frame_stamp_ns"] == recovery["identity_source_frame_stamp_ns"]
    assert semantic["source_frame_stamp_ns"] < recovery["source_frame_stamp_ns"]
    assert semantic["bbox"] is not None and semantic["center"] is not None
    assert semantic["image_width"] == 640 and semantic["image_height"] == 480
    assert first["first_new_post_action_camera"]["source_frame_stamp_ns"] > first["authorizing_camera"]["source_frame_stamp_ns"]
    assert first["tracker_refreshes"][0]["fresh_frames_attempted"] == 3
    assert len([r for r in first["post_action_frames"] if r.get("tracker_evaluated")]) == 3
    assert len(motions(events)) == 2


def test_diagnostic_report_cannot_reauthorize_a_consumed_source_stamp(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    result = run(runtime)
    stamp = result["progress_diagnostics"]["action_summary"][0]["source_frame_stamp_ns"]
    retry = runtime.execute_single_marvin_approach(linear_speed=.10, duration=.5, source_frame_stamp_ns=stamp)
    assert retry["motion_executed"] is False
    assert retry["reason"] == "marvin_approach_observation_already_consumed"
    assert len(motions(events)) == 1


def test_full_mission_keeps_independent_pose_and_first_scan_per_action(tmp_path, monkeypatch):
    runtime, behavior, robot, events, clock = make_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .75), (0, .5)])
    diagnostic = runtime.marvin_progress_diagnostics
    pose_x, odom_stamp = [0.], [0]

    def publish_pose():
        odom_stamp[0] += 1
        diagnostic.record_odom(odom(time.monotonic(), odom_stamp[0], x=pose_x[0]))

    behavior.on_observe = publish_pose
    robot.on_motion = lambda: pose_x.__setitem__(0, pose_x[0] + .018)
    read = runtime.world_model.get_lidar_obstacles

    def producer(**kwargs):
        # Independent producer callbacks are simulated, never generated by
        # diagnostic retention or used by the actual safety policy.
        clock[0] += 1_000_000
        publish_pose()
        state = read(**kwargs)
        runtime.lidar_worker.diagnostic_sample_callback(state)
        return state

    runtime.world_model.get_lidar_obstacles = producer
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert motions(events) == [("forward", .10, .5)] * 2
    rows = result["progress_diagnostics"]["actions"]
    for index, row in enumerate(rows):
        assert row["pre_odom"]["x"] == pytest.approx(index * .018)
        assert row["post_odom"]["x"] == pytest.approx((index + 1) * .018)
        assert row["odom_delta"]["planar_translation_m"] == pytest.approx(.018)
        assert row["first_post_action_lidar"]["acquisition_sequence"] == result["lidar_wait_history"][index]["snapshot"]["acquisition_sequence"]
        assert row["first_post_action_lidar"]["target_associated"] is False
        assert row["first_post_action_lidar"]["acquisition_sequence"] > row["jit_lidar_evidence"]["acquisition_sequence"]
        assert row["command"]["bridge_acknowledgement"]["automatic_stop"] is True
        assert row["command"]["start_monotonic_seconds"] < row["command"]["completion_monotonic_seconds"]


def test_guarded_alignment_keeps_actual_command_and_acknowledgement():
    from test_guarded_turn_execution import FakeRobot, manager, snapshot, SESSION
    from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT
    robot = FakeRobot()
    behavior = manager(snapshot(), robot)
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("alignment")
    diagnostic.prepare_action("ALIGNING", observation())
    behavior.marvin_command_diagnostic_callback = diagnostic.command_event
    result = behavior.execute_guarded_turn("RIGHT", .25, .02,
        expected_lidar_session=SESSION, now=10., target_directed=True,
        safety_mode=ROTATIONAL_SWEPT_FOOTPRINT, dispatch_guard=lambda: True)
    assert result["ok"] is True
    command = diagnostic.snapshot()["actions"][0]["command"]
    assert command["linear_x"] == 0.
    assert command["angular_z"] == -.25
    assert command["duration"] == .02
    assert command["bridge_acknowledgement"] == result["transport_result"]
    assert command["completion_monotonic_seconds"] >= command["start_monotonic_seconds"]


@pytest.mark.parametrize("stop_phase", ["camera", "gemini"])
def test_stop_during_perception_preserves_source_metadata_and_preempts(tmp_path, monkeypatch, stop_phase):
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch,
        [(0, .8), (0, .75), (0, .75), (0, .5)])
    behavior.marvin_command_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("command_event", *args)
    behavior.marvin_perception_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("perception_event", *args)
    if stop_phase == "camera":
        callback = behavior.semantic_vision.fetch_frame

        def stopped_fetch():
            frame = callback()
            episode = behavior._marvin_v2_tracker_episode
            if episode and episode.get("post_action_pending"):
                runtime.submit_intent({"intent": "STOP"})
            return frame

        behavior.semantic_vision.fetch_frame = stopped_fetch
    else:
        callback = behavior.semantic_vision.select_marvin_candidate

        def stopped_gemini(frame, candidates):
            result = callback(frame, candidates)
            if len(motions(events)) == 1:
                runtime.submit_intent({"intent": "STOP"})
            return result

        behavior.semantic_vision.select_marvin_candidate = stopped_gemini
    result = run(runtime)
    assert result["behavior"] == "STOP" and result["state"] == "STOPPED"
    report = runtime.get_status()["marvin_progress_diagnostics"]
    assert report["terminal"]["state"] == "STOPPED"
    assert len(motions(events)) == 1
    if stop_phase == "gemini":
        assert report["actions"][0]["semantic_reacquisitions"][0]["source_frame_stamp_ns"] == behavior.identity_frames[-1]


def test_diagnostic_callback_failure_does_not_change_lidar_publication(tmp_path):
    from lidar_perception import LidarPerceptionWorker
    from test_lidar_perception import payload
    from world_model import WorldModel
    world = WorldModel(str(tmp_path / "world.json"))
    worker = LidarPerceptionWorker(world, fetch=payload, monotonic=lambda: 10.)

    def broken(sample):
        raise OSError("diagnostic storage unavailable")

    worker.diagnostic_sample_callback = broken
    sample = worker.run_once()
    current = world.get_lidar_obstacles(expected_session=worker.session, now=10.)
    assert sample["available"] is True and current["available"] is True
    assert current["acquisition_sequence"] == 1


def test_finished_mission_retains_late_post_stop_sample_without_control_wait(monkeypatch):
    now = [1.5]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    diagnostic.record_odom(odom(.99, 10))
    prepare(diagnostic)
    diagnostic.terminal("STOPPED", "preempted")
    assert diagnostic.snapshot()["actions"][0]["post_odom"] is None
    now[0] = 1.52
    diagnostic.record_odom(odom(1.52, 11, x=.018))
    diagnostic.record_lidar(scan(11, 1.52))
    report = diagnostic.snapshot()
    assert report["terminal"]["state"] == "STOPPED"
    assert report["actions"][0]["post_odom"]["stamp_ns"] == 11
    assert report["actions"][0]["first_post_action_lidar"]["acquisition_sequence"] == 11


def test_camera_cache_poll_metadata_is_not_reacquisition(tmp_path, monkeypatch):
    from test_find_marvin_camera_recovery import camera_schedule
    from test_find_marvin_closed_loop import delayed_runtime
    bundle = delayed_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    runtime, behavior, _, events, _ = bundle
    behavior.marvin_command_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("command_event", *args)
    behavior.marvin_perception_diagnostic_callback = lambda *args: runtime._retain_marvin_diagnostic("perception_event", *args)
    camera_schedule(bundle, ["action", "action", "fresh"])
    result = run(runtime)
    row = result["progress_diagnostics"]["actions"][0]
    assert result["state"] == "ARRIVED" and result["reacquisition_attempts"] == 0
    assert row["reacquisition_used"] is False and row["semantic_reacquisitions"] == []
    assert sum(f["camera_returned_cached_frame"] for f in row["post_action_frames"]) == 2
    assert all(f["image_width"] == 640 and f["image_height"] == 480 for f in row["post_action_frames"])
    assert row["first_new_post_action_camera"]["source_frame_stamp_ns"] > row["authorizing_camera"]["source_frame_stamp_ns"]
    assert row["first_new_post_action_camera"]["tracker_quality"] >= .8
    assert len(motions(events)) == 1


def test_post_odom_is_preserved_when_pre_sample_is_missing():
    diagnostic = MarvinProgressDiagnostics()
    diagnostic.begin("one")
    prepare(diagnostic)
    diagnostic.record_odom(odom(1.51, 11, x=.018))
    row = diagnostic.snapshot()["actions"][0]
    assert row["pre_odom"] is None and row["post_odom"]["stamp_ns_decimal"] == "11"
    assert "odom_delta" not in row


def test_reacquired_alignment_can_retain_diagnostic_target_association_without_control_changes(tmp_path, monkeypatch):
    from test_find_marvin_closed_loop import CALIBRATION, Worker
    runtime, behavior, _, _, _ = make_runtime(tmp_path, monkeypatch, [])
    diagnostic = runtime.marvin_progress_diagnostics
    diagnostic.begin("one", expected_session=Worker.session, camera_model=CALIBRATION)
    before = observation()
    before["arrival"]["producer_session"] = Worker.session
    diagnostic.prepare_action("ADVANCING", before)
    diagnostic.action_result({"motion_executed": True, "approach_result": {"target_standoff": before["arrival"]}}, time.monotonic())
    behavior.sequence = 10
    sample = behavior.lidar()
    diagnostic.record_lidar(sample)
    aligned_later = observation(101)
    aligned_later["identity_confirmed"] = True
    aligned_later["received_monotonic_seconds"] = time.monotonic()
    aligned_later["opencv_tracker"]["matched"] = True
    aligned_later["arrival"] = {"ok": False, "reason": "marvin_not_centered"}
    diagnostic.observe(aligned_later, reacquisition=True)
    row = diagnostic.snapshot()["actions"][0]
    assert row["next_target_association"]["diagnostic_only"] is True
    assert row["next_target_association"]["acquisition_sequence"] == 11
    assert aligned_later["arrival"] == {"ok": False, "reason": "marvin_not_centered"}


def test_failed_bridge_dispatch_retains_command_and_confirmed_terminal_stop(tmp_path, monkeypatch):
    runtime, _, robot, _, _ = make_runtime(tmp_path, monkeypatch, [(0, .8)])

    def failed(**kwargs):
        raise OSError("offline transport error")

    robot.move_forward = failed
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    row = result["progress_diagnostics"]["actions"][0]
    assert row["command"]["linear_x"] == .10 and row["command"]["duration"] == .5
    assert row["stop_time_source"] == "confirmed_terminal_stop"
    assert row["stopped_monotonic_seconds"] is not None
    assert result["progress_diagnostics"]["terminal"]["last_camera"]["bbox"] is not None
