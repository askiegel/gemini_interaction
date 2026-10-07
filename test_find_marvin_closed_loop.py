"""Offline full-mission V2 tests: no HTTP, ROS, services, or robot access."""

from datetime import datetime, timezone
from types import SimpleNamespace
import time
import math

import pytest

from behavior_manager import BehaviorManager
from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT
from mission_manager import MissionManager
from local_motion_safety_envelope import build_local_motion_lidar_geometry
from runtime import CognitiveRuntime
from world_model import WorldModel


CALIBRATION = {
    "fx_pixels": 597.6149561338204, "cx_pixels": 321.6103934690693,
    "image_width": 640, "x_m": 0.081299, "y_m": 0.0,
    "yaw_degrees": 0.0, "range_uncertainty_m": 0.10,
}


class Robot:
    def __init__(self, events, clock):
        self.events, self.clock = events, clock
        self.on_motion = None
        self.ready = True

    def status(self):
        return {"ok": True, "status": "READY", "ros_ready": self.ready,
                "motion": {"linear_x": 0.0, "angular_z": 0.0, "streaming": False}}

    def stop(self):
        self.events.append("stop")
        return {"ok": True}

    def move_forward(self, *, speed, seconds):
        self.events.append(("forward", speed, seconds))
        self.clock[0] += int(seconds * 1_000_000_000)
        if self.on_motion:
            self.on_motion()
        return {"ok": True, "action": "motion", "mode": "bounded",
                "linear_x": speed, "angular_z": 0.0, "duration": seconds,
                "automatic_stop": True, "returned_immediately": False}


class Worker:
    session = "offline-v2-session"
    running = True
    last_error = None
    sequence = 0

    def __init__(self, *_args, **_kwargs):
        pass


class Perception(BehaviorManager):
    """Controlled camera/semantic/tracker and guarded-turn transport only.

    Runtime admission, standoff projection, guarded-forward safety, mission
    management, stamp consumption and STOP handling are production code.
    Existing guarded-turn tests independently exercise its monitor/transport.
    """
    def __init__(self, robot, world, specs, clock, events):
        super().__init__(robot_client=robot, world_model=world)
        self.specs = iter(specs)
        self.clock, self.events = clock, events
        self.stamps = []
        self.sequence = 0
        self.distance = 0.8
        self.unsafe_turn = False
        self.unsafe_forward = False
        self.invalid_lidar = False
        self.freeze_lidar = False
        self.stale_camera = False
        self.repeat_camera = False
        self.on_observe = None
        self.after_guard_check = None
        self.identity_sources = []
        self.source_offset_ns = 0  # Camera and Tony2 need not share a wall clock.

    def lidar(self, **_kwargs):
        if not self.freeze_lidar:
            self.sequence += 1
        x = self.distance + CALIBRATION["range_uncertainty_m"]
        # A bounded target surface spans the image box at this calibrated depth.
        half_width = (x - CALIBRATION["x_m"]) * 58 / CALIBRATION["fx_pixels"]
        points = [{"x_m": x, "y_m": y * half_width / 6} for y in range(-6, 7)]
        if self.unsafe_forward:
            points.append({"x_m": 0.35, "y_m": 0.20})
        geometry = build_local_motion_lidar_geometry({
            "frame_id": "lidar_link", "angle_min": -math.pi,
            "angle_increment": math.tau / 80, "range_min": .02,
            "range_max": 8.0, "ranges": [2.0] * 80})
        for point in points:
            point.update(distance_m=math.hypot(point["x_m"], point["y_m"]),
                         robot_bearing_deg=math.degrees(math.atan2(point["y_m"], point["x_m"])))
        geometry["points"].extend(points)
        geometry["valid"] = not self.invalid_lidar
        return {"available": True, "valid": not self.invalid_lidar,
                "reason": "fresh" if not self.invalid_lidar else "stale",
                "effective_age_seconds": 0.0, "producer_session": Worker.session,
                "received_monotonic_seconds": time.monotonic(), "age_at_receipt_seconds": 0.0,
                "acquisition_sequence": self.sequence,
                "local_motion_geometry": geometry}

    def observe_find_marvin_v2(self):
        self.clock[0] += 10_000_000
        stamp = self.clock[0] + self.source_offset_ns
        receipt = time.monotonic() - (1.000000001 if self.stale_camera else 0)
        if self.repeat_camera and self.stamps:
            stamp = self.stamps[-1]
        self.stamps.append(stamp)
        spec = next(self.specs)
        self.events.append("observe")
        now = datetime.now(timezone.utc).isoformat()
        if self.on_observe:
            self.on_observe()
        if spec in {"absent", "lost", "invalid"}:
            reason = ("Marvin was not found in the current camera frame." if spec == "absent"
                      else "post_action_tracker_continuity_lost" if spec == "lost"
                      else "marvin_tracker_geometry_invalid")
            self._clear_marvin_v2_tracker_episode()
            return {"preview_result": {"ok": False, "target_found": False,
                "source_frame_stamp_ns": stamp, "received_monotonic_seconds": receipt,
                "reason": reason}}
        # Receipt belongs to this frame, independent of the remote stamp.
        error, self.distance = spec
        box = {"x1": 260.0 + error, "y1": 100.0,
               "x2": 380.0 + error, "y2": 300.0}
        episode = self._marvin_v2_tracker_episode
        pending = episode is not None and episode.get("post_action_pending") is True
        source = "marvin_locked_tracker_continuity" if pending else "gemini_marvin_candidate_selection"
        self.identity_sources.append(source)
        opencv = {"active": True, "matched": True, "quality": 0.95, "threshold": 0.8,
                  "bbox": box, "image_width": 640, "image_height": 480,
                  "center_x": 320.0 + error, "horizontal_error": error,
                  "source_frame_stamp_ns": stamp, "received_monotonic_seconds": receipt}
        preview = {"ok": True, "preview": True, "authoritative": False,
                   "target": "marvin", "source": "marvin_local_tracker",
                   "target_found": True, "identity_confirmed": True,
                   "motion_authorized_marvin_candidate": True,
                   "identity_source": source,
                   "identity_source_frame_stamp_ns": (episode["identity_source_frame_stamp_ns"]
                                                       if pending else stamp - 1),
                   "source_frame_stamp_ns": stamp, "source_timestamp": now,
                   "received_monotonic_seconds": receipt,
                   "bbox": box, "image_width": 640, "image_height": 480,
                   "opencv_tracker": opencv}
        if pending:
            preview.update(post_action_tracker_continuity=True,
                           post_action_source_frame_stamp_ns=episode["post_action_source_frame_stamp_ns"],
                           marvin_tracking_episode={"state": "POST_ACTION_TRACKED",
                               "post_action_source_frame_stamp_ns": episode["post_action_source_frame_stamp_ns"]})
        self._marvin_v2_tracker_episode = {
            "identity_source": "gemini_marvin_candidate_selection",
            "identity_source_frame_stamp_ns": preview["identity_source_frame_stamp_ns"],
            "last_tracker_source_frame_stamp_ns": stamp, "tracker_bbox": box,
            "marvin_tracker": episode["marvin_tracker"] if episode else object(),
        }
        return {"preview_result": preview, "target_lock_result": {},
                "target_lock_snapshot": {}, "selected_identity_id": None}

    def _execute_target_directed_turn(self, direction, speed, duration, *,
                                      expected_lidar_session, safety_mode, dispatch_guard):
        assert safety_mode == ROTATIONAL_SWEPT_FOOTPRINT
        assert expected_lidar_session == Worker.session
        if self.after_guard_check:
            self.after_guard_check()
        if self.unsafe_turn or not dispatch_guard():
            return {"ok": False, "permitted": False, "confirmed_forwarded": False}
        lidar = self.world_model.get_lidar_obstacles(expected_session=expected_lidar_session)
        self.events.append(("turn", direction, speed, duration))
        self.clock[0] += int(duration * 1_000_000_000)
        if self.robot.on_motion:
            self.robot.on_motion()
        return {"ok": True, "permitted": True, "confirmed_forwarded": True,
                "action_lidar_evidence": {"producer_session": lidar["producer_session"],
                                          "acquisition_sequence": lidar["acquisition_sequence"]}}


def make_runtime(tmp_path, monkeypatch, specs):
    events, clock = [], [time.time_ns()]
    monkeypatch.setattr("runtime.time.time_ns", lambda: clock[0])
    origin = clock[0]
    monkeypatch.setattr("runtime.time.monotonic", lambda: 100.0 + (clock[0] - origin) / 1e9)
    monkeypatch.setattr("runtime.time.sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + max(1, round(seconds * 1_000_000_000))))
    robot = Robot(events, clock)
    world = WorldModel(str(tmp_path / "world.json"))
    behavior = Perception(robot, world, specs, clock, events)
    world.get_lidar_obstacles = behavior.lidar
    runtime = CognitiveRuntime(
        provider=object(), mission_manager=MissionManager(), world_model=world,
        vision_adapter=object(), robot_client=robot, behavior_manager=behavior,
        lidar_worker_factory=Worker, marvin_camera_model=CALIBRATION,
    )
    runtime.running = True
    return runtime, behavior, robot, events, clock


def run(runtime):
    runtime.submit_intent({"intent": "FIND_OBJECT", "target": "Marvin", "speech": "Finding Marvin."})
    return runtime.run_once()


def motions(events):
    return [e for e in events if isinstance(e, tuple)]


@pytest.mark.parametrize("specs,states", [
    ([(0, .60), (0, .59), (0, .5)], ["ADVANCING", "ADVANCING"]),
    ([(120, .60), (0, .60), (0, .5)], ["ALIGNING", "ADVANCING"]),
    ([(-120, .60), (0, .5)], ["ALIGNING"]),
    (["absent", "absent", (120, .60), (0, .60), (0, .5)],
     ["SEARCHING", "SEARCHING", "ALIGNING", "ADVANCING"]),
])
def test_one_mission_runs_full_observe_guarded_action_stop_loop(tmp_path, monkeypatch, specs, states):
    runtime, behavior, robot, events, clock = make_runtime(tmp_path, monkeypatch, specs)
    result = run(runtime)
    assert result["arrived_at_marvin"] is True and result["state"] == "ARRIVED"
    assert [r["state"] for r in result["history"]] == states
    assert result["actions_executed"] == len(states)
    assert runtime.mission_manager.get_active_mission() is None
    assert runtime.get_status()["runtime_state"] == "IDLE"
    assert runtime.get_status()["tracking"]["active"] is False
    assert len(set(behavior.stamps)) == len(states) + 1
    for i, event in enumerate(events):
        if isinstance(event, tuple):
            assert events[i-1] == "observe"
            assert events[i+1] == "stop"
    assert len(runtime._marvin_alignment_consumed_source_frame_stamps) == len(states)
    assert robot.status()["motion"] == {"linear_x": 0.0, "angular_z": 0.0, "streaming": False}
    if len(states) > 1 and states[0] != "SEARCHING":
        assert behavior.identity_sources[1] == "marvin_locked_tracker_continuity"


def test_search_is_exactly_one_bounded_full_sweep_with_observation_after_every_turn(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, ["absent"] * 27)
    result = run(runtime)
    assert result["state"] == "SEARCH_EXHAUSTED"
    assert result["search_turns"] == 26
    assert len(behavior.stamps) == 27
    assert motions(events) == [("turn", "LEFT", .25, 1.0)] * 26
    assert result["arrived_at_marvin"] is False


@pytest.mark.parametrize("distance", [.5, .49, .45, .4])
def test_standoff_or_hard_envelope_never_advances(tmp_path, monkeypatch, distance):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, distance)])
    result = run(runtime)
    assert result["state"] == ("BLOCKED" if distance == .4 else "ARRIVED")
    assert motions(events) == []
    if distance == .4:
        assert events[-1] == "stop"
        assert not result["arrived_at_marvin"]
        assert result["history"][0]["observation"]["arrival"]["target_range_association_trusted"] is False


@pytest.mark.parametrize("specs,flag", [
    (["absent"], "unsafe_turn"), ([(120, .8)], "unsafe_turn"),
    ([(0, .8)], "unsafe_forward"), ([(0, .8)], "invalid_lidar"),
    ([(0, .8)], "stale_camera"),
])
def test_whole_mission_safety_vetoes_stop_without_avoidance(tmp_path, monkeypatch, specs, flag):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, specs)
    setattr(behavior, flag, True)
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert motions(events) == []
    assert events[-1] == "stop"


def test_identity_loss_after_turn_with_unavailable_semantics_exhausts_recovery(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(120, .8), "lost"])
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert len(motions(events)) == 1
    assert runtime._marvin_alignment_observation is None


@pytest.mark.parametrize("during", ["observe", "turn", "forward"])
def test_stop_preempts_whole_mission_without_any_followup_action(tmp_path, monkeypatch, during):
    runtime, behavior, robot, events, _ = make_runtime(
        tmp_path, monkeypatch, [(120 if during == "turn" else 0, .8)])
    stop = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    if during == "observe":
        behavior.on_observe = stop
    else:
        robot.on_motion = stop
    result = run(runtime)
    assert result["behavior"] == "STOP"
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert len(motions(events)) == (0 if during == "observe" else 1)
    assert events.count("observe") == 1
    assert runtime._marvin_alignment_observation is None


@pytest.mark.parametrize("freeze", ["repeat_camera", "freeze_lidar"])
def test_new_camera_and_lidar_are_required_after_each_action(tmp_path, monkeypatch, freeze):
    runtime, behavior, robot, events, _ = make_runtime(tmp_path, monkeypatch, [(120, .8), (120, .8)])
    robot.on_motion = lambda: setattr(behavior, freeze, True)
    result = run(runtime)
    assert result["state"] in {"BLOCKED", "REVERIFY_REQUIRED"}
    assert len(motions(events)) == 1


def test_expiration_during_jit_checks_prevents_transport_and_consumes_once(tmp_path, monkeypatch):
    runtime, behavior, _, events, clock = make_runtime(tmp_path, monkeypatch, [(120, .8)])
    behavior.after_guard_check = lambda: clock.__setitem__(0, clock[0] + 1_000_000_001)
    result = run(runtime)
    assert result["state"] == "BLOCKED" and motions(events) == []
    assert len(runtime._marvin_alignment_consumed_source_frame_stamps) == 1


def test_unexpected_bridge_state_blocks_before_perception(tmp_path, monkeypatch):
    runtime, _, robot, events, _ = make_runtime(tmp_path, monkeypatch, [])
    robot.ready = False
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert events.count("observe") == 0 and motions(events) == []


def test_invalid_geometry_is_not_misclassified_as_initial_search(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, ["invalid"])
    assert run(runtime)["state"] == "REVERIFY_REQUIRED"
    assert motions(events) == []


def test_other_thread_cannot_supersede_or_execute_mission_observation(tmp_path, monkeypatch):
    import threading

    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(120, .8), (0, .5)])
    replies = []

    def competing_request():
        replies.append(runtime.observe_find_marvin_v2())
        replies.append(runtime.execute_single_marvin_alignment(
            direction="RIGHT", angular_speed=.25, duration=.5,
            source_frame_stamp_ns=behavior.stamps[-1]))

    def during_observe():
        worker = threading.Thread(target=competing_request)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()

    behavior.on_observe = during_observe
    assert run(runtime)["state"] == "ARRIVED"
    assert len(motions(events)) == 1
    assert all(reply["ok"] is False for reply in replies)
    assert all(reply.get("motion_executed", False) is False for reply in replies)


def test_missing_calibration_blocks_approach_without_any_forward(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime.marvin_camera_model = None
    assert run(runtime)["state"] == "BLOCKED"
    assert motions(events) == []


@pytest.mark.parametrize("distance,max_duration", [(.53, .30), (.51, .10)])
def test_near_standoff_shortens_single_forward_and_reobserves(
        tmp_path, monkeypatch, distance, max_duration):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, distance), (0, .5)])
    assert run(runtime)["state"] == "ARRIVED"
    assert motions(events) == [("forward", .10, pytest.approx(max_duration))]
    duration = motions(events)[0][2]
    assert duration <= max_duration + 1e-12
    assert distance - .10 * duration >= .50


def test_another_behavior_owning_motion_blocks_find_marvin(tmp_path, monkeypatch):
    runtime, _, _, events, _ = make_runtime(tmp_path, monkeypatch, [(120, .8)])
    runtime._physical_action_lock.acquire()
    try:
        assert run(runtime)["state"] == "BLOCKED"
    finally:
        runtime._physical_action_lock.release()
    assert motions(events) == []


def test_real_guarded_turn_checks_dispatch_callback_before_robot_transport():
    from test_guarded_turn_execution import FakeRobot, manager, snapshot, SESSION

    robot = FakeRobot()
    behavior = manager(snapshot(), robot)
    result = behavior.execute_guarded_turn(
        "RIGHT", .25, .5, expected_lidar_session=SESSION, now=10.0,
        target_directed=True, safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
        dispatch_guard=lambda: False)
    assert result["ok"] is False
    assert result["confirmed_forwarded"] is False
    assert robot.motion_calls == []
    assert robot.stop_calls >= 1


def test_real_guarded_forward_checks_dispatch_callback_before_robot_transport(tmp_path, monkeypatch):
    _, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [])
    result = behavior.execute_single_marvin_approach_step(
        expected_lidar_session=Worker.session, linear_speed=.10, duration=.5,
        dispatch_guard=lambda: False)
    assert result["reason"] == "marvin_motion_observation_stale_or_preempted"
    assert result["motion_executed"] is False
    assert motions(events) == []


def test_search_transport_exception_stops_and_terminates(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, ["absent"])

    def failed_transport(*_args, **_kwargs):
        raise OSError("bridge unavailable")

    behavior._execute_target_directed_turn = failed_transport
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert motions(events) == [] and events[-1] == "stop"
    assert len(behavior.stamps) == 1
    assert runtime._marvin_alignment_observation is None


def test_bridge_not_ready_cannot_admit_motion_even_with_zero_velocity(tmp_path, monkeypatch):
    runtime, _, robot, events, _ = make_runtime(tmp_path, monkeypatch, [])
    original = robot.status
    robot.status = lambda: dict(original(), status="ERROR")
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert events.count("observe") == 0 and motions(events) == []


class DelayedSemanticPerception(Perception):
    """Real V2 acquisition/confirmation/preview, with local camera spies.

    Only proposal transport, Gemini, frame transport, and template matching
    are fake. The mission, persistent episode, source-stamp selection,
    admission, consumption, and post-action continuation are production code.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.semantic_calls = 0
        self.identity_frames = []
        self.identity_receipts = []
        self.updated_frames = []
        self.updated_receipts = []
        self.created_trackers = []
        self.observations = []
        self.confirm_identity = True
        self.refresh_mode = "fresh"
        self.cached_frame_returned = False
        self.on_semantic = None
        self.on_refresh = None
        self.semantic_done_ns = None
        # Exercise the real bounded polling loop without seconds of offline
        # delay when deliberately simulating a permanently frozen camera.
        self.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS = .10
        self.MARVIN_POST_TURN_FRAME_POLL_SECONDS = .001
        self.semantic_vision = SimpleNamespace(
            fetch_frame=self.frame, select_marvin_candidate=self.select)
        self.marvin_local_tracker_factory = self.tracker

    def current_box(self):
        error = self.current_spec[0] if self.current_spec != "absent" else 0
        return {"x1": 260 + error, "y1": 100,
                "x2": 380 + error, "y2": 300}

    def _confirm_marvin_proposal_candidates_with_status(self, **kwargs):
        self.clock[0] += 10_000_000
        return ([{"bbox": self.current_box(), "image_width": 640,
                  "image_height": 480, "label": "toy", "confidence": .9,
                  "source_frame_stamp_ns": self.clock[0] + self.source_offset_ns}], "target_confirmed",
                {"latest_source_frame_stamp_ns": self.clock[0] + self.source_offset_ns})

    def select(self, frame, candidates):
        self.semantic_calls += 1
        self.identity_frames.append(frame.source_frame_stamp_ns)
        self.identity_receipts.append(frame.received_monotonic_seconds)
        # Deliberately longer than the unchanged physical-action age limit.
        self.clock[0] += 2_500_000_000
        self.semantic_done_ns = self.clock[0]
        if self.on_semantic:
            self.on_semantic()
        return {"confirmed": self.confirm_identity and self.current_spec != "absent",
                "candidate_index": 0, "source": "gemini_marvin_candidate_selection"}

    def frame(self):
        if self.semantic_done_ns is not None and self.on_refresh:
            self.on_refresh()
        if self.semantic_done_ns is not None and self.refresh_mode == "unavailable":
            raise OSError("offline camera unavailable")
        self.clock[0] += 10_000_000
        stamp = self.clock[0] + self.source_offset_ns
        if self.semantic_done_ns is not None:
            if self.refresh_mode == "cached_semantic":
                stamp = self.identity_frames[-1]
            elif self.refresh_mode == "cached_during_gemini":
                stamp = self.semantic_done_ns + self.source_offset_ns - 10_000_000
            elif self.refresh_mode == "cached_once" and not self.cached_frame_returned:
                stamp = self.semantic_done_ns + self.source_offset_ns - 10_000_000
                self.cached_frame_returned = True
            elif self.refresh_mode == "repeat_action":
                stamp = self.updated_frames[-1]
        return SimpleNamespace(width=640, height=480, source_frame_stamp_ns=stamp,
            received_monotonic_seconds=time.monotonic(),
            received_at=datetime.fromtimestamp(self.clock[0] / 1e9, timezone.utc).isoformat())

    def tracker(self, frame, seed):
        owner = self

        class Tracker:
            MIN_MATCH_QUALITY = .8
            last_quality = .95

            def update(self, fresh_frame):
                self.last_quality = .95
                owner.updated_frames.append(fresh_frame.source_frame_stamp_ns)
                owner.updated_receipts.append(fresh_frame.received_monotonic_seconds)
                return owner._expand_marvin_tracker_seed_bbox(owner.current_box(), 640, 480)

        tracker = Tracker()
        self.created_trackers.append(tracker)
        return tracker

    def observe_find_marvin_v2(self):
        self.current_spec = next(self.specs)
        if self.current_spec != "absent":
            self.distance = self.current_spec[1]
        self.events.append("observe")
        observation = BehaviorManager.observe_find_marvin_v2(self)
        self.observations.append(observation["preview_result"])
        return observation


def delayed_runtime(tmp_path, monkeypatch, specs):
    runtime, _, robot, events, clock = make_runtime(tmp_path, monkeypatch, [])
    behavior = DelayedSemanticPerception(robot, runtime.world_model, specs, clock, events)
    runtime.behavior_manager = behavior
    behavior.execution_authorization_provider = runtime._behavior_execution_is_current
    runtime.world_model.get_lidar_obstacles = behavior.lidar
    # Legacy visual policy timestamps are local wall receipt times. Camera
    # source stamps deliberately have an independent offset in skew tests.
    import runtime as runtime_module
    pursuit = runtime_module.evaluate_marvin_pursuit_state
    visual_arrival = runtime_module.evaluate_marvin_visual_arrival
    monkeypatch.setattr(runtime_module, "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: pursuit(*args, **kwargs,
            now=datetime.fromtimestamp(clock[0] / 1e9, timezone.utc)))
    monkeypatch.setattr(runtime_module, "evaluate_marvin_visual_arrival",
        lambda *args, **kwargs: visual_arrival(*args, **kwargs,
            now=datetime.fromtimestamp(clock[0] / 1e9, timezone.utc)))
    return runtime, behavior, robot, events, clock


@pytest.mark.parametrize("error", [120, 0])
def test_slow_gemini_identity_refreshes_action_frame_then_continues_without_gemini(
    tmp_path, monkeypatch, error,
):
    runtime, behavior, _, events, clock = delayed_runtime(
        tmp_path, monkeypatch, [(error, .60), (error, .58), (0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert len(motions(events)) == 2
    assert behavior.semantic_calls == 1
    assert len(behavior.created_trackers) == 1
    first, second = result["history"]
    identity_stamp = behavior.identity_frames[0]
    first_preview = behavior.observations[0]
    action_stamp = first["source_frame_stamp_ns"]
    assert behavior.semantic_done_ns - identity_stamp > 1_000_000_000
    assert first_preview["identity_confirmed"] is True
    assert first_preview["identity_source_frame_stamp_ns"] == identity_stamp
    assert first_preview["source_frame_stamp_ns"] == action_stamp
    assert first["observation"]["identity_source_frame_stamp_ns"] == identity_stamp
    assert action_stamp == first_preview["opencv_tracker"]["source_frame_stamp_ns"]
    assert action_stamp > behavior.semantic_done_ns
    assert second["source_frame_stamp_ns"] > action_stamp
    assert second["observation"]["identity_source"] == "marvin_locked_tracker_continuity"
    assert second["observation"]["post_action_tracker_continuity"] is True
    assert first["observation"]["received_monotonic_seconds"] == behavior.updated_receipts[1]
    assert second["observation"]["received_monotonic_seconds"] == behavior.updated_receipts[2]
    assert behavior.updated_receipts[2] > behavior.updated_receipts[1]
    assert runtime.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.0
    assert not runtime._marvin_motion_stamp_is_fresh(identity_stamp, behavior.identity_receipts[0])
    assert identity_stamp not in runtime._marvin_alignment_consumed_source_frame_stamps
    assert runtime._marvin_alignment_consumed_source_frame_stamps == {
        first["source_frame_stamp_ns"], second["source_frame_stamp_ns"]}
    # Shared stamp protection wins even after the mission's authorization is
    # cleared, and it cannot dispatch either kind of motion a second time.
    for row in (first, second):
        reply = runtime.execute_single_marvin_alignment(
            direction="RIGHT", angular_speed=.25, duration=.5,
            source_frame_stamp_ns=row["source_frame_stamp_ns"])
        assert reply["motion_executed"] is False
        assert reply["reason"] == "marvin_alignment_observation_already_consumed"
    assert len(motions(events)) == 2


@pytest.mark.parametrize("mode", ["cached_semantic", "cached_during_gemini", "unavailable"])
def test_initial_identity_cannot_move_without_post_semantic_tracker_frame(
    tmp_path, monkeypatch, mode,
):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [(120, .8)])
    behavior.refresh_mode = mode
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_post_semantic_tracker_refresh_failed"
    assert motions(events) == []
    assert behavior.updated_frames == []
    assert runtime._marvin_alignment_observation is None
    assert behavior._marvin_v2_tracker_episode is None


def test_cached_camera_frame_is_not_restamped_and_new_source_is_used(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = delayed_runtime(
        tmp_path, monkeypatch, [(120, .8), (0, .5)])
    behavior.refresh_mode = "cached_once"
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert behavior.cached_frame_returned is True
    assert behavior.updated_frames and min(behavior.updated_frames) > behavior.semantic_done_ns
    assert result["history"][0]["source_frame_stamp_ns"] == behavior.updated_frames[1]
    assert len(motions(events)) == 1


@pytest.mark.parametrize("error", [120, 0])
def test_old_semantic_stamp_cannot_dispatch_and_exact_refreshed_stamp_is_one_shot(
    tmp_path, monkeypatch, error,
):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [(error, .8)])
    observation = runtime.observe_find_marvin_v2()
    assert observation["ok"] is True
    old = observation["identity_source_frame_stamp_ns"]
    fresh = observation["source_frame_stamp_ns"]
    assert fresh > old + 1_000_000_000

    def dispatch(stamp):
        if error:
            return runtime.execute_single_marvin_alignment(
                direction="RIGHT", angular_speed=.25, duration=.5,
                source_frame_stamp_ns=stamp)
        return runtime.execute_single_marvin_approach(
            linear_speed=.10, duration=.5, source_frame_stamp_ns=stamp)

    rejected = dispatch(old)
    assert rejected["motion_executed"] is False
    assert motions(events) == []
    assert old not in runtime._marvin_alignment_consumed_source_frame_stamps
    accepted = dispatch(fresh)
    assert accepted["motion_executed"] is True
    assert len(motions(events)) == 1
    duplicate = dispatch(fresh)
    assert duplicate["motion_executed"] is False
    assert duplicate["actions_executed"] == 0
    assert duplicate["reason"].endswith("observation_already_consumed")
    assert len(motions(events)) == 1


def test_slow_semantic_absence_gets_current_camera_before_bounded_search(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = delayed_runtime(
        tmp_path, monkeypatch, ["absent", (0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED" and result["search_turns"] == 1
    row = result["history"][0]
    preview = behavior.observations[0]
    assert row["state"] == "SEARCHING"
    assert preview["identity_source_frame_stamp_ns"] == behavior.identity_frames[0]
    assert preview["identity_source_frame_stamp_ns"] < preview["source_frame_stamp_ns"]
    assert row["source_frame_stamp_ns"] == preview["source_frame_stamp_ns"]
    assert row["source_frame_stamp_ns"] > behavior.identity_frames[0] + 2_500_000_000
    assert len(motions(events)) == 1


@pytest.mark.parametrize("mode", ["cached_semantic", "cached_during_gemini", "unavailable"])
def test_search_does_not_turn_from_old_absence_frame(tmp_path, monkeypatch, mode):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, ["absent"])
    behavior.refresh_mode = mode
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_search_fresh_frame_unavailable"
    assert motions(events) == []


@pytest.mark.parametrize("spec", [(120, .8), "absent"])
@pytest.mark.parametrize("during", ["semantic", "refresh"])
def test_stop_during_slow_semantics_or_action_frame_refresh_prevents_motion(
    tmp_path, monkeypatch, spec, during,
):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [spec])
    stop = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    if during == "semantic":
        behavior.on_semantic = stop
    else:
        behavior.on_refresh = stop
    assert run(runtime)["behavior"] == "STOP"
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert motions(events) == []
    assert behavior._marvin_v2_tracker_episode is None
    assert runtime._marvin_alignment_observation is None


@pytest.mark.parametrize("spec", [(120, .8), (0, .8), "absent"])
@pytest.mark.parametrize("offset_ns", [0, 5_000_000_000])
def test_refreshed_frame_still_expires_at_final_physical_dispatch(tmp_path, monkeypatch, spec, offset_ns):
    runtime, behavior, _, events, clock = delayed_runtime(tmp_path, monkeypatch, [spec])
    behavior.source_offset_ns = offset_ns
    if spec == (0, .8):
        original = behavior.execute_single_marvin_approach_step

        def delayed_guarded_forward(**kwargs):
            clock[0] += 1_000_000_001
            return original(**kwargs)

        behavior.execute_single_marvin_approach_step = delayed_guarded_forward
    else:
        behavior.after_guard_check = lambda: clock.__setitem__(0, clock[0] + 1_000_000_001)
    assert run(runtime)["state"] == "BLOCKED"
    assert motions(events) == []


@pytest.mark.parametrize("spec,flag", [
    ((120, .8), "unsafe_turn"), ((0, .8), "unsafe_forward"),
    ((0, .8), "invalid_lidar"), ("absent", "unsafe_turn"),
])
def test_post_semantic_refresh_does_not_bypass_lidar(tmp_path, monkeypatch, spec, flag):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [spec])
    setattr(behavior, flag, True)
    assert run(runtime)["state"] == "BLOCKED"
    assert motions(events) == []


@pytest.mark.parametrize("offset_ns", [75_000_000, 5_000_000_000, -5_000_000_000])
@pytest.mark.parametrize("spec", [(120, .60), (0, .60), "absent"])
def test_full_mission_uses_local_receipts_with_independent_camera_clock(
    tmp_path, monkeypatch, offset_ns, spec,
):
    # Exercise production semantic refresh, tracker confirmation, search,
    # admission and final dispatch with both signs of cross-host clock skew.
    runtime, behavior, _, events, clock = delayed_runtime(
        tmp_path, monkeypatch, [spec, (0, .5)])
    behavior.source_offset_ns = offset_ns
    result = run(runtime)
    assert result["state"] == "ARRIVED", result
    assert len(motions(events)) == 1
    action = result["history"][0]
    observation = action["observation"]
    stamp = action["source_frame_stamp_ns"]
    preview = behavior.observations[0]
    assert stamp == preview["source_frame_stamp_ns"]
    assert stamp > behavior.identity_frames[0]
    assert runtime._marvin_alignment_consumed_source_frame_stamps == {stamp}
    assert observation["received_monotonic_seconds"] > behavior.identity_receipts[0] + 2.5
    if spec != "absent":
        assert stamp == preview["opencv_tracker"]["source_frame_stamp_ns"]
        assert observation["received_monotonic_seconds"] == behavior.updated_receipts[1]
        assert behavior.semantic_calls == 1
        assert len(behavior.created_trackers) == 1
    # No restamping of the camera token to the Tony2 clock, even when remote
    # time remains seconds ahead of or behind local wall time.
    if offset_ns == 5_000_000_000:
        assert stamp > clock[0]


@pytest.mark.parametrize("spec", [(120, .8), (0, .8), "absent"])
def test_missing_frame_receipt_blocks_all_three_mission_action_types(tmp_path, monkeypatch, spec):
    runtime, behavior, _, events, _ = delayed_runtime(tmp_path, monkeypatch, [spec])
    observe = behavior.observe_find_marvin_v2

    def missing_receipt():
        result = observe()
        preview = result["preview_result"]
        preview.pop("received_monotonic_seconds", None)
        (preview.get("opencv_tracker") or {}).pop("received_monotonic_seconds", None)
        return result

    behavior.observe_find_marvin_v2 = missing_receipt
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "marvin_motion_observation_stale"
    assert motions(events) == []


@pytest.mark.parametrize("spec", [(120, .8), (0, .8), "absent"])
def test_future_remote_stamp_does_not_rescue_expired_local_receipt(tmp_path, monkeypatch, spec):
    runtime, behavior, _, events, clock = delayed_runtime(tmp_path, monkeypatch, [spec])
    behavior.source_offset_ns = 5_000_000_000
    observe = behavior.observe_find_marvin_v2

    def delayed_delivery():
        result = observe()
        clock[0] += 1_000_000_001
        return result

    behavior.observe_find_marvin_v2 = delayed_delivery
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "marvin_motion_observation_stale"
    assert motions(events) == []


def test_post_action_new_receipt_cannot_rejuvenate_same_remote_frame(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = delayed_runtime(
        tmp_path, monkeypatch, [(120, .8), (120, .7)])
    behavior.source_offset_ns = 5_000_000_000
    robot.on_motion = lambda: setattr(behavior, "refresh_mode", "repeat_action")
    result = run(runtime)
    assert result["state"] == "REVERIFY_REQUIRED"
    assert len(motions(events)) == 1
    assert behavior.semantic_calls == 1
    assert len(behavior.created_trackers) == 1
    assert runtime._marvin_alignment_consumed_source_frame_stamps == {behavior.updated_frames[-1]}
    assert runtime._marvin_alignment_observation is None
