"""Offline missions using the production client/interlock; no robot transport."""
import socket
import time

import pytest

from robot_bridge.client import RobotBridgeClient
from robot_bridge.forward_interlock import ForwardMotionInterlock, MAXIMUM_EFFECTIVE_AGE_SECONDS
from test_find_marvin_closed_loop import make_runtime, motions, run


@pytest.fixture(autouse=True)
def forbid_live_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live transport forbidden")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


def interrupted_runtime(tmp_path, monkeypatch, specs, *, interruptions=(True, False), publish=True,
                        factory=make_runtime):
    bundle = factory(tmp_path, monkeypatch, specs)
    runtime, behavior, robot, events, clock = bundle
    behavior.freeze_lidar = True
    behavior.sequence = 42
    original_read = behavior.lidar
    plan = iter(interruptions)
    state = {"outage": False, "sleeps": 0, "on_sleep": None,
             "on_publish": None, "stop_failure": False, "samples": [], "on_read": None,
             "transport_returned": False}

    def telemetry(**kwargs):
        sample = original_read(**kwargs)
        sample["sectors"] = {"front": {"available": True, "state": "CLEAR"}}
        if state["outage"]:
            sample.update(available=False, valid=False, reason="stale", effective_age_seconds=.342)
        state["samples"].append((sample["acquisition_sequence"], sample["reason"], len(motions(events))))
        if state["outage"] and state["transport_returned"] and state["on_read"]:
            state["on_read"]()
        return sample

    runtime.world_model.get_lidar_obstacles = telemetry
    client = RobotBridgeClient(base_url="http://offline.invalid")

    def stop():
        events.append("interlock_stop")
        result = robot.stop()
        return {"ok": False} if state["stop_failure"] else result

    interlock = ForwardMotionInterlock(telemetry, expected_session=runtime.lidar_worker.session,
        stop_callback=stop, monotonic=lambda: time.monotonic())
    client.configure_forward_interlock(interlock)

    def transport(method, path, payload):
        assert method == "POST" and path == "/motion"
        events.append(("forward", payload["linear_x"], payload["duration"]))
        if next(plan, False):
            clock[0] += 120_000_000
            behavior.sequence += 1  # Independent simulated producer acquisition.
            state.update(outage=True, sleeps=0)
            assert interlock.refresh() == (False, "stale")
            assert events[-2:] == ["interlock_stop", "stop"]  # Immediate STOP while pending.
            clock[0] += 10_000_000
        else:
            clock[0] += round(payload["duration"] * 1e9)
            behavior.sequence += 1
        state["transport_returned"] = True
        return {"ok": True, "action": "motion", "mode": "bounded",
                "linear_x": payload["linear_x"], "angular_z": 0., "duration": payload["duration"],
                "automatic_stop": True, "returned_immediately": False}

    client._request = transport

    def forward(*, speed, seconds):
        assert interlock.refresh()[0] is True
        return client.move_forward(speed=speed, seconds=seconds)

    robot.move_forward = forward

    def sleep(seconds):
        assert robot.status()["motion"] == {"linear_x": 0., "angular_z": 0., "streaming": False}
        clock[0] += max(1, round(seconds * 1e9))
        if state["outage"]:
            events.append("stopped_lidar_wait")
            state["sleeps"] += 1
            if state["on_sleep"]:
                state["on_sleep"]()
            if publish and state["sleeps"] == 2:
                behavior.sequence += 1
                state["outage"] = False
                events.append("fresh_lidar_published")
                if state["on_publish"]:
                    state["on_publish"]()

    monkeypatch.setattr("runtime.time.sleep", sleep)
    return bundle, state, interlock


@pytest.mark.parametrize("fresh_distance,replanned_duration", [
    (.60, .50), (.53, .30), (.51, .10),
])
def test_interruption_stops_consumes_stamp_waits_reobserves_replans_and_arrives(
        tmp_path, monkeypatch, fresh_distance, replanned_duration):
    bundle, state, _ = interrupted_runtime(
        tmp_path, monkeypatch, [(0, .60), (0, fresh_distance), (0, .5)])
    runtime, behavior, _, events, _ = bundle
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["completed_forward_actions"] == result["actions_executed"] == 1
    assert result["interrupted_forward_attempts"] == 1
    first, second = result["history"]
    interrupted = first["result"]
    assert interrupted["interrupted"] and interrupted["source_stamp_consumed"]
    assert not interrupted["full_step_completed"] and interrupted["actions_executed"] == 0
    assert interrupted["requested_duration"] == .5
    assert interrupted["actual_confirmed_run_duration_seconds"] is None
    outcome = interrupted["approach_result"]["forward_result"]["interlock_dispatch_outcome"]
    assert outcome["stop_succeeded"] is True
    assert outcome["invalidating_lidar_evidence"]["acquisition_sequence"] == 43
    assert outcome["invalidating_lidar_evidence"]["effective_age_seconds"] == .342
    assert len(outcome["stop_events"]) == 2
    assert all(e["result"]["ok"] for e in outcome["stop_events"])
    assert (outcome["stop_events"][0]["completed_monotonic_seconds"]
            - outcome["dispatch_started_monotonic_seconds"]) == pytest.approx(.12)
    assert first["source_frame_stamp_ns"] < second["source_frame_stamp_ns"]
    assert first["source_frame_stamp_ns"] in runtime._marvin_alignment_consumed_source_frame_stamps
    wait = result["lidar_recovery_history"][0]["wait"]
    assert wait["previous_acquisition_sequence"] == 43 and wait["snapshot"]["acquisition_sequence"] == 44
    assert wait["wait_elapsed_seconds"] == pytest.approx(.10)
    assert state["sleeps"] == 2
    assert events.index("interlock_stop") < events.index("stopped_lidar_wait") < events.index("fresh_lidar_published")
    assert events[events.index("fresh_lidar_published")+1] == "observe"
    assert behavior.identity_sources[1] == "marvin_locked_tracker_continuity"
    assert result["consecutive_lidar_interruptions"] == 0
    diagnostic = result["progress_diagnostics"]["action_summary"][0]
    assert diagnostic["interrupted"] and not diagnostic["full_step_completed"]
    assert diagnostic["nominal_displacement_m"] is None
    assert diagnostic["requested_nominal_displacement_m"] == pytest.approx(.05)
    duplicate = runtime.execute_single_marvin_approach(linear_speed=.10, duration=.5,
        source_frame_stamp_ns=first["source_frame_stamp_ns"])
    assert duplicate["reason"] == "marvin_approach_observation_already_consumed"
    assert motions(events) == [
        ("forward", .10, .50), ("forward", .10, pytest.approx(replanned_duration))]
    # The post-interruption command is recomputed from new range evidence,
    # including its standoff bound, rather than resuming the old 0.50 s step.
    assert motions(events)[1][2] <= replanned_duration + 1e-12
    assert fresh_distance - .10 * motions(events)[1][2] >= .50


def test_fresh_lidar_never_returns_blocks_with_no_additional_motion(tmp_path, monkeypatch):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)], publish=False)
    result = run(bundle[0])
    assert result["state"] == "BLOCKED" and result["reason"] == "find_marvin_new_lidar_evidence_timeout"
    assert len(motions(bundle[3])) == 1
    assert result["lidar_recovery_history"][0]["wait"]["wait_elapsed_seconds"] <= .65
    assert state["sleeps"] <= 13


def test_newer_but_stale_acquisition_keeps_waiting_for_genuinely_fresh_scan(tmp_path, monkeypatch):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .60), (0, .5)])
    def stale_publication():
        bundle[1].sequence += 1
        state["on_read"] = None
    state["on_read"] = stale_publication
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    wait = result["lidar_recovery_history"][0]["wait"]
    assert wait["previous_acquisition_sequence"] == 43
    assert wait["snapshot"]["acquisition_sequence"] == 45
    assert (44, "stale", 1) in state["samples"]
    assert len(motions(bundle[3])) == 1


@pytest.mark.parametrize("fault,reason", [("session", "find_marvin_lidar_producer_session_changed"),
    ("invalid", "find_marvin_lidar_not_current"),
    ("age", "find_marvin_lidar_not_current"),
    ("unsafe", "marvin_single_approach_translation_vetoed")])
def test_returning_scan_session_invalid_or_unsafe_fails_closed(tmp_path, monkeypatch, fault, reason):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8), (0, .7)])
    runtime, behavior, _, events, _ = bundle
    def publish_fault():
        if fault == "session":
            runtime.lidar_worker.session = "replacement"
        elif fault in {"invalid", "age"}:
            old = runtime.world_model.get_lidar_obstacles
            def invalid(**kwargs):
                value = old(**kwargs)
                if fault == "invalid":
                    value.update(valid=False, reason="invalid_geometry")
                else:
                    value.update(effective_age_seconds=.300001)
                return value
            runtime.world_model.get_lidar_obstacles = invalid
        else:
            behavior.unsafe_forward = True
    state["on_publish"] = publish_fault
    if fault == "unsafe":
        original_stop = bundle[2].stop
        def stop_and_publish():
            stopped = original_stop()
            # Unsafe geometry persists in genuinely new scans after STOP.
            if behavior.unsafe_forward:
                behavior.sequence += 1
            return stopped
        bundle[2].stop = stop_and_publish
    result = run(runtime)
    assert result["state"] == "BLOCKED" and result["reason"] == reason
    assert len(motions(events)) == 1


def test_interlock_stop_failure_is_not_recoverable(tmp_path, monkeypatch):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)])
    state["stop_failure"] = True
    result = run(bundle[0])
    assert result["reason"] == "find_marvin_lidar_recovery_stop_unconfirmed"
    assert "stopped_lidar_wait" not in bundle[3]


@pytest.mark.parametrize("during", ["sleep", "poll"])
def test_stop_while_waiting_preempts_without_followup_action(tmp_path, monkeypatch, during):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime = bundle[0]
    state["on_sleep" if during == "sleep" else "on_read"] = lambda: runtime.submit_intent(
        {"intent": "STOP", "speech": "Stopping."})
    result = run(runtime)
    assert result["state"] == "STOPPED"
    assert len(motions(bundle[3])) == 1 and state["sleeps"] == (1 if during == "sleep" else 0)


def test_three_consecutive_interruptions_exhaust_budget(tmp_path, monkeypatch):
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)] * 3,
        interruptions=(True, True, True))
    result = run(bundle[0])
    assert result["state"] == "BLOCKED" and result["reason"] == "find_marvin_lidar_recovery_exhausted"
    assert result["consecutive_lidar_interruptions"] == result["interrupted_forward_attempts"] == 3
    assert result["completed_forward_actions"] == 0
    assert len(motions(bundle[3])) == 3


def test_successful_action_resets_failure_count(tmp_path, monkeypatch):
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .60)] * 5 + [(0, .5)],
        interruptions=(True, False, True, False, True))
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert [x["consecutive_interruptions"] for x in result["lidar_recovery_history"]] == [1, 1, 1]
    assert result["completed_forward_actions"] == 2 and result["interrupted_forward_attempts"] == 3


def test_lifetime_backstop_limits_alternating_success_and_interruptions(tmp_path, monkeypatch):
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)] * 7,
        interruptions=(True, False, True, False, True, False, True))
    bundle[0].MAX_MARVIN_REACQUISITION_EPISODES = 3
    result = run(bundle[0])
    assert result["reason"] == "find_marvin_lidar_recovery_exhausted"
    assert result["interrupted_forward_attempts"] == 4 and result["completed_forward_actions"] == 3


@pytest.mark.parametrize("camera_fault", ["repeat_camera", "stale_camera"])
def test_recovery_cannot_reuse_or_age_out_action_camera(tmp_path, monkeypatch, camera_fault):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8), (0, .7)])
    state["on_publish"] = lambda: setattr(bundle[1], camera_fault, True)
    result = run(bundle[0])
    assert result["state"] in {"REVERIFY_REQUIRED", "BLOCKED"}
    assert len(motions(bundle[3])) == 1


def test_recovery_keeps_all_existing_limits(tmp_path, monkeypatch):
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .60), (0, .5)])
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == .30
    assert bundle[0].MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS == 1.
    assert bundle[0].MARVIN_NEW_LIDAR_TIMEOUT_SECONDS == .60
    arrival = result["final_observation"]["arrival"]
    assert arrival["target_standoff_m"] == .50 and arrival["hard_safety_envelope_m"] == .45
    assert result["final_observation"]["opencv_tracker"]["threshold"] == .80


def test_reobserve_can_change_plan_from_forward_to_alignment(tmp_path, monkeypatch):
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .60), (120, .60), (0, .5)])
    bundle[2].on_motion = lambda: setattr(bundle[1], "sequence", bundle[1].sequence+1)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert [h["state"] for h in result["history"]] == ["ADVANCING", "ALIGNING"]
    assert result["completed_forward_actions"] == 0 and result["interrupted_forward_attempts"] == 1
    assert motions(bundle[3])[1][0] == "turn"


def test_tracker_loss_after_interruption_can_semantically_recover(tmp_path, monkeypatch):
    from test_find_marvin_reacquisition import recovery_runtime
    bundle, _, _ = interrupted_runtime(tmp_path, monkeypatch,
        [(0, .60), (0, .59), (0, .58), (0, .5)], factory=recovery_runtime)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert result["interrupted_forward_attempts"] == 1
    assert result["reacquisition_attempts"] == 1
    assert result["reacquisition_history"][0]["succeeded"] is True


def test_bridge_nonzero_after_stop_denies_recovery(tmp_path, monkeypatch):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8)])
    runtime, _, robot, _, _ = bundle
    read = robot.status
    def status():
        value = read()
        if state["outage"]:
            value["motion"]["linear_x"] = .01
        return value
    robot.status = status
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert result["mission_outcome"] == "safe_failure"
    assert "stopped_lidar_wait" not in bundle[3]


def test_new_jit_safety_is_required_even_after_recovery_wait_passes(tmp_path, monkeypatch):
    bundle, state, _ = interrupted_runtime(tmp_path, monkeypatch, [(0, .8), (0, .7)])
    def observe_fault():
        if not state["outage"] and len(motions(bundle[3])):
            bundle[1].unsafe_forward = True
    bundle[1].on_observe = observe_fault
    result = run(bundle[0])
    assert result["lidar_recovery_history"][0]["wait"]["ok"] is True
    assert result["state"] == "BLOCKED"
    assert len(motions(bundle[3])) == 1
