from marvin_obstacle_phases import plan_phase_action
"""Stopped avoidance handoffs with real freshness, planning and strafe guards.

Only the existing offline sensor/transport fixture is used. Socket/process
access is forbidden, including when the timeout and STOP paths are exercised.
"""
import math
import socket
import time

import pytest

from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS, read_lidar_state
from local_motion_safety_envelope import LOCAL_LIDAR_PROTECTED_RADIUS_M
from marvin_lidar_standoff import TARGET_STANDOFF_M
from marvin_local_obstacle_avoidance import select_marvin_escape_action
from test_find_marvin_closed_loop import motions, run
from test_marvin_lateral_avoidance import strafe_runtime


# These points block the actual 5 cm command capsule and allow a safe strafe.
# The retained .483 m obstacle geometry is tested separately below; the camera
# scene's synthetic arrival label is independent of LiDAR target distance.
LEFT_OPEN = [(.498, 0.), (.499, .003), (.499, -.003), (0., 1.2), (0., -.48)]
RIGHT_OPEN = [(.498, 0.), (.499, .003), (.499, -.003), (0., .48), (0., -1.2)]


@pytest.fixture(autouse=True)
def no_live_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline handoff tests cannot access robot/services")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


def handoff_runtime(tmp_path, monkeypatch, *, publish=True, fault=None,
                    before=LEFT_OPEN, after=LEFT_OPEN, arrival=False,
                    stop_during=None, repeat_camera=False, stop_delay=.32,
                    pursuit_specs=None, advancing_planning=False):
    specs = pursuit_specs or ([(0, .54), (0, .49)] if arrival else [(0, .60), (0, .5)])
    bundle, _, client = strafe_runtime(tmp_path, monkeypatch, specs, [None])
    runtime, behavior, robot, events, clock = bundle
    behavior.freeze_lidar = True
    behavior.sequence = 10925
    behavior.repeat_camera = repeat_camera
    original_read = runtime.world_model.get_lidar_obstacles
    original_stop, original_sleep = robot.stop, time.sleep
    receipts = {}
    state = {"phase": "before", "sleeps": 0, "stopped_actions": 0,
             "reads": [], "candidate_scans": [], "pre_stop_stale": False,
             "published": [], "camera_scans": [], "initial_scan_age_before_refresh": None}
    observe = behavior.observe_find_marvin_v2
    def camera():
        state["camera_scans"].append(behavior.sequence)
        if state["phase"] == "planning":
            state["initial_scan_age_before_refresh"] = time.monotonic() - receipts[10925]
        evidence = observe()
        if arrival and state["phase"] == "planning":
            # Hypothetical future calibrated CAMERA policy only. Updating a
            # LiDAR distance must never manufacture an arrival observation.
            evidence["preview_result"]["simulated_visual_arrival"] = True
        return evidence
    behavior.observe_find_marvin_v2 = camera

    def read(**kwargs):
        scan = original_read(**kwargs)
        sequence = scan["acquisition_sequence"]
        scan["received_monotonic_seconds"] = receipts.setdefault(sequence, time.monotonic())
        scene = (before if state["phase"] in {"before", "waiting"} else after)
        if state["phase"] == "post":
            scene = None
        if scene:
            sectors = scan["local_motion_geometry"]["sectors"]
            names = ("front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right")
            for x, y in scene:
                scan["local_motion_geometry"]["points"].append({
                    "x_m": x, "y_m": y, "distance_m": math.hypot(x, y),
                    "robot_bearing_deg": math.degrees(math.atan2(y, x))})
                sector = sectors[names[int((math.degrees(math.atan2(y, x)) + 22.5) % 360 // 45)]]
                sector["minimum_distance_from_base_m"] = min(
                    sector["minimum_distance_from_base_m"], math.hypot(x, y))
        if state["phase"] == "planning":
            if fault == "session":
                scan["producer_session"] = "another-producer"
            elif fault == "invalid":
                scan.update(available=False, valid=False, reason="invalid_scan")
            elif fault == "stale":
                scan["received_monotonic_seconds"] -= .31
        scan = read_lidar_state(scan, expected_session=runtime.lidar_worker.session)
        state["reads"].append((sequence, scan["valid"], state["phase"]))
        if state["phase"] == "waiting":
            state["pre_stop_stale"] |= scan["reason"] == "stale"
            if stop_during == "poll":
                runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
        return scan

    def stop():
        result = original_stop()
        count = len(motions(events))
        if state["phase"] == "before":
            state["phase"] = "waiting"
            clock[0] += round(stop_delay * 1e9)  # STOP/status transport delay.
        elif count > state["stopped_actions"]:
            state["phase"] = "post"
            state["stopped_actions"] = count
            behavior.sequence += 1  # Producer publishes first post-action scan.
            state["published"].append(behavior.sequence)
        return result

    def sleep(seconds):
        assert all(robot.status()["motion"][axis] == 0 for axis in ("linear_x", "linear_y", "angular_z"))
        assert robot.status()["motion"]["streaming"] is False
        original_sleep(seconds)
        if state["phase"] == "waiting":
            state["sleeps"] += 1
            if stop_during == "sleep":
                runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
            if publish and state["sleeps"] == 2:
                behavior.sequence += 1  # A real mocked producer generation, never runtime synthesis.
                state["published"].append(behavior.sequence)
                state["phase"] = "planning"
        elif state["phase"] == "planning" and advancing_planning:
            # Avoidance has a separate stopped planning refresh. Simulate the
            # producer advancing while stationary, rather than re-stamping a
            # cached scan on reads. Keep the default frozen for the G3 tests.
            behavior.sequence += 1
            state["published"].append(behavior.sequence)

    def select(scan, association, **kwargs):
        assert scan["acquisition_sequence"] == association["acquisition_sequence"]
        assert scan["acquisition_sequence"] > 10925
        state["candidate_scans"].append(scan["acquisition_sequence"])
        return plan_phase_action(scan, association, **kwargs)

    runtime.world_model.get_lidar_obstacles = read
    client.forward_interlock.reader = read
    robot.stop = stop
    monkeypatch.setattr("runtime.time.sleep", sleep)
    monkeypatch.setattr("runtime.plan_phase_action", select)
    return bundle, state


def test_live_stale_10925_handoff_uses_new_scan_strafes_once_and_arrives(tmp_path, monkeypatch):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, advancing_planning=True)
    runtime, behavior, robot, events, _ = bundle
    result = run(runtime)
    assert state["initial_scan_age_before_refresh"] > MAXIMUM_EFFECTIVE_AGE_SECONDS
    assert state["sleeps"] == 2
    assert result["state"] == "ARRIVED" and result["arrived_at_marvin"]
    assert motions(events) == [("strafe", .08, 1.)]
    refresh, = result["avoidance_lidar_refresh_history"]
    camera_scan = state["camera_scans"][1]
    planning_scan = refresh["avoidance_planning_lidar_sequence"]
    assert 10925 < camera_scan < planning_scan
    assert camera_scan in state["published"] and planning_scan in state["published"]
    assert refresh["blocked_forward_lidar_sequence"] == camera_scan
    assert refresh["avoidance_lidar_refresh_wait_seconds"] == pytest.approx(.05)
    assert refresh["association"]["acquisition_sequence"] == planning_scan
    assert refresh["association"]["authority"] == "visual_tracking"
    assert "verified_marvin_distance_m" not in refresh["association"]
    assert refresh["wait"]["snapshot"]["valid"]
    selection = result["local_avoidance_history"][0]["selection"]
    assert set(selection["options"]) == {"STRAFE_LEFT", "STRAFE_RIGHT", "TURN_LEFT", "TURN_RIGHT"}
    assert selection["action_type"] == "STRAFE_LEFT"
    assert selection["acquisition_sequence"] == planning_scan
    assert all(s in state["published"] for s in state["candidate_scans"])
    action = result["history"][0]
    assert action["action_lidar_evidence"][1] == planning_scan
    assert result["lidar_wait_history"][0]["snapshot"]["acquisition_sequence"] > planning_scan
    assert action["result"]["source_stamp_consumed"]
    assert len(behavior.stamps) == 3  # Initial, stopped reacquisition, post-action.
    assert len(set(behavior.stamps)) == 3
    assert events[events.index(("strafe", .08, 1.)) + 1] == "stop"
    assert robot.status()["motion"]["linear_y"] == 0


def test_live_1235_marvin_and_0483_obstacle_resume_guarded_pursuit(tmp_path, monkeypatch):
    # Exact retained Marvin range; commanded bounds permit each later association.
    distance = 1.1353078150041862
    distances = [distance]
    distance -= .04
    while distance > .5:
        distances.append(distance)
        distance -= .05
    distances.append(.5)
    bundle, state = handoff_runtime(tmp_path, monkeypatch,
        advancing_planning=True,
        pursuit_specs=[(0, distance) for distance in distances],
        before=[(.46961726366025214, -.1132902556656369), (0., 1.2), (0., -.48)],
        after=[(.46961726366025214, -.1132902556656369), (0., 1.2), (0., -.48)])
    result = run(bundle[0])
    assert result["state"] == "ARRIVED" and result["arrived_at_marvin"]
    assert state["initial_scan_age_before_refresh"] > MAXIMUM_EFFECTIVE_AGE_SECONDS
    assert result["local_avoidance_actions"] == 1
    commands = motions(bundle[3])
    assert commands[0] == ("strafe", .08, 1.)
    assert all(command[0] == "forward" and command[1] == .10 and command[2] <= .50
               for command in commands[1:])
    refresh, = result["avoidance_lidar_refresh_history"]
    association = refresh["association"]
    assert "verified_marvin_distance_m" not in association
    assert association["authority"] == "visual_tracking"
    assert association["blocking_obstacle_distance_m"] == pytest.approx(.48308907704121)
    assert refresh["blocked_forward_lidar_sequence"] > 10925
    assert refresh["avoidance_planning_lidar_sequence"] > refresh["blocked_forward_lidar_sequence"]
    assert result["final_observation"]["arrival"]["visual_arrival"]["calibration_verified"]
    assert "target_distance_m" not in result["final_observation"]["arrival"]


@pytest.mark.parametrize("fault,reason", [
    ("never", "find_marvin_new_lidar_evidence_timeout"),
    ("session", "find_marvin_lidar_producer_session_changed"),
    ("invalid", "find_marvin_lidar_not_current"),
    ("stale", "find_marvin_lidar_not_current"),
])
def test_bad_new_evidence_fails_closed_without_motion(tmp_path, monkeypatch, fault, reason):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, publish=fault != "never", fault=fault)
    waits = []
    wait = bundle[0]._wait_for_new_marvin_lidar_evidence
    def retained_wait(**kwargs):
        result = wait(**kwargs)
        waits.append(result)
        return result
    monkeypatch.setattr(bundle[0], "_wait_for_new_marvin_lidar_evidence", retained_wait)
    result = run(bundle[0])
    assert result["state"] == "BLOCKED" and result["reason"] == reason
    assert not motions(bundle[3]) and not state["candidate_scans"]
    # Failure now occurs in the initial stopped wait/reacquisition, before a
    # replacement avoidance plan exists. Retain no-camera/no-transport checks.
    assert not result["avoidance_lidar_refresh_history"]
    assert result["stop_result"]["ok"]
    assert state["camera_scans"] == [10925]
    assert waits and all(w["poll_count"] <= 13 and w["wait_elapsed_seconds"] <= .600001 for w in waits)
    assert result["blocked_wait_total_seconds"] <= bundle[0].MARVIN_BLOCKED_WAIT_MAX_SECONDS
    assert bundle[2].status()["motion"] == {
        "linear_x": 0., "linear_y": 0., "angular_z": 0., "streaming": False}
    if fault == "never":
        assert state["pre_stop_stale"]
        assert not state["published"]
    else:
        assert any(seq > 10925 and phase == "planning" for seq, _, phase in state["reads"])


def test_obstacle_disappears_on_new_scan_forward_resumes_without_detour(tmp_path, monkeypatch):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, after=None)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert motions(bundle[3]) == [("forward", .10, .50)]
    assert result["local_avoidance_actions"] == 0 and not state["candidate_scans"]
    assert not result["avoidance_lidar_refresh_history"]
    assert result["blocked_wait_history"][-1]["lidar_sequence"] == 10926
    assert state["camera_scans"] == [10925, 10926, 10927]
    assert result["history"][0]["action_lidar_evidence"][1] == 10926
    assert result["history"][0]["observation"]["controller"]["decision"] == "FORWARD"


def test_new_geometry_reverses_pre_stop_left_ranking_and_selects_right(tmp_path, monkeypatch):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, before=LEFT_OPEN, after=RIGHT_OPEN,
        advancing_planning=True)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED"
    assert motions(bundle[3]) == [("strafe", -.08, 1.)]
    selection = result["local_avoidance_history"][0]["selection"]
    assert selection["action_type"] == "STRAFE_RIGHT"
    assert selection["acquisition_sequence"] > state["camera_scans"][1] > 10925
    assert selection["acquisition_sequence"] in state["published"]
    assert selection["right_clearance_m"] > selection["left_clearance_m"]


def test_trusted_arrival_on_new_scan_does_not_force_avoidance(tmp_path, monkeypatch):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, after=None, arrival=True)
    result = run(bundle[0])
    assert result["state"] == "ARRIVED" and result["arrived_at_marvin"]
    assert not motions(bundle[3]) and not state["candidate_scans"]
    association = result["final_observation"]["arrival"]
    assert association["acquisition_sequence"] == 10926
    assert association["authority"] == "visual_tracking"
    assert association["visual_arrival"]["calibration_verified"]
    assert "target_distance_m" not in association
    assert "target_range_association_trusted" not in association
    assert state["camera_scans"] == [10925, 10926]


@pytest.mark.parametrize("phase", ["poll", "sleep"])
def test_stop_during_avoidance_wait_preempts(tmp_path, monkeypatch, phase):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, publish=False, stop_during=phase)
    result = run(bundle[0])
    assert result["behavior"] == "STOP" and result["state"] == "STOPPED"
    assert not motions(bundle[3]) and not state["candidate_scans"]
    assert state["camera_scans"] == [10925] and not state["published"]
    if phase == "poll":
        assert len([r for r in state["reads"] if r[2] == "waiting"]) == 1
    else:
        assert state["sleeps"] <= 1


def test_refresh_cannot_reuse_consumed_camera_stamp(tmp_path, monkeypatch):
    bundle, _ = handoff_runtime(tmp_path, monkeypatch, repeat_camera=True)
    result = run(bundle[0])
    assert not motions(bundle[3])  # Duplicate stopped frame cannot authorize replacement.
    assert result["state"] == "REVERIFY_REQUIRED"
    assert result["reason"] == "find_marvin_blocked_wait_fresh_identity_required"
    assert bundle[1].stamps[0] == bundle[1].stamps[1]
    assert result["stop_result"]["ok"]


def test_camera_expiring_during_stop_is_not_reauthorized_by_new_lidar(tmp_path, monkeypatch):
    bundle, state = handoff_runtime(tmp_path, monkeypatch, stop_delay=.95, after=None)
    behavior = bundle[1]
    observe = behavior.observe_find_marvin_v2
    def expired_camera():
        # The stopped path now requires a new frame. A deliberately stale
        # replacement must still be refused even with the valid clearing scan.
        behavior.stale_camera = state["phase"] == "planning"
        return observe()
    behavior.observe_find_marvin_v2 = expired_camera
    result = run(bundle[0])
    assert result["state"] == "BLOCKED" and result["reason"] == "marvin_motion_observation_stale"
    assert not motions(bundle[3]) and not state["candidate_scans"]


def test_existing_safety_constants_unchanged():
    assert MAXIMUM_EFFECTIVE_AGE_SECONDS == .30
    assert LOCAL_LIDAR_PROTECTED_RADIUS_M == .45
    assert TARGET_STANDOFF_M == .50
