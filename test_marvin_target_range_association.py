"""Offline association trust and full foreground-obstruction mission regressions."""
import math

import pytest

from marvin_lidar_standoff import evaluate_marvin_lidar_standoff
from marvin_target_range_association import MarvinTargetRangeAssociation
from test_marvin_lidar_standoff import lidar_at, CAMERA, SESSION
from test_find_marvin_runtime import _v2_preview
from test_find_marvin_closed_loop import make_runtime, run, motions, CALIBRATION
from test_find_marvin_local_obstacle_avoidance import obstacle_geometry


def evaluate(gate, distance, stamp, *, commit=True, lidar=None):
    scan = lidar or lidar_at(distance, sequence=stamp)
    tracker = _v2_preview(0, stamp=stamp)["opencv_tracker"]
    candidate = evaluate_marvin_lidar_standoff(tracker, scan, CAMERA, expected_session=SESSION)
    return gate.evaluate(candidate, scan, tracker, expected_session=SESSION, commit=commit)


def test_consistent_close_marvin_arrives_with_trusted_history():
    gate = MarvinTargetRangeAssociation()
    assert evaluate(gate, .51, 1)["target_range_association_trusted"]
    result = evaluate(gate, .49, 2)
    assert result["arrived_at_marvin"] and result["verified_marvin_distance_m"] == .49
    assert result["target_standoff_m"] == .50 and result["hard_safety_envelope_m"] == .45


@pytest.mark.parametrize("distance,trusted", [(.80, True), (.75, True), (.544623, False), (1.05, False)])
def test_pure_turn_has_no_translation_allowance(distance, trusted):
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    result = evaluate(gate, distance, 2)
    assert result["target_range_association_trusted"] is trusted
    assert result["translation_since_verified_range_bound_m"] == 0
    if not trusted:
        assert not result["arrived_at_marvin"]
        assert gate.anchor["measured_distance_m"] == .8
        assert result["verified_marvin_distance_m"] == .8


@pytest.mark.parametrize("distance,trusted", [(.75, True), (.66, True), (.54, False)])
def test_forward_change_bounded_by_command_plus_tolerance(distance, trusted):
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    gate.record_forward_bound(.10, .50)
    result = evaluate(gate, distance, 2)
    assert result["range_change_allowance_m"] == pytest.approx(.15)
    assert result["target_range_association_trusted"] is trusted


def test_ambiguous_candidate_does_not_poison_anchor_and_original_surface_recovers():
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    assert not evaluate(gate, .54, 2)["arrived_at_marvin"]
    recovered = evaluate(gate, .79, 3)
    assert recovered["target_range_association_trusted"]
    assert gate.anchor["measured_distance_m"] == .79


def test_jit_and_duplicate_camera_stamp_cannot_ratchet_anchor():
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    assert evaluate(gate, .71, 2, commit=False)["target_range_association_trusted"]
    assert not evaluate(gate, .62, 3, commit=False)["target_range_association_trusted"]
    assert gate.anchor["measured_distance_m"] == .8
    evaluate(gate, .71, 1)  # same camera stamp cannot advance history either
    assert gate.anchor["measured_distance_m"] == .8


def test_initial_same_bearing_blocker_cannot_authorize_arrival():
    result = evaluate(MarvinTargetRangeAssociation(), .49, 1)
    assert result["direct_path_blocked"]
    assert not result["target_range_association_trusted"] and not result["arrived_at_marvin"]
    assert result["verified_marvin_distance_m"] is None
    assert result["reason"] == "marvin_foreground_obstruction_suspected"


def test_initial_wall_across_image_box_is_ambiguous_even_with_clear_motion_probe():
    scan = lidar_at(.8)
    scan["local_motion_geometry"]["points"].extend(
        {"x_m": .8, "y_m": y / 1000} for y in range(-200, 201))
    result = evaluate(MarvinTargetRangeAssociation(), .8, 1, lidar=scan)
    assert not result["target_range_association_trusted"]
    assert result["reason"] == "marvin_target_range_initial_structure_ambiguous"


def install_projected_target(runtime, behavior, specs):
    read = behavior.lidar
    def projected(**kwargs):
        scan = read(**kwargs)
        spec = specs[min(max(0, len(behavior.stamps)-1), len(specs)-1)]
        error = spec[0]
        x = behavior.distance + .10
        y_center = -error * (x - CALIBRATION["x_m"]) / CALIBRATION["fx_pixels"]
        for p in scan["local_motion_geometry"]["points"]:
            if p["x_m"] == x:
                p["y_m"] += y_center
                p["distance_m"] = math.hypot(x, p["y_m"])
                p["robot_bearing_deg"] = math.degrees(math.atan2(p["y_m"], x))
        return scan
    behavior.lidar = projected
    runtime.world_model.get_lidar_obstacles = projected


def test_live_foreground_after_alignment_selects_one_left_detour_then_reobserves(tmp_path, monkeypatch):
    specs = [(120, .7), (0, .444623), (0, .7), (0, .65), (0, .6), (0, .55), (0, .5)]
    runtime, behavior, robot, events, _ = make_runtime(tmp_path, monkeypatch, specs)
    install_projected_target(runtime, behavior, specs)
    obstacle_geometry(runtime, behavior, events, [None, (1.2, .48), None])
    result = run(runtime)
    assert result["state"] == "ARRIVED", result
    assert [m[:2] for m in motions(events)[:2]] == [("turn", "RIGHT"), ("turn", "LEFT")]
    assert [row["state"] for row in result["history"][:2]] == ["ALIGNING", "AVOIDING"]
    obstructed = result["history"][1]["observation"]["arrival"]
    assert obstructed["candidate_target_return_distance_m"] == pytest.approx(.544623)
    assert obstructed["verified_marvin_distance_m"] > .8
    assert not obstructed["target_range_association_trusted"] and not obstructed["arrived_at_marvin"]
    assert obstructed["direct_path_blocked"]
    assert result["local_avoidance_actions"] == 1
    choice = result["local_avoidance_history"][0]["selection"]
    assert choice["direction"] == "LEFT" and choice["left_clearance_m"] > choice["right_clearance_m"]
    row = result["history"][1]
    wait = result["lidar_wait_history"][1]
    assert wait["snapshot"]["acquisition_sequence"] > row["action_lidar_evidence"][1]
    assert result["history"][2]["source_frame_stamp_ns"] > row["source_frame_stamp_ns"]
    index = events.index(("turn", "LEFT", .25, .50))
    assert events[index+1] == "stop"
    assert result["history"][2]["observation"]["arrival"]["target_range_association_trusted"]
    assert robot.status()["motion"]["streaming"] is False


def test_initial_foreground_with_no_range_history_enters_only_guarded_avoidance(tmp_path, monkeypatch):
    runtime, behavior, robot, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .444623)])
    obstacle_geometry(runtime, behavior, events, [(1.2, .48)])
    robot.on_motion = lambda: runtime.submit_intent({"intent": "STOP", "speech": "Stop."})
    result = run(runtime)
    assert motions(events) == [("turn", "LEFT", .25, .50)]
    assert result["behavior"] == "STOP"
    assert events[-1] == "stop"


@pytest.mark.parametrize("fault", ["stale", "invalid"])
def test_ambiguous_foreground_with_bad_sensor_never_selects_detour(tmp_path, monkeypatch, fault):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .444623)])
    read = obstacle_geometry(runtime, behavior, events, [(1.2, .48)])
    def bad(**kwargs):
        scan = read(**kwargs)
        scan.update(valid=False, reason=fault, effective_age_seconds=.31)
        return scan
    runtime.world_model.get_lidar_obstacles = bad
    monkeypatch.setattr("runtime.select_marvin_detour", lambda *a, **k: pytest.fail("Bad LiDAR cannot trigger avoidance"))
    result = run(runtime)
    assert result["state"] == "BLOCKED" and motions(events) == []


def test_many_turn_frames_cannot_ratchet_down_to_foreground_arrival():
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    assert evaluate(gate, .71, 2)["target_range_association_trusted"]
    assert not evaluate(gate, .62, 3)["target_range_association_trusted"]
    assert not evaluate(gate, .54, 4)["arrived_at_marvin"]


def test_behavior_jit_rejects_new_foreground_range_before_transport():
    from unittest.mock import Mock
    from behavior_manager import BehaviorManager
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    scan = lidar_at(.544623, sequence=2)
    robot = Mock()
    world = Mock()
    world.get_lidar_obstacles.return_value = scan
    behavior = BehaviorManager(robot_client=robot, world_model=world)
    def validate(tracker, lidar):
        candidate = evaluate_marvin_lidar_standoff(tracker, lidar, CAMERA, expected_session=SESSION)
        return gate.evaluate(candidate, lidar, tracker, expected_session=SESSION)
    result = behavior.execute_single_marvin_approach_step(expected_lidar_session=SESSION,
        linear_speed=.10, duration=.50, target_tracker=_v2_preview(0)["opencv_tracker"],
        camera_model=CAMERA, target_range_validator=validate)
    assert result["motion_executed"] is False
    assert not result["target_standoff"]["target_range_association_trusted"]
    robot.move_forward.assert_not_called()


def test_initial_close_center_band_return_without_edge_support_is_not_arrival():
    scan = lidar_at(.544623)
    scan["local_motion_geometry"]["points"] = [
        p for p in scan["local_motion_geometry"]["points"] if p["x_m"] != .544623]
    scan["local_motion_geometry"]["points"].extend(
        {"x_m": .544623, "y_m": n/1000} for n in range(-3,4))
    # Use the deployed uncertainty: measured .544623 -> conservative .444623.
    gate = MarvinTargetRangeAssociation()
    tracker = _v2_preview(0)["opencv_tracker"]
    camera = dict(CAMERA, range_uncertainty_m=.10)
    candidate = evaluate_marvin_lidar_standoff(tracker, scan, camera, expected_session=SESSION)
    result = gate.evaluate(candidate, scan, tracker, expected_session=SESSION, commit=True)
    assert candidate["target_distance_m"] == pytest.approx(.444623)
    assert not result["target_range_association_trusted"] and not result["arrived_at_marvin"]
    assert result["reason"] == "marvin_initial_arrival_association_ambiguous"
    assert gate.anchor is None


def test_saved_live_scan_d0da3104_cannot_declare_arrived(tmp_path, monkeypatch):
    import json
    import time
    from pathlib import Path
    saved = json.loads((Path(__file__).parent / 'test_data/marvin_foreground_mission_d0da3104.json').read_text())
    scan = dict(available=True, valid=True, reason="fresh", effective_age_seconds=0.,
        received_monotonic_seconds=time.monotonic(), age_at_receipt_seconds=0.,
        producer_session=saved['producer_session'], acquisition_sequence=saved['acquisition_sequence'],
        local_motion_geometry=saved['geometry'])
    tracker = saved['tracker']
    candidate = evaluate_marvin_lidar_standoff(tracker, scan, CALIBRATION, expected_session=saved['producer_session'])
    gate = MarvinTargetRangeAssociation()
    result = gate.evaluate(candidate, scan, tracker, expected_session=saved['producer_session'], commit=True)
    assert result['candidate_target_return_distance_m'] == pytest.approx(.544622592911801)
    assert candidate['target_distance_m'] == pytest.approx(.44462259291180106)
    assert not result['target_range_association_trusted'] and not result['arrived_at_marvin']
    assert not result['direct_path_blocked']  # Existing geometry guard permits the 5 cm probe.
    assert result['reason'] == 'marvin_target_range_returns_ambiguous'
    assert result['candidate_surface_point_count'] < 5
    assert not result['candidate_surface_bounded_by_bbox']
    assert gate.anchor is None

    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .444623)])
    # Use the exact scan/bbox with simulated current receipts/session and
    # independently advancing producer generations across the stopped handoff.
    sequence = [scan['acquisition_sequence']]
    def read(**kwargs):
        sequence[0] += 1
        return dict(scan, producer_session=runtime.lidar_worker.session,
                    received_monotonic_seconds=time.monotonic(),
                    acquisition_sequence=sequence[0])
    runtime.world_model.get_lidar_obstacles = read
    observer = behavior.observe_find_marvin_v2
    def observe():
        evidence = observer()
        preview = evidence['preview_result']
        preview['opencv_tracker'].update(bbox=tracker['bbox'])
        preview['bbox'] = tracker['bbox']
        return evidence
    behavior.observe_find_marvin_v2 = observe
    monkeypatch.setattr('runtime.select_marvin_detour', lambda *a, **k: pytest.fail('A clear motion probe cannot trigger detour'))
    mission = run(runtime)
    assert mission['state'] == 'BLOCKED' and not mission['arrived_at_marvin']
    assert mission['reason'] == 'find_marvin_blocked_wait_exhausted'
    assert mission['blocked_wait_reason'] == 'find_marvin_no_safe_local_detour'
    assert mission['final_observation']['route_to_marvin_obstructed']
    assert len(mission['local_avoidance_history'][0]['selection']['options']) == 4
    refresh, = mission['avoidance_lidar_refresh_history']
    assert refresh['avoidance_planning_lidar_sequence'] > refresh['blocked_forward_lidar_sequence']
    assert motions(events) == []


@pytest.mark.parametrize('quality', [.786365, .79, float('nan')])
def test_invalid_tracker_quality_cannot_seed_range_or_arrival(quality):
    gate = MarvinTargetRangeAssociation()
    tracker = _v2_preview(0, quality=quality)['opencv_tracker']
    scan = lidar_at(.544623)
    candidate = evaluate_marvin_lidar_standoff(tracker, scan,
        dict(CAMERA, range_uncertainty_m=.10), expected_session=SESSION)
    result = gate.evaluate(candidate, scan, tracker, expected_session=SESSION, commit=True)
    assert not result['target_range_association_trusted'] and not result['arrived_at_marvin']
    assert gate.anchor is None


def test_initial_range_requires_bounded_scan_surface_and_clear_candidate_probe():
    gate = MarvinTargetRangeAssociation()
    result = evaluate(gate, .8, 1)
    assert result['target_range_association_trusted']
    assert result['verified_marvin_distance_m'] == .8
    assert result['candidate_target_return_distance_m'] == .8
    assert result['nearest_forward_obstacle_distance_m'] == .8
    assert not result['direct_path_blocked']


def test_single_near_return_plus_background_cannot_seed_a_marvin_range():
    gate = MarvinTargetRangeAssociation()
    scan = lidar_at(1.0)
    scan['local_motion_geometry']['points'].append({'x_m': .8, 'y_m': 0.})
    result = evaluate(gate, .8, 1, lidar=scan)
    assert result['candidate_surface_point_count'] == 1
    assert not result['target_range_association_trusted']
    assert result['reason'] == 'marvin_target_range_returns_ambiguous'
    assert gate.anchor is None


def test_close_trusted_range_still_requires_alignment_before_arrival(tmp_path, monkeypatch):
    specs = [(120, .49), (0, .49)]
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, specs)
    install_projected_target(runtime, behavior, specs)
    result = run(runtime)
    assert result['state'] == 'ARRIVED', result
    assert motions(events) == [('turn', 'RIGHT', .25, .50)]
    first = result['history'][0]['observation']
    assert first['arrival']['target_range_association_trusted']
    assert not first['arrival']['arrived_at_marvin']
    assert first['controller']['decision'] == 'TURN_RIGHT'
    assert result['final_observation']['arrival']['target_range_association_trusted']


def test_semantic_reacquisition_cannot_reset_foreground_range_history(tmp_path, monkeypatch):
    from test_find_marvin_reacquisition import recovery_runtime
    specs = [(0, .7), (0, .7), (0, .444623), (0, .7),
             (0, .65), (0, .6), (0, .55), (0, .5)]
    runtime, behavior, _, events, _ = recovery_runtime(tmp_path, monkeypatch, specs)
    obstacle_geometry(runtime, behavior, events, [None, (1.2, .48), None])
    result = run(runtime)
    assert result['state'] == 'ARRIVED', result
    assert result['reacquisition_attempts'] == 1
    assert result['reacquisition_history'][0]['succeeded']
    regained = result['reacquisition_history'][0]['observation']
    assert regained['identity_source'] == 'gemini_marvin_candidate_selection'
    assert not regained['target_range_association_trusted']
    assert regained['verified_marvin_distance_m'] == pytest.approx(.8)
    assert regained['candidate_target_return_distance_m'] == pytest.approx(.544623)
    assert regained['controller']['path_state'] == 'DIRECT_PATH_BLOCKED'
    assert not regained['arrival']['arrived_at_marvin']
    assert motions(events)[1] == ('turn', 'LEFT', .25, .50)
    assert result['local_avoidance_actions'] == 1


def test_new_valid_producer_session_cannot_turn_range_ambiguity_into_avoidance():
    gate = MarvinTargetRangeAssociation()
    evaluate(gate, .8, 1)
    scan = dict(lidar_at(.49, sequence=2), producer_session='new-session')
    tracker = _v2_preview(0)['opencv_tracker']
    candidate = evaluate_marvin_lidar_standoff(tracker, scan, CAMERA, expected_session='new-session')
    assert candidate['ok']  # Fresh sensor data, but not the mission's producer.
    result = gate.evaluate(candidate, scan, tracker, expected_session='new-session', commit=True)
    assert not result['ok'] and not result['direct_path_blocked']
    assert not result['arrived_at_marvin'] and not result['target_range_association_trusted']
    assert result['reason'] == 'marvin_target_range_session_changed'
    assert gate.anchor['producer_session'] == SESSION
