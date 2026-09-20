"""Stationary contracts for the bounded Marvin align-and-approach test."""
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

from mission_manager import MissionManager
from mission_types import create_mission
from behavior_manager import BehaviorManager
from test_marvin_one_step import (
    SequenceInterlock,
    SequenceWorld,
    manager as behavior_manager_factory,
)


def approach_mission():
    return create_mission(
        mission_type="FIND_OBJECT", target="marvin", speech="approach",
        marvin_guarded_approach_test=True,
    )


def bbox_for_error(error):
    center = 320 + error
    return {"x1": center - 50, "y1": 140, "x2": center + 50, "y2": 340}


def payload(bbox, index):
    return {
        "timestamp": f"2026-09-20T17:00:{index:02d}+00:00",
        "camera_running": True, "image_width": 640, "image_height": 480,
        "detections": [{"label": "chair", "confidence": 0.1, **bbox}],
    }


def configured(errors):
    instance, robot = behavior_manager_factory()
    payloads = []
    index = 1
    for error in errors:
        for _ in range(3):
            payloads.append(payload(bbox_for_error(error), index))
            index += 1
    instance.vision.payloads = payloads
    frame_number = [0]

    def fetch_frame():
        frame_number[0] += 1
        instance.semantic_vision.calls.append("frame")
        return SimpleNamespace(
            data=b"", width=640, height=480,
            received_at=f"2026-09-20T17:30:{frame_number[0]:02d}+00:00",
        )

    instance.semantic_vision.fetch_frame = fetch_frame
    turns = []
    instance._execute_target_directed_turn = lambda direction, speed, duration, **kwargs: (
        turns.append((direction, speed, duration)) or {"ok": True}
    )
    instance._marvin_one_step_forward_guard = lambda episode: {
        "authorized": True,
        "guards": {"forward_interlock_result": {"reason": "fresh_clear"}},
        "refresh": {
            "lidar_refresh_attempted": False,
            "lidar_refresh_attempt_count": 0,
            "lidar_refresh_succeeded": False,
            "lidar_refresh_initial_reason": "fresh",
            "lidar_refresh_final_reason": "fresh",
            "lidar_refresh_initial_acquisition_sequence": 1,
            "lidar_refresh_final_acquisition_sequence": 1,
        },
    }
    return instance, robot, turns


def test_three_centered_cycles_issue_three_forward_steps_only():
    instance, robot, turns = configured([0, 0, 0, 0])
    result = instance.execute(approach_mission())
    forwards = [call for call in robot.calls if call[0] == "forward"]
    assert result["state"] == "MARVIN_GUARDED_APPROACH_COMPLETE"
    assert len(forwards) == 3
    assert forwards == [("forward", 0.08, 0.50)] * 3
    assert turns == []
    assert result["motion_actions_attempted"] == 3
    assert result["approach_chunks_completed"] == 3


@pytest.mark.parametrize(
    "errors,expected_turns,expected_forwards",
    [
        ([120, 40, 0, 0, 0], [("RIGHT", 0.25, 0.25)], 3),
        ([-120, -40, 0, 0, 0], [("LEFT", 0.25, 0.25)], 3),
        ([120, 80, 40, 0, 0, 0], [("RIGHT", 0.25, 0.25)] * 2, 3),
        ([90, -70, 20, 0, 0, 0], [("RIGHT", 0.25, 0.25), ("LEFT", 0.25, 0.25)], 3),
    ],
)
def test_alignment_cycles_use_new_measurements(errors, expected_turns, expected_forwards):
    instance, robot, turns = configured(errors)
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_COMPLETE"
    assert turns == expected_turns
    assert len([call for call in robot.calls if call[0] == "forward"]) == expected_forwards
    assert result["motion_actions_attempted"] <= 6


def test_alignment_limit_allows_only_three_turns():
    instance, robot, turns = configured([120, 120, 120, 120])
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_ALIGNMENT_LIMIT"
    assert turns == [("RIGHT", 0.25, 0.25)] * 3
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["motion_actions_attempted"] == 3


def test_reacquisition_failure_after_forward_is_terminal_and_stopped():
    instance, robot, turns = configured([0])
    instance.vision.payloads = instance.vision.payloads[:3]
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_REACQUISITION_FAILED"
    assert len([call for call in robot.calls if call[0] == "forward"]) == 1
    assert turns == []
    assert result["post_step_stop_result"]["ok"] is True


def test_initial_acquisition_failure_has_no_motion():
    instance, robot, turns = configured([0])
    instance.vision.payloads = []
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_REACQUISITION_FAILED"
    assert turns == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_guard_denial_blocks_forward_and_preserves_stop():
    instance, robot, turns = configured([0])
    instance._marvin_one_step_forward_guard = lambda episode: {
        "authorized": False, "guards": {}, "refresh": {},
    }
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_BLOCKED"
    assert turns == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True
    assert result["approach_cycle_results"][-1]["selected_action"] == "forward"
    assert result["approach_cycle_results"][-1]["forward_attempted"] is False


def test_stale_refresh_uses_newer_sample_for_forward():
    instance, robot, turns = configured([0])
    instance.vision.payloads = instance.vision.payloads[:3]
    instance.world_model = SequenceWorld(["stale", "fresh"], [20, 21])
    robot.forward_interlock = SequenceInterlock([
        (False, "stale"), (True, "fresh_clear"),
    ])
    instance._marvin_one_step_forward_guard = BehaviorManager._marvin_one_step_forward_guard.__get__(instance)
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_REACQUISITION_FAILED"
    assert len([call for call in robot.calls if call[0] == "forward"]) == 1


def test_same_stale_sample_and_mixed_denial_never_forward():
    for reasons, sequences, interlocks in (
        (["stale", "stale", "fresh"], [20, 21, 21],
         [(False, "stale"), (False, "stale"), (True, "fresh_clear")]),
        (["stale"], [30], [(False, "blocked")]),
    ):
        instance, robot, _turns = configured([0])
        instance.vision.payloads = instance.vision.payloads[:3]
        instance.world_model = SequenceWorld(reasons, sequences)
        robot.forward_interlock = SequenceInterlock(interlocks)
        instance._marvin_one_step_forward_guard = BehaviorManager._marvin_one_step_forward_guard.__get__(instance)
        result = instance.execute(approach_mission())
        assert result["state"] == "MARVIN_GUARDED_APPROACH_BLOCKED"
        assert not [call for call in robot.calls if call[0] == "forward"]


def test_guarded_real_forward_guard_blocks_nonclear_front_without_retry():
    instance, robot, _turns = configured([0])
    instance.vision.payloads = instance.vision.payloads[:3]
    instance.world_model = SequenceWorld(["fresh"], [30])
    instance.world_model.responses[0]["sectors"]["front"]["state"] = "BLOCKED"
    robot.forward_interlock = SequenceInterlock([(False, "blocked")])
    instance._marvin_one_step_forward_guard = BehaviorManager._marvin_one_step_forward_guard.__get__(instance)
    result = instance.execute(approach_mission())
    assert result["state"] == "MARVIN_GUARDED_APPROACH_BLOCKED"
    assert result["approach_cycle_results"][-1]["selected_action"] == "forward"
    assert robot.forward_interlock.calls == 1
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_guarded_preemption_during_stale_refresh_wait_stops_without_forward():
    instance, robot, _turns = configured([0])
    instance.vision.payloads = instance.vision.payloads[:3]
    instance.world_model = SequenceWorld(["stale"], [30])
    robot.forward_interlock = SequenceInterlock([(False, "stale")])
    instance._marvin_one_step_forward_guard = BehaviorManager._marvin_one_step_forward_guard.__get__(instance)
    allowed = {"value": True}
    instance.execution_authorization_provider = lambda: allowed["value"]

    def revoke(_seconds):
        allowed["value"] = False

    with patch("behavior_manager.time.sleep", side_effect=revoke):
        result = instance.execute(approach_mission())
    assert result["state"] == "PREEMPTED"
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_preemption_before_acquisition_has_no_motion():
    instance, robot, turns = configured([0])
    instance.execution_authorization_provider = lambda: False
    result = instance.execute(approach_mission())
    assert result["state"] == "PREEMPTED"
    assert turns == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_preemption_after_turn_stop_prevents_next_cycle():
    instance, robot, turns = configured([120, 0])
    allowed = {"value": True}
    instance.execution_authorization_provider = lambda: allowed["value"]

    def turn(*args, **kwargs):
        turns.append((args[0], args[1], args[2]))
        allowed["value"] = False
        return {"ok": True}

    instance._execute_target_directed_turn = turn
    result = instance.execute(approach_mission())
    assert result["state"] == "PREEMPTED"
    assert len(turns) == 1
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_preemption_during_post_forward_acquisition_stops_progression():
    instance, robot, turns = configured([0, 0])
    original = instance._acquire_marvin_proposal_tracker_observation
    calls = [0]

    def acquire(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            instance.execution_authorization_provider = lambda: False
        return original(*args, **kwargs)

    instance._acquire_marvin_proposal_tracker_observation = acquire
    result = instance.execute(approach_mission())
    assert result["state"] == "PREEMPTED"
    assert len([call for call in robot.calls if call[0] == "forward"]) == 1
    assert turns == []
    assert result["post_step_stop_result"]["ok"] is True


@pytest.mark.parametrize("error, direction", [(-51, "LEFT"), (51, "RIGHT")])
def test_guarded_boundary_off_center_remains_found_and_turns_once(error, direction):
    instance, robot, turns = configured([error, 0, 0, 0])
    result = instance.execute(approach_mission())
    assert result["target_found"] is True
    assert result["approach_cycle_results"][0]["alignment"] == "OFF_CENTER"
    assert result["approach_cycle_results"][0]["selected_direction"] == direction
    assert turns == [(direction, 0.25, 0.25)]


@pytest.mark.parametrize("error", [-50, 50])
def test_guarded_exact_tolerance_is_centered_without_turn(error):
    instance, robot, turns = configured([error, 0, 0, 0])
    result = instance.execute(approach_mission())
    assert result["approach_cycle_results"][0]["alignment"] == "CENTERED"
    assert turns == []
    assert result["target_found"] is True
    assert len([call for call in robot.calls if call[0] == "forward"]) == 3


def test_browser_rotational_preflight_is_not_forward_coupled():
    html = Path("voice_relay/index.html").read_text(encoding="utf-8")
    assert "function marvinRotationalPreflight(status)" in html
    assert "marvinCenteringSubmitting || !rotationalPreflight.safe" in html
    assert "marvinGuardedApproachSubmitting || !rotationalPreflight.safe" in html
    assert "marvinOneStepSubmitting || !oneStepPreflight.safe" in html
    rotational = html.split("function marvinRotationalPreflight", 1)[1].split(
        "function clearTrackingOverlay", 1
    )[0]
    assert "forward_permitted" not in rotational
    assert "fresh_clear" not in rotational


def test_forward_guard_is_shared_by_one_step_and_guarded_approach():
    source = Path("behavior_manager.py").read_text(encoding="utf-8")
    assert source.count("def _marvin_one_step_forward_guard(") == 1
    assert source.count("self._marvin_one_step_forward_guard(episode)") >= 2


def test_turn_stop_failure_retains_completed_turn_cycle():
    instance, robot, turns = configured([120])
    robot.stop = lambda: {"ok": False, "error": "stop failed"}
    result = instance.execute(approach_mission())
    cycle = result["approach_cycle_results"][-1]
    assert result["state"] == "MARVIN_GUARDED_APPROACH_BLOCKED"
    assert result["reason"] == "turn_stop_failed"
    assert cycle["turn_attempted"] is True
    assert cycle["turn_completed"] is True
    assert cycle["stop_ok"] is False
    assert result["turn_chunks_completed"] == 1
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_forward_stop_failure_retains_completed_forward_cycle():
    instance, robot, turns = configured([0])
    robot.stop = lambda: {"ok": False, "error": "stop failed"}
    result = instance.execute(approach_mission())
    cycle = result["approach_cycle_results"][-1]
    assert result["state"] == "MARVIN_GUARDED_APPROACH_BLOCKED"
    assert result["reason"] == "forward_stop_failed"
    assert cycle["forward_attempted"] is True
    assert cycle["forward_completed"] is True
    assert cycle["stop_ok"] is False
    assert result["approach_chunks_completed"] == 1
    assert len([call for call in robot.calls if call[0] == "forward"]) == 1


def test_route_gate_and_payload_contract():
    from voice_relay.server import VoiceRelayHandler

    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-marvin-guarded-approach"
    handler.read_json_body = lambda: {}
    handler.send_json = Mock()
    handler.submit_marvin_guarded_approach = Mock(return_value=(200, {"dry_run": True}))
    handler.do_POST()
    handler.submit_marvin_guarded_approach.assert_called_once_with(execute=False)

    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: {
        "runtime": {"connected": True, "running": True, "state": "IDLE", "last_error": None,
                     "lidar": {"running": True, "available": True, "valid": True, "reason": "fresh"}},
        "missions": {"active": None, "queue_count": 0},
        "robot": {"connected": True, "status": "READY", "ros_ready": True,
                  "motion": {"linear_x": 0, "angular_z": 0, "streaming": False}},
    }
    with patch("voice_relay.server.request_json", return_value={"status_code": 202, "data": {"accepted": True}, "error": None}) as request:
        code, _ = handler.submit_marvin_guarded_approach(execute=True)
    assert code == 202
    assert request.call_args.kwargs["payload"]["intent"] == {
        "intent": "FIND_OBJECT", "speech": "Marvin Guarded Approach.",
        "target": "marvin", "marvin_guarded_approach_test": True,
    }


def test_guarded_route_does_not_require_forward_clearance():
    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: {
        "runtime": {
            "connected": True, "running": True, "state": "IDLE", "last_error": None,
            "lidar": {"running": True, "available": True, "valid": True, "reason": "fresh",
                      "front_state": "BLOCKED"},
            "forward_interlock": {"configured": True, "monitor_running": True,
                                  "forward_permitted": False, "reason": "front_not_clear",
                                  "active_forward": False, "pending_forward": False},
        },
        "missions": {"active": None, "queue_count": 0},
        "robot": {"connected": True, "status": "READY", "ros_ready": True,
                  "motion": {"linear_x": 0, "angular_z": 0, "streaming": False}},
    }
    with patch("voice_relay.server.request_json", return_value={
        "status_code": 202, "data": {"accepted": True}, "error": None,
    }) as request:
        code, _ = handler.submit_marvin_guarded_approach(execute=True)
    assert code == 202
    request.assert_called_once()


@pytest.mark.parametrize("body", [
    {"execute": "true"}, {"execute": 1}, {"execute": None},
    {"execute": True, "extra": 1},
])
def test_route_rejects_non_boolean_or_extra_fields(body):
    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-marvin-guarded-approach"
    handler.read_json_body = lambda: body
    handler.send_json = Mock()
    handler.submit_marvin_guarded_approach = Mock()
    handler.do_POST()
    handler.submit_marvin_guarded_approach.assert_not_called()
    assert handler.send_json.call_args.args[0] == 400


def test_nonmarvin_flag_rejected_and_browser_contract_is_dedicated():
    rejected = MissionManager().handle_intent({
        "intent": "FIND_OBJECT", "target": "backpack",
        "marvin_guarded_approach_test": True,
    })
    assert rejected.status == "REJECTED"
    html = Path("voice_relay/index.html").read_text(encoding="utf-8")
    feature = html.split(
        "elements.marvinGuardedApproachButton.addEventListener", 1
    )[1].split("document.querySelectorAll", 1)[0]
    assert 'fetch("/dashboard/find-marvin-guarded-approach"' in feature
    assert "body: JSON.stringify({execute: isLiveModeEnabled()})" in feature
