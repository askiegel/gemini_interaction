"""Stationary contracts for the bounded Marvin centering-step mode."""
from unittest.mock import Mock, patch
from pathlib import Path
import pytest

from mission_manager import MissionManager
from mission_types import create_mission

from test_marvin_one_step import manager


def centering_mission():
    return create_mission(
        mission_type="FIND_OBJECT", target="marvin", speech="center",
        marvin_centering_test=True,
    )


def proposal_payload(bbox, index):
    return {
        "timestamp": f"2026-09-19T17:00:{index:02d}+00:00",
        "camera_running": True, "image_width": 640, "image_height": 480,
        "detections": [{"label": "chair", "confidence": 0.1, **bbox}],
    }


def configured(initial, post=None):
    instance, robot = manager()
    post = post or initial
    instance.vision.payloads = [
        proposal_payload(initial, 1), proposal_payload(initial, 2),
        proposal_payload(initial, 3), proposal_payload(post, 4),
        proposal_payload(post, 5), proposal_payload(post, 6),
    ]
    turns = []
    instance._execute_target_directed_turn = lambda direction, speed, duration, **kwargs: (
        turns.append((direction, speed, duration)) or {"ok": True}
    )
    return instance, robot, turns


def bbox_at_error(error):
    center = 320 + error
    return {"x1": center - 50, "y1": 160, "x2": center + 50, "y2": 360}


@pytest.mark.parametrize("error,expected_turn", [
    (-50, None), (50, None), (-51, "LEFT"), (51, "RIGHT"),
])
def test_centering_tolerance_boundaries_preserve_found_status(error, expected_turn):
    instance, robot, turns = configured(bbox_at_error(error))
    result = instance.execute(centering_mission())
    assert result["target_found"] is True
    assert result["yolo_horizontal_error_pixels"] == error
    if expected_turn is None:
        assert result["alignment"] == "CENTERED"
        assert turns == []
    else:
        assert result["alignment"] == "OFF_CENTER"
        assert turns == [(expected_turn, 0.25, 0.25)]
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_off_center_left_turns_once_and_never_forwards():
    instance, robot, turns = configured({"x1": 140, "y1": 160, "x2": 260, "y2": 360})
    result = instance.execute(centering_mission())
    assert result["target_found"] is True
    assert result["alignment"] == "OFF_CENTER"
    assert turns == [("LEFT", 0.25, 0.25)]
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_off_center_right_turn_improves_and_reacquires():
    instance, robot, turns = configured(
        {"x1": 380, "y1": 160, "x2": 500, "y2": 360},
        {"x1": 330, "y1": 160, "x2": 450, "y2": 360},
    )
    result = instance.execute(centering_mission())
    assert result["state"] == "MARVIN_CENTERING_STEP_COMPLETE"
    assert result["target_found"] is True
    assert turns == [("RIGHT", 0.25, 0.25)]
    assert result["centering_improved"] is True
    assert result["post_turn_alignment"] == "OFF_CENTER"
    assert not [call for call in robot.calls if call[0] == "forward"]


@pytest.mark.parametrize("initial_error,post_error,turn,alignment", [
    (120, 70, "RIGHT", "OFF_CENTER"),
    (120, 35, "RIGHT", "CENTERED"),
    (-120, -70, "LEFT", "OFF_CENTER"),
    (-120, -35, "LEFT", "CENTERED"),
    (120, 150, "RIGHT", "OFF_CENTER"),
])
def test_post_turn_reacquisition_reports_improvement_without_second_turn(
    initial_error, post_error, turn, alignment,
):
    instance, robot, turns = configured(
        bbox_at_error(initial_error), bbox_at_error(post_error),
    )
    result = instance.execute(centering_mission())
    assert result["target_found"] is True
    assert result["post_turn_alignment"] == alignment
    assert result["centering_improved"] is (abs(post_error) < abs(initial_error))
    assert turns == [(turn, 0.25, 0.25)]
    assert len([call for call in robot.calls if call[0] == "forward"]) == 0


def test_exact_tolerance_is_already_aligned_without_turn():
    instance, robot, turns = configured(
        {"x1": 270, "y1": 160, "x2": 370, "y2": 360},
    )
    result = instance.execute(centering_mission())
    assert result["state"] == "MARVIN_CENTERING_ALREADY_ALIGNED"
    assert result["target_found"] is True
    assert result["alignment"] == "CENTERED"
    assert turns == []
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_reacquisition_failure_never_turns_twice_and_stops():
    instance, robot, turns = configured(
        {"x1": 380, "y1": 160, "x2": 500, "y2": 360},
    )
    instance.vision.payloads = instance.vision.payloads[:3]
    result = instance.execute(centering_mission())
    assert result["state"] == "MARVIN_CENTERING_REACQUISITION_FAILED"
    assert turns == [("RIGHT", 0.25, 0.25)]
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_stop_precedes_independent_post_turn_acquisition():
    instance, robot, turns = configured(
        bbox_at_error(120), bbox_at_error(70),
    )
    events = []
    original_acquire = instance._acquire_marvin_proposal_tracker_observation
    acquisition_count = [0]

    def acquire(*args, **kwargs):
        acquisition_count[0] += 1
        events.append("acquire")
        return original_acquire(*args, **kwargs)

    instance._acquire_marvin_proposal_tracker_observation = acquire
    instance._execute_target_directed_turn = lambda direction, speed, duration, **kwargs: (
        events.append("turn") or turns.append((direction, speed, duration)) or {"ok": True}
    )
    robot.stop = lambda: (events.append("stop") or {"ok": True})
    result = instance.execute(centering_mission())
    assert result["state"] == "MARVIN_CENTERING_STEP_COMPLETE"
    assert acquisition_count[0] == 2
    assert events == ["acquire", "turn", "stop", "acquire", "stop"]
    assert instance.semantic_vision.calls.count("select_marvin_candidate") == 2


def test_both_acquisitions_use_filtered_tracker_authority():
    instance, _robot, turns = configured(bbox_at_error(120), bbox_at_error(35))
    original_acquire = instance._acquire_marvin_proposal_tracker_observation
    observations = []

    def acquire(*args, **kwargs):
        result = original_acquire(*args, **kwargs)
        observations.append(result)
        return result

    instance._acquire_marvin_proposal_tracker_observation = acquire
    result = instance.execute(centering_mission())
    assert result["state"] == "MARVIN_CENTERING_STEP_COMPLETE"
    assert len(observations) == 2
    assert all(
        observation["source"] == "marvin_local_tracker"
        and observation["confirmation_diagnostics"]["marvin_geometry_filter_applied"] is True
        for observation in observations
    )
    assert instance.semantic_vision.calls.count("select_marvin_candidate") == 2


def test_preemption_before_turn_has_no_motion_and_stop_result():
    instance, robot, turns = configured(bbox_at_error(120))
    instance.execution_authorization_provider = lambda: False
    result = instance.execute(centering_mission())
    assert result["state"] == "PREEMPTED"
    assert turns == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_preemption_after_turn_and_stop_prevents_reacquisition():
    instance, robot, turns = configured(bbox_at_error(120), bbox_at_error(70))
    authorization = {"allowed": True}
    instance.execution_authorization_provider = lambda: authorization["allowed"]

    def turn(*args, **kwargs):
        turns.append((args[0], args[1], args[2]))
        authorization["allowed"] = False
        return {"ok": True}

    instance._execute_target_directed_turn = turn
    result = instance.execute(centering_mission())
    assert result["state"] == "PREEMPTED"
    assert len(turns) == 1
    assert instance.semantic_vision.calls.count("select_marvin_candidate") == 1
    assert result["post_step_stop_result"]["ok"] is True


def test_preemption_during_post_turn_acquisition_prevents_second_motion():
    instance, robot, turns = configured(bbox_at_error(120), bbox_at_error(70))
    original_acquire = instance._acquire_marvin_proposal_tracker_observation
    count = [0]

    def acquire(*args, **kwargs):
        count[0] += 1
        if count[0] == 2:
            instance.execution_authorization_provider = lambda: False
        return original_acquire(*args, **kwargs)

    instance._acquire_marvin_proposal_tracker_observation = acquire
    result = instance.execute(centering_mission())
    assert result["state"] == "PREEMPTED"
    assert len(turns) == 1
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_centering_flag_is_marvin_only_and_route_is_dry_run():
    rejected = MissionManager().handle_intent({
        "intent": "FIND_OBJECT", "target": "backpack",
        "marvin_centering_test": True,
    })
    assert rejected.status == "REJECTED"

    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.submit_marvin_centering_step = Mock(return_value=(200, {"dry_run": True}))
    handler.path = "/dashboard/find-marvin-centering-step"
    handler.read_json_body = lambda: {}
    handler.send_json = Mock()
    handler.do_POST()
    handler.submit_marvin_centering_step.assert_called_once_with(execute=False)
    assert handler.send_json.call_args.args[0] == 200


def test_centering_execute_route_submits_only_centering_mission():
    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: {
        "runtime": {
            "connected": True, "running": True, "state": "IDLE", "last_error": None,
            "lidar": {"running": True, "available": True, "valid": True,
                      "reason": "fresh", "front_state": "CLEAR"},
            "forward_interlock": {"configured": True, "monitor_running": True,
                                  "forward_permitted": True, "reason": "fresh_clear",
                                  "active_forward": False, "pending_forward": False},
        },
        "missions": {"active": None, "queue_count": 0},
        "robot": {"connected": True, "status": "READY", "ros_ready": True,
                  "motion": {"linear_x": 0, "angular_z": 0, "streaming": False}},
    }
    with patch("voice_relay.server.request_json", return_value={
        "status_code": 202, "data": {"accepted": True}, "error": None,
    }) as request:
        code, _ = handler.submit_marvin_centering_step(execute=True)
    assert code == 202
    assert request.call_args.kwargs["payload"]["intent"] == {
        "intent": "FIND_OBJECT", "speech": "Marvin Centering Step.",
        "target": "marvin", "marvin_centering_test": True,
    }


def test_centering_route_does_not_require_forward_clearance():
    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: {
        "runtime": {
            "connected": True, "running": True, "state": "IDLE", "last_error": None,
            "lidar": {"running": True, "available": True, "valid": True, "reason": "fresh"},
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
        code, _ = handler.submit_marvin_centering_step(execute=True)
    assert code == 202
    request.assert_called_once()


def test_centering_button_uses_dedicated_execute_contract():
    html = Path("voice_relay/index.html").read_text(encoding="utf-8")
    feature = html.split(
        'elements.marvinCenteringStepButton.addEventListener', 1
    )[1].split('document.querySelectorAll', 1)[0]
    assert 'fetch("/dashboard/find-marvin-centering-step"' in feature
    assert 'body: JSON.stringify({execute: isLiveModeEnabled()})' in feature
    assert "/dashboard/find-marvin-one-step" not in feature


@pytest.mark.parametrize("payload", [
    {"execute": "true"}, {"execute": 1}, {"execute": None},
    {"execute": True, "unexpected": 1},
])
def test_centering_route_rejects_malformed_or_extra_payload(payload):
    from voice_relay.server import VoiceRelayHandler
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-marvin-centering-step"
    handler.read_json_body = lambda: payload
    handler.send_json = Mock()
    handler.submit_marvin_centering_step = Mock()
    handler.do_POST()
    handler.submit_marvin_centering_step.assert_not_called()
    assert handler.send_json.call_args.args[0] == 400
