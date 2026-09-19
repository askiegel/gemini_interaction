"""Offline contracts for the explicit one-forward Marvin validation mode."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from behavior_manager import BehaviorManager
from mission_manager import MissionManager
from mission_types import create_mission
from tracking_state import build_tracking_state
from voice_relay.server import VoiceRelayHandler


HTML = Path("voice_relay/index.html").read_text(encoding="utf-8")


class Interlock:
    def __init__(self, permitted=True, reason="fresh_clear"):
        self.permitted = permitted
        self.reason = reason

    def refresh(self):
        return self.permitted, self.reason

    def status(self):
        return {"active_forward": False, "pending_forward": False}


class Robot:
    def __init__(self, *, forward_result=None, forward_error=None):
        self.forward_interlock = Interlock()
        self.forward_result = forward_result if forward_result is not None else {"ok": True}
        self.forward_error = forward_error
        self.calls = []

    def move_forward(self, *, speed, seconds):
        self.calls.append(("forward", speed, seconds))
        if self.forward_error:
            raise self.forward_error
        return self.forward_result

    def stop(self):
        self.calls.append(("stop",))
        return {"ok": True, "action": "stop"}


class World:
    def get_lidar_obstacles(self, *, expected_session):
        return {
            "producer_session": expected_session, "available": True,
            "valid": True, "reason": "fresh",
            "sectors": {"front": {"state": "CLEAR"}},
        }


class Semantic:
    def __init__(self, direction="CENTER", found=True):
        self.direction = direction
        self.found = found
        self.calls = 0
        self.frame = 0

    def fetch_frame(self):
        self.frame += 1
        return SimpleNamespace(
            data=b"", width=640, height=480,
            received_at=f"2026-09-19T16:00:0{self.frame}+00:00",
        )

    def describe_marvin(self, _frame):
        self.calls += 1
        return {
            "found": self.found, "source": "gemini_marvin",
            "coarse_direction": self.direction,
            "bbox": {"x1": 260, "y1": 160, "x2": 380, "y2": 360},
            "image_width": 640, "image_height": 480,
        }


class Tracker:
    def __init__(self, _frame, bbox, boxes=None):
        self.bbox = dict(bbox)
        self.boxes = list(boxes or [bbox, bbox])

    def update(self, _frame):
        return dict(self.boxes.pop(0)) if self.boxes else None


def manager(*, direction="CENTER", found=True, boxes=None, robot=None):
    robot = robot or Robot()
    instance = BehaviorManager(robot_client=robot, world_model=World())
    instance.semantic_vision = Semantic(direction, found)
    instance.marvin_local_tracker_factory = lambda frame, bbox: Tracker(frame, bbox, boxes)
    instance.lidar_session = "one-step-session"
    return instance, robot


def mission():
    return create_mission(
        mission_type="FIND_OBJECT", target="marvin", speech="test",
        marvin_one_step_test=True,
    )


def test_centered_confirmed_tracker_allows_one_forward_then_stop():
    instance, robot = manager()
    instance.execute_guarded_turn = lambda *_args, **_kwargs: pytest.fail(
        "one-step mode must never request a guarded turn",
    )
    result = instance.execute(mission())

    assert result["ok"] is True
    assert result["completed"] is True
    assert result["state"] == "MARVIN_ONE_STEP_COMPLETE"
    assert result["authority_source"] == "marvin_local_tracker"
    assert result["approach_chunks_attempted"] == result["approach_chunks_completed"] == 1
    assert result["turn_chunks_attempted"] == result["centering_turn_chunks_attempted"] == 0
    assert robot.calls == [("forward", 0.08, 0.50), ("stop",)]
    assert result["post_step_stop_result"]["ok"] is True


@pytest.mark.parametrize("direction,found", [("LEFT", True), ("RIGHT", True), ("UNKNOWN", True), ("CENTER", False)])
def test_semantic_noncenter_or_absent_never_forwards(direction, found):
    instance, robot = manager(direction=direction, found=found)
    result = instance.execute(mission())

    assert result["completed"] is True
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["turn_chunks_attempted"] == 0
    assert result["post_step_stop_result"]["ok"] is True


def test_one_tracker_frame_or_noncentered_tracker_fails_closed():
    instance, robot = manager(boxes=[{"x1": 260, "y1": 160, "x2": 380, "y2": 360}])
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_TRACKER_UNCONFIRMED"
    assert not [call for call in robot.calls if call[0] == "forward"]

    instance, robot = manager(boxes=[{"x1": 0, "y1": 160, "x2": 120, "y2": 360}] * 2)
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_NOT_CENTERED"
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["turn_chunks_attempted"] == result["centering_turn_chunks_attempted"] == 0


def test_semantic_center_text_with_left_bbox_cannot_become_centered():
    instance, robot = manager(
        boxes=[{"x1": 0, "y1": 160, "x2": 120, "y2": 360}] * 2,
    )
    instance.semantic_vision.describe_marvin = lambda _frame: {
        "found": True, "source": "gemini_marvin", "coarse_direction": "CENTER",
        "bbox": {"x1": 0, "y1": 160, "x2": 120, "y2": 360},
        "image_width": 640, "image_height": 480,
    }
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_NOT_CENTERED"
    assert result["horizontal_error_pixels"] < -50
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_implausibly_huge_marvin_semantic_bbox_fails_closed():
    instance, robot = manager()
    instance.semantic_vision.describe_marvin = lambda _frame: {
        "found": True, "source": "gemini_marvin", "coarse_direction": "CENTER",
        "bbox": {"x1": 0, "y1": 0, "x2": 640, "y2": 480},
        "image_width": 640, "image_height": 480,
    }
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_authoritative_tracker_geometry_is_published_not_semantic_seed():
    tracker_bbox = {"x1": 270, "y1": 160, "x2": 390, "y2": 360}
    instance, _robot = manager(boxes=[tracker_bbox, tracker_bbox])
    result = instance.execute(mission())
    assert result["authority_source"] == "marvin_local_tracker"
    assert result["bbox"] == tracker_bbox
    assert result["bbox"] != result["semantic_reacquisition_result"]["bbox"]
    assert build_tracking_state(result)["bbox"] == {
        key: float(value) for key, value in tracker_bbox.items()
    }


def test_tracker_is_seeded_with_validated_marvin_semantic_bbox():
    instance, _robot = manager()
    seeds = []

    def factory(frame, bbox):
        seeds.append(dict(bbox))
        return Tracker(frame, bbox)

    instance.marvin_local_tracker_factory = factory
    result = instance.execute(mission())
    assert result["ok"] is True
    assert seeds == [{"x1": 260, "y1": 160, "x2": 380, "y2": 360}]


@pytest.mark.parametrize("robot", [Robot(forward_result={"ok": False}), Robot(forward_error=RuntimeError("transport"))])
def test_failed_or_raised_forward_is_terminal_and_stopped(robot):
    instance, robot = manager(robot=robot)
    result = instance.execute(mission())
    assert result["ok"] is False
    assert result["completed"] is True
    assert len([call for call in robot.calls if call[0] == "forward"]) == 1
    assert robot.calls[-1] == ("stop",)


def test_preemption_before_dispatch_prevents_forward_and_is_not_hidden():
    instance, robot = manager()
    instance.execution_authorization_provider = lambda: False
    result = instance.execute(mission())
    assert result["state"] == "PREEMPTED"
    assert result["completed"] is True
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert robot.calls[-1] == ("stop",)


def safe_status(state="IDLE"):
    return {
        "runtime": {"connected": True, "running": True, "state": state, "last_error": None,
                    "lidar": {"running": True, "available": True, "valid": True, "reason": "fresh", "front_state": "CLEAR"},
                    "forward_interlock": {"configured": True, "monitor_running": True, "forward_permitted": True, "reason": "fresh_clear", "active_forward": False, "pending_forward": False}},
        "missions": {"active": None, "queue_count": 0},
        "robot": {"connected": True, "status": "READY", "ros_ready": True,
                  "motion": {"linear_x": 0, "angular_z": 0, "streaming": False}},
    }


def test_one_step_route_dry_run_and_idle_preflight_metadata():
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.dashboard_status = lambda: (_ for _ in ()).throw(AssertionError("dry run queried"))
    assert handler.submit_marvin_one_step_test()[1]["dry_run"] is True

    handler.dashboard_status = lambda: safe_status()
    with patch("voice_relay.server.request_json", return_value={"status_code": 202, "data": {"ok": True}, "error": None}) as request:
        code, _ = handler.submit_marvin_one_step_test(execute=True)
    assert code == 202
    intent = request.call_args.kwargs["payload"]["intent"]
    assert intent["target"] == "marvin"
    assert intent["marvin_one_step_test"] is True


@pytest.mark.parametrize("mutate", [
    lambda status: status["runtime"].update(state="STARTING"),
    lambda status: status["missions"].update(active={"mission_type": "FIND_OBJECT"}),
    lambda status: status["missions"].update(queue_count=1),
    lambda status: status["robot"]["motion"].update(linear_x=0.1),
    lambda status: status["runtime"]["lidar"].update(reason="stale"),
    lambda status: status["runtime"]["forward_interlock"].update(forward_permitted=False),
])
def test_one_step_route_fails_closed_for_unsafe_preflight(mutate):
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    status = safe_status()
    mutate(status)
    handler.dashboard_status = lambda: status
    with patch("voice_relay.server.request_json") as request:
        code, payload = handler.submit_marvin_one_step_test(execute=True)
    assert code == 409
    assert payload["accepted"] is False
    request.assert_not_called()


def test_nonmarvin_cannot_be_marked_as_one_step():
    mission = MissionManager().handle_intent({"intent": "FIND_OBJECT", "target": "backpack", "marvin_one_step_test": True})
    assert mission.status == "REJECTED"


def _route_handler(payload):
    handler = VoiceRelayHandler.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-marvin-one-step"
    handler.read_json_body = lambda: payload
    handler.send_json = Mock()
    return handler


@pytest.mark.parametrize("payload", [{}, {"execute": False}])
def test_one_step_http_missing_or_false_execute_is_dry_run(payload):
    handler = _route_handler(payload)
    handler.submit_marvin_one_step_test = Mock(
        return_value=(200, {"ok": True, "dry_run": True}),
    )
    handler.do_POST()
    handler.submit_marvin_one_step_test.assert_called_once_with(execute=False)
    assert handler.send_json.call_args.args[0] == 200


def test_one_step_http_true_reaches_submit_helper_only():
    handler = _route_handler({"execute": True})
    handler.submit_marvin_one_step_test = Mock(
        return_value=(202, {"ok": True, "accepted": True}),
    )
    handler.do_POST()
    handler.submit_marvin_one_step_test.assert_called_once_with(execute=True)
    assert handler.send_json.call_args.args[0] == 202


@pytest.mark.parametrize("payload", [{"execute": "true"}, {"execute": 1}, {"execute": None}, {"execute": False, "target": "marvin"}])
def test_one_step_http_rejects_malformed_execute_or_extra_fields(payload):
    handler = _route_handler(payload)
    handler.submit_marvin_one_step_test = Mock()
    handler.do_POST()
    handler.submit_marvin_one_step_test.assert_not_called()
    assert handler.send_json.call_args.args[0] == 400


def test_one_step_browser_posts_only_the_dedicated_execute_contract():
    feature = HTML.split('elements.marvinOneStepTestButton.addEventListener', 1)[1].split('document.querySelectorAll', 1)[0]
    assert 'fetch("/dashboard/find-marvin-one-step"' in feature
    assert 'body: JSON.stringify({execute: isLiveModeEnabled()})' in feature
    assert '/dashboard/find-marvin"' not in feature
    assert 'marvin_one_step_test' not in feature
    assert 'target:' not in feature
