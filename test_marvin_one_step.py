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
    def __init__(self, found=True, candidate_index=0):
        self.found = found
        self.candidate_index = candidate_index
        self.calls = []
        self.frame = 0

    def fetch_frame(self):
        self.frame += 1
        self.calls.append("frame")
        return SimpleNamespace(
            data=b"", width=640, height=480,
            received_at=f"2026-09-19T16:00:0{self.frame}+00:00",
        )

    def select_marvin_candidate(self, _frame, candidates):
        self.calls.append("select_marvin_candidate")
        if not self.found:
            return {
                "target": "marvin", "confirmed": False,
                "candidate_index": None,
                "source": "gemini_marvin_candidate_selection",
            }
        return {
            "target": "marvin", "confirmed": True,
            "candidate_index": self.candidate_index,
            "source": "gemini_marvin_candidate_selection",
        }

    def describe_marvin(self, _frame):
        raise AssertionError("one-step must not call describe_marvin")


class ProposalVision:
    def __init__(self, *, boxes=None, labels=None):
        bbox = (
            boxes[0] if isinstance(boxes, list) else boxes
        ) or {"x1": 260, "y1": 160, "x2": 380, "y2": 360}
        labels = labels or ["chair", "toilet", "teddy bear"]
        self.payloads = [
            {
                "timestamp": f"2026-09-19T16:00:0{index}+00:00",
                "camera_running": True,
                "image_width": 640,
                "image_height": 480,
                "detections": [{
                    "label": labels[index % len(labels)],
                    "confidence": 0.10,
                    **bbox,
                }],
            }
            for index in range(3)
        ]
        self.proposal_calls = 0
        self.target_queries = []

    def fetch_detection_proposals(self):
        self.proposal_calls += 1
        return self.payloads.pop(0)

    def fetch_target_candidates(self, target):
        self.target_queries.append(target)
        raise AssertionError("one-step must not query target candidates")

    @staticmethod
    def normalize_detection(item):
        return {
            "label": item["label"],
            "confidence": item["confidence"],
            "cx": (item["x1"] + item["x2"]) / 2.0,
            "cy": (item["y1"] + item["y2"]) / 2.0,
            "area": (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]),
            "bbox": {key: item[key] for key in ("x1", "y1", "x2", "y2")},
            "image_width": 640,
            "image_height": 480,
        }


class Tracker:
    def __init__(self, _frame, bbox, boxes=None):
        self.bbox = dict(bbox)
        self.boxes = list(boxes or [bbox, bbox])

    def update(self, _frame):
        return dict(self.boxes.pop(0)) if self.boxes else None


def manager(*, found=True, boxes=None, robot=None, proposal_boxes=None, candidate_index=0):
    robot = robot or Robot()
    instance = BehaviorManager(
        robot_client=robot,
        world_model=World(),
        vision_adapter=ProposalVision(boxes=proposal_boxes),
    )
    instance.semantic_vision = Semantic(found, candidate_index)
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


def test_unconfirmed_gemini_selection_never_forwards():
    instance, robot = manager(found=False)
    result = instance.execute(mission())

    assert result["completed"] is True
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["turn_chunks_attempted"] == 0
    assert result["post_step_stop_result"]["ok"] is True
    assert instance.semantic_vision.calls == ["frame", "select_marvin_candidate"]


def test_one_step_uses_proposals_and_selects_once_without_teddy_bear_query():
    instance, robot = manager()
    result = instance.execute(mission())
    assert result["ok"] is True
    assert instance.vision.proposal_calls == 3
    assert instance.vision.target_queries == []
    assert instance.semantic_vision.calls.count("select_marvin_candidate") == 1
    assert "describe_marvin" not in instance.semantic_vision.calls
    assert result["proposal_label"] == "chair"
    assert result["geometry_source"] == "yolo_proposal"
    assert result["identity_source"] == "gemini_marvin_candidate_selection"
    assert result["proposal_support"] == 3
    assert result["confirmation_diagnostics"]["maximum_frames"] == 3
    assert result["confirmation_diagnostics"]["minimum_support"] == 2
    assert result["confirmation_diagnostics"]["confirmation_window_seconds"] == 2.0
    assert result["yolo_seed_bbox"] == {
        "x1": 260, "y1": 160, "x2": 380, "y2": 360,
    }
    assert result["tracker_seed_bbox"] == {
        "x1": 236, "y1": 150, "x2": 404, "y2": 370,
    }


def test_one_tracker_frame_or_noncentered_tracker_fails_closed():
    instance, robot = manager(boxes=[{"x1": 260, "y1": 160, "x2": 380, "y2": 360}])
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert not [call for call in robot.calls if call[0] == "forward"]

    instance, robot = manager(boxes=[{"x1": 0, "y1": 160, "x2": 120, "y2": 360}] * 2)
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_NOT_CENTERED"
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["turn_chunks_attempted"] == result["centering_turn_chunks_attempted"] == 0


def test_off_center_yolo_proposal_never_forwards():
    seeds = []
    instance, robot = manager(
        proposal_boxes={"x1": 0, "y1": 160, "x2": 120, "y2": 360},
    )
    instance.world_model.get_lidar_obstacles = lambda **_kwargs: pytest.fail(
        "off-center proposal must not reach LiDAR motion-stage check",
    )
    robot.forward_interlock.refresh = lambda: pytest.fail(
        "off-center proposal must not reach interlock motion-stage check",
    )
    instance.marvin_local_tracker_factory = lambda frame, bbox: seeds.append(bbox)
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_NOT_CENTERED"
    assert result["reason"] == "marvin_yolo_proposal_not_centered"
    assert result["horizontal_error_pixels"] == -260.0
    assert result["steering_direction"] == "LEFT"
    assert result["executed"] is False
    assert result["completed"] is True
    assert seeds == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert robot.calls == [("stop",)]
    assert result["post_step_stop_result"]["ok"] is True


def test_positive_off_center_yolo_proposal_reports_right_without_motion():
    seeds = []
    instance, robot = manager(
        proposal_boxes={"x1": 520, "y1": 160, "x2": 640, "y2": 360},
    )
    instance.world_model.get_lidar_obstacles = lambda **_kwargs: pytest.fail(
        "off-center proposal must not reach LiDAR motion-stage check",
    )
    robot.forward_interlock.refresh = lambda: pytest.fail(
        "off-center proposal must not reach interlock motion-stage check",
    )
    instance.marvin_local_tracker_factory = lambda frame, bbox: seeds.append(bbox)
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_NOT_CENTERED"
    assert result["reason"] == "marvin_yolo_proposal_not_centered"
    assert result["horizontal_error_pixels"] == 260.0
    assert result["steering_direction"] == "RIGHT"
    assert result["executed"] is False
    assert result["completed"] is True
    assert seeds == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert robot.calls == [("stop",)]
    assert result["post_step_stop_result"]["ok"] is True


def test_unrelated_acquisition_value_error_remains_generic_blocked():
    instance, robot = manager()
    instance._acquire_marvin_proposal_tracker_observation = lambda **_kwargs: (
        (_ for _ in ()).throw(ValueError("different_acquisition_failure"))
    )
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert result["reason"] == "marvin_one_step_error"
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert robot.calls == [("stop",)]
    assert result["post_step_stop_result"]["ok"] is True


def test_invalid_proposal_bbox_fails_closed():
    instance, robot = manager(
        proposal_boxes={"x1": 100, "y1": 100, "x2": 90, "y2": 480},
    )
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_wide_centered_proposal_is_filtered_before_gemini():
    instance, robot = manager(
        proposal_boxes={"x1": 0, "y1": 0, "x2": 640, "y2": 480},
    )
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert result["reason"] == "marvin_one_step_error"
    assert instance.semantic_vision.calls == []
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert result["post_step_stop_result"]["ok"] is True


def test_insufficient_proposal_support_fails_closed():
    instance, robot = manager()
    instance.vision.payloads = instance.vision.payloads[:1]
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_invalid_candidate_selection_fails_closed():
    instance, robot = manager(candidate_index=9)
    result = instance.execute(mission())
    assert result["state"] == "MARVIN_ONE_STEP_BLOCKED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_preemption_during_proposal_acquisition_stops_without_forward():
    instance, robot = manager()
    checks = [0]

    def authorization():
        checks[0] += 1
        return checks[0] < 4

    instance.execution_authorization_provider = authorization
    result = instance.execute(mission())
    assert result["state"] == "PREEMPTED"
    assert not [call for call in robot.calls if call[0] == "forward"]
    assert robot.calls[-1] == ("stop",)


def test_preemption_after_gemini_selection_stops_without_forward():
    instance, robot = manager()

    def authorization():
        return "select_marvin_candidate" not in instance.semantic_vision.calls

    instance.execution_authorization_provider = authorization
    result = instance.execute(mission())
    assert result["state"] == "PREEMPTED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_preemption_during_tracker_confirmation_stops_without_forward():
    instance, robot = manager()

    def authorization():
        return instance.semantic_vision.calls.count("frame") < 2

    instance.execution_authorization_provider = authorization
    result = instance.execute(mission())
    assert result["state"] == "PREEMPTED"
    assert not [call for call in robot.calls if call[0] == "forward"]


def test_authoritative_tracker_geometry_is_published_not_semantic_seed():
    tracker_bbox = {"x1": 270, "y1": 160, "x2": 390, "y2": 360}
    instance, _robot = manager(boxes=[tracker_bbox, tracker_bbox])
    result = instance.execute(mission())
    assert result["authority_source"] == "marvin_local_tracker"
    assert result["bbox"] == tracker_bbox
    assert result["bbox"] != result["yolo_seed_bbox"]
    assert build_tracking_state(result)["bbox"] == {
        key: float(value) for key, value in tracker_bbox.items()
    }


def test_tracker_is_seeded_with_expanded_yolo_proposal_bbox():
    instance, _robot = manager()
    seeds = []

    def factory(frame, bbox):
        seeds.append(dict(bbox))
        return Tracker(frame, bbox)

    instance.marvin_local_tracker_factory = factory
    result = instance.execute(mission())
    assert result["ok"] is True
    assert seeds == [{"x1": 236, "y1": 150, "x2": 404, "y2": 370}]
    assert result["yolo_seed_bbox"] == {
        "x1": 260, "y1": 160, "x2": 380, "y2": 360,
    }
    assert result["tracker_seed_bbox"] == seeds[0]


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
