#!/usr/bin/env python3

import json
from types import SimpleNamespace
from unittest.mock import patch

from behavior_manager import BehaviorManager
from mission_types import create_mission
from runtime_api import RuntimeAPIHandler
from tracking_state import build_tracking_state, empty_tracking_state
from voice_relay.server import VoiceRelayHandler


def detection(timestamp="frame-1", cx=145.0):
    return {
        "timestamp": timestamp,
        "camera_running": True,
        "detections": [{
            "label": "backpack",
            "confidence": 0.12,
            "x1": cx - 60,
            "y1": 160,
            "x2": cx + 60,
            "y2": 460,
            "center_x": cx,
            "center_y": 310,
            "area": 36000,
            "image_width": 640,
            "image_height": 480,
        }],
    }


class ReadOnlyRobot:
    def __getattr__(self, name):
        if name in {"motion", "stop", "move_forward", "turn_left", "turn_right"}:
            raise AssertionError(f"preview called robot method {name}")
        raise AttributeError(name)


class WorldModelObservation:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []
        self.writes = 0

    def find_latest_entity_by_label(self, label, **kwargs):
        self.calls.append((label, kwargs))
        return dict(self.observation)

    def update_robot_state(self, **kwargs):
        self.writes += 1
        raise AssertionError("preview wrote World Model")


class CandidateVision:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.process_calls = 0

    def fetch_target_candidates(self, target):
        return dict(self.payloads.pop(0))

    @staticmethod
    def normalize_detection(item):
        return {
            "label": item["label"],
            "confidence": item["confidence"],
            "cx": item["center_x"],
            "cy": item["center_y"],
            "area": item["area"],
            "bbox": {
                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"],
            },
            "image_width": item["image_width"],
            "image_height": item["image_height"],
        }

    def process_detection_frame(self, _detections):
        self.process_calls += 1
        raise AssertionError("preview promoted a candidate")


def test_world_model_preview_is_read_only_and_actionable():
    observation = {
        "found": True,
        "stale": False,
        "label": "backpack",
        "confidence": 0.8,
        "cx": 145.0,
        "cy": 310.0,
        "area": 36000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 85, "y1": 160, "x2": 205, "y2": 460},
    }
    world = WorldModelObservation(observation)
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), world_model=world)
    result = manager.preview_find_object("Backpack")
    assert result["ok"] is True
    assert result["source"] == "world_model"
    assert result["authoritative"] is True
    assert result["state"] == "PREVIEW"
    assert result["bbox"]["x1"] == 85
    assert result["horizontal_error"] == -175.0
    assert world.calls[0][1]["refresh"] is False
    assert world.writes == 0


def test_candidate_preview_confirms_without_promotion_or_motion():
    vision = CandidateVision([
        detection("one"),
        detection("two", cx=146),
        detection("three", cx=147),
    ])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    manager._promote_confirmed_target = lambda *_args: (_ for _ in ()).throw(
        AssertionError("preview promoted a candidate")
    )
    result = manager.preview_find_object("backpack")
    assert result["ok"] is True
    assert result["source"] == "vision_candidate"
    assert result["authoritative"] is False
    assert result["target_found"] is True
    assert result["bbox"] is not None
    assert vision.process_calls == 0
    assert manager._last_target_confirmation_status == "sentinel"


def test_preview_duplicate_frames_fail_closed_without_promotion():
    payload = detection("same")
    vision = CandidateVision([payload, payload, payload])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    result = manager.preview_find_object("backpack")
    assert result["ok"] is False
    assert result["preview"] is True
    assert "confirmed" in result["reason"]
    assert vision.process_calls == 0
    assert manager._last_target_confirmation_status == "sentinel"


def test_production_confirmation_wrapper_updates_shared_status():
    vision = CandidateVision([
        detection("one"),
        detection("two", cx=146),
        detection("three", cx=147),
    ])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert manager._last_target_confirmation_status == "target_confirmed"


def test_preview_result_builds_tracking_without_runtime_mutation():
    result = {
        "behavior": "FIND_OBJECT",
        "state": "PREVIEW",
        "target": "backpack",
        "target_found": True,
        "target_observation": {
            "label": "backpack",
            "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            "cx": 55,
            "cy": 70,
            "area": 9000,
            "image_width": 640,
            "image_height": 480,
        },
        "target_label": "backpack",
        "target_center_x": 55,
        "target_center_y": 70,
        "target_area": 9000,
        "image_width": 640,
        "image_height": 480,
        "horizontal_error": -265,
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
    }
    tracking = build_tracking_state(result, previous=empty_tracking_state())
    assert tracking["state"] == "PREVIEW"
    assert tracking["bbox"]["x2"] == 100.0


class CapturingHandler(RuntimeAPIHandler):
    def __init__(self, runtime):
        self.path = ""
        self.server = SimpleNamespace(runtime=runtime)
        self.responses = []

    def send_json(self, status_code, payload):
        self.responses.append((status_code, payload))


def test_runtime_preview_endpoint_does_not_mutate_runtime_or_execute():
    before_tracking = {"state": "IDLE"}
    calls = []

    class FakeBehavior:
        def preview_find_object(self, target):
            calls.append(target)
            return {
                "ok": True,
                "preview": True,
                "authoritative": False,
                "source": "vision_candidate",
                "target": target,
                "state": "PREVIEW",
                "behavior": "FIND_OBJECT",
                "target_found": True,
                "target_label": target,
                "target_center_x": 320,
                "image_width": 640,
                "target_observation": {},
            }

        def execute(self, _mission):
            raise AssertionError("preview invoked execute")

    class Runtime:
        behavior_manager = FakeBehavior()
        tracking_state = dict(before_tracking)
        active_mission = None
        queue = []
        _control_generation = 0

    before_active_mission = Runtime.active_mission
    before_queue = list(Runtime.queue)

    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-object/preview?target=backpack"
    handler.server = SimpleNamespace(runtime=Runtime())
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_GET()
    assert responses[0][0] == 200
    assert responses[0][1]["preview"] is True
    assert calls == ["backpack"]
    assert handler.server.runtime.tracking_state == before_tracking
    assert handler.server.runtime.active_mission is before_active_mission
    assert handler.server.runtime.queue == before_queue
    assert handler.server.runtime._control_generation == 0


def _call_runtime_preview(result):
    class FakeBehavior:
        def preview_find_object(self, _target):
            return result

    class Runtime:
        behavior_manager = FakeBehavior()

    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-object/preview?target=backpack"
    handler.server = SimpleNamespace(runtime=Runtime())
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_GET()
    return responses[0]


def test_runtime_preview_serializes_candidate_confirmation_diagnostics():
    diagnostics = {
        "confirmation_status": "target_confirmed",
        "fetch_attempts": 2,
        "attempts": [{"response_timestamp": "frame-1"}],
    }
    result = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "source": "vision_candidate",
        "target": "backpack",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": True,
        "target_label": "backpack",
        "target_center_x": 320,
        "target_center_y": 300,
        "target_area": 12000,
        "image_width": 640,
        "image_height": 480,
        "target_observation": {
            "label": "backpack",
            "confidence": 0.12,
            "cx": 320,
            "cy": 300,
            "area": 12000,
            "image_width": 640,
            "image_height": 480,
            "bbox": {"x1": 260, "y1": 200, "x2": 380, "y2": 300},
        },
        "confirmation_diagnostics": diagnostics,
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["source"] == "vision_candidate"
    assert payload["authoritative"] is False
    assert payload["confirmation_diagnostics"] == diagnostics
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_runtime_preview_authoritative_result_keeps_diagnostics_null():
    result = {
        "ok": True,
        "preview": True,
        "authoritative": True,
        "source": "world_model",
        "target": "backpack",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": True,
        "target_label": "backpack",
        "target_center_x": 320,
        "target_center_y": 300,
        "target_area": 12000,
        "image_width": 640,
        "image_height": 480,
        "target_observation": {},
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["authoritative"] is True
    assert payload["confirmation_diagnostics"] is None
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_runtime_preview_failed_result_preserves_supplied_diagnostics():
    diagnostics = {
        "confirmation_status": "target_reconfirmation_failed",
        "terminal_reason": "insufficient_temporal_or_geometric_support",
    }
    result = {
        "ok": False,
        "preview": True,
        "authoritative": False,
        "source": "vision_candidate",
        "target": "backpack",
        "reason": "Target was not confirmed.",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": False,
        "confirmation_diagnostics": diagnostics,
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["ok"] is False
    assert payload["confirmation_diagnostics"] == diagnostics
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_voice_relay_preview_proxy_forwards_only_read_only_request():
    handler = object.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-object-preview?target=backpack"
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    with patch(
        "voice_relay.server.request_json",
        return_value={
            "ok": True,
            "status_code": 200,
            "data": {"ok": True, "preview": True, "target": "backpack"},
            "error": None,
        },
    ) as request:
        handler.do_GET()
    assert responses == [(200, {"ok": True, "preview": True, "target": "backpack"})]
    assert request.call_args.args[0] == "GET"
    assert "/find-object/preview?target=backpack" in request.call_args.args[1]


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("FIND_OBJECT preview tests passed.")
