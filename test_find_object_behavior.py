#!/usr/bin/env python3

import json
import math
import threading

import pytest

import behavior_manager as behavior_module
from behavior_manager import BehaviorManager
from mission_types import create_mission
from runtime import CognitiveRuntime
from tracking_state import build_tracking_state, empty_tracking_state


class FakeRobotBridgeClient:
    def __init__(self):
        self.calls = []

    def move_forward(
        self,
        speed=0.10,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "move_forward",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def turn_left(
        self,
        speed=0.5,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "turn_left",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def turn_right(
        self,
        speed=0.5,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "turn_right",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def stop(self):
        self.calls.append(
            ("stop",)
        )

        return {
            "ok": True,
            "action": "stop",
        }


class SequencedVisionAdapter:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def find_target(self, target):
        if not self.results:
            raise AssertionError(
                "Vision sequence exhausted."
            )

        self.calls += 1

        result = dict(
            self.results.pop(0)
        )

        result.setdefault(
            "target",
            target,
        )

        return result


class CandidateVisionAdapter(SequencedVisionAdapter):
    def __init__(self, observations, candidate_payloads):
        super().__init__(observations)
        self.candidate_payloads = list(candidate_payloads)
        self.candidate_calls = 0
        self.promotions = []

    def fetch_target_candidates(self, target):
        self.candidate_calls += 1
        if not self.candidate_payloads:
            raise AssertionError("Candidate sequence exhausted.")
        return dict(self.candidate_payloads.pop(0))

    @staticmethod
    def normalize_detection(detection):
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        return {
            "label": detection["label"],
            "confidence": float(detection["confidence"]),
            "cx": float(detection["center_x"]),
            "cy": float(detection["center_y"]),
            "area": float(detection["area"]),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_width": float(detection["image_width"]),
            "image_height": float(detection["image_height"]),
        }

    def process_detection_frame(self, detections):
        self.promotions.extend(detections)
        return detections


def candidate_detection(
    timestamp,
    *,
    bbox=(200, 160, 500, 460),
    confidence=0.15,
    label="backpack",
):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return {
        "timestamp": timestamp,
        "camera_running": True,
        "label": label,
        "found": True,
        "detections": [{
            "label": label,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "area": width * height,
            "image_width": 640,
            "image_height": 480,
        }],
    }


class GuardedSearchRobot:
    def __init__(self):
        self.calls = []


def not_found(target="backpack", stale=False):
    return {
        "found": False,
        "stale": stale,
        "target": target,
        "reason": "Target not visible.",
    }


def found_target(target="backpack"):
    return {
        "found": True,
        "stale": False,
        "target": target,
        "entity_id": "backpack-001",
        "confidence": 0.9,
        "cx": 320.0,
        "cy": 240.0,
        "area": 12000.0,
        "image_width": 640.0,
        "image_height": 480.0,
    }


def semantic_only_target(target="backpack"):
    result = found_target(target)
    result.update(
        cx=None,
        cy=None,
        area=None,
        bbox=None,
        image_width=None,
        image_height=None,
    )
    return result


def guarded_search_manager(observations, turn_results=None):
    robot = GuardedSearchRobot()
    vision = SequencedVisionAdapter(observations)
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    manager.lidar_session = "session-1"
    calls = []
    results = list(turn_results or [])

    def execute_guarded_turn(direction, speed, duration, *, expected_lidar_session, now=None):
        calls.append((direction, speed, duration, expected_lidar_session))
        if results:
            return dict(results.pop(0))
        return {"ok": True, "permitted": True, "reason": "completed"}

    manager.execute_guarded_turn = execute_guarded_turn
    return manager, robot, vision, calls


def _mission():
    return create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Find my backpack",
        status="ACTIVE",
    )


def test_guarded_search_visible_before_turn_uses_zero_chunks():
    manager, robot, _vision, calls = guarded_search_manager(
        [found_target()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is True
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 0
    assert calls == []
    assert robot.calls == []


def test_guarded_search_rechecks_camera_after_first_chunk():
    manager, _robot, vision, calls = guarded_search_manager(
        [not_found(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1
    assert calls[0] == ("LEFT", 0.30, 1.0, "session-1")
    assert vision.calls == 2


def test_fresh_semantic_only_target_is_not_acquired_immediately():
    manager, _robot, _vision, calls = guarded_search_manager(
        [semantic_only_target()] * 4
    )
    result = manager.execute(_mission())
    assert result["target_found"] is False
    assert result["state"] == "SEARCH_EXHAUSTED"
    assert result["turn_chunks_attempted"] == 3
    assert len(calls) == 3


def test_semantic_only_target_then_actionable_geometry_is_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [semantic_only_target(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is True
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("cx", math.nan),
        ("cy", math.inf),
        ("area", 0.0),
        ("area", -1.0),
        ("image_width", 0.0),
        ("image_height", -1.0),
    ],
)
def test_invalid_target_geometry_is_not_acquired(field, value):
    invalid = found_target()
    invalid[field] = value
    manager, _robot, _vision, calls = guarded_search_manager(
        [invalid, found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_search_stops_after_second_chunk_when_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found(), not_found(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 2
    assert len(calls) == 2
    assert all(call[2] == 1.0 for call in calls)


def test_guarded_search_exhausts_at_three_independent_chunks():
    manager, _robot, vision, calls = guarded_search_manager(
        [not_found(), not_found(), not_found(), not_found()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["completed"] is False
    assert result["search_exhausted"] is True
    assert result["turn_chunks_attempted"] == 3
    assert result["turn_chunks_completed"] == 3
    assert len(calls) == 3
    assert all(call[1:3] == (0.30, 1.0) for call in calls)
    assert vision.calls == 4


def test_guarded_search_denial_stops_without_replay():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found()],
        [{"ok": False, "permitted": False, "reason": "stale_lidar"}],
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["completed"] is False
    assert result["state"] == "SEARCH_BLOCKED"
    assert result["reason"] == "stale_lidar"
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 0
    assert result["last_guarded_turn_result"]["reason"] == "stale_lidar"
    assert len(calls) == 1


def test_guarded_search_failure_after_successful_chunk_stops_immediately():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found(), not_found(), not_found()],
        [
            {"ok": True, "permitted": True},
            {"ok": False, "permitted": True, "reason": "transport_exception"},
        ],
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["turn_chunks_attempted"] == 2
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 2


def test_stale_camera_detection_is_not_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [found_target(), found_target()]
    )
    # Mark the first observation stale after construction without changing
    # the existing target schema.
    manager.vision.results[0]["stale"] = True
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_search_results_are_json_serializable():
    cases = (
        ([found_target()], None),
        ([not_found()] * 4, None),
        ([not_found()], [{"ok": False, "reason": "turn_side_not_clear"}]),
    )
    for observations, turn_results in cases:
        manager, _robot, _vision, _calls = guarded_search_manager(
            observations, turn_results,
        )
        json.dumps(manager.execute(_mission()))


def test_same_candidate_timestamp_does_not_confirm(monkeypatch):
    payload = candidate_detection("frame-1")
    vision = CandidateVisionAdapter([], [payload, payload, payload])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    clock = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: next(clock, 1.0))
    monkeypatch.setattr(behavior_module.time, "sleep", lambda _seconds: None)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"


def test_malformed_same_label_candidates_are_target_lost():
    payloads = []
    for timestamp in ("bad-1", "bad-2", "bad-3"):
        payload = candidate_detection(timestamp)
        payload["detections"][0]["area"] = float("nan")
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_lost"


def test_two_consistent_distinct_frames_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", confidence=0.12),
            candidate_detection("frame-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["source_timestamp"] in {"frame-1", "frame-2"}


def test_bbox_shape_variation_matches_by_center_and_area():
    first = {
        "label": "backpack",
        "cx": 213.5,
        "cy": 368.0,
        "area": 40836.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 89.0, "y1": 286.0, "x2": 338.0, "y2": 450.0},
    }
    second = {
        "label": "backpack",
        "cx": 229.5,
        "cy": 293.5,
        "area": 23095.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 152.0, "y1": 219.0, "x2": 307.0, "y2": 368.0},
    }
    assert behavior_module.BehaviorManager._target_bbox_iou(first, second) < 0.50
    assert (
        behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
            first, second
        )
        >= 0.50
    )
    assert behavior_module.BehaviorManager._target_observations_match(first, second)


def test_nested_live_backpack_boxes_match_by_intersection_over_smaller():
    first = {
        "label": "backpack",
        "cx": 213.5,
        "cy": 337.5,
        "area": 59474.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 82.0, "y1": 224.0, "x2": 345.0, "y2": 451.0},
    }
    second = {
        "label": "backpack",
        "cx": 232.5,
        "cy": 277.5,
        "area": 17331.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 153.0, "y1": 223.0, "x2": 312.0, "y2": 332.0},
    }
    assert behavior_module.BehaviorManager._target_bbox_iou(first, second) < 0.50
    assert max(first["area"], second["area"]) / min(first["area"], second["area"]) > 2.0
    containment = behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
        first, second
    )
    assert containment >= 0.50
    assert behavior_module.BehaviorManager._target_observations_match(first, second)


def test_nested_association_rejects_nonoverlap_and_different_labels():
    base = {
        "label": "backpack",
        "cx": 200.0,
        "cy": 250.0,
        "area": 10000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 150.0, "y1": 200.0, "x2": 250.0, "y2": 300.0},
    }
    separated = dict(
        base,
        cx=205.0,
        cy=450.0,
        area=25000.0,
        bbox={"x1": 155.0, "y1": 400.0, "x2": 305.0, "y2": 566.67},
    )
    other_label = dict(separated, label="suitcase")
    assert behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
        base, separated
    ) < 0.50
    assert not behavior_module.BehaviorManager._target_observations_match(base, separated)
    assert not behavior_module.BehaviorManager._target_observations_match(base, other_label)


def test_target_association_rejects_area_or_center_outliers():
    base = {
        "label": "backpack",
        "cx": 200.0,
        "cy": 250.0,
        "area": 10000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 150.0, "y1": 200.0, "x2": 250.0, "y2": 300.0},
    }
    area_outlier = dict(
        base,
        area=21000.0,
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    center_outlier = dict(
        base,
        cx=261.0,
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    other_label = dict(
        base,
        label="suitcase",
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, area_outlier
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, center_outlier
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, other_label
    )


def test_target_reconfirmation_failure_is_distinct_from_target_loss():
    payload = candidate_detection("same-frame")
    vision = CandidateVisionAdapter([], [payload, payload, payload])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"

    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [payload, payload, payload],
    )
    result = manager.execute(_mission())
    assert result["state"] == "TARGET_RECONFIRMATION_FAILED"
    assert result["completed"] is False
    assert len(calls) == 1


def test_only_one_consistent_frame_does_not_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", bbox=(20, 20, 120, 120)),
            candidate_detection("frame-3", bbox=(350, 20, 450, 120)),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"


def test_large_cluster_wins_using_all_frame_candidates():
    frame_two = candidate_detection(
        "frame-2",
        bbox=(235, 235, 310, 310),
        confidence=0.30,
    )
    frame_two["detections"].append(
        candidate_detection(
            "frame-2",
            bbox=(205, 165, 505, 465),
            confidence=0.10,
        )["detections"][0]
    )
    vision = CandidateVisionAdapter(
        [],
        [candidate_detection("frame-1"), frame_two, candidate_detection("frame-3")],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["area"] > 50000


def test_umbrella_never_substitutes_for_backpack():
    payloads = []
    for timestamp in ("frame-1", "frame-2", "frame-3"):
        payload = candidate_detection(timestamp, label="umbrella", confidence=0.99)
        payload["detections"].append(
            candidate_detection(timestamp, confidence=0.10)["detections"][0]
        )
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["label"] == "backpack"


def test_malformed_candidate_geometry_does_not_confirm():
    payloads = []
    for timestamp in ("frame-1", "frame-2", "frame-3"):
        payload = candidate_detection(timestamp)
        payload["detections"][0]["area"] = float("nan")
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None


def test_candidate_endpoint_failure_does_not_promote_or_acquire():
    vision = CandidateVisionAdapter([not_found()], [])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: calls.append(args) or {
        "ok": True,
        "permitted": True,
    }
    result = manager.execute(_mission())
    assert result["target_found"] is False
    assert result["turn_chunks_attempted"] == 1
    assert vision.promotions == []
    assert len(calls) == 1


def test_two_consistent_small_candidates_can_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1", bbox=(220, 220, 295, 295)),
            candidate_detection("frame-2", bbox=(221, 221, 296, 296)),
            candidate_detection("frame-3", bbox=(222, 222, 297, 297)),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["area"] == 5625.0


def test_confirmed_candidate_is_promoted_without_turn():
    vision = CandidateVisionAdapter(
        [not_found(), found_target()],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", confidence=0.12),
            candidate_detection("frame-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: calls.append(args) or {
        "ok": True,
        "permitted": True,
    }
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 0
    assert len(vision.promotions) == 1
    assert calls == []


def test_confirmed_candidate_after_one_turn_stops_search():
    first_attempt = [
        candidate_detection("a-1"),
        candidate_detection("a-2", bbox=(20, 20, 120, 120)),
        candidate_detection("a-3", bbox=(350, 20, 450, 120)),
    ]
    second_attempt = [
        candidate_detection("b-1"),
        candidate_detection("b-2", confidence=0.11),
        candidate_detection("b-3", confidence=0.10),
    ]
    vision = CandidateVisionAdapter(
        [not_found(), not_found(), found_target()],
        first_attempt + second_attempt,
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    calls = []

    def turn(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ok": True, "permitted": True}

    manager.execute_guarded_turn = turn
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1


def located_target(cx, target="backpack"):
    result = found_target(target)
    result["cx"] = float(cx)
    return result


def centered_candidate_payloads(prefix, cx=320):
    half_width = 120
    bbox = (cx - half_width, 160, cx + half_width, 460)
    return [
        candidate_detection(f"{prefix}-1", bbox=bbox, confidence=0.12),
        candidate_detection(f"{prefix}-2", bbox=bbox, confidence=0.11),
        candidate_detection(f"{prefix}-3", bbox=bbox, confidence=0.10),
    ]


def make_centering_manager(observations, candidate_payloads, turn_results=None):
    vision = CandidateVisionAdapter(observations, candidate_payloads)
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=vision,
    )
    manager.lidar_session = "session-1"
    calls = []
    results = list(turn_results or [])

    def turn(direction, speed, duration, *, expected_lidar_session, now=None):
        calls.append((direction, speed, duration, expected_lidar_session))
        if results:
            return dict(results.pop(0))
        return {"ok": True, "permitted": True, "reason": "completed"}

    manager.execute_guarded_turn = turn
    return manager, vision, calls


def test_already_centered_target_completes_without_turn():
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.execute_guarded_turn = lambda *args, **kwargs: pytest.fail(
        "centered target must not turn"
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 0
    assert result["centering_turn_chunks_attempted"] == 0
    assert result["steering_direction"] == "CENTERED"


def test_centering_publishes_live_tracking_state_updates():
    manager, _vision, _calls = make_centering_manager(
        [located_target(145), located_target(320)],
        centered_candidate_payloads("telemetry"),
    )
    tracking = empty_tracking_state()
    updates = []

    def publish(result):
        nonlocal tracking
        tracking = build_tracking_state(result, previous=tracking)
        updates.append(dict(tracking))

    manager.tracking_state_callback = publish
    result = manager.execute(_mission())

    centering = next(
        item for item in updates
        if item["state"] == "CENTERING"
    )
    assert centering["behavior"] == "FIND_OBJECT"
    assert centering["target_label"] == "backpack"
    assert centering["horizontal_error"] < -50
    assert centering["steering_direction"] == "LEFT"
    assert centering["target_area"] > 0
    assert result["state"] == "CENTERED"
    assert tracking["state"] == "CENTERED"
    assert tracking["steering_direction"] == "CENTER"
    assert abs(tracking["horizontal_error"]) <= 50
    assert tracking["locked_identity_id"] is None
    assert tracking["locked_entity_id"] is None


def test_runtime_tracking_callback_updates_status_state_without_identity():
    runtime = object.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime._behavior_execution_generation = 7
    runtime._control_generation = 7
    runtime.tracking_state = empty_tracking_state()

    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_label": "backpack",
        "target_confidence": 0.15,
        "target_center_x": 145.0,
        "target_center_y": 240.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "image_center_x": 320.0,
        "horizontal_error": -175.0,
        "target_area": 42000.0,
        "steering_direction": "LEFT",
    })
    assert runtime.tracking_state["behavior"] == "FIND_OBJECT"
    assert runtime.tracking_state["state"] == "CENTERING"
    assert runtime.tracking_state["target_label"] == "backpack"
    assert runtime.tracking_state["horizontal_error"] == -175.0
    assert runtime.tracking_state["target_area"] > 0
    assert runtime.tracking_state["steering_direction"] == "LEFT"
    assert runtime.tracking_state["locked_identity_id"] is None

    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERED",
        "target_label": "backpack",
        "target_center_x": 320.0,
        "image_width": 640.0,
        "horizontal_error": 0.0,
        "steering_direction": "CENTERED",
    })
    assert runtime.tracking_state["state"] == "CENTERED"
    assert runtime.tracking_state["steering_direction"] == "CENTER"

    runtime._control_generation = 8
    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_label": "backpack",
        "horizontal_error": -200.0,
        "steering_direction": "LEFT",
    })
    assert runtime.tracking_state["state"] == "CENTERED"


def test_left_target_uses_guarded_centering_constants():
    manager, _vision, calls = make_centering_manager(
        [located_target(145), located_target(320)],
        centered_candidate_payloads("left"),
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["centering_turn_chunks_completed"] == 1
    assert calls == [("LEFT", 0.20, 0.25, "session-1")]


def test_right_target_uses_right_guarded_centering_direction():
    manager, _vision, calls = make_centering_manager(
        [located_target(500), located_target(320)],
        centered_candidate_payloads("right"),
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert calls[0] == ("RIGHT", 0.20, 0.25, "session-1")


def test_centering_requires_new_confirmation_after_each_turn():
    manager, vision, calls = make_centering_manager(
        [located_target(145), located_target(500), located_target(320)],
        centered_candidate_payloads("first", 500)
        + centered_candidate_payloads("second", 320),
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["centering_turn_chunks_attempted"] == 2
    assert result["centering_turn_chunks_completed"] == 2
    assert len(calls) == 2
    assert vision.candidate_calls == 6


def test_target_loss_after_centering_turn_stops_without_second_turn():
    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [],
    )
    result = manager.execute(_mission())
    assert result["state"] == "TARGET_LOST_DURING_CENTERING"
    assert result["completed"] is False
    assert result["target_found"] is False
    assert result["centering_turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_centering_denial_stops_without_retry():
    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [],
        [{"ok": False, "permitted": False, "reason": "turn_side_not_clear"}],
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERING_BLOCKED"
    assert result["completed"] is False
    assert result["last_guarded_turn_result"]["reason"] == "turn_side_not_clear"
    assert len(calls) == 1


def test_centering_exhausts_at_eight_chunks():
    observations = [located_target(145)] + [located_target(145)] * 8
    payloads = []
    for index in range(8):
        payloads.extend(centered_candidate_payloads(f"exhaust-{index}", 145))
    manager, _vision, calls = make_centering_manager(observations, payloads)
    result = manager.execute(_mission())
    assert result["state"] == "CENTERING_EXHAUSTED"
    assert result["completed"] is False
    assert result["target_found"] is True
    assert result["centering_turn_chunks_attempted"] == 8
    assert result["centering_turn_chunks_completed"] == 8
    assert len(calls) == 8


def test_search_and_centering_counters_remain_separate():
    failed_search = [
        candidate_detection("search-1", bbox=(10, 10, 100, 100)),
        candidate_detection("search-2", bbox=(300, 10, 390, 100)),
        candidate_detection("search-3", bbox=(500, 10, 590, 100)),
    ]
    acquired = centered_candidate_payloads("acquired", 145)
    centered = centered_candidate_payloads("centered", 320)
    manager, _vision, calls = make_centering_manager(
        [not_found(), located_target(145), located_target(320)],
        failed_search + acquired + centered,
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERED"
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["centering_turn_chunks_completed"] == 1
    assert len(calls) == 2


def test_invalid_center_geometry_does_not_start_centering():
    invalid = located_target(float("nan"))
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=SequencedVisionAdapter([invalid]),
    )
    result = manager.execute(_mission())
    assert result["centering_turn_chunks_attempted"] == 0


def detection(cx, area):
    return {
        "found": True,
        "target": "backpack",
        "label": "backpack",
        "confidence": 0.90,
        "cx": float(cx),
        "cy": 240.0,
        "area": float(area),
        "image_width": 640.0,
        "image_height": 480.0,
    }


def main():
    mission = create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Find my backpack",
        status="ACTIVE",
    )

    robot = FakeRobotBridgeClient()

    vision = SequencedVisionAdapter(
        [
            detection(
                cx=180,
                area=35000,
            ),
            detection(
                cx=460,
                area=40000,
            ),
            detection(
                cx=320,
                area=50000,
            ),
            detection(
                cx=320,
                area=100000,
            ),
        ]
    )

    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=vision,
    )

    print(
        "===== CYCLE 1: IMAGE-LEFT TARGET ====="
    )

    left_result = manager.execute(
        mission
    )

    print(left_result)

    assert left_result["ok"] is True
    assert left_result["executed"] is True
    assert left_result["completed"] is False

    assert (
        left_result["state"]
        == "CENTERING_LEFT"
    )

    assert robot.calls[-1][0] == (
        "turn_left"
    )

    print()
    print(
        "===== CYCLE 2: IMAGE-RIGHT TARGET ====="
    )

    right_result = manager.execute(
        mission
    )

    print(right_result)

    assert right_result["ok"] is True
    assert right_result["executed"] is True
    assert right_result["completed"] is False

    assert (
        right_result["state"]
        == "CENTERING_RIGHT"
    )

    assert robot.calls[-1][0] == (
        "turn_right"
    )

    print()
    print(
        "===== CYCLE 3: CENTERED TARGET ====="
    )

    approach_result = manager.execute(
        mission
    )

    print(approach_result)

    assert approach_result["ok"] is True
    assert approach_result["executed"] is True

    assert (
        approach_result["completed"]
        is False
    )

    assert (
        approach_result["state"]
        == "APPROACHING"
    )

    assert robot.calls[-1][0] == (
        "move_forward"
    )

    print()
    print(
        "===== CYCLE 4: ARRIVAL TARGET ====="
    )

    arrived_result = manager.execute(
        mission
    )

    print(arrived_result)

    assert arrived_result["ok"] is True
    assert arrived_result["executed"] is True

    assert (
        arrived_result["completed"]
        is True
    )

    assert (
        arrived_result["state"]
        == "ARRIVED"
    )

    assert robot.calls[-1][0] == (
        "stop"
    )

    assert [
        call[0]
        for call in robot.calls
    ] == [
        "turn_left",
        "turn_right",
        "move_forward",
        "stop",
    ]

    assert vision.calls == 4
    assert vision.results == []

    print()
    print(
        "PASS: image-left target causes "
        "one left correction"
    )

    print(
        "PASS: image-right target causes "
        "one right correction"
    )

    print(
        "PASS: centered target causes "
        "one approach step"
    )

    print(
        "PASS: close target causes ARRIVED"
    )

    print(
        "PASS: intermediate steps keep "
        "the mission active"
    )

    print(
        "PASS: only ARRIVED completes "
        "the mission"
    )

    print()
    print(
        "All single-cycle FIND_OBJECT "
        "steering tests passed."
    )

    print(
        "No commands were sent to "
        "the physical robot."
    )


if __name__ == "__main__":
    main()
