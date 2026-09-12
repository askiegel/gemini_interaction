#!/usr/bin/env python3

import json
import math

import pytest

import behavior_manager as behavior_module
from behavior_manager import BehaviorManager
from mission_types import create_mission


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
    assert result["state"] == "ACQUIRED"
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
    assert result["state"] == "ACQUIRED"
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1


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
