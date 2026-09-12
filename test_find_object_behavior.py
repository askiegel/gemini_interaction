#!/usr/bin/env python3

import json

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
