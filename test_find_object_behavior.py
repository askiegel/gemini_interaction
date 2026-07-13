#!/usr/bin/env python3

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
