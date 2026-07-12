#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeRobotBridgeClient:
    def __init__(self):
        self.calls = []

    def move_forward(self, speed=0.10, seconds=1.0):
        self.calls.append(("move_forward", speed, seconds))
        return {"ok": True, "automatic_stop": True}

    def turn_left(self, speed=0.5, seconds=1.0):
        self.calls.append(("turn_left", speed, seconds))
        return {"ok": True, "automatic_stop": True}

    def turn_right(self, speed=0.5, seconds=1.0):
        self.calls.append(("turn_right", speed, seconds))
        return {"ok": True, "automatic_stop": True}

    def stop(self):
        self.calls.append(("stop",))
        return {"ok": True, "action": "stop"}


class SequencedVisionAdapter:
    def __init__(self, results):
        self.results = list(results)

    def find_target(self, target):
        if not self.results:
            raise AssertionError("Vision sequence exhausted.")

        return dict(self.results.pop(0))


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
            detection(cx=180, area=35000),
            detection(cx=460, area=40000),
            detection(cx=320, area=50000),
            detection(cx=320, area=100000),
        ]
    )

    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=vision,
    )

    manager.FIND_CYCLE_PAUSE = 0.0
    manager.MAX_FIND_CYCLES = 10

    result = manager.execute(mission)

    assert result["ok"] is True
    assert result["completed"] is True
    assert result["state"] == "ARRIVED"

    assert [call[0] for call in robot.calls] == [
        "turn_left",
        "turn_right",
        "move_forward",
        "stop",
    ]

    assert [
        item["state"]
        for item in result["cycle_history"]
    ] == [
        "CENTERING_LEFT",
        "CENTERING_RIGHT",
        "APPROACHING",
        "ARRIVED",
    ]

    print("PASS: image-left target causes left correction")
    print("PASS: image-right target causes right correction")
    print("PASS: centered target causes approach")
    print("PASS: close target causes ARRIVED")
    print()
    print("All FIND_OBJECT steering tests passed.")
    print("No commands were sent to the physical robot.")


if __name__ == "__main__":
    main()
