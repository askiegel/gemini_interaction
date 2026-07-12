#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeRobotBridgeClient:
    def __init__(self):
        self.calls = []

    def move_forward(self, speed=0.10, seconds=1.0):
        self.calls.append(("move_forward", speed, seconds))
        return {
            "ok": True,
            "action": "motion",
            "linear_x": speed,
            "angular_z": 0.0,
            "duration": seconds,
            "automatic_stop": True,
        }

    def turn_left(self, speed=0.5, seconds=1.0):
        self.calls.append(("turn_left", speed, seconds))
        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": abs(speed),
            "duration": seconds,
            "automatic_stop": True,
        }

    def turn_right(self, speed=0.5, seconds=1.0):
        self.calls.append(("turn_right", speed, seconds))
        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": -abs(speed),
            "duration": seconds,
            "automatic_stop": True,
        }

    def stop(self):
        self.calls.append(("stop",))
        return {
            "ok": True,
            "action": "stop",
        }


def make_mission(mission_type, target=None, status="ACTIVE"):
    return create_mission(
        mission_type=mission_type,
        target=target,
        speech=mission_type,
        status=status,
    )



class LegacyFakeVisionAdapter:
    """
    Compatibility perception source for the legacy BehaviorManager test.

    Production runtime behavior uses the shared World Model. This adapter
    verifies that isolated legacy callers can still use find_target().
    """

    def find_target(self, target_name):
        return {
            "found": True,
            "target": target_name,
            "confidence": 0.90,
            "cx": 320.0,
            "cy": 240.0,
            "area": 76000.0,
            "image_width": 640.0,
            "image_height": 480.0,
        }

def main():
    robot = FakeRobotBridgeClient()
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=LegacyFakeVisionAdapter(),
    )

    result = manager.execute(make_mission("MOVE_FORWARD"))
    assert result["ok"] is True
    assert result["behavior"] == "MOVE_FORWARD"
    assert robot.calls[-1][0] == "move_forward"

    result = manager.execute(make_mission("TURN_LEFT"))
    assert result["ok"] is True
    assert result["behavior"] == "TURN_LEFT"
    assert robot.calls[-1][0] == "turn_left"

    result = manager.execute(make_mission("TURN_RIGHT"))
    assert result["ok"] is True
    assert result["behavior"] == "TURN_RIGHT"
    assert robot.calls[-1][0] == "turn_right"

    result = manager.execute(
        make_mission("FOLLOW_PERSON", target="person")
    )
    assert result["ok"] is True
    assert result["behavior"] == "FOLLOW_PERSON"
    assert robot.calls[-1][0] == "move_forward"

    result = manager.execute(make_mission("STOP"))
    assert result["ok"] is True
    assert result["behavior"] == "STOP"
    assert robot.calls[-1][0] == "stop"

    result = manager.execute(
        make_mission("FIND_OBJECT", target="backpack")
    )
    assert result["ok"] is True
    assert result["executed"] is True
    assert result["behavior"] == "FIND_OBJECT"
    assert result["target"] == "backpack"
    assert result["state"] == "ARRIVED"
    assert result["completed"] is True
    assert robot.calls[-1][0] == "stop"

    result = manager.execute(make_mission("RETURN_HOME"))
    assert result["ok"] is True
    assert result["executed"] is False

    result = manager.execute(make_mission("DESCRIBE_SCENE"))
    assert result["ok"] is True
    assert result["executed"] is False

    result = manager.execute(
        make_mission("UNKNOWN", status="REJECTED")
    )
    assert result["ok"] is False
    assert result["executed"] is False

    print("All BehaviorManager refactor tests passed.")
    print("No commands were sent to the physical robot.")


if __name__ == "__main__":
    main()
