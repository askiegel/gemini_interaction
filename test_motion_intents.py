import json

from behavior_manager import BehaviorManager
from intent_parser import validate_intent
from mission_manager import MissionManager


class FakeRobotBridgeClient:
    def __init__(self):
        self.calls = []

    def move_forward(self, speed=0.10, seconds=1.0):
        self.calls.append(
            {
                "method": "move_forward",
                "speed": speed,
                "seconds": seconds,
            }
        )
        return {
            "ok": True,
            "action": "motion",
            "linear_x": speed,
            "angular_z": 0.0,
            "duration": seconds,
            "automatic_stop": True,
        }

    def turn_left(self, speed=0.5, seconds=1.0):
        self.calls.append(
            {
                "method": "turn_left",
                "speed": speed,
                "seconds": seconds,
            }
        )
        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": abs(speed),
            "duration": seconds,
            "automatic_stop": True,
        }

    def turn_right(self, speed=0.5, seconds=1.0):
        self.calls.append(
            {
                "method": "turn_right",
                "speed": speed,
                "seconds": seconds,
            }
        )
        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": -abs(speed),
            "duration": seconds,
            "automatic_stop": True,
        }

    def stop(self):
        self.calls.append({"method": "stop"})
        return {
            "ok": True,
            "action": "stop",
        }


EXPECTED_METHODS = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
}


def test_intent(intent_name):
    parsed = validate_intent(
        {
            "intent": intent_name,
            "speech": intent_name.replace("_", " ").title(),
            "target": None,
        }
    )

    assert parsed["intent"] == intent_name
    assert parsed["target"] is None

    mission_manager = MissionManager()
    mission = mission_manager.handle_intent(parsed)

    assert mission.mission_type == intent_name
    assert mission.status == "ACTIVE"
    assert mission.target is None
    assert mission_manager.get_active_mission() is mission

    fake_robot = FakeRobotBridgeClient()
    behavior_manager = BehaviorManager(robot_client=fake_robot)

    simulation = behavior_manager.simulate(mission)
    result = behavior_manager.execute(mission)

    assert result["ok"] is True
    assert result["executed"] is True
    assert result["behavior"] == intent_name
    assert len(fake_robot.calls) == 1
    assert fake_robot.calls[0]["method"] == EXPECTED_METHODS[intent_name]
    assert result["robot_result"]["automatic_stop"] is True

    print()
    print(f"=== {intent_name} ===")
    print("Parsed intent:")
    print(json.dumps(parsed, indent=2))
    print("Mission:")
    print(json.dumps(mission.to_dict(), indent=2))
    print("Simulation:")
    print(simulation)
    print("Execution:")
    print(json.dumps(result, indent=2))


def main():
    for intent_name in EXPECTED_METHODS:
        test_intent(intent_name)

    print()
    print("All offline motion-intent tests passed.")
    print("No commands were sent to the physical robot.")


if __name__ == "__main__":
    main()
