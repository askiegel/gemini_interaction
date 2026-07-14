import json
import time

from behavior_manager import BehaviorManager
from intent_parser import validate_intent
from mission_manager import MissionManager
from robot_bridge.client import RobotBridgeClient


COMMANDS = [
    {
        "intent": "MOVE_FORWARD",
        "speech": "Move forward",
    },
    {
        "intent": "TURN_LEFT",
        "speech": "Turn left",
    },
    {
        "intent": "TURN_RIGHT",
        "speech": "Turn right",
    },
]


def execute_command(command, behavior_manager):
    parsed = validate_intent(
        {
            "intent": command["intent"],
            "speech": command["speech"],
            "target": None,
        }
    )

    # Use a fresh manager for each short motion so every command becomes active.
    mission_manager = MissionManager()
    mission = mission_manager.handle_intent(parsed)

    print()
    print("=" * 60)
    print(f"COMMAND: {command['speech']}")
    print("=" * 60)
    print("Mission:")
    print(json.dumps(mission.to_dict(), indent=2))

    result = behavior_manager.execute(mission)

    print("Execution result:")
    print(json.dumps(result, indent=2))

    if not result.get("ok"):
        raise RuntimeError(
            f"{command['intent']} failed: "
            f"{result.get('robot_result', result)}"
        )

    return result


def main():
    robot = RobotBridgeClient()
    behavior_manager = BehaviorManager(robot_client=robot)

    print("=== Robot Bridge Status ===")
    status = robot.status()
    print(json.dumps(status, indent=2))

    if not status.get("ok"):
        raise SystemExit(
            "Robot Bridge is not ready. No motion commands were sent."
        )

    print()
    print("Starting controlled live motion test.")

    for index, command in enumerate(COMMANDS, start=1):
        print()
        print(f"Test {index} of {len(COMMANDS)}")

        execute_command(command, behavior_manager)

        # Send an additional explicit stop after the bridge's automatic stop.
        stop_result = robot.stop()
        print("Explicit safety stop:")
        print(json.dumps(stop_result, indent=2))

        if not stop_result.get("ok"):
            raise RuntimeError("Explicit stop command failed.")

        if index < len(COMMANDS):
            print("Pausing before the next command...")
            time.sleep(2.0)

    final_stop = robot.stop()

    print()
    print("=== Final Stop ===")
    print(json.dumps(final_stop, indent=2))

    if not final_stop.get("ok"):
        raise RuntimeError("Final stop command failed.")

    print()
    print("All live motion-intent tests passed.")


if __name__ == "__main__":
    main()
