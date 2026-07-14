import json
from robot_bridge.client import RobotBridgeClient


def main():
    client = RobotBridgeClient()

    print("=== Robot Bridge Status ===")
    print(json.dumps(client.status(), indent=2))

    print("\n=== Forward Motion Test ===")
    print(json.dumps(client.move_forward(speed=0.10, seconds=1.0), indent=2))

    print("\n=== Stop ===")
    print(json.dumps(client.stop(), indent=2))


if __name__ == "__main__":
    main()
