import json
from dataclasses import asdict
from ros2_bridge.bridge import ROS2Bridge


def main():
    bridge = ROS2Bridge()
    print(json.dumps(asdict(bridge.status()), indent=2))


if __name__ == "__main__":
    main()
