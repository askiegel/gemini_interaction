import json
from ros2_bridge.bridge import ROS2Bridge


def main():
    bridge = ROS2Bridge()

    print("ROS2 Bridge Status:")
    print(json.dumps(bridge.status().__dict__, indent=2))

    print("\nSending small forward motion request:")
    result = bridge.motion_request(
        linear_x=0.05,
        angular_z=0.0,
        duration_sec=0.5,
        source="ros2_bridge_motion_test",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
