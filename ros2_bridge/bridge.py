from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class RobotBridgeState:
    timestamp: str
    bridge_status: str
    ros2_available: bool
    motion_topic: str
    last_command: Optional[Dict[str, Any]] = None
    note: str = ""


class ROS2Bridge:
    """
    ROS2 bridge boundary.

    This class intentionally does not require ROS2 at import time.
    It preserves separation between the cognitive platform and ROS2.
    """

    def __init__(self, motion_topic="/cmd_vel"):
        self.motion_topic = motion_topic
        self.last_command = None
        self.ros2_available = self._check_ros2_available()
        self.bridge_status = "READY" if self.ros2_available else "ROS2_NOT_AVAILABLE"

    def _check_ros2_available(self):
        try:
            import rclpy  # noqa: F401
            return True
        except Exception:
            return False

    def status(self):
        return RobotBridgeState(
            timestamp=now_iso(),
            bridge_status=self.bridge_status,
            ros2_available=self.ros2_available,
            motion_topic=self.motion_topic,
            last_command=self.last_command,
            note="ROS2 bridge boundary initialized",
        )

    def motion_request(self, linear_x=0.0, angular_z=0.0, duration_sec=0.5, source="behavior_manager"):
        command = {
            "timestamp": now_iso(),
            "source": source,
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "duration_sec": float(duration_sec),
            "motion_topic": self.motion_topic,
        }

        self.last_command = command

        if not self.ros2_available:
            self.bridge_status = "SIMULATED_COMMAND_ONLY"
            return {
                "ok": False,
                "executed": False,
                "reason": "ROS2 is not available in this Python environment",
                "command": command,
            }

        return self._publish_ros2_motion(command)

    def _publish_ros2_motion(self, command):
        try:
            import time
            import rclpy
            from geometry_msgs.msg import Twist

            rclpy.init(args=None)
            node = rclpy.create_node("cognitive_ros2_bridge")

            publisher = node.create_publisher(Twist, self.motion_topic, 10)

            msg = Twist()
            msg.linear.x = command["linear_x"]
            msg.angular.z = command["angular_z"]

            stop = Twist()

            end_time = time.time() + command["duration_sec"]

            while time.time() < end_time:
                publisher.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.05)

            publisher.publish(stop)

            node.destroy_node()
            rclpy.shutdown()

            self.bridge_status = "COMMAND_EXECUTED"

            return {
                "ok": True,
                "executed": True,
                "reason": "Motion command published to ROS2",
                "command": command,
            }

        except Exception as exc:
            self.bridge_status = "ROS2_COMMAND_FAILED"
            return {
                "ok": False,
                "executed": False,
                "reason": str(exc),
                "command": command,
            }


def bridge_status_dict():
    bridge = ROS2Bridge()
    return asdict(bridge.status())
