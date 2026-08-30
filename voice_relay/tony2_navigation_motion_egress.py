#!/usr/bin/env python3

"""
Guarded Nav2 velocity egress for Tony2.

This adapter consumes planar Twist commands on isolated
Zenoh domain 43 and, only when explicitly enabled, forwards
them through the existing Mayday Robot Bridge streaming
HTTP interface.

It never publishes ROS /cmd_vel directly.

Default state is DISABLED.
"""

import argparse
import math
import time

import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from robot_bridge.client import RobotBridgeClient


INPUT_TOPIC = "/tony2_nav_cmd_vel_egress"

MAX_LINEAR_X = 0.20
MAX_ANGULAR_Z = 1.00

ROBOT_WATCHDOG_SECONDS = 0.50
INPUT_STALE_SECONDS = 0.30

CLIENT_TIMEOUT_SECONDS = 0.25

TIMER_PERIOD_SECONDS = 0.05

ZERO_EPSILON = 1.0e-9


class MotionEgressController:

    def __init__(
        self,
        robot_client,
        enabled=False,
        watchdog_timeout=ROBOT_WATCHDOG_SECONDS,
        stale_timeout=INPUT_STALE_SECONDS,
    ):
        self.robot = robot_client
        self.enabled = bool(enabled)

        self.watchdog_timeout = float(
            watchdog_timeout
        )

        self.stale_timeout = float(
            stale_timeout
        )

        if not (
            0.20
            <= self.watchdog_timeout
            <= 2.00
        ):
            raise ValueError(
                "watchdog_timeout outside "
                "Robot Bridge contract."
            )

        if not (
            0.0
            < self.stale_timeout
            < self.watchdog_timeout
        ):
            raise ValueError(
                "stale_timeout must be positive "
                "and shorter than watchdog_timeout."
            )

        self._active = False
        self._last_input_at = None
        self._last_stop_reason = None

    @property
    def active(self):
        return bool(self._active)

    @property
    def last_stop_reason(self):
        return self._last_stop_reason

    def _safe_stop(
        self,
        reason,
    ):
        self._active = False
        self._last_input_at = None
        self._last_stop_reason = str(reason)

        try:
            result = self.robot.stop()
        except Exception as exc:
            return {
                "ok": False,
                "forwarded": False,
                "stopped": False,
                "reason": str(reason),
                "stop_error": str(exc),
            }

        return {
            "ok": bool(
                isinstance(result, dict)
                and result.get("ok")
            ),
            "forwarded": False,
            "stopped": bool(
                isinstance(result, dict)
                and result.get("ok")
            ),
            "reason": str(reason),
            "stop_result": result,
        }

    def accept(
        self,
        linear_x,
        angular_z,
        *,
        unsupported_axis=False,
        now=None,
    ):
        now = (
            time.monotonic()
            if now is None
            else float(now)
        )

        try:
            linear_x = float(linear_x)
            angular_z = float(angular_z)
        except (TypeError, ValueError):
            if not self.enabled:
                return {
                    "ok": False,
                    "forwarded": False,
                    "reason": "MOTION_OUTPUT_DISABLED",
                }

            return self._safe_stop(
                "NON_NUMERIC_COMMAND"
            )

        if not self.enabled:
            return {
                "ok": False,
                "forwarded": False,
                "reason": "MOTION_OUTPUT_DISABLED",
                "linear_x": linear_x,
                "angular_z": angular_z,
            }

        if unsupported_axis:
            return self._safe_stop(
                "UNSUPPORTED_TWIST_AXIS"
            )

        if (
            not math.isfinite(linear_x)
            or not math.isfinite(angular_z)
        ):
            return self._safe_stop(
                "NONFINITE_COMMAND"
            )

        if abs(linear_x) > MAX_LINEAR_X:
            return self._safe_stop(
                "LINEAR_LIMIT_EXCEEDED"
            )

        if abs(angular_z) > MAX_ANGULAR_Z:
            return self._safe_stop(
                "ANGULAR_LIMIT_EXCEEDED"
            )

        try:
            result = self.robot.streaming_motion(
                linear_x=linear_x,
                angular_z=angular_z,
                watchdog_timeout=self.watchdog_timeout,
            )
        except Exception as exc:
            stopped = self._safe_stop(
                "STREAM_REQUEST_EXCEPTION"
            )

            stopped[
                "stream_error"
            ] = str(exc)

            return stopped

        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
        ):
            stopped = self._safe_stop(
                "STREAM_REQUEST_FAILED"
            )

            stopped[
                "stream_result"
            ] = result

            return stopped

        self._active = True
        self._last_input_at = now
        self._last_stop_reason = None

        return {
            "ok": True,
            "forwarded": True,
            "linear_x": linear_x,
            "angular_z": angular_z,
            "watchdog_timeout":
                self.watchdog_timeout,
            "stream_result": result,
        }

    def tick(
        self,
        now=None,
    ):
        if not self.enabled:
            return {
                "ok": True,
                "action": "DISABLED",
            }

        if (
            not self._active
            or self._last_input_at is None
        ):
            return {
                "ok": True,
                "action": "IDLE",
            }

        now = (
            time.monotonic()
            if now is None
            else float(now)
        )

        age = (
            now
            - self._last_input_at
        )

        if age < self.stale_timeout:
            return {
                "ok": True,
                "action": "FRESH",
                "age_seconds": age,
            }

        result = self._safe_stop(
            "NAV2_COMMAND_STALE"
        )

        result["age_seconds"] = age

        return result

    def shutdown(self):
        if not self.enabled:
            return {
                "ok": True,
                "action": "DISABLED",
            }

        return self._safe_stop(
            "EGRESS_SHUTDOWN"
        )


def reliable_qos(
    depth=10,
):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class MotionEgressNode(Node):

    def __init__(
        self,
        controller,
    ):
        super().__init__(
            "tony2_navigation_motion_egress"
        )

        self.controller = controller

        self.subscription = (
            self.create_subscription(
                Twist,
                INPUT_TOPIC,
                self._on_twist,
                reliable_qos(10),
            )
        )

        self.timer = self.create_timer(
            TIMER_PERIOD_SECONDS,
            self._on_timer,
        )

    def _on_twist(
        self,
        message,
    ):
        unsupported_axis = any(
            abs(float(value))
            > ZERO_EPSILON
            for value in (
                message.linear.y,
                message.linear.z,
                message.angular.x,
                message.angular.y,
            )
        )

        result = self.controller.accept(
            linear_x=message.linear.x,
            angular_z=message.angular.z,
            unsupported_axis=unsupported_axis,
        )

        if (
            self.controller.enabled
            and result.get("ok") is not True
        ):
            self.get_logger().error(
                "Guarded motion egress rejected "
                f"command: {result}"
            )

    def _on_timer(self):
        result = self.controller.tick()

        if (
            self.controller.enabled
            and result.get("reason")
            == "NAV2_COMMAND_STALE"
        ):
            self.get_logger().warning(
                "Nav2 velocity input became stale; "
                "Robot Bridge STOP requested."
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help=(
            "Permit Robot Bridge streaming motion. "
            "Default is disabled."
        ),
    )

    parser.add_argument(
        "--robot-bridge-url",
        default=None,
    )

    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=ROBOT_WATCHDOG_SECONDS,
    )

    args = parser.parse_args()

    robot = RobotBridgeClient(
        base_url=args.robot_bridge_url,
        timeout=CLIENT_TIMEOUT_SECONDS,
    )

    controller = MotionEgressController(
        robot_client=robot,
        enabled=args.enable_motion,
        watchdog_timeout=args.watchdog_timeout,
    )

    rclpy.init()

    node = MotionEgressNode(
        controller
    )

    try:
        rclpy.spin(node)
    finally:
        controller.shutdown()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
