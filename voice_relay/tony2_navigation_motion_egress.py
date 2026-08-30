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
import os
import sys
import time
from pathlib import Path


# The supervisor executes this file directly rather than
# with ``python -m``. In direct-script mode Python places
# voice_relay/ on sys.path, not the Cognitive repository
# root. Add the repository root before importing sibling
# packages such as voice_relay and robot_bridge.
if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(
            Path(__file__)
            .resolve()
            .parents[1]
        ),
    )


import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from voice_relay.tony2_navigation_motion_arm import (
    atomic_write_json,
    read_json_file,
    validate_arm_payload,
)



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

    def arm(
        self,
        robot_client,
    ):
        if self.enabled:
            return {
                "ok": True,
                "action": "ALREADY_ARMED",
            }

        self.robot = robot_client
        self.enabled = True
        self._active = False
        self._last_input_at = None
        self._last_stop_reason = None

        return {
            "ok": True,
            "action": "ARMED",
        }

    def disarm(
        self,
        reason="MOTION_DISARMED",
    ):
        if not self.enabled:
            self._active = False
            self._last_input_at = None
            self._last_stop_reason = str(reason)

            return {
                "ok": True,
                "forwarded": False,
                "stopped": False,
                "reason": str(reason),
            }

        result = self._safe_stop(
            reason
        )

        self.enabled = False

        return result

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

        return self.disarm(
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
        *,
        arm_file,
        status_file,
    ):
        super().__init__(
            "tony2_navigation_motion_egress"
        )

        self.controller = controller

        self.arm_file = Path(
            arm_file
        )

        self.status_file = Path(
            status_file
        )

        self._armed_token = None
        self._last_status = None

        self._write_status(
            running=True,
            armed=False,
            reason="WAITING_FOR_ARM_LEASE",
        )

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

    def _write_status(
        self,
        *,
        running,
        armed,
        reason,
        token=None,
        error=None,
    ):
        payload = {
            "running": bool(running),
            "armed": bool(armed),
            "reason": str(reason),
            "token": token,
            "timestamp": time.time(),
        }

        if error is not None:
            payload["error"] = str(error)

        atomic_write_json(
            self.status_file,
            payload,
        )

        self._last_status = payload

    def _disarm(
        self,
        reason,
    ):
        result = self.controller.disarm(
            reason
        )

        self._armed_token = None

        self._write_status(
            running=True,
            armed=False,
            reason=reason,
        )

        return result

    def _sync_arm_state(self):
        raw = read_json_file(
            self.arm_file
        )

        lease = validate_arm_payload(
            raw
        )

        if lease is None:
            if self.controller.enabled:
                self._disarm(
                    "ARM_LEASE_MISSING_OR_EXPIRED"
                )

            return

        token = lease["token"]

        if (
            self.controller.enabled
            and token == self._armed_token
        ):
            return

        if self.controller.enabled:
            self._disarm(
                "ARM_TOKEN_REPLACED"
            )

        try:
            from robot_bridge.client import (
                RobotBridgeClient,
            )

            robot = RobotBridgeClient(
                base_url=lease[
                    "robot_bridge_url"
                ],
                timeout=CLIENT_TIMEOUT_SECONDS,
            )

            result = self.controller.arm(
                robot
            )

            if result.get("ok") is not True:
                raise RuntimeError(
                    str(result)
                )

            self._armed_token = token

            self._write_status(
                running=True,
                armed=True,
                reason="ARM_LEASE_VALID",
                token=token,
            )

        except Exception as exc:
            self.controller.enabled = False
            self._armed_token = None

            self._write_status(
                running=True,
                armed=False,
                reason="ARM_FAILED",
                error=exc,
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
        self._sync_arm_state()

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
        "--arm-file",
        default=os.environ.get(
            "TONY2_NAVIGATION_MOTION_ARM",
            "/tmp/tony2_navigation_motion_arm.json",
        ),
    )

    parser.add_argument(
        "--status-file",
        default=os.environ.get(
            "TONY2_NAVIGATION_EGRESS_STATUS",
            (
                "/tmp/"
                "tony2_navigation_motion_egress_status.json"
            ),
        ),
    )

    args = parser.parse_args()

    # Always starts disabled. A currently valid transient
    # lease is the only mechanism that can arm this process.
    controller = MotionEgressController(
        robot_client=object(),
        enabled=False,
        watchdog_timeout=ROBOT_WATCHDOG_SECONDS,
    )

    rclpy.init()

    node = MotionEgressNode(
        controller,
        arm_file=args.arm_file,
        status_file=args.status_file,
    )

    try:
        rclpy.spin(node)
    finally:
        controller.shutdown()

        # Runtime status is valid only while this process
        # owns the egress. Remove it on graceful shutdown.
        # Tony2NavigationRuntime.stop() also unlinks this
        # file, covering supervisor-side cleanup.
        try:
            node.status_file.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
