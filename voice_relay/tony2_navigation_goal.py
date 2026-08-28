#!/usr/bin/env python3

"""Execute one bounded NavigateToPose goal on Tony2."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformListener


_CANCEL_REQUESTED = False


def _request_cancel(_signum, _frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def _write_result(path, payload):
    path = Path(path)
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


class GuardedGoalRunner(Node):
    def __init__(self):
        super().__init__(
            "tony2_guarded_navigation_goal"
        )

        self.tf_buffer = Buffer(
            cache_time=Duration(
                seconds=10.0
            ),
            node=self,
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

    def spin_until(
        self,
        predicate,
        timeout_seconds,
    ):
        deadline = (
            time.monotonic()
            + timeout_seconds
        )

        while (
            time.monotonic() < deadline
            and not _CANCEL_REQUESTED
        ):
            if predicate():
                return True

            rclpy.spin_once(
                self,
                timeout_sec=0.05,
            )

        return bool(predicate())

    def current_map_pose(self):
        ready = self.spin_until(
            lambda: self.tf_buffer.can_transform(
                "map",
                "base_link",
                Time(),
                timeout=Duration(
                    seconds=0.0
                ),
            ),
            5.0,
        )

        if not ready:
            raise RuntimeError(
                "Current map-to-base_link transform "
                "is unavailable."
            )

        transform = (
            self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time(),
            )
        )

        return (
            float(
                transform.transform.translation.x
            ),
            float(
                transform.transform.translation.y
            ),
        )

    def wait_for_action_server(self):
        return self.spin_until(
            lambda: self.action_client.wait_for_server(
                timeout_sec=0.0
            ),
            5.0,
        )

    def cancel_goal(self, goal_handle):
        cancel_future = (
            goal_handle.cancel_goal_async()
        )

        self.spin_until(
            cancel_future.done,
            2.0,
        )

    def execute(
        self,
        *,
        x,
        y,
        yaw,
        max_distance,
        timeout_seconds,
    ):
        current_x, current_y = (
            self.current_map_pose()
        )

        distance = math.hypot(
            x - current_x,
            y - current_y,
        )

        if distance > max_distance:
            raise RuntimeError(
                "Goal distance "
                f"{distance:.3f} m exceeds guarded "
                f"maximum {max_distance:.3f} m."
            )

        if _CANCEL_REQUESTED:
            raise RuntimeError(
                "Goal cancelled before submission."
            )

        if not self.wait_for_action_server():
            raise RuntimeError(
                "NavigateToPose action server "
                "is unavailable."
            )

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        half_yaw = yaw / 2.0

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = (
            math.sin(half_yaw)
        )
        goal.pose.pose.orientation.w = (
            math.cos(half_yaw)
        )

        send_future = (
            self.action_client.send_goal_async(
                goal
            )
        )

        if not self.spin_until(
            send_future.done,
            5.0,
        ):
            raise RuntimeError(
                "Timed out waiting for goal "
                "acknowledgement."
            )

        goal_handle = send_future.result()

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):
            raise RuntimeError(
                "NavigateToPose goal was rejected."
            )

        result_future = (
            goal_handle.get_result_async()
        )

        deadline = (
            time.monotonic()
            + timeout_seconds
        )

        while not result_future.done():
            if (
                _CANCEL_REQUESTED
                or time.monotonic() >= deadline
            ):
                self.cancel_goal(
                    goal_handle
                )

                reason = (
                    "cancelled"
                    if _CANCEL_REQUESTED
                    else "timed out"
                )

                raise RuntimeError(
                    "NavigateToPose goal "
                    f"{reason}."
                )

            rclpy.spin_once(
                self,
                timeout_sec=0.05,
            )

        wrapped_result = (
            result_future.result()
        )

        status = int(
            wrapped_result.status
        )

        if (
            status
            != GoalStatus.STATUS_SUCCEEDED
        ):
            raise RuntimeError(
                "NavigateToPose ended with "
                f"status {status}."
            )

        return {
            "ok": True,
            "frame": "map",
            "requested": {
                "x": x,
                "y": y,
                "yaw": yaw,
            },
            "start": {
                "x": current_x,
                "y": current_y,
            },
            "distance_meters": distance,
            "maximum_distance_meters":
                max_distance,
            "timeout_seconds":
                timeout_seconds,
            "status":
                GoalStatus.STATUS_SUCCEEDED,
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--x",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--y",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--yaw",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--result-file",
        required=True,
    )

    args = parser.parse_args()

    for value in (
        args.x,
        args.y,
        args.yaw,
        args.max_distance,
        args.timeout,
    ):
        if not math.isfinite(value):
            raise SystemExit(
                "All numeric arguments must "
                "be finite."
            )

    signal.signal(
        signal.SIGTERM,
        _request_cancel,
    )

    signal.signal(
        signal.SIGINT,
        _request_cancel,
    )

    rclpy.init()

    node = GuardedGoalRunner()

    try:
        payload = node.execute(
            x=args.x,
            y=args.y,
            yaw=args.yaw,
            max_distance=(
                args.max_distance
            ),
            timeout_seconds=(
                args.timeout
            ),
        )

        _write_result(
            args.result_file,
            payload,
        )

        return 0

    except Exception as exc:
        _write_result(
            args.result_file,
            {
                "ok": False,
                "error": str(exc),
            },
        )

        return 1

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
