#!/usr/bin/env python3

"""
Compute one read-only path through Tony2's isolated Nav2 planner.

This helper can call only ComputePathToPose. It never creates a
NavigateToPose action, motion lease, velocity publisher, or robot
motion command.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import rclpy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient


ACTION_NAME = "/compute_path_to_pose"
ACTION_TIMEOUT_SECONDS = 15.0
SERVER_TIMEOUT_SECONDS = 15.0


def write_result(path, payload):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def yaw_from_quaternion(quaternion):
    siny = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )

    cosy = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )

    return math.atan2(
        siny,
        cosy,
    )


class PlannerProbe:

    def __init__(self):
        self.node = rclpy.create_node(
            "tony2_read_only_path_probe"
        )

        self.client = ActionClient(
            self.node,
            ComputePathToPose,
            ACTION_NAME,
        )

    def make_pose(
        self,
        x,
        y,
        yaw,
    ):
        pose = PoseStamped()

        pose.header.frame_id = "map"

        pose.header.stamp = (
            self.node
            .get_clock()
            .now()
            .to_msg()
        )

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        pose.pose.orientation.z = math.sin(
            float(yaw) / 2.0
        )

        pose.pose.orientation.w = math.cos(
            float(yaw) / 2.0
        )

        return pose

    def compute(
        self,
        goal_x,
        goal_y,
        goal_yaw,
    ):
        if not self.client.wait_for_server(
            timeout_sec=SERVER_TIMEOUT_SECONDS
        ):
            raise RuntimeError(
                "Tony2 ComputePathToPose action server "
                "is unavailable."
            )

        request = ComputePathToPose.Goal()

        request.goal = self.make_pose(
            goal_x,
            goal_y,
            goal_yaw,
        )

        # The planner obtains the authoritative current
        # pose from the already-trusted map transform.
        request.use_start = False

        # Empty ID selects the configured planner plugin.
        request.planner_id = ""

        future = self.client.send_goal_async(
            request
        )

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=8.0,
        )

        if not future.done():
            raise RuntimeError(
                "ComputePathToPose acceptance timed out."
            )

        handle = future.result()

        if (
            handle is None
            or handle.accepted is not True
        ):
            raise RuntimeError(
                "Tony2 planner rejected the path request."
            )

        result_future = (
            handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=ACTION_TIMEOUT_SECONDS,
        )

        if not result_future.done():
            cancel_future = (
                handle.cancel_goal_async()
            )

            rclpy.spin_until_future_complete(
                self.node,
                cancel_future,
                timeout_sec=2.0,
            )

            raise RuntimeError(
                "ComputePathToPose timed out."
            )

        wrapped = result_future.result()

        if wrapped is None:
            raise RuntimeError(
                "ComputePathToPose returned no result."
            )

        if (
            wrapped.status
            != GoalStatus.STATUS_SUCCEEDED
        ):
            raise RuntimeError(
                "ComputePathToPose did not succeed; "
                f"status={wrapped.status}."
            )

        result = wrapped.result
        path = result.path

        if path.header.frame_id != "map":
            raise RuntimeError(
                "Planner returned a non-map-frame path."
            )

        poses = []

        for item in path.poses:
            poses.append(
                {
                    "x":
                        float(
                            item.pose.position.x
                        ),
                    "y":
                        float(
                            item.pose.position.y
                        ),
                    "yaw":
                        float(
                            yaw_from_quaternion(
                                item.pose.orientation
                            )
                        ),
                }
            )

        if len(poses) < 2:
            raise RuntimeError(
                "Planner returned an empty path."
            )

        length = 0.0

        for first, second in zip(
            poses,
            poses[1:],
        ):
            length += math.hypot(
                second["x"]
                - first["x"],
                second["y"]
                - first["y"],
            )

        return {
            "ok": True,
            "action":
                "COMPUTE_PATH_TO_POSE_ONLY",
            "read_only": True,
            "executed": False,
            "navigation_goal_executed":
                False,
            "motion_enabled": False,
            "frame_id": "map",
            "pose_count": len(poses),
            "length_meters": length,
            "goal": {
                "x": float(goal_x),
                "y": float(goal_y),
                "yaw": float(goal_yaw),
            },
            "error_code":
                int(
                    getattr(
                        result,
                        "error_code",
                        0,
                    )
                ),
            "poses": poses,
        }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--goal-x",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--goal-y",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--goal-yaw",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--result-file",
        required=True,
    )

    arguments, ros_arguments = (
        parser.parse_known_args()
    )

    for name in (
        "goal_x",
        "goal_y",
        "goal_yaw",
    ):
        if not math.isfinite(
            float(
                getattr(
                    arguments,
                    name,
                )
            )
        ):
            parser.error(
                f"{name} must be finite."
            )

    return (
        arguments,
        ros_arguments,
    )


def main():
    arguments, ros_arguments = (
        parse_arguments()
    )

    result_file = Path(
        arguments.result_file
    )

    rclpy.init(
        args=ros_arguments
    )

    probe = PlannerProbe()

    try:
        result = probe.compute(
            arguments.goal_x,
            arguments.goal_y,
            arguments.goal_yaw,
        )

        write_result(
            result_file,
            result,
        )

        print(
            json.dumps(
                result,
                sort_keys=True,
            )
        )

        return 0

    except Exception as exc:
        failure = {
            "ok": False,
            "action":
                "COMPUTE_PATH_TO_POSE_ONLY",
            "read_only": True,
            "executed": False,
            "navigation_goal_executed":
                False,
            "motion_enabled": False,
            "error": str(exc),
        }

        write_result(
            result_file,
            failure,
        )

        print(
            json.dumps(
                failure,
                sort_keys=True,
            )
        )

        return 2

    finally:
        probe.node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(
        main()
    )
