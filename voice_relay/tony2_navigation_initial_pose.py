#!/usr/bin/env python3

"""Stationary validation of an operator-supplied AMCL pose."""

import argparse
import json
import math
import sys
import time

import rclpy

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.srv import SetInitialPose
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


POSITION_SIGMA_METERS = 0.35
YAW_SIGMA_RADIANS = math.radians(30.0)

NO_MOTION_UPDATES = 12

MAX_POSITION_SIGMA_METERS = 0.25
MAX_YAW_SIGMA_RADIANS = 0.50

MAX_SEED_POSITION_CHANGE_METERS = 0.60
MAX_SEED_YAW_CHANGE_RADIANS = math.radians(45.0)

MAX_ENDPOINT_ERROR_METERS = 0.40
MAX_MEAN_ENDPOINT_ERROR_METERS = 0.18
MIN_WITHIN_10CM_RATIO = 0.35
MIN_KNOWN_RATIO = 0.55
MIN_INSIDE_RATIO = 0.75


def clean_frame(value):
    return str(value).lstrip("/")


def yaw_difference(first, second):
    return math.atan2(
        math.sin(first - second),
        math.cos(first - second),
    )


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (
            q.w * q.z
            + q.x * q.y
        ),
        1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        ),
    )


def call_service(
    node,
    client,
    request,
    label,
    timeout_seconds=4.0,
):
    if not client.wait_for_service(
        timeout_sec=timeout_seconds
    ):
        raise RuntimeError(
            f"{label} service unavailable."
        )

    future = client.call_async(
        request
    )

    rclpy.spin_until_future_complete(
        node,
        future,
        timeout_sec=timeout_seconds,
    )

    if not future.done():
        raise RuntimeError(
            f"{label} timed out."
        )

    if future.exception() is not None:
        raise RuntimeError(
            f"{label} failed: "
            f"{future.exception()}"
        )

    return future.result()


def evenly_sample(items, maximum):
    if len(items) <= maximum:
        return list(items)

    if maximum <= 1:
        return [
            items[0]
        ]

    indexes = [
        round(
            index
            * (len(items) - 1)
            / (maximum - 1)
        )
        for index in range(maximum)
    ]

    return [
        items[index]
        for index in indexes
    ]


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

    args = parser.parse_args()

    for name, value in (
        ("x", args.x),
        ("y", args.y),
        ("yaw", args.yaw),
    ):
        if not math.isfinite(value):
            raise RuntimeError(
                f"{name} must be finite."
            )

    rclpy.init()

    node = Node(
        "tony2_operator_initial_pose"
    )

    map_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    scan_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    pose_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    state = {
        "map": None,
        "scan": None,
        "pose": None,
        "pose_count": 0,
    }

    def map_callback(message):
        state["map"] = message

    def scan_callback(message):
        state["scan"] = message

    def pose_callback(message):
        state["pose"] = message
        state["pose_count"] += 1

    subscriptions = [
        node.create_subscription(
            OccupancyGrid,
            "/map",
            map_callback,
            map_qos,
        ),
        node.create_subscription(
            LaserScan,
            "/tony2_nav_scan",
            scan_callback,
            scan_qos,
        ),
        node.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            pose_callback,
            pose_qos,
        ),
    ]

    tf_buffer = Buffer()

    tf_listener = TransformListener(
        tf_buffer,
        node,
        spin_thread=False,
    )

    set_pose_client = node.create_client(
        SetInitialPose,
        "/set_initial_pose",
    )

    nomotion_client = node.create_client(
        Empty,
        "/request_nomotion_update",
    )

    try:
        # ----------------------------------------------------
        # Require the fixed map and live LiDAR before changing
        # AMCL state.
        # ----------------------------------------------------

        deadline = (
            time.monotonic()
            + 6.0
        )

        while (
            time.monotonic() < deadline
            and (
                state["map"] is None
                or state["scan"] is None
            )
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

        if state["map"] is None:
            raise RuntimeError(
                "Fixed map unavailable."
            )

        if state["scan"] is None:
            raise RuntimeError(
                "Live LiDAR unavailable."
            )

        if not set_pose_client.wait_for_service(
            timeout_sec=4.0
        ):
            raise RuntimeError(
                "AMCL set_initial_pose service unavailable."
            )

        if not nomotion_client.wait_for_service(
            timeout_sec=4.0
        ):
            raise RuntimeError(
                "AMCL request_nomotion_update "
                "service unavailable."
            )

        # ----------------------------------------------------
        # Operator-supplied local pose estimate.
        # ----------------------------------------------------

        request = SetInitialPose.Request()

        request.pose.header.frame_id = "map"

        request.pose.header.stamp = (
            node.get_clock()
            .now()
            .to_msg()
        )

        request.pose.pose.pose.position.x = (
            args.x
        )

        request.pose.pose.pose.position.y = (
            args.y
        )

        request.pose.pose.pose.position.z = 0.0

        half_yaw = (
            args.yaw
            / 2.0
        )

        request.pose.pose.pose.orientation.x = 0.0
        request.pose.pose.pose.orientation.y = 0.0

        request.pose.pose.pose.orientation.z = (
            math.sin(
                half_yaw
            )
        )

        request.pose.pose.pose.orientation.w = (
            math.cos(
                half_yaw
            )
        )

        covariance = [
            0.0
        ] * 36

        covariance[0] = (
            POSITION_SIGMA_METERS
            ** 2
        )

        covariance[7] = (
            POSITION_SIGMA_METERS
            ** 2
        )

        covariance[35] = (
            YAW_SIGMA_RADIANS
            ** 2
        )

        request.pose.pose.covariance = (
            covariance
        )

        call_service(
            node,
            set_pose_client,
            request,
            "set_initial_pose",
        )

        # Do not accept a transient-local pose left over
        # from before this service call as evidence.
        state["pose"] = None

        # ----------------------------------------------------
        # Force repeated stationary laser updates.
        # ----------------------------------------------------

        for index in range(
            NO_MOTION_UPDATES
        ):
            before = (
                state["pose_count"]
            )

            call_service(
                node,
                nomotion_client,
                Empty.Request(),
                (
                    "request_nomotion_update "
                    f"{index + 1}"
                ),
            )

            update_deadline = (
                time.monotonic()
                + 1.5
            )

            while (
                time.monotonic()
                < update_deadline
                and state["pose_count"]
                    <= before
            ):
                rclpy.spin_once(
                    node,
                    timeout_sec=0.05,
                )

            if (
                state["pose_count"]
                <= before
            ):
                raise RuntimeError(
                    "AMCL did not publish a fresh pose "
                    f"after no-motion update {index + 1}."
                )

        pose_message = state["pose"]

        if pose_message is None:
            raise RuntimeError(
                "AMCL produced no final pose."
            )

        pose = (
            pose_message.pose.pose
        )

        covariance = (
            pose_message.pose.covariance
        )

        final_x = float(
            pose.position.x
        )

        final_y = float(
            pose.position.y
        )

        final_yaw = (
            yaw_from_quaternion(
                pose.orientation
            )
        )

        sigma_x = math.sqrt(
            max(
                float(
                    covariance[0]
                ),
                0.0,
            )
        )

        sigma_y = math.sqrt(
            max(
                float(
                    covariance[7]
                ),
                0.0,
            )
        )

        sigma_yaw = math.sqrt(
            max(
                float(
                    covariance[35]
                ),
                0.0,
            )
        )

        seed_position_change = math.hypot(
            final_x - args.x,
            final_y - args.y,
        )

        seed_yaw_change = abs(
            yaw_difference(
                final_yaw,
                args.yaw,
            )
        )

        covariance_tight = (
            sigma_x
                <= MAX_POSITION_SIGMA_METERS
            and sigma_y
                <= MAX_POSITION_SIGMA_METERS
            and sigma_yaw
                <= MAX_YAW_SIGMA_RADIANS
        )

        seed_consistent = (
            seed_position_change
                <= MAX_SEED_POSITION_CHANGE_METERS
            and seed_yaw_change
                <= MAX_SEED_YAW_CHANGE_RADIANS
        )

        # ----------------------------------------------------
        # Obtain map <- lidar_link from AMCL + odom + robot TF.
        # ----------------------------------------------------

        scan = state["scan"]

        scan_frame = clean_frame(
            scan.header.frame_id
        )

        transform = None

        transform_deadline = (
            time.monotonic()
            + 4.0
        )

        while (
            time.monotonic()
            < transform_deadline
        ):
            try:
                transform = (
                    tf_buffer.lookup_transform(
                        "map",
                        scan_frame,
                        Time(),
                        timeout=Duration(
                            seconds=0.5
                        ),
                    )
                )

                break

            except TransformException:
                rclpy.spin_once(
                    node,
                    timeout_sec=0.1,
                )

        if transform is None:
            raise RuntimeError(
                "AMCL did not produce map-to-LiDAR TF."
            )

        translation = (
            transform.transform.translation
        )

        rotation = (
            transform.transform.rotation
        )

        tf_yaw = (
            yaw_from_quaternion(
                rotation
            )
        )

        tf_cos = math.cos(
            tf_yaw
        )

        tf_sin = math.sin(
            tf_yaw
        )

        # ----------------------------------------------------
        # Score this one operator-seeded pose against the
        # fixed occupancy map. No global pose search occurs.
        # ----------------------------------------------------

        grid = state["map"]

        width = int(
            grid.info.width
        )

        height = int(
            grid.info.height
        )

        resolution = float(
            grid.info.resolution
        )

        map_data = list(
            grid.data
        )

        origin = (
            grid.info.origin
        )

        origin_yaw = (
            yaw_from_quaternion(
                origin.orientation
            )
        )

        origin_cos = math.cos(
            origin_yaw
        )

        origin_sin = math.sin(
            origin_yaw
        )

        occupied_cells = []

        for row in range(height):
            offset = (
                row
                * width
            )

            for column in range(width):
                value = (
                    map_data[
                        offset + column
                    ]
                )

                if value >= 65:
                    occupied_cells.append(
                        (
                            row,
                            column,
                        )
                    )

        if not occupied_cells:
            raise RuntimeError(
                "Fixed map has no occupied cells."
            )

        valid_returns = []

        angle = float(
            scan.angle_min
        )

        for raw_range in scan.ranges:
            distance = float(
                raw_range
            )

            if (
                math.isfinite(
                    distance
                )
                and distance
                    >= float(
                        scan.range_min
                    )
                and distance
                    <= (
                        float(
                            scan.range_max
                        )
                        * 0.98
                    )
            ):
                valid_returns.append(
                    (
                        distance,
                        angle,
                    )
                )

            angle += float(
                scan.angle_increment
            )

        valid_returns = evenly_sample(
            valid_returns,
            120,
        )

        if len(valid_returns) < 40:
            raise RuntimeError(
                "Too few live LiDAR obstacle returns."
            )

        endpoint_errors = []

        inside_count = 0
        known_count = 0
        within_10cm_count = 0

        for distance, beam_angle in valid_returns:
            lidar_x = (
                distance
                * math.cos(
                    beam_angle
                )
            )

            lidar_y = (
                distance
                * math.sin(
                    beam_angle
                )
            )

            map_x = (
                float(
                    translation.x
                )
                + tf_cos
                * lidar_x
                - tf_sin
                * lidar_y
            )

            map_y = (
                float(
                    translation.y
                )
                + tf_sin
                * lidar_x
                + tf_cos
                * lidar_y
            )

            dx = (
                map_x
                - float(
                    origin.position.x
                )
            )

            dy = (
                map_y
                - float(
                    origin.position.y
                )
            )

            local_x = (
                origin_cos
                * dx
                + origin_sin
                * dy
            )

            local_y = (
                -origin_sin
                * dx
                + origin_cos
                * dy
            )

            column = math.floor(
                local_x
                / resolution
            )

            row = math.floor(
                local_y
                / resolution
            )

            if (
                row < 0
                or row >= height
                or column < 0
                or column >= width
            ):
                endpoint_errors.append(
                    MAX_ENDPOINT_ERROR_METERS
                )

                continue

            inside_count += 1

            cell_value = (
                map_data[
                    row
                    * width
                    + column
                ]
            )

            if cell_value < 0:
                endpoint_errors.append(
                    MAX_ENDPOINT_ERROR_METERS
                )

                continue

            known_count += 1

            nearest = min(
                math.hypot(
                    row - occupied_row,
                    column - occupied_column,
                )
                * resolution
                for (
                    occupied_row,
                    occupied_column,
                )
                in occupied_cells
            )

            nearest = min(
                nearest,
                MAX_ENDPOINT_ERROR_METERS,
            )

            endpoint_errors.append(
                nearest
            )

            if nearest <= 0.10:
                within_10cm_count += 1

        if not endpoint_errors:
            raise RuntimeError(
                "No LiDAR endpoints were scored."
            )

        sample_count = len(
            endpoint_errors
        )

        mean_endpoint_error = (
            sum(
                endpoint_errors
            )
            / sample_count
        )

        within_10cm_ratio = (
            within_10cm_count
            / sample_count
        )

        known_ratio = (
            known_count
            / sample_count
        )

        inside_ratio = (
            inside_count
            / sample_count
        )

        alignment_good = (
            mean_endpoint_error
                <= MAX_MEAN_ENDPOINT_ERROR_METERS
            and within_10cm_ratio
                >= MIN_WITHIN_10CM_RATIO
            and known_ratio
                >= MIN_KNOWN_RATIO
            and inside_ratio
                >= MIN_INSIDE_RATIO
        )

        trusted = (
            covariance_tight
            and seed_consistent
            and alignment_good
        )

        result = {
            "ok": True,
            "trusted": trusted,
            "frame_id": "map",
            "seed": {
                "x": args.x,
                "y": args.y,
                "yaw_rad": args.yaw,
                "yaw_deg":
                    math.degrees(
                        args.yaw
                    ),
                "sigma_position_m":
                    POSITION_SIGMA_METERS,
                "sigma_yaw_deg":
                    math.degrees(
                        YAW_SIGMA_RADIANS
                    ),
            },
            "final_pose": {
                "x": final_x,
                "y": final_y,
                "yaw_rad": final_yaw,
                "yaw_deg":
                    math.degrees(
                        final_yaw
                    ),
            },
            "uncertainty": {
                "sigma_x_m": sigma_x,
                "sigma_y_m": sigma_y,
                "sigma_yaw_rad":
                    sigma_yaw,
                "sigma_yaw_deg":
                    math.degrees(
                        sigma_yaw
                    ),
            },
            "seed_consistency": {
                "position_change_m":
                    seed_position_change,
                "yaw_change_rad":
                    seed_yaw_change,
                "yaw_change_deg":
                    math.degrees(
                        seed_yaw_change
                    ),
            },
            "scan_alignment": {
                "sample_count":
                    sample_count,
                "mean_endpoint_error_m":
                    mean_endpoint_error,
                "within_0_10m_ratio":
                    within_10cm_ratio,
                "known_ratio":
                    known_ratio,
                "inside_ratio":
                    inside_ratio,
            },
            "diagnostic": {
                "covariance_tight":
                    covariance_tight,
                "seed_consistent":
                    seed_consistent,
                "alignment_good":
                    alignment_good,
                "trusted":
                    trusted,
            },
            "global_localization_requested":
                False,
            "initial_pose_supplied":
                True,
            "nomotion_updates_requested":
                NO_MOTION_UPDATES,
            "stationary_required":
                True,
            "navigation_goal_executed":
                False,
            "motion_enabled":
                False,
        }

        print(
            json.dumps(
                result,
                separators=(",", ":"),
            )
        )

        return 0

    finally:
        # Keep references alive until all subscriptions are done.
        _ = subscriptions
        _ = tf_listener

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "trusted": False,
                    "error": str(exc),
                    "initial_pose_supplied":
                        False,
                    "navigation_goal_executed":
                        False,
                    "motion_enabled":
                        False,
                },
                separators=(",", ":"),
            )
        )

        sys.exit(1)
