#!/usr/bin/env python3

"""Stationary map-wide AMCL localization and trust validation."""

import argparse
import json
import math
import sys
import time

import rclpy

from geometry_msgs.msg import PoseWithCovarianceStamped
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
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from nav2_msgs.srv import SetInitialPose
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


POSITION_SIGMA_METERS = 0.35
YAW_SIGMA_RADIANS = math.radians(30.0)

NO_MOTION_UPDATES = 40

MAX_POSITION_SIGMA_METERS = 0.25
MAX_YAW_SIGMA_RADIANS = 0.50

MAX_SEED_POSITION_CHANGE_METERS = 0.60
MAX_SEED_YAW_CHANGE_RADIANS = math.radians(45.0)

MAX_ENDPOINT_ERROR_METERS = 0.40
MAX_MEAN_ENDPOINT_ERROR_METERS = 0.18
MIN_WITHIN_10CM_RATIO = 0.35
MIN_KNOWN_RATIO = 0.55
MIN_INSIDE_RATIO = 0.75

SCAN_CONFIRMATION_SAMPLES = 5
SCAN_CONFIRMATION_REQUIRED_PASSES = 3
SCAN_CONFIRMATION_SPACING_SECONDS = 0.35


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

    parser.add_argument(
        "--seed-pose",
        action="store_true",
    )

    # argparse must see only application arguments.
    #
    # The runtime intentionally appends ROS arguments such as:
    #
    #   --ros-args -r /tf:=/nav_tf
    #
    # Keep those arguments in sys.argv for rclpy.init(), but
    # remove them from the argv passed to argparse.
    application_args = remove_ros_args(
        args=sys.argv
    )

    args = parser.parse_args(
        application_args[1:]
    )

    # Compatibility only: the runtime historically passes
    # --x/--y/--yaw. Start-anywhere localization intentionally
    # does not use those values to seed AMCL.
    #
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

    global_localization_client = node.create_client(
        Empty,
        "/reinitialize_global_localization",
    )

    initial_pose_client = node.create_client(
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

        if args.seed_pose:
            if not initial_pose_client.wait_for_service(
                timeout_sec=4.0
            ):
                raise RuntimeError(
                    "AMCL set_initial_pose service unavailable."
                )
        else:
            if not global_localization_client.wait_for_service(
                timeout_sec=4.0
            ):
                raise RuntimeError(
                    "AMCL reinitialize_global_localization "
                    "service unavailable."
                )

        if not nomotion_client.wait_for_service(
            timeout_sec=4.0
        ):
            raise RuntimeError(
                "AMCL request_nomotion_update "
                "service unavailable."
            )

        # ----------------------------------------------------
        # Map-wide AMCL localization.
        #
        # Do not publish or supply an initial pose. Ask AMCL to
        # distribute particles across the saved map, then let
        # stationary LiDAR updates converge the distribution.
        #
        # Clear any transient-local pose received before this
        # request so it cannot count as evidence of convergence.
        # ----------------------------------------------------

        state["pose"] = None

        if args.seed_pose:
            request = SetInitialPose.Request()

            request.pose.header.frame_id = "map"
            request.pose.header.stamp = (
                node.get_clock().now().to_msg()
            )

            request.pose.pose.pose.position.x = args.x
            request.pose.pose.pose.position.y = args.y
            request.pose.pose.pose.position.z = 0.0

            half_yaw = args.yaw / 2.0

            request.pose.pose.pose.orientation.x = 0.0
            request.pose.pose.pose.orientation.y = 0.0
            request.pose.pose.pose.orientation.z = (
                math.sin(half_yaw)
            )
            request.pose.pose.pose.orientation.w = (
                math.cos(half_yaw)
            )

            covariance = request.pose.pose.covariance

            covariance[0] = 0.05 ** 2
            covariance[7] = 0.05 ** 2
            covariance[35] = (
                math.radians(10.0) ** 2
            )

            call_service(
                node,
                initial_pose_client,
                request,
                "set_initial_pose",
            )

        else:
            call_service(
                node,
                global_localization_client,
                Empty.Request(),
                "global_localization",
            )

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

        covariance_tight = (
            sigma_x <= MAX_POSITION_SIGMA_METERS
            and sigma_y <= MAX_POSITION_SIGMA_METERS
            and sigma_yaw <= MAX_YAW_SIGMA_RADIANS
        )

        # ----------------------------------------------------
        # Score fresh stationary LiDAR scans independently.
        #
        # Every scan uses the exact same geometric and coverage
        # thresholds. Trust is based on repeatable alignment,
        # rather than one noisy LiDAR frame.
        # ----------------------------------------------------

        def score_scan_alignment(scan):
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
            # Score the globally localized pose against the
            # fixed occupancy map. AMCL global localization has already searched the saved map.
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

            # Coverage is inexpensive and should use every
            # valid LiDAR return. Only the expensive nearest-wall
            # geometry calculation is reduced to 120 samples.
            coverage_returns = list(
                valid_returns
            )

            coverage_sample_count = len(
                coverage_returns
            )

            valid_returns = evenly_sample(
                valid_returns,
                120,
            )

            if len(valid_returns) < 40:
                raise RuntimeError(
                    "Too few live LiDAR obstacle returns."
                )

            coverage_inside_count = 0
            coverage_known_count = 0

            for (
                distance,
                beam_angle,
            ) in coverage_returns:
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
                    continue

                coverage_inside_count += 1

                cell_value = (
                    map_data[
                        row
                        * width
                        + column
                    ]
                )

                if cell_value >= 0:
                    coverage_known_count += 1

            endpoint_errors = []

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
                    # Coverage is validated separately by
                    # inside_ratio. Do not convert an
                    # out-of-map return into an artificial
                    # obstacle-distance error.
                    continue

                cell_value = (
                    map_data[
                        row
                        * width
                        + column
                    ]
                )

                if cell_value < 0:
                    # Unknown-space coverage is validated
                    # separately by known_ratio. It is not
                    # an observed obstacle and must not add
                    # a synthetic endpoint-distance penalty.
                    continue

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

            # Coverage ratios use every selected LiDAR
            # return. Geometric mean error uses only returns
            # that land in known map space.
            #
            # This prevents unknown/out-of-map coverage from
            # being penalized twice: once by known/inside
            # ratios and again as a synthetic 0.40 m error.
            sample_count = len(
                valid_returns
            )

            mean_endpoint_error = (
                sum(
                    endpoint_errors
                )
                / len(
                    endpoint_errors
                )
            )

            # Obstacle-distance quality applies only to
            # endpoints that land in known map space.
            #
            # Unknown/out-of-map coverage is already guarded
            # independently by known_ratio and inside_ratio.
            geometric_sample_count = len(
                endpoint_errors
            )

            within_10cm_ratio = (
                within_10cm_count
                / geometric_sample_count
            )

            if coverage_sample_count <= 0:
                raise RuntimeError(
                    "No valid LiDAR returns available "
                    "for map coverage."
                )

            known_ratio = (
                coverage_known_count
                / coverage_sample_count
            )

            inside_ratio = (
                coverage_inside_count
                / coverage_sample_count
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


            return {
                "sample_count":
                    sample_count,
                "geometric_sample_count":
                    geometric_sample_count,
                "coverage_sample_count":
                    coverage_sample_count,
                "coverage_known_count":
                    coverage_known_count,
                "coverage_inside_count":
                    coverage_inside_count,
                "mean_endpoint_error_m":
                    mean_endpoint_error,
                "within_0_10m_ratio":
                    within_10cm_ratio,
                "known_ratio":
                    known_ratio,
                "inside_ratio":
                    inside_ratio,
                "passed":
                    alignment_good,
            }


        def scan_stamp(message):
            return (
                int(
                    message.header.stamp.sec
                ),
                int(
                    message.header.stamp.nanosec
                ),
            )


        # The scan already present after AMCL convergence is
        # deliberately not counted. Require a fresh frame for
        # every confirmation sample.
        current_scan = state["scan"]

        if current_scan is None:
            raise RuntimeError(
                "Live LiDAR disappeared before "
                "scan confirmation."
            )

        previous_scan_stamp = (
            scan_stamp(
                current_scan
            )
        )

        confirmation_scans = []


        for confirmation_index in range(
            SCAN_CONFIRMATION_SAMPLES
        ):
            fresh_scan = None

            fresh_scan_deadline = (
                time.monotonic()
                + 3.0
            )

            while (
                time.monotonic()
                < fresh_scan_deadline
            ):
                rclpy.spin_once(
                    node,
                    timeout_sec=0.05,
                )

                candidate_scan = (
                    state["scan"]
                )

                if candidate_scan is None:
                    continue

                candidate_stamp = (
                    scan_stamp(
                        candidate_scan
                    )
                )

                if (
                    candidate_stamp
                    == previous_scan_stamp
                ):
                    continue

                fresh_scan = (
                    candidate_scan
                )

                previous_scan_stamp = (
                    candidate_stamp
                )

                break


            if fresh_scan is None:
                raise RuntimeError(
                    "No fresh LiDAR scan for "
                    "stationary confirmation "
                    f"{confirmation_index + 1}."
                )


            confirmation_scans.append(
                fresh_scan
            )


            if (
                confirmation_index
                + 1
                < SCAN_CONFIRMATION_SAMPLES
            ):
                spacing_deadline = (
                    time.monotonic()
                    + SCAN_CONFIRMATION_SPACING_SECONDS
                )

                while (
                    time.monotonic()
                    < spacing_deadline
                ):
                    rclpy.spin_once(
                        node,
                        timeout_sec=0.05,
                    )


        scan_confirmation_samples = [
            score_scan_alignment(
                confirmation_scan
            )
            for confirmation_scan
            in confirmation_scans
        ]


        confirmation_pass_count = sum(
            1
            for confirmation_sample
            in scan_confirmation_samples
            if confirmation_sample[
                "passed"
            ]
        )


        alignment_good = (
            confirmation_pass_count
            >= SCAN_CONFIRMATION_REQUIRED_PASSES
        )


        # Preserve the existing scan_alignment result shape.
        #
        # With five samples, the median is a real observed
        # middle value for each metric. If >= 3 complete scans
        # pass, the median values also lie on the passing side
        # of every unchanged threshold.
        def median_value(name):
            values = sorted(
                confirmation_sample[
                    name
                ]
                for confirmation_sample
                in scan_confirmation_samples
            )

            return values[
                len(values) // 2
            ]


        sample_count = int(
            median_value(
                "sample_count"
            )
        )

        geometric_sample_count = int(
            median_value(
                "geometric_sample_count"
            )
        )

        coverage_sample_count = int(
            median_value(
                "coverage_sample_count"
            )
        )

        coverage_known_count = int(
            median_value(
                "coverage_known_count"
            )
        )

        coverage_inside_count = int(
            median_value(
                "coverage_inside_count"
            )
        )

        mean_endpoint_error = (
            median_value(
                "mean_endpoint_error_m"
            )
        )

        within_10cm_ratio = (
            median_value(
                "within_0_10m_ratio"
            )
        )

        known_ratio = (
            median_value(
                "known_ratio"
            )
        )

        inside_ratio = (
            median_value(
                "inside_ratio"
            )
        )

        trusted = (
            covariance_tight
            and alignment_good
        )

        result = {
            "ok": True,
            "trusted": trusted,
            "frame_id": "map",
            "localization_method":
                (
                    "amcl_seeded"
                    if args.seed_pose
                    else "amcl_global"
                ),
            "search_scope":
                (
                    "known_home_pose"
                    if args.seed_pose
                    else "full_saved_map"
                ),
            "seed_pose_used":
                bool(args.seed_pose),
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
            "localization_search": {
                "method":
                    (
                        "set_initial_pose"
                        if args.seed_pose
                        else "global_localization"
                    ),
                "seed_pose_used":
                    bool(args.seed_pose),
            },
            "scan_alignment": {
                "sample_count":
                    sample_count,
                "geometric_sample_count":
                    geometric_sample_count,
                "coverage_sample_count":
                    coverage_sample_count,
                "coverage_known_count":
                    coverage_known_count,
                "coverage_inside_count":
                    coverage_inside_count,
                "mean_endpoint_error_m":
                    mean_endpoint_error,
                "within_0_10m_ratio":
                    within_10cm_ratio,
                "known_ratio":
                    known_ratio,
                "inside_ratio":
                    inside_ratio,
                "aggregation":
                    "median_of_confirmation_scans",
            },
            "scan_confirmation": {
                "sample_count":
                    SCAN_CONFIRMATION_SAMPLES,
                "pass_count":
                    confirmation_pass_count,
                "required_pass_count":
                    SCAN_CONFIRMATION_REQUIRED_PASSES,
                "spacing_seconds":
                    SCAN_CONFIRMATION_SPACING_SECONDS,
                "samples":
                    scan_confirmation_samples,
            },
            "diagnostic": {
                "covariance_tight":
                    covariance_tight,
                "global_search_completed":
                    not args.seed_pose,
                "seed_pose_applied":
                    bool(args.seed_pose),
                "alignment_good":
                    alignment_good,
                "trusted":
                    trusted,
            },
            "global_localization_requested":
                not args.seed_pose,
            "initial_pose_supplied":
                bool(args.seed_pose),
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
                        bool(args.seed_pose),
                    "global_localization_requested":
                        not args.seed_pose,
                    "navigation_goal_executed":
                        False,
                    "motion_enabled":
                        False,
                },
                separators=(",", ":"),
            )
        )

        sys.exit(1)
