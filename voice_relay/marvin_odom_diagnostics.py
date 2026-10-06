#!/usr/bin/env python3
"""Existing /odom subscriber only: no publishers, estimators or control clients.

Run in the same domain-42 ROS environment as the existing isolation source.
JSON is consumed by a bounded diagnostic cache, never by motion admission.
"""
import json
import math
import time


def main():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node("marvin_odometry_diagnostics")
    publishers = []

    def discover():
        publishers[:] = sorted(info.node_namespace.rstrip("/") + "/" + info.node_name
                               + ":" + bytes(info.endpoint_gid).hex()
                               for info in node.get_publishers_info_by_topic("/odom"))

    def receive(message):
        receipt = time.monotonic()
        pose, twist = message.pose.pose, message.twist.twist
        q = pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if not math.isfinite(norm) or norm < 1e-9:
            return
        x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        sample = {
            "stamp_ns": stamp, "received_monotonic_seconds": receipt,
            "source_age_seconds": (node.get_clock().now().nanoseconds - stamp) / 1e9,
            "frame_id": message.header.frame_id, "child_frame_id": message.child_frame_id,
            "x": pose.position.x, "y": pose.position.y,
            "yaw": math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)),
            "linear_x": twist.linear.x, "linear_y": twist.linear.y, "angular_z": twist.angular.z,
            "pose_covariance": list(message.pose.covariance), "twist_covariance": list(message.twist.covariance),
            "publishers": list(publishers),
        }
        print(json.dumps(sample), flush=True)

    node.create_subscription(Odometry, "/odom", receive, qos_profile_sensor_data)
    node.create_timer(1.0, discover)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
