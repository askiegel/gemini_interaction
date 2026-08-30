#!/usr/bin/env python3

"""
Provide one Tony2 DDS ingress for Nav2's high-rate
Mayday scan and odometry streams.

Mayday:
    /scan
    /odom

Tony2-local fan-out:
    /tony2_nav_scan
    /tony2_nav_odom

This node has no motion capability.
"""

import rclpy

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import LaserScan


SCAN_INPUT = "/scan"
SCAN_OUTPUT = "/tony2_nav_scan"

ODOM_INPUT = "/odom"
ODOM_OUTPUT = "/tony2_nav_odom"


def best_effort_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class Tony2NavigationSensorRelay(Node):
    def __init__(self):
        super().__init__(
            "tony2_navigation_sensor_relay"
        )

        scan_qos = best_effort_qos(5)
        odom_qos = best_effort_qos(10)

        self.scan_publisher = (
            self.create_publisher(
                LaserScan,
                SCAN_OUTPUT,
                scan_qos,
            )
        )

        self.odom_publisher = (
            self.create_publisher(
                Odometry,
                ODOM_OUTPUT,
                odom_qos,
            )
        )

        self.scan_subscription = (
            self.create_subscription(
                LaserScan,
                SCAN_INPUT,
                self.publish_scan,
                scan_qos,
            )
        )

        self.odom_subscription = (
            self.create_subscription(
                Odometry,
                ODOM_INPUT,
                self.publish_odom,
                odom_qos,
            )
        )

        self.get_logger().info(
            "Tony2 Nav2 sensor ingress ready: "
            "/scan -> /tony2_nav_scan, "
            "/odom -> /tony2_nav_odom"
        )

    def publish_scan(self, message):
        self.scan_publisher.publish(message)

    def publish_odom(self, message):
        self.odom_publisher.publish(message)


def main():
    rclpy.init()

    node = Tony2NavigationSensorRelay()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
