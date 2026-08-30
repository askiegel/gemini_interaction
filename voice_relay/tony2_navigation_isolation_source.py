#!/usr/bin/env python3

"""
Read the five navigation inputs from ROS domain 42 and
send their serialized ROS messages over one localhost TCP
connection.

This process has no motion capability.
"""

import os
import socket
import struct
import time

import rclpy

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


HOST = "127.0.0.1"

PORT = int(
    os.getenv(
        "TONY2_NAVIGATION_ISOLATION_PORT",
        "43143",
    )
)

CHANNEL_SCAN = 1
CHANNEL_ODOM = 2
CHANNEL_TF = 3
CHANNEL_TF_STATIC = 4
CHANNEL_MAP = 5

SCAN_INPUT = "/scan"
ODOM_INPUT = "/odom"
TF_INPUT = "/mayday_navigation_tf"
TF_STATIC_INPUT = "/tf_static"
MAP_INPUT = "/map"


def best_effort_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def transient_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class IsolationSource(Node):

    def __init__(self):
        super().__init__(
            "tony2_navigation_isolation_source"
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        deadline = time.monotonic() + 15.0

        while True:
            try:
                self.sock.connect(
                    (HOST, PORT)
                )
                break

            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Navigation isolation sink "
                        "did not become available."
                    )

                time.sleep(0.1)

        self.counts = {
            CHANNEL_SCAN: 0,
            CHANNEL_ODOM: 0,
            CHANNEL_TF: 0,
            CHANNEL_TF_STATIC: 0,
            CHANNEL_MAP: 0,
        }

        self.create_subscription(
            LaserScan,
            SCAN_INPUT,
            lambda message:
                self.forward(
                    CHANNEL_SCAN,
                    message,
                ),
            best_effort_qos(5),
        )

        self.create_subscription(
            Odometry,
            ODOM_INPUT,
            lambda message:
                self.forward(
                    CHANNEL_ODOM,
                    message,
                ),
            best_effort_qos(10),
        )

        self.create_subscription(
            TFMessage,
            TF_INPUT,
            lambda message:
                self.forward(
                    CHANNEL_TF,
                    message,
                ),
            best_effort_qos(100),
        )

        self.create_subscription(
            TFMessage,
            TF_STATIC_INPUT,
            lambda message:
                self.forward(
                    CHANNEL_TF_STATIC,
                    message,
                ),
            transient_qos(1),
        )

        self.create_subscription(
            OccupancyGrid,
            MAP_INPUT,
            lambda message:
                self.forward(
                    CHANNEL_MAP,
                    message,
                ),
            transient_qos(1),
        )

        self.get_logger().info(
            "Navigation isolation domain-42 source ready."
        )

    def forward(
        self,
        channel,
        message,
    ):
        payload = bytes(
            serialize_message(
                message
            )
        )

        header = struct.pack(
            "!BI",
            channel,
            len(payload),
        )

        self.sock.sendall(
            header + payload
        )

        self.counts[channel] += 1

        count = self.counts[channel]

        if (
            channel in (
                CHANNEL_TF_STATIC,
                CHANNEL_MAP,
            )
            or count % 50 == 0
        ):
            self.get_logger().info(
                "Forwarded "
                f"channel={channel} "
                f"count={count} "
                f"bytes={len(payload)}"
            )


def main():
    rclpy.init()

    node = IsolationSource()

    try:
        rclpy.spin(node)

    finally:
        try:
            node.sock.close()
        except OSError:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
