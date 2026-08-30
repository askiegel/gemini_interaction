#!/usr/bin/env python3

"""
Receive serialized navigation inputs over localhost TCP
and publish them into the isolated Zenoh Nav2 graph.

This process has no motion capability.
"""

import os
import socket
import struct
import threading

from collections import deque

import rclpy

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.serialization import deserialize_message
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

SCAN_OUTPUT = "/tony2_nav_scan"
ODOM_OUTPUT = "/tony2_nav_odom"
TF_OUTPUT = "/nav_tf"
TF_STATIC_OUTPUT = "/tf_static"

MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


def reliable_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def transient_qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def recv_exact(
    connection,
    count,
):
    data = bytearray()

    while len(data) < count:
        chunk = connection.recv(
            count - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "Navigation isolation TCP "
                "connection closed."
            )

        data.extend(chunk)

    return bytes(data)


class IsolationSink(Node):

    def __init__(self):
        super().__init__(
            "tony2_navigation_isolation_sink"
        )

        self._channel_publishers = {
            CHANNEL_SCAN:
                self.create_publisher(
                    LaserScan,
                    SCAN_OUTPUT,
                    reliable_qos(10),
                ),
            CHANNEL_ODOM:
                self.create_publisher(
                    Odometry,
                    ODOM_OUTPUT,
                    reliable_qos(20),
                ),
            CHANNEL_TF:
                self.create_publisher(
                    TFMessage,
                    TF_OUTPUT,
                    reliable_qos(100),
                ),
            CHANNEL_TF_STATIC:
                self.create_publisher(
                    TFMessage,
                    TF_STATIC_OUTPUT,
                    transient_qos(1),
                ),
        }

        self.types = {
            CHANNEL_SCAN: LaserScan,
            CHANNEL_ODOM: Odometry,
            CHANNEL_TF: TFMessage,
            CHANNEL_TF_STATIC: TFMessage,
        }

        self.pending = {
            CHANNEL_SCAN:
                deque(maxlen=25),
            CHANNEL_ODOM:
                deque(maxlen=50),
            CHANNEL_TF:
                deque(maxlen=200),
            CHANNEL_TF_STATIC:
                deque(maxlen=2),
        }

        self.counts = {
            channel: 0
            for channel in self.types
        }

        self.lock = threading.Lock()

        self.server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.server.bind(
            (HOST, PORT)
        )

        self.server.listen(1)

        self.get_logger().info(
            "Navigation isolation sink listening "
            f"on {HOST}:{PORT}."
        )

        self.reader_thread = threading.Thread(
            target=self.reader,
            daemon=True,
        )

        self.reader_thread.start()

        self.create_timer(
            0.002,
            self.drain,
        )

    def reader(self):
        connection, address = (
            self.server.accept()
        )

        connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        self.get_logger().info(
            "Navigation isolation source "
            f"connected from {address}."
        )

        try:
            while rclpy.ok():
                header = recv_exact(
                    connection,
                    5,
                )

                channel, size = struct.unpack(
                    "!BI",
                    header,
                )

                if channel not in self.types:
                    raise RuntimeError(
                        "Unknown navigation isolation "
                        f"channel {channel}."
                    )

                if (
                    size <= 0
                    or size > MAX_PAYLOAD_BYTES
                ):
                    raise RuntimeError(
                        "Invalid navigation isolation "
                        f"payload size {size}."
                    )

                payload = recv_exact(
                    connection,
                    size,
                )

                with self.lock:
                    self.pending[
                        channel
                    ].append(
                        payload
                    )

        except (
            ConnectionError,
            OSError,
            RuntimeError,
        ) as exc:
            self.get_logger().error(
                str(exc)
            )

        finally:
            connection.close()

    def drain(self):
        batches = {}

        with self.lock:
            for channel, pending in (
                self.pending.items()
            ):
                if pending:
                    batches[channel] = list(
                        pending
                    )

                    pending.clear()

        for channel, payloads in (
            batches.items()
        ):
            message_type = (
                self.types[channel]
            )

            publisher = (
                self._channel_publishers[
                    channel
                ]
            )

            for payload in payloads:
                message = deserialize_message(
                    payload,
                    message_type,
                )

                publisher.publish(
                    message
                )

                self.counts[channel] += 1

            count = self.counts[channel]

            if (
                channel in (
                    CHANNEL_TF_STATIC,
                )
                or count % 50 == 0
            ):
                self.get_logger().info(
                    "Published isolated input "
                    f"channel={channel} "
                    f"count={count}"
                )


def main():
    rclpy.init()

    node = IsolationSink()

    try:
        rclpy.spin(node)

    finally:
        try:
            node.server.close()
        except OSError:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
