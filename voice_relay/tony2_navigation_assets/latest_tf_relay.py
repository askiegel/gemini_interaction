#!/usr/bin/env python3

from collections import deque
import copy
import threading

import rclpy

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from tf2_msgs.msg import TFMessage


NAVIGATION_FRAME_PAIRS = frozenset(
    {
        ("map", "odom"),
        ("odom", "base_footprint"),
        ("base_footprint", "base_link"),
    }
)


def stamp_nanoseconds(transform):
    stamp = transform.header.stamp

    return (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
    )


class LatestTfRelay(Node):
    """
    Coalesce high-rate dynamic TF into one latest-state
    message at a bounded rate for guarded Nav2.

    Input:
        /tf
        BEST_EFFORT
        KEEP_LAST depth 1

    Output:
        /nav_tf
        RELIABLE
        KEEP_LAST depth 1

    Only the dynamic frame pairs required by guarded Nav2
    are relayed: map -> odom, odom -> base_footprint, and
    base_footprint -> base_link. Articulated leg transforms
    remain on the source /tf tree and are not duplicated.

    A bounded recent history is retained independently for
    each navigation frame pair. This preserves same-pair
    timestamps for delayed sensor messages without allowing
    a stalled relay publication to grow an unbounded outgoing
    TF batch.

    Out-of-order transforms for an individual frame pair
    are rejected.
    """

    def __init__(self):
        super().__init__("latest_tf_relay")

        self.declare_parameter(
            "input_topic",
            "/tf",
        )

        self.declare_parameter(
            "output_topic",
            "/nav_tf",
        )

        self.declare_parameter(
            "publish_frequency",
            10.0,
        )

        self.declare_parameter(
            "history_depth_per_pair",
            8,
        )

        input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )

        output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        frequency = float(
            self.get_parameter(
                "publish_frequency"
            ).value
        )

        history_depth_per_pair = int(
            self.get_parameter(
                "history_depth_per_pair"
            ).value
        )

        if frequency <= 0.0:
            raise ValueError(
                "publish_frequency must be positive"
            )

        if history_depth_per_pair <= 0:
            raise ValueError(
                "history_depth_per_pair must be positive"
            )

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._lock = threading.Lock()
        self._history_depth_per_pair = (
            history_depth_per_pair
        )
        self._latest_stamp_ns = {}
        self._pending = {}

        self._publisher = self.create_publisher(
            TFMessage,
            output_topic,
            output_qos,
        )

        self._subscription = self.create_subscription(
            TFMessage,
            input_topic,
            self._receive,
            input_qos,
        )

        self._timer = self.create_timer(
            1.0 / frequency,
            self._publish_latest,
        )

        self.get_logger().info(
            "Latest TF relay: "
            f"{input_topic} -> {output_topic} "
            f"at {frequency:.1f} Hz"
        )

    def _receive(self, message):
        with self._lock:
            for transform in message.transforms:
                key = (
                    transform.header.frame_id,
                    transform.child_frame_id,
                )

                if key not in NAVIGATION_FRAME_PAIRS:
                    continue

                stamp_ns = stamp_nanoseconds(
                    transform
                )

                previous_stamp_ns = (
                    self._latest_stamp_ns.get(key)
                )

                if (
                    previous_stamp_ns is not None
                    and stamp_ns < previous_stamp_ns
                ):
                    continue

                self._latest_stamp_ns[key] = stamp_ns

                history = self._pending.get(key)

                if history is None:
                    history = deque(
                        maxlen=self._history_depth_per_pair
                    )

                    self._pending[key] = history

                history.append(
                    copy.deepcopy(transform)
                )

    def _publish_latest(self):
        with self._lock:
            if not self._pending:
                return

            pending = self._pending
            self._pending = {}

        transforms = []

        for key in sorted(pending):
            transforms.extend(pending[key])

        message = TFMessage()
        message.transforms = transforms

        self._publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    node = LatestTfRelay()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
