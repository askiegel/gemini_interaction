#!/usr/bin/env python3

"""Read-only Tony2 Cartographer telemetry for the browser dashboard."""

import json
import math
import os
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import rclpy

from cartographer_ros_msgs.msg import SubmapList
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


SNAPSHOT_PATH = Path(
    os.getenv(
        "TONY2_MAPPING_SNAPSHOT",
        "/tmp/tony2_mapping_snapshot.json",
    )
)

MINIMUM_SUBMAPS = 3
MINIMUM_MATURE_SUBMAPS = 2
MINIMUM_MATURE_VERSION = 100


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def yaw_from_quaternion(quaternion):
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class Tony2MappingProbe(Node):
    """Cache /map and /submap_list without controlling the robot."""

    def __init__(self):
        super().__init__("tony2_mapping_dashboard_probe")

        self._map = None
        self._map_received_at = None
        self._map_received_monotonic = None
        self._submaps = []
        self._submap_seen = False
        self._last_error = None

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        submap_qos = QoSProfile(depth=10)
        submap_qos.reliability = ReliabilityPolicy.RELIABLE
        submap_qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._map_callback,
            map_qos,
        )

        self.create_subscription(
            SubmapList,
            "/submap_list",
            self._submap_callback,
            submap_qos,
        )

        self.create_timer(
            0.5,
            self._write_snapshot,
        )

    def _map_callback(self, message):
        width = int(message.info.width)
        height = int(message.info.height)
        resolution = float(message.info.resolution)
        cells = [int(value) for value in message.data]
        expected = width * height

        if width <= 0 or height <= 0:
            self._last_error = "Map dimensions must be positive."
            return

        if resolution <= 0.0:
            self._last_error = "Map resolution must be positive."
            return

        if len(cells) != expected:
            self._last_error = (
                "Occupancy data length does not match map dimensions."
            )
            return

        if any(value < -1 or value > 100 for value in cells):
            self._last_error = (
                "Occupancy data contains a value outside -1 through 100."
            )
            return

        stamp = message.header.stamp
        stamp_seconds = (
            float(stamp.sec)
            + float(stamp.nanosec) / 1_000_000_000.0
        )

        origin = message.info.origin

        self._map = {
            "frame_id": str(message.header.frame_id),
            "name": "live_cartographer_map",
            "stamp_seconds": stamp_seconds,
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin": {
                "x": float(origin.position.x),
                "y": float(origin.position.y),
                "yaw": yaw_from_quaternion(
                    origin.orientation
                ),
            },
            "cell_count": expected,
            "unknown_cell_count": cells.count(-1),
            "free_cell_count": cells.count(0),
            "occupied_cell_count": cells.count(100),
            "probability_cell_count": sum(
                1
                for value in cells
                if 0 < value < 100
            ),
            "encoding": "ros_occupancy_probabilities",
            "unknown_value": -1,
            "free_value": 0,
            "occupied_value": 100,
            "cells": cells,
            "source": {
                "topic": "/map",
                "runtime": "tony2_cartographer",
                "host": "Tony2",
                "mutable": True,
                "authoritative": False,
            },
        }

        self._map_received_at = utc_now()
        self._map_received_monotonic = time.monotonic()
        self._last_error = None

    def _submap_callback(self, message):
        self._submap_seen = True

        self._submaps = [
            {
                "trajectory_id": int(item.trajectory_id),
                "index": int(item.submap_index),
                "version": int(item.submap_version),
            }
            for item in message.submap
        ]

    def _readiness(self):
        count = len(self._submaps)

        mature_count = sum(
            1
            for item in self._submaps
            if item["version"] >= MINIMUM_MATURE_VERSION
        )

        ready = (
            count >= MINIMUM_SUBMAPS
            and mature_count >= MINIMUM_MATURE_SUBMAPS
        )

        if ready:
            status = "READY_TO_SAVE"
        elif count:
            status = "BUILDING_SUBMAPS"
        else:
            status = "WAITING_FOR_SUBMAPS"

        return {
            "available": self._submap_seen,
            "status": status,
            "ready": ready,
            "submap_count": count,
            "mature_submap_count": mature_count,
            "minimum_submap_count": MINIMUM_SUBMAPS,
            "minimum_mature_submap_count": (
                MINIMUM_MATURE_SUBMAPS
            ),
            "minimum_mature_version": MINIMUM_MATURE_VERSION,
            "submap_progress": min(
                1.0,
                count / float(MINIMUM_SUBMAPS),
            ),
            "mature_submap_progress": min(
                1.0,
                mature_count / float(
                    MINIMUM_MATURE_SUBMAPS
                ),
            ),
            "submaps": list(self._submaps),
        }

    def _telemetry(self):
        if self._map is None:
            return {
                "available": False,
                "status": (
                    "INVALID_MAP"
                    if self._last_error
                    else "WAITING_FOR_MAP"
                ),
                "received_at": None,
                "age_seconds": None,
                "error": self._last_error,
                "map": None,
            }

        age = None

        if self._map_received_monotonic is not None:
            age = max(
                0.0,
                time.monotonic()
                - self._map_received_monotonic,
            )

        return {
            "available": True,
            "status": "READY",
            "received_at": self._map_received_at,
            "age_seconds": age,
            "error": None,
            "map": self._map,
        }

    def _write_snapshot(self):
        payload = {
            "ok": True,
            "service": "tony2_mapping_dashboard_probe",
            "host": "Tony2",
            "timestamp": utc_now(),
            "topic": "/map",
            "source": "live_cartographer_map",
            "read_only": True,
            "authoritative": False,
            "telemetry": self._telemetry(),
            "readiness": self._readiness(),
        }

        temporary = SNAPSHOT_PATH.with_suffix(
            SNAPSHOT_PATH.suffix + ".tmp"
        )

        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )

            os.replace(
                temporary,
                SNAPSHOT_PATH,
            )

        except Exception as exc:
            self.get_logger().error(
                f"Could not write mapping snapshot: {exc}"
            )


def main():
    rclpy.init()

    node = Tony2MappingProbe()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
