#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Tony Kiegel
#

"""
Build a local occupancy-map candidate from repeated stationary LaserScans.

The builder deliberately does not perform SLAM. All accepted scans are
assumed to originate from one physically stationary LD06 pose.

Stable returns become occupied endpoints. Cells along stable rays become
free. Cells that have not been observed remain unknown.

The module does not publish ROS messages and contains no robot-motion path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path


TAU = 2.0 * math.pi


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _bresenham(x0, y0, x1, y1):
    """Return integer cells from start through end."""
    cells = []

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1

    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1

    error = dx + dy

    while True:
        cells.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        doubled = 2 * error

        if doubled >= dy:
            error += dy
            x0 += sx

        if doubled <= dx:
            error += dx
            y0 += sy

    return cells


def fetch_json(url, timeout=3.0):
    request = urllib.request.Request(
        url,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        raw = response.read()

    payload = json.loads(raw.decode("utf-8"))

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected JSON object from {url}."
        )

    return payload


def motion_signature(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Robot status payload is not an object."
        )

    if payload.get("status") != "READY":
        raise RuntimeError(
            "Robot Bridge is not READY."
        )

    if payload.get("ros_ready") is not True:
        raise RuntimeError(
            "Robot Bridge ROS state is not ready."
        )

    motion = payload.get("motion")

    if not isinstance(motion, dict):
        raise RuntimeError(
            "Robot status has no motion state."
        )

    linear = _finite_number(
        motion.get("linear_x")
    )
    angular = _finite_number(
        motion.get("angular_z")
    )

    if linear is None or angular is None:
        raise RuntimeError(
            "Robot motion values are invalid."
        )

    if abs(linear) > 1e-12:
        raise RuntimeError(
            "Mayday is reporting nonzero linear velocity."
        )

    if abs(angular) > 1e-12:
        raise RuntimeError(
            "Mayday is reporting nonzero angular velocity."
        )

    if motion.get("streaming") is not False:
        raise RuntimeError(
            "Mayday reports streaming motion."
        )

    return {
        "last_command_at":
            motion.get("last_command_at"),
        "watchdog_stop_count":
            motion.get("watchdog_stop_count"),
    }


def verify_motion_unchanged(payload, baseline):
    current = motion_signature(payload)

    if current != baseline:
        raise RuntimeError(
            "Robot motion history changed during "
            "stationary acquisition."
        )


class StationaryMapBuilder:
    def __init__(
        self,
        *,
        resolution=0.05,
        minimum_scans=100,
        minimum_stable_bins=80,
        persistence_ratio=0.70,
        stable_ratio=0.75,
        base_tolerance_meters=0.08,
        relative_tolerance=0.02,
        maximum_map_range_meters=8.0,
        map_margin_meters=0.25,
    ):
        self.resolution = float(resolution)
        self.minimum_scans = int(
            minimum_scans
        )
        self.minimum_stable_bins = int(
            minimum_stable_bins
        )
        self.persistence_ratio = float(
            persistence_ratio
        )
        self.stable_ratio = float(
            stable_ratio
        )
        self.base_tolerance_meters = float(
            base_tolerance_meters
        )
        self.relative_tolerance = float(
            relative_tolerance
        )
        self.maximum_map_range_meters = float(
            maximum_map_range_meters
        )
        self.map_margin_meters = float(
            map_margin_meters
        )

        if self.resolution <= 0:
            raise ValueError(
                "resolution must be positive."
            )

        if self.minimum_scans < 1:
            raise ValueError(
                "minimum_scans must be positive."
            )

        if self.minimum_stable_bins < 1:
            raise ValueError(
                "minimum_stable_bins must be positive."
            )

        if not 0.0 < self.persistence_ratio <= 1.0:
            raise ValueError(
                "persistence_ratio must be in (0, 1]."
            )

        if not 0.0 < self.stable_ratio <= 1.0:
            raise ValueError(
                "stable_ratio must be in (0, 1]."
            )

        if self.maximum_map_range_meters <= 0:
            raise ValueError(
                "maximum map range must be positive."
            )

        self.scan_count = 0
        self.bin_count = None
        self._bins = None

    def _ensure_bins(self, angle_increment):
        candidate = int(
            round(
                TAU
                / abs(angle_increment)
            )
        )

        if candidate < 90:
            raise ValueError(
                "LaserScan angular resolution is "
                "too coarse."
            )

        if self.bin_count is None:
            self.bin_count = candidate
            self._bins = [
                []
                for _ in range(
                    self.bin_count
                )
            ]

            return

        if abs(candidate - self.bin_count) > 2:
            raise ValueError(
                "LaserScan angular resolution "
                "changed unexpectedly."
            )

    def add_scan(self, scan):
        if not isinstance(scan, dict):
            raise ValueError(
                "scan must be an object."
            )

        angle_min = _finite_number(
            scan.get("angle_min")
        )
        angle_increment = _finite_number(
            scan.get("angle_increment")
        )
        range_min = _finite_number(
            scan.get("range_min")
        )
        range_max = _finite_number(
            scan.get("range_max")
        )

        ranges = scan.get("ranges")

        if (
            angle_min is None
            or angle_increment is None
            or abs(angle_increment) < 1e-9
            or range_min is None
            or range_max is None
            or range_max <= range_min
            or not isinstance(ranges, list)
            or len(ranges) < 90
        ):
            raise ValueError(
                "LaserScan geometry is invalid."
            )

        self._ensure_bins(
            angle_increment
        )

        per_scan = {}

        for index, raw_range in enumerate(
            ranges
        ):
            distance = _finite_number(
                raw_range
            )

            if distance is None:
                continue

            if distance < range_min:
                continue

            if distance > range_max:
                continue

            if (
                distance
                > self.maximum_map_range_meters
            ):
                continue

            angle = (
                angle_min
                + index * angle_increment
            ) % TAU

            bin_index = int(
                round(
                    angle
                    / TAU
                    * self.bin_count
                )
            ) % self.bin_count

            per_scan.setdefault(
                bin_index,
                [],
            ).append(distance)

        for bin_index, values in (
            per_scan.items()
        ):
            # Some LD06 revolutions may produce one
            # extra sample near the wrap boundary.
            # Collapse duplicates within the same
            # angular bin before adding the scan.
            self._bins[bin_index].append(
                statistics.median(values)
            )

        self.scan_count += 1

    def stable_returns(self):
        if self.scan_count == 0:
            return []

        required_observations = int(
            math.ceil(
                self.scan_count
                * self.persistence_ratio
            )
        )

        stable = []

        for index, observations in enumerate(
            self._bins
        ):
            if (
                len(observations)
                < required_observations
            ):
                continue

            median_range = statistics.median(
                observations
            )

            tolerance = max(
                self.base_tolerance_meters,
                abs(median_range)
                * self.relative_tolerance,
            )

            agreeing = [
                value
                for value in observations
                if abs(
                    value - median_range
                ) <= tolerance
            ]

            agreement_ratio = (
                len(agreeing)
                / len(observations)
            )

            if (
                agreement_ratio
                < self.stable_ratio
            ):
                continue

            refined = statistics.median(
                agreeing
            )

            deviations = [
                abs(value - refined)
                for value in agreeing
            ]

            mad = (
                statistics.median(
                    deviations
                )
                if deviations
                else 0.0
            )

            stable.append(
                {
                    "bin_index": index,
                    "angle_radians": (
                        index
                        * TAU
                        / self.bin_count
                    ),
                    "range_meters": refined,
                    "observation_count":
                        len(observations),
                    "persistence_ratio": (
                        len(observations)
                        / self.scan_count
                    ),
                    "agreement_ratio":
                        agreement_ratio,
                    "median_absolute_deviation":
                        mad,
                }
            )

        return stable

    def build_candidate(self):
        if self.scan_count < self.minimum_scans:
            raise RuntimeError(
                "Not enough stationary scans: "
                f"{self.scan_count} received, "
                f"{self.minimum_scans} required."
            )

        stable = self.stable_returns()

        if (
            len(stable)
            < self.minimum_stable_bins
        ):
            raise RuntimeError(
                "Not enough stable LiDAR geometry: "
                f"{len(stable)} stable bins, "
                f"{self.minimum_stable_bins} required."
            )

        maximum_observed = max(
            item["range_meters"]
            for item in stable
        )

        half_extent = (
            min(
                maximum_observed,
                self.maximum_map_range_meters,
            )
            + self.map_margin_meters
        )

        half_extent = max(
            1.0,
            math.ceil(
                half_extent
                / self.resolution
            )
            * self.resolution,
        )

        width = int(
            round(
                2.0
                * half_extent
                / self.resolution
            )
        ) + 1

        height = width

        origin_x = -half_extent
        origin_y = -half_extent

        center_x = int(
            round(
                (0.0 - origin_x)
                / self.resolution
            )
        )
        center_y = int(
            round(
                (0.0 - origin_y)
                / self.resolution
            )
        )

        free_cells = set()
        occupied_cells = set()

        for item in stable:
            angle = item[
                "angle_radians"
            ]
            distance = item[
                "range_meters"
            ]

            endpoint_x = (
                distance
                * math.cos(angle)
            )
            endpoint_y = (
                distance
                * math.sin(angle)
            )

            grid_x = int(
                round(
                    (endpoint_x - origin_x)
                    / self.resolution
                )
            )

            grid_y = int(
                round(
                    (endpoint_y - origin_y)
                    / self.resolution
                )
            )

            grid_x = max(
                0,
                min(width - 1, grid_x),
            )

            grid_y = max(
                0,
                min(height - 1, grid_y),
            )

            ray = _bresenham(
                center_x,
                center_y,
                grid_x,
                grid_y,
            )

            for cell in ray[:-1]:
                free_cells.add(cell)

            occupied_cells.add(
                ray[-1]
            )

        # Occupied evidence wins if two rays
        # disagree at an endpoint.
        free_cells.difference_update(
            occupied_cells
        )

        # The robot's current footprint/origin
        # is observed free.
        free_cells.add(
            (center_x, center_y)
        )

        data = [
            -1
            for _ in range(
                width * height
            )
        ]

        for x, y in free_cells:
            data[
                y * width + x
            ] = 0

        for x, y in occupied_cells:
            data[
                y * width + x
            ] = 100

        free_count = sum(
            value == 0
            for value in data
        )

        occupied_count = sum(
            value == 100
            for value in data
        )

        unknown_count = sum(
            value == -1
            for value in data
        )

        return {
            "frame_id": "map",
            "encoding":
                "ros_occupancy_values",
            "resolution":
                self.resolution,
            "width": width,
            "height": height,
            "origin": {
                "x": origin_x,
                "y": origin_y,
                "yaw": 0.0,
            },
            "robot_pose": {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
            },
            "scan_count":
                self.scan_count,
            "angular_bin_count":
                self.bin_count,
            "stable_bin_count":
                len(stable),
            "free_cell_count":
                free_count,
            "occupied_cell_count":
                occupied_count,
            "unknown_cell_count":
                unknown_count,
            "free_value": 0,
            "occupied_value": 100,
            "unknown_value": -1,
            "maximum_map_range_meters":
                self.maximum_map_range_meters,
            "persistence_ratio":
                self.persistence_ratio,
            "stable_ratio":
                self.stable_ratio,
            "base_tolerance_meters":
                self.base_tolerance_meters,
            "relative_tolerance":
                self.relative_tolerance,
            "stable_returns": stable,
            "cells": data,
        }


def write_candidate(
    candidate,
    output_directory,
    name,
    *,
    metadata_extra=None,
):
    directory = Path(
        output_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pgm_path = directory / f"{name}.pgm"
    yaml_path = directory / f"{name}.yaml"
    metadata_path = (
        directory / f"{name}.json"
    )

    for path in (
        pgm_path,
        yaml_path,
        metadata_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite {path}"
            )

    width = int(candidate["width"])
    height = int(candidate["height"])
    cells = candidate["cells"]

    if len(cells) != width * height:
        raise ValueError(
            "Candidate occupancy array size "
            "does not match map dimensions."
        )

    pixels = bytearray()

    # ROS map_server images are conventionally
    # stored with image row zero at the top while
    # occupancy-grid y increases upward.
    for y in range(
        height - 1,
        -1,
        -1,
    ):
        offset = y * width

        for x in range(width):
            value = cells[
                offset + x
            ]

            if value == 100:
                pixels.append(0)
            elif value == 0:
                pixels.append(254)
            else:
                pixels.append(205)

    header = (
        "P5\n"
        "# Mayday stationary LiDAR map candidate\n"
        f"{width} {height}\n"
        "255\n"
    ).encode("ascii")

    with pgm_path.open("wb") as handle:
        handle.write(header)
        handle.write(pixels)

    origin = candidate["origin"]

    yaml_text = (
        f"image: {pgm_path.name}\n"
        "mode: trinary\n"
        f"resolution: {candidate['resolution']:.6f}\n"
        "origin: "
        f"[{origin['x']:.6f}, "
        f"{origin['y']:.6f}, "
        f"{origin['yaw']:.6f}]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    metadata = {
        key: value
        for key, value in candidate.items()
        if key not in (
            "cells",
            "stable_returns",
        )
    }

    metadata["name"] = name
    metadata["source_type"] = (
        "stationary_lidar"
    )
    metadata["image_name"] = (
        pgm_path.name
    )
    metadata["yaml_name"] = (
        yaml_path.name
    )

    if metadata_extra:
        metadata.update(
            metadata_extra
        )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "pgm": str(pgm_path),
        "yaml": str(yaml_path),
        "metadata": str(
            metadata_path
        ),
        "summary": metadata,
    }


def capture_stationary_scans(
    *,
    lidar_url,
    robot_status_url,
    duration_seconds,
    builder,
    lidar_poll_seconds=0.08,
    status_poll_seconds=0.25,
):
    baseline_status = fetch_json(
        robot_status_url
    )

    baseline_motion = motion_signature(
        baseline_status
    )

    started_monotonic = time.monotonic()
    started_wall = time.time()

    last_status_check = (
        started_monotonic
    )

    last_scan_identity = None
    accepted_scans = 0
    transient_lidar_errors = 0

    while (
        time.monotonic()
        - started_monotonic
        < duration_seconds
    ):
        now = time.monotonic()

        if (
            now - last_status_check
            >= status_poll_seconds
        ):
            verify_motion_unchanged(
                fetch_json(
                    robot_status_url
                ),
                baseline_motion,
            )

            last_status_check = now

        try:
            payload = fetch_json(
                lidar_url
            )

            telemetry = payload.get(
                "telemetry"
            )

            if (
                payload.get("ok") is not True
                or not isinstance(
                    telemetry,
                    dict,
                )
                or telemetry.get(
                    "available"
                ) is not True
                or not isinstance(
                    telemetry.get("scan"),
                    dict,
                )
            ):
                raise RuntimeError(
                    "Live LiDAR telemetry "
                    "is unavailable."
                )

            age = _finite_number(
                telemetry.get(
                    "age_seconds"
                )
            )

            if (
                age is None
                or age < 0.0
                or age > 0.75
            ):
                raise RuntimeError(
                    "Live LiDAR scan is stale."
                )

            identity = (
                telemetry.get(
                    "received_at"
                )
                or telemetry[
                    "scan"
                ].get(
                    "stamp_seconds"
                )
            )

            if (
                identity is not None
                and identity
                == last_scan_identity
            ):
                time.sleep(
                    lidar_poll_seconds
                )
                continue

            builder.add_scan(
                telemetry["scan"]
            )

            accepted_scans += 1
            last_scan_identity = identity
            transient_lidar_errors = 0

        except (
            OSError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ):
            transient_lidar_errors += 1

            if transient_lidar_errors > 20:
                raise

        time.sleep(
            lidar_poll_seconds
        )

    verify_motion_unchanged(
        fetch_json(
            robot_status_url
        ),
        baseline_motion,
    )

    return {
        "accepted_scans":
            accepted_scans,
        "started_unix_seconds":
            started_wall,
        "finished_unix_seconds":
            time.time(),
        "motion_verified": True,
        "motion_baseline":
            baseline_motion,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a stationary Mayday "
            "LiDAR occupancy-map candidate."
        )
    )

    parser.add_argument(
        "--lidar-url",
        required=True,
    )

    parser.add_argument(
        "--robot-status-url",
        required=True,
    )

    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--minimum-scans",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--minimum-stable-bins",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--maximum-map-range-meters",
        type=float,
        default=8.0,
    )

    arguments = parser.parse_args()

    if arguments.duration_seconds <= 0:
        parser.error(
            "--duration-seconds must be positive."
        )

    builder = StationaryMapBuilder(
        resolution=arguments.resolution,
        minimum_scans=(
            arguments.minimum_scans
        ),
        minimum_stable_bins=(
            arguments.minimum_stable_bins
        ),
        maximum_map_range_meters=(
            arguments.maximum_map_range_meters
        ),
    )

    capture = capture_stationary_scans(
        lidar_url=arguments.lidar_url,
        robot_status_url=(
            arguments.robot_status_url
        ),
        duration_seconds=(
            arguments.duration_seconds
        ),
        builder=builder,
    )

    candidate = (
        builder.build_candidate()
    )

    result = write_candidate(
        candidate,
        arguments.output_dir,
        arguments.name,
        metadata_extra={
            "capture":
                capture,
            "validated_map_replaced":
                False,
            "cartographer_used":
                False,
            "navigation_used":
                False,
            "motion_commanded":
                False,
        },
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
