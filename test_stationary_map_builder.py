#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Tony Kiegel
#

import json
import math
from pathlib import Path

import pytest

from voice_relay.stationary_map_builder import (
    StationaryMapBuilder,
    motion_signature,
    verify_motion_unchanged,
    write_candidate,
)


TAU = 2.0 * math.pi


def make_scan(
    *,
    count=500,
    distance=2.0,
    replacements=None,
):
    ranges = [
        float(distance)
        for _ in range(count)
    ]

    if replacements:
        for index, value in (
            replacements.items()
        ):
            ranges[index % count] = value

    return {
        "angle_min": 0.0,
        "angle_max": TAU,
        "angle_increment":
            TAU / count,
        "range_min": 0.02,
        "range_max": 25.0,
        "ranges": ranges,
        "sample_count": count,
        "valid_sample_count":
            sum(
                value is not None
                for value in ranges
            ),
        "frame_id": "lidar_link",
    }


def zero_status(
    *,
    last_command_at=None,
    watchdog_stop_count=0,
):
    return {
        "status": "READY",
        "ros_ready": True,
        "motion": {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "streaming": False,
            "last_command_at":
                last_command_at,
            "watchdog_stop_count":
                watchdog_stop_count,
        },
    }


def test_stationary_geometry_becomes_stable():
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=100,
    )

    for _ in range(20):
        builder.add_scan(
            make_scan()
        )

    stable = builder.stable_returns()

    assert builder.scan_count == 20
    assert builder.bin_count == 500
    assert len(stable) >= 495

    for item in stable[:20]:
        assert item[
            "range_meters"
        ] == pytest.approx(
            2.0
        )

        assert item[
            "persistence_ratio"
        ] == pytest.approx(
            1.0
        )


def test_transient_obstacle_is_rejected():
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=100,
        stable_ratio=0.75,
    )

    transient_bin = 50

    for index in range(20):
        replacement = (
            1.0
            if index < 6
            else 2.0
        )

        builder.add_scan(
            make_scan(
                replacements={
                    transient_bin:
                        replacement,
                }
            )
        )

    stable_bins = {
        item["bin_index"]
        for item in (
            builder.stable_returns()
        )
    }

    assert transient_bin not in stable_bins
    assert len(stable_bins) >= 490


def test_499_and_500_sample_revolutions_align():
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=80,
    )

    for index in range(20):
        builder.add_scan(
            make_scan(
                count=(
                    499
                    if index % 2
                    else 500
                )
            )
        )

    candidate = (
        builder.build_candidate()
    )

    assert candidate[
        "scan_count"
    ] == 20

    assert candidate[
        "stable_bin_count"
    ] >= 450


def test_candidate_contains_ros_occupancy_values():
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=100,
        resolution=0.05,
    )

    for _ in range(20):
        builder.add_scan(
            make_scan()
        )

    candidate = (
        builder.build_candidate()
    )

    assert candidate[
        "frame_id"
    ] == "map"

    assert candidate[
        "encoding"
    ] == "ros_occupancy_values"

    assert candidate[
        "resolution"
    ] == pytest.approx(
        0.05
    )

    assert candidate[
        "robot_pose"
    ] == {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
    }

    assert set(
        candidate["cells"]
    ) <= {-1, 0, 100}

    assert candidate[
        "free_cell_count"
    ] > 100

    assert candidate[
        "occupied_cell_count"
    ] > 100

    assert candidate[
        "unknown_cell_count"
    ] > 0


def test_insufficient_scans_fail_closed():
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=10,
    )

    for _ in range(10):
        builder.add_scan(
            make_scan()
        )

    with pytest.raises(
        RuntimeError,
        match="Not enough stationary scans",
    ):
        builder.build_candidate()


def test_motion_signature_requires_zero():
    baseline = motion_signature(
        zero_status()
    )

    assert baseline == {
        "last_command_at": None,
        "watchdog_stop_count": 0,
    }

    moving = zero_status()
    moving["motion"][
        "linear_x"
    ] = 0.12

    with pytest.raises(
        RuntimeError,
        match="nonzero linear",
    ):
        motion_signature(moving)


def test_motion_history_change_fails_stationary_check():
    baseline = motion_signature(
        zero_status(
            last_command_at="before"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="motion history changed",
    ):
        verify_motion_unchanged(
            zero_status(
                last_command_at="after"
            ),
            baseline,
        )


def test_candidate_writer_emits_map_server_files(
    tmp_path,
):
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=100,
    )

    for _ in range(20):
        builder.add_scan(
            make_scan()
        )

    candidate = (
        builder.build_candidate()
    )

    result = write_candidate(
        candidate,
        tmp_path,
        "stationary_test",
        metadata_extra={
            "motion_verified": True,
        },
    )

    pgm = Path(result["pgm"])
    yaml = Path(result["yaml"])
    metadata = Path(
        result["metadata"]
    )

    assert pgm.exists()
    assert yaml.exists()
    assert metadata.exists()

    assert pgm.read_bytes().startswith(
        b"P5\n"
    )

    yaml_text = yaml.read_text(
        encoding="utf-8"
    )

    assert (
        "image: stationary_test.pgm"
        in yaml_text
    )

    assert (
        "resolution: 0.050000"
        in yaml_text
    )

    assert "mode: trinary" in yaml_text

    metadata_payload = json.loads(
        metadata.read_text(
            encoding="utf-8"
        )
    )

    assert metadata_payload[
        "source_type"
    ] == "stationary_lidar"

    assert metadata_payload[
        "motion_verified"
    ] is True

    assert "cells" not in metadata_payload


def test_writer_refuses_overwrite(tmp_path):
    builder = StationaryMapBuilder(
        minimum_scans=20,
        minimum_stable_bins=100,
    )

    for _ in range(20):
        builder.add_scan(
            make_scan()
        )

    candidate = (
        builder.build_candidate()
    )

    write_candidate(
        candidate,
        tmp_path,
        "stationary_test",
    )

    with pytest.raises(
        FileExistsError
    ):
        write_candidate(
            candidate,
            tmp_path,
            "stationary_test",
        )
