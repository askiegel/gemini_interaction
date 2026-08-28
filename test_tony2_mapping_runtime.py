#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
VOICE_RELAY = ROOT / "voice_relay"

sys.path.insert(
    0,
    str(VOICE_RELAY),
)

from tony2_mapping_runtime import Tony2MappingRuntime


class Tony2MappingRuntimeTests(unittest.TestCase):

    def make_runtime(self, root):
        return Tony2MappingRuntime(
            home=root / "home",
            runtime_dir=root,
            base_environment={
                "ROS_DOMAIN_ID": "99",
                "ROS_LOCALHOST_ONLY": "1",
                "RMW_IMPLEMENTATION": "other",
                "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
                "KEEP_ME": "yes",
            },
        )

    def test_child_environment_uses_default_fastdds_transports(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            environment = runtime.child_environment()

            self.assertEqual(
                environment["ROS_DOMAIN_ID"],
                "42",
            )

            self.assertEqual(
                environment["ROS_LOCALHOST_ONLY"],
                "0",
            )

            self.assertEqual(
                environment["RMW_IMPLEMENTATION"],
                "rmw_fastrtps_cpp",
            )

            self.assertNotIn(
                "FASTDDS_BUILTIN_TRANSPORTS",
                environment,
            )

            self.assertEqual(
                environment["KEEP_ME"],
                "yes",
            )

    def test_stopped_runtime_never_exposes_cached_map(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            runtime.snapshot_file.write_text(
                json.dumps(
                    {
                        "telemetry": {
                            "available": True,
                            "status": "READY",
                            "map": {
                                "width": 10,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            status_code, payload = (
                runtime.live_map_status()
            )

            self.assertEqual(
                status_code,
                503,
            )

            self.assertFalse(
                payload["ok"]
            )

            self.assertFalse(
                payload["runtime_active"]
            )

            self.assertIsNone(
                payload["telemetry"]["map"]
            )

            self.assertEqual(
                payload["telemetry"]["status"],
                "MAPPING_STOPPED",
            )

    def test_running_runtime_preserves_live_map_schema(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            pids = {
                "cartographer": 100,
                "occupancy_grid": 101,
                "telemetry_probe": 102,
            }

            runtime.snapshot_file.write_text(
                json.dumps(
                    {
                        "telemetry": {
                            "available": True,
                            "status": "READY",
                            "received_at": (
                                "2026-08-28T00:00:00+00:00"
                            ),
                            "age_seconds": 0.2,
                            "error": None,
                            "map": {
                                "frame_id": "map",
                                "width": 20,
                                "height": 30,
                                "resolution": 0.05,
                                "origin": {
                                    "x": -1.0,
                                    "y": -2.0,
                                    "yaw": 0.0,
                                },
                                "cells": [
                                    -1,
                                    0,
                                    100,
                                ],
                            },
                        },
                        "readiness": {
                            "available": True,
                            "status": "BUILDING_SUBMAPS",
                            "ready": False,
                            "submap_count": 1,
                            "mature_submap_count": 0,
                            "minimum_submap_count": 3,
                            "minimum_mature_submap_count": 2,
                            "minimum_mature_version": 100,
                            "submap_progress": 1 / 3,
                            "mature_submap_progress": 0.0,
                            "submaps": [
                                {
                                    "trajectory_id": 0,
                                    "index": 0,
                                    "version": 20,
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                runtime,
                "_runtime_pids",
                return_value=pids,
            ):
                status_code, payload = (
                    runtime.live_map_status()
                )

            self.assertEqual(
                status_code,
                200,
            )

            self.assertTrue(
                payload["ok"]
            )

            self.assertTrue(
                payload["runtime_active"]
            )

            self.assertEqual(
                payload["topic"],
                "/map",
            )

            self.assertTrue(
                payload["read_only"]
            )

            self.assertFalse(
                payload["authoritative"]
            )

            self.assertEqual(
                payload[
                    "telemetry"
                ]["map"]["width"],
                20,
            )

            mapping = payload["mapping"]

            self.assertEqual(
                mapping["host"],
                "Tony2",
            )

            self.assertTrue(
                mapping["running"]
            )

            self.assertTrue(
                mapping["owned"]
            )

            self.assertFalse(
                mapping["planning_enabled"]
            )

            self.assertFalse(
                mapping["validated_map_mutable"]
            )

    def test_partial_runtime_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            with patch.object(
                runtime,
                "_runtime_pids",
                return_value={
                    "cartographer": 100,
                    "occupancy_grid": None,
                    "telemetry_probe": None,
                },
            ):
                mapping = runtime.status()

            self.assertEqual(
                mapping["state"],
                "ERROR",
            )

            self.assertFalse(
                mapping["running"]
            )

            self.assertFalse(
                mapping["owned"]
            )

    def test_reset_is_stop_then_start(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            calls = []

            def fake_stop():
                calls.append("stop")

                return {
                    "action": "STOPPED",
                }

            def fake_start():
                calls.append("start")

                return {
                    "action": "STARTED",
                    "mapping": {
                        "state": "RUNNING",
                    },
                }

            with patch.object(
                runtime,
                "stop",
                side_effect=fake_stop,
            ), patch.object(
                runtime,
                "start",
                side_effect=fake_start,
            ):
                result = runtime.reset()

            self.assertEqual(
                calls,
                [
                    "stop",
                    "start",
                ],
            )

            self.assertEqual(
                result["action"],
                "RESET",
            )


if __name__ == "__main__":
    unittest.main()
