#!/usr/bin/env python3

"""Offline tests for validated operator AMCL initialization."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent

VOICE = (
    ROOT
    / "voice_relay"
)

sys.path.insert(
    0,
    str(VOICE),
)

from tony2_navigation_runtime import Tony2NavigationRuntime


HELPER = (
    VOICE
    / "tony2_navigation_initial_pose.py"
)


class OperatorPoseRuntimeTests(
    unittest.TestCase
):
    def make_runtime(self):
        directory = (
            tempfile.TemporaryDirectory()
        )

        self.addCleanup(
            directory.cleanup
        )

        return Tony2NavigationRuntime(
            runtime_dir=directory.name,
            robot_bridge_url=(
                "http://192.168.68.124:8090"
            ),
        )

    @staticmethod
    def preinit_status():
        return {
            "state": "STARTING",
            "running": True,
            "map_server_enabled": True,
            "localization_enabled": True,
            "transform_ready": False,
            "goal_submission_enabled": False,
            "goal_active": False,
            "motion_output_connected": False,
            "motion_egress_ready": True,
            "motion_egress_idle": True,
        }

    @staticmethod
    def ready_status():
        return {
            "state": "READY",
            "running": True,
            "map_server_enabled": True,
            "localization_enabled": True,
            "transform_ready": True,
            "goal_submission_enabled": True,
            "goal_active": False,
            "motion_output_connected": False,
            "motion_egress_ready": True,
            "motion_egress_idle": True,
        }

    def test_trusted_operator_pose_reaches_ready(
        self,
    ):
        runtime = self.make_runtime()

        helper_result = {
            "ok": True,
            "trusted": True,
            "diagnostic": {
                "covariance_tight": True,
                "seed_consistent": True,
                "alignment_good": True,
                "trusted": True,
            },
        }

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "helper log\n"
                + json.dumps(
                    helper_result
                )
                + "\n"
            ),
        )

        with patch.object(
            runtime,
            "status",
            side_effect=[
                self.preinit_status(),
                self.ready_status(),
            ],
        ), patch(
            "tony2_navigation_runtime.subprocess.run",
            return_value=completed,
        ) as run:
            result = (
                runtime.initialize_operator_pose(
                    0.5,
                    0.2,
                    0.75,
                )
            )

        self.assertEqual(
            result["action"],
            "OPERATOR_POSE_VALIDATED",
        )

        self.assertTrue(
            result[
                "localization"
            ][
                "trusted"
            ]
        )

        command = (
            run.call_args.args[0]
        )

        self.assertIn(
            str(HELPER),
            command,
        )

        self.assertIn(
            "--x=0.5",
            command,
        )

        self.assertIn(
            "--y=0.2",
            command,
        )

        self.assertIn(
            "/tf:=/nav_tf",
            command,
        )

    def test_untrusted_pose_stops_runtime(
        self,
    ):
        runtime = self.make_runtime()

        helper_result = {
            "ok": True,
            "trusted": False,
            "diagnostic": {
                "covariance_tight": False,
                "seed_consistent": True,
                "alignment_good": False,
                "trusted": False,
            },
        }

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                json.dumps(
                    helper_result
                )
                + "\n"
            ),
        )

        stopped = {
            "action": "STOPPED",
            "navigation": {
                "state": "STOPPED",
                "running": False,
            },
        }

        with patch.object(
            runtime,
            "status",
            return_value=(
                self.preinit_status()
            ),
        ), patch(
            "tony2_navigation_runtime.subprocess.run",
            return_value=completed,
        ), patch.object(
            runtime,
            "stop",
            return_value=stopped,
        ) as stop:
            result = (
                runtime.initialize_operator_pose(
                    0.5,
                    0.2,
                    0.75,
                )
            )

        stop.assert_called_once_with()

        self.assertEqual(
            result["action"],
            "OPERATOR_POSE_REJECTED",
        )

        self.assertEqual(
            result["navigation"]["state"],
            "STOPPED",
        )

    def test_pose_outside_fixed_map_is_rejected(
        self,
    ):
        runtime = self.make_runtime()

        with self.assertRaisesRegex(
            ValueError,
            "outside",
        ):
            runtime.initialize_operator_pose(
                100.0,
                0.0,
                0.0,
            )

    def test_existing_localization_cannot_be_overwritten(
        self,
    ):
        runtime = self.make_runtime()

        state = (
            self.preinit_status()
        )

        state[
            "transform_ready"
        ] = True

        with patch.object(
            runtime,
            "status",
            return_value=state,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "already initialized",
            ):
                runtime.initialize_operator_pose(
                    0.5,
                    0.2,
                    0.0,
                )

    def test_helper_is_stationary_only(
        self,
    ):
        source = HELPER.read_text(
            encoding="utf-8"
        )

        required = (
            "SetInitialPose",
            '"/set_initial_pose"',
            '"/request_nomotion_update"',
            '"/tony2_nav_scan"',
            '"/map"',
            "covariance_tight",
            "seed_consistent",
            "alignment_good",
            '"motion_enabled":',
        )

        for marker in required:
            self.assertIn(
                marker,
                source,
            )

        forbidden = (
            "NavigateToPose",
            "geometry_msgs.msg.Twist",
            "/cmd_vel",
            "MotionArmLease",
            "reinitialize_global_localization",
        )

        for marker in forbidden:
            self.assertNotIn(
                marker,
                source,
            )


if __name__ == "__main__":
    unittest.main()
