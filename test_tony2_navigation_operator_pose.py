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
            '"/reinitialize_global_localization"',
            '"/request_nomotion_update"',
            '"/tony2_nav_scan"',
            '"/map"',
            "covariance_tight",
            "global_search_completed",
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
        )

        for marker in forbidden:
            self.assertNotIn(
                marker,
                source,
            )


if __name__ == "__main__":
    unittest.main()


def test_initial_pose_helper_accepts_ros_arguments():
    """The helper must strip ROS args before argparse."""

    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent

    helper = (
        root
        / "voice_relay"
        / "tony2_navigation_initial_pose.py"
    )

    environment = dict(os.environ)

    # This invokes argparse --help only. It does not create a
    # ROS node, contact AMCL, set a pose, or command motion.
    process = subprocess.run(
        [
            "/usr/bin/python3",
            str(helper),
            "--help",
            "--ros-args",
            "-r",
            "/tf:=/nav_tf",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        timeout=10.0,
        check=False,
    )

    assert process.returncode == 0, process.stdout

    assert (
        "unrecognized arguments"
        not in process.stdout
    )

    helper_source = helper.read_text(
        encoding="utf-8"
    )

    assert (
        "from rclpy.utilities "
        "import remove_ros_args"
        in helper_source
    )

    assert (
        "application_args = remove_ros_args("
        in helper_source
    )

    assert (
        "parser.parse_args("
        in helper_source
    )

    # ROS arguments must remain available to rclpy itself.
    assert "rclpy.init()" in helper_source


def test_initial_pose_alignment_does_not_double_penalize_unknown_space():
    """Coverage failures must not become synthetic geometry errors."""

    import re
    from pathlib import Path

    helper = (
        Path(__file__).resolve().parent
        / "voice_relay"
        / "tony2_navigation_initial_pose.py"
    )

    source = helper.read_text(
        encoding="utf-8"
    )

    compact = re.sub(
        r"\s+",
        " ",
        source,
    )

    assert (
        "endpoint_errors.append( MAX_ENDPOINT_ERROR_METERS )"
        not in compact
    )

    assert (
        "sample_count = len( valid_returns )"
        in compact
    )

    assert (
        "mean_endpoint_error = ( sum( endpoint_errors ) / len( endpoint_errors ) )"
        in compact
    )

    assert (
        "known_ratio = ( coverage_known_count / coverage_sample_count )"
        in compact
    )

    assert (
        "inside_ratio = ( coverage_inside_count / coverage_sample_count )"
        in compact
    )

def test_initial_pose_within_10cm_uses_geometric_sample_denominator():
    """10-cm geometry uses known-space endpoints."""

    import re
    from pathlib import Path

    helper = (
        Path(__file__).resolve().parent
        / "voice_relay"
        / "tony2_navigation_initial_pose.py"
    )

    source = helper.read_text(
        encoding="utf-8"
    )

    compact = re.sub(
        r"\s+",
        " ",
        source,
    )

    assert (
        "geometric_sample_count = len( endpoint_errors )"
        in compact
    )

    assert (
        "within_10cm_ratio = ( within_10cm_count / geometric_sample_count )"
        in compact
    )

    assert (
        "within_10cm_ratio = ( within_10cm_count / sample_count )"
        not in compact
    )

    assert '"geometric_sample_count": geometric_sample_count' in compact

    assert "MIN_WITHIN_10CM_RATIO = 0.35" in source

def test_initial_pose_coverage_uses_all_valid_lidar_returns():
    """Coverage must use every valid LiDAR return."""

    import re
    from pathlib import Path

    helper = (
        Path(__file__).resolve().parent
        / "voice_relay"
        / "tony2_navigation_initial_pose.py"
    )

    source = helper.read_text(
        encoding="utf-8"
    )

    compact = re.sub(
        r"\s+",
        " ",
        source,
    )

    assert (
        "coverage_returns = list( valid_returns )"
        in compact
    )

    assert (
        "coverage_sample_count = len( coverage_returns )"
        in compact
    )

    assert (
        "known_ratio = ( coverage_known_count / coverage_sample_count )"
        in compact
    )

    assert (
        "inside_ratio = ( coverage_inside_count / coverage_sample_count )"
        in compact
    )

    # Expensive nearest-wall geometry remains bounded.
    assert (
        "valid_returns = evenly_sample( valid_returns, 120, )"
        in compact
    )

    # Coverage thresholds remain unchanged.
    assert "MIN_KNOWN_RATIO = 0.55" in source
    assert "MIN_INSIDE_RATIO = 0.75" in source

def test_initial_pose_requires_stationary_multi_scan_confirmation():
    """Trust requires repeated fresh scan agreement."""

    from pathlib import Path

    helper = (
        Path(__file__).resolve().parent
        / "voice_relay"
        / "tony2_navigation_initial_pose.py"
    )

    source = helper.read_text(
        encoding="utf-8"
    )

    assert "SCAN_CONFIRMATION_SAMPLES = 5" in source

    assert (
        "SCAN_CONFIRMATION_REQUIRED_PASSES = 3"
        in source
    )

    scan_loop = (
        "for confirmation_index in range(\n"
        "            SCAN_CONFIRMATION_SAMPLES\n"
        "        )"
    )

    assert scan_loop in source

    assert (
        "confirmation_pass_count = sum("
        in source
    )

    majority_gate = (
        "confirmation_pass_count\n"
        "            >= SCAN_CONFIRMATION_REQUIRED_PASSES"
    )

    assert majority_gate in source

    assert '"scan_confirmation": {' in source
    assert '"required_pass_count":' in source

    sample_result = (
        '"samples":\n'
        "                    scan_confirmation_samples"
    )

    assert sample_result in source

    # Each scan still uses all original thresholds.
    assert "MAX_MEAN_ENDPOINT_ERROR_METERS = 0.18" in source
    assert "MIN_WITHIN_10CM_RATIO = 0.35" in source
    assert "MIN_KNOWN_RATIO = 0.55" in source
    assert "MIN_INSIDE_RATIO = 0.75" in source

    trust_gate = (
        "trusted = (\n"
        "            covariance_tight\n"
        "            and alignment_good\n"
        "        )"
    )

    assert trust_gate in source

    # Stationary-only safety contract remains.
    assert '"navigation_goal_executed":' in source
    assert '"motion_enabled":' in source
