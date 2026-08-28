#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
VOICE_RELAY = ROOT / "voice_relay"

sys.path.insert(
    0,
    str(VOICE_RELAY),
)

from tony2_navigation_runtime import (
    Tony2NavigationRuntime,
)


class Tony2NavigationRuntimeTests(
    unittest.TestCase
):

    def make_runtime(self, root):
        return Tony2NavigationRuntime(
            runtime_dir=root,
            asset_dir=(
                VOICE_RELAY
                / "tony2_navigation_assets"
            ),
            base_environment={
                "ROS_DOMAIN_ID": "99",
                "ROS_LOCALHOST_ONLY": "1",
                "RMW_IMPLEMENTATION": "other",
                "FASTDDS_BUILTIN_TRANSPORTS":
                    "UDPv4",
                "KEEP_ME": "yes",
            },
        )

    def test_guarded_assets_match_expected_hashes(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            self.assertTrue(
                runtime.validate_assets()
            )

    def test_child_environment_preserves_default_fastdds(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            environment = (
                runtime.child_environment()
            )

            self.assertEqual(
                environment[
                    "ROS_DOMAIN_ID"
                ],
                "42",
            )

            self.assertEqual(
                environment[
                    "ROS_LOCALHOST_ONLY"
                ],
                "0",
            )

            self.assertEqual(
                environment[
                    "RMW_IMPLEMENTATION"
                ],
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

    def test_stopped_runtime_has_no_goal_permission(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            status = runtime.status()

            self.assertEqual(
                status["state"],
                "STOPPED",
            )

            self.assertFalse(
                status["running"]
            )

            self.assertFalse(
                status[
                    "goal_submission_enabled"
                ]
            )

            self.assertTrue(
                status[
                    "goal_execution_implemented"
                ]
            )

            self.assertFalse(
                status["recoveries"]
            )

            self.assertFalse(
                status["retries"]
            )

            self.assertEqual(
                status[
                    "maximum_goal_distance_meters"
                ],
                0.50,
            )

            self.assertEqual(
                status[
                    "execution_timeout_seconds"
                ],
                25.0,
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
                    "supervisor": 100,
                    "probe": None,
                },
            ):
                status = runtime.status()

            self.assertEqual(
                status["state"],
                "ERROR",
            )

            self.assertFalse(
                status["running"]
            )

            self.assertFalse(
                status[
                    "goal_submission_enabled"
                ]
            )

    def test_ready_runtime_enables_one_goal(
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
                    "supervisor": 100,
                    "probe": 101,
                    "goal": None,
                },
            ), patch.object(
                runtime,
                "_read_snapshot",
                return_value={
                    "planner_enabled": True,
                    "controller_enabled": True,
                    "navigator_enabled": True,
                    "action_server_ready": True,
                    "transform_ready": True,
                },
            ), patch.object(
                runtime,
                "mapping_status",
                return_value={
                    "running": True,
                    "cartographer": 200,
                    "occupancy_grid": 201,
                },
            ):
                status = runtime.status()

            self.assertEqual(
                status["state"],
                "READY",
            )

            self.assertTrue(
                status[
                    "goal_submission_enabled"
                ]
            )

            self.assertTrue(
                status[
                    "goal_execution_implemented"
                ]
            )

            self.assertFalse(
                status["goal_active"]
            )

    def test_submit_goal_requires_ready_runtime(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "must be READY",
            ):
                runtime.submit_goal(
                    0.1,
                    0.0,
                    0.0,
                )

    def test_submit_goal_rejects_nonfinite_input(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            with self.assertRaisesRegex(
                ValueError,
                "finite",
            ):
                runtime.submit_goal(
                    float("nan"),
                    0.0,
                    0.0,
                )

    def test_submit_goal_uses_guarded_helper(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            process = Mock()
            process.pid = 321
            process.wait.return_value = 0

            ready = {
                "state": "READY",
                "goal_submission_enabled": True,
                "goal_active": False,
            }

            def fake_popen(
                command,
                **_kwargs,
            ):
                runtime.goal_result_file.write_text(
                    (
                        '{"ok":true,'
                        '"distance_meters":0.1}'
                    ),
                    encoding="utf-8",
                )

                process.command = command

                return process

            with patch.object(
                runtime,
                "status",
                return_value=ready,
            ), patch(
                "tony2_navigation_runtime."
                "subprocess.Popen",
                side_effect=fake_popen,
            ):
                result = runtime.submit_goal(
                    0.1,
                    0.0,
                    0.0,
                )

            self.assertEqual(
                result["action"],
                "SUCCEEDED",
            )

            command = process.command

            self.assertIn(
                "tony2_navigation_goal.py",
                " ".join(command),
            )

            self.assertEqual(
                command[
                    command.index(
                        "--max-distance"
                    )
                    + 1
                ],
                "0.5",
            )

            self.assertEqual(
                command[
                    command.index(
                        "--timeout"
                    )
                    + 1
                ],
                "25.0",
            )

    def test_goal_helper_guards_before_submission(
        self,
    ):
        source = (
            VOICE_RELAY
            / "tony2_navigation_goal.py"
        ).read_text(
            encoding="utf-8"
        )

        for required in (
            '"map",',
            '"base_link",',
            "math.hypot(",
            "distance > max_distance",
            "send_goal_async(",
            "cancel_goal_async()",
            "STATUS_SUCCEEDED",
        ):
            self.assertIn(
                required,
                source,
            )

    def test_supervisor_is_navigation_only(
        self,
    ):
        source = (
            VOICE_RELAY
            / "tony2_navigation_supervisor.py"
        ).read_text(
            encoding="utf-8"
        )

        for required in (
            'package="nav2_planner"',
            'package="nav2_controller"',
            'package="nav2_bt_navigator"',
            'package="nav2_lifecycle_manager"',
            '"cmd_vel",',
            '"/cmd_vel",',
            '"/nav_tf",',
            '"autostart": True',
            (
                '"attempt_respawn_reconnection":'
            ),
        ):
            self.assertIn(
                required,
                source,
            )

        for forbidden in (
            'package="nav2_amcl"',
            'package="nav2_map_server"',
            'package="nav2_behaviors"',
            'package="nav2_waypoint_follower"',
            'package="nav2_velocity_smoother"',
            'package="rviz2"',
        ):
            self.assertNotIn(
                forbidden,
                source,
            )

    def test_probe_does_not_shadow_rclpy_clients(
        self,
    ):
        source = (
            VOICE_RELAY
            / "tony2_navigation_probe.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "self._clients =",
            source,
        )

        self.assertIn(
            "self._state_clients =",
            source,
        )

        self.assertIn(
            "for name, client in self._state_clients.items():",
            source,
        )

    def test_guarded_config_keeps_motion_limits(
        self,
    ):
        source = (
            VOICE_RELAY
            / "tony2_navigation_assets"
            / "mayday_guarded_navigation.yaml"
        ).read_text(
            encoding="utf-8"
        )

        for required in (
            "min_vel_x: 0.12",
            "max_vel_x: 0.14",
            "max_speed_xy: 0.14",
            "max_vel_theta: 0.25",
            "xy_goal_tolerance: 0.03",
            "yaw_goal_tolerance: 0.15",
            "required_movement_radius: 0.03",
            "movement_time_allowance: 12.0",
            "allow_unknown: false",
        ):
            self.assertIn(
                required,
                source,
            )

    def test_guarded_tree_has_no_recovery_sequence(
        self,
    ):
        source = (
            VOICE_RELAY
            / "tony2_navigation_assets"
            / "mayday_guarded_navigate_to_pose.xml"
        ).read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "<ComputePathToPose",
            source,
        )

        self.assertIn(
            "<FollowPath",
            source,
        )

        for forbidden in (
            "Recovery",
            "Spin",
            "BackUp",
            "ClearEntireCostmap",
            "<Wait",
        ):
            self.assertNotIn(
                forbidden,
                source,
            )


if __name__ == "__main__":
    unittest.main()
