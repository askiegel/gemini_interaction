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
            robot_bridge_url=(
                "http://robot.invalid:8090"
            ),
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

    def test_child_environment_uses_isolated_zenoh(
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
                "43",
            )

            self.assertEqual(
                environment[
                    "RMW_IMPLEMENTATION"
                ],
                "rmw_zenoh_cpp",
            )

            self.assertEqual(
                environment[
                    "ZENOH_CONFIG_OVERRIDE"
                ],
                runtime.ZENOH_SESSION_OVERRIDE,
            )

            self.assertNotIn(
                "ROS_LOCALHOST_ONLY",
                environment,
            )

            for removed in (
                "FASTDDS_BUILTIN_TRANSPORTS",
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                "FASTDDS_DEFAULT_PROFILES_FILE",
                "ROS_DISCOVERY_SERVER",
                "ROS_SUPER_CLIENT",
                "CYCLONEDDS_URI",
            ):
                self.assertNotIn(
                    removed,
                    environment,
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

    def test_ready_runtime_allows_go_only_when_egress_idle(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            runtime.motion_egress_status_file.write_text(
                (
                    '{"running":true,'
                    '"armed":false,'
                    '"token":null}'
                ),
                encoding="utf-8",
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
                status["motion_egress_ready"]
            )

            self.assertTrue(
                status["motion_egress_idle"]
            )

            self.assertTrue(
                status[
                    "goal_submission_enabled"
                ]
            )

            self.assertFalse(
                status[
                    "motion_output_connected"
                ]
            )

    def test_ready_runtime_blocks_go_without_idle_egress(
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

            self.assertFalse(
                status["motion_egress_ready"]
            )

            self.assertFalse(
                status["motion_egress_idle"]
            )

            self.assertFalse(
                status[
                    "goal_submission_enabled"
                ]
            )

            self.assertFalse(
                status[
                    "motion_output_connected"
                ]
            )

    def test_armed_unknown_token_blocks_new_go(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            runtime.motion_egress_status_file.write_text(
                (
                    '{"running":true,'
                    '"armed":true,'
                    '"token":"unexpected-token"}'
                ),
                encoding="utf-8",
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

            self.assertTrue(
                status["motion_egress_ready"]
            )

            self.assertFalse(
                status["motion_egress_idle"]
            )

            self.assertFalse(
                status[
                    "goal_submission_enabled"
                ]
            )

            self.assertFalse(
                status[
                    "motion_output_connected"
                ]
            )

    def test_motion_output_requires_matching_owned_token(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            token = "a" * 48

            with runtime._motion_lock:
                runtime._active_motion_lease = object()
                runtime._active_motion_token = token

            runtime.motion_egress_status_file.write_text(
                (
                    '{"running":true,'
                    '"armed":true,'
                    f'"token":"{token}"'
                    '}'
                ),
                encoding="utf-8",
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

            self.assertTrue(
                status[
                    "motion_output_connected"
                ]
            )

            self.assertFalse(
                status[
                    "goal_submission_enabled"
                ]
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
            process.poll.return_value = 0

            lease = Mock()
            token = "b" * 48
            generation = 0

            ready = {
                "state": "READY",
                "goal_submission_enabled": True,
                "goal_active": False,
                "motion_output_connected": False,
            }

            armed = {
                "state": "READY",
                "goal_submission_enabled": False,
                "goal_active": False,
                "motion_output_connected": True,
            }

            final = {
                "state": "READY",
                "goal_submission_enabled": True,
                "goal_active": False,
                "motion_output_connected": False,
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
                side_effect=[
                    ready,
                    armed,
                    final,
                ],
            ), patch.object(
                runtime,
                "_begin_motion_lease",
                return_value=(
                    lease,
                    token,
                    generation,
                ),
            ), patch.object(
                runtime,
                "_wait_for_motion_arm_ack",
                return_value=True,
            ), patch.object(
                runtime,
                "_motion_lease_is_current",
                return_value=True,
            ), patch.object(
                runtime,
                "_release_motion_lease",
                return_value=True,
            ), patch.object(
                runtime,
                "_wait_for_motion_disarm",
                return_value=True,
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

    def test_begin_and_release_motion_lease_keeps_ownership_until_stop(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            lease = Mock()
            token = "c" * 48

            lease.start.return_value = token

            ready = {
                "state": "READY",
                "goal_submission_enabled": True,
            }

            with patch.object(
                runtime,
                "status",
                return_value=ready,
            ), patch(
                "tony2_navigation_runtime."
                "MotionArmLease",
                return_value=lease,
            ):
                (
                    returned_lease,
                    returned_token,
                    generation,
                ) = runtime._begin_motion_lease()

            self.assertIs(
                returned_lease,
                lease,
            )

            self.assertEqual(
                returned_token,
                token,
            )

            self.assertEqual(
                generation,
                0,
            )

            self.assertIs(
                runtime._active_motion_lease,
                lease,
            )

            self.assertEqual(
                runtime._active_motion_token,
                token,
            )

            ownership_during_stop = []

            def lease_stop():
                ownership_during_stop.append(
                    runtime._active_motion_lease
                    is lease
                )

            lease.stop.side_effect = lease_stop

            self.assertTrue(
                runtime._release_motion_lease(
                    lease
                )
            )

            self.assertEqual(
                ownership_during_stop,
                [True],
            )

            self.assertIsNone(
                runtime._active_motion_lease
            )

            self.assertIsNone(
                runtime._active_motion_token
            )

    def test_stop_stops_lease_before_goal_process(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            lease = Mock()

            goal_process = Mock()
            goal_process.pid = 333

            ordering = []

            def lease_stop():
                self.assertIs(
                    runtime._active_motion_lease,
                    lease,
                )

                ordering.append(
                    "lease_stop"
                )

            lease.stop.side_effect = lease_stop

            def fake_terminate(
                pid,
                marker,
            ):
                ordering.append(
                    (
                        "terminate",
                        pid,
                        marker,
                    )
                )

                return True

            with runtime._motion_lock:
                runtime._active_motion_lease = lease
                runtime._active_motion_token = (
                    "d" * 48
                )
                runtime._goal_process = goal_process

            with patch.object(
                runtime,
                "_wait_for_motion_disarm",
                return_value=True,
            ), patch.object(
                runtime,
                "_runtime_pids",
                return_value={
                    "supervisor": 100,
                    "probe": 101,
                    "goal": 333,
                },
            ), patch.object(
                runtime,
                "_terminate_group",
                side_effect=fake_terminate,
            ), patch.object(
                runtime,
                "status",
                return_value={
                    "state": "STOPPED",
                },
            ):
                runtime.stop()

            self.assertEqual(
                ordering[0],
                "lease_stop",
            )

            self.assertEqual(
                ordering[1][0],
                "terminate",
            )

            self.assertEqual(
                ordering[1][1],
                333,
            )

            self.assertEqual(
                runtime._stop_generation,
                1,
            )

            self.assertIsNone(
                runtime._active_motion_lease
            )

            self.assertIsNone(
                runtime._goal_process
            )

    def test_lost_lease_prevents_goal_popen(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            lease = Mock()

            ready = {
                "state": "READY",
                "goal_submission_enabled": True,
                "goal_active": False,
                "motion_output_connected": False,
            }

            with patch.object(
                runtime,
                "status",
                return_value=ready,
            ), patch.object(
                runtime,
                "_begin_motion_lease",
                return_value=(
                    lease,
                    "e" * 48,
                    0,
                ),
            ), patch.object(
                runtime,
                "_wait_for_motion_arm_ack",
                return_value=True,
            ), patch.object(
                runtime,
                "_motion_lease_is_current",
                return_value=False,
            ), patch.object(
                runtime,
                "_release_motion_lease",
                return_value=False,
            ), patch(
                "tony2_navigation_runtime."
                "subprocess.Popen",
            ) as popen:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "cancelled before goal launch",
                ):
                    runtime.submit_goal(
                        0.1,
                        0.0,
                        0.0,
                    )

            popen.assert_not_called()

    def test_stop_generation_invalidates_prior_go(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.make_runtime(
                Path(directory)
            )

            lease = object()
            token = "f" * 48

            with runtime._motion_lock:
                runtime._active_motion_lease = lease
                runtime._active_motion_token = token

            self.assertTrue(
                runtime._motion_lease_is_current(
                    lease,
                    token,
                    0,
                )
            )

            with runtime._motion_lock:
                runtime._stop_generation += 1

            self.assertFalse(
                runtime._motion_lease_is_current(
                    lease,
                    token,
                    0,
                )
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

    def test_supervisor_is_isolated_navigation_only(
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
            '"cmd_vel_egress"',
            '"/nav_tf",',
            '"autostart": True',
            (
                '"attempt_respawn_reconnection":'
            ),
            "rmw_zenoh_cpp/rmw_zenohd",
            "tony2_navigation_isolation_source.py",
            "tony2_navigation_isolation_sink.py",
            '"ROS_DOMAIN_ID=42"',
            '"ROS_DOMAIN_ID": "43"',
        ):
            self.assertIn(
                required,
                source,
            )

        self.assertNotIn(
            '"/cmd_vel",',
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
