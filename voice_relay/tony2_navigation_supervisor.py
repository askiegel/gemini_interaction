#!/usr/bin/env python3

"""Headless guarded Nav2 composition for Tony2."""

import argparse
from pathlib import Path

from launch import LaunchDescription
from launch import LaunchService
from launch.actions import ExecuteProcess
from launch.actions import TimerAction
from launch_ros.actions import Node


def build_launch_description(asset_dir):
    asset_dir = Path(asset_dir).resolve()

    params = (
        asset_dir
        / "mayday_guarded_navigation.yaml"
    )

    navigate_tree = (
        asset_dir
        / "mayday_guarded_navigate_to_pose.xml"
    )

    disabled_through_tree = (
        asset_dir
        / "mayday_disabled_navigate_through_poses.xml"
    )

    tf_relay = (
        asset_dir
        / "latest_tf_relay.py"
    )

    for path in (
        params,
        navigate_tree,
        disabled_through_tree,
        tf_relay,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Required guarded asset is missing: {path}"
            )

    navigation_tf_remap = [
        (
            "/tf",
            "/nav_tf",
        ),
    ]

    relay = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            "-u",
            str(tf_relay),
            "--ros-args",
            "-r",
            "__node:=mapping_navigation_tf_relay",
            "-p",
            "input_topic:=/mayday_navigation_tf",
            "-p",
            "output_topic:=/nav_tf",
            "-p",
            "publish_frequency:=10.0",
        ],
        output="screen",
    )

    delayed_navigation = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[
                    str(params),
                    {
                        "use_sim_time": False,
                    },
                ],
                remappings=navigation_tf_remap,
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[
                    str(params),
                    {
                        "use_sim_time": False,
                    },
                ],
                remappings=(
                    navigation_tf_remap
                    + [
                        (
                            "cmd_vel",
                            "/cmd_vel",
                        ),
                    ]
                ),
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[
                    str(params),
                    {
                        "use_sim_time": False,
                        "default_nav_to_pose_bt_xml":
                            str(navigate_tree),
                        "default_nav_through_poses_bt_xml":
                            str(disabled_through_tree),
                    },
                ],
                remappings=navigation_tf_remap,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name=(
                    "lifecycle_manager_"
                    "mapping_navigation"
                ),
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "autostart": True,
                        "bond_timeout": 4.0,
                        "attempt_respawn_reconnection":
                            False,
                        "node_names": [
                            "planner_server",
                            "controller_server",
                            "bt_navigator",
                        ],
                    },
                ],
            ),
        ],
    )

    return LaunchDescription(
        [
            relay,
            delayed_navigation,
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--asset-dir",
        required=True,
    )

    args = parser.parse_args()

    launch_service = LaunchService()

    launch_service.include_launch_description(
        build_launch_description(
            args.asset_dir
        )
    )

    return launch_service.run()


if __name__ == "__main__":
    raise SystemExit(main())
