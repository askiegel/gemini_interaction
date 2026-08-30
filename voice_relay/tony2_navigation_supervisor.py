#!/usr/bin/env python3

"""
Headless guarded Nav2 composition on isolated Zenoh.

The navigation graph is domain 43 / rmw_zenoh_cpp.

Exactly one domain-42 Fast DDS source reads the five
navigation inputs and transfers them through localhost TCP.

Controller output is deliberately disconnected from
Mayday during this validation feature.
"""

import argparse

from pathlib import Path

from launch import LaunchDescription
from launch import LaunchService
from launch.actions import ExecuteProcess
from launch.actions import TimerAction
from launch_ros.actions import Node


ROUTER_BINARY = (
    "/opt/ros/humble/lib/"
    "rmw_zenoh_cpp/rmw_zenohd"
)

ROUTER_OVERRIDE = (
    'listen/endpoints=["tcp/127.0.0.1:7447"];'
    'scouting/multicast/enabled=false;'
    'transport/shared_memory/enabled=false'
)


def build_launch_description(
    asset_dir,
):
    asset_dir = Path(
        asset_dir
    ).resolve()

    module_dir = Path(
        __file__
    ).resolve().parent

    isolation_source = (
        module_dir
        / "tony2_navigation_isolation_source.py"
    )

    isolation_sink = (
        module_dir
        / "tony2_navigation_isolation_sink.py"
    )

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

    for path in (
        isolation_source,
        isolation_sink,
        params,
        navigate_tree,
        disabled_through_tree,
        Path(ROUTER_BINARY),
    ):
        if not path.is_file():
            raise RuntimeError(
                "Required isolated navigation "
                f"asset is missing: {path}"
            )

    navigation_remap = [
        (
            "/tf",
            "/nav_tf",
        ),
        (
            "/scan",
            "/tony2_nav_scan",
        ),
        (
            "/odom",
            "/tony2_nav_odom",
        ),
    ]

    router = ExecuteProcess(
        cmd=[
            ROUTER_BINARY,
        ],
        additional_env={
            "ROS_DOMAIN_ID": "43",
            "ZENOH_CONFIG_OVERRIDE":
                ROUTER_OVERRIDE,
        },
        output="screen",
    )

    isolation_ingress = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    "-u",
                    str(isolation_sink),
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "/usr/bin/env",
                    "-u",
                    "ZENOH_CONFIG_OVERRIDE",
                    "-u",
                    "CYCLONEDDS_URI",
                    "-u",
                    "ROS_DISCOVERY_SERVER",
                    "-u",
                    "ROS_SUPER_CLIENT",
                    "-u",
                    "FASTRTPS_DEFAULT_PROFILES_FILE",
                    "-u",
                    "FASTDDS_DEFAULT_PROFILES_FILE",
                    "-u",
                    "FASTDDS_BUILTIN_TRANSPORTS",
                    "ROS_DOMAIN_ID=42",
                    "ROS_LOCALHOST_ONLY=0",
                    (
                        "RMW_IMPLEMENTATION="
                        "rmw_fastrtps_cpp"
                    ),
                    "/usr/bin/python3",
                    "-u",
                    str(isolation_source),
                ],
                output="screen",
            ),
        ],
    )

    delayed_navigation = TimerAction(
        period=4.0,
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
                remappings=navigation_remap,
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
                    navigation_remap
                    + [
                        (
                            "cmd_vel",
                            (
                                "/tony2_nav_"
                                "cmd_vel_blocked"
                            ),
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
                            str(
                                disabled_through_tree
                            ),
                    },
                ],
                remappings=navigation_remap,
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
            router,
            isolation_ingress,
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
