#!/usr/bin/env bash
set -e

echo "=== Mini Pupper Cognitive ROS2 Bridge Motion Test ==="

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo

python3 -m ros2_bridge.motion_test
