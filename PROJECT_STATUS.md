# Mini Pupper 2 Cognitive Robotics Platform

## Current Version

v1.7-alpha

## Current Milestone

Robot Bridge Server

## Last Verified Hardware Configuration

Mini Pupper Workspace:
~/ros2_ws

Bringup:

ros2 launch mini_pupper_bringup bringup.launch.py

Verified Motion Topic:

/cmd_vel

Verified Controller:

/quadruped_controller_node

## Development Rules

- One feature per commit.
- Build → Test → Commit → Push.
- ROS2 remains isolated.
- World Model remains the single source of truth.
- Provider-agnostic AI architecture.
- No architectural redesign without discussion.

## Before Every Session

Read:

PROJECT_STATUS.md

Then complete:

docs/checklists/SESSION_START.md

## Next Goal

Implement the Robot Bridge Server on the Mini Pupper.

The Robot Bridge Server will expose:

- GET /status
- POST /motion
- POST /stop

and publish motion commands locally to /cmd_vel.

## End-to-End Demonstration Goal

Find my backpack.
