# Mini Pupper 2 Cognitive Robotics Platform

## Current Version

**v1.0.0 — Stable**

## Release Status

The cognitive platform and Operator Console are feature-complete for the Version 1.0 baseline.

Completed subsystems:

- Conversation Manager and Gemini provider
- Runtime API and persistent Cognitive Runtime
- Mission Manager and deterministic Behavior Manager
- Robot Bridge client integration
- Vision Service, target tracking, and persistent World Model
- Streaming FOLLOW_PERSON control with persistent identity, target lock, predictive recovery, and identity-only reacquisition
- Single-runtime command execution for dashboard, `--runtime`, and backward-compatible `--execute` paths
- Operator Console: Mission Control, Perception, World Model, Conversation, Mission History, Diagnostics, Network, and Administration
- Central configuration and read-only network diagnostics
- Automated regression tests and startup verification

## Stable Architecture

Human → Browser Voice Relay → Conversation Manager → Gemini Provider → Runtime API → Mission Manager → Behavior Manager → Robot Bridge → ROS 2 → Mini Pupper 2

The World Model remains the single source of truth for persistent perceived entities.

## Repository Locations

- Brain PC: `~/robot_services/cognitive`
- Mini Pupper: `~/robot_bridge`
- Branch: `main`
- Release tag: `v1.0.0`

## Version 1.0 Boundaries

Wi-Fi visibility is read-only. Connecting, disconnecting, forgetting profiles, and changing credentials remain disabled until dashboard authentication is implemented.

LiDAR navigation, SLAM, semantic mapping, and multi-robot fleet support are post-v1.0 roadmap items.

## Development Rules

- One feature per commit.
- Build → Test → Commit → Push.
- Keep ROS 2 isolated on the robot.
- Keep the World Model as the single source of truth.
- Never commit `.env`, credentials, runtime state, lock files, or generated logs.
