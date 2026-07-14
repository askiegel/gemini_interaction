# Mini Pupper 2 Cognitive Platform

A modular cognitive robotics platform for the Mini Pupper 2 combining natural-language commands, computer vision, persistent missions, a shared World Model, and safe ROS 2 robot control.

## Project Overview

The Ubuntu PC acts as the cognitive brain, while the Mini Pupper 2 acts as the robot body.

The platform separates:

- Human interaction
- Language understanding
- Mission scheduling
- Perception
- World-state management
- Behavior execution
- Robot control
- ROS 2 hardware access

This separation makes the system easier to test, operate, and extend with future capabilities such as person following, LiDAR navigation, SLAM, semantic mapping, and multi-robot coordination.

## Current Capabilities

- Browser voice command input
- Gemini-based intent recognition
- Mission creation and queuing
- Persistent runtime execution
- STOP preemption
- Shared persistent World Model
- Continuous YOLO object detection
- Vision Service updates
- Robot Bridge HTTP control
- Camera Relay
- Runtime REST API
- Operator Dashboard
- Persistent `FIND_OBJECT`
- Automated regression testing

## Architecture

    Human Operator
           |
           v
    Browser Voice Relay
           |
           v
    Voice Command Interface
           |
           v
    Gemini Provider
           |
           v
    Intent Parser
           |
           v
    Mission Manager
           |
           v
    Persistent Runtime
           |
           +----------------------+
           |                      |
           v                      v
    Behavior Manager        Vision Service
           |                      |
           v                      v
    Robot Bridge Client     Shared World Model
           |
           v
    Robot Bridge Server
           |
           v
         ROS 2
           |
           v
    Mini Pupper 2 Hardware

## Design Principles

- The Runtime owns mission scheduling.
- The Mission Manager owns mission state.
- The World Model is the single source of truth for perception.
- The Vision Service continuously updates the World Model.
- Behaviors perform one bounded action per execution.
- Behaviors do not contain internal loops.
- STOP preempts all active and queued missions.
- ROS 2 remains isolated on the Mini Pupper.
- One focused feature should be completed per commit.
- Regression tests must pass before committing.

## Deployment

### Ubuntu PC — Cognitive Brain

Project directory:

    ~/robot_services/cognitive

Runs:

- Gemini provider
- Intent parser
- Mission manager
- Persistent runtime
- Runtime API
- Vision Server
- Vision Service
- Shared World Model
- Browser Voice Relay
- Operator Dashboard
- Robot Bridge client

### Mini Pupper 2 — Robot Body

Project directory:

    ~/robot_bridge

Runs:

- ROS 2 Humble
- Mini Pupper bringup
- Quadruped controller
- Camera
- LiDAR
- IMU
- Robot Bridge server
- Camera Relay

## Service Ports

| Service | Host | Port |
|---|---|---:|
| Vision Server | Ubuntu PC | 8000 |
| Browser Voice Relay | Ubuntu PC | 8765 |
| Runtime API | Ubuntu PC | 8770 |
| Robot Bridge | Mini Pupper | 8090 |
| Camera Relay | Mini Pupper | 8091 |

## Repository Layout

    cognitive/
    ├── behavior_manager.py
    ├── runtime.py
    ├── runtime_api.py
    ├── mission_manager.py
    ├── mission_types.py
    ├── voice_command.py
    ├── world/
    ├── vision/
    ├── providers/
    ├── robot_bridge/
    ├── voice_relay/
    ├── scripts/
    ├── docs/
    └── test_*.py

The exact layout may expand as new capabilities are added.

## Quick Start

Enter the project:

    cd ~/robot_services/cognitive

Activate the Python environment:

    source .venv/bin/activate

Set the required service URLs:

    export ROBOT_BRIDGE_URL=http://ROBOT_IP:8090
    export CAMERA_RELAY_URL=http://ROBOT_IP:8091/camera/latest.jpg
    export VISION_SERVER_URL=http://127.0.0.1:8000/detections/latest

Replace `ROBOT_IP` with the current Mini Pupper IP address.

Follow the complete startup procedure:

    docs/STARTUP.md

The target automated startup command is:

    python3 scripts/start_platform.py

Expected result:

    ====================
    SYSTEM READY
    ====================

## Health Checks

Robot Bridge:

    curl http://ROBOT_IP:8090/status

Camera Relay:

    curl --output /tmp/mini_pupper_camera.jpg       http://ROBOT_IP:8091/camera/latest.jpg

Vision Server:

    curl http://127.0.0.1:8000/detections/latest

Runtime API:

    curl http://127.0.0.1:8770/health

Runtime status:

    curl http://127.0.0.1:8770/status

## Supported Missions

Currently supported or partially supported:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`
- `FIND_OBJECT`
- `FOLLOW_PERSON`
- `DESCRIBE_SCENE`
- `RETURN_HOME`
- `UNKNOWN`

Persistent `FIND_OBJECT` is currently implemented.

Persistent `FOLLOW_PERSON` is the next major autonomy capability.

## STOP Behavior

STOP is the highest-priority command.

A successful STOP:

1. Sends a stop command to the Mini Pupper.
2. Cancels the active mission.
3. Clears queued missions.
4. Prevents further behavior execution.
5. Returns the Runtime to `STOPPED`.

STOP must be tested before autonomous motion.

## Testing

Activate the environment:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

Run the available regression tests:

    python3 test_world_model.py
    python3 test_entity_registry.py
    python3 test_vision_adapter.py
    python3 test_motion_intents.py
    python3 test_cognitive_runtime.py
    python3 test_behavior_world_model.py
    python3 test_find_object_behavior.py
    python3 test_find_object_persistent.py
    python3 test_runtime_mission_executor.py

All regression tests should pass before committing changes.

## Documentation

| Document | Purpose |
|---|---|
| `docs/SESSION_CHECKLIST.md` | Start-of-session and end-of-session checks |
| `docs/STARTUP.md` | Complete platform startup and shutdown procedure |
| `docs/OPERATIONS.md` | Daily operation, testing, Git, and troubleshooting |
| `docs/ARCHITECTURE.md` | System components, responsibilities, and data flow |
| `docs/ROADMAP.md` | Completed milestones and planned capabilities |

## Development Workflow

For each feature:

1. Verify platform health.
2. Run existing tests.
3. Make one focused change.
4. Run syntax checks.
5. Run targeted tests.
6. Run the full regression suite.
7. Review the Git diff.
8. Stage only intended files.
9. Commit with a clear message.
10. Push to GitHub.
11. Update documentation.

Avoid using `git add .` unless every modified file has been reviewed.

## Roadmap

Near-term priorities:

1. Complete one-command startup automation.
2. Add clean shutdown and process cleanup.
3. Implement persistent `FOLLOW_PERSON`.
4. Add the Dashboard STOP button.
5. Integrate LiDAR obstacle avoidance.
6. Implement Return Home.
7. Add patrol missions.
8. Add SLAM and localization.
9. Add semantic mapping.
10. Add robot-to-robot communication.
11. Add fleet coordination.

See:

    docs/ROADMAP.md

## Related Repositories

Related components include:

- Mini Pupper Robot Bridge
- YOLO Vision Server
- Mini Pupper ROS 2 packages
- Qwen robot experiments
- Duckiebot development repository

Repository links should be added once each related repository is verified and documented.

## Security

Do not commit:

- Gemini API keys
- Passwords
- Wi-Fi credentials
- SSH private keys
- Access tokens
- Machine-specific secrets

Use environment variables or local configuration files excluded by Git.

## Project Status

The project has reached a stable architectural milestone.

The core cognitive pipeline, perception path, robot-control bridge, mission runtime, STOP preemption, and persistent object search are operational.

Current development is focused on:

- Startup automation
- Reliable daily operation
- Persistent person following
- LiDAR navigation
- Future SLAM and multi-robot capabilities
