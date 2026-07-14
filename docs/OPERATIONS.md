# Mini Pupper 2 Cognitive Platform Operations Guide

This document defines the daily operating, development, testing, troubleshooting, and shutdown workflow for the Mini Pupper 2 Cognitive Platform.

## 1. Operating Principles

The platform follows these rules:

- The Runtime owns mission scheduling.
- The World Model is the single source of truth for perception.
- The Vision Service continuously updates the World Model.
- `BehaviorManager` performs one bounded action per execution.
- Behaviors do not contain internal control loops.
- STOP preempts all other missions.
- ROS 2 remains isolated on the Mini Pupper.
- One feature should be completed per commit.
- Regression tests must pass before committing.

## 2. System Roles

### Ubuntu PC

The Ubuntu PC runs the cognitive and perception services:

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

Project directory:

    ~/robot_services/cognitive

### Mini Pupper 2

The robot runs:

- ROS 2 Humble
- Mini Pupper bringup
- Quadruped controller
- Camera driver
- LiDAR driver
- IMU driver
- Robot Bridge
- Camera Relay

Robot Bridge directory:

    ~/robot_bridge

## 3. Start-of-Session Workflow

At the beginning of every session:

1. Review `docs/SESSION_CHECKLIST.md`.
2. Power on the Ubuntu PC.
3. Power on the Mini Pupper.
4. Confirm both systems are on the same network.
5. Start ROS 2 bringup on the Mini Pupper.
6. Start the Robot Bridge.
7. Start the Camera Relay.
8. Start the Vision Server.
9. Start the Vision Service.
10. Start the Runtime API.
11. Start the Browser Voice Relay.
12. Run health checks.
13. Test STOP before any autonomous motion.
14. Run regression tests before editing code.

## 4. Activate the Development Environment

On the Ubuntu PC:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

Confirm Python:

    which python3
    python3 --version

Confirm Git state:

    git status
    git branch --show-current

Expected branch for stable development:

    main

Create a feature branch only when the change is large enough to justify isolation.

## 5. Required Environment Variables

Typical Ubuntu PC variables:

    export ROBOT_BRIDGE_URL=http://ROBOT_IP:8090
    export CAMERA_RELAY_URL=http://ROBOT_IP:8091/camera/latest.jpg
    export VISION_SERVER_URL=http://127.0.0.1:8000/detections/latest

Gemini configuration:

    export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'

Never commit API keys, passwords, tokens, or local secrets.

Typical Mini Pupper ROS variables:

    export ROS_DISTRO=humble
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0

Verify them with:

    env | grep -E 'ROS_DISTRO|ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY'

## 6. Daily Development Workflow

Use the following sequence for each feature:

1. Confirm the platform is healthy.
2. Run existing tests.
3. Review the relevant files.
4. Make one focused change.
5. Run syntax checks.
6. Run targeted tests.
7. Run the full regression suite.
8. Review the Git diff.
9. Stage only intended files.
10. Commit with a clear message.
11. Push to GitHub.
12. Update documentation.

Avoid mixing unrelated changes in one commit.

## 7. Syntax Checks

Run Python compilation checks before tests:

    python3 -m py_compile \
      behavior_manager.py \
      runtime.py \
      runtime_api.py \
      mission_manager.py

Add other changed Python files to the command as needed.

A successful syntax check produces no output.

## 8. Regression Testing

Run the available test files from the project root.

Example:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

    python3 test_world_model.py
    python3 test_entity_registry.py
    python3 test_vision_adapter.py
    python3 test_motion_intents.py
    python3 test_cognitive_runtime.py
    python3 test_behavior_world_model.py
    python3 test_find_object_behavior.py
    python3 test_find_object_persistent.py
    python3 test_runtime_mission_executor.py

The exact test list may grow over time.

All tests must pass before committing.

## 9. Safe Motion Testing

Before motion tests:

- Place the robot on the floor.
- Provide clear space around the robot.
- Keep the robot within reach.
- Confirm battery level.
- Confirm STOP works.
- Avoid testing near stairs or table edges.

Run a short bounded command first.

Example:

    Move forward

Then immediately test:

    Stop

Confirm the robot stops and the runtime returns to `STOPPED`.

## 10. Mission Testing

### Motion Mission

Verify:

- Command parses correctly.
- Mission activates.
- One bounded motion is sent.
- Automatic stop occurs.
- Mission completes.

### FIND_OBJECT

Verify:

- Target label is normalized.
- Vision Service updates the World Model.
- Runtime repeats bounded actions.
- Target observations remain recent.
- Mission completes when the object is found.
- STOP cancels the mission immediately.

### FOLLOW_PERSON

When implemented, verify:

- Person acquisition.
- Horizontal centering.
- Distance maintenance.
- Repeated bounded corrections.
- Loss-of-target recovery.
- STOP preemption.

## 11. Service Health Checks

### Robot Bridge

    curl http://ROBOT_IP:8090/status

### Camera Relay

    curl --output /tmp/mini_pupper_camera.jpg \
      http://ROBOT_IP:8091/camera/latest.jpg

    file /tmp/mini_pupper_camera.jpg

### Vision Server

    curl http://127.0.0.1:8000/detections/latest

### Runtime API

    curl http://127.0.0.1:8770/health
    curl http://127.0.0.1:8770/status

### Browser Voice Relay

Open:

    http://127.0.0.1:8765

## 12. Git Workflow

Review current state:

    git status
    git diff

Stage only the files for the current feature.

Example:

    git add \
      behavior_manager.py \
      runtime.py \
      test_find_object_persistent.py

Review staged files:

    git diff --cached --name-only

Review staged changes:

    git diff --cached

Check for whitespace errors:

    git diff --cached --check

Commit:

    git commit -m "Describe the completed feature"

Push:

    git push origin main

Do not use `git add .` unless every modified file has been reviewed and belongs in the same commit.

## 13. Commit Message Guidelines

Use short, action-oriented messages.

Good examples:

    Add persistent find-object mission execution
    Add startup and operations documentation
    Add runtime STOP preemption
    Add person-follow behavior tests

Avoid vague messages such as:

    updates
    changes
    fix stuff
    work

## 14. Documentation Workflow

Update documentation whenever behavior or startup procedures change.

Relevant files:

    docs/STARTUP.md
    docs/OPERATIONS.md
    docs/ARCHITECTURE.md
    docs/ROADMAP.md
    docs/SESSION_CHECKLIST.md
    README.md

Documentation should reflect the current verified platform, not future assumptions.

Future work should be clearly marked as planned.

## 15. Troubleshooting Workflow

When a failure occurs:

1. Stop robot motion.
2. Identify which layer failed.
3. Check the service process.
4. Check the listening port.
5. Check environment variables.
6. Check logs or terminal output.
7. Run the smallest relevant test.
8. Avoid changing multiple layers simultaneously.
9. Confirm the fix with a regression test.
10. Document recurring failures.

Troubleshoot from the bottom upward:

    Hardware
    ↓
    ROS 2
    ↓
    Robot Bridge
    ↓
    Camera Relay
    ↓
    Vision Server
    ↓
    Vision Service
    ↓
    World Model
    ↓
    Runtime
    ↓
    Voice and Dashboard

## 16. Common Diagnostic Commands

### Running Python Processes

    ps aux | grep python3

### Listening Ports

    ss -ltnp

Specific ports:

    ss -ltnp | grep 8090
    ss -ltnp | grep 8091
    ss -ltnp | grep 8000
    ss -ltnp | grep 8765
    ss -ltnp | grep 8770

### Find a Process Using a Port

    sudo lsof -i :PORT_NUMBER

### ROS 2 Nodes

    ros2 node list

### ROS 2 Topics

    ros2 topic list

### Motion Topic

    ros2 topic info /cmd_vel

### Camera Topic

    ros2 topic info /image_raw
    ros2 topic hz /image_raw

### Recent Git History

    git log --oneline -10

## 17. Port Conflict Procedure

When a service reports that its port is already in use:

1. Identify the process:

       sudo lsof -i :PORT_NUMBER

2. Confirm whether it is a valid running service.
3. Reuse the valid service if healthy.
4. Stop only stale or duplicate processes.
5. Restart the intended service.
6. Re-run its health check.

Do not kill processes without confirming their purpose.

## 18. Failed Test Procedure

If a regression test fails:

1. Read the first failure carefully.
2. Run only the failed test again.
3. Check whether the failure is deterministic.
4. Review the most recent code change.
5. Inspect the relevant Git diff.
6. Restore the last known-good behavior if necessary.
7. Do not commit until the full suite passes.

## 19. World Model Integrity

The World Model must remain the single source of truth for perception.

Behaviors should not query the Vision Server directly when the World Model already provides the required observation.

Verify:

- Entity timestamps are current.
- Labels are normalized.
- Confidence values are preserved.
- Position data is consistent.
- Stale observations are rejected.
- Persistence does not overwrite newer data.

## 20. STOP Procedure

STOP is the highest-priority command.

A successful STOP must:

1. Send zero motion or the stop endpoint to the robot.
2. Cancel the active mission.
3. Clear queued missions.
4. Prevent further mission execution.
5. Return runtime state to `STOPPED`.

Manual Robot Bridge stop:

    curl -X POST http://ROBOT_IP:8090/stop

If STOP does not work, end the session and correct it before further motion testing.

## 21. End-of-Session Workflow

Before ending a session:

1. Stop the robot.
2. Run regression tests.
3. Review Git status.
4. Review all modified files.
5. Commit completed work.
6. Push to GitHub.
7. Update documentation.
8. Record the next task in `docs/ROADMAP.md`.
9. Shut down services cleanly.
10. Power off the robot safely.

## 22. Shutdown Order

Stop services in this order:

1. Browser Voice Relay
2. Runtime API
3. Vision Service
4. Vision Server
5. Camera Relay
6. Robot Bridge
7. ROS 2 bringup

Use `Ctrl+C` in each active terminal.

Confirm no motion remains before powering off the Mini Pupper.

## 23. Release Procedure

Before identifying a version as stable:

1. Run all regression tests.
2. Run a live STOP test.
3. Run a live bounded-motion test.
4. Run a live `FIND_OBJECT` test.
5. Verify all services start from documentation.
6. Review configuration and environment variables.
7. Update README and documentation.
8. Commit all release files.
9. Create a Git tag.
10. Push the tag to GitHub.

Example:

    git tag -a v0.1.0 -m "Mini Pupper cognitive platform milestone"
    git push origin v0.1.0

## 24. Configuration Management

Keep machine-specific configuration out of source code where possible.

Use:

- Environment variables
- Local `.env` files excluded by Git
- Configuration modules with safe defaults
- Documented example configuration files

Never commit:

- API keys
- Passwords
- Personal network credentials
- Private SSH keys
- Access tokens

## 25. Future Operations Expansion

This guide will later include:

- Automated startup and shutdown
- LiDAR health checks
- SLAM startup
- Navigation stack startup
- Map selection
- Return-home validation
- Patrol operations
- Multi-robot discovery
- Robot-to-robot communication
- Fleet startup and shutdown
- Release rollback procedures
