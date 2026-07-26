# Mini Pupper 2 Cognitive Platform Roadmap

This document tracks completed milestones and planned development for the Mini Pupper 2 Cognitive Platform.

## Guiding Principles

- Preserve the current architecture.
- Keep the World Model as the single source of truth.
- Let the Runtime own mission scheduling and repetition.
- Keep each behavior action bounded.
- Maintain immediate STOP preemption.
- Complete one focused feature per commit.
- Keep regression tests passing.

## Current Milestone

Completed:

- [x] Robot Bridge
- [x] Camera Relay
- [x] YOLO Vision Server
- [x] Vision Service
- [x] Shared World Model
- [x] Persistent Cognitive Runtime
- [x] Runtime API
- [x] Browser Voice Relay
- [x] Operator Dashboard
- [x] Mission Queue
- [x] STOP preemption
- [x] Persistent FIND_OBJECT
- [x] Initial project documentation

## Phase 1 — Platform Startup

Goal: Start and verify the platform with one command.

- [x] Complete `scripts/start_platform.py`
- [x] Verify Mini Pupper connectivity
- [x] Verify Robot Bridge
- [x] Verify Camera Relay
- [x] Start Vision Server when needed
- [x] Start Vision Service
- [x] Start Runtime API
- [x] Start Browser Voice Relay
- [x] Run service health checks
- [x] Print `SYSTEM READY`
- [ ] Add clean shutdown support
- [x] Add structured service logs

Target command:

    python3 scripts/start_platform.py

Expected result:

    ====================
    SYSTEM READY
    ====================

## Phase 2 — Persistent FOLLOW_PERSON

Mission lifecycle:

    Acquire Person
          |
          v
    Center Person
          |
          v
    Maintain Distance
          |
          v
        Repeat
          |
          v
         STOP

Tasks:

- [x] Read the latest person observation from the World Model
- [x] Turn toward a person positioned left or right
- [x] Move forward when the person is centered and too far away
- [x] Stop or hold position at the target distance
- [x] Reacquire a temporarily lost person
- [x] Preserve one bounded action per runtime cycle
- [x] Add STOP-preemption tests
- [x] Add persistent mission regression tests

## Phase 3 — Operator Experience

- [x] Dashboard STOP button
- [x] Platform health indicators
- [x] Active mission display
- [x] Mission queue display
- [x] Mission history
- [x] World Model visualization
- [x] Camera and detection preview
- [ ] Battery monitoring

## Phase 4 — LiDAR Navigation

- [ ] Confirm reliable LiDAR data
- [ ] Add LiDAR health checks
- [ ] Add obstacle-distance observations
- [ ] Add collision prevention
- [ ] Add safe local navigation
- [ ] Add waypoint support

## Phase 5 — Return Home and Patrol

- [ ] Define and save a home position
- [ ] Implement RETURN_HOME
- [ ] Add named locations
- [ ] Add patrol routes
- [ ] Add scheduled missions
- [ ] Add battery-aware return behavior

## Phase 6 — SLAM

- [ ] Generate maps
- [ ] Save and load maps
- [ ] Localize within a saved map
- [ ] Integrate navigation planning
- [ ] Validate multi-room operation
- [ ] Document SLAM startup and shutdown

## Phase 7 — Semantic Mapping

- [ ] Associate detected objects with locations
- [ ] Store persistent landmarks
- [ ] Add room identification
- [ ] Add named semantic locations
- [ ] Answer object-location questions
- [ ] Support commands such as `Find the backpack in the office`

## Phase 8 — Human Interaction

- [ ] Improve scene descriptions
- [ ] Add conversational memory
- [ ] Investigate pose estimation
- [x] Implement persistent person identity
- [ ] Add gesture understanding
- [x] Implement identity-only lost-person recovery

## Phase 9 — Robot-to-Robot Communication

- [ ] Robot discovery
- [ ] Robot identity
- [ ] Robot-to-robot mission requests
- [ ] Shared observations
- [ ] Cooperative object search
- [ ] Voice-mediated robot communication

Example:

    Robot A: Search the kitchen for the backpack.
    Robot B: Searching the kitchen.

## Phase 10 — Fleet Coordination

- [ ] Fleet manager
- [ ] Shared mission queue
- [ ] Distributed perception
- [ ] Shared semantic map
- [ ] Multi-robot task assignment
- [ ] Conflict and collision avoidance
- [ ] Fleet health dashboard

## Documentation

Maintain these documents as the platform changes:

- `README.md`
- `docs/STARTUP.md`
- `docs/OPERATIONS.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/SESSION_CHECKLIST.md`

Documentation must distinguish between:

- Verified current behavior
- Work in progress
- Planned future capability

## Release Requirements

Before marking a release stable:

- [ ] All regression tests pass
- [ ] Startup instructions are verified
- [ ] STOP works reliably
- [ ] Robot Bridge and Camera Relay are healthy
- [ ] Vision updates the World Model
- [ ] Runtime API responds
- [ ] A bounded motion test passes
- [ ] A persistent FIND_OBJECT test passes
- [ ] Documentation is current
- [ ] Changes are committed and pushed

## Immediate Next Tasks

1. Add LiDAR observations to the World Model.
2. Add deterministic collision-prevention checks.
3. Add LiDAR health to startup diagnostics.
4. Define safe local-navigation boundaries.
5. Preserve the verified identity and TargetLock contracts.

Persistent identity, occlusion recovery, one-command startup, streaming
FOLLOW_PERSON, and the Operator Console STOP path are hardware-verified.
