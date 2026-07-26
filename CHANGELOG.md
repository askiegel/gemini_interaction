# Changelog

## Unreleased - 2026-07-25

### Persistent person identity

- Stabilized identity across transient World Model entity changes.
- Ordered Entity Registry resolution before identity assignment.
- Deduplicated person detections after registry resolution.
- Preserved TargetLock identity through occlusion and reacquisition.
- Verified repeated waiting-to-locked recovery with zero identity changes.

### Execution ownership

- Routed backward-compatible `--execute` through the persistent Runtime API.
- Prevented command execution from creating a second live VisionAdapter.

## v1.0.0 — 2026-07-23

First stable release of the Mini Pupper 2 Cognitive Robotics Platform.

### Operator Console

- Mission Control, Perception, World Model, Conversation, Mission History, Diagnostics, Network, and Administration pages.
- Persistent mission history with filtering and detail inspection.
- World Model Explorer with entity and observation history.
- Live service and host diagnostics.
- Read-only NetworkManager visibility with explicit unmanaged-interface reporting.
- Central configuration editor and validated configuration API.

### Runtime and robotics

- Persistent cognitive runtime and REST execution boundary.
- Mission queue, STOP preemption, deterministic behavior execution, and Robot Bridge control.
- Streaming FOLLOW_PERSON controller, target lock, prediction, recovery, and identity telemetry.
- Continuous vision updates and persistent World Model state.
- Gemini-backed conversation and structured intent validation.

### Release hardening

- Removed credentials, temporary state, locks, and development backup files from the release archive.
- Added `.env.example` and release verification scripts.
- Added a full Version 1.0 operations and release checklist.
