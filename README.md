# Mini Pupper 2 Cognitive Robotics Platform

**Version 1.0.0**

A modular cognitive robotics platform that combines natural-language interaction, computer vision, persistent missions, a shared World Model, streaming person following, and safe ROS 2 control for Mini Pupper 2.

## Architecture

```text
Human Operator
  → Browser Voice Relay
  → Conversation Manager
  → Gemini Provider
  → Runtime API
  → Mission Manager
  → Behavior Manager
  → Robot Bridge
  → ROS 2
  → Mini Pupper 2
```

The Vision Service continuously updates the World Model, which is the authoritative perception and memory source used by behaviors and the Operator Console.

## Version 1.0 capabilities

- Browser voice and typed commands
- Structured Gemini intent validation
- Persistent runtime and mission queue
- Immediate STOP preemption
- Motion, FIND_OBJECT, FOLLOW_PERSON, scene description, and return-home mission support
- Streaming follow controller with persistent identity, target lock, prediction, occlusion recovery, and identity telemetry
- Single-runtime execution ownership for dashboard, `--runtime`, and backward-compatible `--execute` commands
- YOLO-based perception and persistent World Model
- Robot Bridge and camera relay integration
- Central configuration management
- Operator Console pages:
  - Mission Control
  - Perception
  - World Model
  - Conversation
  - Mission History
  - Diagnostics
  - Network
  - Administration
- Read-only NetworkManager diagnostics
- Automated regression tests

## Hosts and ports

| Service | Host | Port |
|---|---|---:|
| Vision Server | Brain PC | 8000 |
| Browser Voice Relay / Operator Console | Brain PC | 8765 |
| Runtime API | Brain PC | 8770 |
| Robot Bridge | Mini Pupper | 8090 |
| Camera Relay | Mini Pupper | 8091 |

## Installation

```bash
cd ~/robot_services/cognitive
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Enter your Gemini key in `.env`. Never commit or distribute that file.

Review `config/system_config.json`, then start and verify the platform:

```bash
python3 scripts/start_platform.py --check
python3 scripts/start_platform.py --start
```

Open the Operator Console at:

```text
http://localhost:8765
```

## Verification

Run the release verification script:

```bash
bash scripts/verify_release.sh
```

Hardware-dependent and explicitly live tests are not run automatically. Follow `docs/STARTUP.md` and `docs/OPERATIONS.md` for live robot verification.

## Safety

- Keep the physical emergency stop path available during live testing.
- Test with the robot raised or in a clear area before enabling locomotion.
- STOP must preempt active and queued missions.
- Network controls are intentionally read-only in Version 1.0 because the dashboard has no authentication layer.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/PERSISTENT_IDENTITY.md`
- `docs/STARTUP.md`
- `docs/PORTABLE_DEMO.md`
- `docs/OPERATIONS.md`
- `docs/SESSION_CHECKLIST.md`
- `docs/ROADMAP.md`
- `PROJECT_STATUS.md`
- `CHANGELOG.md`
