# Mini Pupper 2 Portable Demo Guide

## Purpose

Run the complete cognitive platform on any compatible shared network without editing IP addresses or source code. Validated on `NETGEAR94` and `iPhone-Tony`. The network must allow direct communication between the Ubuntu PC and Mayday.

## Network Preparation

Mayday uses `minipupperv2.local`. Connect both devices to the same network, power Mayday on, and verify:

```bash
getent ahostsv4 minipupperv2.local
ssh ubuntu@minipupperv2.local
```

For `iPhone-Tony`, enable **Allow Others to Join** and **Maximize Compatibility**, keep the Personal Hotspot screen open, and power-cycle Mayday. Its saved profile is `iPhone-Tony-manual`.

## Start the Platform

```bash
cd ~/robot_services/cognitive
source .venv/bin/activate
python3 scripts/start_platform.py --restart
```

Wait for `SYSTEM READY`, then open `http://localhost:8765`.

Before motion, confirm Runtime, Vision, Robot Bridge, and Voice Relay are green; the live camera is visible; and runtime state is `IDLE`. Keep **STOP ROBOT** ready.

## Validated Demo

1. With live execution disabled, send `What do you see?`.
2. Enable live execution and select `Forward`; confirm automatic stop.
3. Select `Turn Left`; confirm automatic stop.
4. Place a backpack in view and select `Find Backpack`.
5. Stand alone in view and select `Follow Me`.
6. Move left, right, and backward; confirm tracking stays `LOCKED`.
7. Leave view for one or two seconds and return; confirm brief `WAITING_FOR_IDENTITY` followed by reacquisition of the original identity.
8. Select **STOP ROBOT** and confirm `IDLE`, `UNLOCKED`, and zero motion.

## Health and Recovery

```bash
python3 scripts/start_platform.py --check
```

Every component must report `PASS`. After reboot, stale ROS 2, Robot Bridge, and Camera Relay PID files are detected automatically. Recovery messages begin with `STALE:`; manual PID deletion is not required.

The resolved Camera Relay address is passed to the YOLO Vision Server through `VISION_CAMERA_URL`. Mayday's local and fleet identities must both use `minipupperv2.local`.

If Mayday leaves the iPhone hotspot, keep the hotspot screen open, power-cycle Mayday, wait for the hostname to resolve, and run `--restart` again.

## Completion Criteria

The demo is ready when startup prints `SYSTEM READY`, all health items pass, camera detections update, bounded motions stop automatically, backpack detection works, FOLLOW_PERSON preserves identity, STOP cancels motion, and the Git worktree is clean.
