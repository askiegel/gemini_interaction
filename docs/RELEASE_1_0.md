# Version 1.0 Release Procedure

## 1. Safety and prerequisites

1. Place Mini Pupper in a clear test area.
2. Confirm a working STOP command and Robot Bridge watchdog.
3. Confirm `.env` is present locally but not tracked by Git.
4. Confirm the Brain PC and robot are on the intended network.

## 2. Static verification

```bash
cd ~/robot_services/cognitive
source .venv/bin/activate
bash scripts/verify_release.sh
```

## 3. Platform startup

```bash
python3 scripts/start_platform.py --check
python3 scripts/start_platform.py --start
```

Verify:

- Runtime API `/health`
- Robot Bridge `/status`
- Camera image
- Vision detections
- Operator Console pages
- STOP behavior

## 4. Operator Console acceptance

Open `http://localhost:8765` and inspect every page:

1. Mission Control
2. Perception
3. World Model
4. Conversation
5. Mission History
6. Diagnostics
7. Network
8. Administration

The Network page may report that interfaces are unmanaged when Ubuntu is using a renderer other than NetworkManager. This is a valid diagnostic state, not a dashboard failure.

## 5. Live acceptance tests

Run one bounded command at a time:

- STOP
- Move forward
- Turn left
- Turn right
- Find backpack
- Follow person

Confirm automatic stop and watchdog behavior after each motion test.

## 6. Git release

```bash
git status --short
git diff --check
git add .
git commit -m "Release cognitive platform v1.0.0"
git tag -a v1.0.0 -m "Mini Pupper Cognitive Robotics Platform v1.0.0"
git push origin main
git push origin v1.0.0
```
