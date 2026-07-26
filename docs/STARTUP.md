# Mini Pupper 2 Cognitive Platform Startup Guide

This guide starts the complete Mini Pupper 2 Cognitive Platform from a powered-off state.

## System Roles

### Ubuntu PC — Cognitive Brain

The Ubuntu PC runs:

- YOLO Vision Server
- Vision Service
- Shared World Model
- Persistent Cognitive Runtime
- Runtime API
- Browser Voice Relay
- Operator Dashboard
- Gemini provider and intent processing

Project directory:

    ~/robot_services/cognitive

### Mini Pupper 2 — Robot Body

The Mini Pupper runs:

- ROS 2 Humble
- Mini Pupper bringup
- Quadruped controller
- Camera
- LiDAR
- IMU
- Robot Bridge
- Camera Relay

Robot Bridge directory:

    ~/robot_bridge

## Network Requirements

Before startup:

- The Ubuntu PC and Mini Pupper must be on the same network.
- The Mini Pupper IP address must be known.
- Required ports must be reachable.

Default service ports:

| Service | Host | Port |
|---|---|---:|
| Vision Server | Ubuntu PC | 8000 |
| Browser Voice Relay | Ubuntu PC | 8765 |
| Runtime API | Ubuntu PC | 8770 |
| Robot Bridge | Mini Pupper | 8090 |
| Camera Relay | Mini Pupper | 8091 |

Replace `ROBOT_IP` in the commands below with the current Mini Pupper IP address.

Example:

    192.168.68.101

## 1. Power On the Hardware

1. Power on the Ubuntu PC.
2. Power on the Mini Pupper.
3. Confirm that the Mini Pupper completes its boot process.
4. Confirm that both systems are connected to the same network.
5. Place the robot on the floor with clear space around it.

## 2. Verify Mini Pupper Connectivity

From the Ubuntu PC:

    ping -c 4 ROBOT_IP

Expected result:

- Packets are received.
- Packet loss is zero or minimal.

Test SSH:

    ssh ubuntu@ROBOT_IP

Exit the SSH session when finished:

    exit

## 3. Start ROS 2 Bringup

Open a terminal and connect to the Mini Pupper:

    ssh ubuntu@ROBOT_IP

Set the ROS 2 environment:

    export ROS_DISTRO=humble
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0

Start bringup:

    ros2 launch mini_pupper_bringup bringup.launch.py

Keep this terminal open.

Verify that the following components start:

- `/quadruped_controller_node`
- `/robot_state_publisher`
- `/servo_interface`
- `/v4l2_camera`
- `/state_estimation_node`
- `/LD06`
- `/imu_interface`

In another Mini Pupper terminal, verify the nodes:

    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    ros2 node list

Verify the motion topic:

    ros2 topic info /cmd_vel

Expected:

    Type: geometry_msgs/msg/Twist

The quadruped controller should appear as a subscriber.

## 4. Start the Robot Bridge

Open another terminal on the Ubuntu PC and connect to the robot:

    ssh ubuntu@ROBOT_IP

Enter the Robot Bridge project:

    cd ~/robot_bridge

Activate its virtual environment if one is present:

    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi

Set the ROS environment:

    export ROS_DISTRO=humble
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0

Start the Robot Bridge:

    python3 app.py

Keep this terminal open.

From the Ubuntu PC, verify the service:

    curl http://ROBOT_IP:8090/status

Expected result:

- HTTP request succeeds.
- `"ok"` is `true`.
- Robot status is `READY`.

## 5. Start the Camera Relay

Open another terminal and connect to the Mini Pupper:

    ssh ubuntu@ROBOT_IP

Enter the Robot Bridge directory:

    cd ~/robot_bridge

Activate the virtual environment if needed:

    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi

Set the ROS environment:

    export ROS_DISTRO=humble
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0

Start the Camera Relay:

    python3 camera_relay.py

Keep this terminal open.

From the Ubuntu PC, verify that the latest camera image is reachable:

    curl --output /tmp/mini_pupper_camera.jpg \
        http://ROBOT_IP:8091/camera/latest.jpg

Verify the downloaded image:

    file /tmp/mini_pupper_camera.jpg

Expected result:

- The request succeeds.
- The downloaded file is identified as a JPEG image.

## 6. Prepare the Ubuntu Cognitive Environment

On the Ubuntu PC:

    cd ~/robot_services/cognitive

Activate the virtual environment:

    source .venv/bin/activate

Configure the service URLs:

    export ROBOT_BRIDGE_URL=http://ROBOT_IP:8090
    export CAMERA_RELAY_URL=http://ROBOT_IP:8091/camera/latest.jpg
    export VISION_SERVER_URL=http://127.0.0.1:8000/detections/latest

Configure Gemini if required:

    export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'

Do not commit API keys to Git.

## 7. Start the Vision Server

Open another Ubuntu terminal.

Enter the Vision Server project directory.

The exact directory may vary depending on the local installation. A common location is:

    cd ~/vision_server

Activate its virtual environment if present:

    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi

Configure the camera source:

    export CAMERA_URL=http://ROBOT_IP:8091/camera/latest.jpg

Start the Vision Server using its normal project command.

For a FastAPI application, this may be:

    uvicorn app:app --host 0.0.0.0 --port 8000

Keep this terminal open.

Verify the detection endpoint:

    curl http://127.0.0.1:8000/detections/latest

Expected result:

- JSON is returned.
- The response includes detection or camera status fields.
- `last_error` is empty or null during normal operation.

## 8. Start the Vision Service

Open another Ubuntu terminal:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

Set the Vision Server URL:

    export VISION_SERVER_URL=http://127.0.0.1:8000/detections/latest

Start the Vision Service using the current project entry point.

If the service is launched directly:

    python3 vision_service.py

If it is located in a package or subdirectory, use the project-specific command documented in the repository.

Keep this terminal open.

The Vision Service should continuously:

1. Poll the Vision Server.
2. Normalize detection labels.
3. Update entities in the World Model.
4. Refresh observation timestamps.
5. Preserve the World Model as the perception source of truth.

## 9. Start the Persistent Runtime API

Open another Ubuntu terminal:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

Set the required URLs:

    export ROBOT_BRIDGE_URL=http://ROBOT_IP:8090
    export VISION_SERVER_URL=http://127.0.0.1:8000/detections/latest

Start the Runtime API:

    python3 runtime_api.py

Keep this terminal open.

Verify runtime health:

    curl http://127.0.0.1:8770/health

Verify runtime status:

    curl http://127.0.0.1:8770/status

Expected result:

- Runtime API responds.
- Runtime reports its current state.
- The active mission and queue are visible.
- World Model and history information are available when supported.

## 10. Start the Browser Voice Relay

Open another Ubuntu terminal:

    cd ~/robot_services/cognitive
    source .venv/bin/activate

Start the voice relay:

    python3 voice_relay/server.py

Keep this terminal open.

Open the browser interface:

    http://127.0.0.1:8765

Use Google Chrome or another browser that supports speech recognition.

Allow microphone access when prompted.

## 11. Run Platform Health Checks

From the Ubuntu PC:

### Robot Bridge

    curl http://ROBOT_IP:8090/status

### Camera Relay

    curl --output /tmp/mini_pupper_camera.jpg \
        http://ROBOT_IP:8091/camera/latest.jpg

### Vision Server

    curl http://127.0.0.1:8000/detections/latest

### Runtime API

    curl http://127.0.0.1:8770/health

### Runtime Status

    curl http://127.0.0.1:8770/status

All required services must respond before motion testing.

## 12. Run a Safe Motion Test

Place the robot on a clear floor area.

Keep access to the STOP command.

Submit a short forward-motion command through the normal voice or Runtime API interface.

Example voice command:

    Move forward

Immediately verify STOP:

    Stop

Confirm that STOP:

- Sends a stop request to the robot.
- Cancels the active mission.
- Clears queued missions.
- Returns the runtime to `STOPPED`.

Do not continue if STOP does not work reliably.

## 13. Run a FIND_OBJECT Test

Place a recognizable object in view of the camera.

Example:

    Find my backpack

Verify that:

1. The command is parsed as `FIND_OBJECT`.
2. A mission is queued and activated.
3. The Vision Service updates the World Model.
4. The behavior performs one bounded action per runtime cycle.
5. The target is centered or approached.
6. The mission completes when the target is found.
7. STOP can preempt the mission at any time.

## 14. Ready-State Criteria

The platform is ready only when:

- Mini Pupper responds to network requests.
- ROS 2 bringup is running.
- `/cmd_vel` has the quadruped controller subscriber.
- Robot Bridge reports `READY`.
- Camera Relay provides a valid image.
- Vision Server returns current data.
- Vision Service updates the World Model.
- Runtime API responds successfully.
- Browser Voice Relay is available.
- STOP has been tested successfully.
- Regression tests are passing.

When all checks pass:

    ====================
    SYSTEM READY
    ====================

## 15. Troubleshooting

### Mini Pupper Cannot Be Reached

Check:

    ip address
    ping -c 4 ROBOT_IP

Confirm that both systems are on the same network.

### Robot Bridge Does Not Respond

On the Mini Pupper:

    ps aux | grep app.py
    ss -ltnp | grep 8090

Restart the Robot Bridge if needed.

### Port Already in Use

Find the process:

    sudo lsof -i :PORT_NUMBER

Example:

    sudo lsof -i :8090

Stop only the confirmed stale process.

### Camera Image Does Not Load

On the Mini Pupper:

    ros2 topic info /image_raw
    ros2 topic hz /image_raw

Verify that the camera node is running.

Check the Camera Relay process:

    ps aux | grep camera_relay.py
    ss -ltnp | grep 8091

### Vision Server Reports Camera Errors

Verify the relay directly:

    curl --output /tmp/test-camera.jpg \
        http://ROBOT_IP:8091/camera/latest.jpg

Check that `CAMERA_URL` points to the correct robot IP address.

### Robot Does Not Move

Verify bringup:

    ros2 node list

Verify the controller:

    ros2 topic info /cmd_vel

Verify Robot Bridge status:

    curl http://ROBOT_IP:8090/status

Confirm these environment variables on the Mini Pupper:

    echo $ROS_DOMAIN_ID
    echo $ROS_LOCALHOST_ONLY

Expected:

    ROS_DOMAIN_ID=42
    ROS_LOCALHOST_ONLY=0

### Vision Detects Objects but Behavior Cannot Find Them

Check:

- Vision Service is running continuously.
- The World Model is being updated.
- Detection timestamps are recent.
- Label normalization is correct.
- The requested target matches the normalized label.
- The camera is not obstructed.

### Runtime API Does Not Respond

Check:

    ps aux | grep runtime_api.py
    ss -ltnp | grep 8770

Review the runtime terminal for Python exceptions or missing environment variables.

## 16. Manual Shutdown

Issue STOP before shutting down:

    curl -X POST http://ROBOT_IP:8090/stop

Then stop services in this order:

1. Browser Voice Relay
2. Runtime API
3. Vision Service
4. Vision Server
5. Camera Relay
6. Robot Bridge
7. ROS 2 bringup

Use `Ctrl+C` in each service terminal.

Power off the Mini Pupper only after motion has stopped.

## Automated Startup

The implemented startup script is the preferred startup method:

```bash
cd ~/robot_services/cognitive
source .venv/bin/activate
python3 scripts/start_platform.py --start
```

Use `--start` to start missing services, `--restart` after local service code
changes, and `--check` to verify service readiness without starting services.

A successful startup prints `SYSTEM READY`.

## FOLLOW_PERSON Identity Verification

Follow `docs/PERSISTENT_IDENTITY.md`. The required transition is:

```text
LOCKED -> RECOVERING -> WAITING_FOR_IDENTITY -> LOCKED
```

The persistent `locked_identity_id` must remain unchanged. STOP must return
tracking to `UNLOCKED`.
