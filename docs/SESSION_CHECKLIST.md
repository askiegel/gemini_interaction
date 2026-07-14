# Session Checklist

Follow this checklist at the start of every development session.

## Hardware

- [ ] Ubuntu development PC powered on
- [ ] Mini Pupper powered on
- [ ] Robot connected to Wi-Fi
- [ ] Ubuntu PC connected to the same network
- [ ] Robot battery charged

## Robot Verification

SSH into the robot:

    ssh ubuntu@ROBOT_IP

Verify ROS 2:

    echo $ROS_DISTRO

Expected:

    humble

## Start Robot Bringup

On the Mini Pupper:

    ros2 launch mini_pupper_bringup bringup.launch.py

Verify that the following components start:

- Camera
- Quadruped controller
- LiDAR
- IMU

## Verify Robot Bridge

From the Ubuntu PC:

    curl http://ROBOT_IP:8090/status

Expected status:

    READY

## Verify Camera Relay

From the Ubuntu PC:

    curl http://ROBOT_IP:8091/status

## Vision Server

Verify the detection endpoint:

    curl http://127.0.0.1:8000/detections/latest

## Runtime API

Verify runtime health:

    curl http://127.0.0.1:8770/health

## Functional Test

Issue the commands:

    Move forward
    Stop

Verify that:

- The robot moves.
- The robot stops.
- The active mission is cancelled.
- The mission queue is cleared.
- The runtime returns to STOPPED.

## Before Coding

- [ ] Activate the Python virtual environment
- [ ] Check Git status
- [ ] Run regression tests
- [ ] Confirm all services are healthy

## End of Session

- [ ] Run regression tests
- [ ] Review Git changes
- [ ] Commit the completed feature
- [ ] Push to GitHub
- [ ] Update documentation
- [ ] Record the next task
