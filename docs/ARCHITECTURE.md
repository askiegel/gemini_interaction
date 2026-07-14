# Mini Pupper 2 Cognitive Platform Architecture

This document describes the current architecture of the Mini Pupper 2 Cognitive Platform.

## 1. Architectural Goal

The platform separates high-level cognition from low-level robot control.

The Ubuntu PC acts as the cognitive brain.

The Mini Pupper acts as the physical robot body.

The architecture is designed to support:

- Voice commands
- Vision-based perception
- Persistent missions
- Safe bounded motion
- Shared world state
- Future LiDAR navigation
- Future SLAM
- Future multi-robot coordination

## 2. System Overview

The main data flow is:

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
           +--------------------+
           |                    |
           v                    v
    Behavior Manager       Vision Service
           |                    |
           v                    v
    Robot Bridge Client    Shared World Model
           |
           v
    Robot Bridge Server
           |
           v
         ROS 2
           |
           v
    Mini Pupper Hardware

## 3. Deployment Model

### Ubuntu PC

The Ubuntu PC runs:

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
- Regression tests

Project directory:

    ~/robot_services/cognitive

### Mini Pupper 2

The Mini Pupper runs:

- ROS 2 Humble
- Mini Pupper bringup
- Quadruped controller
- Camera driver
- LiDAR driver
- IMU driver
- Robot Bridge server
- Camera Relay

Robot Bridge directory:

    ~/robot_bridge

## 4. Core Design Rules

The platform follows these rules:

1. The Runtime owns mission scheduling.
2. The Mission Manager owns mission lifecycle state.
3. `BehaviorManager` performs one bounded action per execution.
4. Behaviors do not contain internal loops.
5. The Runtime repeats behavior execution when persistence is required.
6. The World Model is the single source of truth for perception.
7. The Vision Service continuously updates the World Model.
8. STOP preempts all active work.
9. ROS 2 remains isolated on the Mini Pupper.
10. Components communicate through explicit interfaces.

## 5. Voice Input Layer

The Browser Voice Relay provides speech input through a web browser.

Typical flow:

    Spoken command
         |
         v
    Browser speech recognition
         |
         v
    HTTP request
         |
         v
    voice_command.py

Example commands:

    Move forward
    Turn left
    Stop
    Find my backpack
    Follow that person

The browser layer is responsible only for capturing and relaying the command.

It does not manage missions or robot motion.

## 6. Provider Layer

The provider layer converts natural-language input into structured intent data.

The current provider is Gemini.

Typical output:

    {
      "intent": "FIND_OBJECT",
      "target": "backpack",
      "speech": "Okay, I'll look for your backpack."
    }

The provider must not directly control the robot.

Its role is language understanding only.

## 7. Intent Parser

The Intent Parser validates and normalizes provider output.

Supported intent examples include:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`
- `FIND_OBJECT`
- `FOLLOW_PERSON`
- `DESCRIBE_SCENE`
- `RETURN_HOME`
- `UNKNOWN`

The parser converts provider output into a form that can be turned into a mission.

## 8. Mission Manager

The Mission Manager owns mission creation, queuing, activation, cancellation, and completion.

A mission contains fields such as:

    mission_id
    mission_type
    status
    target
    speech
    created_at
    started_at
    completed_at
    priority
    source

Typical mission states include:

- `ACTIVE`
- `COMPLETED`
- `CANCELLED`
- `REJECTED`
- `INFO_ONLY`

The Mission Manager does not perform robot motion.

It manages mission state only.

## 9. Persistent Runtime

The Runtime is the central coordinator.

It owns:

- Mission scheduling
- Active mission execution
- Repeated behavior cycles
- Mission completion
- STOP handling
- Runtime state
- Mission history

The Runtime executes one bounded behavior action per cycle.

Typical loop:

    Read active mission
           |
           v
    Execute one bounded behavior action
           |
           v
    Inspect result
           |
           +------ Complete mission
           |
           +------ Continue next cycle
           |
           +------ Cancel mission
           |
           +------ Stop platform

This allows persistent behavior without embedding loops inside behaviors.

## 10. Behavior Manager

The Behavior Manager translates missions into bounded robot actions.

Examples:

### MOVE_FORWARD

- Send one short forward command.
- Allow automatic stop.
- Mark the mission complete.

### TURN_LEFT

- Send one short left-turn command.
- Allow automatic stop.
- Mark the mission complete.

### FIND_OBJECT

- Read the latest target state from the World Model.
- Decide whether to turn, move forward, or finish.
- Perform one bounded action.
- Return control to the Runtime.

The Behavior Manager must not:

- Poll continuously
- Sleep in long loops
- Own mission scheduling
- Query perception indefinitely
- Ignore STOP state

## 11. Vision Server

The Vision Server runs YOLO inference.

It receives camera images from the Camera Relay and publishes detection data.

Typical endpoint:

    http://127.0.0.1:8000/detections/latest

Typical detection fields:

    label
    confidence
    bbox
    center_x
    area
    image_width
    image_height
    timestamp

The Vision Server is responsible for detection only.

It is not the authoritative world state.

## 12. Vision Service

The Vision Service continuously polls the Vision Server.

Its responsibilities are:

1. Read current detections.
2. Normalize labels.
3. Select useful detections.
4. Convert detections into world entities.
5. Update timestamps and confidence.
6. Persist the latest world state.

The Vision Service acts as the bridge between raw perception and the World Model.

## 13. Shared World Model

The World Model is the single source of truth for perception.

It stores entities and observations such as:

    entity_id
    label
    entity_type
    first_seen
    last_seen
    confidence
    location
    attributes
    history

Example:

    backpack-002
    label: backpack
    confidence: 0.79
    center_x: 447
    distance_class: near
    last_seen: current timestamp

Behaviors read the World Model rather than querying the Vision Server directly.

This provides:

- Consistent perception access
- Timestamp validation
- Entity history
- Label normalization
- Future sensor fusion
- Future semantic mapping

## 14. Robot Bridge Client

The Robot Bridge client runs on the Ubuntu PC.

It sends HTTP requests to the Mini Pupper.

Typical actions include:

- Motion command
- Stop command
- Status request

Example motion payload:

    {
      "linear_x": 0.1,
      "angular_z": 0.0,
      "duration": 0.5,
      "automatic_stop": true
    }

The client hides HTTP details from the Behavior Manager.

## 15. Robot Bridge Server

The Robot Bridge server runs on the Mini Pupper.

Default port:

    8090

Typical endpoints:

    GET  /status
    POST /motion
    POST /stop

The Robot Bridge converts HTTP requests into ROS 2 commands.

Its responsibilities include:

- Validate motion requests
- Publish to `/cmd_vel`
- Enforce bounded duration
- Automatically stop
- Report status
- Provide a direct emergency stop endpoint

## 16. Camera Relay

The Camera Relay runs on the Mini Pupper.

Default port:

    8091

It reads the ROS 2 camera topic and provides the latest image over HTTP.

Typical endpoint:

    http://ROBOT_IP:8091/camera/latest.jpg

This avoids requiring ROS 2 image transport across the network.

The Vision Server uses the HTTP camera image as its input.

## 17. ROS 2 Layer

ROS 2 remains on the Mini Pupper.

Current environment:

    ROS_DISTRO=humble
    ROS_DOMAIN_ID=42
    ROS_LOCALHOST_ONLY=0

Important nodes include:

- `/quadruped_controller_node`
- `/robot_state_publisher`
- `/servo_interface`
- `/v4l2_camera`
- `/state_estimation_node`
- `/LD06`
- `/imu_interface`

Important topics include:

- `/cmd_vel`
- `/image_raw`
- `/camera_info`

The cognitive PC does not need to participate directly in ROS 2 discovery for normal operation.

## 18. Runtime API

The Runtime API exposes platform status and mission submission over HTTP.

Default port:

    8770

Typical endpoints:

    GET  /health
    GET  /status
    GET  /missions
    POST /missions

Example mission request:

    {
      "command": "Find my backpack"
    }

The Runtime API provides a stable external interface for:

- Browser tools
- Dashboards
- Automation scripts
- Future mobile interfaces
- Future robot-to-robot communication

## 19. STOP Preemption

STOP is the highest-priority control path.

When STOP is received:

1. A stop request is sent to the Robot Bridge.
2. The active mission is cancelled.
3. Queued missions are cleared.
4. Further behavior execution is prevented.
5. Runtime state becomes `STOPPED`.

STOP must bypass normal mission waiting.

This behavior is required for every future autonomous capability.

## 20. Persistent FIND_OBJECT

Persistent `FIND_OBJECT` uses repeated runtime cycles.

Typical lifecycle:

    Acquire target
         |
         v
    Inspect target position
         |
         +------ Target left: turn left
         |
         +------ Target right: turn right
         |
         +------ Target centered and far: move forward
         |
         +------ Target reached: complete
         |
         +------ Target lost: search
         |
         v
    Return control to Runtime
         |
         v
    Repeat next cycle

Each behavior execution performs only one bounded action.

This preserves STOP responsiveness and mission scheduling.

## 21. Planned Persistent FOLLOW_PERSON

`FOLLOW_PERSON` will reuse the same architecture.

Planned lifecycle:

    Acquire person
         |
         v
    Center person
         |
         v
    Maintain distance
         |
         v
    Repeat
         |
         v
    STOP

The Runtime will own repetition.

The Behavior Manager will perform one correction at a time.

The World Model will provide the latest person observation.

## 22. Service Ports

| Service | Host | Port |
|---|---|---:|
| Vision Server | Ubuntu PC | 8000 |
| Browser Voice Relay | Ubuntu PC | 8765 |
| Runtime API | Ubuntu PC | 8770 |
| Robot Bridge | Mini Pupper | 8090 |
| Camera Relay | Mini Pupper | 8091 |

## 23. Failure Isolation

The architecture supports layer-by-layer diagnosis.

Troubleshooting order:

    Hardware
       |
       v
    ROS 2
       |
       v
    Robot Bridge
       |
       v
    Camera Relay
       |
       v
    Vision Server
       |
       v
    Vision Service
       |
       v
    World Model
       |
       v
    Runtime
       |
       v
    Voice and Dashboard

This reduces the risk of changing multiple layers while diagnosing one fault.

## 24. Future Architecture Extensions

Planned extensions include:

- LiDAR obstacle avoidance
- Navigation stack integration
- SLAM
- Map persistence
- Return Home
- Patrol missions
- Semantic mapping
- Person re-identification
- Pose estimation
- Robot-to-robot communication
- Shared fleet world model
- Fleet mission coordination

These features should preserve the existing architectural rules.

## 25. Non-Goals

The current architecture intentionally avoids:

- Running the cognitive stack on the Mini Pupper
- Embedding long loops inside behaviors
- Allowing behaviors to own mission state
- Allowing multiple perception sources to bypass the World Model
- Coupling the Ubuntu PC directly to low-level ROS 2 control
- Mixing unrelated features in the same implementation change

## 26. Architectural Summary

The platform is organized into four major layers:

### Interaction

- Browser Voice Relay
- Operator Dashboard
- Runtime API

### Cognition

- Gemini provider
- Intent Parser
- Mission Manager
- Persistent Runtime
- Behavior Manager

### Perception

- Camera Relay
- Vision Server
- Vision Service
- Shared World Model

### Actuation

- Robot Bridge client
- Robot Bridge server
- ROS 2
- Mini Pupper hardware

This separation allows the system to grow without redesigning the core platform.
