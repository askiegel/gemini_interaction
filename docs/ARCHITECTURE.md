# Mini Pupper 2 Cognitive Platform Architecture

> **Project:** Mini Pupper 2 Cognitive Platform **Architecture
> Version:** 2.0 **Status:** Current Production Architecture

## Introduction

The Mini Pupper 2 Cognitive Platform is a layered robotics architecture
that separates language understanding, mission planning, perception,
behavior generation, runtime coordination, and hardware execution.

### Design Goals

-   Deterministic behavior
-   Modular services
-   Persistent runtime
-   Hardware abstraction
-   Fleet readiness
-   Safety-first execution

Artificial intelligence determines **what** the operator wants.

Deterministic software determines **which robot owns the command**,
whether it is valid, and how it is executed safely.

## High-Level Architecture

``` text
Human Operator
      |
Browser Voice Recognition
      |
Browser Voice Relay
      |
Robot Address Parser
      |
Conversation Service
      |
Conversation Manager
      |
Gemini Provider
      |
Mission Manager
      |
Behavior Manager
      |
Cognitive Runtime
      |
Robot Bridge
      |
ROS2
      |
Mini Pupper 2
```

## Core Components

### Robot Identity

Each robot has a permanent robot_id plus configurable display name and
aliases.

### Robot Fleet

Fleet configuration is stored in `config/robot_fleet.json`.

### Robot Address Parser

Runs before Gemini and determines which robot owns a command.

### Runtime

Coordinates missions, behaviors, world model, vision, and robot
communication.

### World Model

The single source of truth for perceived entities.

### Robot Bridge

Translates cognitive commands into ROS2 motion and provides watchdog
protection.

## Startup Order

1.  ROS2 Bringup
2.  Robot Bridge
3.  Camera Relay
4.  Vision Server
5.  Vision Service
6.  Cognitive Runtime
7.  Runtime API
8.  Browser Voice Relay
9.  Operator Dashboard

## Current Production Features

-   Robot Identity
-   Robot Fleet
-   Robot Address Parser
-   Deterministic routing
-   Persistent Runtime
-   World Model
-   Vision Service
-   Streaming FOLLOW_PERSON
-   Target Lock
-   Prediction Tracker

## Future

-   Fleet Transport
-   Shared World Model
-   Collaborative missions
-   Distributed SLAM

## Conclusion

The architecture separates AI from execution. AI interprets intent;
deterministic software governs ownership, routing, execution, and
safety.
