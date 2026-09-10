# Mayday Cognitive Platform — Agent Instructions

## Purpose

This repository is part of the Mayday Mini Pupper 2 cognitive robotics platform.

Changes in this repository may eventually control a real physical quadruped robot. Treat hardware-affecting operations as safety-critical.

## Architecture

- ROS 2 runs on Mayday, the physical robot.
- ROS distribution is Humble.
- ROS_DOMAIN_ID=42.
- ROS_LOCALHOST_ONLY=0.
- The World Model is authoritative for cognitive state.
- Preserve the existing architecture unless an architectural change is explicitly requested.
- Do not redesign subsystem boundaries as part of an unrelated feature.

## Development discipline

Use this workflow:

Inspect → modify → build/test → review diff → commit → push.

Additional rules:

- One feature per commit.
- Keep changes narrowly scoped.
- Do not modify unrelated files.
- Do not silently clean up or refactor unrelated code.
- Do not commit or push unless explicitly requested.
- Before modifying code, inspect the relevant implementation and tests.
- After modifying code, run the smallest relevant safe test set first.
- Show the resulting Git diff and test results before proposing a commit.
- Preserve existing behavior unless the requested feature intentionally changes it.

## Physical robot safety

Default state: MAYDAY REMAINS STATIONARY.

Unless the user explicitly authorizes a specific physical-motion test in the current task, NEVER:

- publish a nonzero cmd_vel
- submit a navigation goal
- start autonomous navigation
- command walking, turning, following, or gait motion
- start a motion-producing ROS node or test
- run a test known to command physical motion
- bypass a motion guard or safety interlock

A software task does not imply permission to move the robot.

Past success of a motion test does not imply permission to repeat it.

If it is unclear whether a command can cause motion, treat it as motion-capable and do not execute it without explicit authorization.

## Robot and network access

Unless explicitly permitted for the current task:

- Do not SSH into Mayday.
- Do not send HTTP requests to Mayday.
- Do not start, stop, or restart Mayday services.
- Do not start localization, mapping, planning, or navigation.
- Do not initialize localization.
- Do not issue navigation or planning goals.
- Do not publish ROS control messages.

Read-only local source inspection does not require robot access.

## Testing

Prefer offline/unit tests whenever possible.

Before running a test, determine whether it can:

- access Mayday
- start services
- publish ROS messages
- issue motion commands
- perform navigation
- cause physical motion

If any of those are possible, do not run the test without explicit authorization.

## Repository boundaries

This is the cognitive repository.

Do not modify other repositories as a side effect of a task in this repository.

If a requested feature requires coordinated changes in another repository, stop and report that requirement before making cross-repository changes.

## Git safety

Before making a change:

- verify the expected repository
- verify the current branch
- verify HEAD
- inspect git status

Do not overwrite pre-existing user changes.

Do not use destructive Git commands unless explicitly requested.

## Communication

When reporting work, clearly distinguish:

- what was inspected
- what was changed
- what tests were run
- what was not tested
- whether any robot/network access occurred
- whether any physical motion occurred
