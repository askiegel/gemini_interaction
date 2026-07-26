# Persistent Person Identity and Target Recovery

This document describes the verified identity and FOLLOW_PERSON recovery architecture.

## Ownership Rules

- Vision Service is the only live perception writer.
- Cognitive Runtime is the only mission-execution owner.
- World Model is the single source of truth for perception.
- `PersonIdentityManager` assigns persistent person identities.
- `TargetLock` owns mission-specific target continuity.
- `PredictionTracker` handles short visual interruptions.
- ROS 2 remains isolated on the Mini Pupper.

Both `voice_command.py --runtime` and the backward-compatible
`voice_command.py --execute` option submit through the Runtime API.
Neither creates a second live identity-processing pipeline.

## Perception and Identity Flow

```text
Camera Relay
  -> YOLO Vision Server
  -> Vision Service
  -> VisionAdapter
  -> Entity Registry
  -> PersonIdentityManager
  -> World Model
  -> TargetLock
  -> FOLLOW_PERSON Controller
```

Each vision frame is processed in this order:

1. Fetch and normalize the complete detection frame.
2. Register detections with the Entity Registry.
3. Attach authoritative World Model entity IDs.
4. Deduplicate people that resolve to the same entity.
5. Assign persistent identities.
6. Enrich the existing World Model observation without duplicating it.

## Entity and Identity Continuity

Transient IDs such as `person-001` and `person-002` can change. The
persistent ID, such as `person-identity-91a1f927`, must remain stable.

`PersonIdentityManager.entity_bindings` maps multiple historical transient
entity IDs to one persistent identity. Exact entity bindings take priority
over ambiguous geometric matching.

## TargetLock Modes

| Mode | Meaning |
|---|---|
| `UNLOCKED` | No target is selected. |
| `LOCKED` | The selected target is visible. |
| `RECOVERING` | Predictive recovery is active. |
| `WAITING_FOR_IDENTITY` | Motion is held while the persistent identity is retained. |

Expected occlusion sequence:

```text
LOCKED -> RECOVERING -> WAITING_FOR_IDENTITY -> LOCKED
```

While waiting, `locked_entity_id` is `None`, `locked_identity_id` remains
selected, the robot remains stationary, and generic label-based reacquisition
is blocked. STOP resets tracking to `UNLOCKED`.

## Verified Hardware Result

Live hardware validation confirmed:

- Persistent identity remained stable during movement.
- Two waiting-to-locked recoveries succeeded.
- No persistent identity changes occurred.
- STOP released TargetLock.

```text
locked_entity_id: person-002
locked_identity_id: person-identity-91a1f927
identity_change_count: 0
```

## Startup

```bash
cd ~/robot_services/cognitive
source .venv/bin/activate
python3 scripts/start_platform.py --start
```

After changing local service code, use `--restart`. Use `--check` to verify
services without starting them. Open the Operator Console at
`http://localhost:8765`.

## Acceptance Test

1. Place the robot in a clear area and keep STOP available.
2. Stand alone and issue `Follow me` through the Operator Console.
3. Confirm tracking reaches `LOCKED` and record `locked_identity_id`.
4. Leave view until `WAITING_FOR_IDENTITY`.
5. Reappear and confirm tracking returns to `LOCKED`.
6. Confirm `locked_identity_id` did not change.
7. Issue STOP and confirm `UNLOCKED`.

Direct emergency stop:

```bash
curl -fsS -X POST http://ROBOT_IP:8090/stop | python3 -m json.tool
```

## Diagnostics

Check runtime tracking:

```bash
curl -fsS http://127.0.0.1:8770/status | python3 -m json.tool
```

Relevant logs:

```text
logs/platform/vision_service.log
logs/platform/runtime_api.log
logs/platform/voice_relay.log
```

## Regression Coverage

```text
test_person_identity_entity_binding.py
test_vision_duplicate_person_identity.py
test_vision_entity_identity_order.py
test_vision_adapter_identity.py
test_target_lock_same_entity_identity_refresh.py
test_target_lock_waiting_identity_refresh.py
test_voice_command_execute_runtime.py
```

Relevant commits:

```text
dc9736a Preserve TargetLock continuity through identity refresh
90fb5e2 Expose identity lifecycle telemetry in operator console
dbf8908 Stabilize persistent identity across entity changes
d1065a2 Route legacy command execution through runtime
```
