#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime, timezone

from behavior_manager import BehaviorManager
from target_lock import TargetLock


def now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class Observation:
    location: dict
    attributes: dict


@dataclass
class Entity:
    entity_id: str
    label: str
    confidence: float
    last_seen: str
    history: list
    attributes: dict


@dataclass
class Mission:
    mission_id: str
    mission_type: str = "FOLLOW_PERSON"
    target: str = "person"
    status: str = "ACTIVE"


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.reload_calls = 0

        self.entity = Entity(
            entity_id="person-003",
            label="person",
            confidence=0.94,
            last_seen=now_iso(),
            history=[
                Observation(
                    location={
                        "cx": 520.0,
                        "cy": 230.0,
                    },
                    attributes={
                        "bbox": {
                            "x1": 440.0,
                            "y1": 80.0,
                            "x2": 600.0,
                            "y2": 380.0,
                        },
                        "area": 48000.0,
                        "image_width": 640.0,
                        "image_height": 480.0,
                    },
                )
            ],
            attributes={
                "bbox": {
                    "x1": 440.0,
                    "y1": 80.0,
                    "x2": 600.0,
                    "y2": 380.0,
                },
                "area": 48000.0,
                "image_width": 640.0,
                "image_height": 480.0,
            },
        )

    def reload(self):
        self.reload_calls += 1

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        self.label_queries += 1

        return {
            "found": True,
            "stale": False,
            "target": label,
            "entity_id": self.entity.entity_id,
            "label": self.entity.label,
            "confidence": self.entity.confidence,
            "cx": 520.0,
            "cy": 230.0,
            "area": 48000.0,
            "bbox": {
                "x1": 440.0,
                "y1": 80.0,
                "x2": 600.0,
                "y2": 380.0,
            },
            "image_width": 640.0,
            "image_height": 480.0,
        }

    def get_entity(self, entity_id):
        if (
            self.entity is not None
            and self.entity.entity_id == entity_id
        ):
            return self.entity

        return None


class FakeRobot:
    def __init__(self):
        self.motion_commands = []
        self.stop_calls = 0

    def motion_stream(
        self,
        linear_x,
        angular_z,
        watchdog_timeout,
    ):
        command = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "watchdog_timeout": watchdog_timeout,
        }

        self.motion_commands.append(command)

        return {
            "ok": True,
            "mode": "streaming",
            **command,
        }

    def motion(
        self,
        linear_x,
        angular_z,
        duration,
    ):
        command = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration,
        }

        self.motion_commands.append(command)

        return {
            "ok": True,
            **command,
        }

    def turn_left(self, speed, seconds):
        return {
            "ok": True,
            "direction": "left",
            "speed": speed,
            "seconds": seconds,
        }

    def turn_right(self, speed, seconds):
        return {
            "ok": True,
            "direction": "right",
            "speed": speed,
            "seconds": seconds,
        }

    def move_forward(self, speed, seconds):
        return {
            "ok": True,
            "direction": "forward",
            "speed": speed,
            "seconds": seconds,
        }

    def stop(self):
        self.stop_calls += 1

        return {
            "ok": True,
            "action": "stop",
        }


world_model = FakeWorldModel()
robot = FakeRobot()

manager = BehaviorManager(
    robot_client=robot,
    world_model=world_model,
)

manager.target_lock.recovery_timeout_seconds = 2.0

mission = Mission(
    mission_id="mission-follow-001",
)

print("===== ACQUIRE PERSON ON RIGHT =====")

acquired = manager.execute(mission)
print(acquired)

assert acquired["locked_entity_id"] == "person-003"
assert acquired["tracking_mode"] == TargetLock.MODE_LOCKED
assert (
    manager.target_lock.last_seen_direction
    == TargetLock.DIRECTION_RIGHT
)
assert world_model.label_queries == 1

print()
print("===== PERSON TEMPORARILY DISAPPEARS =====")

world_model.entity = None

recovering = manager.execute(mission)
print(recovering)

assert recovering["state"] == "RECOVERING_TARGET"
assert (
    recovering["tracking_mode"]
    == TargetLock.MODE_RECOVERING
)
assert recovering["locked_entity_id"] == "person-003"
assert recovering["last_seen_direction"] == "RIGHT"
assert recovering["commanded_angular_z"] < 0.0
assert world_model.label_queries == 1

print()
print("===== PERSON RETURNS WITH SAME ID =====")

world_model.entity = Entity(
    entity_id="person-003",
    label="person",
    confidence=0.95,
    last_seen=now_iso(),
    history=[
        Observation(
            location={
                "cx": 500.0,
                "cy": 230.0,
            },
            attributes={
                "bbox": {
                    "x1": 420.0,
                    "y1": 80.0,
                    "x2": 580.0,
                    "y2": 380.0,
                },
                "area": 48000.0,
                "image_width": 640.0,
                "image_height": 480.0,
            },
        )
    ],
    attributes={
        "bbox": {
            "x1": 420.0,
            "y1": 80.0,
            "x2": 580.0,
            "y2": 380.0,
        },
        "area": 48000.0,
        "image_width": 640.0,
        "image_height": 480.0,
    },
)

reacquired = manager.execute(mission)
print(reacquired)

assert (
    reacquired["tracking_mode"]
    == TargetLock.MODE_LOCKED
)
assert reacquired["locked_entity_id"] == "person-003"
assert reacquired["state"] == "CENTERING_RIGHT"
assert world_model.label_queries == 1

print()
print("PASS: last-seen right direction was remembered")
print("PASS: temporary loss preserved the entity lock")
print("PASS: recovery steering turned toward the right")
print("PASS: another label acquisition was not attempted")
print("PASS: the original entity resumed normal tracking")
print()
print("Lost-target directional recovery test passed.")
