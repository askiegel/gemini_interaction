#!/usr/bin/env python3

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from target_lock import TargetLock


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


def iso(value):
    return value.isoformat().replace(
        "+00:00",
        "Z",
    )


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0

        now = datetime.now(timezone.utc)

        self.entity = Entity(
            entity_id="person-003",
            label="person",
            confidence=0.95,
            last_seen=iso(
                now - timedelta(seconds=1.0)
            ),
            history=[
                Observation(
                    location={
                        "cx": 260.0,
                        "cy": 220.0,
                    },
                    attributes={
                        "area": 30000.0,
                        "image_width": 640.0,
                        "image_height": 480.0,
                    },
                )
            ],
            attributes={
                "area": 30000.0,
                "image_width": 640.0,
                "image_height": 480.0,
            },
        )

    def reload(self):
        return None

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
            "entity_id": "person-003",
            "label": "person",
            "confidence": 0.95,
            "cx": 160.0,
            "cy": 220.0,
            "area": 26000.0,
            "image_width": 640.0,
            "image_height": 480.0,
            "last_seen": iso(
                datetime.now(timezone.utc)
                - timedelta(seconds=2.0)
            ),
        }

    def get_entity(self, entity_id):
        if (
            self.entity is not None
            and entity_id == self.entity.entity_id
        ):
            return self.entity

        return None


world_model = FakeWorldModel()

target_lock = TargetLock(
    world_model=world_model,
    max_age_seconds=3.0,
    recovery_timeout_seconds=2.0,
)

target_lock.prediction_tracker.velocity_smoothing = 1.0

print("===== ACQUIRE FIRST MEASUREMENT =====")

first = target_lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

print(first)

assert first["entity_id"] == "person-003"
assert world_model.label_queries == 1

print()
print("===== UPDATE MOVING TARGET =====")

second = target_lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

print(second)

assert second["entity_id"] == "person-003"

tracker_state = (
    target_lock
    .prediction_tracker
    .snapshot()
)

print()
print("===== TRACKER STATE =====")
print(tracker_state)

assert tracker_state["measurement_count"] >= 2
assert tracker_state["horizontal_velocity"] > 0.0

print()
print("===== TARGET DISAPPEARS =====")

world_model.entity = None

recovery = target_lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

print(recovery)

assert recovery["found"] is False
assert recovery["lock_expired"] is False
assert recovery["entity_id"] == "person-003"
assert recovery["prediction"]["available"] is True
assert recovery["predicted_cx"] is not None
assert recovery["horizontal_velocity"] > 0.0
assert (
    recovery["recovery_direction"]
    == TargetLock.DIRECTION_RIGHT
)
assert world_model.label_queries == 1

print()
print("PASS: target motion history was retained")
print("PASS: horizontal velocity was estimated")
print("PASS: target location was predicted during loss")
print("PASS: recovery used predicted direction")
print("PASS: locked identity was preserved")
print("PASS: no replacement person was acquired")
print()
print("Predictive Target Recovery integration passed.")
