#!/usr/bin/env python3

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from target_lock import TargetLock


def now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class FakeObservation:
    location: Dict[str, Any]
    attributes: Dict[str, Any]
    timestamp: str = field(default_factory=now_iso)


@dataclass
class FakeEntity:
    entity_id: str
    label: str
    last_seen: str
    confidence: float
    attributes: Dict[str, Any]
    history: List[FakeObservation]


class FakeWorldModel:
    def __init__(self):
        self.entities = {}

    def reload(self):
        return None

    def get_entity(self, entity_id):
        return self.entities.get(entity_id)

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        for entity in self.entities.values():
            if entity.label == label:
                observation = entity.history[-1]

                return {
                    "found": True,
                    "stale": False,
                    "target": label,
                    "entity_id": entity.entity_id,
                    "identity_id": observation.attributes.get(
                        "identity_id"
                    ),
                    "identity_status": observation.attributes.get(
                        "identity_status"
                    ),
                    "last_seen": entity.last_seen,
                    "confidence": entity.confidence,
                    "cx": observation.location.get("cx"),
                    "area": observation.location.get("area"),
                    "image_width": observation.location.get(
                        "image_width"
                    ),
                }

        return {
            "found": False,
            "stale": False,
            "target": label,
        }

    def find_latest_entity_by_identity(
        self,
        identity_id,
        max_age_seconds=None,
        refresh=True,
    ):
        for entity in self.entities.values():
            observation = entity.history[-1]

            if (
                observation.attributes.get("identity_id")
                == identity_id
            ):
                return {
                    "found": True,
                    "stale": False,
                    "target": entity.label,
                    "entity_id": entity.entity_id,
                    "identity_id": identity_id,
                    "last_seen": entity.last_seen,
                    "confidence": entity.confidence,
                    "cx": observation.location.get("cx"),
                    "area": observation.location.get("area"),
                    "image_width": observation.location.get(
                        "image_width"
                    ),
                }

        return {
            "found": False,
            "stale": False,
            "target": None,
            "identity_id": identity_id,
            "reason": (
                f"No World Model entity matches identity "
                f"'{identity_id}'."
            ),
        }


def make_person(entity_id, identity_id):
    timestamp = now_iso()

    observation = FakeObservation(
        location={
            "cx": 320.0,
            "area": 30000.0,
            "image_width": 640.0,
        },
        attributes={
            "identity_id": identity_id,
            "identity_status": "MATCHED",
        },
        timestamp=timestamp,
    )

    return FakeEntity(
        entity_id=entity_id,
        label="person",
        last_seen=timestamp,
        confidence=0.90,
        attributes={
            "identity_id": identity_id,
        },
        history=[observation],
    )


world = FakeWorldModel()

old_identity = "person-identity-old"
new_identity = "person-identity-stable"

world.entities["person-001"] = make_person(
    entity_id="person-001",
    identity_id=old_identity,
)

lock = TargetLock(
    world_model=world,
    max_age_seconds=3.0,
)

first = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

assert first["found"] is True
assert lock.locked_entity_id == "person-001"
assert lock.locked_identity_id == old_identity
assert lock.tracking_mode == TargetLock.MODE_LOCKED

# The same transient entity remains visible but is now assigned a different
# persistent identity. Mission ownership must remain with the original
# identity rather than silently transferring to the replacement.
world.entities["person-001"] = make_person(
    entity_id="person-001",
    identity_id=new_identity,
)

second = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

assert second["found"] is False
assert second["identity_mismatch"] is True
assert second["reacquisition_blocked"] is True
assert second["observed_identity_id"] == new_identity
assert second["identity_id"] == old_identity

assert lock.locked_entity_id == "person-001"
assert lock.locked_identity_id == old_identity
assert lock.tracking_mode == TargetLock.MODE_LOCKED

third = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

assert third["found"] is False
assert third["identity_mismatch"] is True
assert lock.locked_identity_id == old_identity

print(
    "PASS: Same-entity identity mismatch was blocked and "
    "mission identity ownership remained immutable."
)
