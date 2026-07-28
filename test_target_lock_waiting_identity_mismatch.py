#!/usr/bin/env python3

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from target_lock import TargetLock


def now_iso(offset_seconds=0):
    value = (
        datetime.now(timezone.utc)
        + timedelta(seconds=offset_seconds)
    )

    return value.isoformat().replace("+00:00", "Z")


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
            if entity.label != label:
                continue

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


def make_person(identity_id, timestamp=None):
    timestamp = timestamp or now_iso()

    observation = FakeObservation(
        location={
            "cx": 320.0,
            "area": 32000.0,
            "image_width": 640.0,
        },
        attributes={
            "identity_id": identity_id,
            "identity_status": "MATCHED",
        },
        timestamp=timestamp,
    )

    return FakeEntity(
        entity_id="person-001",
        label="person",
        last_seen=timestamp,
        confidence=0.91,
        attributes={
            "identity_id": identity_id,
        },
        history=[observation],
    )


world = FakeWorldModel()

temporary_identity = "person-identity-temporary"
stable_identity = "person-identity-stable"

world.entities["person-001"] = make_person(
    temporary_identity
)

lock = TargetLock(
    world_model=world,
    max_age_seconds=3.0,
    recovery_timeout_seconds=0.0,
)

acquired = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

assert acquired["found"] is True
assert lock.locked_entity_id == "person-001"
assert lock.locked_identity_id == temporary_identity
assert lock.tracking_mode == TargetLock.MODE_LOCKED

# Enter the same waiting transition used after predictive recovery expires.
waiting = lock._enter_identity_wait()

assert (
    lock.tracking_mode
    == TargetLock.MODE_WAITING_FOR_IDENTITY
)
assert lock.locked_entity_id is None
assert lock.waiting_entity_id == "person-001"
assert lock.locked_identity_id == temporary_identity
assert waiting["entity_id"] is None

# The same transient entity becomes fresh again but now carries a different
# identity. Waiting ownership must remain with the originally selected
# persistent identity.
world.entities["person-001"] = make_person(
    stable_identity
)

blocked = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

assert blocked["found"] is False
assert blocked["identity_mismatch"] is True
assert blocked["reacquisition_blocked"] is True
assert blocked["observed_identity_id"] == stable_identity
assert blocked["identity_id"] == temporary_identity

assert lock.locked_entity_id is None
assert lock.waiting_entity_id == "person-001"
assert lock.locked_identity_id == temporary_identity
assert (
    lock.tracking_mode
    == TargetLock.MODE_WAITING_FOR_IDENTITY
)

print(
    "PASS: Waiting TargetLock rejected a changed identity "
    "and retained original mission ownership."
)
