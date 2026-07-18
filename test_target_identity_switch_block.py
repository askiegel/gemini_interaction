#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from target_lock import TargetLock


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


class FakeWorldModel:
    def __init__(self):
        self.latest_queries = 0
        self.entity_queries = []
        self.original_visible = True
        self.other_person_visible = False

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        self.latest_queries += 1

        if self.other_person_visible:
            return {
                "found": True,
                "stale": False,
                "target": label,
                "entity_id": "person-999",
                "label": "person",
                "confidence": 0.99,
                "cx": 320.0,
                "cy": 220.0,
                "area": 42000.0,
                "image_width": 640.0,
                "image_height": 480.0,
                "last_seen": iso_now(),
            }

        return {
            "found": True,
            "stale": False,
            "target": label,
            "entity_id": "person-003",
            "label": "person",
            "confidence": 0.95,
            "cx": 320.0,
            "cy": 220.0,
            "area": 42000.0,
            "image_width": 640.0,
            "image_height": 480.0,
            "last_seen": iso_now(),
        }

    def reload(self):
        return None

    def get_entity(self, entity_id):
        from types import SimpleNamespace

        self.entity_queries.append(entity_id)

        if (
            entity_id != "person-003"
            or not self.original_visible
        ):
            return None

        observation = SimpleNamespace(
            location={
                "cx": 320.0,
                "cy": 220.0,
            },
            attributes={
                "bbox": {
                    "x1": 240.0,
                    "y1": 80.0,
                    "x2": 400.0,
                    "y2": 420.0,
                },
                "area": 42000.0,
                "image_width": 640.0,
                "image_height": 480.0,
            },
        )

        return SimpleNamespace(
            entity_id="person-003",
            label="person",
            confidence=0.95,
            last_seen=iso_now(),
            attributes={},
            history=[observation],
        )


world_model = FakeWorldModel()

lock = TargetLock(
    world_model=world_model,
    recovery_timeout_seconds=0.1,
)

print("===== ACQUIRE ORIGINAL PERSON =====")

first = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

print(first)

assert first["found"] is True
assert first["entity_id"] == "person-003"
assert lock.locked_entity_id == "person-003"

world_model.original_visible = False
world_model.other_person_visible = True

lock.lost_since = (
    datetime.now(timezone.utc)
    - timedelta(seconds=1.0)
).isoformat().replace("+00:00", "Z")

print()
print("===== RECOVERY TIMEOUT WITH OTHER PERSON VISIBLE =====")

second = lock.resolve(
    mission_id="mission-follow-001",
    target_label="person",
)

print(second)

assert second["found"] is False
assert second["lock_expired"] is True
assert second["identity_lost"] is True
assert second["reacquisition_blocked"] is True
assert second.get("entity_id") != "person-999"
assert lock.locked_entity_id is None
assert world_model.latest_queries == 1

print()
print("PASS: original person was acquired")
print("PASS: recovery timeout released the old lock")
print("PASS: another visible person was not acquired")
print("PASS: automatic identity switching was blocked")
print()
print("Target identity-switch prevention test passed.")
