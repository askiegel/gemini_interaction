#!/usr/bin/env python3

from dataclasses import dataclass

from target_lock import TargetLock


@dataclass
class FakeObservation:
    timestamp: str
    source: str
    confidence: float
    location: dict
    attributes: dict


@dataclass
class FakeEntity:
    entity_id: str
    label: str
    last_seen: str
    confidence: float
    attributes: dict
    history: list


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.entity_queries = []
        self.reload_count = 0

        self.entities = {
            "person-003": FakeEntity(
                entity_id="person-003",
                label="person",
                last_seen="2099-01-01T00:00:00Z",
                confidence=0.94,
                attributes={
                    "bbox": {
                        "x1": 100,
                        "y1": 80,
                        "x2": 260,
                        "y2": 380,
                    },
                    "image_width": 640,
                    "image_height": 480,
                },
                history=[
                    FakeObservation(
                        timestamp=(
                            "2099-01-01T00:00:00Z"
                        ),
                        source="vision",
                        confidence=0.94,
                        location={
                            "cx": 180,
                            "cy": 230,
                        },
                        attributes={},
                    )
                ],
            ),
            "person-009": FakeEntity(
                entity_id="person-009",
                label="person",
                last_seen="2099-01-01T00:00:01Z",
                confidence=0.99,
                attributes={
                    "bbox": {
                        "x1": 400,
                        "y1": 80,
                        "x2": 600,
                        "y2": 400,
                    },
                    "image_width": 640,
                    "image_height": 480,
                },
                history=[
                    FakeObservation(
                        timestamp=(
                            "2099-01-01T00:00:01Z"
                        ),
                        source="vision",
                        confidence=0.99,
                        location={
                            "cx": 500,
                            "cy": 240,
                        },
                        attributes={},
                    )
                ],
            ),
        }

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
            "confidence": 0.94,
            "cx": 180.0,
            "cy": 230.0,
            "area": 48000.0,
            "bbox": {
                "x1": 100,
                "y1": 80,
                "x2": 260,
                "y2": 380,
            },
            "image_width": 640.0,
            "image_height": 480.0,
        }

    def reload(self):
        self.reload_count += 1

    def get_entity(self, entity_id):
        self.entity_queries.append(entity_id)
        return self.entities.get(entity_id)


def main():
    world_model = FakeWorldModel()

    lock = TargetLock(
        world_model=world_model,
        max_age_seconds=3.0,
    )

    print("===== ACQUIRE TARGET =====")

    first = lock.resolve(
        mission_id="mission-follow-001",
        target_label="person",
    )

    print(first)
    print(lock.snapshot())

    assert first["entity_id"] == "person-003"
    assert lock.locked_entity_id == "person-003"
    assert lock.tracking_mode == "LOCKED"
    assert world_model.label_queries == 1
    assert world_model.entity_queries == []

    print()
    print("===== KEEP SAME TARGET =====")

    second = lock.resolve(
        mission_id="mission-follow-001",
        target_label="person",
    )

    print(second)
    print(lock.snapshot())

    assert second["entity_id"] == "person-003"
    assert second["cx"] == 180.0
    assert second["area"] == 48000.0

    # person-009 is newer and has higher confidence, but it must not replace
    # the locked person.
    assert lock.locked_entity_id == "person-003"
    assert world_model.label_queries == 1
    assert world_model.entity_queries == [
        "person-003",
    ]

    print()
    print("===== NEW MISSION RESETS LOCK =====")

    third = lock.resolve(
        mission_id="mission-follow-002",
        target_label="person",
    )

    print(third)
    print(lock.snapshot())

    assert world_model.label_queries == 2
    assert lock.locked_entity_id == "person-003"

    print()
    print("===== EXPLICIT RESET =====")

    lock.reset()
    print(lock.snapshot())

    assert lock.locked_entity_id is None
    assert lock.tracking_mode == "UNLOCKED"
    assert lock.locked_since is None

    print()
    print("PASS: target is acquired by label once")
    print("PASS: later cycles query only the locked entity ID")
    print("PASS: a newer person cannot steal the lock")
    print("PASS: a new mission starts with a fresh lock")
    print("PASS: STOP can explicitly reset the lock")
    print()
    print("Target Lock foundation test passed.")


if __name__ == "__main__":
    main()
