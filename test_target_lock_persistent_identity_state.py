#!/usr/bin/env python3

from dataclasses import dataclass

from target_lock import TargetLock


IDENTITY_ID = "person-identity-alpha"
ENTITY_ID = "person-003"


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

        self.entity = FakeEntity(
            entity_id=ENTITY_ID,
            label="person",
            last_seen="2099-01-01T00:00:00Z",
            confidence=0.94,
            attributes={
                "bbox": {
                    "x1": 100.0,
                    "y1": 80.0,
                    "x2": 260.0,
                    "y2": 380.0,
                },
                "area": 48000.0,
                "image_width": 640.0,
                "image_height": 480.0,
                "identity_id": IDENTITY_ID,
                "identity_match_score": 0.91,
                "identity_status": "MATCHED",
                "identity_ambiguous": False,
            },
            history=[
                FakeObservation(
                    timestamp="2099-01-01T00:00:00Z",
                    source="vision",
                    confidence=0.94,
                    location={
                        "cx": 180.0,
                        "cy": 230.0,
                    },
                    attributes={},
                )
            ],
        )

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
            "entity_id": ENTITY_ID,
            "identity_id": IDENTITY_ID,
            "identity_match_score": 0.91,
            "identity_status": "MATCHED",
            "identity_ambiguous": False,
            "label": "person",
            "confidence": 0.94,
            "cx": 180.0,
            "cy": 230.0,
            "area": 48000.0,
            "bbox": {
                "x1": 100.0,
                "y1": 80.0,
                "x2": 260.0,
                "y2": 380.0,
            },
            "image_width": 640.0,
            "image_height": 480.0,
        }

    def reload(self):
        return None

    def get_entity(self, entity_id):
        self.entity_queries.append(entity_id)

        if entity_id == ENTITY_ID:
            return self.entity

        return None


def main():
    world_model = FakeWorldModel()

    lock = TargetLock(
        world_model=world_model,
        max_age_seconds=3.0,
    )

    print("===== INITIAL STATE =====")

    initial = lock.snapshot()
    print(initial)

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id is None
    assert initial["locked_entity_id"] is None
    assert initial["locked_identity_id"] is None

    print("PASS: unlocked state contains no identity lock")

    print()
    print("===== ACQUIRE TARGET =====")

    first = lock.resolve(
        mission_id="mission-follow-alpha",
        target_label="person",
    )

    first_snapshot = lock.snapshot()

    print(first)
    print(first_snapshot)

    assert first["found"] is True
    assert first["entity_id"] == ENTITY_ID
    assert first["identity_id"] == IDENTITY_ID
    assert lock.locked_entity_id == ENTITY_ID
    assert lock.locked_identity_id == IDENTITY_ID
    assert first_snapshot["locked_entity_id"] == ENTITY_ID
    assert (
        first_snapshot["locked_identity_id"]
        == IDENTITY_ID
    )

    print("PASS: acquisition stores entity and identity IDs")

    print()
    print("===== LOCKED ENTITY CYCLE =====")

    second = lock.resolve(
        mission_id="mission-follow-alpha",
        target_label="person",
    )

    second_snapshot = lock.snapshot()

    print(second)
    print(second_snapshot)

    assert second["found"] is True
    assert lock.locked_entity_id == ENTITY_ID
    assert lock.locked_identity_id == IDENTITY_ID
    assert (
        second_snapshot["locked_identity_id"]
        == IDENTITY_ID
    )
    assert world_model.label_queries == 1
    assert world_model.entity_queries == [ENTITY_ID]

    print("PASS: identity state survives locked cycles")
    print("PASS: lookup behavior remains entity-based")

    print()
    print("===== RELEASE LOCK =====")

    lock._release_lock()

    released = lock.snapshot()
    print(released)

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id is None
    assert released["locked_entity_id"] is None
    assert released["locked_identity_id"] is None

    print("PASS: lock release clears identity state")

    print()
    print("===== REACQUIRE AND RESET =====")

    lock.resolve(
        mission_id="mission-follow-bravo",
        target_label="person",
    )

    assert lock.locked_identity_id == IDENTITY_ID

    lock.reset()

    reset = lock.snapshot()
    print(reset)

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id is None
    assert reset["locked_entity_id"] is None
    assert reset["locked_identity_id"] is None

    print("PASS: explicit reset clears identity state")
    print()
    print("TargetLock persistent identity state test passed.")


if __name__ == "__main__":
    main()
