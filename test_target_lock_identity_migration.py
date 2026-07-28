#!/usr/bin/env python3

import inspect

from target_lock import TargetLock


IDENTITY_ALPHA = "person-identity-alpha"
IDENTITY_BRAVO = "person-identity-bravo"


class IdentityMigrationWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.identity_queries = 0
        self.entity_queries = 0
        self.phase = "acquire"

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
            "target": "person",
            "entity_id": "person-003",
            "identity_id": IDENTITY_ALPHA,
            "label": "person",
            "confidence": 0.91,
            "cx": 160.0,
            "cy": 220.0,
            "area": 42000.0,
            "bbox": {
                "x1": 100.0,
                "y1": 70.0,
                "x2": 220.0,
                "y2": 370.0,
            },
            "image_width": 640,
            "image_height": 480,
        }

    def find_latest_entity_by_identity(
        self,
        identity_id,
        max_age_seconds=None,
        refresh=True,
    ):
        self.identity_queries += 1

        assert identity_id == IDENTITY_ALPHA

        return {
            "found": True,
            "stale": False,
            "target": "person",
            "entity_id": "person-019",
            "identity_id": IDENTITY_ALPHA,
            "label": "person",
            "confidence": 0.96,
            "cx": 400.0,
            "cy": 225.0,
            "area": 46000.0,
            "bbox": {
                "x1": 335.0,
                "y1": 65.0,
                "x2": 465.0,
                "y2": 385.0,
            },
            "image_width": 640,
            "image_height": 480,
            "identity_match_score": 0.93,
            "identity_status": "MATCHED",
            "identity_ambiguous": False,
        }

    def reload(self):
        return None

    def get_entity(self, entity_id):
        self.entity_queries += 1

        # The previously locked transient entity is unavailable, so
        # TargetLock continues through persistent-identity lookup.
        return None


class MismatchedIdentityWorldModel(
    IdentityMigrationWorldModel
):
    def find_latest_entity_by_identity(
        self,
        identity_id,
        max_age_seconds=None,
        refresh=True,
    ):
        self.identity_queries += 1

        return {
            "found": True,
            "stale": False,
            "target": "person",
            "entity_id": "person-025",
            "identity_id": IDENTITY_BRAVO,
            "label": "person",
            "confidence": 0.99,
            "cx": 540.0,
            "cy": 220.0,
            "area": 50000.0,
            "bbox": {
                "x1": 470.0,
                "y1": 60.0,
                "x2": 610.0,
                "y2": 390.0,
            },
            "image_width": 640,
            "image_height": 480,
        }


def create_lock(world_model):
    signature = inspect.signature(TargetLock)

    kwargs = {}

    if "world_model" in signature.parameters:
        kwargs["world_model"] = world_model

    if "max_age_seconds" in signature.parameters:
        kwargs["max_age_seconds"] = 3.0

    if (
        "recovery_timeout_seconds"
        in signature.parameters
    ):
        kwargs["recovery_timeout_seconds"] = 2.0

    if kwargs.get("world_model") is world_model:
        return TargetLock(**kwargs)

    return TargetLock(world_model, **kwargs)


def test_same_identity_migrates_entity():
    world_model = IdentityMigrationWorldModel()
    lock = create_lock(world_model)

    print("===== INITIAL ACQUISITION =====")

    first = lock.resolve(
        mission_id="mission-alpha",
        target_label="person",
    )

    print(first)
    print(lock.snapshot())

    assert first["found"] is True
    assert lock.locked_entity_id == "person-003"
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert world_model.label_queries == 1
    assert world_model.identity_queries == 0

    print("PASS: initial mission acquires by label once")

    print()
    print("===== IDENTITY ENTITY MIGRATION =====")

    second = lock.resolve(
        mission_id="mission-alpha",
        target_label="person",
    )

    print(second)
    print(lock.snapshot())

    assert second["found"] is True
    assert second["identity_id"] == IDENTITY_ALPHA
    assert second["entity_id"] == "person-019"

    assert second["entity_migrated"] is True
    assert (
        second["previous_entity_id"]
        == "person-003"
    )
    assert second["new_entity_id"] == "person-019"

    assert lock.locked_entity_id == "person-019"
    assert lock.locked_identity_id == IDENTITY_ALPHA

    assert world_model.label_queries == 1
    assert world_model.identity_queries == 1
    assert world_model.entity_queries == 1

    migration = (
        lock.snapshot()["last_entity_migration"]
    )

    assert migration is not None
    assert migration["identity_id"] == IDENTITY_ALPHA
    assert (
        migration["previous_entity_id"]
        == "person-003"
    )
    assert migration["new_entity_id"] == "person-019"
    assert migration["timestamp"]

    print(
        "PASS: same persistent identity migrated from "
        "person-003 to person-019"
    )
    print(
        "PASS: label reacquisition was not used"
    )
    print(
        "PASS: migration telemetry was recorded"
    )

    print()
    print("===== SAME ENTITY REMAINS LOCKED =====")

    third = lock.resolve(
        mission_id="mission-alpha",
        target_label="person",
    )

    print(third)

    assert third["found"] is True
    assert third["entity_id"] == "person-019"
    assert third["entity_migrated"] is False
    assert third["previous_entity_id"] is None
    assert third["new_entity_id"] is None

    assert (
        lock.snapshot()["last_entity_migration"]
        == migration
    )

    print(
        "PASS: repeated identity resolution does not "
        "report a false migration"
    )


def test_different_identity_is_blocked():
    world_model = MismatchedIdentityWorldModel()
    lock = create_lock(world_model)

    print()
    print("===== ACQUIRE IDENTITY ALPHA =====")

    first = lock.resolve(
        mission_id="mission-bravo",
        target_label="person",
    )

    print(first)

    assert first["found"] is True
    assert lock.locked_entity_id == "person-003"
    assert lock.locked_identity_id == IDENTITY_ALPHA

    print()
    print("===== BLOCK IDENTITY BRAVO =====")

    blocked = lock.resolve(
        mission_id="mission-bravo",
        target_label="person",
    )

    print(blocked)
    print(lock.snapshot())

    assert blocked["found"] is False
    assert blocked["identity_mismatch"] is True
    assert blocked["reacquisition_blocked"] is True

    assert lock.locked_entity_id == "person-003"
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert (
        lock.snapshot()["last_entity_migration"]
        is None
    )

    assert world_model.label_queries == 1
    assert world_model.identity_queries == 1
    assert world_model.entity_queries == 1

    print(
        "PASS: a different persistent identity cannot "
        "steal the target lock"
    )


def main():
    test_same_identity_migrates_entity()
    test_different_identity_is_blocked()

    print()
    print(
        "TargetLock identity migration test passed."
    )


if __name__ == "__main__":
    main()
