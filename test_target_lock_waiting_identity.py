#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from target_lock import TargetLock


def now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.identity_queries = 0
        self.identity_visible = True

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
            "identity_id": "person-identity-alpha",
            "label": "person",
            "confidence": 0.94,
            "cx": 180.0,
            "cy": 230.0,
            "area": 46000.0,
            "bbox": {
                "x1": 100.0,
                "y1": 70.0,
                "x2": 260.0,
                "y2": 370.0,
            },
            "image_width": 640,
            "image_height": 480,
            "last_seen": now_iso(),
        }

    def find_latest_entity_by_identity(
        self,
        identity_id,
        max_age_seconds=None,
        refresh=True,
    ):
        self.identity_queries += 1

        if not self.identity_visible:
            return {
                "found": False,
                "stale": False,
                "target": "person",
                "entity_id": None,
                "identity_id": identity_id,
                "reason": (
                    "Selected identity is not currently visible."
                ),
            }

        return {
            "found": True,
            "stale": False,
            "target": "person",
            "entity_id": "person-003",
            "identity_id": identity_id,
            "label": "person",
            "confidence": 0.94,
            "cx": 180.0,
            "cy": 230.0,
            "area": 46000.0,
            "bbox": {
                "x1": 100.0,
                "y1": 70.0,
                "x2": 260.0,
                "y2": 370.0,
            },
            "image_width": 640,
            "image_height": 480,
            "last_seen": now_iso(),
        }


def main():
    world_model = FakeWorldModel()

    lock = TargetLock(
        world_model=world_model,
        max_age_seconds=3.0,
        recovery_timeout_seconds=2.0,
    )

    print("===== INITIAL IDENTITY ACQUISITION =====")

    acquired = lock.resolve(
        mission_id="mission-follow-001",
        target_label="person",
    )

    print(acquired)
    print(lock.snapshot())

    assert acquired["found"] is True
    assert lock.locked_entity_id == "person-003"
    assert (
        lock.locked_identity_id
        == "person-identity-alpha"
    )
    assert lock.tracking_mode == TargetLock.MODE_LOCKED
    assert world_model.label_queries == 1

    print(
        "PASS: person and persistent identity "
        "were acquired"
    )

    print()
    print("===== FORCE PREDICTIVE RECOVERY TIMEOUT =====")

    world_model.identity_visible = False

    lock.lost_since = (
        datetime.now(timezone.utc)
        - timedelta(seconds=3.0)
    ).isoformat().replace("+00:00", "Z")

    waiting = lock.resolve(
        mission_id="mission-follow-001",
        target_label="person",
    )

    print(waiting)
    print(lock.snapshot())

    assert waiting["found"] is False
    assert waiting["lock_expired"] is True
    assert waiting["identity_lost"] is True
    assert waiting["identity_retained"] is True
    assert (
        waiting["tracking_mode"]
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )
    assert waiting["expired_entity_id"] == "person-003"
    assert waiting["entity_id"] is None
    assert (
        waiting["identity_id"]
        == "person-identity-alpha"
    )
    assert waiting["label_reacquisition_blocked"] is True
    assert (
        waiting["identity_reacquisition_pending"]
        is True
    )

    assert lock.locked_entity_id is None
    assert (
        lock.locked_identity_id
        == "person-identity-alpha"
    )
    assert (
        lock.tracking_mode
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )
    assert lock.waiting_since is not None
    assert world_model.label_queries == 1

    print(
        "PASS: transient entity lock expired"
    )
    print(
        "PASS: persistent identity remained selected"
    )
    print(
        "PASS: TargetLock entered WAITING_FOR_IDENTITY"
    )

    print()
    print("===== REPEATED WAITING CYCLE =====")

    label_queries_before = world_model.label_queries
    identity_queries_before = world_model.identity_queries

    repeated = lock.resolve(
        mission_id="mission-follow-001",
        target_label="person",
    )

    print(repeated)
    print(lock.snapshot())

    assert repeated["found"] is False
    assert (
        repeated["tracking_mode"]
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )
    assert repeated["identity_retained"] is True
    assert repeated["label_reacquisition_blocked"] is True

    assert (
        world_model.label_queries
        == label_queries_before
    )
    assert (
        world_model.identity_queries
        == identity_queries_before
    )

    print(
        "PASS: repeated waiting did not acquire by label"
    )
    print(
        "PASS: Phase 3A does not yet attempt identity "
        "reacquisition"
    )

    print()
    print("===== RESET CLEARS WAITING IDENTITY =====")

    lock.reset()

    snapshot = lock.snapshot()
    print(snapshot)

    assert snapshot["tracking_mode"] == TargetLock.MODE_UNLOCKED
    assert snapshot["locked_entity_id"] is None
    assert snapshot["locked_identity_id"] is None
    assert snapshot["waiting_since"] is None

    print(
        "PASS: STOP or mission reset clears waiting state"
    )

    print()
    print(
        "TargetLock WAITING_FOR_IDENTITY state test passed."
    )


if __name__ == "__main__":
    main()
