#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from target_lock import TargetLock


IDENTITY_ALPHA = "person-identity-alpha"
IDENTITY_BRAVO = "person-identity-bravo"


def now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.identity_queries = []
        self.lookup_mode = "MISSING"

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
            "identity_id": IDENTITY_ALPHA,
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
        self.identity_queries.append(identity_id)

        if self.lookup_mode == "MISSING":
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

        if self.lookup_mode == "DIFFERENT_IDENTITY":
            return {
                "found": True,
                "stale": False,
                "target": "person",
                "entity_id": "person-999",
                "identity_id": IDENTITY_BRAVO,
                "label": "person",
                "confidence": 0.99,
                "cx": 500.0,
                "cy": 230.0,
                "area": 50000.0,
                "image_width": 640,
                "image_height": 480,
                "last_seen": now_iso(),
            }

        if self.lookup_mode == "SAME_IDENTITY":
            return {
                "found": True,
                "stale": False,
                "target": "person",
                "entity_id": "person-019",
                "identity_id": IDENTITY_ALPHA,
                "label": "person",
                "confidence": 0.96,
                "cx": 410.0,
                "cy": 225.0,
                "area": 47000.0,
                "bbox": {
                    "x1": 340.0,
                    "y1": 65.0,
                    "x2": 480.0,
                    "y2": 385.0,
                },
                "image_width": 640,
                "image_height": 480,
                "last_seen": now_iso(),
                "identity_match_score": 0.93,
                "identity_status": "MATCHED",
                "identity_ambiguous": False,
            }

        raise AssertionError(
            f"Unexpected lookup mode: {self.lookup_mode}"
        )


def main():
    world_model = FakeWorldModel()

    lock = TargetLock(
        world_model=world_model,
        max_age_seconds=3.0,
        recovery_timeout_seconds=2.0,
    )

    mission_id = "mission-follow-001"

    print("===== ACQUIRE ORIGINAL IDENTITY =====")

    acquired = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(acquired)
    print(lock.snapshot())

    assert acquired["found"] is True
    assert lock.locked_entity_id == "person-003"
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert lock.tracking_mode == TargetLock.MODE_LOCKED
    assert world_model.label_queries == 1

    print(
        "PASS: original person and identity were acquired"
    )

    print()
    print("===== ENTER WAITING STATE =====")

    lock.lost_since = (
        datetime.now(timezone.utc)
        - timedelta(seconds=3.0)
    ).isoformat().replace("+00:00", "Z")

    waiting = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(waiting)
    print(lock.snapshot())

    assert waiting["found"] is False
    assert waiting["lock_expired"] is True
    assert waiting["identity_retained"] is True
    assert (
        waiting["tracking_mode"]
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert lock.waiting_since is not None
    assert world_model.label_queries == 1

    print(
        "PASS: predictive timeout preserved identity"
    )
    print(
        "PASS: TargetLock entered WAITING_FOR_IDENTITY"
    )

    print()
    print("===== SAME IDENTITY STILL ABSENT =====")

    identity_queries_before_absent = len(
        world_model.identity_queries
    )

    absent = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(absent)
    print(lock.snapshot())

    assert absent["found"] is False
    assert absent["identity_lookup_attempted"] is True
    assert absent["identity_reacquisition_pending"] is True
    assert (
        absent["tracking_mode"]
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert world_model.label_queries == 1
    assert len(
        world_model.identity_queries
    ) == identity_queries_before_absent + 1

    assert all(
        identity_id == IDENTITY_ALPHA
        for identity_id in world_model.identity_queries
    )

    print(
        "PASS: identity absence keeps TargetLock waiting"
    )
    print(
        "PASS: no label acquisition was attempted"
    )

    print()
    print("===== DIFFERENT IDENTITY IS VISIBLE =====")

    world_model.lookup_mode = "DIFFERENT_IDENTITY"

    identity_queries_before_mismatch = len(
        world_model.identity_queries
    )

    mismatch = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(mismatch)
    print(lock.snapshot())

    assert mismatch["found"] is False
    assert mismatch["identity_mismatch"] is True
    assert mismatch["reacquisition_blocked"] is True
    assert (
        mismatch["resolved_identity_id"]
        == IDENTITY_BRAVO
    )
    assert (
        mismatch["tracking_mode"]
        == TargetLock.MODE_WAITING_FOR_IDENTITY
    )

    assert lock.locked_entity_id is None
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert world_model.label_queries == 1

    assert len(
        world_model.identity_queries
    ) == identity_queries_before_mismatch + 1

    assert all(
        identity_id == IDENTITY_ALPHA
        for identity_id in world_model.identity_queries
    )

    print(
        "PASS: a different identity cannot satisfy the wait"
    )

    print()
    print("===== SAME IDENTITY REAPPEARS =====")

    world_model.lookup_mode = "SAME_IDENTITY"

    identity_queries_before_reacquire = len(
        world_model.identity_queries
    )

    reacquired = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(reacquired)
    print(lock.snapshot())

    assert reacquired["found"] is True
    assert reacquired["identity_reacquired"] is True
    assert reacquired["reacquisition_blocked"] is False
    assert (
        reacquired["identity_reacquisition_pending"]
        is False
    )
    assert (
        reacquired["reacquired_entity_id"]
        == "person-019"
    )
    assert (
        reacquired["tracking_mode"]
        == TargetLock.MODE_LOCKED
    )

    assert lock.locked_entity_id == "person-019"
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert lock.tracking_mode == TargetLock.MODE_LOCKED
    assert lock.waiting_since is None
    assert lock.lost_since is None

    assert world_model.label_queries == 1

    assert len(
        world_model.identity_queries
    ) == identity_queries_before_reacquire + 1

    assert all(
        identity_id == IDENTITY_ALPHA
        for identity_id in world_model.identity_queries
    )

    print(
        "PASS: same identity reacquired a new entity ID"
    )
    print(
        "PASS: tracking returned to LOCKED"
    )
    print(
        "PASS: label acquisition remained blocked"
    )

    print()
    print("===== NORMAL LOCKED CYCLE CONTINUES =====")

    continued = lock.resolve(
        mission_id=mission_id,
        target_label="person",
    )

    print(continued)
    print(lock.snapshot())

    assert continued["found"] is True
    assert lock.locked_entity_id == "person-019"
    assert lock.locked_identity_id == IDENTITY_ALPHA
    assert lock.tracking_mode == TargetLock.MODE_LOCKED
    assert world_model.label_queries == 1

    print(
        "PASS: normal identity tracking continued"
    )

    print()
    print(
        "TargetLock identity-only reacquisition test passed."
    )


if __name__ == "__main__":
    main()
