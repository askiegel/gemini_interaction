#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from world.model import (
    EntityObservation,
    WorldEntity,
    WorldModel,
)


IDENTITY_ALPHA = "person-identity-alpha"
IDENTITY_BRAVO = "person-identity-bravo"


def iso_seconds_ago(seconds):
    return (
        datetime.now(timezone.utc)
        - timedelta(seconds=float(seconds))
    ).isoformat().replace("+00:00", "Z")


def make_person(
    entity_id,
    identity_id,
    seconds_ago,
    cx,
    confidence,
):
    timestamp = iso_seconds_ago(seconds_ago)

    attributes = {
        "identity_id": identity_id,
        "identity_match_score": 0.91,
        "identity_status": "MATCHED",
        "identity_ambiguous": False,
        "bbox": {
            "x1": cx - 40.0,
            "y1": 80.0,
            "x2": cx + 40.0,
            "y2": 360.0,
        },
        "image_width": 640,
        "image_height": 480,
    }

    observation = EntityObservation(
        timestamp=timestamp,
        source="test",
        confidence=confidence,
        location={
            "cx": cx,
            "cy": 220.0,
        },
        attributes=dict(attributes),
    )

    return WorldEntity(
        entity_id=entity_id,
        label="person",
        entity_type="human",
        first_seen=timestamp,
        last_seen=timestamp,
        confidence=confidence,
        attributes=dict(attributes),
        history=[observation],
    )


def main():
    with TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "world.json"
        world_model = WorldModel(
            storage_path=str(storage_path)
        )

        old_alpha = make_person(
            entity_id="person-003",
            identity_id=IDENTITY_ALPHA,
            seconds_ago=1.0,
            cx=150.0,
            confidence=0.82,
        )

        new_alpha = make_person(
            entity_id="person-019",
            identity_id=IDENTITY_ALPHA,
            seconds_ago=0.1,
            cx=410.0,
            confidence=0.94,
        )

        newer_wrong_identity = make_person(
            entity_id="person-025",
            identity_id=IDENTITY_BRAVO,
            seconds_ago=0.01,
            cx=560.0,
            confidence=0.99,
        )

        world_model.entities = {
            old_alpha.entity_id: old_alpha,
            new_alpha.entity_id: new_alpha,
            newer_wrong_identity.entity_id: (
                newer_wrong_identity
            ),
        }

        print("===== EMPTY IDENTITY =====")

        empty = (
            world_model
            .find_latest_entity_by_identity(
                "",
                refresh=False,
            )
        )

        print(empty)

        assert empty["found"] is False
        assert empty["identity_id"] is None

        print("PASS: empty identity is rejected")

        print()
        print("===== UNKNOWN IDENTITY =====")

        missing = (
            world_model
            .find_latest_entity_by_identity(
                "person-identity-missing",
                refresh=False,
            )
        )

        print(missing)

        assert missing["found"] is False
        assert missing["stale"] is False
        assert (
            missing["identity_id"]
            == "person-identity-missing"
        )

        print("PASS: unknown identity returns no match")

        print()
        print("===== NEWEST MATCHING IDENTITY =====")

        result = (
            world_model
            .find_latest_entity_by_identity(
                IDENTITY_ALPHA,
                max_age_seconds=3.0,
                refresh=False,
            )
        )

        print(result)

        assert result["found"] is True
        assert result["stale"] is False
        assert result["identity_id"] == IDENTITY_ALPHA

        # The newer entity for identity alpha must replace
        # the older transient entity ID.
        assert result["entity_id"] == "person-019"
        assert result["cx"] == 410.0
        assert result["confidence"] == 0.94

        # The newer person belonging to identity bravo must
        # never be selected.
        assert result["entity_id"] != "person-025"

        assert result["identity_status"] == "MATCHED"
        assert result["identity_match_score"] == 0.91
        assert result["identity_ambiguous"] is False
        assert result["bbox"] == {
            "x1": 370.0,
            "y1": 80.0,
            "x2": 450.0,
            "y2": 360.0,
        }

        print(
            "PASS: newest entity for the requested identity "
            "is selected"
        )
        print(
            "PASS: another identity cannot steal the result"
        )

        print()
        print("===== STALE MATCH =====")

        stale_person = make_person(
            entity_id="person-030",
            identity_id="person-identity-stale",
            seconds_ago=10.0,
            cx=320.0,
            confidence=0.88,
        )

        world_model.entities[
            stale_person.entity_id
        ] = stale_person

        stale = (
            world_model
            .find_latest_entity_by_identity(
                "person-identity-stale",
                max_age_seconds=1.0,
                refresh=False,
            )
        )

        print(stale)

        assert stale["found"] is False
        assert stale["stale"] is True
        assert stale["entity_id"] == "person-030"
        assert (
            stale["identity_id"]
            == "person-identity-stale"
        )
        assert stale.get("reason")

        print("PASS: stale identity observations are blocked")

    print()
    print(
        "World Model identity lookup test passed."
    )


if __name__ == "__main__":
    main()
