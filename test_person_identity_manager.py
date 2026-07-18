#!/usr/bin/env python3

from person_identity_manager import (
    PersonIdentityManager,
)


def person(
    x1,
    y1,
    x2,
    y2,
    entity_id=None,
    confidence=0.90,
):
    return {
        "label": "person",
        "confidence": confidence,
        "entity_id": entity_id,
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
        "image_width": 640.0,
        "image_height": 480.0,
    }


print("===== SAME PERSON CONTINUITY =====")

manager = PersonIdentityManager()

first = manager.assign_identity(
    person(
        220,
        80,
        380,
        430,
        entity_id="person-001",
    )
)

second = manager.assign_identity(
    person(
        230,
        82,
        390,
        432,
        entity_id="person-001",
    )
)

print(first)
print(second)

assert first["identity_id"]
assert second["identity_id"] == (
    first["identity_id"]
)
assert second["identity_status"] == "MATCHED"

print("PASS: nearby observations retain identity")

print()
print("===== TRANSIENT ENTITY ID CHANGE =====")

third = manager.assign_identity(
    person(
        238,
        84,
        398,
        434,
        entity_id="person-027",
    )
)

print(third)

assert third["identity_id"] == (
    first["identity_id"]
)

print(
    "PASS: persistent identity survives "
    "transient entity-ID change"
)

print()
print("===== DIFFERENT PERSON SEPARATION =====")

different = manager.assign_identity(
    person(
        20,
        90,
        145,
        420,
        entity_id="person-002",
    )
)

print(different)

assert different["identity_id"] != (
    first["identity_id"]
)

print(
    "PASS: spatially separate person "
    "receives another identity"
)

print()
print("===== SIMULTANEOUS FRAME ASSIGNMENT =====")

frame_manager = PersonIdentityManager()

initial_frame = (
    frame_manager.assign_identities(
        [
            person(
                40,
                80,
                180,
                430,
                entity_id="person-001",
            ),
            person(
                440,
                80,
                590,
                430,
                entity_id="person-002",
            ),
        ]
    )
)

next_frame = frame_manager.assign_identities(
    [
        person(
            50,
            82,
            190,
            432,
            entity_id="person-011",
        ),
        person(
            430,
            82,
            580,
            432,
            entity_id="person-012",
        ),
    ]
)

print(initial_frame)
print(next_frame)

assert (
    initial_frame[0]["identity_id"]
    != initial_frame[1]["identity_id"]
)

assert (
    next_frame[0]["identity_id"]
    == initial_frame[0]["identity_id"]
)

assert (
    next_frame[1]["identity_id"]
    == initial_frame[1]["identity_id"]
)

print(
    "PASS: two people retain separate "
    "identities across frames"
)

print()
print("===== NON-PERSON DETECTION =====")

chair = manager.assign_identity(
    {
        "label": "chair",
        "confidence": 0.88,
        "bbox": [
            200,
            200,
            400,
            450,
        ],
        "image_width": 640,
        "image_height": 480,
    }
)

print(chair)

assert chair["identity_id"] is None
assert (
    chair["identity_status"]
    == "NOT_A_PERSON"
)

print("PASS: non-person detections are ignored")

print()
print("===== IDENTITY SNAPSHOT =====")

snapshot = manager.get_identity(
    first["identity_id"]
)

print(snapshot)

assert snapshot is not None
assert (
    snapshot["observation_count"]
    == 3
)

print("PASS: identity state is inspectable")

print()
print(
    "All PersonIdentityManager tests passed."
)
