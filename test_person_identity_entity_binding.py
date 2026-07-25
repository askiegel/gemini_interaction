#!/usr/bin/env python3

from person_identity_manager import (
    PersonIdentityManager,
)


def person(
    entity_id,
    x1,
    y1,
    x2,
    y2,
):
    return {
        "label": "person",
        "entity_id": entity_id,
        "confidence": 0.90,
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
        "image_width": 640.0,
        "image_height": 480.0,
    }


manager = PersonIdentityManager()

print("===== INITIAL ENTITY =====")

first = manager.assign_identity(
    person(
        "person-001",
        40,
        80,
        180,
        430,
    )
)

identity_id = first["identity_id"]

assert first["identity_status"] == "NEW"

print(first)

print()
print("===== TRANSIENT ENTITY MIGRATION =====")

migrated = manager.assign_identity(
    person(
        "person-002",
        50,
        82,
        190,
        432,
    )
)

assert migrated["identity_id"] == identity_id
assert migrated["identity_status"] == "MATCHED"

print(migrated)

print()
print("===== ORIGINAL ENTITY RETURNS FAR AWAY =====")

returned = manager.assign_identity(
    person(
        "person-001",
        430,
        70,
        600,
        440,
    )
)

print(returned)

assert returned["identity_id"] == identity_id
assert returned["identity_status"] == "MATCHED"
assert returned["identity_ambiguous"] is False

print()
print("===== MIGRATED ENTITY RETURNS FAR AWAY =====")

migrated_return = manager.assign_identity(
    person(
        "person-002",
        20,
        75,
        175,
        435,
    )
)

print(migrated_return)

assert (
    migrated_return["identity_id"]
    == identity_id
)
assert (
    migrated_return["identity_status"]
    == "MATCHED"
)

assert (
    manager.entity_bindings["person-001"]
    == identity_id
)
assert (
    manager.entity_bindings["person-002"]
    == identity_id
)

assert len(manager.identities) == 1

print()
print(
    "PASS: One persistent identity retains "
    "multiple transient entity bindings."
)
print(
    "PASS: Returning entity IDs bypass "
    "ambiguous geometric rematching."
)
