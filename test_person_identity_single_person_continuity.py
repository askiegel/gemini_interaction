#!/usr/bin/env python3

from person_identity_manager import PersonIdentityManager


def person(entity_id, x1, y1, x2, y2):
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

first = manager.assign_identities([
    person("person-001", 220, 80, 380, 430),
])[0]

identity_id = first["identity_id"]

second = manager.assign_identities([
    person("person-001", 225, 82, 385, 432),
])[0]

assert second["identity_id"] == identity_id

# Reproduce historical identity pollution from the live failure.
manager._create_identity(
    manager._normalize_detection(
        person("person-old", 226, 82, 386, 432)
    )
)

migrated = manager.assign_identities([
    person("person-002", 230, 84, 390, 434),
])[0]

print("original_identity:", identity_id)
print("migrated:", migrated)
print("identity_count:", len(manager.identities))

assert migrated["identity_id"] == identity_id, (
    "A continuously observed single person changed identity "
    "when its transient Entity Registry ID changed."
)

assert migrated["identity_status"] in {
    "MATCHED",
    "MATCHED_HYSTERESIS",
    "MATCHED_SINGLE_PERSON_CONTINUITY",
}

# Simultaneous people must still receive distinct identities.
multiple = manager.assign_identities([
    person("person-003", 80, 80, 220, 430),
    person("person-004", 420, 80, 560, 430),
])

multiple_ids = {
    detection["identity_id"]
    for detection in multiple
}

assert len(multiple_ids) == 2, (
    "Two simultaneous people shared one persistent identity."
)

print()
print("PASS: Single-person entity migration retained identity.")
print("PASS: Simultaneous people remained identity-distinct.")
