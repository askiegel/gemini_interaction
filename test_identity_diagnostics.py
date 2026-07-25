#!/usr/bin/env python3

import io
import json
import os
from contextlib import redirect_stdout

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


print("===== DIAGNOSTIC CREATION =====")

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

first_diagnostic = first["identity_diagnostics"]

assert first["identity_status"] == "NEW"
assert first_diagnostic["decision"] == "NEW"
assert first_diagnostic["candidates"] == []
assert (
    "No eligible existing identity"
    in first_diagnostic["reason"]
)

print("PASS: first observation explains new identity")

print()
print("===== CANDIDATE SCORE BREAKDOWN =====")

second = manager.assign_identity(
    person(
        230,
        82,
        390,
        432,
        entity_id="person-001",
    )
)

diagnostic = second["identity_diagnostics"]

assert second["identity_status"] == "MATCHED"
assert diagnostic["decision"] == "MATCHED"
assert diagnostic["winner_identity_id"] == (
    first["identity_id"]
)
assert diagnostic["runner_up_identity_id"] is None
assert diagnostic["best_score"] == (
    second["identity_match_score"]
)
assert diagnostic["minimum_match_score"] == round(
    manager.minimum_match_score,
    4,
)
assert diagnostic["minimum_score_margin"] == round(
    manager.minimum_score_margin,
    4,
)
assert len(diagnostic["candidates"]) == 1

candidate = diagnostic["candidates"][0]

required_candidate_fields = {
    "identity_id",
    "age_seconds",
    "center_distance_ratio",
    "center_score",
    "iou_score",
    "area_score",
    "temporal_score",
    "transient_entity_bonus",
    "weak_spatial_match_penalty_applied",
    "final_score",
}

assert required_candidate_fields.issubset(
    candidate.keys()
)

assert candidate["identity_id"] == (
    first["identity_id"]
)
assert candidate["final_score"] == (
    diagnostic["best_score"]
)

print(json.dumps(diagnostic, indent=2))
print("PASS: score components and ranking are exposed")

print()
print("===== AMBIGUITY EXPLANATION =====")

ambiguous_manager = PersonIdentityManager(
    minimum_match_score=0.10,
    minimum_score_margin=0.50,
)

# Seed two distinct existing identities directly. This isolates the
# ambiguity test from the normal process used to create new identities.
left_detection = ambiguous_manager._normalize_detection(
    person(
        40,
        80,
        180,
        430,
        entity_id="person-left",
    )
)

right_detection = ambiguous_manager._normalize_detection(
    person(
        460,
        80,
        600,
        430,
        entity_id="person-right",
    )
)

left_identity = ambiguous_manager._create_identity(
    left_detection
)

right_identity = ambiguous_manager._create_identity(
    right_detection
)

assert (
    left_identity.identity_id
    != right_identity.identity_id
)
assert len(ambiguous_manager.identities) == 2

# This midpoint detection should produce nearly equal scores for the
# two seeded identities. The intentionally large minimum score margin
# therefore forces a NEW_AMBIGUOUS decision.
ambiguous = ambiguous_manager.assign_identity(
    person(
        250,
        80,
        390,
        430,
        entity_id="person-middle",
    )
)

ambiguous_diagnostic = (
    ambiguous["identity_diagnostics"]
)

assert ambiguous["identity_status"] == (
    "NEW_AMBIGUOUS"
)
assert ambiguous["identity_ambiguous"] is True
assert ambiguous_diagnostic["ambiguous"] is True
assert ambiguous_diagnostic["decision"] == (
    "NEW_AMBIGUOUS"
)
assert len(
    ambiguous_diagnostic["candidates"]
) >= 2
assert (
    ambiguous_diagnostic["score_margin"]
    < ambiguous_diagnostic[
        "minimum_score_margin"
    ]
)
assert (
    "required score margin"
    in ambiguous_diagnostic["reason"]
)

candidate_ids = {
    candidate["identity_id"]
    for candidate
    in ambiguous_diagnostic["candidates"]
}

assert left_identity.identity_id in candidate_ids
assert right_identity.identity_id in candidate_ids

print(json.dumps(
    ambiguous_diagnostic,
    indent=2,
))
print("PASS: ambiguous decisions explain the margin")

print()
print("===== OPTIONAL DEBUG OUTPUT =====")

previous_debug = os.environ.get("IDENTITY_DEBUG")
os.environ["IDENTITY_DEBUG"] = "1"

debug_output = io.StringIO()

with redirect_stdout(debug_output):
    manager.assign_identity(
        person(
            235,
            84,
            395,
            434,
            entity_id="person-001",
        )
    )

if previous_debug is None:
    os.environ.pop("IDENTITY_DEBUG", None)
else:
    os.environ["IDENTITY_DEBUG"] = previous_debug

printed = debug_output.getvalue()

assert "[IDENTITY_DIAGNOSTIC]" in printed
assert '"decision": "MATCHED"' in printed
assert '"entity_id": "person-001"' in printed
assert '"assigned_identity_id":' in printed
assert '"best_candidate_identity_id":' in printed
assert '"previous_identity_ids":' in printed
assert '"match_score":' in printed
assert '"threshold":' in printed
assert '"hysteresis_threshold":' in printed
assert '"runner_up_score":' in printed
assert first["identity_id"] in printed

print(printed.strip())
print("PASS: IDENTITY_DEBUG emits JSON diagnostics")

print()
print("===== NON-PERSON EXPLANATION =====")

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

assert chair["identity_status"] == "NOT_A_PERSON"
assert (
    chair["identity_diagnostics"]["decision"]
    == "NOT_A_PERSON"
)

print("PASS: ignored objects are explained")

print()
print("All identity diagnostic tests passed.")
