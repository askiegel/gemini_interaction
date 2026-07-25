#!/usr/bin/env python3

from tracking_state import (
    build_tracking_state,
    empty_tracking_state,
)


print("===== EMPTY TRACKING STATE =====")

empty = empty_tracking_state()

assert empty["tracking_mode"] == "UNLOCKED"
assert empty["locked_entity_id"] is None
assert empty["locked_identity_id"] is None
assert empty["identity_id"] is None
assert empty["waiting_since"] is None
assert empty["waiting_age_seconds"] is None
assert empty["identity_reacquisition_pending"] is False
assert empty["identity_lookup_attempted"] is False
assert empty["identity_reacquired"] is False
assert empty["identity_mismatch"] is False
assert empty["reacquired_entity_id"] is None
assert empty["last_entity_migration"] is None
assert empty["identity_status"] is None
assert empty["identity_match_score"] is None
assert empty["identity_ambiguous"] is False
assert empty["identity_best_score"] is None
assert empty["identity_runner_up_score"] is None
assert empty["identity_score_margin"] is None
assert empty["identity_decision"] is None

print("PASS: empty state exposes identity contract")


print()
print("===== DIRECT IDENTITY FIELDS =====")

direct = build_tracking_state(
    {
        "behavior": "FOLLOW_PERSON",
        "state": "CENTERING_LEFT",
        "target": "person",
        "tracking_mode": "LOCKED",
        "locked_entity_id": "person-003",
        "locked_identity_id": "person-identity-alpha",
        "identity_id": "person-identity-alpha",
        "identity_status": "MATCHED",
        "identity_match_score": 0.94,
        "identity_ambiguous": False,
        "horizontal_error": -120.0,
        "target_area": 42000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {
            "x1": 100.0,
            "y1": 80.0,
            "x2": 260.0,
            "y2": 430.0,
        },
    }
)

assert direct["tracking_mode"] == "LOCKED"
assert direct["locked_entity_id"] == "person-003"
assert (
    direct["locked_identity_id"]
    == "person-identity-alpha"
)
assert direct["identity_id"] == "person-identity-alpha"
assert direct["identity_status"] == "MATCHED"
assert direct["identity_match_score"] == 0.94
assert direct["identity_best_score"] == 0.94
assert direct["identity_ambiguous"] is False

print("PASS: direct identity fields are preserved")


print()
print("===== NESTED IDENTITY DIAGNOSTICS =====")

nested = build_tracking_state(
    {
        "behavior": "FOLLOW_PERSON",
        "state": "CENTERING_RIGHT",
        "locked_entity_id": "person-027",
        "vision_result": {
            "found": True,
            "label": "person",
            "identity_id": "person-identity-bravo",
            "identity_status": "NEW_AMBIGUOUS",
            "identity_match_score": 0.61,
            "identity_ambiguous": True,
            "identity_diagnostics": {
                "best_score": 0.61,
                "second_score": 0.58,
                "score_margin": 0.03,
                "ambiguous": True,
                "decision": "NEW_AMBIGUOUS",
            },
            "cx": 410.0,
            "cy": 220.0,
            "area": 39000.0,
            "image_width": 640.0,
            "image_height": 480.0,
            "bbox": {
                "x1": 330.0,
                "y1": 80.0,
                "x2": 490.0,
                "y2": 430.0,
            },
        },
    }
)

assert nested["locked_entity_id"] == "person-027"
assert nested["identity_id"] == "person-identity-bravo"
assert nested["identity_status"] == "NEW_AMBIGUOUS"
assert nested["identity_match_score"] == 0.61
assert nested["identity_ambiguous"] is True
assert nested["identity_best_score"] == 0.61
assert nested["identity_runner_up_score"] == 0.58
assert nested["identity_score_margin"] == 0.03
assert nested["identity_decision"] == "NEW_AMBIGUOUS"

print("PASS: nested diagnostics reach public tracking state")


print()
print("===== WAITING FOR IDENTITY TELEMETRY =====")

waiting = build_tracking_state(
    {
        "behavior": "FOLLOW_PERSON",
        "state": "TARGET_LOST",
        "target": "person",
        "tracking_mode": "WAITING_FOR_IDENTITY",
        "locked_entity_id": None,
        "locked_identity_id": "person-identity-alpha",
        "identity_id": "person-identity-alpha",
        "waiting_since": "2026-07-25T18:10:00Z",
        "waiting_age_seconds": 1.7,
        "identity_reacquisition_pending": True,
        "identity_lookup_attempted": True,
        "identity_reacquired": False,
        "identity_mismatch": True,
        "reacquired_entity_id": None,
        "last_entity_migration": {
            "previous_entity_id": "person-003",
            "new_entity_id": None,
            "identity_id": "person-identity-alpha",
        },
    },
    previous=direct,
)

assert waiting["tracking_mode"] == "WAITING_FOR_IDENTITY"
assert waiting["locked_entity_id"] is None
assert (
    waiting["locked_identity_id"]
    == "person-identity-alpha"
)
assert waiting["waiting_since"] == "2026-07-25T18:10:00Z"
assert waiting["waiting_age_seconds"] == 1.7
assert waiting["identity_reacquisition_pending"] is True
assert waiting["identity_lookup_attempted"] is True
assert waiting["identity_reacquired"] is False
assert waiting["identity_mismatch"] is True
assert waiting["reacquired_entity_id"] is None
assert waiting["last_entity_migration"] == {
    "previous_entity_id": "person-003",
    "new_entity_id": None,
    "identity_id": "person-identity-alpha",
}

print("PASS: TargetLock lifecycle telemetry is preserved")


print()
print("===== STOP CLEARS IDENTITY =====")

stopped = build_tracking_state(
    {
        "behavior": "STOP",
        "state": "STOPPED",
    },
    previous=nested,
)

assert stopped["state"] == "STOPPED"
assert stopped["tracking_mode"] == "UNLOCKED"
assert stopped["identity_id"] is None
assert stopped["locked_identity_id"] is None
assert stopped["locked_entity_id"] is None
assert stopped["waiting_since"] is None
assert stopped["waiting_age_seconds"] is None
assert stopped["identity_reacquisition_pending"] is False
assert stopped["identity_lookup_attempted"] is False
assert stopped["identity_reacquired"] is False
assert stopped["identity_mismatch"] is False
assert stopped["reacquired_entity_id"] is None
assert stopped["last_entity_migration"] is None
assert stopped["identity_ambiguous"] is False

print("PASS: STOP clears identity telemetry")
print()
print("All TrackingState identity tests passed.")
