#!/usr/bin/env python3

from tracking_state import (
    build_tracking_state,
    empty_tracking_state,
)


print("===== EMPTY TRACKING STATE =====")

empty = empty_tracking_state()

assert empty["locked_entity_id"] is None
assert empty["identity_id"] is None
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
        "locked_entity_id": "person-003",
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

assert direct["locked_entity_id"] == "person-003"
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
print("===== STOP CLEARS IDENTITY =====")

stopped = build_tracking_state(
    {
        "behavior": "STOP",
        "state": "STOPPED",
    },
    previous=nested,
)

assert stopped["state"] == "STOPPED"
assert stopped["identity_id"] is None
assert stopped["locked_entity_id"] is None
assert stopped["identity_ambiguous"] is False

print("PASS: STOP clears identity telemetry")
print()
print("All TrackingState identity tests passed.")
