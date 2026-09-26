from copy import deepcopy

from marvin_preview_reacquisition import (
    evaluate_marvin_preview_reacquisition,
)


STAMP = "2026-09-25T12:00:00+00:00"


def preview(**updates):
    value = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target": "marvin",
        "identity_confirmed": True,  # semantic candidate confirmation only
        "source_timestamp": STAMP,
        "tracking": {
            "target_label": "marvin",
            "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        },
    }
    value.update(updates)
    return value


def lock(identity_id="marvin-1", mode="WAITING_FOR_IDENTITY"):
    return {
        "tracking_mode": mode,
        "locked_identity_id": identity_id,
        "locked_entity_id": None,
    }


def evaluate(candidate, selected="marvin-1", **kwargs):
    return evaluate_marvin_preview_reacquisition(
        candidate,
        lock(selected),
        now=STAMP,
        **kwargs,
    )


def test_exact_identity_reacquires_waiting_target_lock():
    result = evaluate(preview(identity_id="marvin-1"))
    assert result["classification"] == "same_identity_reacquired"
    assert result["pursuit_authorized"] is False


def test_different_persistent_identity_is_rejected():
    result = evaluate(preview(identity_id="person-2"))
    assert result["classification"] == "different_identity"


def test_semantic_candidate_without_persistent_identity_is_candidate_only():
    result = evaluate(preview())
    assert result["classification"] == "candidate_only"


def test_ambiguous_identity_fails_closed():
    result = evaluate(preview(identity_id="marvin-1", identity_ambiguous=True))
    assert result["classification"] == "ambiguous_identity"


def test_stale_preview_is_rejected():
    result = evaluate_marvin_preview_reacquisition(
        preview(identity_id="marvin-1"), lock(),
        now="2026-09-25T12:00:04+00:00", max_age_seconds=3.0,
    )
    assert result["classification"] == "stale_preview"


def test_missing_preview_timestamp_is_insufficient_evidence():
    result = evaluate(preview(source_timestamp=None, identity_id="marvin-1"))
    assert result["classification"] == "insufficient_evidence"


def test_bbox_overlap_does_not_prove_identity():
    result = evaluate(preview())
    assert result["classification"] == "candidate_only"


def test_reused_detector_track_does_not_prove_identity():
    result = evaluate(preview(tracking={
        "target_label": "marvin",
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        "track_id": "reused-track",
    }))
    assert result["classification"] == "candidate_only"


def test_preview_semantic_confirmation_alone_never_authorizes_pursuit():
    result = evaluate(preview())
    assert result["classification"] == "candidate_only"
    assert result["pursuit_authorized"] is False


def test_exact_world_model_entity_identity_link_can_prove_same_observation():
    result = evaluate(
        preview(entity_id="entity-7"),
        identity_evidence={
            "found": True,
            "entity_id": "entity-7",
            "identity_id": "marvin-1",
            "last_seen": STAMP,
        },
    )
    assert result["classification"] == "same_identity_reacquired"


def test_entity_identity_link_for_another_observation_is_insufficient():
    result = evaluate(
        preview(entity_id="entity-7"),
        identity_evidence={
            "found": True,
            "entity_id": "entity-7",
            "identity_id": "marvin-1",
            "last_seen": "2026-09-25T11:59:59+00:00",
        },
    )
    assert result["classification"] == "candidate_only"


def test_wrong_identity_in_waiting_state_stays_blocked():
    result = evaluate(preview(identity_id="person-2"))
    assert result["classification"] == "different_identity"
    assert result["pursuit_authorized"] is False


def test_result_is_deterministic_and_inputs_are_not_mutated():
    candidate = preview(identity_id="marvin-1")
    selected = lock()
    before = deepcopy((candidate, selected))
    first = evaluate_marvin_preview_reacquisition(
        candidate, selected, now=STAMP,
    )
    second = evaluate_marvin_preview_reacquisition(
        candidate, selected, now=STAMP,
    )
    assert first == second
    assert (candidate, selected) == before


def test_invalid_preview_geometry_fails_closed():
    result = evaluate(preview(tracking={"target_label": "marvin", "bbox": None}))
    assert result["classification"] == "insufficient_evidence"


def test_classifier_has_no_motion_capable_dependencies():
    import inspect
    import marvin_preview_reacquisition

    source = inspect.getsource(marvin_preview_reacquisition)
    assert "robot_client" not in source
    assert "execute_guarded_turn" not in source
    assert "local_forward" not in source
    assert "requests." not in source
