from copy import deepcopy

import pytest

from marvin_identity_continuity import evaluate_marvin_identity_continuity


NOW = "2026-09-26T20:00:03Z"
STAMP = "2026-09-26T20:00:02Z"
ENTITY_ID = "marvin-001"
IDENTITY_ID = "person-identity-1"
_UNSET = object()


def previous(entity_id=ENTITY_ID, identity_id=IDENTITY_ID):
    return {
        "ok": True,
        "confirmed": True,
        "identity_confirmed": True,
        "entity_id": entity_id,
        "identity_id": identity_id,
    }


def world_entity(entity_id=ENTITY_ID, identity_id=IDENTITY_ID):
    return {
        "entity_id": entity_id,
        "label": "marvin",
        "entity_type": "person",
        "attributes": {
            "identity_id": identity_id,
            "operator_confirmed": True,
        },
    }


def preview(**overrides):
    result = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target_found": True,
        "target": "marvin",
        "source": "marvin_local_tracker",
        "identity_confirmed": True,
        "entity_id": ENTITY_ID,
        "identity_id": IDENTITY_ID,
        "source_timestamp": STAMP,
        "target_observation": {
            "found": True,
            "stale": False,
            "source_timestamp": STAMP,
            "entity_id": ENTITY_ID,
            "identity_id": IDENTITY_ID,
            "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 400},
            "image_width": 640,
            "image_height": 480,
            "identity_ambiguous": False,
        },
    }
    result.update(overrides)
    return result


def evaluate(candidate=_UNSET, old=_UNSET, entity=_UNSET, **kwargs):
    return evaluate_marvin_identity_continuity(
        preview() if candidate is _UNSET else candidate,
        previous() if old is _UNSET else old,
        world_entity() if entity is _UNSET else entity,
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def test_valid_explicit_identity_continuity_allows_refresh():
    result = evaluate()
    assert result == {
        "ok": True,
        "allow_refresh": True,
        "reason": "same_confirmed_identity_observed",
        "entity_id": ENTITY_ID,
        "identity_id": IDENTITY_ID,
    }


def test_missing_identity_fails_closed():
    result = evaluate(old=previous(identity_id=None))
    assert result["allow_refresh"] is False
    assert result["identity_id"] is None


def test_stale_preview_fails_closed():
    stale = preview(source_timestamp="2026-09-26T19:59:00Z")
    stale["target_observation"]["source_timestamp"] = "2026-09-26T19:59:00Z"
    result = evaluate(candidate=stale)
    assert result["allow_refresh"] is False
    assert result["reason"] == "preview_stale"


def test_changed_identity_fails_closed():
    changed = preview(identity_id="different-identity")
    changed["target_observation"]["identity_id"] = "different-identity"
    result = evaluate(candidate=changed)
    assert result["allow_refresh"] is False
    assert result["reason"] == "preview_identity_link_missing_or_mismatched"


def test_semantic_only_marvin_detection_is_not_identity_evidence():
    candidate = preview(identity_confirmed=True)
    candidate.pop("identity_id")
    candidate["target_observation"].pop("identity_id")
    result = evaluate(candidate=candidate)
    assert result["allow_refresh"] is False


def test_bbox_similarity_only_is_not_identity_evidence():
    candidate = preview()
    candidate.pop("identity_id")
    candidate["target_observation"].pop("identity_id")
    candidate["entity_id"] = "another-entity"
    candidate["target_observation"]["entity_id"] = "another-entity"
    result = evaluate(candidate=candidate)
    assert result["allow_refresh"] is False


def test_tracker_id_only_is_not_identity_evidence():
    candidate = preview()
    candidate.pop("identity_id")
    candidate["target_observation"].pop("identity_id")
    candidate["target_observation"]["track_id"] = 17
    candidate["target_observation"]["candidate_id"] = "track-17"
    result = evaluate(candidate=candidate)
    assert result["allow_refresh"] is False


def test_preview_entity_must_match_confirmed_entity():
    candidate = preview(entity_id="other-entity")
    candidate["target_observation"]["entity_id"] = "other-entity"
    result = evaluate(candidate=candidate)
    assert result["allow_refresh"] is False


def test_world_model_identity_must_match_confirmation():
    result = evaluate(entity=world_entity(identity_id="other-identity"))
    assert result["allow_refresh"] is False
    assert result["reason"] == "world_model_identity_conflict"


def test_invalid_geometry_fails_closed():
    candidate = preview()
    candidate["target_observation"]["bbox"]["x2"] = 700
    result = evaluate(candidate=candidate)
    assert result["allow_refresh"] is False
    assert result["reason"] == "preview_geometry_invalid"


def test_inputs_are_not_mutated_and_output_is_deterministic():
    candidate, old, entity = preview(), previous(), world_entity()
    originals = deepcopy((candidate, old, entity))
    first = evaluate(candidate, old, entity)
    second = evaluate(candidate, old, entity)
    assert first == second
    assert (candidate, old, entity) == originals


@pytest.mark.parametrize(
    ("candidate", "old", "entity"),
    [
        (None, None, None),
        (preview(target="person"), previous(), world_entity()),
        (preview(identity_ambiguous=True), previous(), world_entity()),
        (preview(source_timestamp="bad"), previous(), world_entity()),
    ],
)
def test_invalid_or_inconsistent_evidence_never_allows_refresh(candidate, old, entity):
    if candidate and candidate.get("source_timestamp") == "bad":
        candidate["target_observation"]["source_timestamp"] = "bad"
    result = evaluate(candidate, old, entity)
    assert result["allow_refresh"] is False
