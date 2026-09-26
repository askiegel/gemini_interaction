from copy import deepcopy

import pytest

from marvin_identity_refresh_policy import build_marvin_identity_refresh_update


NOW = "2026-09-26T20:00:03Z"
STAMP = "2026-09-26T20:00:02Z"
ENTITY_ID = "marvin-001"
IDENTITY_ID = "person-identity-1"
_UNSET = object()


def continuity(**overrides):
    value = {
        "ok": True,
        "allow_refresh": True,
        "reason": "same_confirmed_identity_observed",
        "entity_id": ENTITY_ID,
        "identity_id": IDENTITY_ID,
    }
    value.update(overrides)
    return value


def entity(**overrides):
    value = {
        "entity_id": ENTITY_ID,
        "label": "marvin",
        "entity_type": "person",
        "attributes": {
            "identity_id": IDENTITY_ID,
            "operator_confirmed": True,
            "owner": "human-1",
            "identity_confirmation_source": "operator",
        },
    }
    value.update(overrides)
    return value


def preview(**overrides):
    value = {
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
            "entity_id": ENTITY_ID,
            "identity_id": IDENTITY_ID,
            "source_timestamp": STAMP,
            "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 400},
            "image_width": 640,
            "image_height": 480,
            "confidence": 0.82,
        },
        # Malicious or irrelevant fields must never leak into the update.
        "label": "other-label",
        "operator_confirmed": False,
        "ownership": "attacker",
    }
    value.update(overrides)
    return value


def build(continuity_value=_UNSET, entity_value=_UNSET, preview_value=_UNSET, **kwargs):
    return build_marvin_identity_refresh_update(
        continuity() if continuity_value is _UNSET else continuity_value,
        entity() if entity_value is _UNSET else entity_value,
        preview() if preview_value is _UNSET else preview_value,
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def test_valid_refresh_returns_only_allowlisted_observation_fields():
    result = build()
    assert result["ok"] is True and result["allow_refresh"] is True
    update = result["observation_update"]
    assert update == {
        "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 400},
        "image_width": 640,
        "image_height": 480,
        "source_timestamp": STAMP,
        "source": "marvin_local_tracker",
        "confidence": 0.82,
    }
    assert set(update) <= {
        "bbox", "image_width", "image_height", "source_timestamp",
        "confidence", "source",
    }


@pytest.mark.parametrize(
    "continuity_value",
    [None, {"ok": True, "allow_refresh": False}],
)
def test_missing_or_denied_continuity_is_rejected(continuity_value):
    result = build(continuity_value=continuity_value)
    assert result["allow_refresh"] is False
    assert result["observation_update"] is None


def test_identity_mismatch_is_rejected():
    result = build(
        continuity_value=continuity(identity_id="different-identity"),
    )
    assert result["allow_refresh"] is False
    assert result["reason"] == "identity_or_entity_mismatch"


def test_stale_observation_is_rejected():
    old_stamp = "2026-09-26T19:59:00Z"
    candidate = preview(source_timestamp=old_stamp)
    candidate["target_observation"]["source_timestamp"] = old_stamp
    result = build(preview_value=candidate)
    assert result["allow_refresh"] is False
    assert result["reason"] == "preview_stale"


def test_identity_and_ownership_fields_cannot_be_overwritten():
    candidate = preview(
        identity_id="attacker-identity",
        entity_id="attacker-entity",
        identity_confirmation_timestamp="attacker-time",
        owner="attacker",
    )
    result = build(preview_value=candidate)
    assert result["allow_refresh"] is False
    assert result["observation_update"] is None


def test_geometry_only_evidence_is_rejected():
    candidate = preview()
    candidate.pop("identity_id")
    candidate.pop("entity_id")
    candidate["target_observation"].pop("identity_id")
    candidate["target_observation"].pop("entity_id")
    result = build(
        continuity_value=continuity(),
        preview_value=candidate,
    )
    assert result["allow_refresh"] is False
    assert result["observation_update"] is None


def test_input_objects_are_not_mutated():
    continuity_value, entity_value, candidate = continuity(), entity(), preview()
    originals = deepcopy((continuity_value, entity_value, candidate))
    result = build(continuity_value, entity_value, candidate)
    assert result["allow_refresh"] is True
    assert (continuity_value, entity_value, candidate) == originals
