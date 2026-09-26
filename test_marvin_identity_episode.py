"""Offline contracts for bounded, diagnostic Marvin identity episodes."""

from copy import deepcopy

import pytest

from marvin_identity_episode import evaluate_marvin_identity_episode


NOW = "2026-09-26T20:00:02Z"
CONFIRMATION_TIME = "2026-09-26T20:00:00Z"
PREVIEW_TIME = "2026-09-26T20:00:01Z"
ENTITY_ID = "marvin-001"
IDENTITY_ID = "person-identity-1"
EPISODE_ID = "marvin-tracker-episode-1"


def confirmation(**overrides):
    value = {
        "ok": True,
        "confirmed": True,
        "identity_confirmed": True,
        "entity_id": ENTITY_ID,
        "identity_id": IDENTITY_ID,
        "confirmation_timestamp": CONFIRMATION_TIME,
        "preview_candidate": {"tracker_episode_id": EPISODE_ID},
    }
    value.update(overrides)
    return value


def preview(**overrides):
    value = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target": "marvin",
        "source": "marvin_local_tracker",
        "source_timestamp": PREVIEW_TIME,
        "tracker_episode_id": EPISODE_ID,
        "identity_ambiguous": False,
        "tracking": {
            "tracker_episode_id": EPISODE_ID,
            "identity_ambiguous": False,
        },
        "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 400},
        "image_width": 640,
        "image_height": 480,
    }
    value.update(overrides)
    return value


def evaluate(old=None, candidate=None, **kwargs):
    return evaluate_marvin_identity_episode(
        confirmation() if old is None else old,
        preview() if candidate is None else candidate,
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def test_confirmed_identity_same_tracker_episode_allows_diagnostic_continuity():
    result = evaluate()
    assert result == {
        "ok": True,
        "identity_continuity": True,
        "reason": "same_confirmed_tracker_episode",
        "selected_identity_id": IDENTITY_ID,
        "entity_id": ENTITY_ID,
        "episode_valid": True,
    }


def test_different_tracker_episode_fails_closed():
    result = evaluate(candidate=preview(tracker_episode_id="different-episode"))
    assert result["identity_continuity"] is False
    assert result["episode_valid"] is False
    assert result["reason"] == "tracker_episode_mismatch"


def test_stale_confirmation_fails_closed():
    result = evaluate(now="2026-09-26T20:01:00Z")
    assert result["identity_continuity"] is False
    assert result["reason"] == "confirmation_episode_expired"


@pytest.mark.parametrize(
    ("old", "candidate", "reason"),
    [
        (confirmation(identity_id=None), None, "confirmed_identity_id_missing"),
        (None, preview(identity_ambiguous=True), "preview_candidate_invalid"),
        (None, preview(bbox={"x1": 1, "y1": 2, "x2": 1, "y2": 4}), "preview_geometry_invalid"),
        (None, preview(source_timestamp=None), "preview_timestamp_missing_or_invalid"),
    ],
)
def test_missing_or_ambiguous_or_malformed_evidence_fails_closed(old, candidate, reason):
    result = evaluate(old=old, candidate=candidate)
    assert result["identity_continuity"] is False
    assert result["reason"] == reason


def test_evaluator_is_pure_and_cannot_establish_identity_or_return_write_payload():
    old, candidate = confirmation(), preview()
    original = deepcopy((old, candidate))
    result = evaluate(old=old, candidate=candidate)
    assert (old, candidate) == original
    assert "observation_update" not in result
    unconfirmed = evaluate(old=confirmation(confirmed=False))
    assert unconfirmed["identity_continuity"] is False
    assert unconfirmed["reason"] == "previous_identity_not_confirmed"
