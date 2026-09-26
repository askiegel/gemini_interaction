"""Pure policy for refreshing an already-confirmed Marvin observation."""

from datetime import datetime, timezone
import math


DEFAULT_MARVIN_PREVIEW_MAX_AGE_SECONDS = 3.0


def evaluate_marvin_identity_continuity(
    preview_result,
    previous_confirmed_record,
    world_model_entity,
    *,
    now,
    max_age_seconds=DEFAULT_MARVIN_PREVIEW_MAX_AGE_SECONDS,
):
    """Decide whether one Preview is explicitly linked to a known identity.

    Similarity, semantic labels, candidate IDs, tracker IDs, and confidence
    are deliberately not identity evidence.  The current Preview observation
    must carry the same entity and persistent identity IDs as both the prior
    confirmation and the existing World Model record.
    """
    result = {
        "ok": True,
        "allow_refresh": False,
        "reason": "insufficient_identity_evidence",
        "entity_id": None,
        "identity_id": None,
    }

    if not isinstance(previous_confirmed_record, dict):
        return _invalid(result, "previous_confirmation_malformed")
    if not isinstance(world_model_entity, dict):
        return _invalid(result, "world_model_entity_malformed")
    if not isinstance(preview_result, dict):
        return _invalid(result, "preview_result_malformed")

    entity_id = _nonempty(previous_confirmed_record.get("entity_id"))
    identity_id = _nonempty(previous_confirmed_record.get("identity_id"))
    result.update(entity_id=entity_id, identity_id=identity_id)

    if (
        previous_confirmed_record.get("ok") is not True
        or previous_confirmed_record.get("confirmed") is not True
        or previous_confirmed_record.get("identity_confirmed") is not True
    ):
        return dict(result, reason="previous_identity_not_confirmed")
    if entity_id is None:
        return dict(result, reason="confirmed_entity_id_missing")
    if identity_id is None:
        return dict(result, reason="confirmed_identity_id_missing")

    attributes = world_model_entity.get("attributes")
    if not isinstance(attributes, dict):
        return _invalid(result, "world_model_attributes_malformed")
    if (
        _nonempty(world_model_entity.get("entity_id")) != entity_id
        or _nonempty(world_model_entity.get("label")) != "marvin"
        or str(world_model_entity.get("entity_type") or "").strip().lower()
        not in {"person", "human"}
        or _nonempty(attributes.get("identity_id")) != identity_id
        or attributes.get("operator_confirmed") is not True
    ):
        return dict(result, reason="world_model_identity_conflict")

    observation = preview_result.get("target_observation")
    if not isinstance(observation, dict):
        return dict(result, reason="preview_observation_missing")
    if (
        preview_result.get("ok") is not True
        or preview_result.get("preview") is not True
        or preview_result.get("authoritative") is not False
        or preview_result.get("target_found") is not True
        or _nonempty(preview_result.get("target")) != "marvin"
        or _nonempty(preview_result.get("source")) != "marvin_local_tracker"
        or preview_result.get("identity_confirmed") is not True
        or preview_result.get("identity_ambiguous") is True
        or observation.get("found") is not True
        or observation.get("stale") is True
        or observation.get("identity_ambiguous") is True
    ):
        return dict(result, reason="preview_candidate_invalid")

    source_timestamp = _nonempty(preview_result.get("source_timestamp"))
    observation_timestamp = _nonempty(observation.get("source_timestamp"))
    parsed_timestamp = _parse_timestamp(source_timestamp)
    parsed_now = _parse_timestamp(now)
    if (
        source_timestamp is None
        or observation_timestamp != source_timestamp
        or parsed_timestamp is None
        or parsed_now is None
    ):
        return dict(result, reason="preview_timestamp_missing_or_invalid")
    if not _finite_positive(max_age_seconds):
        return _invalid(result, "invalid_max_age_seconds")
    age_seconds = (parsed_now - parsed_timestamp).total_seconds()
    if not math.isfinite(age_seconds) or age_seconds < -1.0:
        return dict(result, reason="preview_timestamp_invalid")
    if max(0.0, age_seconds) > float(max_age_seconds):
        return dict(result, reason="preview_stale")

    bbox = observation.get("bbox")
    image_width = observation.get("image_width")
    image_height = observation.get("image_height")
    if not _valid_geometry(bbox, image_width, image_height):
        return dict(result, reason="preview_geometry_invalid")

    preview_entity_id = _nonempty(preview_result.get("entity_id"))
    observation_entity_id = _nonempty(observation.get("entity_id"))
    preview_identity_id = _nonempty(preview_result.get("identity_id"))
    observation_identity_id = _nonempty(observation.get("identity_id"))
    if (
        preview_entity_id is None
        or observation_entity_id != preview_entity_id
        or preview_entity_id != entity_id
        or preview_identity_id is None
        or observation_identity_id != preview_identity_id
        or preview_identity_id != identity_id
    ):
        return dict(result, reason="preview_identity_link_missing_or_mismatched")

    return dict(result, allow_refresh=True, reason="same_confirmed_identity_observed")


def _invalid(result, reason):
    return dict(result, ok=False, reason=reason)


def _nonempty(value):
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _parse_timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0.0
    )


def _valid_geometry(bbox, image_width, image_height):
    if not isinstance(bbox, dict):
        return False
    try:
        x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError):
        return False
    values = (x1, y1, x2, y2)
    return (
        _finite_positive(image_width)
        and _finite_positive(image_height)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        )
        and 0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    )
