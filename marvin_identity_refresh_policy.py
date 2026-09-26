"""Pure allowlist policy for a future Marvin observation refresh."""

from datetime import datetime, timezone
import math


DEFAULT_MARVIN_REFRESH_MAX_AGE_SECONDS = 3.0


def build_marvin_identity_refresh_update(
    identity_continuity,
    existing_world_model_entity,
    preview_result,
    *,
    now,
    max_age_seconds=DEFAULT_MARVIN_REFRESH_MAX_AGE_SECONDS,
):
    """Return a minimal observation update, without performing any write.

    The output's ``observation_update`` is an allowlist: identity, entity,
    label, and ownership/confirmation fields are never copied or returned.
    """
    result = {
        "ok": True,
        "allow_refresh": False,
        "reason": "refresh_not_authorized",
        "entity_id": None,
        "identity_id": None,
        "observation_update": None,
    }

    if not isinstance(identity_continuity, dict):
        return _invalid(result, "identity_continuity_malformed")
    if identity_continuity.get("ok") is not True:
        return dict(result, reason="identity_continuity_not_ok")
    if identity_continuity.get("allow_refresh") is not True:
        return dict(result, reason="identity_continuity_denied")
    if not isinstance(existing_world_model_entity, dict):
        return _invalid(result, "world_model_entity_malformed")
    if not isinstance(preview_result, dict):
        return _invalid(result, "preview_result_malformed")

    entity_id = _nonempty(existing_world_model_entity.get("entity_id"))
    attributes = existing_world_model_entity.get("attributes")
    if not isinstance(attributes, dict):
        return _invalid(result, "world_model_attributes_malformed")
    identity_id = _nonempty(attributes.get("identity_id"))
    result.update(entity_id=entity_id, identity_id=identity_id)

    if (
        entity_id is None
        or identity_id is None
        or _nonempty(identity_continuity.get("entity_id")) != entity_id
        or _nonempty(identity_continuity.get("identity_id")) != identity_id
        or _nonempty(existing_world_model_entity.get("label")) != "marvin"
        or str(existing_world_model_entity.get("entity_type") or "").strip().lower()
        not in {"person", "human"}
        or attributes.get("operator_confirmed") is not True
    ):
        return dict(result, reason="identity_or_entity_mismatch")

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
        or _nonempty(preview_result.get("entity_id")) != entity_id
        or _nonempty(observation.get("entity_id")) != entity_id
        or _nonempty(preview_result.get("identity_id")) != identity_id
        or _nonempty(observation.get("identity_id")) != identity_id
    ):
        return dict(result, reason="preview_identity_or_candidate_mismatch")

    source_timestamp = _nonempty(preview_result.get("source_timestamp"))
    if (
        source_timestamp is None
        or _nonempty(observation.get("source_timestamp")) != source_timestamp
    ):
        return dict(result, reason="preview_timestamp_missing_or_mismatched")
    parsed_timestamp = _parse_timestamp(source_timestamp)
    parsed_now = _parse_timestamp(now)
    if parsed_timestamp is None or parsed_now is None:
        return dict(result, reason="preview_timestamp_invalid")
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

    observation_update = {
        "bbox": {key: bbox[key] for key in ("x1", "y1", "x2", "y2")},
        "image_width": image_width,
        "image_height": image_height,
        "source_timestamp": source_timestamp,
        "source": "marvin_local_tracker",
    }
    confidence = observation.get("confidence")
    if confidence is None:
        confidence = preview_result.get("target_confidence")
    if confidence is not None:
        if not _valid_confidence(confidence):
            return dict(result, reason="preview_confidence_invalid")
        observation_update["confidence"] = confidence

    return dict(
        result,
        allow_refresh=True,
        reason="fresh_same_identity_observation",
        observation_update=observation_update,
    )


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


def _valid_confidence(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_geometry(bbox, image_width, image_height):
    if not isinstance(bbox, dict):
        return False
    try:
        x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError):
        return False
    return (
        _finite_positive(image_width)
        and _finite_positive(image_height)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (x1, y1, x2, y2)
        )
        and 0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    )
