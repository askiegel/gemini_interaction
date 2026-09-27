"""Pure, fail-closed continuity diagnostics for a confirmed Marvin episode."""

from datetime import datetime, timezone
import math


DEFAULT_PREVIEW_MAX_AGE_SECONDS = 3.0


def evaluate_marvin_identity_episode(
    previous_confirmation,
    current_preview,
    *,
    now=None,
    max_episode_age_seconds=30.0,
):
    """Report whether Preview proves continuity with one confirmation episode.

    This policy is diagnostic only.  It neither establishes identity nor
    returns an update payload, and deliberately rejects visual similarity,
    labels, and tracker IDs without an explicit prior episode relationship.
    """
    result = {
        "ok": True,
        "identity_continuity": False,
        "reason": "previous_confirmation_unavailable",
        "selected_identity_id": None,
        "entity_id": None,
        "episode_valid": False,
        "marvin_continuity": {
            "available": False,
            "tracker_id": None,
            "tracker_source": None,
        },
    }
    if not isinstance(previous_confirmation, dict):
        return _invalid(result, "previous_confirmation_malformed")
    if not isinstance(current_preview, dict):
        return _invalid(result, "preview_result_malformed")
    if not _finite_positive(max_episode_age_seconds):
        return _invalid(result, "invalid_max_episode_age_seconds")

    entity_id = _nonempty(previous_confirmation.get("entity_id"))
    identity_id = _nonempty(previous_confirmation.get("identity_id"))
    result.update(entity_id=entity_id, selected_identity_id=identity_id)
    if (
        previous_confirmation.get("ok") is not True
        or previous_confirmation.get("confirmed") is not True
        or previous_confirmation.get("identity_confirmed") is not True
    ):
        return _fail(result, "previous_identity_not_confirmed")
    if entity_id is None:
        return _fail(result, "confirmed_entity_id_missing")
    if identity_id is None:
        return _fail(result, "confirmed_identity_id_missing")

    confirmation_time = _parse_timestamp(
        previous_confirmation.get("confirmation_timestamp")
    )
    current_time = _parse_timestamp(now) if now is not None else datetime.now(timezone.utc)
    if confirmation_time is None or current_time is None:
        return _fail(result, "confirmation_timestamp_missing_or_invalid")
    confirmation_age = (current_time - confirmation_time).total_seconds()
    if not math.isfinite(confirmation_age) or confirmation_age < -1.0:
        return _fail(result, "confirmation_timestamp_invalid")
    if max(0.0, confirmation_age) > float(max_episode_age_seconds):
        return _fail(result, "confirmation_episode_expired")

    expected_episode_id = _episode_id(previous_confirmation)
    if expected_episode_id is None:
        return _fail(result, "confirmation_tracker_episode_missing")

    tracking = current_preview.get("tracking")
    observation = current_preview.get("target_observation")
    if not isinstance(tracking, dict):
        tracking = {}
    if not isinstance(observation, dict):
        observation = {}
    if (
        current_preview.get("ok") is not True
        or current_preview.get("preview") is not True
        or current_preview.get("authoritative") is not False
        or _nonempty(current_preview.get("target")) != "marvin"
        or _nonempty(current_preview.get("source")) != "marvin_local_tracker"
        or current_preview.get("identity_ambiguous") is True
        or tracking.get("identity_ambiguous") is True
        or observation.get("identity_ambiguous") is True
        or _identity_status_ambiguous(current_preview.get("identity_status"))
        or _identity_status_ambiguous(tracking.get("identity_status"))
        or _identity_status_ambiguous(observation.get("identity_status"))
    ):
        return _fail(result, "preview_candidate_invalid")

    # A tracker ID is diagnostic evidence only.  It is never promoted to an
    # identity ID and cannot establish continuity on its own, but an episode
    # evaluation must fail closed when the provider did not supply it.
    marvin_continuity = _marvin_continuity(current_preview)
    result["marvin_continuity"] = marvin_continuity
    if not marvin_continuity["available"]:
        return _fail(result, "preview_marvin_continuity_missing_or_invalid")

    current_episode_id = _episode_id(current_preview)
    if current_episode_id is None:
        return _fail(result, "preview_tracker_episode_missing")
    if current_episode_id != expected_episode_id:
        return _fail(result, "tracker_episode_mismatch")

    timestamp = _nonempty(current_preview.get("source_timestamp"))
    parsed_timestamp = _parse_timestamp(timestamp)
    if parsed_timestamp is None or current_time is None:
        return _fail(result, "preview_timestamp_missing_or_invalid")
    preview_age = (current_time - parsed_timestamp).total_seconds()
    if not math.isfinite(preview_age) or preview_age < -1.0:
        return _fail(result, "preview_timestamp_invalid")
    if max(0.0, preview_age) > DEFAULT_PREVIEW_MAX_AGE_SECONDS:
        return _fail(result, "preview_stale")

    bbox = current_preview.get("bbox") or tracking.get("bbox") or observation.get("bbox")
    width = current_preview.get("image_width") or tracking.get("image_width") or observation.get("image_width")
    height = current_preview.get("image_height") or tracking.get("image_height") or observation.get("image_height")
    if not _valid_geometry(bbox, width, height):
        return _fail(result, "preview_geometry_invalid")

    return dict(
        result,
        identity_continuity=True,
        reason="same_confirmed_tracker_episode",
        episode_valid=True,
    )


def _fail(result, reason):
    return dict(result, reason=reason)


def _invalid(result, reason):
    return dict(result, ok=False, reason=reason)


def _episode_id(value):
    if not isinstance(value, dict):
        return None
    for candidate in (
        value.get("tracker_episode_id"),
        value.get("preview_candidate"),
        value.get("tracking"),
        value.get("target_observation"),
    ):
        if isinstance(candidate, dict):
            candidate = candidate.get("tracker_episode_id")
        normalized = _nonempty(candidate)
        if normalized is not None:
            return normalized
    return None


def _marvin_continuity(value):
    if not isinstance(value, dict):
        value = {}
    observation = value.get("target_observation")
    candidates = (value.get("marvin_continuity"),)
    if isinstance(observation, dict):
        candidates += (observation.get("marvin_continuity"),)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        tracker_id = candidate.get("tracker_id")
        tracker_source = candidate.get("tracker_source")
        if (
            isinstance(tracker_id, int)
            and not isinstance(tracker_id, bool)
            and tracker_id >= 0
            and isinstance(tracker_source, str)
            and tracker_source.strip()
        ):
            return {
                "available": True,
                "tracker_id": tracker_id,
                "tracker_source": tracker_source.strip(),
            }
    return {
        "available": False,
        "tracker_id": None,
        "tracker_source": None,
    }


def _nonempty(value):
    if not isinstance(value, str):
        return None
    return value.strip() or None


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


def _identity_status_ambiguous(value):
    return str(value or "").strip().upper() in {
        "AMBIGUOUS", "NEW_FRAME_CONFLICT", "IDENTITY_MISMATCH",
    }


def _valid_geometry(bbox, width, height):
    if not isinstance(bbox, dict) or not _finite_positive(width) or not _finite_positive(height):
        return False
    try:
        x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError):
        return False
    return (
        all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in (x1, y1, x2, y2))
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )
