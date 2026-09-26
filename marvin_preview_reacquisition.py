"""Read-only identity check between Marvin Preview and an existing lock."""

from datetime import datetime, timezone
import math


DEFAULT_PREVIEW_MAX_AGE_SECONDS = 3.0


def evaluate_marvin_preview_reacquisition(
    preview_result,
    target_lock_snapshot,
    *,
    identity_evidence=None,
    now=None,
    max_age_seconds=DEFAULT_PREVIEW_MAX_AGE_SECONDS,
):
    """Classify a fresh Marvin preview against the currently selected identity.

    This function only reports identity evidence. It never mutates TargetLock
    and never authorizes or dispatches motion.  Entity evidence is accepted
    only when its entity ID and acquisition timestamp exactly match the
    preview observation; bbox overlap and detector track IDs are ignored.
    """
    base = {
        "ok": True,
        "classification": "insufficient_evidence",
        "reason": None,
        "target_name": "marvin",
        "selected_identity_id": None,
        "candidate_identity_id": None,
        "source_timestamp": None,
        "detection_age_ms": None,
        "pursuit_authorized": False,
    }
    if not _finite_positive(max_age_seconds):
        return dict(base, ok=False, reason="invalid_preview_max_age")
    if not isinstance(preview_result, dict):
        return dict(base, reason="preview_result_unavailable")
    tracking = preview_result.get("tracking")
    if not isinstance(tracking, dict):
        tracking = {}
    if (
        preview_result.get("ok") is not True
        or preview_result.get("preview") is not True
        or preview_result.get("authoritative") is not False
        or str(preview_result.get("target") or "").strip().lower() != "marvin"
        or preview_result.get("identity_confirmed") is not True
        or tracking.get("target_label") != "marvin"
        or not _valid_bbox(tracking.get("bbox"))
    ):
        return dict(base, reason="marvin_preview_candidate_unavailable")

    timestamp = _first_nonempty(
        preview_result.get("source_timestamp"),
        preview_result.get("vision_timestamp"),
        tracking.get("vision_timestamp"),
    )
    if timestamp is None:
        return dict(base, reason="preview_timestamp_missing")
    parsed_timestamp = _parse_timestamp(timestamp)
    parsed_now = _parse_timestamp(now) if now is not None else datetime.now(timezone.utc)
    if parsed_timestamp is None or parsed_now is None:
        return dict(base, reason="preview_timestamp_invalid",
                    source_timestamp=timestamp)
    age_seconds = (parsed_now - parsed_timestamp).total_seconds()
    if not math.isfinite(age_seconds) or age_seconds < -1.0:
        return dict(base, reason="preview_timestamp_invalid",
                    source_timestamp=timestamp)
    age_seconds = max(0.0, age_seconds)
    age_ms = int(round(age_seconds * 1000.0))
    freshness = {
        "source_timestamp": timestamp,
        "detection_age_ms": age_ms,
    }
    if age_seconds > float(max_age_seconds):
        return dict(base, **freshness, classification="stale_preview",
                    reason="marvin_preview_candidate_stale")

    selected_identity_id = _identity_id(target_lock_snapshot)
    if selected_identity_id is None:
        return dict(base, **freshness, reason="selected_identity_unavailable")
    base = dict(base, **freshness, selected_identity_id=selected_identity_id)

    if not isinstance(target_lock_snapshot, dict):
        return dict(base, reason="target_lock_snapshot_unavailable")
    if target_lock_snapshot.get("identity_ambiguous") is True:
        return dict(base, classification="ambiguous_identity",
                    reason="selected_target_lock_identity_ambiguous")

    candidate_identity_id = _identity_id(preview_result)
    tracking_identity_id = _identity_id(tracking)
    if candidate_identity_id is None:
        candidate_identity_id = tracking_identity_id
    elif tracking_identity_id and tracking_identity_id != candidate_identity_id:
        return dict(base, classification="ambiguous_identity",
                    reason="preview_identity_fields_disagree")

    if candidate_identity_id is None:
        candidate_identity_id = _identity_from_entity_evidence(
            preview_result, identity_evidence, timestamp,
        )
    if candidate_identity_id is None:
        return dict(base, classification="candidate_only",
                    reason="persistent_identity_evidence_missing")
    base["candidate_identity_id"] = candidate_identity_id

    ambiguous = (
        preview_result.get("identity_ambiguous") is True
        or tracking.get("identity_ambiguous") is True
        or _identity_status_is_ambiguous(preview_result.get("identity_status"))
        or _identity_status_is_ambiguous(tracking.get("identity_status"))
        or (
            isinstance(identity_evidence, dict)
            and identity_evidence.get("identity_ambiguous") is True
        )
    )
    if ambiguous:
        return dict(base, classification="ambiguous_identity",
                    reason="preview_identity_evidence_ambiguous")
    if candidate_identity_id != selected_identity_id:
        return dict(base, classification="different_identity",
                    reason="preview_candidate_has_different_identity")

    return dict(base, classification="same_identity_reacquired",
                reason="preview_identity_matches_selected_target_lock")


def _identity_from_entity_evidence(preview, evidence, timestamp):
    """Accept only an exact same-entity, same-observation World Model link."""
    if not isinstance(evidence, dict) or evidence.get("found") is not True:
        return None
    entity_id = _nonempty(preview.get("entity_id"))
    evidence_entity_id = _nonempty(evidence.get("entity_id"))
    if entity_id is None or evidence_entity_id != entity_id:
        return None
    evidence_timestamp = _first_nonempty(
        evidence.get("last_seen"), evidence.get("source_timestamp"),
    )
    if evidence_timestamp != timestamp:
        return None
    if evidence.get("identity_ambiguous") is True:
        return None
    return _identity_id(evidence)


def _identity_id(value):
    if not isinstance(value, dict):
        return None
    return _nonempty(value.get("locked_identity_id")) or _nonempty(
        value.get("identity_id")
    )


def _identity_status_is_ambiguous(value):
    return str(value or "").strip().upper() in {
        "AMBIGUOUS", "NEW_FRAME_CONFLICT", "IDENTITY_MISMATCH",
    }


def _valid_bbox(value):
    if not isinstance(value, dict):
        return False
    try:
        x1, y1, x2, y2 = (value[key] for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError):
        return False
    values = (x1, y1, x2, y2)
    return (
        all(isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(item) for item in values)
        and x2 > x1
        and y2 > y1
    )


def _finite_positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0.0
    )


def _nonempty(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _first_nonempty(*values):
    for value in values:
        value = _nonempty(value)
        if value is not None:
            return value
    return None


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
