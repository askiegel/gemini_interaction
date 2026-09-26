"""Pure, read-only orchestration policy for the Sunday Find-Marvin flow.

Preview can find a Marvin candidate and the preview bridge can relate that
candidate to a selected identity.  Neither is an authority to pursue.
Only a fresh, unambiguous current TargetLock observation for that same
persistent identity can produce ``READY_TO_APPROACH``.
"""

from datetime import datetime, timezone
import math

from marvin_preview_reacquisition import (
    DEFAULT_PREVIEW_MAX_AGE_SECONDS,
    evaluate_marvin_preview_reacquisition,
)


SEARCHING = "SEARCHING"
CANDIDATE_SEEN = "CANDIDATE_SEEN"
MARVIN_LOCKED = "MARVIN_LOCKED"
REACQUIRE_REQUIRED = "REACQUIRE_REQUIRED"
SAME_IDENTITY_REACQUIRED = "SAME_IDENTITY_REACQUIRED"
READY_TO_APPROACH = "READY_TO_APPROACH"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def evaluate_marvin_pursuit_state(
    preview_result,
    target_lock_result,
    target_lock_snapshot,
    *,
    selected_identity_id=None,
    identity_evidence=None,
    bridge_result=None,
    now=None,
    max_age_seconds=DEFAULT_PREVIEW_MAX_AGE_SECONDS,
):
    """Return a deterministic, non-mutating Marvin pursuit policy result.

    ``target_lock_result`` is the current result of ``TargetLock.resolve()``;
    ``target_lock_snapshot`` is its companion ``TargetLock.snapshot()``.
    The optional ``bridge_result`` is useful to expose an already evaluated
    bridge diagnostically, but is checked against the supplied inputs rather
    than trusted as an authority.
    """
    base = _base_result()
    if not _finite_positive(max_age_seconds):
        return _result(base, INSUFFICIENT_EVIDENCE, "invalid_max_age_seconds")
    if target_lock_result is not None and not isinstance(target_lock_result, dict):
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_result_malformed")
    if target_lock_snapshot is not None and not isinstance(target_lock_snapshot, dict):
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_snapshot_malformed")
    if preview_result is not None and not isinstance(preview_result, dict):
        return _result(base, INSUFFICIENT_EVIDENCE, "preview_result_malformed")

    snapshot = target_lock_snapshot or {}
    lock_result = target_lock_result or {}
    snapshot_identity = _identity_id(snapshot)
    supplied_identity = _nonempty(selected_identity_id)
    if supplied_identity and snapshot_identity and supplied_identity != snapshot_identity:
        return _result(base, INSUFFICIENT_EVIDENCE, "selected_identity_changed")
    selected_identity = supplied_identity or snapshot_identity
    base["selected_identity_id"] = selected_identity
    base["entity_id"] = _nonempty(lock_result.get("entity_id")) or _nonempty(snapshot.get("locked_entity_id"))

    if _ambiguous(snapshot) or _ambiguous(lock_result):
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_identity_ambiguous")
    if lock_result.get("identity_mismatch") is True:
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_identity_mismatch")

    candidate, candidate_reason = _preview_candidate(preview_result, now, max_age_seconds)
    base["candidate_available"] = candidate
    if candidate_reason in {"preview_timestamp_missing", "preview_timestamp_invalid", "preview_geometry_malformed"}:
        return _result(base, INSUFFICIENT_EVIDENCE, candidate_reason)

    bridge = None
    if selected_identity and preview_result is not None:
        bridge = evaluate_marvin_preview_reacquisition(
            preview_result,
            snapshot,
            identity_evidence=identity_evidence,
            now=now,
            max_age_seconds=max_age_seconds,
        )
        if not _bridge_consistent(bridge_result, bridge):
            return _result(base, INSUFFICIENT_EVIDENCE, "bridge_output_inconsistent")
        classification = bridge.get("classification")
        if classification in {"different_identity", "ambiguous_identity"}:
            return _result(base, INSUFFICIENT_EVIDENCE, "preview_identity_not_selected")
    elif bridge_result is not None:
        return _result(base, INSUFFICIENT_EVIDENCE, "bridge_output_without_selected_identity")

    mode = str(lock_result.get("tracking_mode") or snapshot.get("tracking_mode") or "").strip().upper()
    waiting = mode == "WAITING_FOR_IDENTITY"
    if waiting:
        if bridge and bridge.get("classification") == "same_identity_reacquired":
            return _result(base, SAME_IDENTITY_REACQUIRED, "bridge_confirmed_waiting_for_target_lock")
        return _result(base, REACQUIRE_REQUIRED, "target_lock_waiting_for_identity")

    if not selected_identity:
        if candidate:
            return _result(base, CANDIDATE_SEEN, "fresh_preview_candidate_without_persistent_identity")
        return _result(base, SEARCHING, "no_preview_candidate_or_persistent_identity")

    lock_identity = _identity_id(lock_result)
    if lock_identity and lock_identity != selected_identity:
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_selected_identity_mismatch")
    if lock_result.get("found") is not True or lock_result.get("stale") is True or mode != "LOCKED":
        return _result(base, REACQUIRE_REQUIRED, "selected_identity_not_currently_locked")
    if lock_identity is None:
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_identity_missing")

    lock_freshness = _fresh_timestamp(lock_result.get("last_seen"), now, max_age_seconds)
    if lock_freshness == "missing" or lock_freshness == "invalid":
        return _result(base, INSUFFICIENT_EVIDENCE, "target_lock_timestamp_" + lock_freshness)
    if lock_freshness == "stale":
        return _result(base, REACQUIRE_REQUIRED, "target_lock_observation_stale")
    base["fresh"] = True
    base["identity_locked"] = True

    if not _lock_geometry_usable(lock_result):
        # Identity continuity is valid, but this is deliberately not enough
        # to approach.  Preserve that useful diagnostic distinction while
        # failing closed on missing current visual geometry.
        return _result(base, MARVIN_LOCKED, "target_lock_geometry_unusable")
    base["geometry_usable"] = True
    return _result(base, READY_TO_APPROACH, "fresh_selected_identity_locked", pursuit_authorized=True)


def _base_result():
    return {
        "ok": True, "state": INSUFFICIENT_EVIDENCE,
        "pursuit_authorized": False, "selected_identity_id": None,
        "entity_id": None, "candidate_available": False,
        "identity_locked": False, "fresh": False,
        "geometry_usable": False, "reason": None,
    }


def _result(base, state, reason, *, pursuit_authorized=False):
    return dict(base, state=state, reason=reason,
                pursuit_authorized=bool(pursuit_authorized))


def _preview_candidate(value, now, max_age_seconds):
    if value is None:
        return False, "preview_unavailable"
    if not isinstance(value, dict):
        return False, "preview_malformed"
    if (value.get("ok") is not True or value.get("preview") is not True
            or value.get("authoritative") is not False
            or str(value.get("target") or "").strip().lower() != "marvin"
            or value.get("target_found") is not True
            or value.get("identity_confirmed") is not True):
        return False, "preview_candidate_unavailable"
    if not _bbox(value.get("bbox")):
        return False, "preview_geometry_malformed"
    freshness = _fresh_timestamp(value.get("source_timestamp") or value.get("vision_timestamp"), now, max_age_seconds)
    if freshness in {"missing", "invalid"}:
        return False, "preview_timestamp_" + freshness
    return freshness == "fresh", "preview_candidate_stale" if freshness == "stale" else "preview_candidate_fresh"


def _bridge_consistent(supplied, computed):
    if supplied is None:
        return True
    if not isinstance(supplied, dict):
        return False
    return (supplied.get("classification") == computed.get("classification")
            and supplied.get("selected_identity_id") == computed.get("selected_identity_id")
            and supplied.get("candidate_identity_id") == computed.get("candidate_identity_id"))


def _lock_geometry_usable(value):
    cx = value.get("cx", value.get("target_center_x"))
    width = value.get("image_width")
    return _finite_number(cx) and _finite_number(width) and float(width) > 0.0


def _bbox(value):
    if not isinstance(value, dict):
        return False
    try:
        values = tuple(value[key] for key in ("x1", "y1", "x2", "y2"))
    except KeyError:
        return False
    return all(_finite_number(item) for item in values) and values[2] > values[0] and values[3] > values[1]


def _fresh_timestamp(value, now, max_age_seconds):
    timestamp = _parse_timestamp(value)
    current = _parse_timestamp(now) if now is not None else datetime.now(timezone.utc)
    if value is None or value == "":
        return "missing"
    if timestamp is None or current is None:
        return "invalid"
    age = (current - timestamp).total_seconds()
    if not math.isfinite(age) or age < -1.0:
        return "invalid"
    return "stale" if max(0.0, age) > float(max_age_seconds) else "fresh"


def _identity_id(value):
    if not isinstance(value, dict):
        return None
    return _nonempty(value.get("locked_identity_id")) or _nonempty(value.get("identity_id"))


def _ambiguous(value):
    return (value.get("identity_ambiguous") is True
            or str(value.get("identity_status") or "").strip().upper()
            in {"AMBIGUOUS", "NEW_FRAME_CONFLICT", "IDENTITY_MISMATCH"})


def _nonempty(value):
    return value.strip() or None if isinstance(value, str) else None


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_positive(value):
    return _finite_number(value) and value > 0.0


def _parse_timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
