"""Pure, bounded local scan policy for the first Find-Marvin search layer.

The policy proposes one future guarded scan action only.  It owns no motion,
transport, perception acquisition, or identity assignment.
"""

from datetime import datetime, timezone
import math

from marvin_preview_reacquisition import DEFAULT_PREVIEW_MAX_AGE_SECONDS
from marvin_preview_schema import normalize_marvin_preview


MAX_SCAN_TURNS = 26
SCAN_DIRECTION = "LEFT"
DEFAULT_MAX_SEARCH_ACTIONS = MAX_SCAN_TURNS
DEFAULT_LOCAL_SCAN_PLAN = ("turn_left",) * MAX_SCAN_TURNS


def plan_marvin_search_step(
    pursuit_state,
    *,
    scan_turn_index=0,
    selected_identity_id=None,
    preview_result=None,
    target_lock_snapshot=None,
    bridge_result=None,
    max_search_actions=DEFAULT_MAX_SEARCH_ACTIONS,
    now=None,
    max_preview_age_seconds=DEFAULT_PREVIEW_MAX_AGE_SECONDS,
    scan_direction=SCAN_DIRECTION,
):
    """Propose at most one deterministic local scan action without motion.

    Scan progress is an explicit mission-scoped completed-turn index. It is
    never reconstructed from controller or action history.
    """
    base = {
        "ok": False,
        "completed": False,
        "reason": None,
        "search_state": None,
        "selected_search_action": "fail_closed",
        "search_actions_used": 0,
        "max_search_actions": max_search_actions,
        "selected_identity_id": _nonempty(selected_identity_id),
        "reacquired": False,
        "candidate_available": False,
        "preview_status": None,
    }
    if not _valid_limit(max_search_actions):
        return dict(base, reason="invalid_marvin_search_action_limit")
    if (not isinstance(scan_turn_index, int)
            or isinstance(scan_turn_index, bool)
            or not 0 <= scan_turn_index <= max_search_actions):
        return dict(base, reason="marvin_search_scan_index_invalid")
    if scan_direction != SCAN_DIRECTION:
        return dict(base, reason="marvin_search_direction_invalid")
    if not _valid_positive(max_preview_age_seconds):
        return dict(base, reason="marvin_search_preview_age_invalid")
    if not isinstance(pursuit_state, dict):
        return dict(base, reason="marvin_search_pursuit_state_malformed")

    state = pursuit_state.get("state")
    if state not in {"SEARCHING", "REACQUIRE_REQUIRED"}:
        return dict(base, search_state=state, reason="marvin_search_state_not_searchable")
    selected_identity = _nonempty(selected_identity_id) or _nonempty(
        pursuit_state.get("selected_identity_id")
    )
    result = dict(base, search_state=state, selected_identity_id=selected_identity)
    used = scan_turn_index
    result["search_actions_used"] = used

    preview_status = _preview_status(
        preview_result, now=now, max_age_seconds=max_preview_age_seconds,
    )
    result["preview_status"] = preview_status
    if preview_status == "malformed":
        return dict(result, reason="marvin_search_preview_malformed")
    candidate_available = preview_status == "fresh_candidate"
    result["candidate_available"] = candidate_available

    if state == "SEARCHING" and candidate_available:
        return dict(
            result, ok=True, reason="marvin_candidate_requires_identity_confirmation",
            selected_search_action="preview_only",
        )

    if state == "REACQUIRE_REQUIRED":
        bridge_classification = (
            bridge_result.get("classification")
            if isinstance(bridge_result, dict) else None
        )
        if bridge_classification in {"different_identity", "ambiguous_identity"}:
            return dict(result, reason="marvin_reacquisition_identity_not_selected")
        if _identity_restored(target_lock_snapshot, selected_identity):
            return dict(
                result, ok=True, reason="marvin_selected_identity_reacquired",
                selected_search_action="reacquired", reacquired=True,
            )
        if candidate_available:
            return dict(
                result, ok=True,
                reason="marvin_candidate_requires_persistent_reacquisition",
                selected_search_action="preview_only",
            )

    if used >= max_search_actions:
        return dict(
            result, ok=True, completed=True, reason="marvin_search_exhausted",
            selected_search_action="search_complete",
        )
    return dict(
        result, ok=True, reason="search_action_planned",
        selected_search_action="turn_left", search_actions_used=used + 1,
        scan_turn_index=used,
        scan_direction=SCAN_DIRECTION,
    )


def _identity_restored(snapshot, selected_identity_id):
    if not isinstance(snapshot, dict) or selected_identity_id is None:
        return False
    return (
        str(snapshot.get("tracking_mode") or "").strip().upper() == "LOCKED"
        and _nonempty(snapshot.get("locked_identity_id")) == selected_identity_id
    )


def _preview_status(value, *, now, max_age_seconds):
    if value is None:
        return "unavailable"
    if not isinstance(value, dict):
        return "malformed"
    preview = normalize_marvin_preview(value)
    if preview is None:
        return "malformed"
    if value.get("ok") is not True:
        return "unavailable"
    if (preview.get("preview") is not True
            or preview.get("authoritative") is not False
            or preview.get("target") != "marvin"):
        return "malformed"
    if preview.get("target_found") is not True:
        return "no_target"
    if preview.get("ambiguous") is True:
        return "ambiguous"
    if (preview.get("identity_confirmed") is not True
            or preview.get("source") != "marvin_local_tracker"):
        return "unconfirmed"
    # A semantic/tracker-confirmed Preview remains inspectable, but it is not
    # a usable Marvin search candidate unless it is compatible with the
    # Marvin visual-session motion-authority contract.  Missing legacy fields
    # fail closed here too, so they cannot suppress bounded search turns.
    if preview.get("motion_authorized_marvin_candidate") is not True:
        return "no_target"
    bbox = preview.get("bbox")
    width, height = preview.get("image_width"), preview.get("image_height")
    if (not _valid_bbox(bbox) or not _valid_positive(width)
            or not _valid_positive(height)
            or not (0 <= bbox["x1"] < bbox["x2"] <= width
                    and 0 <= bbox["y1"] < bbox["y2"] <= height)):
        return "malformed"
    timestamp = _parse_timestamp(
        preview.get("source_timestamp") or preview.get("vision_timestamp")
    )
    current = _parse_timestamp(now) if now is not None else datetime.now(timezone.utc)
    if timestamp is None:
        return "malformed"
    if current is None:
        return "malformed"
    age = (current - timestamp).total_seconds()
    if not math.isfinite(age) or age < -1.0:
        return "malformed"
    return "stale" if max(0.0, age) > float(max_age_seconds) else "fresh_candidate"


def _valid_bbox(value):
    if not isinstance(value, dict):
        return False
    try:
        values = tuple(value[key] for key in ("x1", "y1", "x2", "y2"))
    except KeyError:
        return False
    return (all(_finite(item) for item in values)
            and values[2] > values[0] and values[3] > values[1])


def _valid_limit(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_positive(value):
    return _finite(value) and value > 0.0


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonempty(value):
    return value.strip() or None if isinstance(value, str) else None


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
