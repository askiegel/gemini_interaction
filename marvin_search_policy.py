"""Pure, bounded local scan policy for the first Find-Marvin search layer.

The policy proposes one future guarded scan action only.  It owns no motion,
transport, perception acquisition, or identity assignment.
"""

from datetime import datetime, timezone
import math

from marvin_preview_reacquisition import DEFAULT_PREVIEW_MAX_AGE_SECONDS


DEFAULT_MAX_SEARCH_ACTIONS = 4
# Each turn is separated by a fresh Preview/TargetLock evaluation.  The two
# right turns inspect past center before the final left turn restores heading.
DEFAULT_LOCAL_SCAN_PLAN = ("turn_left", "turn_right", "turn_right", "turn_left")


def plan_marvin_search_step(
    pursuit_state,
    *,
    prior_search_history=None,
    selected_identity_id=None,
    preview_result=None,
    target_lock_snapshot=None,
    bridge_result=None,
    max_search_actions=DEFAULT_MAX_SEARCH_ACTIONS,
    now=None,
    max_preview_age_seconds=DEFAULT_PREVIEW_MAX_AGE_SECONDS,
    scan_plan=DEFAULT_LOCAL_SCAN_PLAN,
):
    """Propose at most one deterministic local scan action without motion.

    ``prior_search_history`` is prior planner output, not an execution log.
    It is used only to count already proposed scan turns; callers must obtain
    fresh perception and TargetLock evidence before each subsequent call.
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
    }
    if not _valid_limit(max_search_actions):
        return dict(base, reason="invalid_marvin_search_action_limit")
    if not isinstance(prior_search_history, (list, tuple, type(None))):
        return dict(base, reason="marvin_search_history_malformed")
    if not _valid_scan_plan(scan_plan):
        return dict(base, reason="marvin_search_scan_plan_invalid")
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
    used = _count_actions(prior_search_history)
    if used is None:
        return dict(result, reason="marvin_search_history_entry_malformed")
    result["search_actions_used"] = used

    preview_status = _preview_status(
        preview_result, now=now, max_age_seconds=max_preview_age_seconds,
    )
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

    if used >= max_search_actions or used >= len(scan_plan):
        return dict(
            result, ok=True, completed=True, reason="marvin_search_exhausted",
            selected_search_action="search_complete",
        )
    action = scan_plan[used]
    return dict(
        result, ok=True, reason="search_action_planned",
        selected_search_action=action, search_actions_used=used + 1,
    )


def _count_actions(history):
    if history is None:
        return 0
    count = 0
    for entry in history:
        if not isinstance(entry, dict):
            return None
        action = entry.get("selected_search_action")
        if action in {"turn_left", "turn_right"}:
            count += 1
        elif action not in {"preview_only", "reacquired", "search_complete", "fail_closed", None}:
            return None
    return count


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
    if (value.get("ok") is not True or value.get("preview") is not True
            or value.get("authoritative") is not False
            or str(value.get("target") or "").strip().lower() != "marvin"):
        return "malformed"
    if value.get("target_found") is not True or value.get("identity_confirmed") is not True:
        return "unavailable"
    if not _valid_bbox(value.get("bbox")):
        return "malformed"
    timestamp = _parse_timestamp(value.get("source_timestamp") or value.get("vision_timestamp"))
    current = _parse_timestamp(now) if now is not None else datetime.now(timezone.utc)
    if timestamp is None or current is None:
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


def _valid_scan_plan(value):
    return (isinstance(value, (tuple, list)) and bool(value)
            and all(item in {"turn_left", "turn_right"} for item in value))


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
