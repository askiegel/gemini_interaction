"""Pure calibrated visual stand-off policy for a persistently locked Marvin."""

from datetime import datetime, timezone
import math

from marvin_preview_reacquisition import DEFAULT_PREVIEW_MAX_AGE_SECONDS


MARVIN_ARRIVAL_HEIGHT_FRACTION = 0.545833
MARVIN_ARRIVAL_AREA_FRACTION = 0.160339


def evaluate_marvin_arrival(
    target_lock_result,
    target_lock_snapshot,
    *,
    selected_identity_id=None,
    now=None,
    max_age_seconds=DEFAULT_PREVIEW_MAX_AGE_SECONDS,
    **_extras,
):
    """Evaluate calibrated visual arrival without changing any state.

    A current TargetLock observation remains the sole identity authority.
    The calibrated height fraction is the arrival criterion; area is exposed
    only as corroborating diagnostic information.
    """
    result = _base_result(selected_identity_id)
    if not _positive(max_age_seconds):
        return _fail(result, "invalid_max_age_seconds", ok=False)
    if not isinstance(target_lock_result, dict):
        return _fail(result, "target_lock_result_malformed", ok=False)
    if not isinstance(target_lock_snapshot, dict):
        return _fail(result, "target_lock_snapshot_malformed", ok=False)

    selected = _nonempty(selected_identity_id)
    result["selected_identity_id"] = selected
    if selected is None:
        return _fail(result, "selected_identity_missing")

    snapshot_identity = _identity(target_lock_snapshot)
    result_identity = _identity(target_lock_result)
    if snapshot_identity is not None and snapshot_identity != selected:
        return _fail(result, "target_lock_snapshot_identity_mismatch")
    if result_identity is not None and result_identity != selected:
        return _fail(result, "target_lock_identity_mismatch")
    if _ambiguous(target_lock_snapshot) or _ambiguous(target_lock_result):
        return _fail(result, "target_lock_identity_ambiguous")

    snapshot_mode = _mode(target_lock_snapshot)
    result_mode = _mode(target_lock_result)
    if snapshot_mode and result_mode and snapshot_mode != result_mode:
        return _fail(result, "target_lock_mode_inconsistent")
    if (result_mode or snapshot_mode) != "LOCKED":
        return _fail(result, "target_lock_not_locked")
    if result_identity is None:
        return _fail(result, "target_lock_identity_missing")
    if target_lock_result.get("found") is not True or target_lock_result.get("stale") is True:
        return _fail(result, "target_lock_observation_not_current")
    result["identity_authorized"] = True

    freshness = _freshness(target_lock_result.get("last_seen"), now, max_age_seconds)
    if freshness != "fresh":
        return _fail(result, "target_lock_timestamp_" + freshness)
    result["fresh"] = True

    geometry = _geometry(target_lock_result)
    if geometry is None:
        return _fail(result, "target_lock_geometry_invalid")
    result.update(geometry)
    result["geometry_valid"] = True
    result["area_threshold_met"] = (
        result["area_fraction"] >= MARVIN_ARRIVAL_AREA_FRACTION
    )
    if result["height_fraction"] < MARVIN_ARRIVAL_HEIGHT_FRACTION:
        return _fail(result, "marvin_visual_standoff_not_reached")
    return dict(
        result,
        arrived_at_marvin=True,
        reason="marvin_visual_standoff_reached",
    )


def _base_result(selected_identity_id):
    return {
        "ok": True,
        "arrived_at_marvin": False,
        "reason": None,
        "selected_identity_id": _nonempty(selected_identity_id),
        "identity_authorized": False,
        "fresh": False,
        "geometry_valid": False,
        "bbox_width": None,
        "bbox_height": None,
        "width_fraction": None,
        "height_fraction": None,
        "area_fraction": None,
        "height_threshold": MARVIN_ARRIVAL_HEIGHT_FRACTION,
        "area_threshold": MARVIN_ARRIVAL_AREA_FRACTION,
        "area_threshold_met": False,
    }


def _fail(result, reason, *, ok=True):
    return dict(result, ok=ok, arrived_at_marvin=False, reason=reason)


def _geometry(value):
    bbox = value.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
    except KeyError:
        return None
    width, height = value.get("image_width"), value.get("image_height")
    if not all(_number(item) for item in (x1, y1, x2, y2, width, height)):
        return None
    if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    bbox_width, bbox_height = x2 - x1, y2 - y1
    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "width_fraction": bbox_width / width,
        "height_fraction": bbox_height / height,
        "area_fraction": (bbox_width * bbox_height) / (width * height),
    }


def _freshness(value, now, max_age_seconds):
    if value is None or value == "":
        return "missing"
    timestamp = _timestamp(value)
    current = _timestamp(now) if now is not None else datetime.now(timezone.utc)
    if timestamp is None or current is None:
        return "invalid"
    age = (current - timestamp).total_seconds()
    if not math.isfinite(age) or age < -1.0:
        return "invalid"
    return "stale" if max(0.0, age) > float(max_age_seconds) else "fresh"


def _timestamp(value):
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


def _mode(value):
    return str(value.get("tracking_mode") or "").strip().upper()


def _identity(value):
    return _nonempty(value.get("identity_id")) or _nonempty(value.get("locked_identity_id"))


def _ambiguous(value):
    return (value.get("identity_ambiguous") is True
            or str(value.get("identity_status") or "").strip().upper()
            in {"AMBIGUOUS", "NEW_FRAME_CONFLICT", "IDENTITY_MISMATCH"})


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value):
    return _number(value) and value > 0.0


def _nonempty(value):
    return value.strip() or None if isinstance(value, str) else None
