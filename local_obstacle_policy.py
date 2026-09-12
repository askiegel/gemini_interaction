"""Pure advisory local-obstacle recommendation from authoritative LiDAR state."""

import math

from lidar_perception import read_lidar_state


TIE_MARGIN_METERS = 0.05
RECOMMENDATIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "HOLD")
TRUSTED_SIDE_STATES = {"CLEAR"}
BLOCKING_FRONT_STATES = {"CAUTION", "BLOCKED"}


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sector_status(sector):
    if not isinstance(sector, dict):
        return "UNKNOWN", None, False
    state = sector.get("state", "UNKNOWN")
    clearance = sector.get("robust_clearance_m")
    minimum_clearance = sector.get("minimum_clearance_m")
    trusted = (
        state in TRUSTED_SIDE_STATES
        and sector.get("available") is True
        and _finite(clearance)
        and _finite(minimum_clearance)
    )
    return state, clearance if _finite(clearance) else None, trusted


def _status(*, recommendation, reason, trusted, state, front_state,
            front_left_state, front_right_state, front_left_clearance,
            front_right_clearance):
    return {
        "recommendation": recommendation,
        "reason": reason,
        "trusted": trusted,
        "fresh": bool(
            isinstance(state, dict)
            and state.get("available") is True
            and state.get("valid") is True
            and state.get("reason") == "fresh"
        ),
        "producer_session": state.get("producer_session") if isinstance(state, dict) else None,
        "effective_age_seconds": state.get("effective_age_seconds") if isinstance(state, dict) else None,
        "front_state": front_state,
        "front_left_state": front_left_state,
        "front_right_state": front_right_state,
        "front_left_robust_clearance_m": front_left_clearance,
        "front_right_robust_clearance_m": front_right_clearance,
    }


def recommend_local_avoidance(state, *, expected_session, now=None,
                              tie_margin_m=TIE_MARGIN_METERS):
    """Return an advisory recommendation without authorizing or executing motion.

    The input is normally the result of ``WorldModel.get_lidar_obstacles``.
    Freshness and producer-session validation are repeated here so callers
    cannot accidentally use an unvalidated snapshot.
    """
    if not _finite(tie_margin_m) or tie_margin_m < 0:
        tie_margin_m = TIE_MARGIN_METERS

    if not isinstance(state, dict):
        return _status(
            recommendation="HOLD", reason="malformed_lidar_state", trusted=False,
            state=None, front_state="UNKNOWN", front_left_state="UNKNOWN",
            front_right_state="UNKNOWN", front_left_clearance=None,
            front_right_clearance=None,
        )

    validated = read_lidar_state(state, expected_session=expected_session, now=now)
    sectors = validated.get("sectors")
    if not validated.get("available") or not validated.get("valid") or not isinstance(sectors, dict):
        front_state, front_clearance, _ = _sector_status(
            sectors.get("front") if isinstance(sectors, dict) else None
        )
        left_state, left_clearance, _ = _sector_status(
            sectors.get("front_left") if isinstance(sectors, dict) else None
        )
        right_state, right_clearance, _ = _sector_status(
            sectors.get("front_right") if isinstance(sectors, dict) else None
        )
        return _status(
            recommendation="HOLD",
            reason=validated.get("reason") or "untrusted_lidar_state",
            trusted=False,
            state=validated,
            front_state=front_state,
            front_left_state=left_state,
            front_right_state=right_state,
            front_left_clearance=left_clearance,
            front_right_clearance=right_clearance,
        )

    front_state, _, front_trusted = _sector_status(sectors.get("front"))
    left_state, left_clearance, left_trusted = _sector_status(sectors.get("front_left"))
    right_state, right_clearance, right_trusted = _sector_status(sectors.get("front_right"))
    common = {
        "state": validated,
        "front_state": front_state,
        "front_left_state": left_state,
        "front_right_state": right_state,
        "front_left_clearance": left_clearance,
        "front_right_clearance": right_clearance,
    }

    if front_state == "CLEAR" and front_trusted:
        return _status(recommendation="FORWARD", reason="front_clear_advisory", trusted=True, **common)
    if front_state not in BLOCKING_FRONT_STATES:
        return _status(recommendation="HOLD", reason="front_not_trustworthy", trusted=True, **common)

    if left_trusted and not right_trusted:
        return _status(recommendation="TURN_LEFT", reason="left_clear_right_not_clear", trusted=True, **common)
    if right_trusted and not left_trusted:
        return _status(recommendation="TURN_RIGHT", reason="right_clear_left_not_clear", trusted=True, **common)
    if not left_trusted and not right_trusted:
        return _status(recommendation="HOLD", reason="no_clear_side", trusted=True, **common)

    difference = left_clearance - right_clearance
    if difference > tie_margin_m:
        recommendation, reason = "TURN_LEFT", "left_clearance_greater"
    elif difference < -tie_margin_m:
        recommendation, reason = "TURN_RIGHT", "right_clearance_greater"
    else:
        recommendation, reason = "TURN_LEFT", "clearance_near_tie_left_preferred"
    return _status(recommendation=recommendation, reason=reason, trusted=True, **common)
