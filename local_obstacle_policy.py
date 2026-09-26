"""Pure advisory local-obstacle recommendation from authoritative LiDAR state."""

import math

from lidar_perception import read_lidar_state


TIE_MARGIN_METERS = 0.05
RECOMMENDATIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "HOLD")
TRUSTED_SIDE_STATES = {"CLEAR"}
BLOCKING_FRONT_STATES = {"CAUTION", "BLOCKED"}

# These are recommendations only.  They deliberately mirror the bounded
# commands already understood by the local safety envelope; this module never
# transports one of them.
LOCAL_AVOIDANCE_CANDIDATES = {
    "forward": {
        "action": "forward", "linear_x": 0.10, "linear_y": 0.0,
        "angular_z": 0.0, "duration": 0.50,
    },
    "forward_left": {
        "action": "forward_left", "linear_x": 0.10, "linear_y": 0.10,
        "angular_z": 0.0, "duration": 0.50,
    },
    "forward_right": {
        "action": "forward_right", "linear_x": 0.10, "linear_y": -0.10,
        "angular_z": 0.0, "duration": 0.50,
    },
    "left_turn": {
        "action": "left_turn", "linear_x": 0.0, "linear_y": 0.0,
        "angular_z": 0.50, "duration": 0.40,
    },
    "right_turn": {
        "action": "right_turn", "linear_x": 0.0, "linear_y": 0.0,
        "angular_z": -0.50, "duration": 0.40,
    },
}

_FORWARD_BLOCKING_REASON = "translation_protected_region_violated"
_LATERAL_EPSILON_M = 1e-9


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


def _candidate_evaluation(state, expected_session, command, now):
    """Call the sole local collision authority for one hypothetical action."""
    from local_motion_safety_envelope import evaluate_local_motion_safety

    try:
        return evaluate_local_motion_safety(
            state,
            expected_session=expected_session,
            linear_x=command["linear_x"],
            linear_y=command["linear_y"],
            angular_z=command["angular_z"],
            duration=command["duration"],
            now=now,
        )
    except Exception as exc:
        # A policy exception is not an authorization.  Preserve its type for
        # diagnostics while making the candidate explicitly fail closed.
        return {
            "permitted": False,
            "reason": "local_motion_safety_evaluator_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def _away_from_blocker_priority(blocking_point):
    """Deterministic non-executing preference order from base-link XY."""
    y = blocking_point.get("y_m") if isinstance(blocking_point, dict) else None
    if not _finite(y):
        return ()
    if y > _LATERAL_EPSILON_M:
        # The blocker is to Mayday's left: prefer the right-side options.
        return ("forward_right", "right_turn", "forward_left", "left_turn")
    if y < -_LATERAL_EPSILON_M:
        # The blocker is to Mayday's right: prefer the left-side options.
        return ("forward_left", "left_turn", "forward_right", "right_turn")
    # A centered blocker has no geometry-supported side preference.  The
    # fixed order is deterministic and only chooses already-permitted actions.
    return ("forward_left", "forward_right", "left_turn", "right_turn")


def _blocking_point_summary(point):
    if not isinstance(point, dict):
        return None
    values = (point.get("x_m"), point.get("y_m"))
    if not all(_finite(value) for value in values):
        return None
    x_m, y_m = values
    distance = point.get("distance_m")
    if not _finite(distance):
        distance = math.hypot(x_m, y_m)
    bearing = point.get("robot_bearing_deg")
    if not _finite(bearing):
        bearing = math.degrees(math.atan2(y_m, x_m))
    return {
        "x_m": x_m,
        "y_m": y_m,
        "distance_m": distance,
        "bearing_deg": bearing,
    }


def plan_local_obstacle_avoidance(state, *, expected_session, now=None):
    """Purely recommend a safe bounded action for a desired forward move.

    The local-motion envelope is the only safety authority: all candidates,
    including pure turns, are evaluated through it.  This function returns a
    recommendation only and intentionally has no Robot Bridge, ownership, or
    BehaviorManager dependency.
    """
    evaluations = {
        name: _candidate_evaluation(state, expected_session, command, now)
        for name, command in LOCAL_AVOIDANCE_CANDIDATES.items()
    }
    forward = evaluations["forward"]
    base = {
        "ok": False,
        "desired_action": "forward",
        "selected_action": None,
        "reason": None,
        "producer_session": (
            state.get("producer_session") if isinstance(state, dict) else None
        ),
        "candidate_evaluations": evaluations,
        "blocking_point": _blocking_point_summary(
            forward.get("violating_point") if isinstance(forward, dict) else None
        ),
    }
    if forward.get("permitted") is True:
        return dict(base, ok=True, selected_action="forward",
                    reason="desired_motion_clear")

    # A translated desired move can safely drive a side choice only when its
    # lower-level geometry identified an actual approaching obstacle.  All
    # freshness, session, coverage, malformed-data, and footprint denials
    # stay fail-closed rather than becoming turn authorizations.
    if forward.get("reason") != _FORWARD_BLOCKING_REASON:
        return dict(base, reason=forward.get("reason", "forward_not_permitted"))
    blocking_point = base["blocking_point"]
    priority = _away_from_blocker_priority(blocking_point)
    if not priority:
        return dict(base, reason="blocking_geometry_unavailable")
    for candidate in priority:
        if evaluations[candidate].get("permitted") is True:
            lateral = blocking_point["y_m"]
            if lateral > _LATERAL_EPSILON_M:
                reason = "forward_blocked_prefer_away_from_left_obstacle"
            elif lateral < -_LATERAL_EPSILON_M:
                reason = "forward_blocked_prefer_away_from_right_obstacle"
            else:
                reason = "forward_blocked_centered_deterministic_safe_side"
            return dict(base, ok=True, selected_action=candidate, reason=reason)
    return dict(base, reason="no_safe_local_avoidance")
