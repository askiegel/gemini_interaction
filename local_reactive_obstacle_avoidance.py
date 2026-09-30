"""Pure local reactive obstacle-avoidance decisions for one forward intent.

This module is deliberately not an executor.  It evaluates the established
local translation envelope for the caller's intended bounded forward movement
and the approved circular rotational safety envelope for either turn.  It
returns only a next-action class; it has no Robot Bridge, navigation, map,
camera, World Model, or localization dependency.
"""

import math

from local_motion_safety_envelope import evaluate_local_motion_safety
from rotational_swept_footprint import evaluate_rotational_swept_footprint


FORWARD_CLEAR = "FORWARD_CLEAR"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
STOP_BLOCKED = "STOP_BLOCKED"
DECISIONS = {FORWARD_CLEAR, TURN_LEFT, TURN_RIGHT, STOP_BLOCKED}

# The first version uses the existing proven bounded primitives only as
# hypothetical safety inputs.  They are never returned or dispatched here.
DEFAULT_FORWARD_LINEAR_SPEED_MPS = 0.08
DEFAULT_FORWARD_DURATION_SECONDS = 0.50
DEFAULT_TURN_ANGULAR_SPEED_RADPS = 0.25
DEFAULT_TURN_DURATION_SECONDS = 0.50

_SIDE_SECTORS = {
    TURN_LEFT: ("front_left", "left", "rear_left"),
    TURN_RIGHT: ("front_right", "right", "rear_right"),
}


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _lidar_flags(state):
    geometry = state.get("local_motion_geometry") if isinstance(state, dict) else None
    return {
        "lidar_fresh": bool(
            isinstance(state, dict)
            and state.get("available") is True
            and state.get("valid") is True
            and state.get("reason") == "fresh"
        ),
        "geometry_valid": bool(
            isinstance(geometry, dict)
            and geometry.get("valid") is True
        ),
    }


def _side_clearance(evaluation, direction):
    """Return the conservative nearest sector clearance used for ranking.

    The score ranks already-rotationally-permitted options only.  It is not a
    separate collision threshold, and a CAUTION label is never a turn veto.
    """
    geometry = evaluation.get("geometry") if isinstance(evaluation, dict) else None
    sectors = geometry.get("sectors") if isinstance(geometry, dict) else None
    if not isinstance(sectors, dict):
        return None
    clearances = []
    for name in _SIDE_SECTORS[direction]:
        sector = sectors.get(name)
        value = (
            sector.get("minimum_distance_from_base_m")
            if isinstance(sector, dict)
            else None
        )
        if not _finite(value):
            return None
        clearances.append(value)
    return min(clearances)


def _evidence(evaluation, *, clearance=None):
    return {
        "permitted": bool(isinstance(evaluation, dict) and evaluation.get("permitted") is True),
        "reason": evaluation.get("reason") if isinstance(evaluation, dict) else "safety_evaluator_error",
        "relevant_clearance_m": clearance,
        "violating_point": (
            evaluation.get("violating_point") if isinstance(evaluation, dict) else None
        ),
    }


def _result(*, decision, reason, flags, forward, left, right):
    return {
        "decision": decision,
        "reason": reason,
        **flags,
        "forward": forward,
        "left": left,
        "right": right,
    }


def decide_forward_reaction(
    state,
    *,
    expected_session,
    forward_linear_speed=DEFAULT_FORWARD_LINEAR_SPEED_MPS,
    forward_duration=DEFAULT_FORWARD_DURATION_SECONDS,
    turn_angular_speed=DEFAULT_TURN_ANGULAR_SPEED_RADPS,
    turn_duration=DEFAULT_TURN_DURATION_SECONDS,
    now=None,
):
    """Choose a non-executable local reaction to one intended forward move.

    ``state`` is the existing producer-bound local LiDAR snapshot.  The
    function has no global-pose input: map and AMCL authority are irrelevant
    to this purely robot-relative collision decision.
    """
    flags = _lidar_flags(state)
    forward_safety = evaluate_local_motion_safety(
        state,
        expected_session=expected_session,
        linear_x=forward_linear_speed,
        duration=forward_duration,
        now=now,
    )
    forward = _evidence(forward_safety)

    if forward["permitted"]:
        return _result(
            decision=FORWARD_CLEAR,
            reason="forward_translation_protected_region_clear",
            flags=flags,
            forward=forward,
            left={"permitted": None, "reason": "not_evaluated_forward_clear", "relevant_clearance_m": None, "violating_point": None},
            right={"permitted": None, "reason": "not_evaluated_forward_clear", "relevant_clearance_m": None, "violating_point": None},
        )

    left_safety = evaluate_rotational_swept_footprint(
        state,
        expected_session=expected_session,
        direction="LEFT",
        angular_speed=turn_angular_speed,
        duration=turn_duration,
        now=now,
    )
    right_safety = evaluate_rotational_swept_footprint(
        state,
        expected_session=expected_session,
        direction="RIGHT",
        angular_speed=turn_angular_speed,
        duration=turn_duration,
        now=now,
    )
    left = _evidence(left_safety, clearance=_side_clearance(left_safety, TURN_LEFT))
    right = _evidence(right_safety, clearance=_side_clearance(right_safety, TURN_RIGHT))

    if not flags["lidar_fresh"] or not flags["geometry_valid"]:
        return _result(decision=STOP_BLOCKED, reason=forward["reason"], flags=flags,
                       forward=forward, left=left, right=right)
    if not left["permitted"] and not right["permitted"]:
        return _result(decision=STOP_BLOCKED, reason="no_rotationally_safe_turn", flags=flags,
                       forward=forward, left=left, right=right)
    if left["permitted"] and not right["permitted"]:
        return _result(decision=TURN_LEFT, reason="only_left_rotation_safe", flags=flags,
                       forward=forward, left=left, right=right)
    if right["permitted"] and not left["permitted"]:
        return _result(decision=TURN_RIGHT, reason="only_right_rotation_safe", flags=flags,
                       forward=forward, left=left, right=right)

    # The circular rotational safety contract makes collision permission
    # direction-invariant for identical geometry.  Direction choice therefore
    # ranks the nearest usable side-sector clearance, with LEFT as the stable
    # deterministic tie-breaker.
    if not _finite(left["relevant_clearance_m"]) or not _finite(right["relevant_clearance_m"]):
        return _result(decision=STOP_BLOCKED, reason="side_clearance_unavailable", flags=flags,
                       forward=forward, left=left, right=right)
    if left["relevant_clearance_m"] > right["relevant_clearance_m"]:
        return _result(decision=TURN_LEFT, reason="left_usable_clearance_greater", flags=flags,
                       forward=forward, left=left, right=right)
    if right["relevant_clearance_m"] > left["relevant_clearance_m"]:
        return _result(decision=TURN_RIGHT, reason="right_usable_clearance_greater", flags=flags,
                       forward=forward, left=left, right=right)
    return _result(decision=TURN_LEFT, reason="usable_clearance_tie_left_preferred", flags=flags,
                   forward=forward, left=left, right=right)
