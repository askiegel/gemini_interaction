"""Pure advisory validation for a future bounded turn primitive."""

import math

from lidar_perception import read_lidar_state
from local_motion_safety_envelope import evaluate_local_motion_safety


MAX_ABSOLUTE_ANGULAR_SPEED = 1.0
MAX_TURN_DURATION_SECONDS = 1.0
TURN_DIRECTIONS = {"LEFT", "RIGHT"}
TRUSTWORTHY_FRONT_STATES = {"CLEAR", "CAUTION", "BLOCKED"}


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sector_status(sector):
    if not isinstance(sector, dict):
        return "UNKNOWN", None, False, None
    state = sector.get("state", "UNKNOWN")
    clearance = sector.get("robust_clearance_m")
    minimum = sector.get("minimum_clearance_m")
    trusted = (
        sector.get("available") is True
        and state == "CLEAR"
        and _finite(clearance)
        and _finite(minimum)
    )
    return (
        state,
        clearance if _finite(clearance) else None,
        trusted,
        minimum if _finite(minimum) else None,
    )


def _front_status(sector):
    state, clearance, _, minimum = _sector_status(sector)
    if not isinstance(sector, dict):
        return state, clearance, False, minimum
    trustworthy = (
        sector.get("available") is True
        and isinstance(state, str)
        and state in TRUSTWORTHY_FRONT_STATES
        and _finite(sector.get("robust_clearance_m"))
        and _finite(sector.get("minimum_clearance_m"))
    )
    return state, clearance, trustworthy, minimum


def _result(*, permitted, reason, direction, angular_z, duration,
            state, relevant, local_motion_safety=None):
    unknown = ("UNKNOWN", None, False, None)
    return {
        "permitted": permitted,
        "reason": reason,
        "direction": direction,
        "angular_z": angular_z,
        "duration": duration,
        "producer_session": state.get("producer_session") if isinstance(state, dict) else None,
        "effective_age_seconds": state.get("effective_age_seconds") if isinstance(state, dict) else None,
        "front_state": relevant.get("front", unknown)[0],
        "left_state": relevant.get("left", unknown)[0],
        "front_left_state": relevant.get("front_left", unknown)[0],
        "right_state": relevant.get("right", unknown)[0],
        "front_right_state": relevant.get("front_right", unknown)[0],
        "left_robust_clearance_m": relevant.get("left", unknown)[1],
        "front_left_robust_clearance_m": relevant.get("front_left", unknown)[1],
        "right_robust_clearance_m": relevant.get("right", unknown)[1],
        "front_right_robust_clearance_m": relevant.get("front_right", unknown)[1],
        "front_minimum_clearance_m": relevant.get("front", unknown)[3],
        "left_minimum_clearance_m": relevant.get("left", unknown)[3],
        "front_left_minimum_clearance_m": relevant.get("front_left", unknown)[3],
        "right_minimum_clearance_m": relevant.get("right", unknown)[3],
        "front_right_minimum_clearance_m": relevant.get("front_right", unknown)[3],
        "local_motion_safety": local_motion_safety,
    }


def validate_guarded_turn(direction, angular_speed, duration, state, *, expected_session, now=None,
                          target_directed=False):
    """Validate a bounded turn without authorizing or executing it.

    STOP remains unconditional and is intentionally outside this policy.
    """
    angular_value = angular_speed if _finite(angular_speed) else None
    duration_value = duration if _finite(duration) else None
    if not isinstance(direction, str) or direction not in TURN_DIRECTIONS:
        return _result(permitted=False, reason="invalid_direction", direction=direction,
                       angular_z=None, duration=duration_value, state=state, relevant={})
    if angular_value is None or angular_value <= 0:
        return _result(permitted=False, reason="invalid_angular_speed", direction=direction,
                       angular_z=None, duration=duration_value, state=state, relevant={})
    if angular_value > MAX_ABSOLUTE_ANGULAR_SPEED:
        return _result(permitted=False, reason="angular_speed_exceeds_limit", direction=direction,
                       angular_z=None, duration=duration_value, state=state, relevant={})
    if duration_value is None or duration_value <= 0:
        return _result(permitted=False, reason="invalid_duration", direction=direction,
                       angular_z=None, duration=duration_value, state=state, relevant={})
    if duration_value > MAX_TURN_DURATION_SECONDS:
        return _result(permitted=False, reason="duration_exceeds_limit", direction=direction,
                       angular_z=None, duration=duration_value, state=state, relevant={})

    validated = read_lidar_state(state, expected_session=expected_session, now=now)
    sectors = validated.get("sectors") if isinstance(validated, dict) else None
    if (not validated.get("available") or not validated.get("valid")
            or not isinstance(sectors, dict)):
        return _result(permitted=False, reason=validated.get("reason", "untrusted_lidar_state"),
                       direction=direction, angular_z=None, duration=duration_value,
                       state=validated, relevant={})

    # Target-directed turns use the forward corridor while retaining a
    # directional side veto for BLOCKED sectors. Broad side CLEAR requirements
    # remain the policy for search/avoidance turns.
    relevant_names = (
        ("left", "front_left") if direction == "LEFT" else ("right", "front_right")
    )
    relevant = {name: _sector_status(sectors.get(name)) for name in relevant_names}
    relevant["front"] = _front_status(sectors.get("front"))
    for name in ("left", "front_left", "right", "front_right"):
        relevant.setdefault(name, _sector_status(sectors.get(name)))
    if not relevant["front"][2]:
        return _result(permitted=False, reason="front_not_trustworthy", direction=direction,
                       angular_z=None, duration=duration_value, state=validated, relevant=relevant)
    if target_directed and relevant["front"][0] != "CLEAR":
        return _result(permitted=False, reason="front_not_clear", direction=direction,
                       angular_z=None, duration=duration_value, state=validated, relevant=relevant)
    if target_directed:
        side_clear = all(
            isinstance(sectors.get(name), dict)
            and sectors[name].get("available") is True
            and sectors[name].get("state") in {"CLEAR", "CAUTION"}
            and _finite(sectors[name].get("robust_clearance_m"))
            and _finite(sectors[name].get("minimum_clearance_m"))
            for name in relevant_names
        )
        if not side_clear:
            return _result(permitted=False, reason="turn_side_not_clear", direction=direction,
                           angular_z=None, duration=duration_value, state=validated, relevant=relevant)
    elif not all(relevant[name][2] for name in relevant_names):
        return _result(permitted=False, reason="turn_side_not_clear", direction=direction,
                       angular_z=None, duration=duration_value, state=validated, relevant=relevant)

    signed_speed = angular_value if direction == "LEFT" else -angular_value
    envelope = evaluate_local_motion_safety(
        validated, expected_session=expected_session, angular_z=signed_speed,
        duration=duration_value, now=now,
    )
    if not envelope["permitted"]:
        return _result(permitted=False, reason=envelope["reason"], direction=direction,
                       angular_z=None, duration=duration_value, state=validated,
                       relevant=relevant, local_motion_safety=envelope)
    return _result(permitted=True, reason="turn_side_clear_advisory", direction=direction,
                   angular_z=signed_speed, duration=duration_value,
                   state=validated, relevant=relevant,
                   local_motion_safety=envelope)
