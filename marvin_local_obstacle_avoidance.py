"""Advisory local detour selection; never an executor or motion authority.

Turns use the existing circular rotational guard; strafes use full lateral
capsule geometry. Selection never commands motion or authorizes a later step.
Every physical step needs JIT validation and a new identity observation.
"""

import math

from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT, validate_guarded_turn
from local_motion_safety_envelope import evaluate_local_motion_safety


TURN_SPEED = 0.25
TURN_DURATION = 0.50
MAX_LOCAL_AVOIDANCE_ACTIONS = 6
CLEARANCE_TIE_TOLERANCE_M = 0.01
_SIDE_SECTORS = {
    "LEFT": ("front_left", "left", "rear_left"),
    "RIGHT": ("front_right", "right", "rear_right"),
}


def select_marvin_detour(state, *, expected_session, forward_speed,
                         forward_duration, previous_direction=None,
                         previous_clearances=None, now=None):
    """Choose at most one turn from one fresh, producer-bound snapshot."""
    forward = evaluate_local_motion_safety(
        state, expected_session=expected_session, linear_x=forward_speed,
        duration=forward_duration, now=now)
    result = {"direction": None, "reason": forward["reason"],
              "direct_path_blocked": forward["permitted"] is not True,
              "producer_session": state.get("producer_session"),
              "acquisition_sequence": state.get("acquisition_sequence"),
              "left_clearance_m": None, "right_clearance_m": None,
              "options": {}}
    # A sensor/coverage/footprint failure must never become an escape request.
    if forward["permitted"] or forward["reason"] != "translation_protected_region_violated":
        return result
    for direction, sign in (("LEFT", 1), ("RIGHT", -1)):
        rotation = validate_guarded_turn(
            direction, TURN_SPEED, TURN_DURATION, state,
            expected_session=expected_session, target_directed=True,
            safety_mode=ROTATIONAL_SWEPT_FOOTPRINT, now=now)
        escape = evaluate_local_motion_safety(
            state, expected_session=expected_session, linear_y=sign * forward_speed,
            duration=forward_duration, now=now)
        sectors = (forward.get("geometry") or {}).get("sectors", {})
        distances = [sectors.get(name, {}).get("minimum_distance_from_base_m")
                     for name in _SIDE_SECTORS[direction]]
        clearance = (min(distances) if all(type(x) in (int, float)
                     and math.isfinite(x) and x >= 0 for x in distances) else None)
        result[direction.lower() + "_clearance_m"] = clearance
        result["options"][direction] = {
            "permitted": rotation["permitted"] and escape["permitted"] and clearance is not None,
            "rotation_reason": rotation["reason"], "escape_probe_reason": escape["reason"],
            "clearance_m": clearance,
        }
    eligible = [side for side, option in result["options"].items() if option["permitted"]]
    if not eligible:
        reason = ("marvin_single_approach_translation_vetoed" if all(
            option["rotation_reason"] == "rotational_protected_region_violated"
            for option in result["options"].values()) else "find_marvin_no_safe_local_detour")
        return dict(result, reason=reason)
    if len(eligible) == 1:
        chosen = eligible[0]
    else:
        chosen = ("RIGHT" if result["right_clearance_m"] >
                  result["left_clearance_m"] + CLEARANCE_TIE_TOLERANCE_M else "LEFT")
    if previous_direction in _SIDE_SECTORS and chosen != previous_direction:
        if result["options"][previous_direction]["permitted"]:
            # Do not undo a still-safe turn merely because rankings fluctuated.
            chosen = previous_direction
        else:
            # A reversal needs both a newly unsafe old option and independently
            # improved clearance on the alternative, beyond one probe length.
            old = (previous_clearances or {}).get(chosen)
            if (type(old) not in (int, float) or not math.isfinite(old)
                    or result[chosen.lower() + "_clearance_m"] <=
                    old + forward_speed * forward_duration):
                return dict(result, reason="find_marvin_local_avoidance_oscillation_blocked")
    return dict(result, direction=chosen, reason="find_marvin_local_detour_selected")


LOCAL_AVOIDANCE_STRAFE_SPEED_MPS = 0.08
LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS = 1.00
LOCAL_AVOIDANCE_STRAFE_MIN_SECONDS = 0.25
ROUTE_IMPROVEMENT_NEAR_TIE_M = 0.003


def safe_marvin_strafe_duration(state, *, expected_session, linear_y,
                               maximum_seconds=LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS):
    """Advisory bounded search; every tested path uses the strict full capsule.

    Collision-free lateral capsules are nested as duration grows. Retain the
    safe lower bound, never the colliding upper bound. The executor independently
    checks exactly the selected path again immediately before transport.
    """
    def evaluate(seconds):
        return evaluate_local_motion_safety(state, expected_session=expected_session,
            linear_y=linear_y, duration=seconds, lateral_swept_footprint=True)

    if (type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
            or not LOCAL_AVOIDANCE_STRAFE_MIN_SECONDS <= maximum_seconds <= LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS):
        return 0.0, {'permitted': False, 'reason': 'marvin_lateral_parameters_invalid'}
    full = evaluate(maximum_seconds)
    if full['permitted']:
        return maximum_seconds, full
    # Coverage, stale data and starting-footprint failures never admit an escape.
    if full['reason'] != 'translation_protected_region_violated':
        return 0.0, full
    lower = LOCAL_AVOIDANCE_STRAFE_MIN_SECONDS
    safe = evaluate(lower)
    if not safe['permitted']:
        return 0.0, safe
    upper = maximum_seconds
    for _ in range(12):
        middle = (lower + upper) / 2
        probe = evaluate(middle)
        if probe['permitted']:
            lower, safe = middle, probe
        else:
            upper = middle
    return lower, safe


def rank_marvin_escape_options(options, eligible):
    """Rank every useful primitive; strafe is only a comparable-score tie break.

    Route benefit is first. Within 3 mm, compare occupancy, clearance at 1 cm
    resolution and heading error at 1 degree resolution before strafe/LEFT.
    This helper is advisory and cannot admit motion.
    """
    def score(kind):
        option = options[kind]
        progress = option['route_progress']
        gain = max(progress['corridor_overlap_reduction_m'] or 0.0,
                   progress['centerline_clearance_improvement_m'] or 0.0, 0.0)
        return (not option['predicted_route']['route_to_marvin_obstructed'], gain,
                progress['route_occupancy_reduction'] or 0,
                round(option['side_clearance_m'] / CLEARANCE_TIE_TOLERANCE_M)
                    if option['side_clearance_m'] is not None else -1,
                -round(abs(option['heading_error_deg'] or 0.0)), kind.startswith('STRAFE'), kind.endswith('LEFT'))

    for kind in options:
        options[kind]['ranking_score'] = score(kind)
    if not eligible:
        return None
    best = max(eligible, key=score)
    first, gain = score(best)[:2]
    near = [kind for kind in eligible if score(kind)[0] == first
            and gain - score(kind)[1] <= ROUTE_IMPROVEMENT_NEAR_TIE_M]
    return max(near, key=lambda kind: score(kind)[2:])


def select_marvin_escape_action(state, association, *, expected_session,
                                allow_strafe, previous_selection=None,
                                strafe_duration_limit=LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS):
    """Evaluate four primitives from one scan; only JIT executors admit motion."""
    from marvin_route_obstruction import evaluate_marvin_route, evaluate_route_progress
    route = evaluate_marvin_route(state, association, expected_session=expected_session)
    result = {'action_type': None, 'direction': None, 'reason': route['reason'],
        'producer_session': state.get('producer_session'),
        'acquisition_sequence': state.get('acquisition_sequence'), 'route': route,
        'direct_path_blocked': association.get('direct_path_blocked', False),
        'left_clearance_m': None, 'right_clearance_m': None, 'options': {},
        'progress_improved': None}
    if not route.get('valid') or not route['route_to_marvin_obstructed']:
        return result
    sectors = state['local_motion_geometry']['sectors']
    for side in ('LEFT', 'RIGHT'):
        values = [(sectors.get(n) or {}).get('minimum_distance_from_base_m') for n in _SIDE_SECTORS[side]]
        result[side.lower() + '_clearance_m'] = min(values) if all(type(v) in (int, float) and math.isfinite(v) for v in values) else None
    for kind in ('STRAFE_LEFT', 'STRAFE_RIGHT', 'TURN_LEFT', 'TURN_RIGHT'):
        side = kind.split('_')[1]; sign = 1 if side == 'LEFT' else -1
        strafe = kind.startswith('STRAFE')
        if strafe:
            duration, safe = safe_marvin_strafe_duration(state, expected_session=expected_session,
                linear_y=sign * LOCAL_AVOIDANCE_STRAFE_SPEED_MPS,
                maximum_seconds=strafe_duration_limit)
        else:
            duration = TURN_DURATION
            safe = validate_guarded_turn(side, TURN_SPEED, TURN_DURATION, state,
                expected_session=expected_session, target_directed=True,
                safety_mode=ROTATIONAL_SWEPT_FOOTPRINT)
        dy = sign * LOCAL_AVOIDANCE_STRAFE_SPEED_MPS * duration if strafe else 0.0
        heading = 0.0 if strafe else sign * TURN_SPEED * TURN_DURATION
        prediction = evaluate_marvin_route(state, association, expected_session=expected_session,
            translation_y=dy, heading_change=heading)
        clearance = result[side.lower() + '_clearance_m']
        progress = evaluate_route_progress(route, prediction)
        improves = progress['meaningful_progress']
        permitted = safe['permitted'] and (allow_strafe or not strafe) and clearance is not None
        result['options'][kind] = {'permitted': permitted, 'hard_safety_permitted': safe['permitted'],
            'reason': safe['reason'] if allow_strafe or not strafe else 'bridge_lateral_support_unavailable',
            'side_clearance_m': clearance, 'requested_duration': duration,
            'nominal_lateral_displacement_m': abs(dy) if strafe else None,
            'predicted_route': prediction,
            'predicted_obstacle_clearance_m': prediction['blocking_obstacle_distance_m'],
            'predicted_route_occupancy': prediction['route_occupancy'],
            'heading_error_deg': prediction.get('heading_error_after_deg'),
            'predicted_max_overlap_m': prediction['corridor_overlap_m'],
            'predicted_blocker_centerline_clearance_m': prediction.get('blocking_obstacle_centerline_clearance_m'),
            'route_progress': progress,
            'improves_route': improves, 'reverses_previous_direction': False,
            'undoes_previous_progress': False}
    rank_marvin_escape_options(result['options'], [])  # Retain scores even when every option loses.
    eligible = [k for k, o in result['options'].items() if o['permitted'] and o['improves_route']]
    if not eligible:
        return dict(result, reason='find_marvin_no_safe_local_detour')
    old = (previous_selection or {}).get('action_type')
    previous_route = (previous_selection or {}).get('route')
    if old:
        progress = evaluate_route_progress(previous_route, route)
        improved = progress['meaningful_progress']
        result['progress_improved'] = improved
        result.update(meaningful_progress=improved,
                      meaningful_progress_reason=progress['meaningful_progress_reason'],
                      actual_route_progress=progress)
        old_side = old.split('_')[1]
        for k, o in result['options'].items():
            o['reverses_previous_direction'] = k.split('_')[1] != old_side
            o['undoes_previous_progress'] = o['reverses_previous_direction'] and improved
        # Never repeat a non-improving primitive. A change of primitive on the
        # same side is allowed; reversing requires an unsafe/useless old option
        # and independently improved alternative clearance since last scan.
        ineffective = set((previous_selection or {}).get("ineffective_action_types", []))
        if improved:
            ineffective.clear()
        else:
            ineffective.add(old)
        result["ineffective_action_types"] = sorted(ineffective)
        eligible = [k for k in eligible if k not in ineffective]
        for k in list(eligible):
            if k.split('_')[1] != old_side:
                prior = (previous_selection or {}).get(k.split('_')[1].lower() + '_clearance_m')
                now = result[k.split('_')[1].lower() + '_clearance_m']
                old_useful = old in eligible
                if old_useful or type(prior) not in (int, float) or now <= prior + 0.005:
                    eligible.remove(k)
        if not eligible:
            return dict(result, reason='find_marvin_local_avoidance_no_progress')
    chosen = rank_marvin_escape_options(result['options'], eligible)
    return dict(result, action_type=chosen, direction=chosen.split('_')[1], reason='find_marvin_local_detour_selected')
