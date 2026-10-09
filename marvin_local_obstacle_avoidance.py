"""Advisory local detour selection; never an executor or motion authority.

Turns use the existing circular rotational guard; strafes use full lateral
capsule geometry. Selection never commands motion or authorizes a later step.
Every physical step needs JIT validation and a new identity observation.
"""

import math

from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT, validate_guarded_turn
from local_motion_safety_envelope import evaluate_local_motion_safety
from marvin_blocked_wait import stationary_lateral_reconsideration


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


def _select_marvin_escape_action(state, association, *, expected_session,
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
    old = (previous_selection or {}).get('action_type')
    previous_route = (previous_selection or {}).get('route')
    if old:
        progress = evaluate_route_progress(previous_route, route)
        # A recovery's first stopped reassessment owns its outcome. An
        # intervening alignment cannot turn a failed recovery into progress.
        frozen = previous_selection.get('first_post_action_lateral_recovery_progress')
        if ((previous_selection.get('post_bypass_lateral_recovery_selected') is True
                or previous_selection.get('stationary_lateral_reconsidered') is True)
                and isinstance(frozen, dict) and type(frozen.get('meaningful_progress')) is bool):
            progress = dict(frozen)
        improved = progress['meaningful_progress']
        result['progress_improved'] = improved
        result.update(meaningful_progress=improved,
                      meaningful_progress_reason=progress['meaningful_progress_reason'],
                      actual_route_progress=progress)
        old_side = previous_selection.get('direction') or old.split('_')[1]
        for k, o in result['options'].items():
            o['reverses_previous_direction'] = k.split('_')[1] != old_side
            o['undoes_previous_progress'] = o['reverses_previous_direction'] and improved
        # Never repeat a non-improving primitive. A change of primitive on the
        # same side is allowed; reversing requires an unsafe/useless old option
        # and independently improved alternative clearance since last scan.
        ineffective = set((previous_selection or {}).get("ineffective_action_types", []))
        recovery_used = previous_selection.get('post_bypass_lateral_recovery_used')
        if recovery_used is not None:
            recovery_used = recovery_used is not False
            result['post_bypass_lateral_recovery_used'] = recovery_used
        if improved:
            if recovery_used is True:
                # A successful recovery establishes a new epoch, but does not
                # itself restore the failed bypass. A subsequent physical
                # action needs its own measured progress under ordinary policy.
                ineffective.discard(old)
                result['post_bypass_lateral_recovery_used'] = False
            else:
                ineffective.clear()
        else:
            ineffective.add(old)
        lateral = 'STRAFE_' + old_side
        option = result['options'].get(lateral, {})
        if (stationary_lateral_reconsideration(previous_selection, route)
                and option.get('permitted') is True and option.get('improves_route') is True):
            ineffective.discard(lateral)
            result['stationary_lateral_reconsidered'] = True
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
        return dict(result, reason='find_marvin_local_avoidance_no_progress' if old else 'find_marvin_no_safe_local_detour')
    chosen = rank_marvin_escape_options(result['options'], eligible)
    return dict(result, action_type=chosen, direction=chosen.split('_')[1], reason='find_marvin_local_detour_selected')


def _bypass_handoff(result, previous_selection, bypass, progress, *, expected_session,
                    remaining_avoidance_actions):
    """Compare lateral gain with forward passage after a measured lateral action.

    The first stopped outcome is history, not a permission. A route-clearing
    strafe wins; otherwise forward passage may win within the existing 3 mm
    ranking tolerance. Nominal motion is used only for the fresh prediction.
    """
    old = previous_selection or {}
    kind, side = result.get('action_type'), old.get('direction')
    lateral = result['options'].get(kind, {})
    lateral_progress = lateral.get('route_progress') or {}
    gain = max(lateral_progress.get('corridor_overlap_reduction_m') or 0.,
               lateral_progress.get('centerline_clearance_improvement_m') or 0., 0.)
    evidence = old.get('first_post_action_strafe_progress')
    evidence = evidence if isinstance(evidence, dict) else {}
    measured = evidence.get('progress')
    measured = measured if isinstance(measured, dict) else {}
    diagnostic = dict(eligible=False, selected=False, reason='same_side_lateral_history_required',
        established_side=side, current_avoidance_count=MAX_LOCAL_AVOIDANCE_ACTIONS-remaining_avoidance_actions,
        remaining_avoidance_actions=remaining_avoidance_actions,
        predicted_strafe_overlap_improvement_m=lateral_progress.get('corridor_overlap_reduction_m'),
        predicted_strafe_centerline_improvement_m=lateral_progress.get('centerline_clearance_improvement_m'),
        predicted_bypass_longitudinal_improvement_m=progress.get('bypass_longitudinal_progress_m'),
        bypass_corridor_occupancy=bypass.get('bypass_corridor_occupancy'),
        bypass_corridor_overlap_m=bypass.get('bypass_corridor_overlap_m'),
        current_side_clearance_m=result.get(str(side).lower()+'_clearance_m'),
        measured_lateral_progress=measured.get('meaningful_progress'),
        near_tie_tolerance_m=ROUTE_IMPROVEMENT_NEAR_TIE_M)
    if (side not in {'LEFT', 'RIGHT'} or old.get('action_type') != 'STRAFE_'+side
            or kind != 'STRAFE_'+side or result.get('direction') != side):
        return diagnostic
    diagnostic['reason'] = 'measured_lateral_progress_required'
    action_sequence, outcome_sequence = evidence.get('action_acquisition_sequence'), evidence.get('acquisition_sequence')
    current_sequence = result.get('acquisition_sequence')
    if (measured.get('meaningful_progress') is not True or result.get('progress_improved') is not True
            or evidence.get('producer_session') != expected_session
            or not all(type(v) is int for v in (action_sequence, outcome_sequence, current_sequence))
            or not action_sequence < outcome_sequence <= current_sequence):
        return diagnostic
    diagnostic['reason'] = 'failed_bypass_or_recovery_suppressed'
    if ('BYPASS_FORWARD' in old.get('ineffective_action_types', [])
            or 'BYPASS_FORWARD' in result.get('ineffective_action_types', [])
            or old.get('post_bypass_lateral_recovery_used', False) is not False
            or old.get('post_bypass_lateral_recovery_selected') is True
            or old.get('stationary_lateral_reconsidered') is True):
        return diagnostic
    diagnostic['reason'] = 'fresh_bypass_passage_required'
    passage = progress.get('bypass_longitudinal_progress_m')
    from marvin_local_bypass import MIN_BYPASS_PROGRESS_M
    if (bypass.get('local_bypass_side') != side or bypass.get('bypass_forward_permitted') is not True
            or bypass.get('bypass_corridor_occupancy') != 0 or bypass.get('bypass_corridor_overlap_m') != 0.
            or progress.get('meaningful_progress') is not True
            or type(passage) not in (int, float) or not math.isfinite(passage)
            or passage < MIN_BYPASS_PROGRESS_M - 1e-12):
        return diagnostic
    diagnostic['eligible'] = True
    if not lateral.get('predicted_route', {}).get('route_to_marvin_obstructed', True):
        diagnostic['reason'] = 'strafe_clears_direct_route'
    elif gain > passage + ROUTE_IMPROVEMENT_NEAR_TIE_M:
        diagnostic['reason'] = 'strafe_gain_exceeds_bypass_passage'
    else:
        diagnostic.update(selected=True, reason='established_corridor_forward_passage_preferred')
    return diagnostic


def select_marvin_escape_action(state, association, *, expected_session, allow_strafe,
                                previous_selection=None,
                                strafe_duration_limit=LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
                                remaining_avoidance_actions=MAX_LOCAL_AVOIDANCE_ACTIONS):
    """Rank lateral clearance against bounded passage on an established corridor.

    A pure turn cannot magically clear the same Marvin ray. Forward along a
    separate short free-space corridor is a different planning objective.
    The selected side persists; an unsafe old side can switch only with new,
    materially better clearance. No source data here authorizes transport.
    """
    from marvin_local_bypass import plan_local_bypass, evaluate_avoidance_progress
    from marvin_bypass_episode import start_bypass_episode, bypass_continuation, end_bypass_episode
    old = previous_selection or {}
    old_bypass = old.get('action_type') == 'BYPASS_FORWARD'
    result = _select_marvin_escape_action(state, association, expected_session=expected_session,
        allow_strafe=allow_strafe, previous_selection=None if old_bypass else previous_selection,
        strafe_duration_limit=strafe_duration_limit)
    if type(remaining_avoidance_actions) is not int or remaining_avoidance_actions <= 0:
        if old.get('bypass_episode'):
            ended = end_bypass_episode(old, 'avoidance_budget_exhausted')
            result.update(bypass_episode=ended['bypass_episode'],
                          ineffective_action_types=ended['ineffective_action_types'])
        return dict(result, action_type=None, direction=None,
                    bypass_continuation=False, bypass_episode_active=False,
                    bypass_episode_reason='avoidance_budget_exhausted',
                    reason='find_marvin_local_avoidance_exhausted')
    route = result['route']
    if not route.get('valid') or not route['route_to_marvin_obstructed']:
        if old.get('bypass_episode'):
            result.update(bypass_continuation=False, bypass_episode_active=False,
                bypass_episode_reason='direct_route_clear' if route.get('valid') else 'current_route_invalid')
        return result
    if old_bypass and old.get('direction') not in {'LEFT', 'RIGHT'}:
        return dict(result, action_type=None, direction=None,
                    reason='find_marvin_local_avoidance_history_invalid')
    side = old.get('direction')
    if side not in {'LEFT', 'RIGHT'}:
        left, right = result['left_clearance_m'], result['right_clearance_m']
        if left is None or right is None:
            return result
        # The temporary target must be on the other side of the blocker.
        # Rear-sector clearance alone cannot select an unrelated hemisphere.
        blocker_y = route.get('blocking_obstacle_y_m')
        side = ('LEFT' if blocker_y < 0 else 'RIGHT') if type(blocker_y) in (int, float) else (
            'RIGHT' if right > left + CLEARANCE_TIE_TOLERANCE_M else 'LEFT')
    bypass = plan_local_bypass(state, association, route, expected_session=expected_session, side=side)
    measured = evaluate_avoidance_progress(old, route, bypass) if old_bypass else None
    continuing = (bypass_continuation(old, bypass, measured,
        expected_session=expected_session, current_sequence=result.get('acquisition_sequence'))
        if old_bypass else {'eligible': False, 'reason': 'previous_bypass_required'})
    if old_bypass:
        result.update(bypass_episode_diagnostics=continuing, bypass_continuation=False,
            bypass_episode_active=False, bypass_episode_reason=continuing['reason'])
        if old.get('bypass_episode') and not continuing['eligible']:
            result['bypass_episode'] = dict(old['bypass_episode'], active=False,
                                           ended_reason=continuing['reason'])
    side_change_allowed = False
    if (old_bypass and not bypass['bypass_forward_permitted']
            and (not old.get('bypass_episode') or measured['meaningful_progress'])):
        alternative = 'RIGHT' if side == 'LEFT' else 'LEFT'
        prior = old.get(alternative.lower() + '_clearance_m')
        current = result.get(alternative.lower() + '_clearance_m')
        if (type(prior) in (int, float) and type(current) in (int, float)
                and current > prior + .01):
            other = plan_local_bypass(state, association, route, expected_session=expected_session, side=alternative)
            if other['bypass_forward_permitted']:
                side, bypass = alternative, other
                side_change_allowed = True
    result['local_bypass'] = bypass
    result['bypass_side_change_allowed'] = side_change_allowed
    if old_bypass:
        progress = evaluate_avoidance_progress(old, route, bypass)
        ineffective = set(old.get('ineffective_action_types', []))
        if progress['meaningful_progress']:
            # Measured passage establishes new geometry in which a previously
            # ineffective lateral primitive can be reconsidered, never replayed.
            ineffective.clear()
        if not progress['meaningful_progress'] and not side_change_allowed and not continuing['eligible']:
            ineffective.add('BYPASS_FORWARD')
            if old.get('bypass_handoff_from_strafe') is True or old.get('bypass_episode'):
                # Handoff did not mark the successful strafe ineffective. A
                # failed bypass nevertheless permits only the existing one-use
                # lateral recovery, never an unmarked ordinary retry.
                ineffective.add('STRAFE_' + old['direction'])
        if side_change_allowed:
            ineffective.discard('BYPASS_FORWARD')
        used = old.get('post_bypass_lateral_recovery_used', False) is not False
        recovery = 'STRAFE_' + old['direction']
        if used and not progress['meaningful_progress'] and not side_change_allowed:
            ineffective.add(recovery)
        option = result['options'].get(recovery, {})
        reconsider = (not progress['meaningful_progress'] and not side_change_allowed and not continuing['eligible']
            and (not used or stationary_lateral_reconsideration(old, route)) and recovery in ineffective
            and option.get('permitted') is True and option.get('hard_safety_permitted') is True
            and option.get('improves_route') is True
            and (option.get('route_progress') or {}).get('meaningful_progress') is True)
        if reconsider:
            # Remove only this fresh, material, same-side lateral prediction.
            # The failed bypass and unrelated ineffective primitives survive.
            ineffective.discard(recovery)
        result['ineffective_action_types'] = sorted(ineffective)
        eligible = [kind for kind, option in result['options'].items()
            if option['permitted'] and option['improves_route'] and kind not in ineffective
            and (kind.endswith('_' + old['direction']) or side_change_allowed)]
        chosen = rank_marvin_escape_options(result['options'], eligible) if eligible else None
        result.update(action_type=chosen, direction=chosen.split('_')[1] if chosen else None)
        if not chosen:
            result['reason'] = 'find_marvin_local_avoidance_no_progress'
        if used or reconsider:
            # Advisory until adopted as previous_selection after dispatch.
            # Repeated planning/JIT reads cannot accumulate recovery credits.
            result['post_bypass_lateral_recovery_used'] = used or chosen == recovery
            result['post_bypass_lateral_recovery_selected'] = reconsider and chosen == recovery
        if reconsider and chosen == recovery:
            result['reason'] = 'find_marvin_post_bypass_lateral_recovery_selected'
        result.update(actual_route_progress=progress, progress_improved=progress['meaningful_progress'],
            meaningful_progress=progress['meaningful_progress'],
            meaningful_progress_reason=progress['meaningful_progress_reason'])
        # Do not reverse away from a useful, still-safe bypass. Other primitives
        # may be reconsidered on the same side when forward progress stops.
        chosen = result.get('action_type')
        if chosen and result['direction'] != old['direction'] and not side_change_allowed:
            result.update(action_type=None, direction=None, reason='find_marvin_local_avoidance_no_progress')
    # Preserve the fallback/continuation and failed-recovery policies. Evaluate
    # fresh bypass passage alongside a useful strafe, without admitting motion.
    # Genuine progress uses the old path. A bounded no-progress episode needs
    # completion history and a fresh corridor without claiming measured success.
    use_bypass = (result['action_type'] is None or old_bypass and result['progress_improved']
                  or continuing['eligible'])
    if not bypass['bypass_forward_permitted']:
        result['bypass_handoff'] = {'eligible': False, 'selected': False,
            'reason': bypass['local_bypass_reason']}
        return result
    if old_bypass and not use_bypass:
        # A selected one-use recovery retains its reason and suppression. The
        # new handoff only compares ordinary same-side strafes, never recovery.
        return result
    if old_bypass and not result['progress_improved'] and not side_change_allowed and not continuing['eligible']:
        return dict(result, reason='find_marvin_local_bypass_no_progress')
    if 'BYPASS_FORWARD' in result.get('ineffective_action_types', old.get('ineffective_action_types', [])):
        result['bypass_handoff'] = {'eligible': False, 'selected': False,
            'reason': 'failed_bypass_or_recovery_suppressed'}
        return result
    from marvin_route_obstruction import evaluate_marvin_route
    prediction = evaluate_marvin_route(
        state, association, expected_session=expected_session, translation_x=.05)
    predicted_selection = dict(result, action_type='BYPASS_FORWARD', direction=side)
    predicted_bypass = dict(bypass)
    progress = evaluate_avoidance_progress(predicted_selection, prediction, predicted_bypass)
    handoff = _bypass_handoff(result, old, bypass, progress, expected_session=expected_session,
        remaining_avoidance_actions=remaining_avoidance_actions)
    result['bypass_handoff'] = handoff
    if not use_bypass and not handoff['selected']:
        return result
    result['options']['BYPASS_FORWARD'] = dict(permitted=True, hard_safety_permitted=True,
        reason='local_bypass_corridor_clear', requested_duration=.50,
        predicted_route=prediction, predicted_route_occupancy=prediction['route_occupancy'],
        predicted_max_overlap_m=prediction['corridor_overlap_m'],
        predicted_blocker_centerline_clearance_m=prediction.get('blocking_obstacle_centerline_clearance_m'),
        route_progress=progress, improves_route=progress['meaningful_progress'],
        side_clearance_m=result[side.lower() + '_clearance_m'],
        heading_error_deg=0., nominal_forward_displacement_m=.05, local_bypass=bypass)
    if not progress['meaningful_progress']:
        if continuing['eligible']:
            result.update(bypass_continuation=False, bypass_episode_active=False,
                bypass_episode=dict(old['bypass_episode'], active=False,
                    ended_reason='fresh_bypass_prediction_no_progress'),
                bypass_episode_reason='fresh_bypass_prediction_no_progress',
                ineffective_action_types=sorted(set(result.get('ineffective_action_types', [])) | {'BYPASS_FORWARD'}))
        return dict(result, reason='find_marvin_local_bypass_no_progress')
    if handoff['selected']:
        result['bypass_handoff_from_strafe'] = True  # History, never permission.
    if continuing['eligible']:
        episode = dict(old['bypass_episode'], step=continuing['next_step'])
    else:
        episode = start_bypass_episode(bypass, expected_session)
    result.update(bypass_episode=episode, bypass_episode_active=True,
        bypass_episode_step=episode['step'], bypass_episode_max_steps=episode['max_steps'],
        bypass_continuation=continuing['eligible'],
        bypass_episode_reason=continuing['reason'] if old_bypass else 'bypass_episode_started')
    if continuing['eligible'] and old.get('bypass_handoff_from_strafe') is True:
        result['bypass_handoff_from_strafe'] = True
    return dict(result, action_type='BYPASS_FORWARD', direction=side,
                reason='find_marvin_local_bypass_continuation_selected' if continuing['eligible'] else
                    'find_marvin_local_bypass_handoff_selected' if handoff['selected'] else
                    'find_marvin_local_bypass_selected')
