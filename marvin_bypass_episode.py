"""Bounded detour history. No I/O, cached permission or displacement estimate."""
import copy
import math

from marvin_local_bypass import (
    LOCAL_BYPASS_HORIZON_M, BYPASS_FORWARD_SPEED_MPS, BYPASS_FORWARD_SECONDS,
    MIN_BYPASS_LATERAL_SEPARATION_M,
)
from local_motion_safety_envelope import LOCAL_LIDAR_PROTECTED_RADIUS_M


MAX_BYPASS_EPISODE_STEPS = 3


def start_bypass_episode(bypass, expected_session):
    """The selected step is advisory until physically dispatched and stopped.

    Count attempts, not odometry. Shorter commands never earn extra steps.
    A shorter initial horizon can only reduce the maximum.
    """
    horizon = bypass['bypass_distance_m']
    maximum = min(MAX_BYPASS_EPISODE_STEPS, math.floor(
        (min(horizon, LOCAL_BYPASS_HORIZON_M) + 1e-12)
        / (BYPASS_FORWARD_SPEED_MPS * BYPASS_FORWARD_SECONDS)))
    return dict(active=True, side=bypass['local_bypass_side'],
                producer_session=expected_session, step=1, max_steps=maximum)


def bypass_completion(result, bridge, action_lidar_evidence):
    """Certify a completed physical step, independently of its route progress."""
    selection = result.get('local_detour') or {}
    episode = selection.get('bypass_episode') or {}
    transport = (result.get('approach_result') or {}).get('forward_result') or {}
    motion = bridge.get('motion') or {}
    sequence = selection.get('acquisition_sequence')
    stamp = result.get('source_frame_stamp_ns')
    safe = (result.get('action_type') == 'BYPASS_FORWARD'
        and all(result.get(key) is True for key in (
            'ok', 'execution_authorized', 'motion_executed', 'full_step_completed', 'source_stamp_consumed'))
        and result.get('interrupted') is False
        and result.get('delivery_uncertain') is not True
        and result.get('bridge_stop_confirmed') is True
        and not result.get('error') and not transport.get('error')
        and transport.get('delivery_uncertain') is not True
        and transport.get('ok') is True and transport.get('executed') is True
        and transport.get('automatic_stop') is True and transport.get('returned_immediately') is False
        and transport.get('linear_x') == BYPASS_FORWARD_SPEED_MPS
        and transport.get('linear_y', 0.) == 0. and transport.get('angular_z') == 0.
        and type(transport.get('duration')) in (int, float)
        and 0 < transport['duration'] <= BYPASS_FORWARD_SECONDS
        and (result.get('stop_result') or {}).get('ok') is True
        and bridge.get('ok') is True and bridge.get('status') == 'READY' and bridge.get('ros_ready') is True
        and all(motion.get(key) == 0. for key in ('linear_x', 'linear_y', 'angular_z'))
        and motion.get('streaming') is False
        and type(sequence) is int and type(stamp) is int and stamp > 0
        and tuple(action_lidar_evidence or ()) == (selection.get('producer_session'), sequence))
    return dict(confirmed=safe, producer_session=selection.get('producer_session'),
                acquisition_sequence=sequence, source_frame_stamp_ns=stamp,
                side=selection.get('direction'), step=episode.get('step'))


def end_bypass_episode(selection, reason):
    """A failed episode cannot be reopened by repeated reads or alignment."""
    result = copy.deepcopy(selection)
    episode = result.get('bypass_episode')
    if isinstance(episode, dict):
        episode.update(active=False, ended_reason=reason)
        result['bypass_continuation'] = False
        ineffective = set(result.get('ineffective_action_types', []))
        ineffective.add('BYPASS_FORWARD')
        result['ineffective_action_types'] = sorted(ineffective)
    return result


def bypass_continuation(previous, bypass, progress, *, expected_session, current_sequence):
    """Fresh corridor admission after a STOPped, measured no-progress step.

    This never changes meaningful_progress. The next executor must independently
    acquire its strict observation, consume its stamp and perform ordinary JIT.
    """
    episode = previous.get('bypass_episode')
    diagnostic = dict(eligible=False, reason='bypass_episode_history_required')
    if not isinstance(episode, dict):
        return diagnostic
    diagnostic.update(step=episode.get('step'), max_steps=episode.get('max_steps'))
    diagnostic['reason'] = 'bypass_episode_inactive'
    if episode.get('active') is not True:
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_measured_progress'
    if progress.get('meaningful_progress') is True:
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_bound_reached'
    step, maximum = episode.get('step'), episode.get('max_steps')
    if (type(step) is not int or type(maximum) is not int
            or not 1 <= step < maximum <= MAX_BYPASS_EPISODE_STEPS):
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_side_or_session_changed'
    side = previous.get('direction')
    if (previous.get('action_type') != 'BYPASS_FORWARD' or side not in {'LEFT', 'RIGHT'}
            or episode.get('side') != side or bypass.get('local_bypass_side') != side
            or episode.get('producer_session') != expected_session
            or bypass.get('producer_session') != expected_session):
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_completion_required'
    completion = previous.get('bypass_step_completion')
    outcome = previous.get('bypass_step_outcome')
    if not isinstance(completion, dict) or not isinstance(outcome, dict):
        return diagnostic
    action_sequence, post_sequence = completion.get('acquisition_sequence'), outcome.get('acquisition_sequence')
    frozen = previous.get('first_post_action_bypass_progress')
    if (completion.get('confirmed') is not True or completion.get('side') != side
            or type(completion.get('step')) is not int or completion['step'] != step
            or type(completion.get('source_frame_stamp_ns')) is not int
            or completion['source_frame_stamp_ns'] <= 0
            or completion.get('producer_session') != expected_session
            or outcome.get('producer_session') != expected_session
            or type(outcome.get('action_acquisition_sequence')) is not int
            or outcome['action_acquisition_sequence'] != action_sequence
            or not isinstance(frozen, dict) or frozen.get('meaningful_progress') is not False
            or outcome.get('progress') != frozen
            or not all(type(v) is int for v in (action_sequence, post_sequence, current_sequence))
            or not action_sequence < post_sequence <= current_sequence):
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_suppressed'
    if ('BYPASS_FORWARD' in previous.get('ineffective_action_types', [])
            or previous.get('post_bypass_lateral_recovery_used', False) is not False
            or previous.get('post_bypass_lateral_recovery_selected') is True
            or previous.get('stationary_lateral_reconsidered') is True):
        return diagnostic
    diagnostic['reason'] = 'bypass_episode_fresh_corridor_required'
    capsule = bypass.get('forward_safety') or {}
    horizon, blocker_y = bypass.get('bypass_distance_m'), bypass.get('blocking_obstacle_y_m')
    if (bypass.get('bypass_forward_permitted') is not True
            or type(bypass.get('bypass_corridor_occupancy')) is not int
            or bypass['bypass_corridor_occupancy'] != 0
            or bypass.get('bypass_corridor_overlap_m') != 0.
            or bypass.get('route_to_bypass_obstructed') is not False
            or bypass.get('protected_radius_m') != LOCAL_LIDAR_PROTECTED_RADIUS_M
            or capsule.get('permitted') is not True
            or capsule.get('protected_radius_m') != LOCAL_LIDAR_PROTECTED_RADIUS_M
            or type(horizon) not in (int, float) or not math.isfinite(horizon)
            or not BYPASS_FORWARD_SPEED_MPS * BYPASS_FORWARD_SECONDS <= horizon <= LOCAL_BYPASS_HORIZON_M
            or bypass.get('bypass_target_x_m') != horizon or bypass.get('bypass_target_y_m') != 0.
            or type(blocker_y) not in (int, float) or not math.isfinite(blocker_y)
            or -(1 if side == 'LEFT' else -1) * blocker_y < MIN_BYPASS_LATERAL_SEPARATION_M
            or bypass.get('acquisition_sequence') != current_sequence):
        return diagnostic
    diagnostic.update(eligible=True, reason='bounded_safe_bypass_continuation', next_step=step+1)
    return diagnostic
