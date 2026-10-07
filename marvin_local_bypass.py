"""Ephemeral, base-frame bypass planning. Never a mission goal or executor.

The first bypass policy follows a short forward corridor on the established
open side. It never issues mixed-axis motion or assumes commanded displacement
was realized. Every target is rebuilt from the next scan.
"""
import math

from lidar_perception import read_lidar_state
from local_motion_safety_envelope import (
    evaluate_local_motion_safety, _distance_to_segment,
    LOCAL_LIDAR_PROTECTED_RADIUS_M, MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
    OCTANT_SECTORS,
)
from marvin_lidar_standoff import TARGET_STANDOFF_M
from marvin_route_obstruction import evaluate_route_progress

LOCAL_BYPASS_HORIZON_M = 0.15  # Three bounded forward steps of observed free space.
MIN_BYPASS_LATERAL_SEPARATION_M = 0.15
MIN_BYPASS_PROGRESS_M = 0.01
BYPASS_FORWARD_SPEED_MPS = 0.10
BYPASS_FORWARD_SECONDS = 0.50


def plan_local_bypass(lidar, association, route, *, expected_session, side):
    """A full 0.45 m capsule to a short target alongside the current blocker.

    Range history limits the horizon, but cannot authorize motion. All octants,
    every local point, freshness and the exact forward envelope are checked.
    A point inside either starting or endpoint protected circle blocks bypass.
    Unknown Marvin depth does not permit a forward bypass toward an ambiguous
    close surface; lateral/turn planning can still run independently.
    """
    result = dict(local_bypass_active=False, local_bypass_side=side,
        local_bypass_target_x_m=None, local_bypass_target_y_m=None,
        bypass_target_x_m=None, bypass_target_y_m=None, bypass_distance_m=None,
        bypass_bearing_deg=None, route_to_bypass_obstructed=True,
        bypass_corridor_occupancy=None, bypass_corridor_overlap_m=None,
        bypass_forward_permitted=False, local_bypass_reason='bypass_evidence_unavailable',
        producer_session=lidar.get('producer_session'),
        acquisition_sequence=lidar.get('acquisition_sequence'),
        blocking_obstacle_x_m=route.get('blocking_obstacle_x_m'),
        blocking_obstacle_y_m=route.get('blocking_obstacle_y_m'))
    state = read_lidar_state(lidar, expected_session=expected_session)
    if not state.get('available') or not state.get('valid'):
        return dict(result, local_bypass_reason=state.get('reason'))
    if type(state.get('acquisition_sequence')) is not int or state['acquisition_sequence'] < 0:
        return dict(result, local_bypass_reason='bypass_lidar_sequence_invalid')
    geometry = state.get('local_motion_geometry') or {}
    if (not geometry.get('valid') or not route.get('valid')
            or not route.get('route_to_marvin_obstructed') or side not in {'LEFT', 'RIGHT'}):
        return result
    distance = association.get('verified_marvin_distance_m')
    if type(distance) not in (int, float) or not math.isfinite(distance) or distance <= TARGET_STANDOFF_M:
        return dict(result, local_bypass_reason='bypass_target_depth_unverified')
    conservative = association.get('verified_marvin_conservative_distance_m')
    if (type(conservative) not in (int, float) or not math.isfinite(conservative)
            or conservative - TARGET_STANDOFF_M < BYPASS_FORWARD_SPEED_MPS * BYPASS_FORWARD_SECONDS - 1e-12):
        return dict(result, local_bypass_reason='bypass_verified_standoff_margin_insufficient')
    x, y = route.get('blocking_obstacle_x_m'), route.get('blocking_obstacle_y_m')
    sign = 1 if side == 'LEFT' else -1
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (x, y))
            or x <= 0 or -sign * y < MIN_BYPASS_LATERAL_SEPARATION_M):
        return dict(result, local_bypass_reason='bypass_lateral_clearance_not_established')
    sectors = geometry.get('sectors') or {}
    if any((sectors.get(name) or {}).get('valid_sample_count', 0)
           < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR for name, _, _ in OCTANT_SECTORS):
        return dict(result, local_bypass_reason='insufficient_lidar_samples')
    horizon = min(LOCAL_BYPASS_HORIZON_M, distance - TARGET_STANDOFF_M)
    if horizon < BYPASS_FORWARD_SPEED_MPS * BYPASS_FORWARD_SECONDS:
        return dict(result, local_bypass_reason='bypass_horizon_too_short')
    points = geometry.get('points')
    if not isinstance(points, list) or not points or any(
            not isinstance(p, dict) or not all(type(p.get(k)) in (int, float)
                and math.isfinite(p[k]) for k in ('x_m', 'y_m')) for p in points):
        return dict(result, local_bypass_reason='invalid_lidar_geometry')
    gaps = [_distance_to_segment(p['x_m'], p['y_m'], horizon, 0.) for p in points]
    occupancy = sum(gap <= LOCAL_LIDAR_PROTECTED_RADIUS_M for gap in gaps)
    overlap = max(0., LOCAL_LIDAR_PROTECTED_RADIUS_M - min(gaps))
    safe = evaluate_local_motion_safety(state, expected_session=expected_session,
        linear_x=BYPASS_FORWARD_SPEED_MPS, duration=BYPASS_FORWARD_SECONDS)
    permitted = not occupancy and safe['permitted']
    return dict(result, local_bypass_active=permitted,
        local_bypass_target_x_m=horizon, local_bypass_target_y_m=0.,
        bypass_target_x_m=horizon, bypass_target_y_m=0.,
        bypass_distance_m=horizon, bypass_bearing_deg=0.,
        route_to_bypass_obstructed=bool(occupancy), bypass_corridor_occupancy=occupancy,
        bypass_corridor_overlap_m=overlap, bypass_forward_permitted=permitted,
        forward_safety={k: v for k, v in safe.items() if k != "geometry"}, protected_radius_m=LOCAL_LIDAR_PROTECTED_RADIUS_M,
        local_bypass_reason='local_bypass_corridor_clear' if permitted else
            'local_bypass_corridor_blocked' if occupancy else safe['reason'])


def evaluate_avoidance_progress(previous_selection, route, bypass=None):
    """Route progress, or measured longitudinal passage on a clear bypass.

    Only LiDAR geometry counts. A recreated target or nominal command distance
    is never evidence of progress. This measures the completed action; current
    corridor permission is separately required before another forward bypass.
    """
    before = (previous_selection or {}).get('route')
    progress = evaluate_route_progress(before, route)
    if (previous_selection or {}).get('action_type') != 'BYPASS_FORWARD':
        return progress
    if not route.get('route_to_marvin_obstructed'):
        return progress
    if (not before or not before.get('valid') or not route.get('valid') or not bypass
            or previous_selection.get('direction') != bypass.get('local_bypass_side')):
        return dict(progress, meaningful_progress=False, meaningful_progress_reason='bypass_progress_evidence_invalid')
    first_progress = previous_selection.get('first_post_action_bypass_progress')
    if isinstance(first_progress, dict) and type(first_progress.get('meaningful_progress')) is bool:
        return dict(first_progress)
    values = [row.get(key) for row in (before or {}, route)
              for key in ('blocking_obstacle_x_m', 'blocking_obstacle_y_m')]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        return dict(progress, meaningful_progress=False, meaningful_progress_reason='bypass_progress_evidence_invalid')
    old_x, old_y, x, y = values
    old_extent, extent = before.get('blocking_obstacle_longitudinal_extent_m'), route.get('blocking_obstacle_longitudinal_extent_m')
    if all(type(v) in (int, float) and math.isfinite(v) for v in (old_extent, extent)):
        old_x, x = old_extent, extent
    gain = old_x - x
    same_side = old_y * y > 0
    side_regression = abs(old_y) - abs(y)
    result = dict(progress, bypass_longitudinal_progress_m=gain,
                  bypass_side_clearance_change_m=-side_regression)
    if same_side and gain >= MIN_BYPASS_PROGRESS_M - 1e-12 and side_regression <= MIN_BYPASS_PROGRESS_M:
        return dict(result, meaningful_progress=True, meaningful_progress_reason='bypass_longitudinal_passage_improved')
    return result
