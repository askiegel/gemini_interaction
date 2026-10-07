"""Fresh local corridor geometry, advisory only; never a motion admission gate."""
import math
from lidar_perception import read_lidar_state
from local_motion_safety_envelope import (
    LOCAL_LIDAR_PROTECTED_RADIUS_M, _distance_to_segment,
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
)
from marvin_lidar_standoff import TARGET_STANDOFF_M

LOCAL_ROUTE_LOOKAHEAD_M = 1.0
PROGRESS_EPSILON_M = 0.001  # Diagnostic geometry improvement, never a safety margin.


def evaluate_marvin_route(lidar, association, *, expected_session, translation_y=0.0,
                          heading_change=0.0):
    """Project one bounded candidate on current points, not a future scan.

    Trusted range limits the pursuit segment to the standoff. With ambiguous
    depth, inspect a bounded 1 m local lookahead instead of inventing Marvin's
    distance. No visual size or predicted geometry can authorize an action.
    """
    state = read_lidar_state(lidar, expected_session=expected_session)
    base = {'valid': False, 'route_to_marvin_obstructed': False,
            'blocking_obstacle_distance_m': None, 'blocking_obstacle_bearing_deg': None,
            'blocking_obstacle_x_m': None, 'blocking_obstacle_y_m': None,
            'route_occupancy': 0, 'corridor_overlap_m': 0.0, 'blocking_obstacle_overlap_m': 0.0,
            'route_width_m': 2 * LOCAL_LIDAR_PROTECTED_RADIUS_M,
            'reason': state.get('reason')}
    geometry = state.get('local_motion_geometry') or {}
    if not state.get('valid') or not state.get('available') or not geometry.get('valid'):
        return base
    sectors = geometry.get('sectors') or {}
    if any((sectors.get(n) or {}).get('valid_sample_count', 0) < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
           for n in ('front', 'front_left', 'front_right')):
        return dict(base, reason='insufficient_lidar_samples')
    bearing = association.get('target_bearing_degrees')
    if type(bearing) not in (int, float) or not math.isfinite(bearing):
        return dict(base, reason='marvin_route_bearing_unavailable')
    distance = association.get('verified_marvin_distance_m')
    trusted_history = type(distance) in (int, float) and math.isfinite(distance)
    horizon = min(LOCAL_ROUTE_LOOKAHEAD_M, max(0.0, distance - TARGET_STANDOFF_M)) if trusted_history else LOCAL_ROUTE_LOOKAHEAD_M
    # A strafe preserves the world-frame ray to the currently observed target;
    # rotation predicts an escape heading, which is reassessed after STOP.
    theta = math.radians(bearing) + heading_change
    target_x, target_y = horizon * math.cos(theta), horizon * math.sin(theta) - translation_y
    length = math.hypot(target_x, target_y)
    blockers = []
    for point in geometry.get('points', []):
        x, y = point.get('x_m'), point.get('y_m')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (x, y)):
            return dict(base, reason='invalid_lidar_geometry')
        y -= translation_y
        if length == 0 or x * target_x + y * target_y <= 0:
            continue
        gap = _distance_to_segment(x, y, target_x, target_y)
        if gap <= LOCAL_LIDAR_PROTECTED_RADIUS_M:
            blockers.append((gap, math.hypot(x, y), x, y))
    overlap = max((LOCAL_LIDAR_PROTECTED_RADIUS_M - p[0] for p in blockers), default=0.0)
    nearest = min(blockers, key=lambda p: p[1]) if blockers else None
    return dict(base, valid=True, reason='marvin_route_obstructed' if blockers else 'marvin_route_clear',
        route_to_marvin_obstructed=bool(blockers), route_occupancy=len(blockers),
        corridor_overlap_m=overlap,
        blocking_obstacle_overlap_m=LOCAL_LIDAR_PROTECTED_RADIUS_M - nearest[0] if nearest else 0.0,
        route_lookahead_m=horizon,
        route_depth_source='verified_range_history' if trusted_history else 'bounded_local_lookahead',
        marvin_bearing_deg=bearing,
        heading_error_after_deg=math.degrees(math.atan2(target_y, target_x)),
        blocking_obstacle_distance_m=nearest[1] if nearest else None,
        blocking_obstacle_bearing_deg=math.degrees(math.atan2(nearest[3], nearest[2])) if nearest else None,
        blocking_obstacle_x_m=nearest[2] if nearest else None,
        blocking_obstacle_y_m=nearest[3] if nearest else None)


def route_progress(before, after):
    """Compare independently refreshed geometry; association may have changed."""
    if not before or not after or not before.get('valid') or not after.get('valid'):
        return False
    a, b = before.get('blocking_obstacle_distance_m'), after.get('blocking_obstacle_distance_m')
    return (not after['route_to_marvin_obstructed']
            or after['route_occupancy'] < before['route_occupancy']
            or after['corridor_overlap_m'] < before['corridor_overlap_m'] - PROGRESS_EPSILON_M
            or after.get('blocking_obstacle_overlap_m', 0) < before.get('blocking_obstacle_overlap_m', 0) - PROGRESS_EPSILON_M
            or type(a) in (int, float) and type(b) in (int, float) and b > a + PROGRESS_EPSILON_M)
