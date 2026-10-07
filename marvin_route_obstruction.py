"""Fresh local corridor geometry, advisory only; never a motion admission gate."""
import math
from lidar_perception import read_lidar_state
from local_motion_safety_envelope import (
    LOCAL_LIDAR_PROTECTED_RADIUS_M, _distance_to_segment,
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
)
from marvin_lidar_standoff import TARGET_STANDOFF_M

LOCAL_ROUTE_LOOKAHEAD_M = 1.0
MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M = 0.01
MIN_ROUTE_CENTERLINE_CLEARANCE_IMPROVEMENT_M = 0.01


def evaluate_marvin_route(lidar, association, *, expected_session, translation_y=0.0, translation_x=0.0,
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
    # Transform the SAME target and all returns into the candidate robot frame.
    # Move the full target, then rebuild the standoff endpoint: translating an
    # already-truncated endpoint exaggerates the change in target bearing.
    theta = math.radians(bearing)
    target_depth = distance if trusted_history else LOCAL_ROUTE_LOOKAHEAD_M
    c, s = math.cos(heading_change), math.sin(heading_change)

    def transform(x, y):
        x -= translation_x
        y -= translation_y
        return c * x + s * y, -s * x + c * y

    world_tx, world_ty = target_depth * math.cos(theta) - translation_x, target_depth * math.sin(theta) - translation_y
    target_length = math.hypot(world_tx, world_ty)
    tx, ty = c * world_tx + s * world_ty, -s * world_tx + c * world_ty
    if trusted_history:
        horizon = min(LOCAL_ROUTE_LOOKAHEAD_M, max(0.0, target_length - TARGET_STANDOFF_M))
    predicted_bearing = math.degrees(math.atan2(ty, tx))
    target_x, target_y = ((horizon * world_tx / target_length, horizon * world_ty / target_length)
                          if target_length else (0.0, 0.0))
    length = math.hypot(target_x, target_y)
    blockers = []
    for point in geometry.get('points', []):
        x, y = point.get('x_m'), point.get('y_m')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (x, y)):
            return dict(base, reason='invalid_lidar_geometry')
        translated_x, translated_y = x - translation_x, y - translation_y
        if length == 0 or translated_x * target_x + translated_y * target_y <= 0:
            continue
        # Distances are invariant under the common rotation. Compute before
        # rotation so an exact boundary return cannot flicker by roundoff and
        # manufacture one fewer blocker / an apparent pure-turn improvement.
        gap = _distance_to_segment(translated_x, translated_y, target_x, target_y)
        if gap <= LOCAL_LIDAR_PROTECTED_RADIUS_M:
            bx, by = transform(x, y)
            blockers.append((gap, math.hypot(translated_x, translated_y), bx, by))
    overlap = max((LOCAL_LIDAR_PROTECTED_RADIUS_M - p[0] for p in blockers), default=0.0)
    nearest = min(blockers, key=lambda p: p[1]) if blockers else None
    return dict(base, valid=True, reason='marvin_route_obstructed' if blockers else 'marvin_route_clear',
        route_to_marvin_obstructed=bool(blockers), route_occupancy=len(blockers),
        corridor_overlap_m=overlap,
        blocking_obstacle_overlap_m=LOCAL_LIDAR_PROTECTED_RADIUS_M - nearest[0] if nearest else 0.0,
        blocking_obstacle_longitudinal_extent_m=max((p[2] for p in blockers), default=None),
        blocking_obstacle_minimum_side_separation_m=min((abs(p[3]) for p in blockers), default=None),
        route_lookahead_m=horizon,
        route_depth_source='verified_range_history' if trusted_history else 'bounded_local_lookahead',
        marvin_bearing_deg=predicted_bearing,
        heading_error_after_deg=predicted_bearing,
        blocking_obstacle_centerline_clearance_m=nearest[0] if nearest else None,
        blocking_obstacle_distance_m=nearest[1] if nearest else None,
        blocking_obstacle_bearing_deg=math.degrees(math.atan2(nearest[3], nearest[2])) if nearest else None,
        blocking_obstacle_x_m=nearest[2] if nearest else None,
        blocking_obstacle_y_m=nearest[3] if nearest else None)


def evaluate_route_progress(before, after):
    """Material route improvement; radial range is diagnostic only.

    One fewer return counts only without a material geometry regression.
    This is advisory planning evidence, never a change to motion clearance.
    """
    result = {'meaningful_progress': False, 'meaningful_progress_reason': 'route_evidence_invalid',
              'corridor_overlap_reduction_m': None, 'centerline_clearance_improvement_m': None,
              'route_occupancy_reduction': None, 'blocker_distance_change_m': None}
    if not before or not after or not before.get('valid') or not after.get('valid'):
        return result
    if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
           for row in (before, after) for key in
           ('corridor_overlap_m', 'blocking_obstacle_overlap_m', 'route_occupancy')):
        return result
    overlap = before['corridor_overlap_m'] - after['corridor_overlap_m']
    centerline = before.get('blocking_obstacle_overlap_m', 0) - after.get('blocking_obstacle_overlap_m', 0)
    occupancy = before['route_occupancy'] - after['route_occupancy']
    a, b = before.get('blocking_obstacle_distance_m'), after.get('blocking_obstacle_distance_m')
    result.update(corridor_overlap_reduction_m=overlap,
                  centerline_clearance_improvement_m=centerline, route_occupancy_reduction=occupancy,
                  blocker_distance_change_m=b - a if type(a) in (int, float) and type(b) in (int, float) else None)
    if not after['route_to_marvin_obstructed']:
        return dict(result, meaningful_progress=True, meaningful_progress_reason='route_cleared')
    if (overlap < -MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M + 1e-12
            or centerline < -MIN_ROUTE_CENTERLINE_CLEARANCE_IMPROVEMENT_M + 1e-12):
        return dict(result, meaningful_progress_reason='route_geometry_worsened')
    if overlap >= MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M - 1e-12:
        return dict(result, meaningful_progress=True, meaningful_progress_reason='corridor_overlap_improved')
    if centerline >= MIN_ROUTE_CENTERLINE_CLEARANCE_IMPROVEMENT_M - 1e-12:
        return dict(result, meaningful_progress=True, meaningful_progress_reason='blocker_centerline_clearance_improved')
    if occupancy >= 1:
        return dict(result, meaningful_progress=True, meaningful_progress_reason='blocking_return_count_reduced')
    return dict(result, meaningful_progress_reason='no_material_route_improvement')


def route_progress(before, after):
    return evaluate_route_progress(before, after)['meaningful_progress']
