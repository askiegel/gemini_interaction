"""Fail-closed base-frame safety for bounded pure in-place turns.

The approved rotational safety zone is an exact 0.45 m radius circle centered
on ``base_link``. A circle is invariant under pure yaw, so direction and the
requested angle are command diagnostics rather than collision geometry.
Translation safety remains exclusively in ``local_motion_safety_envelope``.
"""

import math

from lidar_perception import read_lidar_state
from local_motion_safety_envelope import (
    EXPECTED_LIDAR_FRAME,
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
    OCTANT_SECTORS,
)
from voice_relay.lidar_sectors import finite_number


ROTATIONAL_PROTECTED_RADIUS_M = 0.45


def _finite_positive(value):
    return finite_number(value) and value > 0


def _validate_geometry(validated):
    geometry = validated.get("local_motion_geometry")
    if not isinstance(geometry, dict) or geometry.get("valid") is not True:
        return (None, geometry.get("reason", "missing_lidar_geometry")
                if isinstance(geometry, dict) else "missing_lidar_geometry")
    if geometry.get("frame_id") != EXPECTED_LIDAR_FRAME:
        return None, "unsupported_lidar_frame"
    points = geometry.get("points")
    sectors = geometry.get("sectors")
    if not isinstance(points, list) or not isinstance(sectors, dict):
        return None, "invalid_lidar_geometry"
    if any(
        not isinstance(sectors.get(name), dict)
        or sectors[name].get("valid_sample_count", 0)
        < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
        for name, _, _ in OCTANT_SECTORS
    ):
        return None, "insufficient_lidar_samples"
    valid_points = []
    for point in points:
        if not isinstance(point, dict) or not all(
                finite_number(point.get(key)) for key in ("x_m", "y_m")):
            return None, "invalid_lidar_geometry"
        valid_points.append(point)
    if not valid_points:
        return None, "insufficient_lidar_samples"
    return valid_points, None


def _result(*, permitted, reason, direction, requested_angle, geometry=None,
            violating_point=None):
    return {
        "permitted": permitted,
        "reason": reason,
        "direction": direction,
        "requested_angle_radians": requested_angle,
        "requested_angle_degrees": (
            math.degrees(requested_angle) if requested_angle is not None else None
        ),
        "model": "base_link_circular_rotational_envelope",
        "protected_radius_m": ROTATIONAL_PROTECTED_RADIUS_M,
        "angle_affects_collision_geometry": False,
        "geometry": geometry,
        "violating_point": violating_point,
    }


def evaluate_rotational_swept_footprint(state, *, expected_session, direction,
                                        angular_speed, duration, now=None):
    """Evaluate one pure in-place turn against the 0.45 m base-frame circle."""
    if direction not in {"LEFT", "RIGHT"}:
        return _result(permitted=False, reason="invalid_direction", direction=direction,
                       requested_angle=None)
    if not _finite_positive(angular_speed):
        return _result(permitted=False, reason="invalid_angular_speed", direction=direction,
                       requested_angle=None)
    if not _finite_positive(duration):
        return _result(permitted=False, reason="invalid_duration", direction=direction,
                       requested_angle=None)
    requested_angle = abs(angular_speed * duration)
    if not _finite_positive(requested_angle):
        return _result(permitted=False, reason="invalid_requested_angle", direction=direction,
                       requested_angle=None)

    validated = read_lidar_state(state, expected_session=expected_session, now=now)
    if not validated.get("available") or not validated.get("valid"):
        return _result(permitted=False,
                       reason=validated.get("reason", "untrusted_lidar_state"),
                       direction=direction, requested_angle=requested_angle)
    points, failure = _validate_geometry(validated)
    if failure:
        return _result(permitted=False, reason=failure, direction=direction,
                       requested_angle=requested_angle,
                       geometry=validated.get("local_motion_geometry"))
    violations = [
        point for point in points
        if math.hypot(point["x_m"], point["y_m"])
        <= ROTATIONAL_PROTECTED_RADIUS_M
    ]
    if violations:
        nearest = min(violations, key=lambda point: math.hypot(point["x_m"], point["y_m"]))
        return _result(permitted=False, reason="rotational_protected_region_violated",
                       direction=direction, requested_angle=requested_angle,
                       geometry=validated.get("local_motion_geometry"),
                       violating_point=nearest)
    return _result(permitted=True, reason="rotational_protected_region_clear",
                   direction=direction, requested_angle=requested_angle,
                   geometry=validated.get("local_motion_geometry"))
