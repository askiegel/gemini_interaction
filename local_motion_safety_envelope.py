"""Pure base-link LiDAR geometry for bounded local-motion safety.

This policy neither performs transport nor authorizes ownership.  It converts
the existing ``lidar_link`` scan into base-link coordinates so established
callers can make fail-closed swept-footprint decisions from one producer-bound
World Model snapshot.
"""

import math

from voice_relay.lidar_sectors import finite_number, is_mayday_self_return


# Approved operational safety parameter; not a measured chassis/stance radius.
MAYDAY_OPERATIONAL_FOOTPRINT_RADIUS_M = 0.22
LOCAL_LIDAR_SAFETY_CLEARANCE_M = 0.45
LOCAL_LIDAR_PROTECTED_RADIUS_M = (
    MAYDAY_OPERATIONAL_FOOTPRINT_RADIUS_M + LOCAL_LIDAR_SAFETY_CLEARANCE_M
)

# Authoritative Mini Pupper 2 base_link -> lidar_link transform.
LIDAR_TO_BASE_X_M = -0.078701
LIDAR_TO_BASE_Y_M = 0.000495
LIDAR_TO_BASE_Z_M = 0.068457
LIDAR_TO_BASE_YAW_RADIANS = math.pi / 2
EXPECTED_LIDAR_FRAME = "lidar_link"
MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR = 5

OCTANT_SECTORS = (
    ("front", -22.5, 22.5),
    ("front_left", 22.5, 67.5),
    ("left", 67.5, 112.5),
    ("rear_left", 112.5, 157.5),
    ("rear", 157.5, -157.5),
    ("rear_right", -157.5, -112.5),
    ("right", -112.5, -67.5),
    ("front_right", -67.5, -22.5),
)

# Explicit centers keep the wrapped rear interval centered on ±180°, rather
# than accidentally averaging its endpoints to zero.
OCTANT_CENTER_BEARINGS_DEG = {
    "front": 0.0,
    "front_left": 45.0,
    "left": 90.0,
    "rear_left": 135.0,
    "rear": 180.0,
    "rear_right": -135.0,
    "right": -90.0,
    "front_right": -45.0,
}


def _normalize(angle):
    return (angle + math.pi) % math.tau - math.pi


def _sector_name(bearing_degrees):
    for name, lower, upper in OCTANT_SECTORS:
        if name == "rear":
            if bearing_degrees >= lower or bearing_degrees < upper:
                return name
        elif lower <= bearing_degrees < upper:
            return name
    return None


def _geometry_error(reason):
    return {"valid": False, "reason": reason, "points": [],
            "sectors": {name: {"valid_sample_count": 0, "available": False}
                        for name, _, _ in OCTANT_SECTORS}}


def build_local_motion_lidar_geometry(scan):
    """Return self-filtered valid ``lidar_link`` points in base_link XY."""
    if not isinstance(scan, dict) or scan.get("frame_id") != EXPECTED_LIDAR_FRAME:
        return _geometry_error("unsupported_lidar_frame")
    values = [scan.get(key) for key in
              ("angle_min", "angle_increment", "range_min", "range_max")]
    if (not all(finite_number(value) for value in values)
            or not isinstance(scan.get("ranges"), list)):
        return _geometry_error("invalid_lidar_geometry")
    angle_min, increment, range_min, range_max = values
    if increment == 0 or range_min < 0 or range_max <= range_min:
        return _geometry_error("invalid_lidar_geometry")

    sectors = {name: [] for name, _, _ in OCTANT_SECTORS}
    points = []
    filtered = 0
    for index, distance in enumerate(scan["ranges"]):
        if not finite_number(distance) or not range_min <= distance <= range_max:
            continue
        raw_angle = angle_min + index * increment
        base_angle = _normalize(raw_angle + LIDAR_TO_BASE_YAW_RADIANS)
        bearing_degrees = math.degrees(base_angle)
        # The known self mask is intentionally evaluated in the established
        # robot-relative angular convention before XY footprint geometry.
        if is_mayday_self_return(bearing_degrees, distance):
            filtered += 1
            continue
        x = LIDAR_TO_BASE_X_M + distance * math.cos(base_angle)
        y = LIDAR_TO_BASE_Y_M + distance * math.sin(base_angle)
        point = {"x_m": x, "y_m": y, "distance_m": distance,
                 "robot_bearing_deg": bearing_degrees}
        points.append(point)
        name = _sector_name(math.degrees(math.atan2(y, x)))
        if name is not None:
            sectors[name].append(math.hypot(x, y))

    diagnostics = {}
    for name, distances in sectors.items():
        diagnostics[name] = {
            "available": bool(distances),
            "valid_sample_count": len(distances),
            "minimum_distance_from_base_m": min(distances) if distances else None,
        }
    return {
        "valid": True,
        "reason": "fresh_geometry",
        "frame_id": EXPECTED_LIDAR_FRAME,
        "lidar_to_base": {"x_m": LIDAR_TO_BASE_X_M, "y_m": LIDAR_TO_BASE_Y_M,
                          "z_m": LIDAR_TO_BASE_Z_M,
                          "yaw_radians": LIDAR_TO_BASE_YAW_RADIANS},
        "self_return_filtered_count": filtered,
        "points": points,
        "sectors": diagnostics,
    }


def _finite_nonnegative(value):
    return finite_number(value) and value >= 0


def _distance_to_segment(x, y, end_x, end_y):
    length_squared = end_x * end_x + end_y * end_y
    if length_squared == 0:
        return math.hypot(x, y)
    projection = max(0.0, min(1.0, (x * end_x + y * end_y) / length_squared))
    return math.hypot(x - projection * end_x, y - projection * end_y)


def _translation_approaches_point(point, path_x, path_y):
    """Whether a bounded translation materially closes on an obstacle.

    A point must lie within 45 degrees of the translation direction, become
    closer at the bounded endpoint, and enter the swept protected tube.  The
    direction test prevents a nearby lateral point from becoming a veto merely
    because the capsule includes the robot's starting footprint.
    """
    path_length = math.hypot(path_x, path_y)
    if path_length == 0:
        return False
    x, y = point["x_m"], point["y_m"]
    longitudinal = (x * path_x + y * path_y) / path_length
    lateral = abs(x * path_y - y * path_x) / path_length
    start_distance = math.hypot(x, y)
    end_distance = math.hypot(x - path_x, y - path_y)
    return (
        longitudinal > lateral
        and end_distance < start_distance
        and _distance_to_segment(x, y, path_x, path_y)
        <= LOCAL_LIDAR_PROTECTED_RADIUS_M
    )


def _required_sectors(linear_x, linear_y, angular_z):
    if angular_z:
        return [name for name, _, _ in OCTANT_SECTORS]
    direction = math.degrees(math.atan2(linear_y, linear_x))
    return [
        name for name, _, _ in OCTANT_SECTORS
        if abs(math.degrees(_normalize(math.radians(
            direction - OCTANT_CENTER_BEARINGS_DEG[name]
        )))) <= 67.5
    ] or ["front"]


def evaluate_local_motion_safety(state, *, expected_session, linear_x=0.0,
                                 linear_y=0.0, angular_z=0.0, duration=0.0,
                                 now=None):
    """Evaluate a bounded command's circular-footprint clearance.

    Pure result only. Translation vetoes a point only when the bounded path
    materially approaches it and enters the protected tube. A circular
    footprint is invariant under pure rotation, so rotation alone does not
    create a static-clearance veto. Combined commands use their translation
    component without rotational enlargement.
    """
    # Delayed import avoids a construction-time cycle: lidar_perception
    # publishes geometry made by this pure module.
    from lidar_perception import read_lidar_state
    validated = read_lidar_state(state, expected_session=expected_session, now=now)
    result = {"permitted": False, "reason": None,
              "operational_footprint_radius_m": MAYDAY_OPERATIONAL_FOOTPRINT_RADIUS_M,
              "required_clearance_m": LOCAL_LIDAR_SAFETY_CLEARANCE_M,
              "protected_radius_m": LOCAL_LIDAR_PROTECTED_RADIUS_M,
              "required_sectors": [], "geometry": None}
    if not validated.get("available") or not validated.get("valid"):
        result["reason"] = validated.get("reason", "untrusted_lidar_state")
        return result
    values = (linear_x, linear_y, angular_z, duration)
    if (not all(finite_number(value) for value in values)
            or duration <= 0):
        result["reason"] = "invalid_motion_geometry"
        return result
    geometry = validated.get("local_motion_geometry")
    if not isinstance(geometry, dict) or geometry.get("valid") is not True:
        result["reason"] = (geometry.get("reason", "missing_lidar_geometry")
                            if isinstance(geometry, dict) else "missing_lidar_geometry")
        return result
    if geometry.get("frame_id") != EXPECTED_LIDAR_FRAME:
        result["reason"] = "unsupported_lidar_frame"
        return result
    points = geometry.get("points")
    if not isinstance(points, list):
        result["reason"] = "invalid_lidar_geometry"
        return result
    if linear_x == linear_y == angular_z == 0:
        result["reason"] = "invalid_motion_geometry"
        return result
    required = _required_sectors(linear_x, linear_y, angular_z)
    result["required_sectors"] = required
    sectors = geometry.get("sectors")
    if not isinstance(sectors, dict) or any(
            not isinstance(sectors.get(name), dict)
            or sectors[name].get("valid_sample_count", 0)
            < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
            for name in required):
        result["reason"] = "insufficient_lidar_samples"
        return result
    valid_points = []
    for point in points:
        if not isinstance(point, dict) or not all(finite_number(point.get(key))
                                                   for key in ("x_m", "y_m")):
            result["reason"] = "invalid_lidar_geometry"
            return result
        valid_points.append(point)
    if not valid_points:
        result["reason"] = "insufficient_lidar_samples"
        return result
    path_x, path_y = linear_x * duration, linear_y * duration
    result["geometry"] = geometry
    footprint_violations = [
        point for point in valid_points
        if math.hypot(point["x_m"], point["y_m"])
        <= MAYDAY_OPERATIONAL_FOOTPRINT_RADIUS_M
    ]
    if footprint_violations:
        nearest = min(footprint_violations,
                      key=lambda point: math.hypot(point["x_m"], point["y_m"]))
        result.update(reason="operational_footprint_violated", violating_point=nearest)
        return result
    violates = [
        point for point in valid_points
        if _translation_approaches_point(point, path_x, path_y)
    ]
    reason = "translation_protected_region_violated"
    if violates:
        nearest = min(violates, key=lambda point: math.hypot(point["x_m"], point["y_m"]))
        result.update(reason=reason, violating_point=nearest)
        return result
    result.update(permitted=True, reason="protected_region_clear")
    return result
