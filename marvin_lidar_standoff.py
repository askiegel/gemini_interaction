"""Pure candidate-range measurement for the strict V2 camera target.

Camera calibration is explicit: fx_pixels, cx_pixels, image_width, x_m,
y_m, yaw_degrees (camera forward in base_link), and range_uncertainty_m.
No FOV or camera pose is inferred from JPEG dimensions. The uncertainty must
bound range/calibration error; deployment must supply measured values.
"""

import math

from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
from local_motion_safety_envelope import (
    LOCAL_LIDAR_PROTECTED_RADIUS_M,
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
    OCTANT_SECTORS,
)
from marvin_pursuit_state import FIND_CENTER_TOLERANCE_PIXELS


TARGET_STANDOFF_M = 0.50
HARD_SAFETY_ENVELOPE_M = LOCAL_LIDAR_PROTECTED_RADIUS_M
# Association evidence tolerances only; never collision / freshness margins.
TARGET_RANGE_CLUSTER_TOLERANCE_M = 0.10
TARGET_RANGE_EDGE_PIXEL_MARGIN = 10.0
TRACKER_SEED_PADDING_FRACTION = 0.20


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def evaluate_marvin_lidar_standoff(tracker, lidar, camera_model, *, expected_session):
    """Measure the nearest surface projecting into the target's central band.

    Project base-link scan points into camera horizontal pixels, accounting
    for camera translation and yaw. Use at most the existing +/-50 px center
    tolerance, clipped to the actual bbox, rather than the whole front scan
    or a wide padded tracker box. Unrelated bearings never set target range.
    This is a bearing candidate, not verified Marvin depth or arrival authority.
    Runtime applies mission range continuity/structure before ARRIVED. The
    independent local-motion policy still checks surrounding obstacles.
    """
    result = {
        "ok": False, "arrived_at_marvin": False, "candidate_at_standoff": False,
        "target_range_association_trusted": False,
        "target_range_association_reason": "bearing_candidate_only",
        "verified_marvin_distance_m": None, "candidate_target_return_distance_m": None,
        "reason": None,
        "authority": "target_bearing_lidar", "target_standoff_m": TARGET_STANDOFF_M,
        "hard_safety_envelope_m": HARD_SAFETY_ENVELOPE_M,
        "hard_safety_condition": False, "target_distance_m": None,
        "measured_distance_m": None, "point_count": 0,
        "target_bearing_degrees": None, "bearing_interval_degrees": None,
        "producer_session": expected_session, "acquisition_sequence": None,
    }

    def fail(reason):
        return dict(result, reason=reason)

    if not isinstance(camera_model, dict):
        return fail("target_camera_calibration_unavailable")
    names = ("fx_pixels", "cx_pixels", "image_width", "x_m", "y_m",
             "yaw_degrees", "range_uncertainty_m")
    if (not all(_number(camera_model.get(key)) for key in names)
            or camera_model["fx_pixels"] <= 0 or camera_model["image_width"] <= 0
            or not 0 <= camera_model["cx_pixels"] <= camera_model["image_width"]
            or camera_model["range_uncertainty_m"] < 0):
        return fail("target_camera_calibration_invalid")
    if not isinstance(tracker, dict) or not isinstance(tracker.get("bbox"), dict):
        return fail("target_geometry_invalid")
    box = tracker["bbox"]
    width = tracker.get("image_width")
    height = tracker.get("image_height")
    if (not all(_number(box.get(key)) for key in ("x1", "y1", "x2", "y2"))
            or not _number(width) or not _number(height)
            or width != camera_model["image_width"] or height <= 0
            or not 0 <= box["x1"] < box["x2"] <= width
            or not 0 <= box["y1"] < box["y2"] <= height):
        return fail("target_geometry_invalid")
    geometry = lidar.get("local_motion_geometry") if isinstance(lidar, dict) else None
    if (not isinstance(lidar, dict) or lidar.get("available") is not True
            or lidar.get("valid") is not True or lidar.get("reason") != "fresh"
            or not expected_session or lidar.get("producer_session") != expected_session
            or not _number(lidar.get("effective_age_seconds"))
            or not 0 <= lidar["effective_age_seconds"] <= MAXIMUM_EFFECTIVE_AGE_SECONDS
            or type(lidar.get("acquisition_sequence")) is not int
            or lidar["acquisition_sequence"] < 0
            or not isinstance(geometry, dict) or geometry.get("valid") is not True
            or geometry.get("frame_id") != "lidar_link"
            or not isinstance(geometry.get("points"), list)):
        return fail("target_lidar_not_current")
    result["acquisition_sequence"] = lidar["acquisition_sequence"]
    center = (box["x1"] + box["x2"]) / 2.0
    left = max(box["x1"], center - FIND_CENTER_TOLERANCE_PIXELS)
    right = min(box["x2"], center + FIND_CENTER_TOLERANCE_PIXELS)
    fx, cx = camera_model["fx_pixels"], camera_model["cx_pixels"]
    yaw = math.radians(camera_model["yaw_degrees"])
    result["target_bearing_degrees"] = math.degrees(yaw + math.atan((cx - center) / fx))
    result["bearing_interval_degrees"] = [
        math.degrees(yaw + math.atan((cx - right) / fx)),
        math.degrees(yaw + math.atan((cx - left) / fx)),
    ]
    distances = []
    projected_points = []
    selected_points = {}
    for point in geometry["points"]:
        if (not isinstance(point, dict) or not _number(point.get("x_m"))
                or not _number(point.get("y_m"))):
            return fail("target_lidar_points_invalid")
        dx, dy = point["x_m"] - camera_model["x_m"], point["y_m"] - camera_model["y_m"]
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if forward <= 0:
            continue
        pixel = cx - fx * lateral / forward
        projected_points.append((pixel, math.hypot(point["x_m"], point["y_m"])))
        if left <= pixel <= right:
            distance_and_bearing = (math.hypot(point["x_m"], point["y_m"]),
                                    math.degrees(math.atan2(point["y_m"], point["x_m"])))
            distances.append(distance_and_bearing)
            selected_points.setdefault(distance_and_bearing, {
                "x_m": point["x_m"], "y_m": point["y_m"],
                "distance_m": distance_and_bearing[0], "robot_bearing_deg": distance_and_bearing[1],
            })
    result["point_count"] = len(distances)
    if len(distances) < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR:
        return fail("target_lidar_returns_insufficient")
    measured, measured_bearing = min(distances)
    # Supporting initial association evidence only: inspect nearby horizontal
    # bearings outside the full bbox for a continuation of the same near surface.
    result["candidate_surface_bounded_by_bbox"] = not any(
        box["x1"] - FIND_CENTER_TOLERANCE_PIXELS <= pixel <= box["x2"] + FIND_CENTER_TOLERANCE_PIXELS
        and not box["x1"] - TARGET_RANGE_EDGE_PIXEL_MARGIN <= pixel <= box["x2"] + TARGET_RANGE_EDGE_PIXEL_MARGIN
        and abs(depth - measured) <= TARGET_RANGE_CLUSTER_TOLERANCE_M
        for pixel, depth in projected_points)
    near_pixels = [pixel for pixel, depth in projected_points
                   if abs(depth - measured) <= TARGET_RANGE_CLUSTER_TOLERANCE_M
                   and box["x1"] - TARGET_RANGE_EDGE_PIXEL_MARGIN <= pixel
                   <= box["x2"] + TARGET_RANGE_EDGE_PIXEL_MARGIN]
    result["candidate_surface_point_count"] = len(near_pixels)
    # Strict tracker templates expand the semantic seed by 20% per side.
    # Require range support across the inner 60% of that tracked window.
    edge_margin = max(TARGET_RANGE_EDGE_PIXEL_MARGIN,
                      (box["x2"] - box["x1"]) * TRACKER_SEED_PADDING_FRACTION)
    result["candidate_surface_edges_supported"] = bool(near_pixels
        and min(near_pixels) <= box["x1"] + edge_margin
        and max(near_pixels) >= box["x2"] - edge_margin)
    result["selected_return"] = selected_points[(measured, measured_bearing)]
    result["measured_bearing_degrees"] = measured_bearing
    _, front_lower, front_upper = OCTANT_SECTORS[0]
    if not front_lower <= measured_bearing < front_upper:
        return fail("target_lidar_not_in_forward_sector")
    distance = max(0.0, measured - camera_model["range_uncertainty_m"])
    arrived = distance <= TARGET_STANDOFF_M
    return dict(result, ok=True, candidate_at_standoff=arrived,
                candidate_target_return_distance_m=measured,
                measured_distance_m=measured, target_distance_m=distance,
                hard_safety_condition=distance <= HARD_SAFETY_ENVELOPE_M,
                reason="marvin_lidar_standoff_reached" if arrived else "marvin_lidar_standoff_not_reached")
