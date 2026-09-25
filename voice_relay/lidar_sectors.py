"""Pure, read-only local obstacle geometry from Robot Bridge LaserScans.

Bounds are lower-inclusive, upper-exclusive (including +120 for left).
Distances are sensor-origin ranges, not robot-footprint clearances.
The robust metric is the linearly interpolated 10th percentile at (n-1)*0.1;
small samples offer limited outlier resistance, so counts accompany distances.
"""

import math


MAYDAY_SELF_BEARING_MIN_DEG = -15.0
MAYDAY_SELF_BEARING_MAX_DEG = -2.0
MAYDAY_SELF_MAX_RANGE_M = 0.15

def is_mayday_self_return(bearing_deg, distance):
    return MAYDAY_SELF_BEARING_MIN_DEG <= bearing_deg < MAYDAY_SELF_BEARING_MAX_DEG and distance <= MAYDAY_SELF_MAX_RANGE_M


SECTOR_BOUNDS = {
    "front": (-20, 20),
    "front_left": (20, 60),
    "front_right": (-60, -20),
    "left": (60, 120),
    "right": (-120, -60),
}

# Provisional sensor-origin perception thresholds, in meters. Order defines
# precedence: BLOCKED before CAUTION, then robust before minimum within a state.
CLASSIFICATION_RULES = (
    ("BLOCKED", "robust_clearance_m", 0.45, "robust_at_or_below_blocked_threshold"),
    ("BLOCKED", "minimum_clearance_m", 0.30, "minimum_at_or_below_blocked_threshold"),
    ("CAUTION", "robust_clearance_m", 0.75, "robust_at_or_below_caution_threshold"),
    ("CAUTION", "minimum_clearance_m", 0.60, "minimum_at_or_below_caution_threshold"),
)


def classify_sector(sector):
    """Return an enriched copy; CLEAR never authorizes physical motion."""
    state, reason = "UNKNOWN", "unavailable"
    if sector.get("available") is True and all(
        finite_number(sector.get(metric))
        for metric in ("robust_clearance_m", "minimum_clearance_m")
    ):
        state, reason = "CLEAR", "clear_of_provisional_thresholds"
        for candidate, metric, threshold, match_reason in CLASSIFICATION_RULES:
            if sector[metric] <= threshold:
                state, reason = candidate, match_reason
                break
    return {**sector, "state": state, "classification_reason": reason}


def classification_metadata():
    """Describe the same ordered rules used by the pure classifier."""
    return {
        "provisional": True,
        "read_only": True,
        "distance_reference": "sensor_origin",
        "units": "meters",
        "clear_authorizes_motion": False,
        "clearance_note": "Not guaranteed body or foot clearances.",
        "unknown_when": "unavailable or either distance metric missing/non-finite",
        "precedence": "first matching rule; otherwise CLEAR",
        "rules": [
            {"state": state, "metric": metric, "operator": "<=",
             "threshold_m": threshold, "classification_reason": reason}
            for state, metric, threshold, reason in CLASSIFICATION_RULES
        ],
    }


def empty_sectors():
    return {
        name: classify_sector({
            "valid_sample_count": 0,
            "minimum_clearance_m": None,
            "robust_clearance_m": None,
            "available": False,
        })
        for name in SECTOR_BOUNDS
    }


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def calculate_sectors(scan, *, rotation_radians=0.0):
    """Calculate robot bearings: zero forward, positive left, negative right.

    Callers explicitly supply the scan-to-robot rotation; no TF or ROS needed.
    Invalid geometry raises ValueError rather than fabricating distances.
    """
    if not isinstance(scan, dict):
        raise ValueError("LiDAR scan must be an object.")
    values = [scan.get(key) for key in (
        "angle_min", "angle_increment", "range_min", "range_max"
    )]
    if not all(finite_number(value) for value in values + [rotation_radians]):
        raise ValueError("LiDAR scan geometry must be finite.")
    angle_min, increment, range_min, range_max = values
    if (increment == 0 or range_min < 0 or range_max <= range_min
            or not isinstance(scan.get("ranges"), list)):
        raise ValueError("LiDAR scan geometry is invalid.")

    samples = {name: [] for name in SECTOR_BOUNDS}
    for index, distance in enumerate(scan["ranges"]):
        if not finite_number(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + index * increment + rotation_radians
        if not math.isfinite(angle):
            raise ValueError("LiDAR sample angle is invalid.")
        bearing = math.degrees((angle + math.pi) % math.tau - math.pi)
        if is_mayday_self_return(bearing, distance):
            continue
        # Stabilize exact boundary angles after radians/modulo roundoff.
        for boundary in (-120, -60, -20, 20, 60, 120):
            if abs(bearing - boundary) < 1e-10:
                bearing = boundary
                break
        for name, (lower, upper) in SECTOR_BOUNDS.items():
            if lower <= bearing < upper or (name == "left" and bearing == 120):
                samples[name].append(distance)
                break

    result = empty_sectors()
    for name, distances in samples.items():
        if not distances:
            continue
        distances.sort()
        position = (len(distances) - 1) * 0.1
        lower = math.floor(position)
        upper = math.ceil(position)
        result[name] = classify_sector({
            "valid_sample_count": len(distances),
            "minimum_clearance_m": distances[0],
            "robust_clearance_m": (
                distances[lower]
                + (distances[upper] - distances[lower]) * (position - lower)
            ),
            "available": True,
        })
    return result


def lidar_sector_payload(payload):
    """Adapt the existing telemetry.scan envelope without requiring a map."""
    payload = payload if isinstance(payload, dict) else {}
    telemetry = payload.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    scan = telemetry.get("scan")
    scan = scan if isinstance(scan, dict) else {}
    result = {
        "ok": False,
        "read_only": True,
        "perception": "local_obstacle_sectors",
        "coordinate_convention": "robot-relative: 0 forward, positive left",
        "distance_reference": "sensor_origin",
        "robust_statistic": "linear_10th_percentile",
        "classification_thresholds": classification_metadata(),
        "self_return_filter": {"enabled": True, "bearing_min_deg": MAYDAY_SELF_BEARING_MIN_DEG, "bearing_max_deg": MAYDAY_SELF_BEARING_MAX_DEG, "max_range_m": MAYDAY_SELF_MAX_RANGE_M},
        "source": {
            "frame_id": scan.get("frame_id"),
            "stamp_seconds": scan.get("stamp_seconds"),
            "received_at": telemetry.get("received_at"),
            "age_seconds": telemetry.get("age_seconds"),
        },
        "sectors": empty_sectors(),
    }
    if payload.get("ok") is not True or telemetry.get("available") is not True:
        result["error"] = payload.get("error") or "LiDAR telemetry is unavailable."
        return result
    # Same hardware-validated correction as operator_console.js's local overlay.
    if scan.get("frame_id") != "lidar_link":
        result["error"] = "Unsupported LiDAR frame; robot orientation is unknown."
        return result
    try:
        result["sectors"] = calculate_sectors(scan, rotation_radians=math.pi / 2)
    except ValueError as error:
        result["error"] = str(error)
        return result
    result["source"]["rotation_to_robot_radians"] = math.pi / 2
    result["ok"] = True
    return result
