"""Pure, read-only local obstacle geometry from Robot Bridge LaserScans.

Bounds are lower-inclusive, upper-exclusive (including +120 for left).
Distances are sensor-origin ranges, not robot-footprint clearances.
The robust metric is the linearly interpolated 10th percentile at (n-1)*0.1;
small samples offer limited outlier resistance, so counts accompany distances.
"""

import math


SECTOR_BOUNDS = {
    "front": (-20, 20),
    "front_left": (20, 60),
    "front_right": (-60, -20),
    "left": (60, 120),
    "right": (-120, -60),
}


def empty_sectors():
    return {
        name: {
            "valid_sample_count": 0,
            "minimum_clearance_m": None,
            "robust_clearance_m": None,
            "available": False,
        }
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
        result[name] = {
            "valid_sample_count": len(distances),
            "minimum_clearance_m": distances[0],
            "robust_clearance_m": (
                distances[lower]
                + (distances[upper] - distances[lower]) * (position - lower)
            ),
            "available": True,
        }
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
