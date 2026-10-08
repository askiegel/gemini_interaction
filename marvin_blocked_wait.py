"""Stationary route-change diagnostics, never motion or identity authority."""
import copy
import math

from marvin_route_obstruction import (
    MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M,
    MIN_ROUTE_CENTERLINE_CLEARANCE_IMPROVEMENT_M,
    evaluate_route_progress,
)

BLOCKED_WAIT_REASONS = frozenset({
    "find_marvin_local_avoidance_no_progress",
    "find_marvin_local_bypass_no_progress",
    "find_marvin_no_safe_local_detour",
})
INITIAL_RECHECK_SECONDS = 0.5
MAX_RECHECK_SECONDS = 2.0
MAX_RECHECKS = 12
MAX_STATIONARY_SECONDS = 30.0


def blocked_wait_diagnostics():
    return {
        "blocked_wait_active": False,
        "blocked_wait_reason": None,
        "blocked_wait_recheck_count": 0,
        "blocked_wait_interval_seconds": INITIAL_RECHECK_SECONDS,
        "blocked_wait_initial_lidar_sequence": None,
        "blocked_wait_latest_lidar_sequence": None,
        "blocked_wait_route_obstructed": None,
        "blocked_wait_geometry_changed": False,
        "blocked_wait_camera_recheck_performed": False,
        "blocked_wait_resume_reason": None,
        "blocked_wait_total_seconds": 0.0,
    }


def material_route_change(before, after):
    """Compare stationary geometry to the entry scan, including accumulated drift.

    The same material overlap/centerline thresholds used by the selector apply.
    A blocker translation of 3 cm also warrants a new visual/route evaluation.
    A count-only fluctuation of one return is not a semantic trigger.
    """
    if not before.get("valid") or not after.get("valid"):
        return False
    if before.get("route_to_marvin_obstructed") != after.get("route_to_marvin_obstructed"):
        return True
    for key, threshold in (
        ("corridor_overlap_m", MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M),
        ("blocking_obstacle_overlap_m", MIN_ROUTE_CENTERLINE_CLEARANCE_IMPROVEMENT_M),
        ("blocking_obstacle_x_m", 0.03),
        ("blocking_obstacle_y_m", 0.03),
    ):
        a, b = before.get(key), after.get(key)
        if (all(type(v) in (int, float) and math.isfinite(v) for v in (a, b))
                and abs(a - b) >= threshold - 1e-12):
            return True
    return False


def stationary_geometry_epoch(previous_selection, entry_route, fresh_route):
    """Selective same-side reconsideration after measured external improvement.

    Frozen action outcomes remain historical truth. Neither waiting nor a pure
    alignment resets them. The shared selector must still independently predict
    improvement and reacquire hard safety/JIT evidence before any dispatch.
    """
    if (not previous_selection or not material_route_change(entry_route, fresh_route)
            or not evaluate_route_progress(entry_route, fresh_route)["meaningful_progress"]):
        return previous_selection
    side = previous_selection.get("direction")
    if side not in {"LEFT", "RIGHT"}:
        return previous_selection
    result = copy.deepcopy(previous_selection)
    result["stationary_geometry_epoch"] = {"entry_route": copy.deepcopy(entry_route)}
    return result


def stationary_lateral_reconsideration(previous_selection, current_route):
    """Re-check the epoch against CURRENT planning/JIT geometry, not a cached grant."""
    epoch = (previous_selection or {}).get("stationary_geometry_epoch")
    entry = epoch.get("entry_route") if isinstance(epoch, dict) else None
    return bool(isinstance(entry, dict) and material_route_change(entry, current_route)
                and evaluate_route_progress(entry, current_route)["meaningful_progress"])
