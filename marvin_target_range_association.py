"""Mission-owned range plausibility; candidate depth is not object identity.

No transport, odometry, visual-size arrival or alternate safety policy. A new
camera observation may update a trusted anchor; JIT checks never advance it.
Semantic tracker reseeding does not reset the range anchor.
"""
import math

from local_motion_safety_envelope import (
    evaluate_local_motion_safety, _translation_approaches_point,
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
)
from marvin_lidar_standoff import TARGET_STANDOFF_M, TARGET_RANGE_CLUSTER_TOLERANCE_M
from marvin_route_obstruction import evaluate_marvin_route

# Bounds association noise, not LiDAR freshness, collision clearance or standoff.
# More than 10 cm unexplained range change requires re-association. Forward
# commands add their maximum possible translation, including interrupted ones.
RANGE_ASSOCIATION_TOLERANCE_M = TARGET_RANGE_CLUSTER_TOLERANCE_M
FORWARD_PROBE_SPEED = 0.10
FORWARD_PROBE_DURATION = 0.50


class MarvinTargetRangeAssociation:
    def __init__(self):
        self.anchor = None
        self.initial_anchor = None
        self.commanded_translation_bound_m = 0.0
        self.last_committed_stamp = None

    def record_forward_bound(self, speed, duration):
        """Accumulate the maximum commanded translation, forward or lateral.

        This permits range continuity, never proves realized displacement.
        Pure turns contribute zero; interrupted steps retain a permissive full
        requested bound rather than claiming completion.
        """
        if (type(speed) in (int, float) and type(duration) in (int, float)
                and math.isfinite(speed) and math.isfinite(duration)
                and speed >= 0 and duration >= 0):
            self.commanded_translation_bound_m += speed * duration

    def _with_route(self, result, lidar, session):
        route = evaluate_marvin_route(lidar, result, expected_session=session)
        return dict(result, route=route, **{k: route.get(k) for k in (
            "route_to_marvin_obstructed", "blocking_obstacle_distance_m",
            "blocking_obstacle_bearing_deg", "blocking_obstacle_x_m", "blocking_obstacle_y_m")})

    def evaluate(self, candidate, lidar, tracker, *, expected_session, commit=False):
        anchor = self.anchor
        result = dict(candidate, target_range_association_trusted=False,
                      target_range_association_reason=candidate.get("reason"),
                      candidate_target_return_distance_m=candidate.get("measured_distance_m"),
                      verified_marvin_distance_m=anchor["measured_distance_m"] if anchor else None,
                      verified_marvin_conservative_distance_m=anchor["target_distance_m"] if anchor else None,
                      nearest_forward_obstacle_distance_m=None,
                      arrived_at_marvin=False, direct_path_blocked=False)
        if candidate.get("ok") is not True:
            return result
        if anchor and anchor["producer_session"] != expected_session:
            # A valid new producer is still a sensor ownership change, never
            # an ambiguous foreground return eligible for an escape maneuver.
            return dict(result, ok=False, reason="marvin_target_range_session_changed",
                        target_range_association_reason="marvin_target_range_session_changed")
        quality, threshold = tracker.get("quality"), tracker.get("threshold")
        if (tracker.get("active") is not True or tracker.get("matched") is not True
                or type(quality) not in (int, float) or not math.isfinite(quality)
                or type(threshold) not in (int, float) or not math.isfinite(threshold)
                or quality < max(0.80, threshold)
                or type(tracker.get("source_frame_stamp_ns")) is not int
                or tracker["source_frame_stamp_ns"] < 0):
            return dict(result, reason="marvin_target_range_tracker_invalid",
                        target_range_association_reason="marvin_target_range_tracker_invalid")
        measured = candidate["measured_distance_m"]
        points = lidar["local_motion_geometry"]["points"]
        forward = [math.hypot(p["x_m"], p["y_m"]) for p in points
                   if p["x_m"] > abs(p["y_m"])]
        result["nearest_forward_obstacle_distance_m"] = min(forward) if forward else None
        direct = evaluate_local_motion_safety(lidar, expected_session=expected_session,
            linear_x=FORWARD_PROBE_SPEED, duration=FORWARD_PROBE_DURATION)
        result["association_forward_probe"] = {key: direct.get(key) for key in (
            "permitted", "reason", "violating_point", "required_sectors", "protected_radius_m")}
        result["direct_path_blocked"] = direct.get("reason") == "translation_protected_region_violated"
        reason = None
        if candidate.get("candidate_surface_point_count", 0) < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR:
            reason = "marvin_target_range_returns_ambiguous"
        elif direct.get("reason") not in {"protected_region_clear", "translation_protected_region_violated"}:
            reason = "marvin_target_range_geometry_unavailable"
        elif anchor:
            translation = self.commanded_translation_bound_m - anchor["translation_bound_m"]
            allowance = RANGE_ASSOCIATION_TOLERANCE_M + translation
            result.update(previous_verified_marvin_distance_m=anchor["measured_distance_m"],
                          range_change_allowance_m=allowance,
                          translation_since_verified_range_bound_m=translation)
            if (abs(measured - anchor["measured_distance_m"]) > allowance + 1e-9
                  or (self.initial_anchor is not None and
                      abs(measured - self.initial_anchor["measured_distance_m"]) >
                      RANGE_ASSOCIATION_TOLERANCE_M + self.commanded_translation_bound_m + 1e-9)):
                reason = ("marvin_foreground_obstruction_suspected" if measured < anchor["measured_distance_m"]
                          else "marvin_target_range_ambiguous")
        else:
            # A broad near surface extending outside the confirmed image box
            # is not an isolated Marvin association. A forward-blocking near
            # cluster is ambiguous even with many points / a high tracker score.
            if candidate.get("candidate_surface_bounded_by_bbox") is not True:
                reason = "marvin_target_range_initial_structure_ambiguous"
            elif any(_translation_approaches_point(p, FORWARD_PROBE_SPEED * FORWARD_PROBE_DURATION, 0)
                     for p in points if abs(math.hypot(p["x_m"], p["y_m"]) - measured)
                     <= RANGE_ASSOCIATION_TOLERANCE_M):
                reason = "marvin_foreground_obstruction_suspected"
            elif (candidate.get("candidate_at_standoff")
                  and candidate.get("candidate_surface_edges_supported") is not True):
                # A close center-band return alone cannot seed arrival. Need
                # near-surface support at both calibrated bbox edges, or a
                # previous trustworthy range; visual box size never sets depth.
                reason = "marvin_initial_arrival_association_ambiguous"
        if reason:
            return self._with_route(dict(result, reason=reason, target_range_association_reason=reason),
                                    lidar, expected_session)
        result.update(target_range_association_trusted=True,
                      target_range_association_reason="marvin_range_continuity_verified" if anchor
                          else "marvin_initial_bounded_surface_clear",
                      verified_marvin_distance_m=measured,
                      verified_marvin_conservative_distance_m=candidate["target_distance_m"],
                      arrived_at_marvin=candidate["target_distance_m"] <= TARGET_STANDOFF_M)
        stamp = tracker.get("source_frame_stamp_ns")
        if (commit and type(stamp) is int
                and (self.last_committed_stamp is None or stamp > self.last_committed_stamp)):
            self.anchor = {"measured_distance_m": measured,
                           "target_distance_m": candidate["target_distance_m"],
                           "producer_session": expected_session,
                           "acquisition_sequence": candidate["acquisition_sequence"],
                           "source_frame_stamp_ns": stamp,
                           "translation_bound_m": self.commanded_translation_bound_m}
            if self.initial_anchor is None:
                self.initial_anchor = dict(self.anchor)
            self.last_committed_stamp = stamp
        return self._with_route(result, lidar, expected_session)
