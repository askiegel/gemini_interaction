"""Mission-owned range plausibility; candidate depth is not object identity.

No transport, odometry, visual-size arrival or alternate safety policy. A new
camera observation may update a trusted anchor; JIT checks never advance it.
Semantic tracker reseeding does not reset the range anchor.
"""
import math
import copy

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
# Three distinct adjacent rays support two intervals of a compact surface.
# This floor alone NEVER admits a target: two stopped, independent scan/frame
# pairs and an existing range anchor are additionally required.
SPARSE_MINIMUM_RAYS = 3
SPARSE_REQUIRED_OBSERVATIONS = 2


class MarvinTargetRangeAssociation:
    def __init__(self):
        self.anchor = None
        self.initial_anchor = None
        self.commanded_translation_bound_m = 0.0
        self.last_committed_stamp = None
        self.sparse_recovery = None
        self.sparse_confirmation = None

    def clear_sparse(self):
        self.sparse_recovery = None
        self.sparse_confirmation = None

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
            if speed * duration:
                self.clear_sparse()

    @staticmethod
    def _sparse_sample(candidate, lidar, tracker, identity):
        """Pure target-surface coherence; never infers distance from bbox size."""
        from lidar_perception import read_lidar_state
        session = lidar.get("producer_session")
        fresh = read_lidar_state(lidar, expected_session=session)
        native = lidar.get("bridge_scan_identity") or {}
        support = candidate.get("sparse_surface") or {}
        points = support.get("returns") or []
        competing = support.get("competing_ranges_m")
        quality, threshold = tracker.get("quality"), tracker.get("threshold")
        stamp = tracker.get("source_frame_stamp_ns")
        sequence = lidar.get("acquisition_sequence")
        if (not isinstance(identity, str) or not identity or not session
                or fresh.get("valid") is not True or fresh.get("available") is not True
                or fresh.get("reason") != "fresh" or candidate.get("ok") is not True
                or candidate.get("producer_session") != session
                or candidate.get("acquisition_sequence") != sequence
                or type(sequence) is not int or sequence < 0
                or type(stamp) is not int or stamp < 0
                or tracker.get("active") is not True or tracker.get("matched") is not True
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in (quality, threshold)) or quality < max(.80, threshold)
                or candidate.get("candidate_surface_bounded_by_bbox") is not True
                or not isinstance(competing, list)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in competing)
                or not SPARSE_MINIMUM_RAYS <= len(points) < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
                or len(points) != candidate.get("candidate_surface_point_count")
                or not isinstance(native.get("producer_session"), str) or not native["producer_session"]
                or any(type(native.get(k)) is not int or native[k] < 0
                       for k in ("acquisition_sequence", "source_stamp_ns"))):
            return None
        step = support.get("ray_step_degrees")
        interval = support.get("pixel_interval")
        if (type(step) not in (int, float) or not math.isfinite(step) or step <= 0
                or not isinstance(interval, list) or len(interval) != 2):
            return None
        fields = ("x_m", "y_m", "base_range_m", "pixel_x", "robot_bearing_deg")
        if any(any(type(p.get(k)) not in (int, float) or not math.isfinite(p[k])
                   for k in fields) or not interval[0] <= p["pixel_x"] <= interval[1]
               for p in points):
            return None
        ordered = sorted(points, key=lambda p: p["robot_bearing_deg"])
        if any(not 0 < b["robot_bearing_deg"] - a["robot_bearing_deg"] <= 1.5 * step
               for a, b in zip(ordered, ordered[1:])):
            return None
        if (max(p["base_range_m"] for p in points) - min(p["base_range_m"] for p in points)
                > RANGE_ASSOCIATION_TOLERANCE_M
                or max(math.hypot(a["x_m"]-b["x_m"], a["y_m"]-b["y_m"])
                       for a in points for b in points) > RANGE_ASSOCIATION_TOLERANCE_M):
            return None
        return {"identity": identity, "session": session, "sequence": sequence,
                "native": dict(native), "camera_stamp": stamp,
                "range_m": candidate["measured_distance_m"],
                "competing_ranges_m": competing,
                "center": [sum(p[k] for p in points)/len(points) for k in ("x_m", "y_m")]}

    def _sparse_range_continues(self, sample):
        anchor = self.anchor
        if not anchor:
            return False
        def plausible(distance):
            return (abs(distance-anchor["measured_distance_m"])
                <= RANGE_ASSOCIATION_TOLERANCE_M + self.commanded_translation_bound_m
                    - anchor["translation_bound_m"] + 1e-9
                and (self.initial_anchor is None or
                    abs(distance-self.initial_anchor["measured_distance_m"])
                        <= RANGE_ASSOCIATION_TOLERANCE_M + self.commanded_translation_bound_m + 1e-9))
        return bool(anchor and anchor.get("identity") == sample["identity"]
            and anchor["producer_session"] == sample["session"]
            and anchor.get("native_producer_session") == sample["native"]["producer_session"]
            and plausible(sample["range_m"])
            and not any(abs(distance-sample["range_m"]) <= 2*RANGE_ASSOCIATION_TOLERANCE_M
                        or plausible(distance) for distance in sample["competing_ranges_m"]))

    def begin_sparse_recovery(self, candidate, lidar, tracker, *, identity):
        self.clear_sparse()
        sample = self._sparse_sample(candidate, lidar, tracker, identity)
        if (sample is None or not self._sparse_range_continues(sample)
                or candidate.get("candidate_at_standoff") is not False):
            return False
        self.sparse_recovery = {"floor": sample, "observations": [],
                                "translation_bound": self.commanded_translation_bound_m}
        return True

    def corroborate_sparse(self, candidate, lidar, tracker, *, identity):
        """Called ONLY by the mission's verified-stopped recovery, not JIT."""
        pending = self.sparse_recovery
        sample = self._sparse_sample(candidate, lidar, tracker, identity)
        if (pending is None or sample is None or not self._sparse_range_continues(sample)
                or candidate.get("candidate_at_standoff") is not False):
            self.clear_sparse()
            return False
        prior = pending["observations"][-1] if pending["observations"] else pending["floor"]
        native, previous = sample["native"], prior["native"]
        if (sample["identity"] != prior["identity"] or sample["session"] != prior["session"]
                or native["producer_session"] != previous["producer_session"]
                or self.commanded_translation_bound_m != pending["translation_bound"]
                or any(sample[k] <= prior[k] for k in ("sequence", "camera_stamp"))
                or any(native[k] <= previous[k] for k in ("acquisition_sequence", "source_stamp_ns"))
                or abs(sample["range_m"] - prior["range_m"]) > RANGE_ASSOCIATION_TOLERANCE_M
                or math.dist(sample["center"], prior["center"]) > RANGE_ASSOCIATION_TOLERANCE_M):
            self.clear_sparse()
            return False
        pending["observations"].append(sample)
        if len(pending["observations"]) >= SPARSE_REQUIRED_OBSERVATIONS:
            self.sparse_confirmation = copy.deepcopy(sample)
            self.sparse_confirmation["translation_bound"] = self.commanded_translation_bound_m
            self.sparse_recovery = None
            return True
        return False

    def _sparse_confirmed(self, candidate, lidar, tracker, identity):
        confirmed = self.sparse_confirmation
        sample = self._sparse_sample(candidate, lidar, tracker, identity)
        if (confirmed is None or sample is None
                or not self._sparse_range_continues(sample)
                or candidate.get("candidate_at_standoff") is not False
                or sample["identity"] != confirmed["identity"]
                or sample["session"] != confirmed["session"]
                or sample["camera_stamp"] != confirmed["camera_stamp"]
                or self.commanded_translation_bound_m != confirmed["translation_bound"]
                or sample["native"]["producer_session"] != confirmed["native"]["producer_session"]
                or sample["sequence"] < confirmed["sequence"]
                or any(sample["native"][k] < confirmed["native"][k]
                       for k in ("acquisition_sequence", "source_stamp_ns"))
                or abs(sample["range_m"]-confirmed["range_m"]) > RANGE_ASSOCIATION_TOLERANCE_M
                or math.dist(sample["center"], confirmed["center"]) > RANGE_ASSOCIATION_TOLERANCE_M):
            self.sparse_confirmation = None
            return False
        return True

    def _with_route(self, result, lidar, session):
        route = evaluate_marvin_route(lidar, result, expected_session=session)
        return dict(result, route=route, **{k: route.get(k) for k in (
            "route_to_marvin_obstructed", "blocking_obstacle_distance_m",
            "blocking_obstacle_bearing_deg", "blocking_obstacle_x_m", "blocking_obstacle_y_m")})

    def evaluate(self, candidate, lidar, tracker, *, expected_session, commit=False, identity=None):
        anchor = self.anchor
        result = dict(candidate, target_range_association_trusted=False,
                      target_range_association_reason=candidate.get("reason"),
                      candidate_target_return_distance_m=candidate.get("measured_distance_m"),
                      verified_marvin_distance_m=anchor["measured_distance_m"] if anchor else None,
                      verified_marvin_conservative_distance_m=anchor["target_distance_m"] if anchor else None,
                      nearest_forward_obstacle_distance_m=None,
                      arrived_at_marvin=False, direct_path_blocked=False)
        if candidate.get("ok") is not True:
            self.clear_sparse()
            return result
        if anchor and anchor["producer_session"] != expected_session:
            # A valid new producer is still a sensor ownership change, never
            # an ambiguous foreground return eligible for an escape maneuver.
            self.clear_sparse()
            return dict(result, ok=False, reason="marvin_target_range_session_changed",
                        target_range_association_reason="marvin_target_range_session_changed")
        quality, threshold = tracker.get("quality"), tracker.get("threshold")
        if (tracker.get("active") is not True or tracker.get("matched") is not True
                or type(quality) not in (int, float) or not math.isfinite(quality)
                or type(threshold) not in (int, float) or not math.isfinite(threshold)
                or quality < max(0.80, threshold)
                or type(tracker.get("source_frame_stamp_ns")) is not int
                or tracker["source_frame_stamp_ns"] < 0):
            self.clear_sparse()
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
        sparse = candidate.get("candidate_surface_point_count", 0) < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
        corroborated = sparse and anchor is not None and anchor.get("identity") == identity and self._sparse_confirmed(
            candidate, lidar, tracker, identity)
        if sparse and not corroborated:
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
            if corroborated:
                self.clear_sparse()
            return self._with_route(dict(result, reason=reason, target_range_association_reason=reason),
                                    lidar, expected_session)
        result.update(target_range_association_trusted=True,
                      target_range_association_reason="marvin_sparse_surface_corroborated" if corroborated else
                          "marvin_range_continuity_verified" if anchor
                          else "marvin_initial_bounded_surface_clear",
                      verified_marvin_distance_m=measured,
                      verified_marvin_conservative_distance_m=candidate["target_distance_m"],
                      arrived_at_marvin=candidate["target_distance_m"] <= TARGET_STANDOFF_M)
        if corroborated:
            # Pursuit evidence only. Never updates the dense range anchor or
            # proves arrival, even after many sparse observations.
            result.update(arrived_at_marvin=False, sparse_surface_corroborated=True)
            return self._with_route(result, lidar, expected_session)
        stamp = tracker.get("source_frame_stamp_ns")
        if (commit and type(stamp) is int
                and (self.last_committed_stamp is None or stamp > self.last_committed_stamp)):
            self.anchor = {"measured_distance_m": measured,
                           "target_distance_m": candidate["target_distance_m"],
                           "producer_session": expected_session,
                           "acquisition_sequence": candidate["acquisition_sequence"],
                           "source_frame_stamp_ns": stamp,
                           "identity": identity,
                           "native_producer_session": (lidar.get("bridge_scan_identity") or {}).get("producer_session"),
                           "translation_bound_m": self.commanded_translation_bound_m}
            if self.initial_anchor is None:
                self.initial_anchor = dict(self.anchor)
            self.last_committed_stamp = stamp
        return self._with_route(result, lidar, expected_session)
