"""Immutable evidence interface for non-authoritative diagnostics.

Navigation intent is never motion authorization. These records contain facts
and exact source identifiers, not reusable transport permission. They have no
executor/JIT handles, callbacks, dispatch tokens, or stamp-consumption methods.
They are not cryptographic attestations: existing safety/execution code must
issue and validate the facts. Only the diagnostic worker issues these records.
Historical completion facts need not remain fresh forever; current observations
and geometry must pass age/session/mission checks at their actual use time.
"""
from dataclasses import dataclass
from enum import Enum
import math

from marvin_navigation_phases import Side


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError('exact positive integer required; never float-converted nanoseconds')


def number(value, *, nonnegative=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('finite measured value required')


def boolean(value):
    if type(value) is not bool:
        raise ValueError('explicit boolean required')


def text(value):
    if type(value) is not str or not value:
        raise ValueError('immutable nonempty identifier required')


class VisibilityClass(str, Enum):
    """Abstract visibility evidence; detour image margins remain uncalibrated."""
    CENTERING = 'CENTERING'
    VISIBILITY_MARGIN_LOW = 'VISIBILITY_MARGIN_LOW'
    TARGET_LOST = 'TARGET_LOST_OR_AUTHORITY_INVALID'
    CALIBRATION_REQUIRED = 'CALIBRATION_REQUIRED'


class DispatchedPrimitive(str, Enum):
    FORWARD = 'FORWARD'
    STRAFE_LEFT = 'STRAFE_LEFT'
    STRAFE_RIGHT = 'STRAFE_RIGHT'
    BYPASS_FORWARD = 'BYPASS_FORWARD'
    TURN_LEFT = 'TURN_LEFT'
    TURN_RIGHT = 'TURN_RIGHT'


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    mission_id: str
    producer_session: str
    sequence: int
    received_monotonic_seconds: float
    age_at_receipt_seconds: float

    def __post_init__(self):
        text(self.mission_id); text(self.producer_session); positive_int(self.sequence)
        number(self.received_monotonic_seconds); number(self.age_at_receipt_seconds)

    def require_current(self, *, mission_id: str, session: str, now: float, max_age: float):
        number(now); number(max_age)
        if max_age <= 0 or self.mission_id != mission_id or self.producer_session != session:
            raise ValueError('ownership/session/freshness policy mismatch')
        if now < self.received_monotonic_seconds or now-self.received_monotonic_seconds+self.age_at_receipt_seconds > max_age:
            raise ValueError('stale, future-dated, or foreign monotonic-clock evidence')


@dataclass(frozen=True, slots=True)
class ObservationCertificate:
    key: EvidenceKey
    source_stamp_ns: int
    identity_valid: bool
    tracker_accepted: bool
    tracker_quality: float | None
    tracker_threshold: float | None
    image_width: int | None
    bbox: tuple | None
    visibility: VisibilityClass
    target_range_trusted: bool
    target_range_m: float | None
    trusted_standoff: bool

    def __post_init__(self):
        if type(self.key) is not EvidenceKey or type(self.visibility) is not VisibilityClass:
            raise ValueError('typed immutable observation')
        positive_int(self.source_stamp_ns)
        for x in (self.identity_valid, self.tracker_accepted, self.target_range_trusted, self.trusted_standoff):
            boolean(x)
        for x in (self.tracker_quality, self.tracker_threshold):
            if x is not None:
                number(x)
                if x > 1: raise ValueError('normalized tracker metric')
        if self.image_width is not None: positive_int(self.image_width)
        if self.bbox is not None:
            if type(self.bbox) is not tuple or len(self.bbox) != 4: raise ValueError('immutable bbox')
            for x in self.bbox: number(x)
            if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]: raise ValueError('bbox extent')
            if self.image_width is None or self.bbox[2] > self.image_width: raise ValueError('bbox width')
        if self.target_range_m is not None: number(self.target_range_m)
        if self.target_range_trusted and self.target_range_m is None: raise ValueError('missing trusted range')
        if self.trusted_standoff and not self.target_range_trusted: raise ValueError('untrusted arrival')
        if self.tracker_accepted and (self.tracker_quality is None or self.tracker_threshold is None
                or self.tracker_quality < self.tracker_threshold or self.tracker_threshold < .80): raise ValueError('unsupported tracker acceptance')

    def require_strict_current(self, *, newer_than_stamp: int | None = None, **current):
        self.key.require_current(**current)
        if (not self.identity_valid or not self.tracker_accepted or self.bbox is None
                or self.visibility == VisibilityClass.TARGET_LOST):
            raise ValueError('strict target evidence absent')
        if newer_than_stamp is not None:
            positive_int(newer_than_stamp)
            if self.source_stamp_ns <= newer_than_stamp: raise ValueError('new action needs newer exact stamp')


@dataclass(frozen=True, slots=True)
class SideGeometry:
    side: Side
    lateral_feasible: bool | None
    lateral_separation_m: float | None
    side_clearance_m: float | None
    target_xy_m: tuple | None
    target_recomputed_sequence: int | None
    pass_occupancy: int | None
    pass_overlap_m: float | None
    pass_obstructed: bool | None
    protected_capsule_clear: bool | None
    predicted_longitudinal_gain_m: float | None

    def __post_init__(self):
        if type(self.side) is not Side: raise ValueError('side enum')
        for x in (self.lateral_feasible, self.pass_obstructed, self.protected_capsule_clear):
            if x is not None: boolean(x)
        for x in (self.lateral_separation_m, self.side_clearance_m, self.pass_overlap_m):
            if x is not None: number(x)
        if self.predicted_longitudinal_gain_m is not None: number(self.predicted_longitudinal_gain_m, nonnegative=False)
        if self.pass_occupancy is not None:
            if type(self.pass_occupancy) is not int or self.pass_occupancy < 0: raise ValueError('occupancy')
        if self.target_recomputed_sequence is not None: positive_int(self.target_recomputed_sequence)
        if self.target_xy_m is not None:
            if type(self.target_xy_m) is not tuple or len(self.target_xy_m) != 2: raise ValueError('immutable target')
            for x in self.target_xy_m: number(x, nonnegative=False)

    def pass_feasible(self, sequence: int) -> bool:
        # Feasibility snapshot only; ordinary safety/JIT must check again later.
        return (self.lateral_separation_m is not None and self.lateral_separation_m >= .15
            and self.target_xy_m is not None and self.target_xy_m[0] > 0
            and self.target_recomputed_sequence == sequence and self.pass_occupancy == 0
            and self.pass_overlap_m == 0 and self.pass_obstructed is False
            and self.protected_capsule_clear is True and self.predicted_longitudinal_gain_m is not None
            and self.predicted_longitudinal_gain_m >= .01)


@dataclass(frozen=True, slots=True)
class GeometryCertificate:
    key: EvidenceKey
    valid: bool
    required_sectors_valid: bool
    direct_route_valid: bool
    direct_route_obstructed: bool | None
    route_occupancy: int | None
    route_overlap_m: float | None
    blocker_xy_m: tuple | None
    sides: tuple  # Exactly LEFT and RIGHT, typed immutable geometry.
    protected_radius_m: float = .45

    def __post_init__(self):
        if type(self.key) is not EvidenceKey: raise ValueError('geometry key')
        for x in (self.valid, self.required_sectors_valid, self.direct_route_valid): boolean(x)
        if self.direct_route_obstructed is not None: boolean(self.direct_route_obstructed)
        if self.route_occupancy is not None and (type(self.route_occupancy) is not int or self.route_occupancy < 0): raise ValueError('occupancy')
        if self.route_overlap_m is not None: number(self.route_overlap_m)
        if type(self.sides) is not tuple or len(self.sides) != 2 or any(type(x) is not SideGeometry for x in self.sides):
            raise ValueError('immutable pair of side geometry')
        if {x.side for x in self.sides} != {Side.LEFT, Side.RIGHT}: raise ValueError('unique side coverage')
        if self.blocker_xy_m is not None:
            if type(self.blocker_xy_m) is not tuple or len(self.blocker_xy_m) != 2: raise ValueError('blocker tuple')
            for x in self.blocker_xy_m: number(x, nonnegative=False)
        number(self.protected_radius_m)
        if self.protected_radius_m != .45: raise ValueError('protected radius unchanged')

    def require_current(self, **current):
        self.key.require_current(**current)
        if (not self.valid or not self.required_sectors_valid or not self.direct_route_valid
                or self.direct_route_obstructed is None or self.route_occupancy is None or self.route_overlap_m is None):
            raise ValueError('current required geometry incomplete/invalid')
        if not self.direct_route_obstructed and (self.route_occupancy != 0 or self.route_overlap_m != 0):
            raise ValueError('inconsistent clear route')


@dataclass(frozen=True, slots=True)
class CompletionCertificate:
    primitive: DispatchedPrimitive
    used_source_stamp_ns: int
    planning: EvidenceKey
    jit: EvidenceKey
    outcome_geometry: GeometryCertificate
    outcome_observation: ObservationCertificate
    transport_confirmed: bool
    full_completion: bool
    stop_confirmed: bool
    bridge_zero: bool
    delivery_uncertain: bool
    measured_direct_route_progress: bool | None
    stopped_monotonic_seconds: float
    source_stamp_consumed: bool
    stop_lidar_sequence: int
    measured_lateral_gain_m: float | None = None
    # No commanded distance. A gain is comparative geometry, not robot odometry.

    def __post_init__(self):
        if type(self.primitive) is not DispatchedPrimitive or type(self.planning) is not EvidenceKey or type(self.jit) is not EvidenceKey:
            raise ValueError('typed completion')
        if type(self.outcome_geometry) is not GeometryCertificate or type(self.outcome_observation) is not ObservationCertificate:
            raise ValueError('immutable stopped outcome certificates')
        positive_int(self.used_source_stamp_ns)
        for x in (self.transport_confirmed, self.full_completion, self.stop_confirmed, self.bridge_zero, self.delivery_uncertain, self.source_stamp_consumed): boolean(x)
        number(self.stopped_monotonic_seconds); positive_int(self.stop_lidar_sequence)
        if self.measured_direct_route_progress is not None: boolean(self.measured_direct_route_progress)
        if self.measured_lateral_gain_m is not None: number(self.measured_lateral_gain_m, nonnegative=False)

    def require_completed(self, *, mission_id: str, session: str):
        keys=(self.planning, self.jit, self.outcome_geometry.key, self.outcome_observation.key)
        if any(k.mission_id != mission_id or k.producer_session != session for k in keys): raise ValueError('session/ownership failure')
        if not all((self.transport_confirmed, self.full_completion, self.stop_confirmed, self.bridge_zero, self.source_stamp_consumed)) or self.delivery_uncertain:
            raise ValueError('uncertain/incomplete physical action is fail-closed')
        if self.jit.sequence < self.planning.sequence or self.stop_lidar_sequence < self.jit.sequence or self.outcome_geometry.key.sequence <= self.stop_lidar_sequence:
            raise ValueError('new stopped same-session LiDAR required')
        if self.outcome_observation.source_stamp_ns <= self.used_source_stamp_ns:
            raise ValueError('new accepted post-action camera required')
        if not self.outcome_observation.identity_valid or not self.outcome_observation.tracker_accepted:
            raise ValueError('accepted strict outcome required')
        if not self.outcome_geometry.valid or not self.outcome_geometry.required_sectors_valid:
            raise ValueError('invalid stopped geometry')
        if self.jit.received_monotonic_seconds < self.planning.received_monotonic_seconds:
            raise ValueError('JIT chronology')
        if min(self.outcome_geometry.key.received_monotonic_seconds, self.outcome_observation.key.received_monotonic_seconds) <= self.stopped_monotonic_seconds or self.stopped_monotonic_seconds < self.jit.received_monotonic_seconds:
            raise ValueError('outcome must follow confirmed STOP')


@dataclass(frozen=True, slots=True)
class JitVetoCertificate:
    vetoed_primitive: DispatchedPrimitive
    retired_source_stamp_ns: int
    planning: EvidenceKey
    jit_geometry: GeometryCertificate
    deterministic_reason: str
    zero_transport_certified: bool
    delivery_uncertain: bool
    old_authority_retired: bool
    stop_confirmed: bool
    bridge_ready_ros_stationary: bool
    active_forward: bool
    pending_forward: bool
    stationary_certified_monotonic_seconds: float

    def __post_init__(self):
        if type(self.vetoed_primitive) is not DispatchedPrimitive or type(self.planning) is not EvidenceKey or type(self.jit_geometry) is not GeometryCertificate:
            raise ValueError('typed JIT veto')
        positive_int(self.retired_source_stamp_ns); text(self.deterministic_reason)
        number(self.stationary_certified_monotonic_seconds)
        for x in (self.zero_transport_certified, self.delivery_uncertain, self.old_authority_retired,
                  self.stop_confirmed, self.bridge_ready_ros_stationary, self.active_forward, self.pending_forward): boolean(x)

    def require_safe_replan(self, *, allowlisted_reasons: frozenset, **current):
        # Allowlist is owned by existing admission code, not invented here.
        if type(allowlisted_reasons) is not frozenset or self.deterministic_reason not in allowlisted_reasons:
            raise ValueError('not a certified deterministic veto')
        self.jit_geometry.require_current(**current)
        if self.planning.mission_id != current['mission_id'] or self.planning.producer_session != current['session']:
            raise ValueError('planning ownership/session mismatch')
        if self.jit_geometry.key.sequence < self.planning.sequence or self.jit_geometry.key.received_monotonic_seconds < self.planning.received_monotonic_seconds:
            raise ValueError('JIT must not precede planning')
        if not self.jit_geometry.key.received_monotonic_seconds <= self.stationary_certified_monotonic_seconds <= current['now']:
            raise ValueError('stationary STOP certificate chronology')
        if (not all((self.zero_transport_certified, self.old_authority_retired, self.stop_confirmed, self.bridge_ready_ros_stationary))
                or self.delivery_uncertain or self.active_forward or self.pending_forward):
            raise ValueError('no navigation replan from unsafe or uncertain veto')

    def require_new_observation(self, observation: ObservationCertificate, **current):
        if type(observation) is not ObservationCertificate: raise ValueError('typed new observation')
        observation.require_strict_current(newer_than_stamp=self.retired_source_stamp_ns, **current)
        if observation.key.received_monotonic_seconds <= self.stationary_certified_monotonic_seconds:
            raise ValueError('replan observation must follow JIT/stationary certification')
