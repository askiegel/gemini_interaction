"""Immutable, observation-only policy vocabulary for OFFLINE SHADOW use.

A navigation intent is never motion authorization. Frame references identify
retained evidence only; they cannot be consumed here or reused for dispatch.
No runtime, Bridge handles, cached JIT permissions or transport APIs appear in
this interface. Safety status is supplied by existing admission components.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
import math

from marvin_navigation_phases import Event, Phase, Primitive, Side


class EvidenceHealth(str, Enum):
    OK = 'OK'
    UNKNOWN = 'UNKNOWN'
    BAD = 'BAD'


class Alignment(str, Enum):
    CENTERED = 'CENTERED'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    UNKNOWN = 'UNKNOWN'


class TargetVisibility(str, Enum):
    TRACK_OK = 'TRACK_OK'
    MAINTAIN_LEFT = 'MAINTAIN_LEFT'
    MAINTAIN_RIGHT = 'MAINTAIN_RIGHT'
    LOST = 'LOST'
    CALIBRATION_REQUIRED = 'CALIBRATION_REQUIRED'


class IntentKind(str, Enum):
    DIRECT_FORWARD = 'DIRECT_FORWARD'
    DIRECT_ALIGN_LEFT = 'DIRECT_ALIGN_LEFT'
    DIRECT_ALIGN_RIGHT = 'DIRECT_ALIGN_RIGHT'
    STRAFE_LEFT = 'STRAFE_LEFT'
    STRAFE_RIGHT = 'STRAFE_RIGHT'
    PASS_FORWARD = 'PASS_FORWARD'
    VISIBILITY_TURN_LEFT = 'VISIBILITY_TURN_LEFT'
    VISIBILITY_TURN_RIGHT = 'VISIBILITY_TURN_RIGHT'
    STOP_REVERIFY = 'STOP_REVERIFY'
    WAIT = 'WAIT'
    ARRIVED = 'ARRIVED'
    SAFE_INCOMPLETE = 'SAFE_INCOMPLETE'
    FAIL_CLOSED = 'FAIL_CLOSED'


PHYSICAL_INTENTS = frozenset({IntentKind.DIRECT_FORWARD, IntentKind.DIRECT_ALIGN_LEFT,
    IntentKind.DIRECT_ALIGN_RIGHT, IntentKind.STRAFE_LEFT, IntentKind.STRAFE_RIGHT,
    IntentKind.PASS_FORWARD, IntentKind.VISIBILITY_TURN_LEFT, IntentKind.VISIBILITY_TURN_RIGHT})


@dataclass(frozen=True)
class NavigationIntent:
    kind: IntentKind
    reason: str
    motion_authority: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.kind, IntentKind) or not isinstance(self.reason, str):
            raise ValueError('IntentKind required')


@dataclass(frozen=True)
class EvidenceReference:
    now: float
    producer_session: str | None
    lidar_sequence: int | None
    frame_stamp_ns: int | None  # Read-only identification; NEVER authority.
    current: bool

    def __post_init__(self):
        if type(self.now) not in (int, float) or not math.isfinite(self.now) or self.now < 0:
            raise ValueError('Finite monotonic reference time required')
        for value in (self.lidar_sequence, self.frame_stamp_ns):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Evidence identifiers must be exact nonnegative integers')
        if self.producer_session is not None and not isinstance(self.producer_session, str):
            raise ValueError('Session reference must be an immutable string')
        if type(self.current) is not bool:
            raise ValueError('Current evidence flag must be boolean')


@dataclass(frozen=True)
class TargetEvidence:
    identity_tracker: EvidenceHealth = EvidenceHealth.UNKNOWN
    visible: EvidenceHealth = EvidenceHealth.UNKNOWN
    trusted_standoff: bool = False
    direct_pursuit_admissible: bool = False
    alignment: Alignment = Alignment.UNKNOWN
    visibility: TargetVisibility = TargetVisibility.CALIBRATION_REQUIRED
    horizontal_error_px: float | None = None
    tracker_quality: float | None = None


@dataclass(frozen=True)
class HealthEvidence:
    ownership: EvidenceHealth = EvidenceHealth.UNKNOWN
    session: EvidenceHealth = EvidenceHealth.UNKNOWN
    geometry: EvidenceHealth = EvidenceHealth.UNKNOWN
    required_sectors: EvidenceHealth = EvidenceHealth.UNKNOWN
    stopped_bridge: EvidenceHealth = EvidenceHealth.UNKNOWN
    delivery_resolved: EvidenceHealth = EvidenceHealth.UNKNOWN
    required_evidence: EvidenceHealth = EvidenceHealth.UNKNOWN


@dataclass(frozen=True)
class SideEvidence:
    side: Side
    lateral_feasible: bool | None = None
    separation_m: float | None = None
    pass_feasible: bool | None = None
    pass_occupancy: int | None = None
    pass_overlap_m: float | None = None
    protected_capsule_clear: bool | None = None
    target_recomputed: bool | None = None

    @property
    def passage_feasible(self):
        # Combine supplied facts, never calculate or authorize a safety capsule.
        return (self.pass_feasible is True and self.pass_occupancy == 0
            and self.pass_overlap_m == 0. and self.protected_capsule_clear is True
            and self.target_recomputed is True)


@dataclass(frozen=True)
class OutcomeEvidence:
    last_completed_primitive: Primitive | None = None
    completion_confirmed: bool = False
    comparable_geometry: bool = False
    observed_lateral_gain_m: float | None = None
    observed_pass_gain_m: float | None = None
    recorded_meaningful_route_progress: bool | None = None  # Diagnostic only.
    material_external_change: bool = False
    event_frame_stamp_ns: int | None = None  # Completion/veto reference only.
    zero_transport_certified: bool = False
    stationary_stop_confirmed: bool = False
    old_authority_retired: bool = False  # Supplied fact; no ledger mutation.


@dataclass(frozen=True)
class WatchdogEvidence:
    avoidance_count: int = 0
    prior_phase: Phase = Phase.DIRECT
    committed_side: Side | None = None
    lateral_stagnation: int = 0
    passage_stagnation: int = 0
    equivalent_vetoes: int = 0
    wait_rechecks: int = 0
    wait_seconds: float = 0.


@dataclass(frozen=True)
class PolicyInput:
    reference: EvidenceReference
    target: TargetEvidence
    health: HealthEvidence
    left: SideEvidence
    right: SideEvidence
    route_obstructed: bool | None = None
    route_occupancy: int | None = None
    route_overlap_m: float | None = None
    blocker_x_m: float | None = None
    blocker_y_m: float | None = None
    preferred_side: Side | None = None
    outcome: OutcomeEvidence = field(default_factory=OutcomeEvidence)
    watchdog: WatchdogEvidence = field(default_factory=WatchdogEvidence)
    event: Event = Event.OBSERVATION
    provenance: str = 'UNSPECIFIED'

    def __post_init__(self):
        if (not isinstance(self.event, Event) or not isinstance(self.left, SideEvidence)
                or not isinstance(self.right, SideEvidence)
                or not isinstance(self.left.side, Side) or not isinstance(self.right.side, Side)
                or self.left.side != Side.LEFT or self.right.side != Side.RIGHT):
            raise ValueError('Typed event and correctly positioned sides required')
        for obj, typ in ((self.reference, EvidenceReference), (self.target, TargetEvidence),
                         (self.health, HealthEvidence), (self.left, SideEvidence),
                         (self.right, SideEvidence), (self.outcome, OutcomeEvidence),
                         (self.watchdog, WatchdogEvidence)):
            if not isinstance(obj, typ):
                raise ValueError('Immutable typed evidence required')
        if not isinstance(self.provenance, str):
            raise ValueError('Immutable provenance label required')
        for value in (self.target.trusted_standoff, self.target.direct_pursuit_admissible):
            if type(value) is not bool:
                raise ValueError('Target navigation conditions must be boolean')
        for name in ('avoidance_count', 'lateral_stagnation', 'passage_stagnation',
                     'equivalent_vetoes', 'wait_rechecks'):
            value = getattr(self.watchdog, name)
            if type(value) is not int or value < 0:
                raise ValueError('Watchdog counters must be nonnegative integers')
        if (type(self.watchdog.wait_seconds) not in (int, float)
                or not math.isfinite(self.watchdog.wait_seconds) or self.watchdog.wait_seconds < 0):
            raise ValueError('Finite nonnegative wait accounting required')
        health_values = [getattr(self.health, name) for name in self.health.__dataclass_fields__]
        health_values += [self.target.identity_tracker, self.target.visible]
        if any(not isinstance(v, EvidenceHealth) for v in health_values):
            raise ValueError('Typed health evidence required')
        if (not isinstance(self.target.alignment, Alignment)
                or not isinstance(self.target.visibility, TargetVisibility)
                or not isinstance(self.watchdog.prior_phase, Phase)):
            raise ValueError('Typed target and phase classifications required')
        for value in (self.preferred_side, self.watchdog.committed_side):
            if value is not None and not isinstance(value, Side):
                raise ValueError('Typed side required')
        for name in ('completion_confirmed', 'comparable_geometry', 'material_external_change',
                     'zero_transport_certified', 'stationary_stop_confirmed', 'old_authority_retired'):
            if type(getattr(self.outcome, name)) is not bool:
                raise ValueError('Completion/veto facts must be boolean')
        if (self.outcome.last_completed_primitive is not None
                and not isinstance(self.outcome.last_completed_primitive, Primitive)):
            raise ValueError('Typed completed primitive required')
        for value in (self.route_obstructed, self.target.trusted_standoff,
                      self.target.direct_pursuit_admissible,
                      self.outcome.recorded_meaningful_route_progress):
            if value is not None and type(value) is not bool:
                raise ValueError('Boolean or absent evidence required')
        for side in (self.left, self.right):
            for value in (side.lateral_feasible, side.pass_feasible,
                          side.protected_capsule_clear, side.target_recomputed):
                if value is not None and type(value) is not bool:
                    raise ValueError('Boolean feasibility required')
        for value in (self.route_occupancy, self.left.pass_occupancy, self.right.pass_occupancy,
                      self.watchdog.avoidance_count, self.watchdog.lateral_stagnation,
                      self.watchdog.passage_stagnation, self.watchdog.equivalent_vetoes,
                      self.watchdog.wait_rechecks, self.outcome.event_frame_stamp_ns):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Exact nonnegative counts/references required')
        for value in (self.route_overlap_m, self.blocker_x_m, self.blocker_y_m,
                      self.left.separation_m, self.right.separation_m,
                      self.left.pass_overlap_m, self.right.pass_overlap_m,
                      self.target.horizontal_error_px, self.target.tracker_quality,
                      self.outcome.observed_lateral_gain_m, self.outcome.observed_pass_gain_m,
                      self.watchdog.wait_seconds):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError('Finite observation metric or absent evidence required')

    def side(self, side):
        return self.left if side == Side.LEFT else self.right

    def record(self):
        return asdict(self)


def normalize_policy_input(*, reference, target, health, route, bypasses,
                           lateral_feasibility, outcome=None, watchdog=None,
                           event=Event.OBSERVATION, preferred_side=None, provenance='UNSPECIFIED'):
    """Adapt existing planner results without mutation or safety reimplementation.

    Missing occupancy/overlap/capsule evidence stays None, never becomes zero.
    Consumers supply already-established health/identity facts; this function
    does not certify them from a bounding box or a successful selector return.
    """
    sides = []
    for side in Side:
        plan = bypasses.get(side, {})
        y = route.get('blocking_obstacle_y_m')
        separation = None if y is None else (-y if side == Side.LEFT else y)
        recomputed = None if not plan else (
            plan.get('producer_session') == reference.producer_session
            and type(plan.get('acquisition_sequence')) is int
            and plan.get('acquisition_sequence') == reference.lidar_sequence
            and all(type(plan.get(key)) in (int, float) and math.isfinite(plan[key])
                    for key in ('bypass_target_x_m', 'bypass_target_y_m')))
        sides.append(SideEvidence(side=side, lateral_feasible=lateral_feasibility.get(side),
            separation_m=separation, pass_feasible=plan.get('bypass_forward_permitted'),
            pass_occupancy=plan.get('bypass_corridor_occupancy'),
            pass_overlap_m=plan.get('bypass_corridor_overlap_m'),
            protected_capsule_clear=(plan.get('forward_safety') or {}).get('permitted'),
            target_recomputed=recomputed))
    return PolicyInput(reference=reference, target=target, health=health,
        left=sides[0], right=sides[1], route_obstructed=route.get('route_to_marvin_obstructed'),
        route_occupancy=route.get('route_occupancy'), route_overlap_m=route.get('corridor_overlap_m'),
        blocker_x_m=route.get('blocking_obstacle_x_m'), blocker_y_m=route.get('blocking_obstacle_y_m'),
        preferred_side=preferred_side, outcome=outcome or OutcomeEvidence(),
        watchdog=watchdog or WatchdogEvidence(), event=event, provenance=provenance)
