"""Offline navigation-policy prototype; deliberately not imported by runtime.

Navigation intent is not motion authority. Feasibility inputs are summaries from
existing planners/admission checks, not cached permissions. An eventual runner
must still obtain strict identity/tracker evidence, consume a new exact stamp,
run ordinary JIT, dispatch bounded motion, STOP, verify Bridge zero, and obtain
newer LiDAR/camera evidence. This module performs none of those operations.
"""
from dataclasses import dataclass, replace
from enum import Enum
import math

from marvin_local_bypass import MIN_BYPASS_LATERAL_SEPARATION_M
from marvin_local_obstacle_avoidance import MAX_LOCAL_AVOIDANCE_ACTIONS
from marvin_route_obstruction import MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M


class Phase(str, Enum):
    DIRECT = 'DIRECT'
    CLEAR_SIDE = 'CLEAR_SIDE'
    PASS_OBSTACLE = 'PASS_OBSTACLE'
    REJOIN = 'REJOIN'
    BLOCKED_WAIT = 'BLOCKED_WAIT'
    ARRIVED = 'ARRIVED'


class Side(str, Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'


class Intent(str, Enum):
    PROPOSE_DIRECT_FORWARD = 'PROPOSE_DIRECT_FORWARD'
    PROPOSE_STRAFE_LEFT = 'PROPOSE_STRAFE_LEFT'
    PROPOSE_STRAFE_RIGHT = 'PROPOSE_STRAFE_RIGHT'
    PROPOSE_PASS_FORWARD = 'PROPOSE_PASS_FORWARD'
    REQUEST_DIRECT_ALIGNMENT = 'REQUEST_DIRECT_ALIGNMENT'
    REQUEST_VISIBILITY_TURN = 'REQUEST_VISIBILITY_TURN'
    WAIT = 'WAIT'
    STOP_REVERIFY = 'STOP_REVERIFY'
    ARRIVED = 'ARRIVED'
    SAFE_INCOMPLETE = 'SAFE_INCOMPLETE'
    FAIL_CLOSED = 'FAIL_CLOSED'


class Visibility(str, Enum):
    TRACK_OK = 'TRACK_OK'
    VISIBILITY_MAINTENANCE_REQUIRED = 'VISIBILITY_MAINTENANCE_REQUIRED'
    TARGET_LOST = 'TARGET_LOST'


class Event(str, Enum):
    OBSERVATION = 'OBSERVATION'
    ACTION_COMPLETED = 'ACTION_COMPLETED'
    ACTION_JIT_VETO = 'ACTION_JIT_VETO'
    WAIT_RECHECK = 'WAIT_RECHECK'


class Primitive(str, Enum):
    DIRECT_FORWARD = 'FORWARD'
    STRAFE_LEFT = 'STRAFE_LEFT'
    STRAFE_RIGHT = 'STRAFE_RIGHT'
    PASS_FORWARD = 'BYPASS_FORWARD'
    TURN = 'TURN'


class SafetyDisposition(str, Enum):
    """Supplied by the existing safety layer; never certified by this reducer."""
    HEALTHY = 'HEALTHY'
    CERTIFIED_ZERO_TRANSPORT = 'CERTIFIED_ZERO_TRANSPORT'
    CONFIRMED_COMPLETION = 'CONFIRMED_COMPLETION'
    DELIVERY_UNCERTAIN = 'DELIVERY_UNCERTAIN'
    STOP_FAILURE = 'STOP_FAILURE'
    BRIDGE_FAILURE = 'BRIDGE_FAILURE'
    SESSION_FAILURE = 'SESSION_FAILURE'
    OWNERSHIP_LOST = 'OWNERSHIP_LOST'
    MALFORMED = 'MALFORMED'


@dataclass(frozen=True)
class Watchdogs:
    # Entry calibration is intentionally required. 0.16 m used in tests is
    # provisional, not a proposed production calibration. JIT stays >=0.15 m.
    pass_entry_separation_m: float
    legacy_six_action_cap: bool = True
    max_detour_seconds: float | None = None  # Future calibration required.
    max_clear_stagnation: int | None = None  # Future calibration required.
    max_pass_stagnation: int = 3  # Conservative unmeasured-passage allowance.
    max_equivalent_vetoes: int = 2
    max_wait_rechecks: int = 12
    max_wait_seconds: float = 30.
    wait_intervals_seconds: tuple = (.5, 1., 2.)  # Reference, not a wait loop.

    def __post_init__(self):
        if (not math.isfinite(self.pass_entry_separation_m)
                or self.pass_entry_separation_m <= MIN_BYPASS_LATERAL_SEPARATION_M):
            raise ValueError('A provisional entry margin above the hard floor is required')
        for value in (self.max_detour_seconds, self.max_wait_seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Watchdog times must be positive and finite')
        for value in (self.max_clear_stagnation, self.max_pass_stagnation,
                      self.max_equivalent_vetoes, self.max_wait_rechecks):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError('Watchdog counts must be positive integers')


@dataclass(frozen=True)
class Context:
    phase: Phase = Phase.DIRECT
    committed_side: Side | None = None
    producer_session: str | None = None
    phase_entered_at: float = 0.
    detour_started_at: float | None = None
    phase_action_count: int = 0
    avoidance_count: int = 0
    clear_stagnation_count: int = 0
    pass_stagnation_count: int = 0
    last_lateral_separation: float | None = None
    equivalent_veto_count: int = 0
    wait_rechecks: int = 0
    wait_elapsed_seconds: float = 0.
    # Retirement frontier, NOT reusable motion authority. Never stores a target,
    # JIT permission, previous_selection, or source frame for later dispatch.
    retired_stamp_ns: int = 0
    last_completion_stamp_ns: int = 0
    terminal: Intent | None = None
    last_event_at: float = 0.

    def __post_init__(self):
        if (not isinstance(self.phase, Phase)
                or self.committed_side is not None and not isinstance(self.committed_side, Side)
                or self.terminal not in (None, Intent.SAFE_INCOMPLETE, Intent.FAIL_CLOSED)):
            raise ValueError('Invalid phase context')
        for name in ('phase_action_count', 'avoidance_count', 'clear_stagnation_count',
                     'pass_stagnation_count', 'equivalent_veto_count', 'wait_rechecks',
                     'retired_stamp_ns', 'last_completion_stamp_ns'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('Counters and retirement stamps must be nonnegative integers')
        for value in (self.phase_entered_at, self.detour_started_at,
                      self.wait_elapsed_seconds, self.last_event_at):
            if value is not None and (type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0):
                raise ValueError('Context times must be finite and nonnegative')


@dataclass(frozen=True)
class Evidence:
    """Typed policy inputs; no raw sensor/transport safety implementation here.

    pass_feasible denotes existing fresh full-capsule/corridor feasibility for
    the committed side, not permission to transport. Side separation and
    comparable_blocker_advance are observed geometry, never command integration.
    visibility classification requires a future validated image-margin adapter.
    Healthy safety status includes current owner/session/sectors/Bridge/STOP.
    """
    now: float
    producer_session: str
    strict_target: bool = False
    observation_stamp_ns: int = 0
    geometry_current: bool = False
    route_obstructed: bool | None = None
    trusted_standoff: bool = False
    direct_pursuit_admissible: bool = False
    centered: bool = True
    visibility: Visibility = Visibility.TRACK_OK
    left_lateral_feasible: bool = False
    right_lateral_feasible: bool = False
    preferred_side: Side | None = None
    passage_side: Side | None = None
    lateral_separation_m: float | None = None
    pass_feasible: bool = False
    comparable_geometry: bool = False
    comparable_blocker_advance_m: float | None = None
    material_external_change: bool = False
    event: Event = Event.OBSERVATION
    safety: SafetyDisposition = SafetyDisposition.HEALTHY
    completed_primitive: Primitive | None = None
    action_stamp_ns: int = 0


@dataclass(frozen=True)
class Decision:
    context: Context
    intent: Intent
    reason: str
    measured_phase_progress: bool = False
    motion_authority: bool = False


def reduce_navigation(context: Context, evidence: Evidence, config: Watchdogs) -> Decision:
    """Pure deterministic reducer. All proposed motion requires later admission."""
    c, e = context, evidence

    def result(intent, reason, progress=False):
        return Decision(c, intent, reason, progress)

    def terminal(intent, reason):
        nonlocal c
        c = replace(c, terminal=intent)
        return result(intent, reason)

    def enter(phase):
        nonlocal c
        if c.phase != phase:
            c = replace(c, phase=phase, phase_entered_at=e.now,
                        phase_action_count=0)
        if phase == Phase.REJOIN:
            # Keep cumulative detour time until fresh direct pursuit is restored.
            c = replace(c, committed_side=None, last_lateral_separation=None,
                        clear_stagnation_count=0, pass_stagnation_count=0,
                        retired_stamp_ns=max(c.retired_stamp_ns, e.observation_stamp_ns))

    if c.terminal is not None:
        return result(c.terminal, 'terminal_disposition_retained')
    booleans = (e.strict_target, e.geometry_current, e.trusted_standoff,
                e.direct_pursuit_admissible, e.centered, e.left_lateral_feasible,
                e.right_lateral_feasible, e.pass_feasible, e.comparable_geometry,
                e.material_external_change)
    if (any(type(value) is not bool for value in booleans)
            or type(e.observation_stamp_ns) is not int or e.observation_stamp_ns < 0
            or any(value is not None and not isinstance(value, Side)
                   for value in (e.preferred_side, e.passage_side))
            or not isinstance(e.safety, SafetyDisposition) or not isinstance(e.event, Event)
            or not isinstance(e.visibility, Visibility)
            or e.safety not in {SafetyDisposition.HEALTHY,
                SafetyDisposition.CERTIFIED_ZERO_TRANSPORT,
                SafetyDisposition.CONFIRMED_COMPLETION}
            or not isinstance(c.phase, Phase)
            or c.committed_side is not None and not isinstance(c.committed_side, Side)
            or not e.producer_session or not isinstance(e.producer_session, str)
            or type(e.now) not in (int, float) or not math.isfinite(e.now) or e.now < c.last_event_at
            or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                   for v in (e.lateral_separation_m, e.comparable_blocker_advance_m))
            or e.route_obstructed is not None and type(e.route_obstructed) is not bool):
        return terminal(Intent.FAIL_CLOSED, 'invalid_or_failed_safety_evidence')
    if c.producer_session is not None and c.producer_session != e.producer_session:
        return terminal(Intent.FAIL_CLOSED, 'producer_session_changed')
    c = replace(c, producer_session=e.producer_session, last_event_at=e.now,
                wait_elapsed_seconds=c.wait_elapsed_seconds + (e.now-c.last_event_at
                    if c.phase == Phase.BLOCKED_WAIT else 0.))
    if ((e.event == Event.ACTION_JIT_VETO and e.safety != SafetyDisposition.CERTIFIED_ZERO_TRANSPORT)
            or (e.event == Event.ACTION_COMPLETED and e.safety != SafetyDisposition.CONFIRMED_COMPLETION)):
        return terminal(Intent.FAIL_CLOSED, 'event_requires_external_safety_certificate')
    progress = False
    if e.event in {Event.ACTION_COMPLETED, Event.ACTION_JIT_VETO}:
        if type(e.action_stamp_ns) is not int or e.action_stamp_ns <= 0:
            return terminal(Intent.FAIL_CLOSED, 'exact_retired_action_stamp_required')
        if (e.event == Event.ACTION_COMPLETED and e.action_stamp_ns <= c.retired_stamp_ns
                and e.action_stamp_ns > c.last_completion_stamp_ns):
            return terminal(Intent.FAIL_CLOSED, 'retired_veto_authority_cannot_complete')
        c = replace(c, retired_stamp_ns=max(c.retired_stamp_ns, e.action_stamp_ns))
    if e.event == Event.ACTION_COMPLETED:
        if e.action_stamp_ns <= c.last_completion_stamp_ns:
            return result(Intent.STOP_REVERIFY, 'completion_already_counted')
        if not isinstance(e.completed_primitive, Primitive):
            return terminal(Intent.FAIL_CLOSED, 'completed_primitive_required')
        lateral = e.completed_primitive in {Primitive.STRAFE_LEFT, Primitive.STRAFE_RIGHT}
        passing = e.completed_primitive == Primitive.PASS_FORWARD
        if (config.legacy_six_action_cap and (lateral or passing)
                and c.avoidance_count >= MAX_LOCAL_AVOIDANCE_ACTIONS):
            return terminal(Intent.FAIL_CLOSED, 'unexpected_completion_exceeds_legacy_bound')
        if ((lateral and c.phase != Phase.CLEAR_SIDE)
                or (passing and c.phase != Phase.PASS_OBSTACLE)
                or (e.completed_primitive == Primitive.DIRECT_FORWARD and c.phase != Phase.DIRECT)):
            return terminal(Intent.FAIL_CLOSED, 'completion_incompatible_with_phase')
        if lateral and ((e.completed_primitive == Primitive.STRAFE_LEFT and c.committed_side != Side.LEFT)
                        or (e.completed_primitive == Primitive.STRAFE_RIGHT and c.committed_side != Side.RIGHT)):
            return terminal(Intent.FAIL_CLOSED, 'completed_side_does_not_match_commitment')
        c = replace(c, last_completion_stamp_ns=e.action_stamp_ns,
                    avoidance_count=c.avoidance_count + int(lateral or passing),
                    phase_action_count=c.phase_action_count + int(lateral or passing))
        if lateral:
            progress = (e.comparable_geometry and e.lateral_separation_m is not None
                        and c.last_lateral_separation is not None
                        and e.lateral_separation_m - c.last_lateral_separation
                        >= MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M)
            if (e.comparable_geometry and c.last_lateral_separation is not None
                    and e.lateral_separation_m is not None
                    and c.last_lateral_separation < config.pass_entry_separation_m
                    <= e.lateral_separation_m):
                progress = True
            c = replace(c, clear_stagnation_count=0 if progress else c.clear_stagnation_count+1)
        if passing:
            progress = (e.comparable_geometry and e.comparable_blocker_advance_m is not None
                        and e.comparable_blocker_advance_m >= MIN_CORRIDOR_OVERLAP_IMPROVEMENT_M)
            c = replace(c, pass_stagnation_count=0 if progress else c.pass_stagnation_count+1)
        # Rotations never earn traversal progress or reset stagnation/veto credit.
        if progress:
            c = replace(c, equivalent_veto_count=0)
    if e.material_external_change and e.geometry_current:
        c = replace(c, equivalent_veto_count=0)
    if e.event == Event.ACTION_JIT_VETO:
        c = replace(c, equivalent_veto_count=c.equivalent_veto_count+1)
    if (c.detour_started_at is not None and config.max_detour_seconds is not None
            and e.now-c.detour_started_at >= config.max_detour_seconds):
        return terminal(Intent.SAFE_INCOMPLETE, 'cumulative_detour_time_bound')
    if not e.geometry_current or e.route_obstructed is None:
        return result(Intent.STOP_REVERIFY, 'current_navigation_geometry_required')
    if e.visibility == Visibility.TARGET_LOST:
        return result(Intent.STOP_REVERIFY, 'target_lost_reverify')
    if c.phase == Phase.ARRIVED:
        return result(Intent.ARRIVED, 'trusted_arrival_retained')
    if e.trusted_standoff and e.strict_target:
        enter(Phase.ARRIVED)
        return result(Intent.ARRIVED, 'existing_trusted_standoff')
    if c.phase != Phase.DIRECT and not e.route_obstructed:
        if c.phase != Phase.REJOIN:
            enter(Phase.REJOIN)
            return result(Intent.STOP_REVERIFY, 'route_clear_discard_detour', True)
        if (e.direct_pursuit_admissible and e.strict_target
                and type(e.observation_stamp_ns) is int and e.observation_stamp_ns > c.retired_stamp_ns
                and e.event != Event.ACTION_JIT_VETO):
            if not e.centered:
                return result(Intent.REQUEST_DIRECT_ALIGNMENT, 'rejoin_centering')
            enter(Phase.DIRECT)
            c = replace(c, detour_started_at=None, equivalent_veto_count=0)
        else:
            return result(Intent.STOP_REVERIFY, 'fresh_direct_pursuit_required')
    if e.route_obstructed:
        if config.legacy_six_action_cap and c.avoidance_count >= MAX_LOCAL_AVOIDANCE_ACTIONS:
            return terminal(Intent.SAFE_INCOMPLETE, 'legacy_six_action_compatibility_bound')
        if ((config.max_clear_stagnation is not None and c.clear_stagnation_count >= config.max_clear_stagnation)
                or c.pass_stagnation_count >= config.max_pass_stagnation):
            return terminal(Intent.SAFE_INCOMPLETE, 'phase_stagnation_bound')
        if c.detour_started_at is None:
            c = replace(c, detour_started_at=e.now)
        if c.committed_side is None:
            available = [side for side, feasible in ((Side.LEFT, e.left_lateral_feasible),
                         (Side.RIGHT, e.right_lateral_feasible)) if feasible]
            if (e.pass_feasible and e.passage_side is not None
                    and e.lateral_separation_m is not None
                    and e.lateral_separation_m >= config.pass_entry_separation_m
                    and e.passage_side not in available):
                available.append(e.passage_side)
            side = e.preferred_side if e.preferred_side in available else (available[0] if available else None)
            c = replace(c, committed_side=side)
        repair = ((c.committed_side == Side.LEFT and e.left_lateral_feasible)
                  or (c.committed_side == Side.RIGHT and e.right_lateral_feasible))
        entry_floor = (MIN_BYPASS_LATERAL_SEPARATION_M if c.phase == Phase.PASS_OBSTACLE
                       else config.pass_entry_separation_m)
        passage = (c.committed_side is not None and e.passage_side == c.committed_side
                   and e.pass_feasible and e.lateral_separation_m is not None
                   and e.lateral_separation_m >= entry_floor)
        if c.equivalent_veto_count >= config.max_equivalent_vetoes:
            enter(Phase.BLOCKED_WAIT)
        elif passage:
            enter(Phase.PASS_OBSTACLE)
        elif repair:
            enter(Phase.CLEAR_SIDE)
        else:
            enter(Phase.BLOCKED_WAIT)
        if e.lateral_separation_m is not None and e.passage_side == c.committed_side:
            c = replace(c, last_lateral_separation=e.lateral_separation_m)
    if c.phase == Phase.BLOCKED_WAIT:
        if e.event == Event.WAIT_RECHECK:
            c = replace(c, wait_rechecks=c.wait_rechecks+1)
        if (c.wait_rechecks >= config.max_wait_rechecks
                or c.wait_elapsed_seconds >= config.max_wait_seconds):
            return terminal(Intent.SAFE_INCOMPLETE, 'blocked_wait_bound')
        return result(Intent.WAIT, 'equivalent_veto_bound_wait_for_material_change'
                      if c.equivalent_veto_count >= config.max_equivalent_vetoes
                      else 'no_safe_phase_admissible_action')
    if e.event == Event.ACTION_JIT_VETO:
        return result(Intent.STOP_REVERIFY, 'veto_authority_retired_new_observation_required')
    if (not e.strict_target or type(e.observation_stamp_ns) is not int
            or e.observation_stamp_ns <= c.retired_stamp_ns):
        return result(Intent.STOP_REVERIFY, 'new_strict_action_observation_required')
    if c.phase in {Phase.CLEAR_SIDE, Phase.PASS_OBSTACLE}:
        if e.visibility == Visibility.VISIBILITY_MAINTENANCE_REQUIRED:
            return result(Intent.REQUEST_VISIBILITY_TURN, 'validated_visibility_input', progress)
        if c.phase == Phase.CLEAR_SIDE:
            intent = Intent.PROPOSE_STRAFE_LEFT if c.committed_side == Side.LEFT else Intent.PROPOSE_STRAFE_RIGHT
            return result(intent, 'establish_or_restore_lateral_separation', progress)
        return result(Intent.PROPOSE_PASS_FORWARD, 'pass_corridor_feasible_not_measured_success', progress)
    if not e.centered:
        return result(Intent.REQUEST_DIRECT_ALIGNMENT, 'normal_direct_centering')
    if not e.direct_pursuit_admissible:
        return result(Intent.STOP_REVERIFY, 'direct_pursuit_evidence_required')
    return result(Intent.PROPOSE_DIRECT_FORWARD, 'direct_route_clear', progress)


def context_record(context: Context) -> dict:
    """Serializable policy state only; carries no reusable action authority."""
    from dataclasses import asdict
    return asdict(context)


def restore_context(record: dict) -> Context:
    data = dict(record)
    data['phase'] = Phase(data['phase'])
    if data.get('committed_side') is not None:
        data['committed_side'] = Side(data['committed_side'])
    if data.get('terminal') is not None:
        data['terminal'] = Intent(data['terminal'])
    return Context(**data)
