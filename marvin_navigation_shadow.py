"""Non-authoritative shadow adapters: no runtime import, I/O, stamp ledger or dispatcher.

The existing controller remains authoritative. This module compares diagnostic
intents only. All reducer contexts/accounting are private immutable shadow data.
"""
from dataclasses import asdict, dataclass, replace
from enum import Enum

from marvin_blocked_wait import BLOCKED_WAIT_REASONS
from marvin_navigation_phases import (
    Context, Evidence, Event, Intent, Phase, Primitive, SafetyDisposition, Side,
    Visibility, Watchdogs, reduce_navigation,
)
from marvin_navigation_policy import (
    Alignment, EvidenceHealth, IntentKind, NavigationIntent, PHYSICAL_INTENTS,
    PolicyInput, TargetVisibility,
)


class ShadowMode(str, Enum):
    COMPATIBILITY = 'COMPATIBILITY'
    ARCHITECTURE = 'ARCHITECTURE'


class DifferenceCategory(str, Enum):
    EQUIVALENT = 'equivalent_behavior'
    INTENTIONAL = 'intentional_simplification'
    UNSAFE = 'potentially_unsafe_divergence'
    CONSERVATIVE = 'overly_conservative_phase_policy'
    OLD_ARTIFACT = 'old_policy_special_case_artifact'
    INSUFFICIENT = 'insufficient_evidence'


@dataclass(frozen=True)
class PhaseResult:
    context: Context
    intent: NavigationIntent
    measured_phase_progress: bool = False
    calibration_required: bool = False


def initial_context(data):
    w = data.watchdog
    side = w.committed_side
    separation = data.side(side).separation_m if side is not None else None
    if (data.event == Event.ACTION_COMPLETED and separation is not None
            and data.outcome.comparable_geometry and data.outcome.observed_lateral_gain_m is not None):
        separation -= data.outcome.observed_lateral_gain_m  # Observed difference, never a command.
    return Context(phase=w.prior_phase, committed_side=side,
        producer_session=data.reference.producer_session, avoidance_count=w.avoidance_count,
        clear_stagnation_count=w.lateral_stagnation, pass_stagnation_count=w.passage_stagnation,
        equivalent_veto_count=w.equivalent_vetoes, wait_rechecks=w.wait_rechecks,
        wait_elapsed_seconds=w.wait_seconds, last_lateral_separation=separation,
        phase_entered_at=data.reference.now, last_event_at=data.reference.now,
        detour_started_at=data.reference.now if w.prior_phase != Phase.DIRECT else None)


def evidence_block(data):
    """Reject represented unhealthy/unknown facts; does not certify safety."""
    h, t, r = data.health, data.target, data.reference
    if any(getattr(h, name) == EvidenceHealth.BAD for name in (
            'ownership', 'session', 'stopped_bridge', 'delivery_resolved')):
        return NavigationIntent(IntentKind.FAIL_CLOSED, 'represented_terminal_safety_failure')
    if any(getattr(h, name) != EvidenceHealth.OK for name in h.__dataclass_fields__):
        return NavigationIntent(IntentKind.STOP_REVERIFY, 'healthy_required_evidence_not_established')
    if (t.identity_tracker != EvidenceHealth.OK or t.visible != EvidenceHealth.OK
            or t.visibility == TargetVisibility.LOST):
        return NavigationIntent(IntentKind.STOP_REVERIFY, 'strict_target_continuity_required')
    if (not r.current or not r.producer_session or r.lidar_sequence is None or r.lidar_sequence <= 0
            or r.frame_stamp_ns is None or r.frame_stamp_ns <= 0 or data.route_obstructed is None):
        return NavigationIntent(IntentKind.STOP_REVERIFY, 'current_observation_reference_required')
    return None


def phase_policy(data: PolicyInput, config: Watchdogs, *, context=None,
                 mode=ShadowMode.COMPATIBILITY) -> PhaseResult:
    """Normalize into the unchanged pure reducer; NEVER authorizes transport."""
    if not isinstance(mode, ShadowMode):
        raise ValueError('Explicit shadow mode required')
    c = initial_context(data) if context is None else context
    blocked = evidence_block(data)
    if (blocked is not None and blocked.kind != IntentKind.FAIL_CLOSED
            and data.event == Event.ACTION_COMPLETED and data.outcome.completion_confirmed
            and all(getattr(data.health, name) == EvidenceHealth.OK for name in
                    ('ownership', 'session', 'stopped_bridge', 'delivery_resolved'))):
        # Count certified physical completion even while target re-acquisition
        # is pending. Missing navigation evidence can never create an intent.
        e = Evidence(now=data.reference.now, producer_session=data.reference.producer_session,
            event=Event.ACTION_COMPLETED, safety=SafetyDisposition.CONFIRMED_COMPLETION,
            completed_primitive=data.outcome.last_completed_primitive,
            action_stamp_ns=data.outcome.event_frame_stamp_ns or 0)
        d = reduce_navigation(c, e, replace(config, legacy_six_action_cap=mode == ShadowMode.COMPATIBILITY))
        kind = IntentKind.FAIL_CLOSED if d.intent == Intent.FAIL_CLOSED else blocked.kind
        return PhaseResult(d.context, NavigationIntent(kind, blocked.reason))
    if blocked is not None:
        if blocked.kind == IntentKind.FAIL_CLOSED:
            c = replace(c, terminal=Intent.FAIL_CLOSED)
        return PhaseResult(c, blocked)
    if c.terminal is not None:
        return PhaseResult(c, NavigationIntent(IntentKind(c.terminal.value), 'terminal_shadow_disposition'))
    o, t = data.outcome, data.target
    if data.event == Event.OBSERVATION:
        # Read confirmed controller accounting; never decrement/replenish budget.
        c = replace(c, avoidance_count=max(c.avoidance_count, data.watchdog.avoidance_count))
    if (c.phase == Phase.DIRECT and data.event == Event.OBSERVATION
            and t.alignment in {Alignment.LEFT, Alignment.RIGHT}
            and data.reference.frame_stamp_ns > c.retired_stamp_ns):
        kind = IntentKind.DIRECT_ALIGN_LEFT if t.alignment == Alignment.LEFT else IntentKind.DIRECT_ALIGN_RIGHT
        return PhaseResult(c, NavigationIntent(kind, 'normal_direct_alignment_before_detour'))
    if data.event == Event.ACTION_JIT_VETO and not all((
            o.zero_transport_certified, o.stationary_stop_confirmed, o.old_authority_retired)):
        return PhaseResult(replace(c, terminal=Intent.FAIL_CLOSED),
            NavigationIntent(IntentKind.FAIL_CLOSED, 'veto_not_certified_and_retired_by_existing_layer'))
    if data.event == Event.ACTION_COMPLETED and not o.completion_confirmed:
        return PhaseResult(replace(c, terminal=Intent.FAIL_CLOSED),
            NavigationIntent(IntentKind.FAIL_CLOSED, 'completion_not_confirmed_by_existing_layer'))
    side = c.committed_side or data.preferred_side
    if side is None:
        feasible = [s for s in Side if data.side(s).lateral_feasible is True or data.side(s).passage_feasible]
        side = feasible[0] if feasible else Side.LEFT  # Input reference only, reducer chooses commitment.
    geometry = data.side(side)
    detour_alignment_unknown = (t.visibility == TargetVisibility.CALIBRATION_REQUIRED
        and t.alignment != Alignment.CENTERED
        and (c.phase in {Phase.CLEAR_SIDE, Phase.PASS_OBSTACLE, Phase.BLOCKED_WAIT}
             or data.route_obstructed is True))
    # Unknown visibility is not silently converted to TRACK_OK motion authority.
    # Geometry can still determine a phase; strict=False ensures STOP_REVERIFY.
    visibility = (Visibility.VISIBILITY_MAINTENANCE_REQUIRED
        if t.visibility in {TargetVisibility.MAINTAIN_LEFT, TargetVisibility.MAINTAIN_RIGHT}
        else Visibility.TRACK_OK)
    safety = (SafetyDisposition.CERTIFIED_ZERO_TRANSPORT if data.event == Event.ACTION_JIT_VETO
        else SafetyDisposition.CONFIRMED_COMPLETION if data.event == Event.ACTION_COMPLETED
        else SafetyDisposition.HEALTHY)
    e = Evidence(now=data.reference.now, producer_session=data.reference.producer_session,
        strict_target=not detour_alignment_unknown, observation_stamp_ns=data.reference.frame_stamp_ns,
        geometry_current=True, route_obstructed=data.route_obstructed,
        trusted_standoff=t.trusted_standoff, direct_pursuit_admissible=t.direct_pursuit_admissible,
        centered=t.alignment == Alignment.CENTERED, visibility=visibility,
        left_lateral_feasible=data.left.lateral_feasible is True,
        right_lateral_feasible=data.right.lateral_feasible is True,
        preferred_side=data.preferred_side, passage_side=side,
        lateral_separation_m=geometry.separation_m, pass_feasible=geometry.passage_feasible,
        comparable_geometry=o.comparable_geometry,
        comparable_blocker_advance_m=o.observed_pass_gain_m,
        material_external_change=o.material_external_change,
        event=data.event, safety=safety, completed_primitive=o.last_completed_primitive,
        action_stamp_ns=o.event_frame_stamp_ns or 0)
    d = reduce_navigation(c, e, replace(config, legacy_six_action_cap=mode == ShadowMode.COMPATIBILITY))
    if d.motion_authority is not False:
        raise AssertionError('A navigation intent is never motion authorization')
    mapped = {
        Intent.PROPOSE_DIRECT_FORWARD: IntentKind.DIRECT_FORWARD,
        Intent.PROPOSE_STRAFE_LEFT: IntentKind.STRAFE_LEFT,
        Intent.PROPOSE_STRAFE_RIGHT: IntentKind.STRAFE_RIGHT,
        Intent.PROPOSE_PASS_FORWARD: IntentKind.PASS_FORWARD,
        Intent.STOP_REVERIFY: IntentKind.STOP_REVERIFY,
        Intent.WAIT: IntentKind.WAIT, Intent.ARRIVED: IntentKind.ARRIVED,
        Intent.SAFE_INCOMPLETE: IntentKind.SAFE_INCOMPLETE, Intent.FAIL_CLOSED: IntentKind.FAIL_CLOSED,
    }
    if d.intent == Intent.REQUEST_DIRECT_ALIGNMENT:
        kind = (IntentKind.DIRECT_ALIGN_LEFT if t.alignment == Alignment.LEFT else
                IntentKind.DIRECT_ALIGN_RIGHT if t.alignment == Alignment.RIGHT else IntentKind.STOP_REVERIFY)
    elif d.intent == Intent.REQUEST_VISIBILITY_TURN:
        kind = (IntentKind.VISIBILITY_TURN_LEFT if t.visibility == TargetVisibility.MAINTAIN_LEFT else
                IntentKind.VISIBILITY_TURN_RIGHT if t.visibility == TargetVisibility.MAINTAIN_RIGHT
                else IntentKind.STOP_REVERIFY)
    else:
        kind = mapped[d.intent]
    reason = 'CALIBRATION_REQUIRED: validated detour visibility classification absent' if detour_alignment_unknown else d.reason
    return PhaseResult(d.context, NavigationIntent(kind, reason), d.measured_phase_progress, detour_alignment_unknown)


def existing_policy_intent(outcome, data: PolicyInput) -> NavigationIntent:
    """Read-only mapping of an actual selected controller outcome. No reranking."""
    action = outcome.get('action_type') or outcome.get('decision')
    reason = outcome.get('reason') or 'existing_controller_outcome'
    if outcome.get('terminal_safety_failure') is True or outcome.get('delivery_uncertain') is True:
        return NavigationIntent(IntentKind.FAIL_CLOSED, reason)
    mappings = {'STRAFE_LEFT': IntentKind.STRAFE_LEFT, 'STRAFE_RIGHT': IntentKind.STRAFE_RIGHT,
        'BYPASS_FORWARD': IntentKind.PASS_FORWARD, 'FORWARD': IntentKind.DIRECT_FORWARD,
        'TURN_LEFT': IntentKind.DIRECT_ALIGN_LEFT, 'TURN_RIGHT': IntentKind.DIRECT_ALIGN_RIGHT,
        'BLOCKED_WAIT': IntentKind.WAIT, 'WAIT': IntentKind.WAIT, 'ARRIVED': IntentKind.ARRIVED,
        'SAFE_INCOMPLETE': IntentKind.SAFE_INCOMPLETE, 'FAIL_CLOSED': IntentKind.FAIL_CLOSED,
        'STOP_REVERIFY': IntentKind.STOP_REVERIFY}
    if action in mappings:
        return NavigationIntent(mappings[action], reason)
    # Preserve actual wait/terminal semantics rather than treating every None as WAIT.
    if reason == 'find_marvin_local_avoidance_exhausted':
        return NavigationIntent(IntentKind.SAFE_INCOMPLETE, reason)
    if reason in BLOCKED_WAIT_REASONS:
        return NavigationIntent(IntentKind.WAIT, reason)
    return NavigationIntent(IntentKind.STOP_REVERIFY, 'insufficient_existing_outcome: '+reason)




@dataclass(frozen=True)
class Comparison:
    step: str
    normalized_input: PolicyInput
    old_intent: NavigationIntent
    phase_result: PhaseResult
    category: DifferenceCategory
    expected_divergence: bool
    reason: str
    safety_significance: str
    provenance: str

    def record(self):
        d = asdict(self)
        d['same'] = self.old_intent.kind == self.phase_result.intent.kind
        return d


def compare(step, data, old, phase, *, provenance='CURRENT_BASELINE'):
    same = old.kind == phase.intent.kind
    physical = phase.intent.kind in PHYSICAL_INTENTS
    invalid = evidence_block(data)
    if physical and (invalid is not None or phase.calibration_required):
        category, expected, reason = DifferenceCategory.UNSAFE, False, 'Motion-shaped proposal under unhealthy/uncalibrated input'
    elif phase.intent.kind == IntentKind.PASS_FORWARD and not data.side(phase.context.committed_side).passage_feasible:
        category, expected, reason = DifferenceCategory.UNSAFE, False, 'Pass proposal without independently supplied clear corridor'
    elif same:
        category, expected, reason = DifferenceCategory.EQUIVALENT, False, 'Common navigation intents agree'
    elif phase.calibration_required:
        category, expected, reason = DifferenceCategory.INSUFFICIENT, False, 'CALIBRATION_REQUIRED; fewer turns not established as correct'
    elif invalid is not None or old.reason.startswith('insufficient_existing_outcome'):
        category, expected, reason = DifferenceCategory.INSUFFICIENT, False, 'Retained input/outcome lacks required comparable evidence'
    elif (old.kind == IntentKind.PASS_FORWARD and phase.intent.kind in {IntentKind.STRAFE_LEFT, IntentKind.STRAFE_RIGHT}
            and data.side(phase.context.committed_side).separation_m is not None
            and data.side(phase.context.committed_side).separation_m >= .15):
        category, expected, reason = DifferenceCategory.CONSERVATIVE, True, 'Provisional entry hysteresis retains lateral phase'
    elif (old.kind == IntentKind.WAIT and phase.intent.kind in {IntentKind.STRAFE_LEFT, IntentKind.STRAFE_RIGHT}
            and phase.context.phase == Phase.CLEAR_SIDE):
        category, expected, reason = DifferenceCategory.INTENTIONAL, True, 'Fresh safe same-side repair replaces unconditional waiting'
    elif (old.kind == IntentKind.SAFE_INCOMPLETE and physical
            and data.watchdog.avoidance_count + int(data.event == Event.ACTION_COMPLETED
                and data.outcome.completion_confirmed
                and data.outcome.last_completed_primitive in {Primitive.STRAFE_LEFT, Primitive.STRAFE_RIGHT, Primitive.PASS_FORWARD}) >= 6):
        category, expected, reason = DifferenceCategory.INTENTIONAL, True, 'Architecture-only legacy-cap comparison; replacement watchdog calibration required'
    elif old.kind in {IntentKind.WAIT, IntentKind.STRAFE_LEFT, IntentKind.STRAFE_RIGHT, IntentKind.SAFE_INCOMPLETE} and (
            phase.intent.kind == IntentKind.PASS_FORWARD or data.event == Event.ACTION_JIT_VETO):
        artifact = any(word in old.reason for word in ('recovery', 'no_progress', 'suppression'))
        category = DifferenceCategory.OLD_ARTIFACT if artifact else DifferenceCategory.INTENTIONAL
        expected, reason = True, 'Explicit phase feasibility replaces history-derived ranking/wait special case'
    elif old.kind in {IntentKind.DIRECT_ALIGN_LEFT, IntentKind.DIRECT_ALIGN_RIGHT} and physical:
        category, expected, reason = DifferenceCategory.INTENTIONAL, True, 'Validated visibility permits phase-aware alignment deferral'
    elif phase.context.phase == Phase.REJOIN or old.kind == IntentKind.STOP_REVERIFY:
        category, expected, reason = DifferenceCategory.INTENTIONAL, True, 'Fresh rejoin/action evidence boundary'
    else:
        category, expected, reason = DifferenceCategory.INSUFFICIENT, False, 'Unexplained difference needs scenario/evidence review'
    significance = ('INVESTIGATE; no authority or transport exists' if category == DifferenceCategory.UNSAFE
        else 'Observation-only comparison; every future physical action still needs universal admission')
    return Comparison(str(step), data, old, phase, category, expected, reason, significance, provenance)


@dataclass(frozen=True)
class Accounting:
    last_at: float | None = None
    last_phase: Phase | None = None
    detour_elapsed_seconds: float = 0.
    clear_side_completed_actions: int = 0
    pass_obstacle_completed_actions: int = 0
    phase_transition_count: int = 0
    blocked_wait_total_seconds: float = 0.
    lateral_stagnation: int = 0
    passage_stagnation: int = 0
    equivalent_veto_count: int = 0
    last_completion_reference_ns: int = 0
    unproven_advancement_completions: int = 0  # Unknown/comparability absent, not measured immobility.


def account(previous: Accounting, data: PolicyInput, result: PhaseResult):
    delta = 0. if previous.last_at is None else data.reference.now-previous.last_at
    if delta < 0:
        raise ValueError('Shadow diagnostic clock moved backwards')
    detour = previous.last_phase in {Phase.CLEAR_SIDE, Phase.PASS_OBSTACLE, Phase.REJOIN, Phase.BLOCKED_WAIT}
    o = data.outcome
    completed = (result.intent.kind != IntentKind.FAIL_CLOSED
        and data.event == Event.ACTION_COMPLETED and o.completion_confirmed
        and o.event_frame_stamp_ns is not None and o.event_frame_stamp_ns > previous.last_completion_reference_ns)
    lateral = completed and o.last_completed_primitive in {Primitive.STRAFE_LEFT, Primitive.STRAFE_RIGHT}
    passing = completed and o.last_completed_primitive == Primitive.PASS_FORWARD
    c = result.context
    return Accounting(last_at=data.reference.now, last_phase=c.phase,
        detour_elapsed_seconds=previous.detour_elapsed_seconds+(delta if detour else 0.),
        clear_side_completed_actions=previous.clear_side_completed_actions+int(lateral),
        pass_obstacle_completed_actions=previous.pass_obstacle_completed_actions+int(passing),
        phase_transition_count=previous.phase_transition_count+int(previous.last_phase is not None and previous.last_phase != c.phase),
        blocked_wait_total_seconds=previous.blocked_wait_total_seconds+(delta if previous.last_phase == Phase.BLOCKED_WAIT else 0.),
        lateral_stagnation=c.clear_stagnation_count, passage_stagnation=c.pass_stagnation_count,
        equivalent_veto_count=c.equivalent_veto_count,
        last_completion_reference_ns=o.event_frame_stamp_ns if completed else previous.last_completion_reference_ns,
        unproven_advancement_completions=previous.unproven_advancement_completions
            + int((lateral or passing) and not o.comparable_geometry))
