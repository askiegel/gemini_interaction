"""Mission-owned obstacle phases; proposals still require the native JIT executor.

No sensor getter, transport, shadow, replay or calibration dependency belongs
here. Previous action selection is diagnostic history, never phase ownership.
"""
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from marvin_detour_watchdog import DetourWatchdog, MATERIAL_GEOMETRY_CHANGE_M, finite

from marvin_local_obstacle_avoidance import (
    _select_marvin_escape_action, rank_marvin_escape_options,
    LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
    MAX_LOCAL_AVOIDANCE_ACTIONS,
)
from marvin_local_bypass import plan_local_bypass
from marvin_route_obstruction import evaluate_marvin_route
from local_motion_safety_envelope import OCTANT_SECTORS, MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR


class Phase(str, Enum):
    DIRECT = 'DIRECT'
    CLEAR_SIDE = 'CLEAR_SIDE'
    PASS_OBSTACLE = 'PASS_OBSTACLE'
    REJOIN = 'REJOIN'
    BLOCKED_WAIT = 'BLOCKED_WAIT'
    ARRIVED = 'ARRIVED'


@dataclass(frozen=True)
class Frontier:
    producer_session: str
    acquisition_sequence: int
    source_frame_stamp_ns: int

    def __post_init__(self):
        if (not isinstance(self.producer_session, str) or not self.producer_session
                or type(self.acquisition_sequence) is not int or self.acquisition_sequence < 0
                or type(self.source_frame_stamp_ns) is not int or self.source_frame_stamp_ns <= 0):
            raise ValueError('Exact native evidence frontier required')


@dataclass(frozen=True)
class DetourContext:
    phase: Phase = Phase.DIRECT
    committed_side: str | None = None
    phase_entry_evidence: Frontier | None = None
    phase_action_count: int = 0
    last_completion_evidence: Frontier | None = None
    retired_source_stamp_ns: int = 0
    watchdog: DetourWatchdog = field(default_factory=DetourWatchdog)

    def __post_init__(self):
        if (not isinstance(self.watchdog, DetourWatchdog)
                or not isinstance(self.phase, Phase) or self.committed_side not in {None, 'LEFT', 'RIGHT'}
                or type(self.phase_action_count) is not int or self.phase_action_count < 0
                or type(self.retired_source_stamp_ns) is not int or self.retired_source_stamp_ns < 0
                or any(e is not None and not isinstance(e, Frontier)
                    for e in (self.phase_entry_evidence, self.last_completion_evidence))):
            raise ValueError('Invalid mission detour context')

    def enter(self, phase, evidence, *, side=None, now=None):
        previous = self.phase_entry_evidence or self.last_completion_evidence
        if previous is not None and previous.producer_session != evidence.producer_session:
            raise ValueError('Detour producer session changed')
        if previous is not None and evidence.acquisition_sequence < previous.acquisition_sequence:
            raise ValueError('Detour evidence sequence regressed')
        committed = self.committed_side if side is None else side
        if committed not in {None, 'LEFT', 'RIGHT'}:
            raise ValueError('Invalid committed side')
        if phase in {Phase.REJOIN, Phase.DIRECT, Phase.ARRIVED}:
            committed = None
        return replace(self, phase=phase, committed_side=committed,
            watchdog=self.watchdog.enter(phase.value, now) if now is not None else self.watchdog,
            phase_entry_evidence=evidence if phase != self.phase else self.phase_entry_evidence,
            phase_action_count=0 if phase != self.phase else self.phase_action_count)

    def retire(self, evidence):
        return replace(self, retired_source_stamp_ns=max(
            self.retired_source_stamp_ns, evidence.source_frame_stamp_ns))

    def completed(self, evidence, *, traversal):
        if evidence.source_frame_stamp_ns <= self.retired_source_stamp_ns:
            raise ValueError('Retired completion authority')
        return replace(self.retire(evidence), last_completion_evidence=evidence,
            phase_action_count=self.phase_action_count + int(traversal))

    def record(self):
        return dict(asdict(self), watchdog=self.watchdog.record())


def phase_from_admission(*, route_obstructed, pass_safe, repair_safe):
    """Ownership only. Unknown admission cannot masquerade as healthy WAIT."""
    if route_obstructed is False:
        return Phase.REJOIN
    if route_obstructed is not True:
        return None
    if pass_safe is True:
        return Phase.PASS_OBSTACLE
    if repair_safe is True:
        return Phase.CLEAR_SIDE
    if pass_safe is False and repair_safe is False:
        return Phase.BLOCKED_WAIT
    return None


def plan_phase_action(state, association, *, expected_session, allow_strafe,
                      committed_side=None,
                      entering_detour=False, lateral_progress=None,
                      strafe_duration_limit=LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
                      remaining_avoidance_actions=None):
    """Recompute WHAT from one scan; never authorize transport or reuse a grant.

    Native primitive geometry, ranking and bypass admission are retained. Only
    high-level ownership changes. Completed lateral outcomes may suppress an
    ineffective strafe; predicted gain never credits progress. Safe passage
    still wins without a direct-route progress requirement.
    """
    result = _select_marvin_escape_action(state, association,
        expected_session=expected_session, allow_strafe=allow_strafe,
        strafe_duration_limit=strafe_duration_limit)
    result.update(action_type=None, direction=None, phase=None,
        phase_owned=True, committed_side=committed_side)
    route = result['route']
    if not route.get('valid'):
        return result
    if not route['route_to_marvin_obstructed']:
        return dict(result, phase=Phase.REJOIN.value, reason='direct_path_restored')
    # Explicit legacy/proof budgets remain compatible. Normal phase navigation
    # has a mission watchdog, never a global six-action planning/JIT allowance.
    if remaining_avoidance_actions is not None and (
            type(remaining_avoidance_actions) is not int or remaining_avoidance_actions <= 0):
        return dict(result, reason='find_marvin_local_avoidance_exhausted')
    if committed_side not in {None, 'LEFT', 'RIGHT'}:
        return dict(result, reason='find_marvin_local_avoidance_history_invalid')
    # Unhealthy required coverage is fail-closed, never healthy BLOCKED_WAIT.
    sectors = state['local_motion_geometry'].get('sectors') or {}
    if any(type((sectors.get(name) or {}).get('valid_sample_count')) is not int
            or (sectors.get(name) or {}).get('valid_sample_count', 0)
                < MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
            for name, _, _ in OCTANT_SECTORS):
        return dict(result, reason='insufficient_lidar_samples')
    side = committed_side
    if side is None:
        # Reuse native primitive ranking, restricted to safe lateral options.
        eligible = [kind for kind, option in result['options'].items()
            if kind.startswith('STRAFE') and option.get('permitted') is True
            and option.get('improves_route') is True]
        chosen = rank_marvin_escape_options(result['options'], eligible)
        side = chosen.split('_')[1] if chosen else None
        if side is None:
            # An already established safe corridor can be used even if no
            # lateral step is available. Keep native blocker-side semantics.
            y = route.get('blocking_obstacle_y_m')
            if type(y) in (int, float):
                side = 'LEFT' if y < 0 else 'RIGHT'
    def admission(candidate_side):
        lateral = result['options'].get('STRAFE_' + candidate_side, {}) if candidate_side else {}
        bypass = plan_local_bypass(state, association, route,
            expected_session=expected_session, side=candidate_side)
        # Only the first stopped same-session outcome of a completed strafe
        # establishes insufficiency. Vetoes, attempts and predictions cannot.
        outcome = lateral_progress or {}
        insufficient = (outcome.get('completed_strafe') is True
            and outcome.get('producer_session') == expected_session
            and outcome.get('side') == candidate_side
            and type(outcome.get('acquisition_sequence')) is int
            and 0 < outcome['acquisition_sequence'] <= state['acquisition_sequence']
            and outcome.get('meaningful_progress') is False)
        # A bounded strafe may improve committed blocker separation even when
        # the target-route metric stays neutral (e.g. a foreground return).
        predicted = lateral.get('predicted_route') or {}
        old_y, new_y = route.get('blocking_obstacle_y_m'), predicted.get('blocking_obstacle_y_m')
        old_x, new_x = route.get('blocking_obstacle_x_m'), predicted.get('blocking_obstacle_x_m')
        sign = -1 if candidate_side == 'LEFT' else 1
        clearance_gain = (sign * (new_y - old_y)
            if all(finite(v) for v in (old_y, new_y, old_x, new_x))
                and abs(new_x - old_x) < 1e-9 else None)
        predicted_useful = (lateral.get('improves_route') is True
            or finite(clearance_gain) and clearance_gain >= MATERIAL_GEOMETRY_CHANGE_M)
        useful = (lateral.get('permitted') is True and predicted_useful and not insufficient)
        return lateral, bypass, useful, insufficient

    lateral, bypass, useful, insufficient = admission(side)
    initial_lateral = entering_detour and useful
    # Retain the committed corridor whenever passage or useful same-side
    # clearance is available. A better-ranked opposite side is not a reason
    # to reverse. Fresh unsafe admission or a measured ineffective strafe is.
    side_change_reason = None
    if (side and bypass.get('bypass_forward_permitted') is not True and not useful
            and (lateral.get('permitted') is not True or insufficient)):
        alternative = 'RIGHT' if side == 'LEFT' else 'LEFT'
        alt_lateral, alt_bypass, alt_useful, alt_insufficient = admission(alternative)
        prior_clearance = (lateral_progress or {}).get('alternative_clearance_m')
        clearance = result[alternative.lower() + '_clearance_m']
        alternative_gain = (clearance - prior_clearance
            if finite(clearance) and finite(prior_clearance) else None)
        # Ineffectiveness alone may justify a same-side reorientation, not an
        # oscillating reversal. Reversal after ineffective traversal additionally
        # needs a measured material opening on the alternative side.
        alternative_opened = (finite(alternative_gain)
            and alternative_gain >= MATERIAL_GEOMETRY_CHANGE_M)
        if alt_useful and (lateral.get('permitted') is not True or alternative_opened):
            side_change_reason = ('committed_lateral_unsafe' if lateral.get('permitted') is not True
                                  else 'committed_lateral_ineffective')
            side = alternative
            lateral, bypass, useful, insufficient = alt_lateral, alt_bypass, alt_useful, alt_insufficient
            initial_lateral = True  # Establish the new side before passing.
    result.update(committed_side=side, local_bypass=bypass,
        lateral_progress=lateral_progress, side_change_reason=side_change_reason)
    # Explicitly the production CENTER metric, not minimum-side separation.
    y = route.get('blocking_obstacle_y_m')
    result['blocker_center_separation_m'] = (
        -(1 if side == 'LEFT' else -1) * y if side and type(y) in (int, float) else None)
    result['minimum_side_separation_m'] = route.get('blocking_obstacle_minimum_side_separation_m')
    turn_kind = 'TURN_' + side if side else None
    turn_safe = (result['options'].get(turn_kind) or {}).get('permitted') is True
    phase = phase_from_admission(route_obstructed=True,
        pass_safe=bypass.get('bypass_forward_permitted') is True and not initial_lateral,
        repair_safe=useful or turn_safe)
    if phase == Phase.PASS_OBSTACLE:
        prediction = evaluate_marvin_route(state, association,
            expected_session=expected_session, translation_x=.05)
        result['options']['BYPASS_FORWARD'] = dict(permitted=True,
            hard_safety_permitted=True, requested_duration=.50,
            predicted_route=prediction, predicted_route_occupancy=prediction['route_occupancy'],
            predicted_max_overlap_m=prediction['corridor_overlap_m'],
            predicted_blocker_centerline_clearance_m=prediction.get('blocking_obstacle_centerline_clearance_m'),
            reason='local_bypass_corridor_clear')
        return dict(result, action_type='BYPASS_FORWARD', direction=side,
            phase=Phase.PASS_OBSTACLE.value, reason='find_marvin_phase_pass_corridor_safe')
    if phase == Phase.CLEAR_SIDE:
        kind = 'STRAFE_' + side if useful else turn_kind
        reason = ('find_marvin_phase_lateral_repair_safe' if useful else
            'find_marvin_phase_lateral_progress_insufficient' if insufficient else
            'find_marvin_phase_lateral_unsafe_or_not_useful')
        return dict(result, action_type=kind, direction=side,
            phase=Phase.CLEAR_SIDE.value, reason=reason)
    return dict(result, phase=Phase.BLOCKED_WAIT.value, reason='find_marvin_no_safe_local_detour')
