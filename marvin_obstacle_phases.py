"""Mission-owned obstacle phases; proposals still require the native JIT executor.

No sensor getter, transport, shadow, replay or calibration dependency belongs
here. Previous action selection is diagnostic history, never phase ownership.
"""
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from marvin_detour_watchdog import DetourWatchdog

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
                      entering_detour=False,
                      strafe_duration_limit=LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
                      remaining_avoidance_actions=None):
    """Recompute WHAT from one scan; never authorize transport or reuse a grant.

    Native primitive geometry, ranking and bypass admission are retained. Only
    high-level ownership changes: no direct-route progress credit, continuation
    episode or recovery blacklist controls the next phase-admissible primitive.
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
    result['committed_side'] = side
    bypass = plan_local_bypass(state, association, route,
        expected_session=expected_session, side=side)
    result['local_bypass'] = bypass
    # Explicitly the production CENTER metric, not minimum-side separation.
    y = route.get('blocking_obstacle_y_m')
    result['blocker_center_separation_m'] = (
        -(1 if side == 'LEFT' else -1) * y if side and type(y) in (int, float) else None)
    result['minimum_side_separation_m'] = route.get('blocking_obstacle_minimum_side_separation_m')
    lateral_kind = 'STRAFE_' + side if side else None
    initial_lateral = (entering_detour and
        (result['options'].get(lateral_kind) or {}).get('permitted') is True)
    phase = phase_from_admission(route_obstructed=True,
        pass_safe=bypass.get('bypass_forward_permitted') is True and not initial_lateral,
        repair_safe=(result['options'].get(lateral_kind) or {}).get('permitted') is True)
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
    kind = 'STRAFE_' + side if side else None
    if phase == Phase.CLEAR_SIDE:
        return dict(result, action_type=kind, direction=side,
            phase=Phase.CLEAR_SIDE.value, reason='find_marvin_phase_lateral_repair_safe')
    return dict(result, phase=Phase.BLOCKED_WAIT.value, reason='find_marvin_no_safe_local_detour')
