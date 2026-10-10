"""Mission-local navigation bounds, never evidence or permission to move.

120 s is provisional: the retained 34.78 s mission was still safely passing.
No transport, sensor acquisition, worker, writer or shadow dependency belongs here.
"""
from dataclasses import asdict, dataclass, replace
import math

DETOUR_MAX_SECONDS = 120.0
CLEAR_SIDE_MAX_UNPROVEN_ACTIONS = 4
CLEAR_SIDE_MAX_ACTIONS = 8
PASS_MAX_UNPROVEN_ACTIONS = 12
PASS_MAX_ACTIONS = 24
DETOUR_EMERGENCY_PHYSICAL_ACTIONS = 64  # Includes detour alignment, not only traversal.
MAX_EQUIVALENT_VETOES = 6
MAX_TOTAL_VETOES = 24
MAX_PHASE_TRANSITIONS = 24
MATERIAL_GEOMETRY_CHANGE_M = .02


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Passage:
    session: str
    sequence: int
    side: str
    center_separation_m: float
    blocker_x_m: float
    longitudinal_extent_m: float
    pass_safe: bool

    @classmethod
    def from_native(cls, association, side):
        """Current measured geometry only; never use a predicted/commanded pose."""
        route = association.get('route') or {}
        session, sequence = association.get('producer_session'), association.get('acquisition_sequence')
        x, y = route.get('blocking_obstacle_x_m'), route.get('blocking_obstacle_y_m')
        extent = route.get('blocking_obstacle_longitudinal_extent_m')
        if (not session or type(sequence) is not int or side not in {'LEFT', 'RIGHT'}
                or route.get('valid') is not True or route.get('route_to_marvin_obstructed') is not True
                or not all(finite(v) for v in (x, y, extent))):
            return None
        bypass = (association.get('local_bypass_candidates') or {}).get(side) or {}
        return cls(session, sequence, side, -y if side == 'LEFT' else y, x, extent,
            bypass.get('bypass_forward_permitted') is True)

    @classmethod
    def from_selection(cls, selection):
        return cls.from_native(dict(selection, local_bypass_candidates={
            selection.get('committed_side'): selection.get('local_bypass') or {}}),
            selection.get('committed_side'))


@dataclass(frozen=True)
class DetourWatchdog:
    detour_started_monotonic: float | None = None
    phase_started_monotonic: float | None = None
    phase: str = 'DIRECT'
    total_detour_actions: int = 0
    phase_actions: int = 0
    phase_stagnation_count: int = 0
    equivalent_veto_count: int = 0
    total_veto_count: int = 0
    phase_transitions: int = 0
    last_veto: tuple | None = None
    pending_passage: Passage | None = None
    pending_reassessment: bool = False
    last_progress: str = 'not_evaluated'

    def __post_init__(self):
        if (self.phase not in {'DIRECT', 'CLEAR_SIDE', 'PASS_OBSTACLE', 'REJOIN', 'BLOCKED_WAIT', 'ARRIVED'}
                or any(v is not None and (not finite(v) or v < 0)
                    for v in (self.detour_started_monotonic, self.phase_started_monotonic))
                or any(type(v) is not int or v < 0 for v in (self.total_detour_actions,
                    self.phase_actions, self.phase_stagnation_count, self.equivalent_veto_count,
                    self.total_veto_count, self.phase_transitions))):
            raise ValueError('Invalid detour watchdog context')

    @property
    def active(self):
        return self.detour_started_monotonic is not None and self.phase not in {'DIRECT', 'ARRIVED'}

    def enter(self, phase, now):
        if not finite(now) or now < 0 or (self.phase_started_monotonic is not None
                and now < self.phase_started_monotonic):
            raise ValueError('Invalid detour monotonic clock')
        if phase == self.phase:
            return self
        started = self.detour_started_monotonic
        if started is None and phase not in {'DIRECT', 'ARRIVED'}:
            started = now
        # The mission's original detour frontier NEVER resets, including at REJOIN.
        return replace(self, phase=phase, detour_started_monotonic=started,
            phase_started_monotonic=now, phase_actions=0, phase_stagnation_count=0,
            pending_passage=None, pending_reassessment=False, last_progress='phase_entry',
            phase_transitions=self.phase_transitions + int(self.detour_started_monotonic is not None))

    def physical_action(self):
        if not self.active:
            return self
        return replace(self, total_detour_actions=self.total_detour_actions + 1,
            equivalent_veto_count=0, last_veto=None)

    def completed_traversal(self, selection):
        if not self.active or self.phase not in {'CLEAR_SIDE', 'PASS_OBSTACLE'}:
            return self
        return replace(self, phase_actions=self.phase_actions + 1,
            phase_stagnation_count=self.phase_stagnation_count + 1,
            pending_passage=Passage.from_selection(selection), pending_reassessment=True,
            last_progress='unproven_pending_fresh_geometry')

    def reassess(self, association, *, side):
        if not self.pending_reassessment:
            return self
        before, after = self.pending_passage, Passage.from_native(association, side)
        progress = False
        reason = 'unproven_noncomparable_geometry'
        if (before is not None and after is not None and before.session == after.session
                and before.side == after.side and after.sequence > before.sequence):
            if self.phase == 'CLEAR_SIDE':
                # Passage feasibility, independent of the direct Marvin route metric.
                progress = (after.pass_safe and not before.pass_safe
                    or after.center_separation_m - before.center_separation_m >= MATERIAL_GEOMETRY_CHANGE_M)
                reason = 'measured_passage_feasibility_improved' if progress else 'no_material_passage_gain'
            elif self.phase == 'PASS_OBSTACLE':
                # Two independent longitudinal facts must agree, with stable lateral
                # location and cluster shape. This is obstacle-relative geometry
                # change, NOT a displacement estimate or a command-distance credit.
                comparable = (abs(after.center_separation_m - before.center_separation_m) <= .03
                    and abs((after.longitudinal_extent_m - after.blocker_x_m)
                        - (before.longitudinal_extent_m - before.blocker_x_m)) <= .02)
                progress = (comparable and after.pass_safe
                    and before.blocker_x_m - after.blocker_x_m >= MATERIAL_GEOMETRY_CHANGE_M
                    and before.longitudinal_extent_m - after.longitudinal_extent_m >= MATERIAL_GEOMETRY_CHANGE_M)
                reason = ('measured_obstacle_relative_advancement' if progress else
                    'no_material_passage_gain' if comparable else 'unproven_noncomparable_geometry')
        return replace(self, phase_stagnation_count=0 if progress else self.phase_stagnation_count,
            pending_passage=None, pending_reassessment=False, last_progress=reason)

    def veto(self, primitive, side, reason):
        key = (primitive, side, reason)
        return replace(self, total_veto_count=self.total_veto_count + 1, last_veto=key,
            equivalent_veto_count=self.equivalent_veto_count + 1 if key == self.last_veto else 1)

    def action_fits_deadline(self, now, duration):
        """A navigation budget check, never a grant or grandfathered admission."""
        return (not self.active or finite(now) and finite(duration) and duration > 0
            and now + duration <= self.detour_started_monotonic + DETOUR_MAX_SECONDS)

    def exhaustion(self, now, *, include_phase=True):
        if not self.active:
            return None
        if not finite(now) or now < max(self.detour_started_monotonic, self.phase_started_monotonic):
            return 'find_marvin_detour_clock_invalid'
        if now - self.detour_started_monotonic >= DETOUR_MAX_SECONDS:
            return 'find_marvin_detour_watchdog_exhausted'
        if self.total_detour_actions >= DETOUR_EMERGENCY_PHYSICAL_ACTIONS:
            return 'find_marvin_detour_emergency_action_ceiling_exhausted'
        if self.equivalent_veto_count >= MAX_EQUIVALENT_VETOES:
            return 'find_marvin_equivalent_jit_veto_exhausted'
        if self.total_veto_count >= MAX_TOTAL_VETOES or self.phase_transitions >= MAX_PHASE_TRANSITIONS:
            return 'find_marvin_detour_antispin_exhausted'
        # Allow the just-completed action ONE fresh reassessment before deciding
        # whether its outcome was stagnant. This is not motion authorization.
        if include_phase and not self.pending_reassessment:
            if self.phase == 'CLEAR_SIDE':
                if self.phase_stagnation_count >= CLEAR_SIDE_MAX_UNPROVEN_ACTIONS:
                    return 'find_marvin_clear_side_stagnation_exhausted'
                if self.phase_actions >= CLEAR_SIDE_MAX_ACTIONS:
                    return 'find_marvin_clear_side_attempts_exhausted'
            if self.phase == 'PASS_OBSTACLE':
                if self.phase_stagnation_count >= PASS_MAX_UNPROVEN_ACTIONS:
                    return 'find_marvin_pass_stagnation_exhausted'
                if self.phase_actions >= PASS_MAX_ACTIONS:
                    return 'find_marvin_pass_attempts_exhausted'
        return None

    def record(self):
        return dict(asdict(self), limits={
            'detour_seconds': DETOUR_MAX_SECONDS, 'clear_side_unproven_actions': CLEAR_SIDE_MAX_UNPROVEN_ACTIONS,
            'clear_side_actions': CLEAR_SIDE_MAX_ACTIONS, 'pass_unproven_actions': PASS_MAX_UNPROVEN_ACTIONS,
            'pass_actions': PASS_MAX_ACTIONS, 'emergency_physical_actions': DETOUR_EMERGENCY_PHYSICAL_ACTIONS,
            'equivalent_vetoes': MAX_EQUIVALENT_VETOES, 'total_vetoes': MAX_TOTAL_VETOES,
            'phase_transitions': MAX_PHASE_TRANSITIONS})
