"""Fail-closed local LiDAR prerequisite for positive streaming motion."""

import copy
import math
import threading
import time

MONITOR_INTERVAL_SECONDS = 0.05
MAXIMUM_EFFECTIVE_AGE_SECONDS = 0.30
FORWARD_POLICY_STRICT = "strict"
FORWARD_POLICY_TARGET_APPROACH = "target_approach"
_FORWARD_POLICIES = {
    FORWARD_POLICY_STRICT,
    FORWARD_POLICY_TARGET_APPROACH,
}


def evaluate_lidar_state(
    state,
    expected_session,
    *,
    policy=FORWARD_POLICY_STRICT,
):
    if policy not in _FORWARD_POLICIES:
        return False, "invalid_forward_policy"
    if not isinstance(state, dict):
        return False, "missing_lidar_state"
    if not expected_session or state.get("producer_session") != expected_session:
        return False, "producer_session_mismatch"
    if state.get("available") is not True or state.get("valid") is not True:
        return False, state.get("reason") or "invalid_lidar_state"
    if state.get("reason") != "fresh":
        return False, "not_fresh"
    age = state.get("effective_age_seconds")
    sectors = state.get("sectors")
    if not isinstance(sectors, dict) or not isinstance(sectors.get("front"), dict):
        return False, "malformed_lidar_state"
    front = sectors["front"]
    if (not isinstance(age, (int, float)) or isinstance(age, bool)
            or not math.isfinite(age)):
        return False, "invalid_effective_age"
    if age < 0 or age > MAXIMUM_EFFECTIVE_AGE_SECONDS:
        return False, "stale_lidar"
    front_state = front.get("state")
    permitted_front_states = {"CLEAR"}
    if policy == FORWARD_POLICY_TARGET_APPROACH:
        permitted_front_states.add("CAUTION")
    if (
        front.get("available") is not True
        or front_state not in permitted_front_states
    ):
        return False, "front_not_clear"
    return (
        True,
        "fresh_target_caution"
        if front_state == "CAUTION"
        else "fresh_clear",
    )


class ForwardMotionInterlock:
    """Monitor a supplied cached-state reader; never performs acquisition."""

    def __init__(self, reader, *, expected_session, stop_callback,
                 monotonic=time.monotonic, interval=MONITOR_INTERVAL_SECONDS):
        self.reader = reader
        self.expected_session = expected_session
        self.stop_callback = stop_callback
        self.monotonic = monotonic
        self.interval = float(interval)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._configured = bool(expected_session and callable(reader))
        self._permitted = False
        self._inhibited = True
        self._reason = "not_started" if self._configured else "not_configured"
        self._state = None
        self._active_forward = False
        self._pending_dispatches = {}
        self._dispatch_outcomes = {}
        self._invalidation_reasons = {}
        self._generation = 0
        self._dispatch_sequence = 0
        self._dispatch_epochs = {}
        self._dispatch_modes = {}
        self._dispatch_policies = {}
        self._active_forward_policy = None
        self._authorization_policy = FORWARD_POLICY_STRICT
        self._bounded_authorization_consumed = False
        self._last_stop_error = None
        self._stopped = False

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def _read(self):
        try:
            return self.reader(expected_session=self.expected_session)
        except Exception as exc:
            return {"available": False, "valid": False, "reason": "reader_exception",
                    "producer_session": self.expected_session, "error": str(exc)}

    def refresh(self, *, policy=FORWARD_POLICY_STRICT):
        state = self._read()
        should_stop = False
        with self._lock:
            was_active = self._active_forward or bool(self._pending_dispatches.get(self._generation))
            pending_policies = {
                self._dispatch_policies.get(dispatch_id)
                for dispatch_id, epoch in self._dispatch_epochs.items()
                if epoch == self._generation
            }
            active_policies = {
                value
                for value in pending_policies | {self._active_forward_policy}
                if value is not None
            }
            if active_policies:
                effective_policy = (
                    FORWARD_POLICY_TARGET_APPROACH
                    if active_policies == {FORWARD_POLICY_TARGET_APPROACH}
                    else FORWARD_POLICY_STRICT
                )
            else:
                effective_policy = (
                    self._authorization_policy
                    if policy is None
                    else policy
                )
            permitted, reason = evaluate_lidar_state(
                state,
                self.expected_session,
                policy=effective_policy,
            )
            self._state = copy.deepcopy(state) if isinstance(state, dict) else None
            self._permitted = permitted
            if not permitted:
                self._inhibited = True
                self._invalidation_reasons[self._generation] = reason
                self._generation += 1
                self._reason = reason
                should_stop = was_active
                self._active_forward = False
                self._active_forward_policy = None
                self._authorization_policy = FORWARD_POLICY_STRICT
            else:
                self._bounded_authorization_consumed = False
                self._inhibited = self._stopped or any(
                    generation != self._generation for generation in self._pending_dispatches
                )
                self._reason = reason
                self._authorization_policy = effective_policy
        if should_stop:
            self._dispatch_stop()
        return permitted, reason

    def _dispatch_stop(self):
        """STOP is ungated and always called outside the interlock lock."""
        error = None
        try:
            result = self.stop_callback()
            if not isinstance(result, dict) or result.get("ok") is not True:
                error = str(result)
        except Exception as exc:
            error = str(exc)
        if error is not None:
            with self._lock:
                self._last_stop_error = error

    def begin_positive_dispatch(
        self,
        *,
        streaming,
        policy=FORWARD_POLICY_STRICT,
    ):
        with self._lock:
            if policy not in _FORWARD_POLICIES:
                raise PermissionError("invalid_forward_policy")
            if policy == FORWARD_POLICY_TARGET_APPROACH and streaming:
                raise PermissionError(
                    "target_approach_requires_bounded_forward"
                )
            if not self._configured:
                raise PermissionError("forward_interlock_not_configured")
            if not self._permitted or self._inhibited:
                raise PermissionError(self._reason)
            if policy != self._authorization_policy:
                raise PermissionError(
                    "forward_interlock_policy_refresh_required"
                )
            if (
                not streaming
                and self._bounded_authorization_consumed
            ):
                raise PermissionError(
                    "bounded_forward_authorization_refresh_required"
                )
            epoch = self._generation
            dispatch_id = self._dispatch_sequence
            self._dispatch_sequence += 1
            self._dispatch_epochs[dispatch_id] = epoch
            self._dispatch_modes[dispatch_id] = bool(streaming)
            self._dispatch_policies[dispatch_id] = policy
            self._pending_dispatches[epoch] = (
                self._pending_dispatches.get(epoch, 0) + 1
            )
            return dispatch_id

    def finalize_positive_dispatch(self, generation, transport_result):
        """Finalize an attempted transport, including failures and exceptions.

        An invalidated request may have reached the remote side after the
        monitor's STOP. Always send another STOP now that transport has returned.
        """
        with self._lock:
            epoch = self._dispatch_epochs.pop(generation, generation)
            streaming = self._dispatch_modes.pop(generation, True)
            dispatch_policy = self._dispatch_policies.pop(
                generation,
                FORWARD_POLICY_STRICT,
            )
            transport_uncertain = transport_result is None
            invalidated = (
                epoch != self._generation
                or self._inhibited
                or transport_uncertain
            )
            invalidation_reason = self._invalidation_reasons.pop(
                epoch,
                "transport_exception"
                if transport_uncertain and epoch == self._generation
                else self._reason if invalidated else None,
            )
            if invalidated:
                self._inhibited = True
                self._active_forward = False
                self._active_forward_policy = None
                if transport_uncertain and epoch == self._generation:
                    self._reason = "transport_exception"
                    self._generation += 1
            elif isinstance(transport_result, dict) and transport_result.get("ok") is True:
                self._active_forward = True
                self._active_forward_policy = dispatch_policy
            if not streaming:
                self._bounded_authorization_consumed = True
                if invalidated:
                    self._dispatch_outcomes[generation] = {
                        "valid": False,
                        "reason": invalidation_reason,
                    }
        if invalidated:
            self._dispatch_stop()
        with self._lock:
            remaining = self._pending_dispatches[epoch] - 1
            if remaining:
                self._pending_dispatches[epoch] = remaining
            else:
                del self._pending_dispatches[epoch]
        return not invalidated

    def dispatch_outcome(self, generation):
        """Return the completion outcome for one finalized generation."""
        with self._lock:
            return copy.deepcopy(
                self._dispatch_outcomes.pop(generation, None)
            )

    def stop_active(self):
        with self._lock:
            self._active_forward = False
            self._active_forward_policy = None
            self._authorization_policy = FORWARD_POLICY_STRICT

    def _run(self):
        while not self._stop.is_set():
            self.refresh(policy=None)
            self._stop.wait(self.interval)

    def start(self):
        with self._lock:
            if self._stopped or not self._configured or self.running:
                return
        self.refresh()
        with self._lock:
            if self._stopped or not self._configured or self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="forward-motion-interlock", daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            was_active = self._active_forward or bool(self._pending_dispatches.get(self._generation))
            self._active_forward = False
            self._active_forward_policy = None
            self._stop.set()
            self._inhibited = True
            self._permitted = False
            self._reason = "interlock_stopped"
            self._invalidation_reasons[self._generation] = (
                "interlock_stopped"
            )
            self._generation += 1
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.interval * 4)
        if was_active:
            self._dispatch_stop()

    def status(self):
        with self._lock:
            state = self._state or {}
            return {
                "configured": self._configured,
                "monitor_running": self.running,
                "forward_permitted": self._permitted and not self._inhibited,
                "inhibited": self._inhibited,
                "reason": self._reason,
                "producer_session": self.expected_session,
                "effective_age_seconds": state.get("effective_age_seconds"),
                "front_state": (
                    state.get("sectors", {}).get("front", {}).get("state", "UNKNOWN")
                    if state.get("valid") is True else "UNKNOWN"
                ),
                "active_forward": self._active_forward,
                "pending_forward": bool(self._pending_dispatches),
                "active_forward_policy": self._active_forward_policy,
                "last_stop_error": self._last_stop_error,
            }
