"""LiDAR admission and command/STOP fencing for guarded motion.

Legacy forward method/state names cover all configured bounded dispatches. Collision
geometry is deliberately not decided here. Guarded forward and lateral motion
perform their just-in-time base-frame swept-path checks before reaching
the Robot Bridge client. Native bounded Bridge commands own continuous sensing;
HTTP observations gate their next dispatch, not their active command. Other
transports retain the existing continuous health veto.
"""

import copy
import math
import threading
import time

MONITOR_INTERVAL_SECONDS = 0.05
MAXIMUM_EFFECTIVE_AGE_SECONDS = 0.30


def evaluate_lidar_state(state, expected_session):
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
    # A missing or unknown front sector is incomplete monitoring evidence and
    # must remain fail-closed.  CAUTION/BLOCKED are diagnostic obstacle labels,
    # however, not a second collision authority: guarded forward dispatch has
    # already passed evaluate_local_motion_safety()'s current base-frame swept
    # geometry immediately before transport.
    if front.get("available") is not True or front.get("state") == "UNKNOWN":
        return False, "front_unavailable_or_unknown"
    return True, "fresh_clear"


class ForwardMotionInterlock:
    """Monitor a supplied cached-state reader; never performs acquisition."""

    def __init__(self, reader, *, expected_session, stop_callback,
                 monotonic=time.monotonic, interval=MONITOR_INTERVAL_SECONDS,
                 bridge_guarded_bounded=False):
        self.reader = reader
        self.expected_session = expected_session
        self.stop_callback = stop_callback
        self.monotonic = monotonic
        self.interval = float(interval)
        self.bridge_guarded_bounded = bridge_guarded_bounded is True
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
        self._dispatch_details = {}
        self._bounded_authorization_consumed = False
        self._last_stop_error = None
        self._stops_inflight = 0
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

    def refresh(self):
        state = self._read()
        permitted, reason = evaluate_lidar_state(state, self.expected_session)
        should_stop = False
        pending = []
        with self._lock:
            native_pending = self.bridge_guarded_bounded and any(
                not streaming for streaming in self._dispatch_modes.values())
            watched_pending = any(
                epoch == self._generation
                and (not self.bridge_guarded_bounded or self._dispatch_modes[dispatch])
                for dispatch, epoch in self._dispatch_epochs.items())
            was_active = self._active_forward or watched_pending
            self._state = copy.deepcopy(state) if isinstance(state, dict) else None
            self._permitted = permitted
            if not permitted:
                self._inhibited = True
                # A delayed HTTP observation cannot revoke a native bounded
                # command. Keep admission inhibited until fresh evidence returns.
                # Actual STOP/shutdown still advances the shared owner fence.
                if was_active or not native_pending:
                    self._invalidation_reasons[self._generation] = reason
                    self._generation += 1
                self._reason = reason
                should_stop = was_active
                self._active_forward = False
            else:
                self._bounded_authorization_consumed = False
                self._inhibited = self._stopped or any(
                    generation != self._generation for generation in self._pending_dispatches
                ) or self._stops_inflight > 0 or self._last_stop_error is not None
                self._reason = ("forward_interlock_stop_unverified"
                                if self._last_stop_error is not None else reason)
            if should_stop:
                # Reserve before releasing the lock: completion cannot admit a
                # successor while this old command's STOP callback is delayed.
                self._stops_inflight += 1
                pending = list(self._dispatch_epochs)
                for dispatch in pending:
                    details = self._dispatch_details[dispatch]
                    details.setdefault("interruption_monotonic_seconds", self.monotonic())
                    details.setdefault("invalidating_lidar_evidence", copy.deepcopy({
                        key: (self._state or {}).get(key) for key in (
                            "producer_session", "acquisition_sequence", "received_monotonic_seconds",
                            "effective_age_seconds", "available", "valid", "reason", "source",
                            "age_at_receipt_seconds", "request_latency_seconds",
                            "acquisition_started_at", "completed_at")}))
                    details["stop_pending"] += 1
        if should_stop:
            event = self._dispatch_stop(reserved=True)
            with self._lock:
                for dispatch in pending:
                    if dispatch in self._dispatch_details:
                        details = self._dispatch_details[dispatch]
                        details["stop_events"].append(event)
                        details["stop_pending"] -= 1
        return permitted, reason

    def _dispatch_stop(self, *, reserved=False):
        """STOP is ungated and always called outside the interlock lock."""
        if not reserved:
            with self._lock:
                self._stops_inflight += 1
        try:
            return self._dispatch_stop_reserved()
        finally:
            with self._lock:
                self._stops_inflight -= 1

    def _dispatch_stop_reserved(self):
        error = None
        started = self.monotonic()
        result = None
        try:
            result = self.stop_callback()
            if not isinstance(result, dict) or result.get("ok") is not True:
                error = str(result)
        except Exception as exc:
            error = str(exc)
        if error is not None:
            with self._lock:
                self._last_stop_error = error
                self._inhibited = True
        return {"started_monotonic_seconds": started,
                "completed_monotonic_seconds": self.monotonic(),
                "result": copy.deepcopy(result), "error": error}

    def begin_positive_dispatch(self, *, streaming):
        with self._lock:
            if not self._configured:
                raise PermissionError("forward_interlock_not_configured")
            if self._stops_inflight:
                raise PermissionError("forward_interlock_stop_pending")
            if self._last_stop_error is not None:
                # A callback timeout/failure can still leave a late STOP in
                # flight. Fresh scans cannot certify resolved motion delivery.
                raise PermissionError("forward_interlock_stop_unverified")
            if not self._permitted or self._inhibited:
                raise PermissionError(self._reason)
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
            self._dispatch_details[dispatch_id] = {
                "dispatch_started_monotonic_seconds": self.monotonic(),
                "stop_events": [], "stop_pending": 0}
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
            native_bounded = self.bridge_guarded_bounded and not streaming
            transport_uncertain = transport_result is None or (native_bounded and (
                not isinstance(transport_result, dict)
                or transport_result.get("evidence_valid") is not True
                or transport_result.get("delivery_uncertain") is True))
            invalidated = (
                epoch != self._generation
                or (self._inhibited and not native_bounded)
                or self._stopped or self._stops_inflight > 0 or self._last_stop_error is not None
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
                if transport_uncertain and epoch == self._generation:
                    self._reason = "transport_exception"
                    self._generation += 1
            elif (not native_bounded and isinstance(transport_result, dict)
                  and transport_result.get("ok") is True):
                self._active_forward = True
            if not streaming:
                self._bounded_authorization_consumed = True
        if invalidated:
            event = self._dispatch_stop()
        with self._lock:
            details = self._dispatch_details.pop(generation)
            if invalidated:
                details["stop_events"].append(event)
                if not streaming:
                    self._dispatch_outcomes[generation] = {
                        "valid": False, "reason": invalidation_reason, **details,
                        "stop_succeeded": (details["stop_pending"] == 0
                                           and all(e["error"] is None for e in details["stop_events"])),
                    }
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

    def _run(self):
        while not self._stop.is_set():
            self.refresh()
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
                "bounded_active_lidar_owner": (
                    "robot_bridge" if self.bridge_guarded_bounded else "cognitive"),
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
                "last_stop_error": self._last_stop_error,
                "stop_pending": self._stops_inflight > 0,
            }
