import math
import threading
import time

from robot_bridge.client import RobotBridgeClient
from guarded_turn_policy import validate_guarded_turn
from target_lock import TargetLock


class _GuardedTurnMonitor:
    """Monitor one explicit bounded turn without owning transport locks."""

    INTERVAL_SECONDS = 0.05
    # Allow the synchronous bounded Robot Bridge request to return after the
    # physical window.  This covers the bridge's documented post-zero delay
    # and ordinary scheduling/HTTP overhead; it never changes the requested
    # motion duration.
    TRANSPORT_COMPLETION_ALLOWANCE_SECONDS = 0.25

    def __init__(
        self,
        *,
        world_model,
        robot,
        direction,
        angular_speed,
        duration,
        expected_session,
        generation,
        initial_validation,
        now=None,
    ):
        self.world_model = world_model
        self.robot = robot
        self.direction = direction
        self.angular_speed = angular_speed
        self.duration = duration
        self.expected_session = expected_session
        self.generation = generation
        self.now = now
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._pending = True
        self._active = False
        self._dispatch_started = None
        self._deadline = None
        self._completion_deadline = None
        self._transport_began = False
        self._window_complete = False
        self._late_transport = False
        self._deadline_stop_attempted = False
        self._deadline_stop_result = None
        self._deadline_stop_error = None
        self._deadline_stop_error_type = None
        self._transport_completion_pending = False
        self._transport_completion_pending_seen = False
        self._transport_completion_timed_out = False
        self._normal_completion = False
        self._completed_after_deadline = False
        self._physical_deadline_reached = False
        self._transport_returned = False
        self._transport_accepted = False
        self._inhibited = False
        self._invalidated = False
        self._reason = initial_validation.get("reason")
        self._validation = dict(initial_validation)
        self._last_stop_result = None
        self._last_stop_error = None
        self._last_stop_error_type = None
        self._stop_count = 0
        self._stop_events = []

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self.world_model is None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="guarded-turn-monitor",
            daemon=True,
        )
        self._thread.start()

    def begin_transport(self, dispatch_started):
        with self._lock:
            if self._invalidated:
                return False
            self._pending = False
            self._active = True
            self._dispatch_started = dispatch_started
            self._deadline = dispatch_started + self.duration
            self._completion_deadline = (
                self._deadline + self.TRANSPORT_COMPLETION_ALLOWANCE_SECONDS
            )
            self._transport_began = True
            return True

    def cancel(self, reason="operator_stop"):
        """Invalidate this generation without dispatching a duplicate STOP."""
        with self._lock:
            was_invalidated = self._invalidated
            self._invalidated = True
            self._inhibited = True
            self._reason = reason
            return not was_invalidated

    def is_invalidated(self):
        with self._lock:
            return self._invalidated

    def mark_transport_returned(self, accepted):
        with self._lock:
            self._transport_returned = True
            self._transport_accepted = bool(accepted)
            self._transport_completion_pending = False
            now_monotonic = time.monotonic()
            after_deadline = bool(
                self._deadline is not None
                and now_monotonic >= self._deadline
            )
            if after_deadline:
                self._physical_deadline_reached = True
            self._normal_completion = bool(accepted and not self._invalidated)
            self._completed_after_deadline = bool(
                self._normal_completion
                and after_deadline
            )
            needs_deadline_stop = bool(
                accepted
                and after_deadline
                and not self._deadline_stop_attempted
                and not self._invalidated
            )
            if needs_deadline_stop:
                self._transport_completion_pending_seen = True
            if needs_deadline_stop:
                self._deadline_stop_attempted = True
            return needs_deadline_stop

    def window_expired(self):
        with self._lock:
            return self._deadline is not None and time.monotonic() >= self._deadline

    def wait_for_window(self):
        """Keep the monitor alive through the approved monotonic window."""
        while not self._stop.is_set():
            with self._lock:
                if self._invalidated:
                    return False
                deadline = self._deadline
            if deadline is None:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    self._window_complete = True
                return True
            self._stop.wait(min(remaining, self.INTERVAL_SECONDS))
        return False

    def _read_validation(self):
        try:
            state = self.world_model.get_lidar_obstacles(
                expected_session=self.expected_session,
                now=self.now,
            )
        except Exception:
            state = None
        return validate_guarded_turn(
            self.direction,
            self.angular_speed,
            self.duration,
            state,
            expected_session=self.expected_session,
            now=self.now,
        )

    def _run(self):
        while not self._stop.is_set():
            try:
                validation = self._read_validation()
            except Exception as exc:
                validation = {
                    "permitted": False,
                    "reason": "monitor_validation_error",
                    "producer_session": self.expected_session,
                    "effective_age_seconds": None,
                    "front_state": "UNKNOWN",
                    "left_state": "UNKNOWN",
                    "front_left_state": "UNKNOWN",
                    "right_state": "UNKNOWN",
                    "front_right_state": "UNKNOWN",
                    "monitor_error": str(exc),
                }
            should_stop = False
            deadline_stop = False
            with self._lock:
                self._validation = validation
                now_monotonic = time.monotonic()
                deadline_expired = (
                    self._active
                    and self._deadline is not None
                    and now_monotonic >= self._deadline
                )
                if deadline_expired:
                    self._window_complete = True
                    self._physical_deadline_reached = True
                if not validation.get("permitted"):
                    if not self._invalidated:
                        self._reason = validation.get("reason")
                        self._inhibited = True
                    if not self._invalidated and (self._pending or self._active):
                        self._invalidated = True
                        should_stop = True
                elif deadline_expired and not self._transport_returned:
                    self._transport_completion_pending = True
                    self._transport_completion_pending_seen = True
                    if not self._deadline_stop_attempted:
                        self._deadline_stop_attempted = True
                        deadline_stop = True
                    if (
                        self._completion_deadline is not None
                        and now_monotonic >= self._completion_deadline
                        and not self._invalidated
                    ):
                        self._invalidated = True
                        self._inhibited = True
                        self._transport_completion_timed_out = True
                        self._reason = "transport_completion_timeout"
                        should_stop = True
            if deadline_stop:
                stop_result = self._dispatch_stop(source="deadline")
                self._record_deadline_stop_outcome(stop_result)
            if should_stop:
                self._dispatch_stop(source="monitor")
            self._stop.wait(self.INTERVAL_SECONDS)

    def _dispatch_stop(self, source="monitor"):
        """Issue ungated STOP with no monitor lock held across transport."""
        result = None
        error = None
        error_type = None
        try:
            result = self.robot.stop()
            if not isinstance(result, dict) or result.get("ok") is not True:
                error = str(result)
        except Exception as exc:
            error = str(exc)
            error_type = type(exc).__name__
        with self._lock:
            self._stop_count += 1
            self._last_stop_result = result
            self._last_stop_error = error
            self._last_stop_error_type = error_type
            event = {
                "result": result,
                "error": error,
                "error_type": error_type,
                "source": source,
            }
            self._stop_events.append(event)
            if source == "deadline":
                self._deadline_stop_result = result
                self._deadline_stop_error = error
                self._deadline_stop_error_type = error_type
        return result, error, error_type

    def _record_deadline_stop_outcome(self, stop_outcome):
        """Invalidate if the normal-window STOP was not confirmed."""
        result, error, _error_type = stop_outcome
        confirmed = (
            error is None
            and isinstance(result, dict)
            and result.get("ok") is True
        )
        if confirmed:
            return False
        with self._lock:
            if self._invalidated:
                return False
            self._invalidated = True
            self._inhibited = True
            self._reason = "deadline_stop_failed"
        return True

    def record_external_stop(self, result, error=None, error_type=None):
        """Record an operator STOP that already served immediate cancellation."""
        if error is None and (
            not isinstance(result, dict) or result.get("ok") is not True
        ):
            error = str(result)
        with self._lock:
            self._stop_count += 1
            self._last_stop_result = result
            self._last_stop_error = error
            self._last_stop_error_type = error_type
            self._stop_events.append({
                "result": result,
                "error": error,
                "error_type": error_type,
                "source": "operator",
            })

    def stop_monitor(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def finalize(self, *, force_post_stop=False):
        with self._lock:
            invalidated = self._invalidated
            self._pending = False
            self._active = False
            if invalidated or force_post_stop:
                self._inhibited = True
            transport_began = self._transport_began
        post_return = None
        if transport_began and (invalidated or force_post_stop):
            post_return = self._dispatch_stop()
        return invalidated, force_post_stop, post_return

    def status(self):
        with self._lock:
            validation = dict(self._validation)
            return {
                "monitor_configured": self.world_model is not None,
                "monitor_running": self.running,
                "direction": self.direction,
                "pending_turn": self._pending,
                "active_turn": self._active,
                "inhibited": self._inhibited,
                "dispatch_started_monotonic": self._dispatch_started,
                "window_complete": self._window_complete,
                "physical_deadline_reached": self._physical_deadline_reached,
                "late_transport": self._late_transport,
                "transport_completion_allowance_seconds": (
                    self.TRANSPORT_COMPLETION_ALLOWANCE_SECONDS
                ),
                "transport_completion_deadline_monotonic": (
                    self._completion_deadline
                ),
                "transport_completion_pending": (
                    self._transport_completion_pending
                ),
                "transport_completion_pending_seen": (
                    self._transport_completion_pending_seen
                ),
                "transport_completion_timed_out": (
                    self._transport_completion_timed_out
                ),
                "normal_completion": self._normal_completion,
                "completed_after_deadline": self._completed_after_deadline,
                "deadline_stop_attempted": self._deadline_stop_attempted,
                "deadline_stop_result": self._deadline_stop_result,
                "deadline_stop_error": self._deadline_stop_error,
                "deadline_stop_error_type": self._deadline_stop_error_type,
                "transport_returned": self._transport_returned,
                "transport_accepted": self._transport_accepted,
                "transport_began": self._transport_began,
                "generation": self.generation,
                "generation_invalidated": self._invalidated,
                "reason": self._reason,
                "producer_session": validation.get("producer_session"),
                "effective_age_seconds": validation.get("effective_age_seconds"),
                "front_state": validation.get("front_state", "UNKNOWN"),
                "left_state": validation.get("left_state", "UNKNOWN"),
                "front_left_state": validation.get("front_left_state", "UNKNOWN"),
                "right_state": validation.get("right_state", "UNKNOWN"),
                "front_right_state": validation.get("front_right_state", "UNKNOWN"),
                "last_stop_result": self._last_stop_result,
                "last_stop_error": self._last_stop_error,
                "last_stop_error_type": self._last_stop_error_type,
                "stop_count": self._stop_count,
                "stop_events": list(self._stop_events),
            }


class BehaviorManager:
    SEARCH_TURN_SPEED = 0.30
    SEARCH_TURN_SECONDS = 1.0
    SEARCH_MAX_TURN_CHUNKS = 3
    SEARCH_DIRECTION = "LEFT"
    TARGET_CONFIRMATION_MAX_FRAMES = 3
    TARGET_CONFIRMATION_MIN_SUPPORT = 2
    TARGET_CONFIRMATION_WINDOW_SECONDS = 0.90
    TARGET_CONFIRMATION_POLL_SECONDS = 0.05

    FIND_CENTER_TOLERANCE_PIXELS = 50.0
    FIND_CENTER_TURN_SPEED = 0.20
    FIND_CENTER_TURN_SECONDS = 0.25
    FIND_CENTER_MAX_TURN_CHUNKS = 8

    CENTER_TURN_SPEED = 0.60
    CENTER_TURN_SECONDS = 0.40

    FIND_FORWARD_SPEED = 0.08
    FIND_FORWARD_SECONDS = 0.80
    FIND_ARRIVAL_AREA = 75000.0

    FOLLOW_SEARCH_TURN_SPEED = 0.50
    FOLLOW_SEARCH_TURN_SECONDS = 0.30

    FOLLOW_CENTER_TURN_SPEED = 0.65
    FOLLOW_CENTER_TURN_SECONDS = 0.28

    # Adaptive FOLLOW_PERSON steering controller.
    #
    # The horizontal pixel error is converted into an angular velocity.
    # Every command remains short and automatically stops at the Robot
    # Bridge after FOLLOW_CENTER_TURN_SECONDS.
    FOLLOW_TURN_KP = 0.0030
    FOLLOW_MIN_TURN_SPEED = 0.32
    FOLLOW_MAX_TURN_SPEED = 0.95

    # While the target is inside the center tolerance region, the robot may
    # move forward and apply a small simultaneous steering correction.
    FOLLOW_APPROACH_TURN_KP = 0.0025
    FOLLOW_MAX_APPROACH_TURN_SPEED = 0.28

    FOLLOW_FORWARD_SPEED = 0.14
    FOLLOW_FORWARD_SECONDS = 0.45
    FOLLOW_STOP_AREA = 60000.0

    # FOLLOW_PERSON continuously refreshes the Robot Bridge deadman
    # watchdog. If updates stop, the bridge automatically publishes zero
    # velocity.
    FOLLOW_STREAM_WATCHDOG_SECONDS = 0.50

    DEFAULT_IMAGE_WIDTH = 640.0
    CENTER_TOLERANCE_PIXELS = 95.0

    # FOLLOW_PERSON servo smoothing.
    #
    # The object detector naturally moves the reported bounding box slightly
    # between frames. Filtering prevents that perception noise from becoming
    # physical steering jitter.
    FOLLOW_ERROR_FILTER_ALPHA = 0.25

    # Steering begins outside CENTER_TOLERANCE_PIXELS, but an active turn is
    # not released until the target enters this tighter center region.
    FOLLOW_CENTER_EXIT_TOLERANCE_PIXELS = 55.0

    # Maximum permitted angular_z change during one cognitive runtime cycle.
    FOLLOW_MAX_ANGULAR_STEP = 0.09

    MAX_FIND_CYCLES = 30
    FIND_CYCLE_PAUSE = 1.50

    TARGET_MAX_AGE_SECONDS = 3.0

    def __init__(
        self,
        robot_client=None,
        vision_adapter=None,
        world_model=None,
    ):
        self.robot = robot_client or RobotBridgeClient()
        self.vision = vision_adapter
        self.world_model = (
            world_model
            or getattr(vision_adapter, "world_model", None)
        )

        self.target_lock = (
            TargetLock(
                world_model=self.world_model,
                max_age_seconds=self.TARGET_MAX_AGE_SECONDS,
            )
            if self.world_model is not None
            else None
        )

        self._follow_mission_id = None
        self._guarded_turn_generation = 0
        self._guarded_turn_slot_lock = threading.Lock()
        self._guarded_turn_owner_generation = None
        self._guarded_turn_monitor = None
        # Optional runtime-owned hook for live, dashboard-facing telemetry.
        # BehaviorManager remains usable without a runtime callback.
        self.tracking_state_callback = None

    def _publish_tracking_state(self, result):
        callback = getattr(self, "tracking_state_callback", None)
        if not callable(callback):
            return
        try:
            callback(result)
        except Exception:
            # Telemetry must never affect guarded behavior or motion safety.
            return

    def simulate(self, mission):
        """
        Describe the intended behavior without moving the robot.
        """
        mission_type = mission.mission_type
        status = mission.status
        target = mission.target

        if status == "REJECTED":
            return "Behavior: Mission rejected. No robot action taken."

        if status == "CANCELLED":
            return "Behavior: Active mission cancelled. Robot should stop."

        simulations = {
            "FOLLOW_PERSON": (
                f"Behavior: Tracking target '{target}'. "
                "Robot Bridge is ready for safe motion execution."
            ),
            "MOVE_FORWARD": (
                "Behavior: Moving forward briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "TURN_LEFT": (
                "Behavior: Turning left briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "TURN_RIGHT": (
                "Behavior: Turning right briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "FIND_OBJECT": (
                f"Behavior: Query vision for '{target}', then execute one "
                "bounded search, centering, approach, or arrival action."
            ),
            "RETURN_HOME": (
                "Behavior: Return-home mission prepared. "
                "Navigation is not implemented yet."
            ),
            "DESCRIBE_SCENE": (
                "Behavior: Scene description requested. "
                "Using current robot context."
            ),
            "STOP": "Behavior: Robot stop requested.",
        }

        return simulations.get(
            mission_type,
            "Behavior: No simulated behavior available.",
        )

    def execute(self, mission):
        """
        Execute one bounded mission action through the Robot Bridge.
        """
        if mission.status == "REJECTED":
            return {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": "Mission rejected.",
            }

        if mission.status == "CANCELLED" or mission.mission_type == "STOP":
            return self._execute_stop()

        handlers = {
            "FOLLOW_PERSON": self._execute_follow_person,
            "MOVE_FORWARD": self._execute_move_forward,
            "TURN_LEFT": self._execute_turn_left,
            "TURN_RIGHT": self._execute_turn_right,
            "FIND_OBJECT": self._execute_find_object,
            "RETURN_HOME": self._execute_return_home,
            "DESCRIBE_SCENE": self._execute_describe_scene,
        }

        handler = handlers.get(mission.mission_type)

        if handler is None:
            return {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": (
                    f"No executable behavior for "
                    f"'{mission.mission_type}'."
                ),
            }

        return handler(mission)

    def _execute_stop(self):
        if self.target_lock is not None:
            self.target_lock.reset()

        self._follow_mission_id = None
        with self._guarded_turn_slot_lock:
            guarded_monitor = self._guarded_turn_monitor
            if guarded_monitor is not None:
                guarded_monitor.cancel("operator_stop")
        try:
            robot_result = self.robot.stop()
        except Exception as exc:
            if guarded_monitor is not None:
                guarded_monitor.record_external_stop(
                    None,
                    str(exc),
                    type(exc).__name__,
                )
            raise
        if guarded_monitor is not None:
            guarded_monitor.record_external_stop(robot_result)

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "STOP",
            "state": "STOPPED",
            "reason": "Robot stop command sent.",
            "robot_result": robot_result,
        }

    def _release_guarded_turn(self, monitor):
        """Release only the slot owned by this completed generation."""
        with self._guarded_turn_slot_lock:
            if self._guarded_turn_monitor is monitor:
                self._guarded_turn_monitor = None
                self._guarded_turn_owner_generation = None

    def execute_guarded_turn(
        self,
        direction,
        angular_speed,
        duration,
        *,
        expected_lidar_session,
        now=None,
    ):
        """Validate and execute one explicit bounded turn request.

        This is an advisory caller's execution boundary, not an autonomous
        behavior.  The transient World Model snapshot is validated before a
        single bounded angular-only Robot Bridge request is sent.  STOP
        remains independent and unconditional through ``execute``.
        """
        state = None
        if self.world_model is not None:
            try:
                state = self.world_model.get_lidar_obstacles(
                    expected_session=expected_lidar_session,
                    now=now,
                )
            except Exception:
                state = None

        validation = validate_guarded_turn(
            direction,
            angular_speed,
            duration,
            state,
            expected_session=expected_lidar_session,
            now=now,
        )
        result = dict(validation)
        result.update(
            ok=False,
            forwarded=False,
            confirmed_forwarded=False,
            transport_attempted=False,
            delivery_uncertain=False,
            validation_reason=validation.get("reason"),
            transport_result=None,
            transport_error=None,
            transport_error_type=None,
            stop_fallback_attempted=False,
            stop_fallback_result=None,
            stop_fallback_error=None,
            stop_fallback_error_type=None,
            pending_turn=False,
            active_turn=False,
            generation=None,
            generation_invalidated=False,
            monitor_configured=self.world_model is not None,
            monitor_running=False,
            inhibited=not validation.get("permitted"),
            last_stop_result=None,
            last_stop_error=None,
            last_stop_error_type=None,
            stop_count=0,
            stop_events=[],
        )

        with self._guarded_turn_slot_lock:
            slot_occupied = self._guarded_turn_owner_generation is not None
        if slot_occupied:
            result.update(
                permitted=False,
                reason="turn_already_active",
                validation_reason="turn_already_active",
                inhibited=True,
            )
            return result

        if not validation.get("permitted"):
            return result

        with self._guarded_turn_slot_lock:
            if self._guarded_turn_owner_generation is not None:
                result.update(
                    permitted=False,
                    reason="turn_already_active",
                    validation_reason="turn_already_active",
                    inhibited=True,
                )
                return result
            self._guarded_turn_generation += 1
            generation = self._guarded_turn_generation
            monitor = _GuardedTurnMonitor(
                world_model=self.world_model,
                robot=self.robot,
                direction=direction,
                angular_speed=angular_speed,
                duration=validation["duration"],
                expected_session=expected_lidar_session,
                generation=generation,
                initial_validation=validation,
                now=now,
            )
            self._guarded_turn_owner_generation = generation
            self._guarded_turn_monitor = monitor
        result["generation"] = generation
        result["pending_turn"] = True
        result["monitor_configured"] = True
        result["monitor_running"] = True
        monitor.start()
        dispatch_started = time.monotonic()
        transport_began = monitor.begin_transport(dispatch_started)
        if not transport_began:
            monitor.stop_monitor()
            invalidated, forced_post, post_return_stop = monitor.finalize()
            status = monitor.status()
            self._release_guarded_turn(monitor)
            result.update(status)
            result.update(
                ok=False,
                forwarded=False,
                confirmed_forwarded=False,
                transport_attempted=False,
                reason=status["reason"] or "turn_cancelled_before_transport",
                validation_reason=status["reason"] or "turn_cancelled_before_transport",
                generation_invalidated=invalidated,
                pending_turn=False,
                active_turn=False,
            )
            result["monitor_reason"] = status["reason"]
            return result
        result["dispatch_started_monotonic"] = dispatch_started
        result["active_turn"] = True
        result["pending_turn"] = False
        result["transport_attempted"] = True
        transport_result = None
        transport_error = None
        try:
            transport_result = self.robot.motion(
                linear_x=0.0,
                angular_z=validation["angular_z"],
                duration=validation["duration"],
                streaming=False,
            )
        except Exception as exc:
            transport_error = exc

        transport_ok = (
            transport_error is None
            and isinstance(transport_result, dict)
            and transport_result.get("ok") is True
        )
        deadline_stop_needed = monitor.mark_transport_returned(transport_ok)
        if deadline_stop_needed:
            # A response that arrives just after the physical deadline may
            # race the monitor's sampling tick.  Preserve the same normal
            # deadline-stop semantics, including fail-closed failure handling.
            deadline_stop = monitor._dispatch_stop(source="deadline")
            monitor._record_deadline_stop_outcome(deadline_stop)
        if transport_error is None and transport_ok:
            if not monitor.is_invalidated() and not monitor.window_expired():
                monitor.wait_for_window()
        force_post_stop = False
        monitor.stop_monitor()
        invalidated, forced_post, post_return_stop = monitor.finalize(
            force_post_stop=force_post_stop,
        )
        status = monitor.status()
        self._release_guarded_turn(monitor)
        result.update(status)
        result["monitor_reason"] = status["reason"]
        result["pending_turn"] = False
        result["active_turn"] = False
        if invalidated:
            result["inhibited"] = True
            result["reason"] = status["reason"] or "turn_invalidated"
        if transport_error is not None:
            result["delivery_uncertain"] = True
            result["transport_error"] = str(transport_error)
            result["transport_error_type"] = type(transport_error).__name__
            result["reason"] = "transport_exception"
            result["transport_result"] = {
                "ok": False,
                "error": str(transport_error),
            }
        else:
            result["transport_result"] = transport_result
        if invalidated or forced_post:
            result["stop_fallback_attempted"] = True
            result["stop_fallback_result"] = post_return_stop[0]
            result["stop_fallback_error"] = post_return_stop[1]
            result["stop_fallback_error_type"] = post_return_stop[2]
        elif not transport_ok:
            result["delivery_uncertain"] = True
            result["stop_fallback_attempted"] = True
            fallback_result = None
            fallback_error = None
            fallback_error_type = None
            try:
                fallback_result = self.robot.stop()
                result["stop_fallback_result"] = fallback_result
                if not isinstance(fallback_result, dict) or fallback_result.get("ok") is not True:
                    fallback_error = str(fallback_result)
            except Exception as stop_exc:
                fallback_error = str(stop_exc)
                fallback_error_type = type(stop_exc).__name__
                result["stop_fallback_error"] = fallback_error
                result["stop_fallback_error_type"] = fallback_error_type
            if fallback_error is not None:
                result["stop_fallback_error"] = fallback_error
                result["stop_fallback_error_type"] = fallback_error_type
            result["last_stop_result"] = fallback_result
            result["last_stop_error"] = fallback_error
            result["last_stop_error_type"] = fallback_error_type
            result["stop_count"] = 1
            result["stop_events"] = [{
                "result": fallback_result,
                "error": fallback_error,
                "error_type": fallback_error_type,
            }]
        result.update(
            ok=transport_ok and not invalidated and not forced_post,
            forwarded=transport_ok,
            confirmed_forwarded=transport_ok,
            reason=(result["reason"] if invalidated else (
                "transport_exception"
                if transport_error is not None
                else status["reason"]
                if forced_post
                else (
                    validation.get("reason")
                    if transport_ok
                    else "transport_failed"
                )
            )),
        )
        return result

    def _execute_follow_person(self, mission):
        """
        Execute one bounded FOLLOW_PERSON cycle.

        The CognitiveRuntime owns repetition. This mission remains active
        while the person is being searched for, centered, approached, or
        held at the desired following distance. STOP ends the mission.
        """
        target_name = "person"
        self._follow_mission_id = getattr(
            mission,
            "mission_id",
            None,
        )

        if (
            self.world_model is None
            and self.vision is None
        ):
            return {
                "ok": False,
                "executed": False,
                "completed": True,
                "behavior": "FOLLOW_PERSON",
                "target": target_name,
                "reason": (
                    "World Model perception source "
                    "is not configured."
                ),
            }

        result = self._execute_visual_servo_cycle(
            behavior="FOLLOW_PERSON",
            target_name=target_name,
            cycle_number=1,
            stop_area=self.FOLLOW_STOP_AREA,
            search_turn_speed=self.FOLLOW_SEARCH_TURN_SPEED,
            search_turn_seconds=self.FOLLOW_SEARCH_TURN_SECONDS,
            center_turn_speed=self.FOLLOW_CENTER_TURN_SPEED,
            center_turn_seconds=self.FOLLOW_CENTER_TURN_SECONDS,
            forward_speed=self.FOLLOW_FORWARD_SPEED,
            forward_seconds=self.FOLLOW_FORWARD_SECONDS,
            complete_when_close=False,
            close_state="MAINTAINING_DISTANCE",
            close_reason=(
                "Person is centered and within the "
                "desired following distance."
            ),
        )

        if self.target_lock is not None:
            result.update(
                self.target_lock.snapshot()
            )

        return result

    def _execute_move_forward(self, mission):
        robot_result = self.robot.move_forward(
            speed=0.08,
            seconds=0.50,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "MOVE_FORWARD",
            "reason": "Executed short forward movement.",
            "robot_result": robot_result,
        }

    def _execute_turn_left(self, mission):
        robot_result = self.robot.turn_left(
            speed=0.50,
            seconds=0.40,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "TURN_LEFT",
            "reason": "Executed short left turn.",
            "robot_result": robot_result,
        }

    def _execute_turn_right(self, mission):
        robot_result = self.robot.turn_right(
            speed=0.50,
            seconds=0.40,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "TURN_RIGHT",
            "reason": "Executed short right turn.",
            "robot_result": robot_result,
        }

    def _execute_find_object(self, mission):
        """
        Execute exactly one bounded FIND_OBJECT cycle.

        The CognitiveRuntime owns repetition. If the returned result contains
        completed=False, the active mission remains active and this method is
        called again during a later runtime cycle.
        """
        target_name = str(
            mission.target or ""
        ).strip().lower()

        if not target_name:
            return {
                "ok": False,
                "executed": False,
                "completed": True,
                "behavior": "FIND_OBJECT",
                "reason": "FIND_OBJECT requires a target.",
            }

        if (
            self.world_model is None
            and self.vision is None
        ):
            return {
                "ok": False,
                "executed": False,
                "completed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "reason": (
                    "World Model perception source "
                    "is not configured."
                ),
            }

        return self._execute_guarded_find_search(target_name)

    def _current_lidar_session(self):
        provider = getattr(self, "lidar_session_provider", None)
        if callable(provider):
            try:
                return provider()
            except Exception:
                return None
        return getattr(self, "lidar_session", None)

    @staticmethod
    def _target_is_fresh_and_acquired(target):
        def finite(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        return (
            isinstance(target, dict)
            and target.get("found") is True
            and target.get("stale") is not True
            and finite(target.get("cx"))
            and finite(target.get("cy"))
            and finite(target.get("area"))
            and target.get("area") > 0
            and finite(target.get("image_width"))
            and target.get("image_width") > 0
            and finite(target.get("image_height"))
            and target.get("image_height") > 0
        )

    @staticmethod
    def _target_bbox(target):
        bbox = target.get("bbox") if isinstance(target, dict) else None
        if not isinstance(bbox, dict):
            return None
        values = (
            bbox.get("x1"),
            bbox.get("y1"),
            bbox.get("x2"),
            bbox.get("y2"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            return None
        if values[2] <= values[0] or values[3] <= values[1]:
            return None
        return {
            "x1": float(values[0]),
            "y1": float(values[1]),
            "x2": float(values[2]),
            "y2": float(values[3]),
        }

    @classmethod
    def _target_bbox_iou(cls, first, second):
        first_bbox = cls._target_bbox(first)
        second_bbox = cls._target_bbox(second)
        if first_bbox is None or second_bbox is None:
            return 0.0

        x1 = max(first_bbox["x1"], second_bbox["x1"])
        y1 = max(first_bbox["y1"], second_bbox["y1"])
        x2 = min(first_bbox["x2"], second_bbox["x2"])
        y2 = min(first_bbox["y2"], second_bbox["y2"])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = (
            first_bbox["x2"] - first_bbox["x1"]
        ) * (first_bbox["y2"] - first_bbox["y1"])
        second_area = (
            second_bbox["x2"] - second_bbox["x1"]
        ) * (second_bbox["y2"] - second_bbox["y1"])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    @classmethod
    def _target_bbox_intersection_over_smaller(cls, first, second):
        """Return intersection area divided by the smaller bbox area."""
        first_bbox = cls._target_bbox(first)
        second_bbox = cls._target_bbox(second)
        if first_bbox is None or second_bbox is None:
            return 0.0

        x1 = max(first_bbox["x1"], second_bbox["x1"])
        y1 = max(first_bbox["y1"], second_bbox["y1"])
        x2 = min(first_bbox["x2"], second_bbox["x2"])
        y2 = min(first_bbox["y2"], second_bbox["y2"])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = (
            first_bbox["x2"] - first_bbox["x1"]
        ) * (first_bbox["y2"] - first_bbox["y1"])
        second_area = (
            second_bbox["x2"] - second_bbox["x1"]
        ) * (second_bbox["y2"] - second_bbox["y1"])
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area > 0.0 else 0.0

    @classmethod
    def _target_observations_match(cls, first, second):
        """Return whether two same-label observations can share a cluster."""
        first_label = first.get("label") if isinstance(first, dict) else None
        second_label = second.get("label") if isinstance(second, dict) else None
        if (
            not isinstance(first_label, str)
            or not isinstance(second_label, str)
            or first_label.casefold() != second_label.casefold()
        ):
            return False

        if cls._target_bbox_iou(first, second) >= 0.50:
            return True

        def finite_positive(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0.0
            )

        metrics = (
            first.get("cx"),
            first.get("cy"),
            first.get("area"),
            first.get("image_width"),
            first.get("image_height"),
            second.get("cx"),
            second.get("cy"),
            second.get("area"),
            second.get("image_width"),
            second.get("image_height"),
        )
        if not all(finite_positive(value) for value in metrics):
            return False

        # FIND_OBJECT turns on horizontal image error.  Associate detector
        # shape variants by horizontal center so changes in box height do not
        # split one visible target into separate temporal clusters.
        center_distance = abs(float(first["cx"]) - float(second["cx"]))
        area_ratio = max(float(first["area"]), float(second["area"])) / min(
            float(first["area"]), float(second["area"])
        )
        if center_distance > 60.0:
            return False

        if area_ratio <= 2.0:
            return True

        return cls._target_bbox_intersection_over_smaller(first, second) >= 0.50

    def _confirm_target_candidates(self, target_name):
        """Confirm a target across distinct cached Vision Server frames."""
        self._last_target_confirmation_status = "target_lost"
        fetch = getattr(self.vision, "fetch_target_candidates", None)
        normalize = getattr(self.vision, "normalize_detection", None)
        if not callable(fetch) or not callable(normalize):
            return None

        started = time.monotonic()
        seen_timestamps = set()
        clusters = []
        candidate_seen = False

        while (
            len(seen_timestamps) < self.TARGET_CONFIRMATION_MAX_FRAMES
            and time.monotonic() - started
            <= self.TARGET_CONFIRMATION_WINDOW_SECONDS
        ):
            try:
                payload = fetch(target_name)
            except Exception:
                self._last_target_confirmation_status = (
                    "target_reconfirmation_failed"
                    if candidate_seen
                    else "target_lost"
                )
                return None

            if not isinstance(payload, dict):
                return None
            if payload.get("camera_running") is not True:
                return None

            timestamp = payload.get("timestamp")
            if not isinstance(timestamp, str) or not timestamp.strip():
                return None
            if timestamp in seen_timestamps:
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue

            seen_timestamps.add(timestamp)
            raw_detections = payload.get("detections")
            if not isinstance(raw_detections, list):
                continue

            observations = []
            for raw_detection in raw_detections:
                if not isinstance(raw_detection, dict):
                    continue
                label = str(raw_detection.get("label", ""))
                if label.casefold() != str(target_name).casefold():
                    continue
                try:
                    normalized = normalize(raw_detection)
                except Exception:
                    continue
                if not isinstance(normalized, dict):
                    continue
                normalized["found"] = True
                normalized["stale"] = False
                normalized["target"] = target_name
                normalized["source_timestamp"] = timestamp
                normalized["raw_detection"] = dict(raw_detection)
                if (
                    self._target_is_fresh_and_acquired(normalized)
                    and self._target_bbox(normalized) is not None
                ):
                    candidate_seen = True
                    observations.append(normalized)

            for observation in sorted(
                observations,
                key=lambda item: (
                    -(item.get("confidence") or 0.0),
                    -(item.get("area") or 0.0),
                ),
            ):
                matching = []
                for index, cluster in enumerate(clusters):
                    if timestamp in cluster["timestamps"]:
                        continue
                    if any(
                        self._target_observations_match(observation, member)
                        for member in cluster["observations"]
                    ):
                        matching.append((1.0, index))

                if matching:
                    _, cluster_index = max(
                        matching,
                        key=lambda item: (item[0], -item[1]),
                    )
                    cluster = clusters[cluster_index]
                    cluster["observations"].append(observation)
                    cluster["timestamps"].add(timestamp)
                else:
                    clusters.append({
                        "observations": [observation],
                        "timestamps": {timestamp},
                    })

        eligible = [
            cluster
            for cluster in clusters
            if len(cluster["timestamps"])
            >= self.TARGET_CONFIRMATION_MIN_SUPPORT
        ]
        if not eligible:
            self._last_target_confirmation_status = (
                "target_reconfirmation_failed"
                if candidate_seen
                else "target_lost"
            )
            return None

        def cluster_key(cluster):
            observations = cluster["observations"]
            mean_confidence = sum(
                float(item.get("confidence") or 0.0)
                for item in observations
            ) / len(observations)
            mean_area = sum(
                float(item.get("area") or 0.0)
                for item in observations
            ) / len(observations)
            return (
                len(cluster["timestamps"]),
                mean_confidence,
                mean_area,
            )

        winning = max(
            eligible,
            key=cluster_key,
        )
        return max(
            winning["observations"],
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                float(item.get("area") or 0.0),
            ),
        )

    def _promote_confirmed_target(self, target):
        processor = getattr(self.vision, "process_detection_frame", None)
        if not callable(processor):
            return None
        raw_detection = target.get("raw_detection")
        if not isinstance(raw_detection, dict):
            return None
        try:
            processor([dict(raw_detection)])
            promoted = self._get_target_observation(
                target.get("target", "")
            )
        except Exception:
            return None
        return promoted if self._target_is_fresh_and_acquired(promoted) else None

    def _center_acquired_target(self, target_name, observation, base):
        """Center an acquired FIND_OBJECT target in bounded guarded chunks."""
        centering_attempted = 0
        centering_completed = 0
        current = observation
        last_guarded_result = base.get("last_guarded_turn_result")

        def result(**fields):
            value = dict(
                base,
                target=target_name,
                target_found=fields.pop("target_found", True),
                centering_turn_chunks_attempted=centering_attempted,
                centering_turn_chunks_completed=centering_completed,
                maximum_centering_turn_chunks=(
                    self.FIND_CENTER_MAX_TURN_CHUNKS
                ),
                center_tolerance_pixels=(
                    self.FIND_CENTER_TOLERANCE_PIXELS
                ),
                last_guarded_turn_result=last_guarded_result,
                **fields,
            )
            self._publish_tracking_state(value)
            return value

        while True:
            if not self._target_is_fresh_and_acquired(current):
                return result(
                    ok=False,
                    completed=False,
                    state="TARGET_LOST_DURING_CENTERING",
                    reason="Target observation is no longer actionable.",
                    target_observation=current,
                )

            cx = current.get("cx")
            image_width = current.get("image_width")
            image_center_x = float(image_width) / 2.0
            horizontal_error = float(cx) - image_center_x
            telemetry = {
                "target_label": target_name,
                "target_confidence": current.get("confidence"),
                "target_center_x": float(cx),
                "target_center_y": float(current.get("cy")),
                "target_area": float(current.get("area")),
                "bbox": current.get("bbox"),
                "image_width": float(image_width),
                "image_height": float(current.get("image_height")),
                "image_center_x": image_center_x,
                "horizontal_error": horizontal_error,
                "horizontal_error_pixels": horizontal_error,
                "steering_direction": (
                    "LEFT"
                    if horizontal_error < -self.FIND_CENTER_TOLERANCE_PIXELS
                    else (
                        "RIGHT"
                        if horizontal_error > self.FIND_CENTER_TOLERANCE_PIXELS
                        else "CENTERED"
                    )
                ),
                "centering_direction": (
                    "LEFT"
                    if horizontal_error < -self.FIND_CENTER_TOLERANCE_PIXELS
                    else (
                        "RIGHT"
                        if horizontal_error > self.FIND_CENTER_TOLERANCE_PIXELS
                        else "CENTERED"
                    )
                ),
                "target_observation": current,
            }

            # Publish the state before any guarded turn transport begins so
            # the runtime status endpoint reflects the action in progress.
            self._publish_tracking_state(dict(
                base,
                **telemetry,
                target=target_name,
                target_found=True,
                state="CENTERING",
                ok=True,
                completed=False,
            ))

            if abs(horizontal_error) <= self.FIND_CENTER_TOLERANCE_PIXELS:
                return result(
                    ok=True,
                    completed=True,
                    executed=base["turn_chunks_attempted"] > 0
                    or centering_attempted > 0,
                    state="CENTERED",
                    reason=f"Centered {target_name}.",
                    **telemetry,
                )

            if centering_attempted >= self.FIND_CENTER_MAX_TURN_CHUNKS:
                return result(
                    ok=False,
                    completed=False,
                    state="CENTERING_EXHAUSTED",
                    reason=f"{target_name} remained outside center tolerance.",
                    **telemetry,
                )

            direction = (
                "LEFT"
                if horizontal_error < 0.0
                else "RIGHT"
            )
            session = self._current_lidar_session()
            if session is None:
                return result(
                    ok=False,
                    completed=False,
                    state="CENTERING_BLOCKED",
                    reason="LiDAR producer session is unavailable.",
                    **telemetry,
                )

            centering_attempted += 1
            try:
                guarded_result = self.execute_guarded_turn(
                    direction,
                    self.FIND_CENTER_TURN_SPEED,
                    self.FIND_CENTER_TURN_SECONDS,
                    expected_lidar_session=session,
                )
            except Exception as exc:
                return result(
                    ok=False,
                    completed=False,
                    state="CENTERING_BLOCKED",
                    reason="guarded_turn_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    **telemetry,
                )

            last_guarded_result = guarded_result
            if (
                not isinstance(guarded_result, dict)
                or guarded_result.get("ok") is not True
                or guarded_result.get("permitted") is not True
            ):
                return result(
                    ok=False,
                    completed=False,
                    state="CENTERING_BLOCKED",
                    reason=(
                        guarded_result.get("reason", "guarded_turn_failed")
                        if isinstance(guarded_result, dict)
                        else "guarded_turn_invalid_result"
                    ),
                    stop_reason=(
                        guarded_result.get("reason")
                        if isinstance(guarded_result, dict)
                        else None
                    ),
                    **telemetry,
                )

            centering_completed += 1
            confirmed = self._confirm_target_candidates(target_name)
            if confirmed is None:
                confirmation_status = getattr(
                    self,
                    "_last_target_confirmation_status",
                    "target_lost",
                )
                confirmation_failed = (
                    confirmation_status == "target_reconfirmation_failed"
                )
                return result(
                    ok=False,
                    completed=False,
                    target_found=False,
                    state=(
                        "TARGET_RECONFIRMATION_FAILED"
                        if confirmation_failed
                        else "TARGET_LOST_DURING_CENTERING"
                    ),
                    reason=(
                        "Target candidates were not temporally re-confirmed."
                        if confirmation_failed
                        else "Target was not re-confirmed after centering turn."
                    ),
                    **telemetry,
                )
            promoted = self._promote_confirmed_target(confirmed)
            if promoted is None:
                return result(
                    ok=False,
                    completed=False,
                    target_found=False,
                    state="TARGET_LOST_DURING_CENTERING",
                    reason="Target promotion failed after centering turn.",
                    **telemetry,
                )
            current = promoted

    def _execute_guarded_find_search(self, target_name):
        """Search in independent guarded turn chunks with camera rechecks."""
        base = {
            "ok": False,
            "executed": False,
            "completed": False,
            "behavior": "FIND_OBJECT",
            "target": target_name,
            "target_found": False,
            "search_exhausted": False,
            "turn_chunks_attempted": 0,
            "turn_chunks_completed": 0,
            "maximum_turn_chunks": self.SEARCH_MAX_TURN_CHUNKS,
            "turn_angular_speed": self.SEARCH_TURN_SPEED,
            "turn_duration": self.SEARCH_TURN_SECONDS,
            "search_direction": self.SEARCH_DIRECTION,
            "centering_turn_chunks_attempted": 0,
            "centering_turn_chunks_completed": 0,
            "maximum_centering_turn_chunks": (
                self.FIND_CENTER_MAX_TURN_CHUNKS
            ),
            "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
            "horizontal_error_pixels": None,
            "image_center_x": None,
            "centering_direction": None,
            "last_guarded_turn_result": None,
            "stop_reason": None,
        }

        for _chunk_number in range(self.SEARCH_MAX_TURN_CHUNKS + 1):
            try:
                target = self._get_target_observation(target_name)
            except Exception as exc:
                return dict(
                    base,
                    reason="perception_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            if not isinstance(target, dict):
                return dict(
                    base,
                    state="SEARCH_BLOCKED",
                    reason="invalid_target_observation",
                )

            if self._target_is_fresh_and_acquired(target):
                return self._center_acquired_target(
                    target_name,
                    target,
                    base,
                )

            confirmed = self._confirm_target_candidates(target_name)
            if confirmed is not None:
                promoted = self._promote_confirmed_target(confirmed)
                if promoted is not None:
                    return self._center_acquired_target(
                        target_name,
                        promoted,
                        base,
                    )

            if base["turn_chunks_attempted"] >= self.SEARCH_MAX_TURN_CHUNKS:
                return dict(
                    base,
                    ok=False,
                    state="SEARCH_EXHAUSTED",
                    search_exhausted=True,
                    reason=f"{target_name} not found after bounded search.",
                    target_observation=target,
                )

            session = self._current_lidar_session()
            if session is None:
                return dict(
                    base,
                    state="SEARCH_BLOCKED",
                    reason="LiDAR producer session is unavailable.",
                    target_observation=target,
                )

            base["turn_chunks_attempted"] += 1
            base["executed"] = True
            try:
                guarded_result = self.execute_guarded_turn(
                    self.SEARCH_DIRECTION,
                    self.SEARCH_TURN_SPEED,
                    self.SEARCH_TURN_SECONDS,
                    expected_lidar_session=session,
                )
            except Exception as exc:
                return dict(
                    base,
                    state="SEARCH_BLOCKED",
                    reason="guarded_turn_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            base["last_guarded_turn_result"] = guarded_result

            if not isinstance(guarded_result, dict):
                return dict(
                    base,
                    state="SEARCH_BLOCKED",
                    reason="guarded_turn_invalid_result",
                )

            if (
                guarded_result.get("ok") is not True
                or guarded_result.get("permitted") is not True
            ):
                return dict(
                    base,
                    state="SEARCH_BLOCKED",
                    reason=guarded_result.get(
                        "reason", "guarded_turn_failed"
                    ),
                    stop_reason=guarded_result.get("reason"),
                    target_observation=target,
                )

            base["turn_chunks_completed"] += 1

        return dict(
            base,
            state="SEARCH_EXHAUSTED",
            search_exhausted=True,
            reason=f"{target_name} not found after bounded search.",
        )

    def _get_target_observation(self, target_name):
        """
        Read the newest target observation from the shared World Model.

        The compatibility fallback is retained only for isolated legacy tests
        that construct BehaviorManager without a World Model.
        """
        if (
            self.world_model is not None
            and hasattr(
                self.world_model,
                "find_latest_entity_by_label",
            )
        ):
            return self.world_model.find_latest_entity_by_label(
                target_name,
                max_age_seconds=self.TARGET_MAX_AGE_SECONDS,
                refresh=True,
            )

        if (
            self.vision is not None
            and hasattr(self.vision, "find_target")
        ):
            return self.vision.find_target(target_name)

        raise RuntimeError(
            "No World Model perception source is available."
        )

    def _execute_find_object_cycle(
        self,
        target_name,
        cycle_number,
    ):
        return self._execute_guarded_find_search(target_name)

    @staticmethod
    def _clamp(value, minimum, maximum):
        """
        Clamp a numeric value to an inclusive range.
        """
        return max(
            float(minimum),
            min(float(maximum), float(value)),
        )

    def _reset_follow_servo_state(self):
        """
        Reset transient FOLLOW_PERSON steering-controller state.

        Target identity and predictive tracking remain owned by TargetLock.
        These fields control only smoothing of robot motion commands.
        """
        self._follow_filtered_horizontal_error = None
        self._follow_steering_latch = "CENTER"
        self._follow_previous_angular_command = 0.0
        self._follow_servo_mission_id = None

    def _ensure_follow_servo_state(self):
        """
        Lazily initialize controller state.

        Lazy initialization preserves compatibility with isolated tests that
        construct BehaviorManager through __new__ without calling __init__.
        """
        if not hasattr(
            self,
            "_follow_filtered_horizontal_error",
        ):
            self._follow_filtered_horizontal_error = None

        if not hasattr(
            self,
            "_follow_steering_latch",
        ):
            self._follow_steering_latch = "CENTER"

        if not hasattr(
            self,
            "_follow_previous_angular_command",
        ):
            self._follow_previous_angular_command = 0.0

        if not hasattr(
            self,
            "_follow_servo_mission_id",
        ):
            self._follow_servo_mission_id = None

    def _prepare_follow_servo_mission(
        self,
        mission_id,
    ):
        """
        Reset smoothing history when a different follow mission starts.
        """
        self._ensure_follow_servo_state()

        if self._follow_servo_mission_id == mission_id:
            return False

        self._follow_filtered_horizontal_error = None
        self._follow_steering_latch = "CENTER"
        self._follow_previous_angular_command = 0.0
        self._follow_servo_mission_id = mission_id

        return True

    def _filter_follow_horizontal_error(
        self,
        horizontal_error,
    ):
        """
        Smooth FOLLOW_PERSON horizontal error while preserving
        fast response when the target crosses the image center.

        Small frame-to-frame changes use low-pass filtering.
        A large sign change resets the filter immediately so the
        controller does not continue steering in the old direction.
        """
        horizontal_error = float(horizontal_error)

        previous = (
            self._follow_filtered_horizontal_error
        )

        if previous is None or previous == 0.0:
            # A reset servo starts with no measurement history.
            # Use the first real observation directly rather than
            # blending it with the reset value of zero.
            filtered = horizontal_error
        else:
            sign_changed = (
                previous != 0.0
                and horizontal_error != 0.0
                and (
                    previous < 0.0
                    < horizontal_error
                    or horizontal_error < 0.0
                    < previous
                )
            )

            large_crossing = (
                abs(horizontal_error)
                >= self.CENTER_TOLERANCE_PIXELS
            )

            if sign_changed and large_crossing:
                # This is likely real target motion rather than
                # center-line detection noise. Respond immediately.
                filtered = horizontal_error
            else:
                alpha = float(
                    self.FOLLOW_ERROR_FILTER_ALPHA
                )

                filtered = (
                    alpha * horizontal_error
                    + (1.0 - alpha) * previous
                )

        self._follow_filtered_horizontal_error = (
            filtered
        )

        return filtered

    def _follow_effective_horizontal_error(
        self,
        horizontal_error,
    ):
        """
        Filter FOLLOW_PERSON horizontal error and apply steering
        hysteresis.

        Steering begins outside CENTER_TOLERANCE_PIXELS. Once a
        direction is active, the raw camera measurement must return
        inside FOLLOW_CENTER_EXIT_TOLERANCE_PIXELS before steering is
        released.

        The raw measurement is used for latch transitions so old
        filtered history cannot keep the robot turning after the
        target has returned near image center.
        """
        raw_error = float(horizontal_error)

        filtered_error = (
            self._filter_follow_horizontal_error(
                raw_error
            )
        )

        enter_tolerance = float(
            self.CENTER_TOLERANCE_PIXELS
        )

        exit_tolerance = float(
            self.FOLLOW_CENTER_EXIT_TOLERANCE_PIXELS
        )

        latch = self._follow_steering_latch

        # Once the raw camera measurement is genuinely centered,
        # discard stale filtered steering history. Without this reset,
        # a previous large LEFT or RIGHT error can re-enter a centering
        # state on the next frame even though the target remains near
        # the image center.
        if abs(raw_error) <= exit_tolerance:
            self._follow_steering_latch = "CENTER"
            self._follow_filtered_horizontal_error = raw_error
            return raw_error

        if latch == "LEFT":
            if raw_error > enter_tolerance:
                self._follow_steering_latch = "RIGHT"
                return max(
                    filtered_error,
                    enter_tolerance + 0.001,
                )

            if raw_error >= -exit_tolerance:
                self._follow_steering_latch = "CENTER"
                return raw_error

            return min(
                filtered_error,
                -(enter_tolerance + 0.001),
            )

        if latch == "RIGHT":
            if raw_error < -enter_tolerance:
                self._follow_steering_latch = "LEFT"
                return min(
                    filtered_error,
                    -(enter_tolerance + 0.001),
                )

            if raw_error <= exit_tolerance:
                self._follow_steering_latch = "CENTER"
                return raw_error

            return max(
                filtered_error,
                enter_tolerance + 0.001,
            )

        if filtered_error < -enter_tolerance:
            self._follow_steering_latch = "LEFT"

            return min(
                filtered_error,
                -(enter_tolerance + 0.001),
            )

        if filtered_error > enter_tolerance:
            self._follow_steering_latch = "RIGHT"

            return max(
                filtered_error,
                enter_tolerance + 0.001,
            )

        self._follow_steering_latch = "CENTER"

        return filtered_error

    def _limit_follow_angular_command(
        self,
        angular_z,
    ):
        """
        Limit angular velocity changes between control cycles.

        A requested direction reversal must therefore pass progressively
        through zero instead of changing direction in one camera frame.
        """
        self._ensure_follow_servo_state()

        requested = float(angular_z)
        previous = float(
            self._follow_previous_angular_command
        )
        maximum_step = abs(
            float(self.FOLLOW_MAX_ANGULAR_STEP)
        )

        limited = self._clamp(
            requested,
            previous - maximum_step,
            previous + maximum_step,
        )

        if (
            abs(requested) < 1e-9
            and abs(limited) <= maximum_step
        ):
            limited = 0.0

        self._follow_previous_angular_command = limited

        return limited

    def _follow_turn_speed(self, horizontal_error):
        """
        Convert absolute camera error into a safe proportional turn speed.
        """
        proportional_speed = (
            abs(float(horizontal_error))
            * self.FOLLOW_TURN_KP
        )

        return self._clamp(
            proportional_speed,
            self.FOLLOW_MIN_TURN_SPEED,
            self.FOLLOW_MAX_TURN_SPEED,
        )

    def _follow_approach_turn_speed(
        self,
        horizontal_error,
    ):
        """
        Calculate a small steering correction during forward approach.
        """
        # Camera-left is a negative pixel error, but the Robot Bridge
        # uses positive angular_z for a physical left turn.
        proportional_speed = (
            -float(horizontal_error)
            * self.FOLLOW_APPROACH_TURN_KP
        )

        return self._clamp(
            proportional_speed,
            -self.FOLLOW_MAX_APPROACH_TURN_SPEED,
            self.FOLLOW_MAX_APPROACH_TURN_SPEED,
        )

    def _execute_follow_streaming_motion(
        self,
        linear_x,
        angular_z,
    ):
        """
        Refresh one FOLLOW_PERSON streaming velocity command.

        The preferred client method explicitly requests streaming mode.
        Compatibility fallbacks keep isolated legacy tests functional.
        """
        angular_z = self._limit_follow_angular_command(
            angular_z
        )

        if hasattr(self.robot, "streaming_motion"):
            return self.robot.streaming_motion(
                linear_x=linear_x,
                angular_z=angular_z,
                watchdog_timeout=(
                    self.FOLLOW_STREAM_WATCHDOG_SECONDS
                ),
            )

        if hasattr(self.robot, "motion"):
            try:
                return self.robot.motion(
                    linear_x=linear_x,
                    angular_z=angular_z,
                    duration=0.25,
                    streaming=True,
                    watchdog_timeout=(
                        self.FOLLOW_STREAM_WATCHDOG_SECONDS
                    ),
                )
            except TypeError:
                return self.robot.motion(
                    linear_x=linear_x,
                    angular_z=angular_z,
                    duration=0.25,
                )

        if abs(float(linear_x)) > 0.0:
            return self.robot.move_forward(
                speed=abs(float(linear_x)),
                seconds=0.25,
            )

        if float(angular_z) > 0.0:
            return self.robot.turn_left(
                speed=abs(float(angular_z)),
                seconds=0.25,
            )

        if float(angular_z) < 0.0:
            return self.robot.turn_right(
                speed=abs(float(angular_z)),
                seconds=0.25,
            )

        return self.robot.stop()

    def _execute_bounded_motion(
        self,
        linear_x,
        angular_z,
        seconds,
    ):
        """
        Send one combined bounded motion command when supported.

        The fallback preserves compatibility with older fake clients and
        isolated tests that expose only move_forward/turn_left/turn_right.
        """
        if hasattr(self.robot, "motion"):
            return self.robot.motion(
                linear_x=linear_x,
                angular_z=angular_z,
                duration=seconds,
            )

        if abs(float(linear_x)) > 0.0:
            return self.robot.move_forward(
                speed=abs(float(linear_x)),
                seconds=seconds,
            )

        if float(angular_z) > 0.0:
            return self.robot.turn_left(
                speed=abs(float(angular_z)),
                seconds=seconds,
            )

        if float(angular_z) < 0.0:
            return self.robot.turn_right(
                speed=abs(float(angular_z)),
                seconds=seconds,
            )

        return self.robot.stop()

    def _execute_visual_servo_cycle(
        self,
        behavior,
        target_name,
        cycle_number,
        stop_area,
        search_turn_speed,
        search_turn_seconds,
        center_turn_speed,
        center_turn_seconds,
        forward_speed,
        forward_seconds,
        complete_when_close,
        close_state,
        close_reason,
    ):
        """
        Perform one bounded camera-guided steering cycle.

        This helper is shared by FIND_OBJECT and FOLLOW_PERSON. It never
        loops internally. The CognitiveRuntime decides whether another
        cycle should run.
        """
        if behavior == "FIND_OBJECT":
            return self._execute_guarded_find_search(target_name)

        try:
            if (
                behavior == "FOLLOW_PERSON"
                and self.target_lock is not None
            ):
                target = self.target_lock.resolve(
                    mission_id=self._follow_mission_id,
                    target_label=target_name,
                )
            else:
                target = self._get_target_observation(
                    target_name
                )
        except Exception as exc:
            stop_result = self.robot.stop()

            return {
                "ok": False,
                "executed": True,
                "completed": True,
                "behavior": behavior,
                "target": target_name,
                "state": "PERCEPTION_ERROR",
                "cycle": cycle_number,
                "reason": (
                    f"World Model target query failed: {exc}"
                ),
                "robot_result": stop_result,
            }

        if not target.get("found"):
            commanded_linear_x = 0.0

            if behavior == "FOLLOW_PERSON":
                streaming = True

                recovery_direction = target.get(
                    "recovery_direction"
                )

                predicted_cx = target.get(
                    "predicted_cx"
                )
                predicted_image_width = (
                    target.get("image_width")
                    or target.get(
                        "prediction",
                        {},
                    ).get("image_width")
                    or self.DEFAULT_IMAGE_WIDTH
                )

                if recovery_direction == "CENTER":
                    commanded_angular_z = 0.0
                    state = "HOLDING_PREDICTED_TARGET"
                    reason = (
                        f"{target_name} temporarily lost "
                        "near the predicted image center. "
                        "Holding position while continuing "
                        "to observe."
                    )

                elif recovery_direction in (
                    "LEFT",
                    "RIGHT",
                ):
                    if predicted_cx is not None:
                        predicted_center = (
                            float(predicted_image_width)
                            / 2.0
                        )
                        predicted_error = (
                            float(predicted_cx)
                            - predicted_center
                        )

                        normalized_error = min(
                            1.0,
                            abs(predicted_error)
                            / max(predicted_center, 1.0),
                        )

                        recovery_turn_speed = max(
                            0.12,
                            float(search_turn_speed)
                            * normalized_error,
                        )

                        recovery_turn_speed = min(
                            float(search_turn_speed),
                            recovery_turn_speed,
                        )

                        commanded_angular_z = (
                            -recovery_turn_speed
                            if predicted_error > 0.0
                            else recovery_turn_speed
                        )
                    elif recovery_direction == "RIGHT":
                        commanded_angular_z = -float(
                            search_turn_speed
                        )
                    else:
                        commanded_angular_z = float(
                            search_turn_speed
                        )

                    state = "RECOVERING_TARGET"
                    reason = (
                        f"{target_name} temporarily lost. "
                        "Steering toward the predicted "
                        f"{recovery_direction.lower()} "
                        "location."
                    )

                else:
                    commanded_angular_z = 0.0
                    state = "HOLDING_NO_PREDICTION"
                    reason = (
                        f"{target_name} is not visible and "
                        "no reliable recovery direction is "
                        "available. Holding position while "
                        "continuing to observe."
                    )

                robot_result = (
                    self._execute_follow_streaming_motion(
                        linear_x=commanded_linear_x,
                        angular_z=commanded_angular_z,
                    )
                )

            else:
                streaming = False
                commanded_angular_z = float(
                    search_turn_speed
                )

                robot_result = self.robot.turn_left(
                    commanded_angular_z,
                    search_turn_seconds,
                )

                state = "SEARCHING"
                reason = (
                    f"{target_name} not visible. "
                    "Applying one bounded left search turn."
                )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "completed": False,
                "behavior": behavior,
                "target": target_name,
                "state": state,
                "cycle": cycle_number,
                "reason": reason,
                "commanded_linear_x": commanded_linear_x,
                "commanded_angular_z": commanded_angular_z,
                "streaming": streaming,
                "watchdog_timeout": (
                    self.FOLLOW_STREAM_WATCHDOG_SECONDS
                    if streaming
                    else None
                ),
                "vision_result": target,
                "robot_result": robot_result,
            }

        cx = target.get("cx")
        area = target.get("area")
        image_width = (
            target.get("image_width")
            or self.DEFAULT_IMAGE_WIDTH
        )

        if cx is None:
            stop_result = self.robot.stop()

            return {
                "ok": False,
                "executed": True,
                "completed": True,
                "behavior": behavior,
                "target": target_name,
                "state": "INVALID_DETECTION",
                "cycle": cycle_number,
                "reason": (
                    "Target detection has no horizontal center."
                ),
                "vision_result": target,
                "robot_result": stop_result,
            }

        image_center = float(image_width) / 2.0
        raw_horizontal_error = float(cx) - image_center

        if behavior == "FOLLOW_PERSON":
            self._prepare_follow_servo_mission(
                self._follow_mission_id
            )

            horizontal_error = (
                self._follow_effective_horizontal_error(
                    raw_horizontal_error
                )
            )
        else:
            horizontal_error = raw_horizontal_error

        if (
            area is not None
            and float(area) >= float(stop_area)
            and abs(horizontal_error)
            <= self.CENTER_TOLERANCE_PIXELS
        ):
            robot_result = self.robot.stop()

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "completed": bool(complete_when_close),
                "behavior": behavior,
                "target": target_name,
                "state": close_state,
                "cycle": cycle_number,
                "reason": close_reason,
                "horizontal_error": horizontal_error,
                "vision_result": target,
                "robot_result": robot_result,
            }

        if horizontal_error < -self.CENTER_TOLERANCE_PIXELS:
            if behavior == "FOLLOW_PERSON":
                commanded_turn_speed = (
                    self._follow_turn_speed(
                        horizontal_error
                    )
                )

                commanded_angular_z = (
                    commanded_turn_speed
                )

                robot_result = (
                    self._execute_follow_streaming_motion(
                        linear_x=0.0,
                        angular_z=commanded_angular_z,
                    )
                )
            else:
                commanded_turn_speed = (
                    float(center_turn_speed)
                )

                commanded_angular_z = (
                    commanded_turn_speed
                )

                robot_result = self.robot.turn_left(
                    speed=commanded_turn_speed,
                    seconds=center_turn_seconds,
                )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "completed": False,
                "behavior": behavior,
                "target": target_name,
                "state": "CENTERING_LEFT",
                "cycle": cycle_number,
                "reason": (
                    f"{target_name} is left in the camera image. "
                    "Applying one bounded left correction."
                ),
                "horizontal_error": horizontal_error,
                "commanded_linear_x": 0.0,
                "commanded_angular_z": commanded_angular_z,
                "commanded_duration": None,
                "streaming": (
                    behavior == "FOLLOW_PERSON"
                ),
                "watchdog_timeout": (
                    self.FOLLOW_STREAM_WATCHDOG_SECONDS
                    if behavior == "FOLLOW_PERSON"
                    else None
                ),
                "vision_result": target,
                "robot_result": robot_result,
            }

        if horizontal_error > self.CENTER_TOLERANCE_PIXELS:
            if behavior == "FOLLOW_PERSON":
                commanded_turn_speed = (
                    self._follow_turn_speed(
                        horizontal_error
                    )
                )

                commanded_angular_z = (
                    -commanded_turn_speed
                )

                robot_result = (
                    self._execute_follow_streaming_motion(
                        linear_x=0.0,
                        angular_z=commanded_angular_z,
                    )
                )
            else:
                commanded_turn_speed = (
                    float(center_turn_speed)
                )

                commanded_angular_z = (
                    -commanded_turn_speed
                )

                robot_result = self.robot.turn_right(
                    speed=commanded_turn_speed,
                    seconds=center_turn_seconds,
                )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "completed": False,
                "behavior": behavior,
                "target": target_name,
                "state": "CENTERING_RIGHT",
                "cycle": cycle_number,
                "reason": (
                    f"{target_name} is right in the camera image. "
                    "Applying one bounded right correction."
                ),
                "horizontal_error": horizontal_error,
                "commanded_linear_x": 0.0,
                "commanded_angular_z": commanded_angular_z,
                "commanded_duration": None,
                "streaming": (
                    behavior == "FOLLOW_PERSON"
                ),
                "watchdog_timeout": (
                    self.FOLLOW_STREAM_WATCHDOG_SECONDS
                    if behavior == "FOLLOW_PERSON"
                    else None
                ),
                "vision_result": target,
                "robot_result": robot_result,
            }

        if behavior == "FOLLOW_PERSON":
            commanded_angular_z = (
                self._follow_approach_turn_speed(
                    horizontal_error
                )
            )

            robot_result = (
                self._execute_follow_streaming_motion(
                    linear_x=forward_speed,
                    angular_z=commanded_angular_z,
                )
            )
        else:
            commanded_angular_z = 0.0

            robot_result = self.robot.move_forward(
                speed=forward_speed,
                seconds=forward_seconds,
            )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "completed": False,
            "behavior": behavior,
            "target": target_name,
            "state": "APPROACHING",
            "cycle": cycle_number,
            "reason": (
                f"{target_name} is centered. "
                "Executing one bounded approach step."
            ),
            "horizontal_error": horizontal_error,
            "commanded_linear_x": float(
                forward_speed
            ),
            "commanded_angular_z": (
                commanded_angular_z
            ),
            "commanded_duration": (
                None
                if behavior == "FOLLOW_PERSON"
                else forward_seconds
            ),
            "streaming": (
                behavior == "FOLLOW_PERSON"
            ),
            "watchdog_timeout": (
                self.FOLLOW_STREAM_WATCHDOG_SECONDS
                if behavior == "FOLLOW_PERSON"
                else None
            ),
            "vision_result": target,
            "robot_result": robot_result,
        }

    def _execute_return_home(self, mission):
        return {
            "ok": True,
            "executed": False,
            "behavior": "RETURN_HOME",
            "reason": "Navigation is not implemented yet.",
        }

    def _execute_describe_scene(self, mission):
        return {
            "ok": True,
            "executed": False,
            "behavior": "DESCRIBE_SCENE",
            "reason": "Information-only mission.",
        }
