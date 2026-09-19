import copy
import math
import threading
import time
from datetime import datetime, timezone

from robot_bridge.client import RobotBridgeClient
from robot_bridge.forward_interlock import (
    FORWARD_POLICY_STRICT,
    FORWARD_POLICY_TARGET_APPROACH,
)
from guarded_turn_policy import validate_guarded_turn
from local_obstacle_policy import recommend_local_avoidance
from target_lock import TargetLock


class _GuardedTurnMonitor:
    """Monitor one explicit bounded turn without owning transport locks."""

    INTERVAL_SECONDS = 0.05
    # A short producer scheduling gap is tolerated only after the turn has
    # already passed its initial LiDAR validation.  This does not change the
    # global LiDAR freshness contract or any forward-motion authorization.
    TRANSIENT_STALE_GRACE_SECONDS = 0.15
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
        target_directed=None,
    ):
        self.world_model = world_model
        self.robot = robot
        self.direction = direction
        self.angular_speed = angular_speed
        self.duration = duration
        self.expected_session = expected_session
        self.generation = generation
        self.now = now
        self.target_directed = bool(target_directed)
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
        self._first_invalidating_validation = None
        self._reason = initial_validation.get("reason")
        self._validation = dict(initial_validation)
        self._last_stop_result = None
        self._last_stop_error = None
        self._last_stop_error_type = None
        self._stop_count = 0
        self._stop_events = []
        self._stale_started_monotonic = None
        self._transient_stale_observed = False
        self._transient_stale_recovered = False
        self._transient_stale_max_duration_seconds = 0.0

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
            target_directed=self.target_directed,
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
                stale_transient = False
                if validation.get("reason") == "stale":
                    if self._stale_started_monotonic is None:
                        self._stale_started_monotonic = now_monotonic
                        self._transient_stale_observed = True
                    stale_duration = (
                        now_monotonic - self._stale_started_monotonic
                    )
                    self._transient_stale_max_duration_seconds = max(
                        self._transient_stale_max_duration_seconds,
                        stale_duration,
                    )
                    stale_transient = bool(
                        not deadline_expired
                        and stale_duration
                        <= self.TRANSIENT_STALE_GRACE_SECONDS
                    )
                elif (
                    validation.get("permitted")
                    and self._stale_started_monotonic is not None
                ):
                    self._transient_stale_recovered = True
                    self._stale_started_monotonic = None
                if deadline_expired:
                    self._window_complete = True
                    self._physical_deadline_reached = True
                if stale_transient:
                    # Keep the initial successful validation reason while a
                    # single continuous stale interval is within grace.
                    pass
                elif not validation.get("permitted"):
                    if not self._invalidated:
                        self._first_invalidating_validation = dict(validation)
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
            if self._first_invalidating_validation is not None:
                event["monitor_validation"] = dict(self._first_invalidating_validation)
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
                "monitor_validation": (
                    dict(self._first_invalidating_validation)
                    if self._first_invalidating_validation is not None else None
                ),
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
                "transient_stale_observed": self._transient_stale_observed,
                "transient_stale_recovered": self._transient_stale_recovered,
                "transient_stale_max_duration_seconds": (
                    self._transient_stale_max_duration_seconds
                ),
            }


class _SemanticPreempted(Exception):
    """Unwind FIND_OBJECT without promoting late perception or issuing motion."""


class BehaviorManager:
    SEARCH_TURN_SPEED = 0.30
    SEARCH_TURN_SECONDS = 1.0
    SEARCH_MAX_TURN_CHUNKS = 3
    SEARCH_DIRECTION = "LEFT"
    TARGET_CONFIRMATION_MAX_FRAMES = 3
    TARGET_CONFIRMATION_MIN_SUPPORT = 2
    TARGET_CONFIRMATION_WINDOW_SECONDS = 0.90
    TARGET_CONFIRMATION_POLL_SECONDS = 0.05
    FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS = 1.50
    FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS = (
        FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS
    )
    FIND_STALE_RECOVERY_WINDOW_SECONDS = 0.50

    FIND_CENTER_TOLERANCE_PIXELS = 50.0
    FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS = 5
    FIND_CENTER_MIN_PROGRESS_PIXELS = 2.0
    FIND_CENTER_TURN_SPEED = 0.20
    FIND_CENTER_TURN_SECONDS = 0.50
    FIND_AVOIDANCE_MAX_MANEUVERS = 1
    FIND_AVOIDANCE_MAX_TURN_CHUNKS = 3
    FIND_AVOIDANCE_TURN_SPEED = FIND_CENTER_TURN_SPEED
    FIND_AVOIDANCE_TURN_SECONDS = FIND_CENTER_TURN_SECONDS

    CENTER_TURN_SPEED = 0.60
    CENTER_TURN_SECONDS = 0.40

    FIND_FORWARD_SPEED = 0.08
    FIND_FORWARD_SECONDS = 0.80
    FIND_APPROACH_FORWARD_SPEED = 0.08
    FIND_APPROACH_FORWARD_SECONDS = 0.50
    FIND_APPROACH_MAX_CHUNKS = 4
    FIND_ARRIVAL_AREA = 75000.0
    FIND_ARRIVAL_AREA_FRACTION = FIND_ARRIVAL_AREA / (640.0 * 480.0)
    FIND_ARRIVAL_SUPPORT_REQUIRED = 2

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
        semantic_vision=None,
    ):
        self.robot = robot_client or RobotBridgeClient()
        self.vision = vision_adapter
        self.world_model = (
            world_model
            or getattr(vision_adapter, "world_model", None)
        )

        self.semantic_vision = semantic_vision
        self._semantic_episode = None

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
        # Runtime-owned execution-generation hook.  A missing hook preserves
        # direct/offline BehaviorManager use; CognitiveRuntime installs it so
        # STOP can invalidate a multi-action FIND_OBJECT execution.
        self.execution_authorization_provider = None

    def _publish_tracking_state(self, result):
        callback = getattr(self, "tracking_state_callback", None)
        if not callable(callback):
            return
        try:
            callback(result)
        except Exception:
            # Telemetry must never affect guarded behavior or motion safety.
            return

    def _execution_is_current(self):
        """Return whether the runtime still authorizes this execution."""
        provider = getattr(self, "execution_authorization_provider", None)
        if not callable(provider):
            return True
        try:
            return bool(provider())
        except Exception:
            return False

    SEMANTIC_FRAME_TIMEOUT_SECONDS = 5.0
    SEMANTIC_IMAGE_TIMEOUT_SECONDS = 13.0

    def _semantic_motion_idle(self):
        # These are local snapshots; never query the Robot Bridge over HTTP.
        with self._guarded_turn_slot_lock:
            if self._guarded_turn_owner_generation is not None:
                return False
        interlock = getattr(self.robot, "forward_interlock", None)
        if interlock is None:
            return True
        try:
            state = interlock.status()
            return (
                isinstance(state, dict)
                and state.get("active_forward") is False
                and state.get("pending_forward") is False
            )
        except Exception:
            return False

    def _semantic_check_current(self, episode):
        if self._semantic_episode is not episode or not self._execution_is_current():
            raise _SemanticPreempted()

    def _semantic_bounded_call(self, callback, timeout, episode):
        """Bound caller latency; late I/O can never schedule followup work.

        Each worker owns only one I/O operation, never motion or promotion.
        The helper additionally applies transport timeouts and disables retries.
        """
        done = threading.Event()
        outcome = []
        deadline = time.monotonic() + timeout

        def run():
            try:
                self._semantic_check_current(episode)
                if time.monotonic() >= deadline:
                    raise TimeoutError("semantic_deadline_expired")
                outcome.append((True, callback()))
            except Exception as exc:
                outcome.append((False, exc))
            finally:
                done.set()

        self._semantic_check_current(episode)
        threading.Thread(target=run, daemon=True, name="semantic-one-shot").start()
        while True:
            self._semantic_check_current(episode)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("semantic_deadline_expired")
            if done.wait(min(remaining, 0.02)):
                self._semantic_check_current(episode)
                if time.monotonic() >= deadline:
                    raise TimeoutError("semantic_deadline_expired")
                ok, value = outcome[0]
                if not ok:
                    raise value
                return value

    def _confirm_find_target_with_semantic(
        self, target_name, *, semantic_turn_budget=1, **kwargs
    ):
        """Return only normal YOLO confirmation, optionally after one hint.

        The episode spans the entire FIND_OBJECT execution (including approach).
        No recovery path resets its request or turn budget.
        """
        confirmed, status, diagnostics = self._confirm_target_candidates_with_status(
            target_name, **kwargs
        )
        episode = self._semantic_episode
        if episode is None:
            return confirmed, status, diagnostics
        self._semantic_check_current(episode)
        if confirmed is not None or episode["used"] or self.semantic_vision is None:
            return confirmed, status, diagnostics
        label = str(target_name or "").strip().lower()
        if not label or not self._semantic_motion_idle():
            return confirmed, status, diagnostics
        if any(
            attempt.get("camera_running") is False
            for attempt in diagnostics.get("attempts", [])
            if isinstance(attempt, dict)
        ):
            return confirmed, status, diagnostics

        # Reserve before any I/O, including unsuccessful frame retrieval.
        episode["used"] = True
        telemetry = episode["telemetry"]
        telemetry["semantic_reacquisition_attempted"] = True
        try:
            frame = self._semantic_bounded_call(
                self.semantic_vision.fetch_frame,
                self.SEMANTIC_FRAME_TIMEOUT_SECONDS, episode,
            )
            self._semantic_check_current(episode)
            if not self._semantic_motion_idle():
                raise ValueError("semantic_motion_busy")
            semantic = self._semantic_bounded_call(
                lambda: self.semantic_vision.describe(label, frame),
                self.SEMANTIC_IMAGE_TIMEOUT_SECONDS, episode,
            )
            self._semantic_check_current(episode)
            telemetry.update(
                semantic_reacquisition_completed=True,
                semantic_reacquisition_found=semantic["found"],
                semantic_reacquisition_direction=semantic["coarse_direction"],
                semantic_reacquisition_result=semantic,
            )
            direction = semantic["coarse_direction"]
            if semantic["found"] is True and direction in {"LEFT", "RIGHT"}:
                if semantic_turn_budget <= 0:
                    raise ValueError("semantic_turn_budget_exhausted")
                session = self._current_lidar_session()
                if session is None or not self._semantic_motion_idle():
                    raise ValueError("semantic_turn_unavailable")
                self._semantic_check_current(episode)
                # One call site, reached at most once per reserved episode.
                episode["turn_attempts"] += 1
                turn = self._execute_target_directed_turn(
                    direction, self.FIND_CENTER_TURN_SPEED,
                    self.FIND_CENTER_TURN_SECONDS,
                    expected_lidar_session=session,
                )
                self._semantic_check_current(episode)
                telemetry["semantic_reacquisition_turn_result"] = turn
                if not isinstance(turn, dict) or (
                    turn.get("ok") is not True or turn.get("permitted") is not True
                ):
                    if isinstance(turn, dict):
                        telemetry["semantic_reacquisition_failure_reason"] = turn.get("reason")
                        telemetry["semantic_reacquisition_monitor_reason"] = turn.get("monitor_reason")
                        if turn.get("monitor_validation") is not None:
                            telemetry["semantic_reacquisition_monitor_validation"] = dict(turn["monitor_validation"])
                    raise ValueError("semantic_guarded_turn_denied")
                episode["turn_completions"] += 1
        except _SemanticPreempted:
            raise
        except Exception as exc:
            self._semantic_check_current(episode)
            telemetry["semantic_reacquisition_error"] = type(exc).__name__
            return None, status, diagnostics

        # Even CENTER/UNKNOWN/absent hints require a new temporal YOLO window.
        # Gemini geometry never enters this return value or the World Model.
        self._semantic_check_current(episode)
        fresh_kwargs = dict(kwargs, minimum_timestamp=datetime.now(timezone.utc).isoformat())
        confirmed, status, diagnostics = self._confirm_target_candidates_with_status(
            label, **fresh_kwargs
        )
        self._semantic_check_current(episode)
        telemetry["semantic_reacquisition_post_yolo_status"] = status
        telemetry["semantic_reacquisition_post_yolo_diagnostics"] = diagnostics
        return confirmed, status, diagnostics

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

    def _execute_target_directed_turn(
        self, direction, angular_speed, duration, *, expected_lidar_session,
    ):
        previous = getattr(self, "_target_directed_turn_context", False)
        self._target_directed_turn_context = True
        try:
            return self.execute_guarded_turn(
                direction,
                angular_speed,
                duration,
                expected_lidar_session=expected_lidar_session,
            )
        finally:
            self._target_directed_turn_context = previous

    def execute_guarded_turn(
        self,
        direction,
        angular_speed,
        duration,
        *,
        expected_lidar_session,
        now=None,
        target_directed=None,
    ):
        """Validate and execute one explicit bounded turn request.

        This is an advisory caller's execution boundary, not an autonomous
        behavior.  The transient World Model snapshot is validated before a
        single bounded angular-only Robot Bridge request is sent.  STOP
        remains independent and unconditional through ``execute``.
        """
        if target_directed is None:
            target_directed = bool(
                getattr(self, "_target_directed_turn_context", False)
            )
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
            target_directed=target_directed,
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
                target_directed=target_directed,
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
        Execute one bounded FIND_OBJECT acquisition and approach sequence.

        The sequence itself owns its bounded forward-step limit and returns a
        terminal result after completion or any safety/perception failure.
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

        episode = {
            "used": False,
            "turn_attempts": 0,
            "turn_completions": 0,
            "telemetry": {
                "semantic_reacquisition_attempted": False,
                "semantic_reacquisition_completed": False,
            },
        }
        self._semantic_episode = episode
        try:
            outcome = self._execute_guarded_find_search(target_name)
        except _SemanticPreempted:
            outcome = {
                "ok": False, "completed": True, "target_found": False,
                "behavior": "FIND_OBJECT", "target": target_name,
                "state": "PREEMPTED",
                "reason": "FIND_OBJECT execution was preempted.",
            }
        finally:
            self._semantic_episode = None
        if episode["used"] and outcome.get("ok") is not True:
            outcome["completed"] = True
        outcome.update(episode["telemetry"])
        return outcome

    def preview_find_object(self, target_name):
        """Preview FIND_OBJECT perception without promotion or motion."""
        normalized_target = str(target_name or "").strip().lower()
        base = {
            "ok": False,
            "preview": True,
            "authoritative": False,
            "executed": False,
            "completed": True,
            "behavior": "FIND_OBJECT",
            "state": "PREVIEW",
            "target": normalized_target,
            "target_found": False,
        }
        if not normalized_target:
            return dict(
                base,
                reason="FIND_OBJECT preview requires a target.",
            )

        observation = None
        if (
            self.world_model is not None
            and hasattr(self.world_model, "find_latest_entity_by_label")
        ):
            try:
                observation = self.world_model.find_latest_entity_by_label(
                    normalized_target,
                    max_age_seconds=self.TARGET_MAX_AGE_SECONDS,
                    refresh=False,
                )
            except Exception:
                observation = None

        if self._target_is_fresh_and_acquired(observation):
            return self._build_find_object_preview(
                normalized_target,
                observation,
                source="world_model",
                authoritative=True,
            )

        confirmed, confirmation_status, confirmation_diagnostics = (
            self._confirm_target_candidates_with_status(
                normalized_target,
                return_diagnostics=True,
                confirmation_window_seconds=(
                    self.FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS
                ),
            )
        )
        if confirmed is None:
            return dict(
                base,
                reason=(
                    "Target candidates were not temporally confirmed."
                    if confirmation_status == "target_reconfirmation_failed"
                    else "No actionable target candidate is available."
                ),
                confirmation_diagnostics=confirmation_diagnostics,
            )

        # Deliberately do not call _promote_confirmed_target() here. The
        # candidate remains diagnostic and non-authoritative for preview.
        return self._build_find_object_preview(
            normalized_target,
            confirmed,
            source="vision_candidate",
            authoritative=False,
            confirmation_diagnostics=confirmation_diagnostics,
        )

    def _build_find_object_preview(
        self,
        target_name,
        observation,
        *,
        source,
        authoritative,
        confirmation_diagnostics=None,
    ):
        cx = observation.get("cx")
        cy = observation.get("cy")
        area = observation.get("area")
        image_width = observation.get("image_width")
        image_height = observation.get("image_height")
        image_center_x = (
            float(image_width) / 2.0
            if isinstance(image_width, (int, float))
            and not isinstance(image_width, bool)
            and math.isfinite(image_width)
            else None
        )
        horizontal_error = (
            float(cx) - image_center_x
            if image_center_x is not None
            and isinstance(cx, (int, float))
            and not isinstance(cx, bool)
            and math.isfinite(cx)
            else None
        )
        if horizontal_error is None:
            direction = None
        elif horizontal_error < -self.FIND_CENTER_TOLERANCE_PIXELS:
            direction = "LEFT"
        elif horizontal_error > self.FIND_CENTER_TOLERANCE_PIXELS:
            direction = "RIGHT"
        else:
            direction = "CENTERED"

        result = {
            "ok": True,
            "preview": True,
            "authoritative": bool(authoritative),
            "source": source,
            "executed": False,
            "completed": True,
            "behavior": "FIND_OBJECT",
            "state": "PREVIEW",
            "target": target_name,
            "target_found": True,
            "target_label": observation.get("label", target_name),
            "target_confidence": observation.get("confidence"),
            "target_center_x": cx,
            "target_center_y": cy,
            "target_area": area,
            "image_width": image_width,
            "image_height": image_height,
            "image_center_x": image_center_x,
            "horizontal_error": horizontal_error,
            "horizontal_error_pixels": horizontal_error,
            "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
            "steering_direction": direction,
            "bbox": observation.get("bbox"),
            "target_observation": observation,
        }
        if confirmation_diagnostics is not None:
            result["confirmation_diagnostics"] = confirmation_diagnostics
        return result

    def _current_lidar_session(self):
        provider = getattr(self, "lidar_session_provider", None)
        if callable(provider):
            try:
                return provider()
            except Exception:
                return None
        return getattr(self, "lidar_session", None)

    def _wait_for_find_stale_recovery(self, interlock, expected_session):
        """Wait briefly for a stopped FIND_OBJECT forward's LiDAR to recover."""
        deadline = time.monotonic() + self.FIND_STALE_RECOVERY_WINDOW_SECONDS

        while True:
            if not self._execution_is_current():
                return False, "preempted"
            if self._current_lidar_session() != expected_session:
                return False, "producer_session_mismatch"

            status = getattr(interlock, "status", None)
            refresh = getattr(interlock, "refresh", None)
            if not callable(status) or not callable(refresh):
                return False, "forward_interlock_status_unavailable"
            before = status()
            if (
                not isinstance(before, dict)
                or before.get("active_forward") is not False
                or before.get("pending_forward") is not False
            ):
                return False, "forward_motion_not_stopped"

            try:
                permitted, interlock_reason = refresh()
                lidar = self.world_model.get_lidar_obstacles(
                    expected_session=expected_session
                )
            except Exception:
                return False, "stale_recovery_state_unavailable"

            after = status()
            if (
                not isinstance(after, dict)
                or after.get("active_forward") is not False
                or after.get("pending_forward") is not False
            ):
                return False, "forward_motion_not_stopped"
            if self._current_lidar_session() != expected_session:
                return False, "producer_session_mismatch"

            front = (
                lidar.get("sectors", {}).get("front", {})
                if isinstance(lidar, dict)
                else {}
            )
            recovered = bool(
                isinstance(lidar, dict)
                and lidar.get("producer_session") == expected_session
                and lidar.get("available") is True
                and lidar.get("valid") is True
                and lidar.get("reason") == "fresh"
                and isinstance(front, dict)
                and front.get("state") == "CLEAR"
                and permitted is True
                and interlock_reason == "fresh_clear"
                and after.get("producer_session") == expected_session
                and after.get("forward_permitted") is True
                and after.get("reason") == "fresh_clear"
                and after.get("front_state") == "CLEAR"
            )
            if recovered:
                return True, "fresh_clear"

            lidar_reason = (
                lidar.get("reason") if isinstance(lidar, dict) else None
            )
            stale = bool(
                lidar_reason == "stale"
                or (
                    lidar_reason == "fresh"
                    and interlock_reason in {
                        "stale", "stale_lidar", "not_fresh"
                    }
                )
            )
            if not stale:
                return False, interlock_reason or lidar_reason or "unknown"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, "stale_recovery_timeout"
            time.sleep(min(self.TARGET_CONFIRMATION_POLL_SECONDS, remaining))

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
        return cls._target_observation_match_details(first, second)["matched"]

    @classmethod
    def _target_observation_match_details(cls, first, second):
        """Explain the existing target-association decision."""
        first_label = first.get("label") if isinstance(first, dict) else None
        second_label = second.get("label") if isinstance(second, dict) else None
        details = {
            "matched": False,
            "rule": None,
            "iou": cls._target_bbox_iou(first, second),
            "center_distance": None,
            "area_ratio": None,
            "intersection_over_smaller": cls._target_bbox_intersection_over_smaller(
                first, second
            ),
        }
        if (
            not isinstance(first_label, str)
            or not isinstance(second_label, str)
            or first_label.casefold() != second_label.casefold()
        ):
            details["rejection_reason"] = "label_mismatch"
            return details

        if details["iou"] >= 0.50:
            details.update(matched=True, rule="iou")
            return details

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
            details["rejection_reason"] = "invalid_geometry"
            return details

        # FIND_OBJECT turns on horizontal image error.  Associate detector
        # shape variants by horizontal center so changes in box height do not
        # split one visible target into separate temporal clusters.
        center_distance = abs(float(first["cx"]) - float(second["cx"]))
        area_ratio = max(float(first["area"]), float(second["area"])) / min(
            float(first["area"]), float(second["area"])
        )
        details["center_distance"] = center_distance
        details["area_ratio"] = area_ratio
        if center_distance > 60.0:
            details["rejection_reason"] = "center_distance"
            return details

        if area_ratio <= 2.0:
            details.update(matched=True, rule="center_and_area_ratio")
            return details

        if details["intersection_over_smaller"] >= 0.50:
            details.update(matched=True, rule="center_and_intersection_over_smaller")
            return details
        details["rejection_reason"] = "geometric_thresholds"
        return details

    @staticmethod
    def _vision_timestamp_is_newer(timestamp, minimum_timestamp):
        if minimum_timestamp is None:
            return True
        if not isinstance(timestamp, str) or not isinstance(
            minimum_timestamp, str
        ):
            return False
        timestamp = timestamp.strip()
        minimum_timestamp = minimum_timestamp.strip()
        if not timestamp or not minimum_timestamp or timestamp == minimum_timestamp:
            return False

        def parse(value):
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                return None

        parsed_timestamp = parse(timestamp)
        parsed_minimum = parse(minimum_timestamp)
        if parsed_timestamp is not None and parsed_minimum is not None:
            if (parsed_timestamp.tzinfo is None) == (
                parsed_minimum.tzinfo is None
            ):
                return parsed_timestamp > parsed_minimum
        return timestamp > minimum_timestamp

    @staticmethod
    def _vision_timestamp_is_iso(timestamp):
        if not isinstance(timestamp, str) or not timestamp.strip():
            return False
        value = timestamp.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            datetime.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _confirm_target_candidates_with_status(
        self,
        target_name,
        *,
        minimum_timestamp=None,
        return_diagnostics=False,
        confirmation_window_seconds=None,
    ):
        """Confirm a target and optionally return bounded diagnostics."""
        started = time.monotonic()
        confirmation_window = (
            self.TARGET_CONFIRMATION_WINDOW_SECONDS
            if confirmation_window_seconds is None
            else float(confirmation_window_seconds)
        )
        diagnostics = {
            "confirmation_status": None,
            "elapsed_seconds": 0.0,
            "fetch_attempts": 0,
            "distinct_fresh_timestamps": 0,
            "evidence_frames_evaluated": 0,
            "actionable_frames": 0,
            "maximum_frames": self.TARGET_CONFIRMATION_MAX_FRAMES,
            "minimum_support": self.TARGET_CONFIRMATION_MIN_SUPPORT,
            "confirmation_window_seconds": confirmation_window,
            "poll_seconds": self.TARGET_CONFIRMATION_POLL_SECONDS,
            "minimum_timestamp": minimum_timestamp,
            "attempts": [],
            "terminal_reason": None,
            "qualified_support_reached": False,
            "qualified_fallback_used": False,
            "winning_cluster_support": 0,
            "winning_cluster_arrival_area_fractions": [],
        }

        def finish(candidate, status, terminal_reason=None):
            diagnostics["confirmation_status"] = status
            diagnostics["elapsed_seconds"] = round(
                max(0.0, time.monotonic() - started),
                6,
            )
            diagnostics["distinct_fresh_timestamps"] = len(
                fresh_timestamps
            )
            diagnostics["terminal_reason"] = terminal_reason or (
                diagnostics["attempts"][-1].get("fetch_error", {}).get(
                    "type"
                )
                if diagnostics["attempts"]
                and diagnostics["attempts"][-1].get("fetch_error")
                else (
                    "support_reached"
                    if status == "target_confirmed"
                    else (
                        "insufficient_temporal_or_geometric_support"
                        if status == "target_reconfirmation_failed"
                        else status
                    )
                )
            )
            if isinstance(candidate, dict):
                candidate = dict(candidate)
                candidate["_find_confirmation_diagnostics"] = copy.deepcopy(
                    diagnostics
                )
            if return_diagnostics:
                return candidate, status, diagnostics
            return candidate, status

        fetch = getattr(self.vision, "fetch_target_candidates", None)
        normalize = getattr(self.vision, "normalize_detection", None)
        if not callable(fetch) or not callable(normalize):
            fresh_timestamps = set()
            return finish(None, "target_lost")

        seen_timestamps = set()
        fresh_timestamps = set()
        clusters = []
        actionable_candidate_seen = False
        qualified_support_reached = False

        def eligible_clusters():
            return [
                cluster
                for cluster in clusters
                if len(cluster["timestamps"])
                >= self.TARGET_CONFIRMATION_MIN_SUPPORT
            ]

        def select_winner(eligible):
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

            winning = max(eligible, key=cluster_key)
            selected = max(
                winning["observations"],
                key=lambda item: (
                    float(item.get("confidence") or 0.0),
                    float(item.get("area") or 0.0),
                ),
            )
            arrival_evidence = self._find_arrival_evidence_for_cluster(
                winning["observations"], winning["timestamps"]
            )
            diagnostics["winning_cluster_support"] = len(
                winning["timestamps"]
            )
            diagnostics["winning_cluster_arrival_area_fractions"] = list(
                arrival_evidence["arrival_area_fractions"]
            )
            selected = dict(selected)
            selected["_find_arrival_evidence"] = arrival_evidence
            return selected

        def fallback_after_optional_failure():
            nonlocal qualified_support_reached
            eligible = eligible_clusters()
            if not qualified_support_reached or not eligible:
                return None
            diagnostics["qualified_fallback_used"] = True
            return finish(
                select_winner(eligible),
                "target_confirmed",
                "qualified_support_preserved_after_fetch_error",
            )

        while (
            diagnostics["evidence_frames_evaluated"]
            < self.TARGET_CONFIRMATION_MAX_FRAMES
            and time.monotonic() - started
            <= confirmation_window
        ):
            diagnostics["fetch_attempts"] += 1
            attempt = {
                "attempt_index": diagnostics["fetch_attempts"],
                "elapsed_seconds": round(
                    max(0.0, time.monotonic() - started),
                    6,
                ),
                "response_timestamp": None,
                "minimum_timestamp": minimum_timestamp,
                "timestamp_newer_than_cutoff": None,
                "before_cutoff": False,
                "duplicate_timestamp": False,
                "camera_running": None,
                "raw_candidate_count": 0,
                "actionable_candidate_count": 0,
                "evidence_frame_evaluated": False,
                "actionable_candidates": [],
                "association_outcomes": [],
                "cluster_count": len(clusters),
                "cluster_support_counts": [
                    len(cluster["timestamps"]) for cluster in clusters
                ],
                "cluster_reached_support": False,
            }
            diagnostics["attempts"].append(attempt)
            try:
                payload = fetch(target_name)
            except Exception as exc:
                attempt["fetch_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                fallback = fallback_after_optional_failure()
                if fallback is not None:
                    return fallback
                return finish(
                    None,
                    "target_reconfirmation_failed"
                    if actionable_candidate_seen else "target_lost",
                )

            if not isinstance(payload, dict):
                attempt["fetch_error"] = {
                    "type": "invalid_payload",
                    "message": "candidate response was not an object",
                }
                fallback = fallback_after_optional_failure()
                if fallback is not None:
                    return fallback
                return finish(
                    None,
                    "target_reconfirmation_failed"
                    if actionable_candidate_seen else "target_lost",
                )
            attempt["response_timestamp"] = payload.get("timestamp")
            attempt["camera_running"] = payload.get("camera_running")
            if payload.get("camera_running") is not True:
                attempt["fetch_error"] = {
                    "type": "camera_not_running",
                    "message": "camera_running was not true",
                }
                return finish(
                    None,
                    "target_reconfirmation_failed"
                    if actionable_candidate_seen else "target_lost",
                )

            timestamp = payload.get("timestamp")
            if not isinstance(timestamp, str) or not timestamp.strip():
                attempt["fetch_error"] = {
                    "type": "invalid_timestamp",
                    "message": "candidate response had no timestamp",
                }
                fallback = fallback_after_optional_failure()
                if fallback is not None:
                    return fallback
                return finish(
                    None,
                    "target_reconfirmation_failed"
                    if actionable_candidate_seen else "target_lost",
                )
            if timestamp in seen_timestamps:
                attempt["duplicate_timestamp"] = True
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue

            seen_timestamps.add(timestamp)
            timestamp_newer = self._vision_timestamp_is_newer(
                timestamp,
                minimum_timestamp,
            )
            attempt["timestamp_newer_than_cutoff"] = timestamp_newer
            attempt["before_cutoff"] = not timestamp_newer
            if not timestamp_newer:
                continue
            fresh_timestamps.add(timestamp)
            raw_detections = payload.get("detections")
            if not isinstance(raw_detections, list):
                attempt["fetch_error"] = {
                    "type": "invalid_detections",
                    "message": "detections was not a list",
                }
                continue
            attempt["raw_candidate_count"] = len(raw_detections)

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
                    actionable_candidate_seen = True
                    observations.append(normalized)
                    attempt["actionable_candidate_count"] += 1
                    attempt["actionable_candidates"].append(
                        self._target_diagnostic_summary(normalized)
                    )

            if observations:
                diagnostics["actionable_frames"] += 1
                diagnostics["evidence_frames_evaluated"] += 1
                attempt["evidence_frame_evaluated"] = True

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
                    for member in cluster["observations"]:
                        match_details = self._target_observation_match_details(
                            observation, member
                        )
                        attempt["association_outcomes"].append({
                            "observation": self._target_diagnostic_summary(
                                observation
                            ),
                            "member": self._target_diagnostic_summary(member),
                            **match_details,
                        })
                        if match_details["matched"]:
                            matching.append((1.0, index))
                            break

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

            attempt["cluster_count"] = len(clusters)
            attempt["cluster_support_counts"] = [
                len(cluster["timestamps"]) for cluster in clusters
            ]
            attempt["cluster_reached_support"] = any(
                support >= self.TARGET_CONFIRMATION_MIN_SUPPORT
                for support in attempt["cluster_support_counts"]
            )
            if attempt["cluster_reached_support"]:
                qualified_support_reached = True
                diagnostics["qualified_support_reached"] = True

        eligible = eligible_clusters()
        if not eligible:
            return finish(None, (
                "target_reconfirmation_failed"
                if actionable_candidate_seen else "target_lost"
            ))

        return finish(select_winner(eligible), "target_confirmed")

    @staticmethod
    def _target_diagnostic_summary(observation):
        """Return bounded, JSON-safe geometry for confirmation diagnostics."""
        def number(value):
            return (
                value
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                else None
            )

        bbox = observation.get("bbox")
        safe_bbox = None
        if isinstance(bbox, dict):
            safe_bbox = {
                key: number(bbox.get(key))
                for key in ("x1", "y1", "x2", "y2")
            }
            if any(value is None for value in safe_bbox.values()):
                safe_bbox = None
        return {
            "label": observation.get("label"),
            "confidence": number(observation.get("confidence")),
            "center_x": number(observation.get("cx")),
            "center_y": number(observation.get("cy")),
            "area": number(observation.get("area")),
            "bbox": safe_bbox,
        }

    @staticmethod
    def _find_target_area_fraction(observation):
        """Return finite positive bbox area as an image fraction, or None."""
        if not isinstance(observation, dict):
            return None
        values = (
            observation.get("area"),
            observation.get("image_width"),
            observation.get("image_height"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0.0
            for value in values
        ):
            return None
        area, width, height = (float(value) for value in values)
        fraction = area / (width * height)
        return (
            fraction
            if math.isfinite(fraction) and 0.0 < fraction <= 1.0
            else None
        )

    def _find_arrival_evidence_for_cluster(self, observations, timestamps):
        """Count at most one qualifying arrival observation per timestamp."""
        cluster_timestamps = {
            value
            for value in timestamps
            if isinstance(value, str) and value.strip()
        }
        qualifying_by_timestamp = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            timestamp = observation.get("source_timestamp")
            if timestamp not in cluster_timestamps:
                continue
            fraction = self._find_target_area_fraction(observation)
            if (
                fraction is not None
                and fraction >= self.FIND_ARRIVAL_AREA_FRACTION
            ):
                qualifying_by_timestamp[timestamp] = max(
                    fraction,
                    qualifying_by_timestamp.get(timestamp, 0.0),
                )
        support_timestamps = sorted(qualifying_by_timestamp)
        return {
            "arrival_area_fraction_threshold": (
                self.FIND_ARRIVAL_AREA_FRACTION
            ),
            "arrival_support_required": self.FIND_ARRIVAL_SUPPORT_REQUIRED,
            "arrival_support_count": len(support_timestamps),
            "arrival_support_timestamps": support_timestamps,
            "arrival_area_fractions": [
                qualifying_by_timestamp[timestamp]
                for timestamp in support_timestamps
            ],
        }

    def _confirm_target_candidates(self, target_name):
        """Compatibility wrapper for production FIND_OBJECT execution."""
        confirmed, status = self._confirm_target_candidates_with_status(
            target_name,
            confirmation_window_seconds=(
                self.FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS
            ),
        )
        self._last_target_confirmation_status = status
        return confirmed

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
        if not self._target_is_fresh_and_acquired(promoted):
            return None
        for key in (
            "_find_arrival_evidence",
            "_find_confirmation_diagnostics",
        ):
            if key in target:
                promoted[key] = copy.deepcopy(target[key])
        return promoted

    def _center_acquired_target(
        self,
        target_name,
        observation,
        base,
        *,
        continue_to_approach=True,
    ):
        """Center an acquired FIND_OBJECT target in bounded guarded chunks."""
        centering_attempted = 0
        centering_completed = 0
        centering_best_abs_error = None
        centering_stagnant_observations = 0
        current = observation
        last_guarded_result = base.get("last_guarded_turn_result")

        def result(**fields):
            value = dict(
                base,
                target=target_name,
                target_found=fields.pop("target_found", True),
                centering_turn_chunks_attempted=centering_attempted,
                centering_turn_chunks_completed=centering_completed,
                centering_no_progress_max_observations=(
                    self.FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS
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
            if not self._execution_is_current():
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="PREEMPTED",
                    reason="FIND_OBJECT execution was preempted.",
                    target_observation=current,
                )
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

            absolute_error = abs(horizontal_error)
            if centering_best_abs_error is None:
                centering_best_abs_error = absolute_error
                centering_stagnant_observations = 0
            elif absolute_error <= (
                centering_best_abs_error - self.FIND_CENTER_MIN_PROGRESS_PIXELS
            ):
                centering_best_abs_error = absolute_error
                centering_stagnant_observations = 0
            else:
                centering_stagnant_observations += 1

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

            if absolute_error <= self.FIND_CENTER_TOLERANCE_PIXELS:
                if not continue_to_approach:
                    return result(
                        ok=True,
                        completed=False,
                        state="CENTERED",
                        reason="Target centered after bounded correction.",
                        **telemetry,
                    )
                return self._execute_find_object_approach(
                    target_name,
                    current,
                    base,
                    centering_attempted=centering_attempted,
                    centering_completed=centering_completed,
                    last_guarded_result=last_guarded_result,
                    telemetry=telemetry,
                )

            if (
                centering_stagnant_observations
                >= self.FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS
            ):
                return result(
                    ok=False,
                    completed=False,
                    state="CENTERING_NO_PROGRESS",
                    reason=(
                        f"{target_name} centering made no meaningful progress."
                    ),
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

            if not self._execution_is_current():
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="PREEMPTED",
                    reason="FIND_OBJECT execution was preempted.",
                    **telemetry,
                )

            centering_attempted += 1
            try:
                guarded_result = self._execute_target_directed_turn(
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
            post_turn_cutoff = None
            for key in ("source_timestamp", "timestamp", "last_seen"):
                value = current.get(key)
                if isinstance(value, str) and value.strip():
                    post_turn_cutoff = value.strip()
                    break
            if self._vision_timestamp_is_iso(post_turn_cutoff):
                # This is deliberately created only after the guarded turn
                # returned, so cached pre-turn ISO frames cannot confirm the
                # post-turn observation. Synthetic/non-ISO test timestamps
                # retain their deterministic ordering semantics.
                post_turn_cutoff = datetime.now(timezone.utc).isoformat()

            episode = self._semantic_episode or {}
            semantic_attempts_before = episode.get("turn_attempts", 0)
            semantic_completions_before = episode.get("turn_completions", 0)
            (
                confirmed,
                confirmation_status,
                confirmation_diagnostics,
            ) = self._confirm_find_target_with_semantic(
                target_name,
                semantic_turn_budget=1,
                minimum_timestamp=post_turn_cutoff,
                return_diagnostics=True,
                confirmation_window_seconds=(
                    self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                ),
            )
            centering_attempted += episode.get("turn_attempts", 0) - semantic_attempts_before
            centering_completed += episode.get("turn_completions", 0) - semantic_completions_before
            self._last_target_confirmation_status = confirmation_status
            if confirmed is None:
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
                    confirmation_status=confirmation_status,
                    confirmation_diagnostics=confirmation_diagnostics,
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
                    confirmation_status=confirmation_status,
                    confirmation_diagnostics=confirmation_diagnostics,
                    **telemetry,
                )
            current = promoted

    def _execute_find_object_approach(
        self,
        target_name,
        observation,
        base,
        *,
        centering_attempted,
        centering_completed,
        last_guarded_result,
        telemetry,
    ):
        """Execute a bounded, independently guarded FIND_OBJECT approach."""
        approach_attempted = 0
        approach_completed = 0
        approach_result = None
        approach_steps = []
        avoidance_attempted = 0
        avoidance_completed = 0
        avoidance_steps = []
        bypass_pending = False
        pending_avoidance_step = None
        clearance_forward_attempted = False
        clearance_forward_completed = False
        clearance_forward_trigger_reason = None
        clearance_forward_result = None
        clearance_forward_post_confirmation_status = None
        clearance_forward_post_confirmation_diagnostics = None
        centering_total_attempted = centering_attempted
        centering_total_completed = centering_completed
        current = observation
        initial_arrival_confirmation_attempted = False

        def target_telemetry(target, previous=None):
            value = dict(previous or {})
            public_target = {
                key: item
                for key, item in target.items()
                if not key.startswith("_find_")
            }
            image_width = target.get("image_width")
            center_x = target.get("cx")
            image_center_x = None
            horizontal_error = None
            if (
                isinstance(image_width, (int, float))
                and not isinstance(image_width, bool)
                and math.isfinite(image_width)
                and image_width > 0
                and isinstance(center_x, (int, float))
                and not isinstance(center_x, bool)
                and math.isfinite(center_x)
            ):
                image_center_x = float(image_width) / 2.0
                horizontal_error = float(center_x) - image_center_x
            direction = (
                "LEFT"
                if horizontal_error is not None
                and horizontal_error < -self.FIND_CENTER_TOLERANCE_PIXELS
                else (
                    "RIGHT"
                    if horizontal_error is not None
                    and horizontal_error > self.FIND_CENTER_TOLERANCE_PIXELS
                    else "CENTERED"
                )
            )
            value.update({
                "target_label": target_name,
                "target_confidence": target.get("confidence"),
                "target_center_x": target.get("cx"),
                "target_center_y": target.get("cy"),
                "target_area": target.get("area"),
                "image_width": image_width,
                "image_height": target.get("image_height"),
                "image_center_x": image_center_x,
                "horizontal_error": horizontal_error,
                "horizontal_error_pixels": horizontal_error,
                "steering_direction": direction,
                "centering_direction": direction,
                "bbox": target.get("bbox"),
                "target_observation": public_target,
            })
            return value

        def post_motion_cutoff(target):
            cutoff = None
            for key in ("source_timestamp", "timestamp", "last_seen"):
                value = target.get(key)
                if isinstance(value, str) and value.strip():
                    cutoff = value.strip()
                    break
            if self._vision_timestamp_is_iso(cutoff):
                return datetime.now(timezone.utc).isoformat()
            return cutoff

        def result(**fields):
            value = dict(base)
            value.update({
                "target": target_name,
                "target_found": fields.pop("target_found", True),
                "centering_turn_chunks_attempted": (
                    centering_total_attempted
                ),
                "centering_turn_chunks_completed": (
                    centering_total_completed
                ),
                "centering_no_progress_max_observations": (
                    self.FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS
                ),
                "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
                "last_guarded_turn_result": last_guarded_result,
                "approach_chunks_attempted": approach_attempted,
                "approach_chunks_completed": approach_completed,
                "maximum_approach_chunks": self.FIND_APPROACH_MAX_CHUNKS,
                "approach_forward_speed": self.FIND_APPROACH_FORWARD_SPEED,
                "approach_forward_duration": self.FIND_APPROACH_FORWARD_SECONDS,
                "approach_result": fields.pop(
                    "approach_result", approach_result
                ),
                "approach_steps": list(approach_steps),
                "avoidance_maneuvers_attempted": avoidance_attempted,
                "avoidance_maneuvers_completed": avoidance_completed,
                "maximum_avoidance_maneuvers": (
                    self.FIND_AVOIDANCE_MAX_MANEUVERS
                ),
                "avoidance_steps": list(avoidance_steps),
                "clearance_forward_attempted": clearance_forward_attempted,
                "clearance_forward_completed": clearance_forward_completed,
                "clearance_forward_trigger_reason": (
                    clearance_forward_trigger_reason
                ),
                "clearance_forward_result": clearance_forward_result,
                "clearance_forward_post_confirmation_status": (
                    clearance_forward_post_confirmation_status
                ),
                "clearance_forward_post_confirmation_diagnostics": (
                    clearance_forward_post_confirmation_diagnostics
                ),
            })
            value.update(fields)
            self._publish_tracking_state(value)
            return value

        def target_reached_result(target, telemetry_fields):
            evidence = target.get("_find_arrival_evidence")
            support = (
                evidence.get("arrival_support_count")
                if isinstance(evidence, dict)
                else None
            )
            if (
                not isinstance(support, int)
                or isinstance(support, bool)
                or support < self.FIND_ARRIVAL_SUPPORT_REQUIRED
            ):
                return None
            diagnostics = target.get("_find_confirmation_diagnostics")
            return result(
                ok=True,
                executed=True,
                completed=True,
                target_found=True,
                state="TARGET_REACHED",
                reason=(
                    "Target reached from temporally confirmed visual area."
                ),
                arrival_area_fraction_threshold=(
                    self.FIND_ARRIVAL_AREA_FRACTION
                ),
                arrival_support_required=(
                    self.FIND_ARRIVAL_SUPPORT_REQUIRED
                ),
                arrival_support_count=support,
                arrival_support_timestamps=list(
                    evidence.get("arrival_support_timestamps", [])
                ),
                arrival_area_fractions=list(
                    evidence.get("arrival_area_fractions", [])
                ),
                confirmation_status="target_confirmed",
                confirmation_diagnostics=diagnostics,
                **telemetry_fields,
            )

        telemetry = target_telemetry(current, telemetry)
        telemetry["approach_forward_speed"] = self.FIND_APPROACH_FORWARD_SPEED
        telemetry["approach_forward_duration"] = self.FIND_APPROACH_FORWARD_SECONDS
        base["executed"] = True

        while approach_completed < self.FIND_APPROACH_MAX_CHUNKS:
            if not self._execution_is_current():
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="PREEMPTED",
                    reason="FIND_OBJECT execution was preempted.",
                    **telemetry,
                )
            if not self._target_is_fresh_and_acquired(current):
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="TARGET_LOST_AFTER_APPROACH",
                    reason="Target observation is no longer actionable.",
                    **telemetry,
                )

            telemetry = target_telemetry(current, telemetry)
            horizontal_error = telemetry.get("horizontal_error")
            reached = target_reached_result(current, telemetry)
            if reached is not None:
                return reached

            current_area_fraction = self._find_target_area_fraction(current)
            if (
                not initial_arrival_confirmation_attempted
                and current.get("_find_arrival_evidence") is None
                and current_area_fraction is not None
                and current_area_fraction >= self.FIND_ARRIVAL_AREA_FRACTION
                and horizontal_error is not None
                and abs(horizontal_error) <= self.FIND_CENTER_TOLERANCE_PIXELS
            ):
                # A cached close observation cannot declare arrival. Obtain
                # the same fresh temporal YOLO support used elsewhere, but a
                # failed optional arrival check preserves existing approach.
                initial_arrival_confirmation_attempted = True
                cutoff = post_motion_cutoff(current)
                (
                    close_confirmed,
                    _close_status,
                    _close_diagnostics,
                ) = self._confirm_target_candidates_with_status(
                    target_name,
                    minimum_timestamp=cutoff,
                    return_diagnostics=True,
                    confirmation_window_seconds=(
                        self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                    ),
                )
                if close_confirmed is not None:
                    close_promoted = self._promote_confirmed_target(
                        close_confirmed
                    )
                    if close_promoted is not None:
                        current = close_promoted
                        telemetry = target_telemetry(current, telemetry)
                        horizontal_error = telemetry.get("horizontal_error")
                        reached = target_reached_result(current, telemetry)
                        if reached is not None:
                            return reached
            if (
                horizontal_error is not None
                and abs(horizontal_error) > self.FIND_CENTER_TOLERANCE_PIXELS
                and not bypass_pending
            ):
                centered_result = self._center_acquired_target(
                    target_name,
                    current,
                    base,
                    continue_to_approach=False,
                )
                centering_total_attempted += centered_result.get(
                    "centering_turn_chunks_attempted", 0
                )
                centering_total_completed += centered_result.get(
                    "centering_turn_chunks_completed", 0
                )
                last_guarded_result = centered_result.get(
                    "last_guarded_turn_result", last_guarded_result
                )
                if centered_result.get("state") != "CENTERED":
                    centered_target = centered_result.get(
                        "target_observation", current
                    )
                    can_clearance_forward = bool(
                        centered_result.get("reason") == "turn_side_not_clear"
                        and avoidance_completed > 0
                        and not clearance_forward_attempted
                        and approach_attempted < self.FIND_APPROACH_MAX_CHUNKS
                        and isinstance(centered_target, dict)
                        and self._target_is_fresh_and_acquired(centered_target)
                    )
                    if can_clearance_forward:
                        fallback_interlock = getattr(
                            self.robot, "forward_interlock", None
                        )
                        fallback_session = self._current_lidar_session()
                        fallback_lidar = None
                        if (
                            fallback_interlock is not None
                            and fallback_session is not None
                            and self.world_model is not None
                        ):
                            try:
                                fallback_lidar = (
                                    self.world_model.get_lidar_obstacles(
                                        expected_session=fallback_session
                                    )
                                )
                                expected_session = getattr(
                                    fallback_interlock,
                                    "expected_session",
                                    fallback_session,
                                )
                                permitted, interlock_reason = (
                                    fallback_interlock.refresh()
                                )
                                front = (
                                    fallback_lidar.get("sectors", {})
                                    .get("front", {})
                                    if isinstance(fallback_lidar, dict)
                                    else {}
                                )
                                can_clearance_forward = bool(
                                    expected_session == fallback_session
                                    and permitted is True
                                    and interlock_reason == "fresh_clear"
                                    and isinstance(fallback_lidar, dict)
                                    and fallback_lidar.get("available") is True
                                    and fallback_lidar.get("valid") is True
                                    and fallback_lidar.get("reason") == "fresh"
                                    and front.get("state") == "CLEAR"
                                )
                            except Exception:
                                can_clearance_forward = False
                        else:
                            can_clearance_forward = False

                    if can_clearance_forward:
                        if not self._execution_is_current():
                            return result(
                                ok=False,
                                completed=True,
                                target_found=False,
                                state="PREEMPTED",
                                reason="FIND_OBJECT execution was preempted.",
                                **telemetry,
                            )

                        clearance_forward_attempted = True
                        clearance_forward_trigger_reason = (
                            "turn_side_not_clear"
                        )
                        approach_attempted += 1
                        clearance_step = {
                            "step_index": approach_attempted,
                            "pre_forward_horizontal_error": (
                                centered_result.get("horizontal_error")
                            ),
                            "centering_chunks_before_step": (
                                centering_total_attempted
                            ),
                            "bypass_forward": False,
                            "clearance_forward": True,
                            "forward_result": None,
                            "post_motion_confirmation_status": None,
                            "post_motion_confirmation_diagnostics": None,
                        }
                        approach_steps.append(clearance_step)
                        self._publish_tracking_state(dict(
                            base,
                            **target_telemetry(centered_target, telemetry),
                            target=target_name,
                            target_found=True,
                            state="APPROACHING",
                            ok=True,
                            completed=False,
                            approach_chunks_attempted=approach_attempted,
                            approach_chunks_completed=approach_completed,
                            approach_steps=list(approach_steps),
                            clearance_forward_attempted=True,
                            clearance_forward_completed=False,
                            clearance_forward_trigger_reason=(
                                clearance_forward_trigger_reason
                            ),
                        ))
                        try:
                            clearance_forward_result = self.robot.move_forward(
                                speed=self.FIND_APPROACH_FORWARD_SPEED,
                                seconds=self.FIND_APPROACH_FORWARD_SECONDS,
                            )
                        except Exception as exc:
                            clearance_forward_result = {
                                "ok": False,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "transport_attempted": True,
                            }
                        if not isinstance(clearance_forward_result, dict):
                            clearance_forward_result = {
                                "ok": False,
                                "error": "invalid_bounded_forward_result",
                            }
                        clearance_step["forward_result"] = (
                            clearance_forward_result
                        )
                        bounded_invalidated = bool(
                            clearance_forward_result.get(
                                "bounded_forward_invalidated"
                            )
                        )
                        delivery_uncertain = bool(
                            clearance_forward_result.get("delivery_uncertain")
                        )
                        confirmed_forward_failed = (
                            "confirmed_forwarded" in clearance_forward_result
                            and clearance_forward_result.get(
                                "confirmed_forwarded"
                            ) is not True
                        )
                        if (
                            clearance_forward_result.get("ok") is not True
                            or bounded_invalidated
                            or delivery_uncertain
                            or confirmed_forward_failed
                        ):
                            blocked = bool(
                                bounded_invalidated
                                or clearance_forward_result.get("forwarded")
                                is False
                            )
                            return result(
                                ok=False,
                                completed=True,
                                state=(
                                    "APPROACH_BLOCKED"
                                    if blocked
                                    else "APPROACH_FAILED"
                                ),
                                reason=clearance_forward_result.get(
                                    "reason",
                                    clearance_forward_result.get(
                                        "error", "bounded forward failed"
                                    ),
                                ),
                                approach_result=clearance_forward_result,
                                **telemetry,
                            )

                        approach_completed += 1
                        # The bounded transport completed successfully; keep
                        # this physical completion visible even if the
                        # required post-motion target confirmation later
                        # fails.
                        clearance_forward_completed = True
                        cutoff = post_motion_cutoff(centered_target)
                        (
                            confirmed,
                            confirmation_status,
                            confirmation_diagnostics,
                        ) = self._confirm_find_target_with_semantic(
                            target_name,
                            minimum_timestamp=cutoff,
                            return_diagnostics=True,
                            confirmation_window_seconds=(
                                self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                            ),
                        )
                        clearance_forward_post_confirmation_status = (
                            confirmation_status
                        )
                        clearance_forward_post_confirmation_diagnostics = (
                            confirmation_diagnostics
                        )
                        clearance_step[
                            "post_motion_confirmation_status"
                        ] = confirmation_status
                        clearance_step[
                            "post_motion_confirmation_diagnostics"
                        ] = confirmation_diagnostics
                        if confirmed is None:
                            return result(
                                ok=False,
                                completed=True,
                                target_found=False,
                                state="TARGET_LOST_AFTER_APPROACH",
                                reason=(
                                    "Target candidates were not freshly "
                                    "re-confirmed after clearance forward."
                                ),
                                confirmation_status=confirmation_status,
                                confirmation_diagnostics=confirmation_diagnostics,
                                approach_result=clearance_forward_result,
                                **telemetry,
                            )
                        promoted = self._promote_confirmed_target(confirmed)
                        if promoted is None:
                            return result(
                                ok=False,
                                completed=True,
                                target_found=False,
                                state="TARGET_LOST_AFTER_APPROACH",
                                reason=(
                                    "Target promotion failed after clearance "
                                    "forward."
                                ),
                                confirmation_status=confirmation_status,
                                confirmation_diagnostics=confirmation_diagnostics,
                                approach_result=clearance_forward_result,
                                **telemetry,
                            )
                        current = promoted
                        telemetry = target_telemetry(current, telemetry)
                        reached = target_reached_result(current, telemetry)
                        if reached is not None:
                            return reached
                        continue

                    failure_telemetry = dict(telemetry)
                    for key in failure_telemetry:
                        if key in centered_result:
                            failure_telemetry[key] = centered_result[key]
                    failure_telemetry["target_observation"] = (
                        centered_result.get("target_observation", current)
                    )
                    return result(
                        ok=bool(centered_result.get("ok")),
                        completed=bool(centered_result.get("completed")),
                        target_found=bool(
                            centered_result.get("target_found", False)
                        ),
                        state=centered_result.get(
                            "state", "CENTERING_BLOCKED"
                        ),
                        reason=centered_result.get(
                            "reason", "Target centering failed."
                        ),
                        confirmation_status=centered_result.get(
                            "confirmation_status"
                        ),
                        confirmation_diagnostics=centered_result.get(
                            "confirmation_diagnostics"
                        ),
                        **failure_telemetry,
                    )
                current = centered_result.get("target_observation")
                if not isinstance(current, dict):
                    return result(
                        ok=False,
                        completed=False,
                        target_found=False,
                        state="TARGET_LOST_DURING_CENTERING",
                        reason="Centered target observation is unavailable.",
                        **telemetry,
                    )
                telemetry = target_telemetry(current, telemetry)
                continue

            interlock = getattr(self.robot, "forward_interlock", None)
            if interlock is None:
                approach_result = {
                    "ok": False,
                    "permitted": False,
                    "reason": "forward_interlock_not_configured",
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="forward_interlock_not_configured",
                    approach_result=approach_result,
                    **telemetry,
                )
            refresh = getattr(interlock, "refresh", None)
            if not callable(refresh):
                approach_result = {
                    "ok": False,
                    "error": "forward_interlock_refresh_unavailable",
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="forward_interlock_refresh_unavailable",
                    approach_result=approach_result,
                    **telemetry,
                )
            forward_policy = (
                FORWARD_POLICY_STRICT
                if bypass_pending
                else FORWARD_POLICY_TARGET_APPROACH
            )
            try:
                permitted, reason = refresh(policy=forward_policy)
            except Exception as exc:
                approach_result = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="forward_interlock_refresh_failed",
                    approach_result=approach_result,
                    **telemetry,
                )
            session = self._current_lidar_session()
            if session is None:
                approach_result = {
                    "ok": False,
                    "permitted": False,
                    "reason": "LiDAR producer session is unavailable.",
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="LiDAR producer session is unavailable.",
                    approach_result=approach_result,
                    **telemetry,
                )

            if bypass_pending and permitted is True and reason != "fresh_clear":
                approach_result = {
                    "ok": False,
                    "permitted": permitted,
                    "reason": reason,
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason=reason,
                    approach_result=approach_result,
                    **telemetry,
                )

            if permitted is not True:
                # Local avoidance is only a recovery for the interlock's
                # explicit front-clearance denial.  A separately readable
                # blocked LiDAR snapshot must not turn unrelated fail-closed
                # denials into a maneuver authorization.
                if reason != "front_not_clear":
                    approach_result = {
                        "ok": False,
                        "permitted": False,
                        "reason": reason,
                    }
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_BLOCKED",
                        reason=reason,
                        approach_result=approach_result,
                        **telemetry,
                    )

                recommendation = None
                if self.world_model is not None:
                    try:
                        lidar_state = self.world_model.get_lidar_obstacles(
                            expected_session=session
                        )
                        recommendation = recommend_local_avoidance(
                            lidar_state,
                            expected_session=session,
                        )
                    except Exception as exc:
                        recommendation = {
                            "recommendation": "HOLD",
                            "reason": "avoidance_lidar_read_failed",
                            "error": str(exc),
                        }

                direction = (
                    {
                        "TURN_LEFT": "LEFT",
                        "TURN_RIGHT": "RIGHT",
                    }.get(
                        recommendation.get("recommendation")
                        if isinstance(recommendation, dict)
                        else None
                    )
                )
                recommendation_usable = bool(
                    isinstance(recommendation, dict)
                    and recommendation.get("trusted") is True
                    and recommendation.get("fresh") is True
                    and recommendation.get("producer_session") == session
                    and recommendation.get("front_state") in {
                        "CAUTION", "BLOCKED"
                    }
                    and direction is not None
                    and recommendation.get("reason")
                    != "clearance_near_tie_left_preferred"
                )
                if (
                    avoidance_attempted
                    >= self.FIND_AVOIDANCE_MAX_MANEUVERS
                    or not recommendation_usable
                ):
                    approach_result = {
                        "ok": False,
                        "permitted": False,
                        "reason": reason,
                        "avoidance_recommendation": recommendation,
                    }
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_BLOCKED",
                        reason=(
                            "avoidance_maneuver_limit_reached"
                            if avoidance_attempted
                            >= self.FIND_AVOIDANCE_MAX_MANEUVERS
                            else (
                                recommendation.get("reason")
                                if isinstance(recommendation, dict)
                                else reason
                            )
                        ),
                        approach_result=approach_result,
                        **telemetry,
                    )

                if not self._execution_is_current():
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="PREEMPTED",
                        reason="FIND_OBJECT execution was preempted.",
                        **telemetry,
                    )

                avoidance_attempted += 1
                avoidance_step = {
                    "maneuver_index": avoidance_attempted,
                    "trigger_reason": reason,
                    "front_state": recommendation.get("front_state"),
                    "left_state": recommendation.get("front_left_state"),
                    "right_state": recommendation.get("front_right_state"),
                    "chosen_direction": direction,
                    "turn_result": None,
                    "avoidance_turn_chunks_attempted": 0,
                    "avoidance_turn_chunks_completed": 0,
                    "maximum_avoidance_turn_chunks": (
                        self.FIND_AVOIDANCE_MAX_TURN_CHUNKS
                    ),
                    "turn_chunks": [],
                    "bypass_target_confirmation_status": None,
                    "bypass_target_confirmation_diagnostics": None,
                    "post_turn_confirmation_status": None,
                    "post_turn_confirmation_diagnostics": None,
                    "bypass_forward_result": None,
                    "post_bypass_confirmation_status": None,
                    "post_bypass_confirmation_diagnostics": None,
                }
                avoidance_steps.append(avoidance_step)
                front_clear = False

                for chunk_index in range(
                    1, self.FIND_AVOIDANCE_MAX_TURN_CHUNKS + 1
                ):
                    if not self._execution_is_current():
                        return result(
                            ok=False,
                            completed=True,
                            target_found=False,
                            state="PREEMPTED",
                            reason="FIND_OBJECT execution was preempted.",
                            **telemetry,
                        )

                    try:
                        turn_result = self.execute_guarded_turn(
                            direction,
                            self.FIND_AVOIDANCE_TURN_SPEED,
                            self.FIND_AVOIDANCE_TURN_SECONDS,
                            expected_lidar_session=session,
                        )
                    except Exception as exc:
                        turn_result = {
                            "ok": False,
                            "permitted": False,
                            "reason": "avoidance_turn_exception",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    chunk = {
                        "chunk_index": chunk_index,
                        "direction": direction,
                        "turn_result": turn_result,
                        "post_turn_front_state": None,
                        "post_turn_lidar_reason": None,
                        "producer_session": session,
                    }
                    avoidance_step["turn_chunks"].append(chunk)
                    avoidance_step["turn_result"] = turn_result
                    avoidance_step["avoidance_turn_chunks_attempted"] = (
                        chunk_index
                    )
                    if (
                        not isinstance(turn_result, dict)
                        or turn_result.get("ok") is not True
                        or turn_result.get("permitted") is not True
                    ):
                        return result(
                            ok=False,
                            completed=True,
                            state="APPROACH_BLOCKED",
                            reason=(
                                turn_result.get(
                                    "reason", "avoidance_turn_failed"
                                )
                                if isinstance(turn_result, dict)
                                else "avoidance_turn_failed"
                            ),
                            approach_result=turn_result,
                            **telemetry,
                        )

                    avoidance_step["avoidance_turn_chunks_completed"] = (
                        chunk_index
                    )
                    last_guarded_result = turn_result
                    if not self._execution_is_current():
                        return result(
                            ok=False,
                            completed=True,
                            target_found=False,
                            state="PREEMPTED",
                            reason="FIND_OBJECT execution was preempted.",
                            **telemetry,
                        )

                    post_session = self._current_lidar_session()
                    if post_session is None:
                        return result(
                            ok=False,
                            completed=True,
                            state="APPROACH_BLOCKED",
                            reason="LiDAR producer session is unavailable.",
                            approach_result=turn_result,
                            **telemetry,
                        )
                    post_recommendation = None
                    try:
                        post_lidar = self.world_model.get_lidar_obstacles(
                            expected_session=post_session
                        )
                        post_recommendation = recommend_local_avoidance(
                            post_lidar,
                            expected_session=post_session,
                        )
                    except Exception as exc:
                        post_recommendation = {
                            "recommendation": "HOLD",
                            "reason": "avoidance_lidar_read_failed",
                            "error": str(exc),
                        }
                    post_valid = bool(
                        isinstance(post_recommendation, dict)
                        and post_recommendation.get("trusted") is True
                        and post_recommendation.get("fresh") is True
                        and post_recommendation.get("producer_session")
                        == post_session
                    )
                    post_sectors = (
                        post_lidar.get("sectors")
                        if isinstance(post_lidar, dict)
                        else None
                    )

                    def clear_sector(name):
                        sector = (
                            post_sectors.get(name)
                            if isinstance(post_sectors, dict)
                            else None
                        )
                        if not isinstance(sector, dict):
                            return False
                        clearance = sector.get("robust_clearance_m")
                        minimum = sector.get("minimum_clearance_m")
                        return bool(
                            sector.get("available") is True
                            and sector.get("state") == "CLEAR"
                            and isinstance(clearance, (int, float))
                            and not isinstance(clearance, bool)
                            and math.isfinite(clearance)
                            and isinstance(minimum, (int, float))
                            and not isinstance(minimum, bool)
                            and math.isfinite(minimum)
                        )
                    post_front = (
                        post_recommendation.get("front_state")
                        if isinstance(post_recommendation, dict)
                        else None
                    )
                    post_reason = (
                        post_recommendation.get("reason")
                        if isinstance(post_recommendation, dict)
                        else "avoidance_lidar_read_failed"
                    )
                    chunk["post_turn_front_state"] = post_front
                    chunk["post_turn_lidar_reason"] = post_reason
                    chunk["producer_session"] = post_session
                    if not post_valid:
                        return result(
                            ok=False,
                            completed=True,
                            state="APPROACH_BLOCKED",
                            reason=post_reason,
                            approach_result=turn_result,
                            **telemetry,
                        )
                    if post_front == "CLEAR" and clear_sector("front"):
                        front_clear = True
                        session = post_session
                        break
                    chosen_side_state = (
                        post_recommendation.get("front_left_state")
                        if direction == "LEFT"
                        else post_recommendation.get("front_right_state")
                    )
                    chosen_side_name = (
                        "front_left" if direction == "LEFT" else "front_right"
                    )
                    if (
                        post_front not in {"CAUTION", "BLOCKED"}
                        or chosen_side_state != "CLEAR"
                        or not clear_sector(chosen_side_name)
                    ):
                        return result(
                            ok=False,
                            completed=True,
                            state="APPROACH_BLOCKED",
                            reason="chosen_avoidance_side_no_longer_clear",
                            approach_result=turn_result,
                            **telemetry,
                        )
                    session = post_session

                if not front_clear:
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_BLOCKED",
                        reason="avoidance_turn_limit_reached",
                        approach_result=last_guarded_result,
                        **telemetry,
                    )

                avoidance_completed += 1
                if not self._execution_is_current():
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="PREEMPTED",
                        reason="FIND_OBJECT execution was preempted.",
                        **telemetry,
                    )
                cutoff = post_motion_cutoff(current)
                (
                    confirmed,
                    confirmation_status,
                    confirmation_diagnostics,
                ) = self._confirm_find_target_with_semantic(
                    target_name,
                    minimum_timestamp=cutoff,
                    return_diagnostics=True,
                    confirmation_window_seconds=(
                        self.FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS
                    ),
                )
                avoidance_step["bypass_target_confirmation_status"] = (
                    confirmation_status
                )
                avoidance_step["bypass_target_confirmation_diagnostics"] = (
                    confirmation_diagnostics
                )
                avoidance_step["post_turn_confirmation_status"] = (
                    confirmation_status
                )
                avoidance_step["post_turn_confirmation_diagnostics"] = (
                    confirmation_diagnostics
                )
                if confirmed is None:
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="TARGET_LOST_AFTER_APPROACH",
                        reason=(
                            "Target was not re-confirmed after obstacle "
                            "avoidance."
                        ),
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        approach_result=last_guarded_result,
                        **telemetry,
                    )
                promoted = self._promote_confirmed_target(confirmed)
                if promoted is None:
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="TARGET_LOST_AFTER_APPROACH",
                        reason="Target promotion failed after obstacle avoidance.",
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        approach_result=last_guarded_result,
                        **telemetry,
                    )
                current = promoted
                telemetry = target_telemetry(current, telemetry)
                reached = target_reached_result(current, telemetry)
                if reached is not None:
                    return reached
                bypass_pending = True
                pending_avoidance_step = avoidance_step
                continue

            cutoff = None
            for key in ("source_timestamp", "timestamp", "last_seen"):
                value = current.get(key)
                if isinstance(value, str) and value.strip():
                    cutoff = value.strip()
                    break
            if cutoff is None:
                approach_result = {
                    "ok": False,
                    "permitted": False,
                    "reason": "target_timestamp_unavailable",
                }
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="target_timestamp_unavailable",
                    approach_result=approach_result,
                    **telemetry,
                )

            if approach_attempted >= self.FIND_APPROACH_MAX_CHUNKS:
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_BLOCKED",
                    reason="approach_budget_exhausted_after_stale",
                    approach_result=approach_result,
                    **telemetry,
                )
            approach_attempted += 1
            step = {
                "step_index": approach_attempted,
                "pre_forward_horizontal_error": horizontal_error,
                "centering_chunks_before_step": (
                    centering_total_attempted
                ),
                "bypass_forward": bool(bypass_pending),
                "forward_result": None,
                "post_motion_confirmation_status": None,
                "post_motion_confirmation_diagnostics": None,
            }
            approach_steps.append(step)
            self._publish_tracking_state(dict(
                base,
                **telemetry,
                target=target_name,
                target_found=True,
                state="APPROACHING",
                ok=True,
                completed=False,
                centering_turn_chunks_attempted=centering_total_attempted,
                centering_turn_chunks_completed=centering_total_completed,
                centering_no_progress_max_observations=(
                    self.FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS
                ),
                approach_chunks_attempted=approach_attempted,
                approach_chunks_completed=approach_completed,
                maximum_approach_chunks=self.FIND_APPROACH_MAX_CHUNKS,
                approach_steps=list(approach_steps),
                last_guarded_turn_result=last_guarded_result,
                approach_result=None,
            ))

            if not self._execution_is_current():
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="PREEMPTED",
                    reason="FIND_OBJECT execution was preempted.",
                    **telemetry,
                )

            bypass_pending = False
            try:
                approach_result = self.robot.move_forward(
                    speed=self.FIND_APPROACH_FORWARD_SPEED,
                    seconds=self.FIND_APPROACH_FORWARD_SECONDS,
                    forward_policy=forward_policy,
                )
            except Exception as exc:
                approach_result = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "transport_attempted": True,
                }
                step["forward_result"] = approach_result
                return result(
                    ok=False,
                    completed=True,
                    state="APPROACH_FAILED",
                    reason="bounded forward transport exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    approach_result=approach_result,
                    **telemetry,
                )

            if not isinstance(approach_result, dict):
                approach_result = {
                    "ok": False,
                    "error": "invalid_bounded_forward_result",
                }
            step["forward_result"] = approach_result
            if pending_avoidance_step is not None:
                pending_avoidance_step["bypass_forward_result"] = (
                    approach_result
                )
            bounded_invalidated = bool(
                approach_result.get("bounded_forward_invalidated")
            )
            delivery_uncertain = bool(
                approach_result.get("delivery_uncertain")
            )
            confirmed_forward_failed = (
                "confirmed_forwarded" in approach_result
                and approach_result.get("confirmed_forwarded") is not True
            )
            if (
                approach_result.get("ok") is not True
                or bounded_invalidated
                or delivery_uncertain
                or confirmed_forward_failed
            ):
                recoverable_stale = bool(
                    bounded_invalidated
                    and approach_result.get("reason") == "stale"
                    and approach_result.get("transport_attempted") is True
                    and approach_result.get("forwarded") is True
                    and not delivery_uncertain
                    and isinstance(approach_result.get("transport_result"), dict)
                    and approach_result["transport_result"].get("ok") is True
                )
                if recoverable_stale:
                    recovery_cutoff = datetime.now(timezone.utc).isoformat()
                    if approach_attempted >= self.FIND_APPROACH_MAX_CHUNKS:
                        return result(
                            ok=False,
                            completed=True,
                            state="APPROACH_BLOCKED",
                            reason="approach_budget_exhausted_after_stale",
                            approach_result=approach_result,
                            **telemetry,
                        )
                    recovered, recovery_reason = (
                        self._wait_for_find_stale_recovery(interlock, session)
                    )
                    step["stale_recovery_reason"] = recovery_reason
                    if not recovered:
                        return result(
                            ok=False,
                            completed=True,
                            target_found=False,
                            state=(
                                "PREEMPTED"
                                if recovery_reason == "preempted"
                                else "APPROACH_BLOCKED"
                            ),
                            reason=recovery_reason,
                            approach_result=approach_result,
                            **telemetry,
                        )
                    (
                        confirmed,
                        confirmation_status,
                        confirmation_diagnostics,
                    ) = self._confirm_find_target_with_semantic(
                        target_name,
                        minimum_timestamp=recovery_cutoff,
                        return_diagnostics=True,
                        confirmation_window_seconds=(
                            self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                        ),
                    )
                    step["post_motion_confirmation_status"] = (
                        confirmation_status
                    )
                    step["post_motion_confirmation_diagnostics"] = (
                        confirmation_diagnostics
                    )
                    if confirmed is None or not self._execution_is_current():
                        preempted = not self._execution_is_current()
                        return result(
                            ok=False,
                            completed=True,
                            target_found=False,
                            state=(
                                "PREEMPTED"
                                if preempted
                                else "TARGET_LOST_AFTER_APPROACH"
                            ),
                            reason=(
                                "FIND_OBJECT execution was preempted."
                                if preempted
                                else "Target was not freshly re-confirmed after stale recovery."
                            ),
                            confirmation_status=confirmation_status,
                            confirmation_diagnostics=confirmation_diagnostics,
                            approach_result=approach_result,
                            **telemetry,
                        )
                    promoted = self._promote_confirmed_target(confirmed)
                    if promoted is None:
                        return result(
                            ok=False,
                            completed=True,
                            target_found=False,
                            state="TARGET_LOST_AFTER_APPROACH",
                            reason="Target promotion failed after stale recovery.",
                            confirmation_status=confirmation_status,
                            confirmation_diagnostics=confirmation_diagnostics,
                            approach_result=approach_result,
                            **telemetry,
                        )
                    current = promoted
                    telemetry = target_telemetry(current, telemetry)
                    reached = target_reached_result(current, telemetry)
                    if reached is not None:
                        return reached
                    continue
                blocked = bool(
                    bounded_invalidated
                    or approach_result.get("forwarded") is False
                )
                return result(
                    ok=False,
                    completed=True,
                    state=(
                        "APPROACH_BLOCKED" if blocked else "APPROACH_FAILED"
                    ),
                    reason=approach_result.get(
                        "reason",
                        approach_result.get(
                            "error", "bounded forward failed"
                        ),
                    ),
                    approach_result=approach_result,
                    **telemetry,
                )

            approach_completed += 1
            post_forward_cutoff = cutoff
            if self._vision_timestamp_is_iso(cutoff):
                post_forward_cutoff = datetime.now(timezone.utc).isoformat()
            (
                confirmed,
                confirmation_status,
                confirmation_diagnostics,
            ) = self._confirm_find_target_with_semantic(
                target_name,
                minimum_timestamp=post_forward_cutoff,
                return_diagnostics=True,
                confirmation_window_seconds=(
                    self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                ),
            )
            step["post_motion_confirmation_status"] = confirmation_status
            step["post_motion_confirmation_diagnostics"] = (
                confirmation_diagnostics
            )
            if confirmed is None:
                if pending_avoidance_step is not None:
                    pending_avoidance_step[
                        "post_bypass_confirmation_status"
                    ] = confirmation_status
                    pending_avoidance_step[
                        "post_bypass_confirmation_diagnostics"
                    ] = confirmation_diagnostics
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="TARGET_LOST_AFTER_APPROACH",
                    reason=(
                        "Target candidates were not freshly re-confirmed "
                        "after the bounded approach step."
                    ),
                    confirmation_status=confirmation_status,
                    confirmation_diagnostics=confirmation_diagnostics,
                    approach_result=approach_result,
                    **telemetry,
                )

            promoted = self._promote_confirmed_target(confirmed)
            if promoted is None:
                if pending_avoidance_step is not None:
                    pending_avoidance_step[
                        "post_bypass_confirmation_status"
                    ] = confirmation_status
                    pending_avoidance_step[
                        "post_bypass_confirmation_diagnostics"
                    ] = confirmation_diagnostics
                return result(
                    ok=False,
                    completed=True,
                    target_found=False,
                    state="TARGET_LOST_AFTER_APPROACH",
                    reason="Target promotion failed after bounded approach step.",
                    confirmation_status=confirmation_status,
                    confirmation_diagnostics=confirmation_diagnostics,
                    approach_result=approach_result,
                    **telemetry,
                )

            current = promoted
            telemetry = target_telemetry(current, telemetry)
            reached = target_reached_result(current, telemetry)
            if reached is not None:
                return reached
            if pending_avoidance_step is not None:
                pending_avoidance_step[
                    "post_bypass_confirmation_status"
                ] = confirmation_status
                pending_avoidance_step[
                    "post_bypass_confirmation_diagnostics"
                ] = confirmation_diagnostics
                pending_avoidance_step = None
            if approach_completed >= self.FIND_APPROACH_MAX_CHUNKS:
                terminal_state = (
                    "APPROACH_STEP_COMPLETE"
                    if self.FIND_APPROACH_MAX_CHUNKS == 1
                    else "APPROACH_SEQUENCE_COMPLETE"
                )
                terminal_reason = (
                    "One bounded approach step completed and the target was "
                    "freshly re-confirmed."
                    if self.FIND_APPROACH_MAX_CHUNKS == 1
                    else (
                        "Bounded approach sequence completed and the target "
                        "was freshly re-confirmed."
                    )
                )
                return result(
                    ok=True,
                    completed=True,
                    state=terminal_state,
                    reason=terminal_reason,
                    confirmation_status=confirmation_status,
                    confirmation_diagnostics=confirmation_diagnostics,
                    approach_result=approach_result,
                    **telemetry,
                )

        return result(
            ok=True,
            completed=True,
            state="APPROACH_SEQUENCE_COMPLETE",
            reason="Bounded approach sequence completed.",
            approach_result=approach_result,
            **telemetry,
        )

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
            "centering_no_progress_max_observations": (
                self.FIND_CENTER_NO_PROGRESS_MAX_OBSERVATIONS
            ),
            "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
            "horizontal_error_pixels": None,
            "image_center_x": None,
            "centering_direction": None,
            "approach_chunks_attempted": 0,
            "approach_chunks_completed": 0,
            "maximum_approach_chunks": self.FIND_APPROACH_MAX_CHUNKS,
            "approach_forward_speed": self.FIND_APPROACH_FORWARD_SPEED,
            "approach_forward_duration": self.FIND_APPROACH_FORWARD_SECONDS,
            "approach_result": None,
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

            episode = self._semantic_episode or {}
            semantic_attempts_before = episode.get("turn_attempts", 0)
            semantic_completions_before = episode.get("turn_completions", 0)
            confirmed, status, _diagnostics = self._confirm_find_target_with_semantic(
                target_name,
                semantic_turn_budget=self.SEARCH_MAX_TURN_CHUNKS - base["turn_chunks_attempted"],
                return_diagnostics=True,
                confirmation_window_seconds=self.FIND_OBJECT_CONFIRMATION_WINDOW_SECONDS,
            )
            base["turn_chunks_attempted"] += episode.get("turn_attempts", 0) - semantic_attempts_before
            base["turn_chunks_completed"] += episode.get("turn_completions", 0) - semantic_completions_before
            base["executed"] = base["executed"] or base["turn_chunks_attempted"] > 0
            self._last_target_confirmation_status = status
            if confirmed is not None:
                promoted = self._promote_confirmed_target(confirmed)
                if promoted is not None:
                    return self._center_acquired_target(
                        target_name,
                        promoted,
                        base,
                    )

            # A failed semantic episode is terminal; do not start another
            # search cycle or expand the existing initial search-turn budget.
            if self._semantic_episode and self._semantic_episode["used"]:
                return dict(
                    base, completed=True, state="TARGET_LOST",
                    reason="Target was not freshly confirmed after semantic reacquisition.",
                    confirmation_status=status,
                    confirmation_diagnostics=_diagnostics,
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
