import math
import threading
import time
from datetime import datetime, timezone

from robot_bridge.client import RobotBridgeClient
from guarded_turn_policy import (
    ROTATIONAL_SWEPT_FOOTPRINT,
    validate_guarded_turn,
)
from lidar_perception import MAXIMUM_EFFECTIVE_AGE_SECONDS
from local_obstacle_policy import (
    plan_local_obstacle_avoidance,
    recommend_local_avoidance,
)
from target_lock import TargetLock
from entity_registry import EntityRegistry
from person_identity_manager import PersonIdentityManager
from marvin_local_tracker import MarvinLocalTracker
from marvin_pursuit_state import (
    VISUAL_READY_TO_ALIGN,
    VISUAL_READY_TO_APPROACH,
    evaluate_marvin_pursuit_state,
)
from marvin_arrival_policy import (
    evaluate_marvin_arrival,
    evaluate_marvin_visual_arrival,
)
from marvin_preview_schema import normalize_marvin_preview
from marvin_identity_continuity import evaluate_marvin_identity_continuity
from marvin_identity_episode import evaluate_marvin_identity_episode
from marvin_identity_refresh_policy import build_marvin_identity_refresh_update
from marvin_preview_reacquisition import evaluate_marvin_preview_reacquisition
from marvin_search_policy import (
    MAX_SCAN_TURNS,
    SCAN_DIRECTION,
    plan_marvin_search_step,
)
from local_motion_safety_envelope import (
    EXPECTED_LIDAR_FRAME,
    evaluate_local_motion_safety,
)
from camera_motion_gate import evaluate_camera_gate


FIND_MARVIN_FORWARD_SPEED_MPS = 0.10


def _valid_source_frame_stamp(value):
    return value if type(value) is int and value >= 0 else None


def _bridge_status_zero(status):
    motion = status.get("motion") if isinstance(status, dict) else None
    return bool(
        isinstance(status, dict)
        and status.get("ok") is True
        and status.get("ros_ready") is True
        and isinstance(motion, dict)
        and motion.get("linear_x") == 0
        and motion.get("linear_y", 0.0) == 0
        and motion.get("angular_z") == 0
        and motion.get("streaming") is False
)


LOCAL_AVOIDANCE_LIDAR_WAIT_TIMEOUT_SECONDS = 1.0
LOCAL_AVOIDANCE_LIDAR_POLL_INTERVAL_SECONDS = 0.05


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
        safety_mode="LEGACY_BROAD_SIDE",
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
        self.safety_mode = safety_mode
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
            safety_mode=self.safety_mode,
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
        completed_monotonic_seconds = time.monotonic()
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
                "completed_monotonic_seconds": completed_monotonic_seconds,
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


class _MarvinProposalNotCentered(Exception):
    """Carry the first-stage Marvin proposal geometry to finalization."""

    def __init__(self, geometry):
        super().__init__("marvin_yolo_proposal_not_centered")
        self.geometry = dict(geometry)


class _MarvinProposalGeometryInvalid(ValueError):
    """Carry camera-frame identity through a fail-closed geometry rejection."""

    def __init__(self, source_frame_stamp_ns):
        super().__init__("marvin_yolo_proposal_geometry_invalid")
        self.source_frame_stamp_ns = source_frame_stamp_ns


class _MarvinLocalTrackerConfirmationRequired(ValueError):
    """Carry the failed preview tracker's bounded diagnostics to its API."""

    def __init__(self, opencv_tracker):
        super().__init__("marvin_local_tracker_confirmation_required")
        self.opencv_tracker = dict(opencv_tracker)


class BehaviorManager:
    MARVIN_SEMANTIC_TARGET = "marvin"
    # A seed spanning almost the whole image is a semantic region, not a
    # usable local-template target. Keep this deliberately conservative for a
    # legitimately close Marvin.
    MARVIN_SEMANTIC_MAX_BBOX_AREA_FRACTION = 0.90
    MARVIN_DETECTOR_ALIAS = "teddy bear"
    MARVIN_LOCAL_TRACKER_MAX_FRAMES = 3
    MARVIN_LOCAL_TRACKER_MIN_SUPPORT = 2
    MARVIN_PREVIEW_CONFIRMATION_WINDOW_SECONDS = 2.0
    MARVIN_PREVIEW_MAX_SEMANTIC_CANDIDATES = 8
    MARVIN_PROPOSAL_MAX_WIDTH_TO_HEIGHT_RATIO = 1.25
    MARVIN_TRACKER_HORIZONTAL_PADDING_FRACTION = 0.20
    MARVIN_TRACKER_VERTICAL_PADDING_FRACTION = 0.05

    SEARCH_TURN_SPEED = 0.30
    SEARCH_TURN_SECONDS = 1.0
    MARVIN_SEARCH_TURN_SPEED = 0.25
    MARVIN_SEARCH_TURN_SECONDS = 1.0
    MARVIN_SCAN_MAX_TURNS = MAX_SCAN_TURNS
    MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS = 3.0
    MARVIN_POST_TURN_FRAME_POLL_SECONDS = 0.05
    MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS = 60.0
    MARVIN_CLEARANCE_RECHECK_INTERVAL_SECONDS = 0.25
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

    FIND_FORWARD_SPEED = FIND_MARVIN_FORWARD_SPEED_MPS
    FIND_FORWARD_SECONDS = 0.80
    FIND_APPROACH_FORWARD_SPEED = FIND_MARVIN_FORWARD_SPEED_MPS
    FIND_APPROACH_FORWARD_SECONDS = 0.50
    FIND_APPROACH_MAX_CHUNKS = 4
    # A stale LiDAR veto proven to have occurred before transport begins is
    # not a physical action.  Permit one such fresh-perception replan in a
    # bounded autonomous episode; every other executor attempt remains under
    # the physical-action budget.
    MAX_NONPHYSICAL_STALE_REPLANS = 1
    FIND_ARRIVAL_AREA = 75000.0
    MARVIN_ONE_STEP_LIDAR_REFRESH_MAX_ATTEMPTS = 3
    MARVIN_ONE_STEP_LIDAR_REFRESH_POLL_SECONDS = 0.05
    MARVIN_CENTERING_TURN_SPEED = 0.25
    MARVIN_CENTERING_TURN_DURATION = 0.25
    MARVIN_CENTERING_MAX_TURNS = 1
    MARVIN_GUARDED_APPROACH_MAX_TURNS = 3
    MARVIN_GUARDED_APPROACH_MAX_FORWARD_STEPS = 3
    MARVIN_GUARDED_APPROACH_MAX_MOTION_ACTIONS = 6
    FIND_MARVIN_CONTROLLER_MAX_ACTIONS = 6

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
        self.marvin_local_tracker_factory = MarvinLocalTracker
        self._semantic_episode = None
        # Strict V2 previews are read-only, but must keep one local tracker
        # episode across GETs so fresh Gemini selections cannot silently
        # replace its geometry with a competing proposal.
        self._marvin_v2_tracker_episode_lock = threading.RLock()
        self._marvin_v2_tracker_episode = None

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
        self._marvin_identity_confirmation_lock = threading.Lock()
        # Preview continuity is process-local, and can only be established by
        # a successful Gemini selection.  Provider tracker metadata is never
        # an identity by itself; it merely proves that a fresh candidate is
        # still the same one as that already-semantic-confirmed session.
        self._marvin_preview_continuity_lock = threading.Lock()
        self._marvin_preview_continuity = None
        # Optional runtime-owned hook for live, dashboard-facing telemetry.
        # BehaviorManager remains usable without a runtime callback.
        self.tracking_state_callback = None
        # Runtime-owned execution-generation hook.  A missing hook preserves
        # direct/offline BehaviorManager use; CognitiveRuntime installs it so
        # STOP can invalidate a multi-action FIND_OBJECT execution.
        self.execution_authorization_provider = None
        # Runtime installs this only for mission-owned local forward progress.
        # Standalone BehaviorManager use retains its existing behavior.
        self.local_progress_with_avoidance_handler = None
        # Find Marvin uses the same handoff only after its pursuit policy has
        # authorized a local forward-progress request.
        self.marvin_local_progress_with_avoidance_handler = None
        self._marvin_room_scan_lock = threading.RLock()
        self._marvin_room_scan = None

    def begin_find_marvin_room_scan(self, mission_id, *, source_frame_stamp_ns=None):
        """Start mission-scoped state for the bounded nominal room sweep."""
        if not isinstance(mission_id, str) or not mission_id.strip():
            raise ValueError("marvin_room_scan_mission_id_invalid")
        stamp = _valid_source_frame_stamp(source_frame_stamp_ns)
        with self._marvin_room_scan_lock:
            self._marvin_room_scan = {
                "mission_id": mission_id,
                "scan_active": True,
                "scan_turn_index": 0,
                "scan_max_turns": self.MARVIN_SCAN_MAX_TURNS,
                "scan_direction": SCAN_DIRECTION,
                "scan_started_at": datetime.now(timezone.utc).isoformat(),
                "last_completed_scan_turn": None,
                "last_seen_source_frame_stamp_ns": stamp,
                "pre_turn_source_frame_stamp_ns": None,
                "awaiting_new_source_frame": False,
                "scan_exhausted": False,
                "scan_target_acquired": False,
                "scan_transition_pending": False,
                "search_state": "SEARCHING",
                "clearance_wait_active": False,
                "clearance_wait_started_monotonic": None,
                "clearance_wait_origin": None,
                "clearance_recheck_count": 0,
                "last_rotational_safety_reason": None,
                "pending_scan_turn_index": None,
                "pending_scan_direction": None,
                "interrupted_turn_detected": False,
                "interrupted_turn_monitor_reason": None,
            }
            return dict(self._marvin_room_scan)

    def clear_find_marvin_room_scan(self, mission_id=None):
        with self._marvin_room_scan_lock:
            current = self._marvin_room_scan
            if current is not None and (
                mission_id is None or current.get("mission_id") == mission_id
            ):
                self._marvin_room_scan = None

    def _room_scan_snapshot(self):
        with self._marvin_room_scan_lock:
            return dict(self._marvin_room_scan) if self._marvin_room_scan else None

    def _room_scan_update(self, **values):
        with self._marvin_room_scan_lock:
            if self._marvin_room_scan is None:
                return None
            self._marvin_room_scan.update(values)
            return dict(self._marvin_room_scan)

    @staticmethod
    def _is_valid_rotational_occupancy_validation(validation, expected_session):
        """Validate the structured fresh-JIT proof of a protected-circle hit."""
        if not isinstance(validation, dict) or not (
            validation.get("permitted") is False
            and validation.get("reason") == "rotational_protected_region_violated"
            and validation.get("producer_session") == expected_session
        ):
            return False
        age = validation.get("effective_age_seconds")
        footprint = validation.get("rotational_swept_footprint")
        point = footprint.get("violating_point") if isinstance(footprint, dict) else None
        geometry = footprint.get("geometry") if isinstance(footprint, dict) else None
        if not (
            isinstance(age, (int, float)) and not isinstance(age, bool)
            and math.isfinite(age) and 0.0 <= age <= MAXIMUM_EFFECTIVE_AGE_SECONDS
            and isinstance(footprint, dict)
            and footprint.get("permitted") is False
            and footprint.get("reason") == "rotational_protected_region_violated"
            and footprint.get("model") == "base_link_circular_rotational_envelope"
            and footprint.get("protected_radius_m") == 0.45
            and isinstance(geometry, dict) and geometry.get("valid") is True
            and geometry.get("frame_id") == EXPECTED_LIDAR_FRAME
            and isinstance(geometry.get("points"), list)
            and isinstance(geometry.get("sectors"), dict)
            and isinstance(point, dict)
        ):
            return False
        x_value, y_value = point.get("x_m"), point.get("y_m")
        return bool(
            isinstance(x_value, (int, float)) and not isinstance(x_value, bool)
            and isinstance(y_value, (int, float)) and not isinstance(y_value, bool)
            and math.isfinite(x_value) and math.isfinite(y_value)
            and math.hypot(x_value, y_value) <= 0.45
        )

    @classmethod
    def _is_rotational_clearance_block(cls, result, expected_session):
        """Accept a pre-transport fresh geometric occupancy veto only."""
        return bool(
            isinstance(result, dict)
            and result.get("validation_reason") == "rotational_protected_region_violated"
            and cls._is_valid_rotational_occupancy_validation(result, expected_session)
        )

    @classmethod
    def _is_active_turn_rotational_clearance_stop(cls, result, expected_session):
        """Recognize only a successful transport stopped by valid monitor geometry."""
        transport_result = result.get("transport_result") if isinstance(result, dict) else None
        stop_fallback = result.get("stop_fallback_result") if isinstance(result, dict) else None
        initial_footprint = (
            result.get("rotational_swept_footprint")
            if isinstance(result, dict) else None
        )
        if not isinstance(result, dict) or not (
            result.get("reason") == "rotational_protected_region_violated"
            and result.get("monitor_reason") == "rotational_protected_region_violated"
            and result.get("permitted") is True
            and result.get("validation_reason") == "rotational_swept_footprint_clear"
            and isinstance(initial_footprint, dict)
            and initial_footprint.get("permitted") is True
            and initial_footprint.get("model") == "base_link_circular_rotational_envelope"
            and initial_footprint.get("protected_radius_m") == 0.45
            and result.get("generation_invalidated") is True
            and result.get("transport_began") is True
            and result.get("transport_accepted") is True
            and result.get("transport_returned") is True
            and isinstance(transport_result, dict)
            and transport_result.get("ok") is True
            and result.get("delivery_uncertain") is False
            and result.get("transport_error") is None
            and result.get("stop_fallback_attempted") is True
            and isinstance(stop_fallback, dict)
            and stop_fallback.get("ok") is True
            and result.get("stop_fallback_error") is None
        ):
            return False
        monitor_validation = result.get("monitor_validation")
        if not cls._is_valid_rotational_occupancy_validation(
            monitor_validation, expected_session,
        ):
            return False
        events = result.get("stop_events")
        if not isinstance(events, list):
            return False
        return any(
            isinstance(event, dict)
            and event.get("source") == "monitor"
            and isinstance(event.get("result"), dict)
            and event["result"].get("ok") is True
            and cls._is_valid_rotational_occupancy_validation(
                event.get("monitor_validation"), expected_session,
            )
            for event in events
        )

    def _stop_and_verify_bridge_zero(self):
        """Establish and independently verify zero before a scan clearance wait."""
        try:
            stopped = self.robot.stop()
        except Exception as exc:
            return False, {"ok": False, "reason": "clearance_wait_stop_exception",
                           "error": str(exc), "error_type": type(exc).__name__}, None
        if not isinstance(stopped, dict) or stopped.get("ok") is not True:
            return False, {"ok": False, "reason": "clearance_wait_stop_failed",
                           "stop_result": stopped}, None
        try:
            status = self.robot.status()
        except Exception as exc:
            return False, {"ok": False, "reason": "clearance_wait_bridge_status_exception",
                           "error": str(exc), "error_type": type(exc).__name__}, None
        if not _bridge_status_zero(status):
            return False, {"ok": False, "reason": "clearance_wait_bridge_not_zero",
                           "bridge_status": status}, status
        return True, {"ok": True, "stop_result": stopped}, status

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
        label = str(target_name or "").strip().lower()
        if confirmed is None and label == self.MARVIN_SEMANTIC_TARGET and episode.get("marvin_tracker"):
            local = self._confirm_marvin_local_tracker(episode, minimum_timestamp=kwargs.get("minimum_timestamp"))
            if local is not None:
                return local, "marvin_local_tracker_confirmed", diagnostics
        if confirmed is not None or episode["used"] or self.semantic_vision is None:
            return confirmed, status, diagnostics
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
            semantic_call = (
                (lambda: self.semantic_vision.describe_marvin(frame))
                if label == self.MARVIN_SEMANTIC_TARGET
                else (lambda: self.semantic_vision.describe(label, frame))
            )
            semantic = self._semantic_bounded_call(
                semantic_call,
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
            if label == self.MARVIN_SEMANTIC_TARGET:
                if semantic.get("found") is not True:
                    return None, status, diagnostics
                geometry = self._marvin_semantic_geometry(semantic)
                # Gemini's direction is diagnostic only for Marvin. The
                # locally derived bbox center controls any semantic turn hint.
                direction = geometry["direction"]
                telemetry.update(
                    marvin_semantic_bbox=geometry["bbox"],
                    marvin_semantic_center_x=geometry["cx"],
                    marvin_semantic_horizontal_error_pixels=geometry["horizontal_error"],
                    marvin_semantic_bbox_direction=direction,
                )
                tracker = self.marvin_local_tracker_factory(frame, geometry["bbox"])
                episode["marvin_tracker"] = tracker
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

        if label == self.MARVIN_SEMANTIC_TARGET:
            local = self._confirm_marvin_local_tracker(episode, minimum_timestamp=frame.received_at)
            telemetry["marvin_local_tracker_confirmed"] = local is not None
            return local, ("marvin_local_tracker_confirmed" if local else "marvin_local_tracker_unconfirmed"), diagnostics

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

    def _marvin_semantic_geometry(self, semantic):
        """Validate a Marvin-only semantic seed and derive its pixel geometry."""
        if not isinstance(semantic, dict):
            raise ValueError("marvin_semantic_result_invalid")
        width = MarvinLocalTracker._valid_dimension(semantic.get("image_width"))
        height = MarvinLocalTracker._valid_dimension(semantic.get("image_height"))
        raw_bbox = semantic.get("bbox")
        bbox_tuple = MarvinLocalTracker._validate_bbox(raw_bbox, width, height)
        bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox_tuple))
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        if area > width * height * self.MARVIN_SEMANTIC_MAX_BBOX_AREA_FRACTION:
            raise ValueError("marvin_semantic_bbox_implausibly_large")
        cx = (bbox["x1"] + bbox["x2"]) / 2.0
        horizontal_error = cx - width / 2.0
        direction = (
            "LEFT" if horizontal_error < -self.FIND_CENTER_TOLERANCE_PIXELS
            else "RIGHT" if horizontal_error > self.FIND_CENTER_TOLERANCE_PIXELS
            else "CENTER"
        )
        return {
            "bbox": bbox, "cx": cx, "cy": (bbox["y1"] + bbox["y2"]) / 2.0,
            "area": area, "horizontal_error": horizontal_error,
            "direction": direction,
        }

    def _confirm_marvin_local_tracker(self, episode, *, minimum_timestamp=None):
        """Require fresh, continuous local tracker support before motion use."""
        tracker = episode.get("marvin_tracker")
        if tracker is None or self.semantic_vision is None:
            return None
        self._semantic_check_current(episode)
        return self._confirm_marvin_local_tracker_frames(
            tracker,
            minimum_timestamp=minimum_timestamp,
            fetch_frame=lambda: self._semantic_bounded_call(
                self.semantic_vision.fetch_frame,
                self.SEMANTIC_FRAME_TIMEOUT_SECONDS,
                episode,
            ),
            check_current=lambda: self._semantic_check_current(episode),
        )

    def _confirm_marvin_local_tracker_frames(
        self, tracker, *, minimum_timestamp=None,
        minimum_source_frame_stamp_ns=None, fetch_frame, check_current=None,
        diagnostics=None, post_action=False,
    ):
        """Confirm a seeded tracker from fresh frames.

        The frame/continuity core is authority-neutral. Mission callers supply
        execution/preemption checks; preview callers intentionally do not.
        """
        observations = []
        if diagnostics is None:
            diagnostics = {}
        diagnostics.update(frames_attempted=0, fresh_frames_attempted=0,
                           cached_frame_count=0, frames=[], failure_reason=None)
        deadline = (time.monotonic() + self.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS
                    if post_action else None)
        if post_action:
            diagnostics.update(maximum_fresh_frames=self.MARVIN_LOCAL_TRACKER_MAX_FRAMES,
                               refresh_timeout_seconds=self.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS)
        last_timestamp = minimum_timestamp
        previous_source_stamp = minimum_source_frame_stamp_ns
        previous = None
        for _ in range(self.MARVIN_LOCAL_TRACKER_MAX_FRAMES):
            if check_current is not None:
                check_current()
            sample = None
            def record_frame(frame):
                sample = {"update_returned_no_bbox": False, "bbox_invalid": False,
                          "tracker_evaluated": False}
                diagnostics["frames_attempted"] += 1
                diagnostics["frames"].append(sample)
                stamp = getattr(frame, "source_frame_stamp_ns", None)
                cached = (type(stamp) is int and previous_source_stamp is not None
                          and stamp <= previous_source_stamp)
                sample.update(source_frame_stamp_ns=stamp,
                              camera_returned_cached_frame=cached,
                              received_at=getattr(frame, "received_at", None),
                              received_monotonic_seconds=getattr(frame, "received_monotonic_seconds", None),
                              image_width=getattr(frame, "width", None),
                              image_height=getattr(frame, "height", None))
                diagnostics["cached_frame_count"] += int(cached)
                if post_action:
                    sample["identity_source"] = "marvin_locked_tracker_continuity"
                    self._emit_marvin_perception_diagnostic("post_action_frame", sample)

            try:
                if post_action:
                    frame = self._fetch_strict_v2_frame_after(
                        previous_source_stamp, execution_guard=check_current,
                        deadline=deadline, fetch_frame=fetch_frame, on_frame=record_frame,
                    )
                else:
                    frame = fetch_frame()
                    record_frame(frame)
                if check_current is not None:
                    check_current()
                sample = diagnostics["frames"][-1]
                diagnostics["fresh_frames_attempted"] += 1
                source_stamp = _valid_source_frame_stamp(getattr(frame, "source_frame_stamp_ns", None))
                if post_action:
                    # Even a rejected fresh frame is consumed; its cached
                    # copies never spend another tracker evaluation.
                    previous_source_stamp = source_stamp
                timestamp = getattr(frame, "received_at", None)
                if not self._vision_timestamp_is_newer(timestamp, last_timestamp):
                    diagnostics["failure_reason"] = "local_receipt_timestamp_not_newer"
                    if post_action:
                        continue
                    return None
                source_stamp = _valid_source_frame_stamp(
                    getattr(frame, "source_frame_stamp_ns", None)
                )
                if (not post_action and previous_source_stamp is not None
                        and (source_stamp is None or source_stamp <= previous_source_stamp)):
                    sample["camera_returned_cached_frame"] = source_stamp == previous_source_stamp
                    diagnostics["failure_reason"] = "source_frame_stamp_not_newer"
                    return None
                width = MarvinLocalTracker._valid_dimension(frame.width)
                height = MarvinLocalTracker._valid_dimension(frame.height)
                sample["tracker_evaluated"] = True
                bbox = tracker.update(frame)
                if check_current is not None:
                    check_current()
                if post_action and time.monotonic() >= deadline:
                    diagnostics["failure_reason"] = "post_action_tracker_refresh_timeout"
                    return None
                sample["tracker_bbox"] = dict(bbox) if isinstance(bbox, dict) else None
                sample["tracker_candidate_bbox"] = getattr(tracker, "last_candidate_bbox", None)
                sample["tracker_search_roi"] = getattr(tracker, "last_search_roi", None)
                if bbox is None:
                    sample["opencv_tracker"] = self._opencv_tracker_diagnostic(tracker, frame)
                    sample["update_returned_no_bbox"] = True
                    diagnostics["failure_reason"] = sample["opencv_tracker"].get("reason") or "tracker_update_no_bbox"
                    if post_action:
                        continue
                    return None
                try:
                    bbox = MarvinLocalTracker._validate_bbox(bbox, frame.width, frame.height)
                except ValueError:
                    sample["bbox_invalid"] = True
                    sample["opencv_tracker"] = self._opencv_tracker_diagnostic(tracker, frame)
                    sample["opencv_tracker"]["reason"] = "invalid_bbox"
                    diagnostics["failure_reason"] = "invalid_bbox"
                    if post_action:
                        continue
                    return None
                tracker_bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox))
                opencv_tracker = self._opencv_tracker_diagnostic(
                    tracker, frame, tracker_bbox,
                )
                sample["opencv_tracker"] = opencv_tracker
                if post_action:
                    receipt = getattr(frame, "received_monotonic_seconds", None)
                    quality = opencv_tracker.get("quality")
                    threshold = opencv_tracker.get("threshold")
                    if (opencv_tracker.get("matched") is not True
                            or type(quality) not in (int, float) or not math.isfinite(quality)
                            or type(threshold) not in (int, float) or not math.isfinite(threshold)
                            or quality < max(threshold, MarvinLocalTracker.MIN_MATCH_QUALITY)):
                        diagnostics["failure_reason"] = "below_threshold"
                        continue
                    if (type(receipt) not in (int, float) or not math.isfinite(receipt)
                            or not 0 <= time.monotonic() - receipt <= 1.0):
                        diagnostics["failure_reason"] = "tracker_local_receipt_not_current"
                        continue
                observation = {
                    "found": True, "stale": False, "target": "marvin", "label": "marvin",
                    "source": "marvin_local_tracker", "source_timestamp": timestamp,
                    "received_monotonic_seconds": getattr(frame, "received_monotonic_seconds", None),
                    "bbox": tracker_bbox,
                    "image_width": width, "image_height": height,
                    "opencv_tracker": opencv_tracker,
                }
                observation["cx"] = (bbox[0] + bbox[2]) / 2.0
                observation["cy"] = (bbox[1] + bbox[3]) / 2.0
                observation["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if previous is not None:
                    sample["continuity"] = self._target_observation_match_details(previous, observation)
                    if not sample["continuity"]["matched"]:
                        diagnostics["failure_reason"] = "tracker_frame_geometry_discontinuity"
                        return None
            except _SemanticPreempted:
                raise
            except TimeoutError as exc:
                if not post_action:
                    diagnostics["failure_reason"] = str(exc) or type(exc).__name__
                    if diagnostics["frames"]:
                        diagnostics["frames"][-1]["error_type"] = type(exc).__name__
                    return None
                diagnostics["camera_new_frame_timeout"] = True
                if not diagnostics["fresh_frames_attempted"]:
                    diagnostics["failure_reason"] = "find_marvin_post_action_camera_new_frame_timeout"
                return None
            except Exception as exc:
                diagnostics["failure_reason"] = str(exc) or type(exc).__name__
                if diagnostics["frames"]:
                    diagnostics["frames"][-1]["error_type"] = type(exc).__name__
                if post_action and diagnostics["fresh_frames_attempted"]:
                    continue
                return None
            finally:
                if post_action and sample is not None and sample.get("tracker_evaluated") is True:
                    self._emit_marvin_perception_diagnostic("post_action_frame", sample)
            observations.append(observation)
            previous = observation
            last_timestamp = timestamp
            previous_source_stamp = source_stamp
            if post_action or len(observations) >= self.MARVIN_LOCAL_TRACKER_MIN_SUPPORT:
                diagnostics["failure_reason"] = None
                return observation
        diagnostics["failure_reason"] = diagnostics.get("failure_reason") or "insufficient_tracker_support"
        return None

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
        safety_mode="LEGACY_BROAD_SIDE", dispatch_guard=None,
    ):
        previous = getattr(self, "_target_directed_turn_context", False)
        self._target_directed_turn_context = True
        try:
            extra = {"dispatch_guard": dispatch_guard} if dispatch_guard is not None else {}
            return self.execute_guarded_turn(
                direction,
                angular_speed,
                duration,
                expected_lidar_session=expected_lidar_session,
                safety_mode=safety_mode,
                **extra,
            )
        finally:
            self._target_directed_turn_context = previous

    def _emit_marvin_perception_diagnostic(self, phase, metadata):
        callback = getattr(self, "marvin_perception_diagnostic_callback", None)
        if callable(callback):
            try:
                callback(phase, metadata)
            except Exception:
                pass

    def _emit_marvin_command_diagnostic(self, phase, **metadata):
        """Best-effort cached retention; its return value has no authority."""
        callback = getattr(self, "marvin_command_diagnostic_callback", None)
        if callable(callback):
            try:
                callback(phase, metadata)
            except Exception:
                pass

    def _emit_marvin_semantic_frame_diagnostic(self, frame, candidate=None):
        try:
            box = self._target_bbox(candidate) if candidate is not None else None
            center = ({"x": (box["x1"] + box["x2"]) / 2,
                       "y": (box["y1"] + box["y2"]) / 2} if box is not None else None)
            self._emit_marvin_perception_diagnostic("semantic", {
                "source_frame_stamp_ns": getattr(frame, "source_frame_stamp_ns", None),
                "received_monotonic_seconds": getattr(frame, "received_monotonic_seconds", None),
                "image_width": getattr(frame, "width", None), "image_height": getattr(frame, "height", None),
                "identity_source": "gemini_marvin_candidate_selection", "bbox": box,
                "center": center, "tracker_quality": None, "confirmed": candidate is not None,
            })
        except Exception:
            pass

    def execute_guarded_turn(
        self,
        direction,
        angular_speed,
        duration,
        *,
        expected_lidar_session,
        now=None,
        target_directed=None,
        safety_mode="LEGACY_BROAD_SIDE",
        validate_only=False,
        dispatch_guard=None,
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
            safety_mode=safety_mode,
        )
        result = dict(validation)
        # Preserve the snapshot that authorized this turn, rather than a later
        # monitor or post-STOP acquisition. Marvin's next cycle must exceed it.
        if isinstance(state, dict):
            result["action_lidar_evidence"] = {
                "producer_session": state.get("producer_session"),
                "acquisition_sequence": state.get("acquisition_sequence"),
            }
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

        if validate_only:
            # Clearance polling may inspect a fresh JIT result without
            # authorizing a turn after its mission-scoped deadline.
            result["validation_only"] = True
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
                safety_mode=safety_mode,
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
            if dispatch_guard is not None and dispatch_guard() is not True:
                raise RuntimeError("marvin_motion_observation_stale_or_preempted")
            self._emit_marvin_command_diagnostic(
                "start", start_monotonic_seconds=time.monotonic(), linear_x=0.0, linear_y=0.0,
                angular_z=validation["angular_z"], duration=validation["duration"],
            )
            transport_result = self.robot.motion(
                linear_x=0.0,
                angular_z=validation["angular_z"],
                duration=validation["duration"],
                streaming=False,
            )
        except Exception as exc:
            transport_error = exc

        self._emit_marvin_command_diagnostic(
            "complete", completion_monotonic_seconds=time.monotonic(),
            bridge_acknowledgement=transport_result,
            transport_error=str(transport_error) if transport_error is not None else None,
        )

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
        try:
            payload = self.vision.fetch_vision_payload()
            camera_gate = evaluate_camera_gate(payload)
        except Exception as exc:
            camera_gate = {"camera_semantic_clear": False, "reason": "camera_request_failed", "error": str(exc)}
        if not camera_gate["camera_semantic_clear"]:
            stop_result = self.robot.stop()
            return {"ok": False, "executed": False, "behavior": "MOVE_FORWARD", "reason": camera_gate["reason"], "camera_gate": camera_gate, "robot_result": stop_result}
        try:
            robot_result = self.robot.local_forward()
        except Exception as exc:
            stop_result = self.robot.stop()
            return {"ok": False, "executed": False, "behavior": "MOVE_FORWARD", "reason": "local_forward_request_failed", "camera_gate": camera_gate, "error": str(exc), "emergency_stop_result": stop_result}
        if not isinstance(robot_result, dict) or robot_result.get("ok") is not True:
            stop_result = self.robot.stop()
            return {"ok": False, "executed": bool(isinstance(robot_result, dict) and robot_result.get("executed")), "behavior": "MOVE_FORWARD", "reason": (robot_result.get("stop_reason", "local_forward_failed") if isinstance(robot_result, dict) else "local_forward_failed"), "camera_gate": camera_gate, "robot_result": robot_result, "emergency_stop_result": stop_result}
        return {"ok": True, "executed": bool(robot_result.get("executed")), "behavior": "MOVE_FORWARD", "reason": robot_result.get("stop_reason", "local_forward_complete"), "camera_gate": camera_gate, "robot_result": robot_result}

    def _execute_explicit_turn(self, behavior, direction):
        session = self._current_lidar_session()
        if session is None:
            return {
                "ok": False,
                "executed": False,
                "behavior": behavior,
                "reason": "lidar_producer_session_unavailable",
            }
        result = self.execute_guarded_turn(
            direction,
            angular_speed=0.50,
            duration=0.40,
            expected_lidar_session=session,
        )
        return {
            **result,
            "behavior": behavior,
            "executed": bool(result.get("confirmed_forwarded")),
        }

    def _execute_turn_left(self, mission):
        return self._execute_explicit_turn("TURN_LEFT", "LEFT")

    def _execute_turn_right(self, mission):
        return self._execute_explicit_turn("TURN_RIGHT", "RIGHT")

    def execute_marvin_search_step(
        self,
        pursuit_state,
        *,
        scan_turn_index=0,
        selected_identity_id=None,
        preview_result=None,
        target_lock_snapshot=None,
        bridge_result=None,
        max_search_actions=None,
        now=None,
    ):
        """Plan and execute at most one existing guarded Marvin scan turn.

        This deliberately owns neither a search session nor a scan loop.  The
        caller supplies fresh Preview/TargetLock evidence and calls again only
        after the result requests replanning.
        """
        base = {
            "ok": False,
            "decision": "fail_closed",
            "search_action": "fail_closed",
            "planner": None,
            "motion_executed": False,
            "executed_primitive": None,
            "replan_required": False,
            "guarded_turn_result": None,
            "reason": None,
        }
        planner_kwargs = {
            "scan_turn_index": scan_turn_index,
            "selected_identity_id": selected_identity_id,
            "preview_result": preview_result,
            "target_lock_snapshot": target_lock_snapshot,
            "bridge_result": bridge_result,
            "now": now,
        }
        if max_search_actions is not None:
            planner_kwargs["max_search_actions"] = max_search_actions
        try:
            planner = plan_marvin_search_step(pursuit_state, **planner_kwargs)
        except Exception as exc:
            return dict(
                base,
                reason="marvin_search_planner_exception",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if not isinstance(planner, dict):
            return dict(base, reason="marvin_search_planner_result_malformed")
        action = planner.get("selected_search_action")
        result = dict(base, planner=planner, search_action=action)
        requested_identity = (
            str(selected_identity_id).strip()
            if isinstance(selected_identity_id, str) else None
        ) or None
        planned_identity = planner.get("selected_identity_id")
        if (
            requested_identity is not None
            and planned_identity is not None
            and planned_identity != requested_identity
        ):
            return dict(result, reason="marvin_search_selected_identity_changed")
        if action in {"preview_only", "reacquired", "search_complete"}:
            return dict(
                result, ok=planner.get("ok") is True,
                decision=action, reason=planner.get("reason"),
            )
        if action != "turn_left":
            return dict(result, reason="marvin_search_action_not_permitted")

        session = self._current_lidar_session()
        if session is None:
            return dict(result, reason="lidar_producer_session_unavailable")
        direction = SCAN_DIRECTION
        primitive = "guarded_turn_left"
        scan = self._room_scan_snapshot()
        wait_started = (
            scan.get("clearance_wait_started_monotonic")
            if scan and scan.get("clearance_wait_active") is True else None
        )
        recheck_count = scan.get("clearance_recheck_count", 0) if scan else 0
        wait_origin = scan.get("clearance_wait_origin") if scan else None
        interrupted_turn = scan.get("interrupted_turn_detected") is True if scan else False
        interrupted_monitor_reason = (
            scan.get("interrupted_turn_monitor_reason") if scan else None
        )
        last_block = scan.get("last_rotational_safety_reason") if scan else None
        while True:
            if wait_started is not None:
                if not self._execution_is_current():
                    ok_zero, stop_evidence, bridge_status = self._stop_and_verify_bridge_zero()
                    return dict(
                        result,
                        decision="clearance_wait_preempted",
                        stop_evidence=stop_evidence,
                        bridge_status=bridge_status,
                        bridge_zero_reestablished=ok_zero,
                        reason="find_marvin_clearance_wait_preempted",
                    )
                try:
                    waiting_status = self.robot.status()
                except Exception:
                    waiting_status = None
                if not _bridge_status_zero(waiting_status):
                    ok_zero, stop_evidence, bridge_status = self._stop_and_verify_bridge_zero()
                    if not ok_zero:
                        return dict(
                            result,
                            decision="clearance_wait_bridge_not_zero",
                            stop_evidence=stop_evidence,
                            bridge_status=bridge_status,
                            reason=stop_evidence.get(
                                "reason", "find_marvin_clearance_wait_bridge_not_zero"
                            ),
                        )
            if (
                wait_started is not None
                and time.monotonic() - wait_started
                >= self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS
            ):
                try:
                    deadline_check = self.execute_guarded_turn(
                        direction,
                        self.MARVIN_SEARCH_TURN_SPEED,
                        self.MARVIN_SEARCH_TURN_SECONDS,
                        expected_lidar_session=session,
                        now=None,
                        safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                        validate_only=True,
                    )
                except Exception as exc:
                    return dict(
                        result,
                        decision="clearance_wait_safety_recheck_failed",
                        error=str(exc), error_type=type(exc).__name__,
                        reason="marvin_search_guarded_turn_exception",
                    )
                if not (
                    self._is_rotational_clearance_block(deadline_check, session)
                    or (
                        isinstance(deadline_check, dict)
                        and deadline_check.get("permitted") is True
                        and deadline_check.get("validation_only") is True
                    )
                ):
                    return dict(
                        result,
                        decision="clearance_wait_safety_recheck_failed",
                        guarded_turn_result=deadline_check,
                        reason=(
                            deadline_check.get("reason", "marvin_search_guarded_turn_failed")
                            if isinstance(deadline_check, dict)
                            else "marvin_search_guarded_turn_failed"
                        ),
                    )
                if not self._is_rotational_clearance_block(deadline_check, session):
                    # A newly clear result at/after the deadline is too late
                    # to resume; importantly, validation_only sent no motion.
                    last_block = deadline_check.get("reason")
                ok_zero, stop_evidence, bridge_status = self._stop_and_verify_bridge_zero()
                if not ok_zero:
                    return dict(
                        result,
                        decision="clearance_wait_bridge_not_zero",
                        guarded_turn_result=deadline_check,
                        stop_evidence=stop_evidence,
                        bridge_status=bridge_status,
                        reason=stop_evidence.get("reason", "clearance_wait_stop_failed"),
                    )
                elapsed = max(0.0, time.monotonic() - wait_started)
                diagnostics = {
                    "clearance_wait_active": False,
                    "clearance_wait_origin": wait_origin,
                    "clearance_wait_elapsed_seconds": elapsed,
                    "clearance_wait_timeout_seconds": self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS,
                    "pending_scan_turn_index": scan_turn_index,
                    "pending_scan_direction": "LEFT",
                    "last_rotational_safety_reason": last_block,
                    "clearance_recheck_count": recheck_count + 1,
                    "interrupted_turn_detected": interrupted_turn,
                    "interrupted_turn_monitor_reason": interrupted_monitor_reason,
                    "guarded_turn_result": deadline_check,
                    "stop_evidence": stop_evidence,
                    "bridge_status": bridge_status,
                }
                if scan is not None:
                    self._room_scan_update(
                        search_state="CLEARANCE_WAIT_TIMED_OUT",
                        clearance_wait_active=False,
                        clearance_recheck_count=recheck_count + 1,
                        last_rotational_safety_reason=last_block,
                    )
                return dict(
                    result,
                    ok=True,
                    decision="clearance_wait_timeout",
                    motion_executed=False,
                    clearance_wait_timed_out=True,
                    **diagnostics,
                    reason="find_marvin_clearance_wait_timeout",
                )
            try:
                guarded_turn = self.execute_guarded_turn(
                    direction,
                    self.MARVIN_SEARCH_TURN_SPEED,
                    self.MARVIN_SEARCH_TURN_SECONDS,
                    expected_lidar_session=session,
                    # Only the initial decision inherits the controller's
                    # supplied clock. Rechecks must evaluate live monotonic
                    # freshness from a newly acquired World Model snapshot.
                    now=now if wait_started is None else None,
                    safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                )
            except Exception as exc:
                if wait_started is not None:
                    self._room_scan_update(
                        clearance_wait_active=False,
                        search_state="SEARCHING",
                    )
                return dict(
                    result,
                    decision="search_turn",
                    executed_primitive=primitive,
                    reason="marvin_search_guarded_turn_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            turn_ok = bool(
                isinstance(guarded_turn, dict)
                and guarded_turn.get("ok") is True
                and guarded_turn.get("permitted") is True
                and guarded_turn.get("confirmed_forwarded") is True
            )
            if turn_ok:
                diagnostics = {
                    "clearance_wait_active": False,
                    "clearance_wait_origin": wait_origin,
                    "clearance_wait_timeout_seconds": self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS,
                    "pending_scan_turn_index": scan_turn_index,
                    "pending_scan_direction": direction,
                    "last_rotational_safety_reason": last_block,
                    "clearance_recheck_count": recheck_count,
                    "interrupted_turn_detected": interrupted_turn,
                    "interrupted_turn_monitor_reason": interrupted_monitor_reason,
                    "clearance_wait_elapsed_seconds": (
                        max(0.0, time.monotonic() - wait_started)
                        if wait_started is not None else 0.0
                    ),
                }
                if wait_started is not None:
                    self._room_scan_update(
                        clearance_wait_active=False,
                        clearance_wait_started_monotonic=None,
                        clearance_wait_origin=None,
                        clearance_recheck_count=recheck_count,
                        last_rotational_safety_reason=last_block,
                        pending_scan_turn_index=None,
                        pending_scan_direction=None,
                        interrupted_turn_detected=False,
                        interrupted_turn_monitor_reason=None,
                        search_state="SEARCHING",
                    )
                return dict(
                    result,
                    ok=True,
                    decision="search_turn",
                    motion_executed=True,
                    executed_primitive=primitive,
                    replan_required=True,
                    guarded_turn_result=guarded_turn,
                    reason="marvin_search_guarded_turn_complete",
                    **diagnostics,
                )

            pre_turn_block = self._is_rotational_clearance_block(guarded_turn, session)
            active_turn_block = self._is_active_turn_rotational_clearance_stop(
                guarded_turn, session,
            )
            if not (pre_turn_block or active_turn_block):
                if wait_started is not None:
                    self._room_scan_update(
                        clearance_wait_active=False,
                        search_state="SEARCHING",
                    )
                return dict(
                    result,
                    decision="search_turn",
                    executed_primitive=primitive,
                    guarded_turn_result=guarded_turn,
                    reason=(
                        guarded_turn.get("reason", "marvin_search_guarded_turn_failed")
                        if isinstance(guarded_turn, dict)
                        else "marvin_search_guarded_turn_failed"
                    ),
                )

            if wait_started is None and pre_turn_block:
                wait_started = time.monotonic()
            last_block = guarded_turn.get("reason")
            ok_zero, stop_evidence, bridge_status = self._stop_and_verify_bridge_zero()
            if not ok_zero:
                self._room_scan_update(
                    clearance_wait_active=False,
                    last_rotational_safety_reason=last_block,
                    search_state="SEARCHING",
                )
                return dict(
                    result,
                    decision="search_turn",
                    executed_primitive=primitive,
                    guarded_turn_result=guarded_turn,
                    stop_evidence=stop_evidence,
                    bridge_status=bridge_status,
                    clearance_wait_elapsed_seconds=(
                        max(0.0, time.monotonic() - wait_started)
                        if wait_started is not None else 0.0
                    ),
                    reason=stop_evidence.get("reason", "clearance_wait_stop_failed"),
                )
            if not self._execution_is_current():
                self._room_scan_update(
                    clearance_wait_active=False,
                    search_state="SEARCHING",
                )
                return dict(
                    result,
                    decision="clearance_wait_preempted",
                    guarded_turn_result=guarded_turn,
                    stop_evidence=stop_evidence,
                    bridge_status=bridge_status,
                    reason="find_marvin_clearance_wait_preempted",
                )
            if wait_started is None:
                # An active-turn monitor already issued STOP; the clearance
                # deadline starts only after this explicit zero verification.
                wait_started = time.monotonic()
            if active_turn_block:
                interrupted_turn = True
                interrupted_monitor_reason = "rotational_protected_region_violated"
            if wait_origin is None:
                wait_origin = "active_turn_monitor" if active_turn_block else "pre_turn"
            elapsed = max(0.0, time.monotonic() - wait_started)
            if scan is not None:
                self._room_scan_update(
                    search_state="FIND_MARVIN_WAITING_FOR_CLEARANCE",
                    clearance_wait_active=True,
                    clearance_wait_started_monotonic=wait_started,
                    clearance_wait_origin=wait_origin,
                    clearance_recheck_count=recheck_count,
                    pending_scan_turn_index=scan_turn_index,
                    pending_scan_direction="LEFT",
                    last_rotational_safety_reason=last_block,
                    interrupted_turn_detected=interrupted_turn,
                    interrupted_turn_monitor_reason=interrupted_monitor_reason,
                )
            if elapsed >= self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS:
                diagnostics = {
                    "clearance_wait_active": False,
                    "clearance_wait_origin": wait_origin,
                    "clearance_wait_elapsed_seconds": elapsed,
                    "clearance_wait_timeout_seconds": self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS,
                    "pending_scan_turn_index": scan_turn_index,
                    "pending_scan_direction": "LEFT",
                    "last_rotational_safety_reason": last_block,
                    "clearance_recheck_count": recheck_count,
                    "interrupted_turn_detected": interrupted_turn,
                    "interrupted_turn_monitor_reason": interrupted_monitor_reason,
                    "stop_evidence": stop_evidence,
                    "bridge_status": bridge_status,
                }
                if scan is not None:
                    self._room_scan_update(
                        search_state="CLEARANCE_WAIT_TIMED_OUT",
                        clearance_wait_active=False,
                        clearance_recheck_count=recheck_count,
                    )
                return dict(
                    result,
                    ok=True,
                    decision="clearance_wait_timeout",
                    motion_executed=False,
                    clearance_wait_timed_out=True,
                    **diagnostics,
                    reason="find_marvin_clearance_wait_timeout",
                )
            if not self._execution_is_current():
                self._room_scan_update(
                    clearance_wait_active=False,
                    search_state="SEARCHING",
                )
                return dict(
                    result, decision="clearance_wait_preempted",
                    guarded_turn_result=guarded_turn,
                    stop_evidence=stop_evidence,
                    bridge_status=bridge_status,
                    reason="find_marvin_clearance_wait_preempted",
                )
            time.sleep(min(
                self.MARVIN_CLEARANCE_RECHECK_INTERVAL_SECONDS,
                self.MARVIN_CLEARANCE_WAIT_TIMEOUT_SECONDS - elapsed,
            ))
            # Verify zero at every passive wait boundary before another fresh
            # JIT guard evaluation is allowed.
            try:
                bridge_status = self.robot.status()
            except Exception as exc:
                bridge_status = None
                status_error = {"error": str(exc), "error_type": type(exc).__name__}
            else:
                status_error = None
            if status_error is not None or not _bridge_status_zero(bridge_status):
                stop_ok, stop_evidence, stopped_status = self._stop_and_verify_bridge_zero()
                return dict(
                    result,
                    decision="clearance_wait_bridge_not_zero",
                    guarded_turn_result=guarded_turn,
                    bridge_status=(stopped_status if stopped_status is not None else bridge_status),
                    status_error=status_error,
                    stop_evidence=stop_evidence,
                    bridge_zero_reestablished=stop_ok,
                    reason="find_marvin_clearance_wait_bridge_not_zero",
                )
            recheck_count += 1
            if scan is not None:
                self._room_scan_update(clearance_recheck_count=recheck_count)

    def build_find_marvin_controller_state(
        self, *, now=None, require_fresh_gemini=False, read_only=False,
    ):
        """Read current Marvin evidence for one controller decision.

        This is perception/identity acquisition only.  It deliberately does
        not invoke the controller or any action primitive.
        """
        if self.target_lock is None:
            raise RuntimeError("marvin_target_lock_unavailable")
        if not isinstance(read_only, bool):
            raise RuntimeError("marvin_read_only_policy_invalid")
        target_label = (
            str(self.target_lock.target_label or "").strip().lower()
            or self.MARVIN_SEMANTIC_TARGET
        )
        if target_label != self.MARVIN_SEMANTIC_TARGET:
            raise RuntimeError("marvin_target_lock_target_mismatch")
        lock_snapshot = self.target_lock.snapshot()
        if not isinstance(lock_snapshot, dict):
            raise RuntimeError("marvin_target_lock_snapshot_malformed")
        if read_only:
            # This is the controller builder's strict V2 Preview acquisition,
            # deliberately before room-scan bookkeeping, entity refresh, and
            # TargetLock.resolve().
            preview = self.preview_find_object(
                self.MARVIN_SEMANTIC_TARGET,
                require_fresh_gemini=require_fresh_gemini,
            )
            if not isinstance(preview, dict):
                raise RuntimeError("marvin_preview_result_malformed")
            return {
                "preview_result": preview,
                "target_lock_result": {},
                "target_lock_snapshot": lock_snapshot,
                "selected_identity_id": None,
                "identity_evidence": None,
                "bridge_result": None,
                "read_only": True,
            }
        scan = self._room_scan_snapshot()
        if scan and scan.get("awaiting_new_source_frame") is True:
            baseline = scan.get("pre_turn_source_frame_stamp_ns")
            fetch = getattr(self.vision, "fetch_detection_proposals", None)
            if type(baseline) is not int or baseline < 0 or not callable(fetch):
                return {"post_turn_frame_status": "unavailable", "preview_result": None}
            deadline = time.monotonic() + self.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS
            latest_stamp = None
            while time.monotonic() < deadline:
                if not self._execution_is_current():
                    return {"post_turn_frame_status": "preempted", "preview_result": None}
                try:
                    payload = fetch()
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    stamp = _valid_source_frame_stamp(payload.get("source_frame_stamp_ns"))
                    if stamp is not None:
                        latest_stamp = stamp
                        if stamp > baseline:
                            break
                time.sleep(self.MARVIN_POST_TURN_FRAME_POLL_SECONDS)
            if latest_stamp is None or latest_stamp <= baseline:
                return {
                    "post_turn_frame_status": "timeout",
                    "post_turn_frame_baseline_ns": baseline,
                    "post_turn_frame_latest_ns": latest_stamp,
                    "preview_result": None,
                }
            self._room_scan_update(
                awaiting_new_source_frame=False,
                last_seen_source_frame_stamp_ns=latest_stamp,
            )
            scan = self._room_scan_snapshot()
        required_source_stamp = (
            scan.get("pre_turn_source_frame_stamp_ns")
            if scan and scan.get("awaiting_new_source_frame") is False
            and scan.get("last_completed_scan_turn") is not None
            else None
        )
        preview_options = {}
        if required_source_stamp is not None:
            preview_options["minimum_source_frame_stamp_ns"] = required_source_stamp
        if require_fresh_gemini:
            preview_options["require_fresh_gemini"] = True
        preview = self.preview_find_object(
            self.MARVIN_SEMANTIC_TARGET, **preview_options,
        )
        if not isinstance(preview, dict):
            raise RuntimeError("marvin_preview_result_malformed")
        preview_stamp = _valid_source_frame_stamp(preview.get("source_frame_stamp_ns"))
        if scan and scan.get("pre_turn_source_frame_stamp_ns") is not None:
            baseline = scan.get("pre_turn_source_frame_stamp_ns")
            if (
                type(baseline) is not int
                or preview_stamp is None
                or preview_stamp <= baseline
            ):
                return {
                    "post_turn_frame_status": "preview_not_bound_to_new_frame",
                    "preview_result": preview,
                    "target_lock_result": {},
                    "target_lock_snapshot": lock_snapshot,
                    "selected_identity_id": None,
                }
            self._room_scan_update(
                last_seen_source_frame_stamp_ns=preview_stamp,
                pre_turn_source_frame_stamp_ns=None,
            )
        selected_identity_id = (
            lock_snapshot.get("locked_identity_id")
        )

        # A runtime restart drops only the in-memory lock.  A current,
        # semantic-confirmed Preview may still drive the bounded visual
        # session branch, but it must not refresh persistent state or ask
        # TargetLock to resolve/create anything merely to do so.
        if not self._marvin_target_lock_snapshot_is_locked(lock_snapshot):
            continuity_now = now if now is not None else datetime.now(timezone.utc)
            return {
                "preview_result": preview,
                "target_lock_result": {},
                "target_lock_snapshot": lock_snapshot,
                "selected_identity_id": None,
                "identity_evidence": None,
                "bridge_result": None,
                "identity_continuity": {
                    "ok": True,
                    "allow_refresh": False,
                    "reason": "target_lock_unavailable_for_persistent_refresh",
                    "entity_id": None,
                    "identity_id": None,
                },
                "identity_episode_continuity": evaluate_marvin_identity_episode(
                    None, preview, now=continuity_now,
                ),
                "identity_refresh": {
                    "ok": True,
                    "allow_refresh": False,
                    "reason": "visual_session_has_no_persistent_refresh",
                    "entity_id": None,
                    "identity_id": None,
                    "observation_update": None,
                },
                "visual_session": True,
            }

        confirmed_entity = None
        previous_confirmation = None
        identity_refresh = {
            "ok": True,
            "allow_refresh": False,
            "reason": "confirmed_marvin_entity_unavailable",
            "entity_id": None,
            "identity_id": None,
            "observation_update": None,
        }
        if selected_identity_id and self.world_model is not None:
            try:
                reload_world_model = getattr(self.world_model, "reload", None)
                if callable(reload_world_model):
                    reload_world_model()
                entities = self.world_model.get_entities()
            except Exception as exc:
                raise RuntimeError("marvin_world_model_entity_lookup_failed") from exc
            matches = [
                entity for entity in entities
                if isinstance(entity, dict)
                and str(entity.get("label") or "").strip().lower() == "marvin"
                and isinstance(entity.get("attributes"), dict)
                and str(entity["attributes"].get("identity_id") or "").strip()
                == str(selected_identity_id).strip()
            ] if isinstance(entities, list) else []
            if len(matches) == 1:
                confirmed_entity = matches[0]
                attributes = confirmed_entity.get("attributes", {})
                previously_operator_confirmed = (
                    attributes.get("operator_confirmed") is True
                    and attributes.get("identity_confirmation_source")
                    == "marvin_local_tracker_preview"
                    and bool(attributes.get("identity_confirmation_timestamp"))
                )
                previous_confirmation = {
                    "ok": previously_operator_confirmed,
                    "confirmed": previously_operator_confirmed,
                    "identity_confirmed": previously_operator_confirmed,
                    "entity_id": confirmed_entity.get("entity_id"),
                    "identity_id": attributes.get("identity_id"),
                    "confirmation_timestamp": attributes.get(
                        "identity_confirmation_timestamp"
                    ),
                    "preview_candidate": attributes.get("preview_candidate"),
                }

        continuity_now = now if now is not None else datetime.now(timezone.utc)
        identity_continuity = evaluate_marvin_identity_continuity(
            preview,
            previous_confirmation,
            confirmed_entity,
            now=continuity_now,
        )
        # Diagnostic only: this bounded episode policy neither changes the
        # existing refresh authority nor supplies a World Model write payload.
        identity_episode_continuity = evaluate_marvin_identity_episode(
            previous_confirmation,
            preview,
            now=continuity_now,
        )
        if confirmed_entity is not None:
            identity_refresh = build_marvin_identity_refresh_update(
                identity_continuity,
                confirmed_entity,
                preview,
                now=continuity_now,
            )
            if identity_refresh.get("allow_refresh") is True:
                update = identity_refresh.get("observation_update")
                if not isinstance(update, dict):
                    raise RuntimeError("marvin_identity_refresh_payload_malformed")
                bbox = update["bbox"]
                location = {
                    "cx": (bbox["x1"] + bbox["x2"]) / 2.0,
                    "cy": (bbox["y1"] + bbox["y2"]) / 2.0,
                    "frame": "camera",
                }
                prior_confidence = confirmed_entity.get("confidence", 0.0)
                refresh_confidence = update.get("confidence", prior_confidence)
                refresh_attributes = {
                    key: value
                    for key, value in update.items()
                    if key != "confidence"
                }
                try:
                    existing_record = self.world_model.get_entity(
                        confirmed_entity["entity_id"]
                    )
                    if isinstance(existing_record, dict):
                        current_entity_id = existing_record.get("entity_id")
                        current_attributes = existing_record.get("attributes", {})
                    else:
                        current_entity_id = getattr(
                            existing_record, "entity_id", None
                        )
                        current_attributes = getattr(
                            existing_record, "attributes", {}
                        )
                    if (
                        existing_record is None
                        or current_entity_id != confirmed_entity["entity_id"]
                        or not isinstance(current_attributes, dict)
                        or current_attributes.get("identity_id")
                        != identity_refresh.get("identity_id")
                    ):
                        raise RuntimeError("confirmed_entity_no_longer_exists")
                    self.world_model.update_entity(
                        entity_id=confirmed_entity["entity_id"],
                        label=confirmed_entity["label"],
                        entity_type=confirmed_entity["entity_type"],
                        confidence=refresh_confidence,
                        source=update["source"],
                        location=location,
                        attributes=refresh_attributes,
                    )
                except Exception as exc:
                    raise RuntimeError("marvin_identity_refresh_write_failed") from exc

        # Resolve only after any continuity-authorized observation refresh so
        # TargetLock consumes the ordinary World Model path and remains the
        # sole runtime identity-lock authority.
        lock_result = self.target_lock.resolve(
            mission_id=self.target_lock.mission_id,
            target_label=self.MARVIN_SEMANTIC_TARGET,
        )
        lock_snapshot = self.target_lock.snapshot()
        if not isinstance(lock_result, dict):
            raise RuntimeError("marvin_target_lock_result_malformed")
        if not isinstance(lock_snapshot, dict):
            raise RuntimeError("marvin_target_lock_snapshot_malformed")
        bridge_result = evaluate_marvin_preview_reacquisition(
            preview,
            lock_snapshot,
            identity_evidence=lock_result,
            now=now,
        )
        return {
            "preview_result": preview,
            "target_lock_result": lock_result,
            "target_lock_snapshot": lock_snapshot,
            "selected_identity_id": selected_identity_id,
            "identity_evidence": lock_result,
            "bridge_result": bridge_result,
            "identity_continuity": identity_continuity,
            "identity_episode_continuity": identity_episode_continuity,
            "identity_refresh": identity_refresh,
        }

    def observe_find_marvin_v2(self, *, now=None):
        """Read strict V2 Marvin evidence without resolving or promoting identity.

        This intentionally shares the controller builder's fresh-Gemini
        preview path, but stops before the builder's persistent World Model
        refresh and TargetLock.resolve() boundary.  It is therefore suitable
        for operator inspection only, never for execution.
        """
        continuity = self._continue_strict_v2_tracker_after_action()
        if continuity is not None:
            if continuity.get("found") is not True:
                preview = {
                    "ok": False, "preview": True, "authoritative": False,
                    "executed": False, "completed": True,
                    "behavior": "FIND_OBJECT", "state": "PREVIEW",
                    "target": "marvin", "target_found": False,
                    "identity_confirmed": False,
                    "motion_authorized_marvin_candidate": False,
                    "reason": continuity.get("reason"),
                    "source_frame_stamp_ns": continuity.get("source_frame_stamp_ns"),
                    "opencv_tracker": continuity.get("opencv_tracker"),
                    "strict_tracker_episode": continuity.get("strict_tracker_episode"),
                    "post_action_tracker_diagnostics": continuity.get("post_action_tracker_diagnostics"),
                }
            else:
                preview = self._build_find_object_preview(
                    self.MARVIN_SEMANTIC_TARGET, continuity,
                    source="marvin_local_tracker", authoritative=False,
                )
                for key in (
                    "identity_confirmed", "identity_source",
                    "identity_source_frame_stamp_ns",
                    "motion_authorized_marvin_candidate", "opencv_tracker",
                    "strict_tracker_episode", "marvin_tracking_episode",
                    "post_action_tracker_continuity",
                    "post_action_source_frame_stamp_ns", "post_action_tracker_diagnostics",
                ):
                    if key in continuity:
                        preview[key] = continuity[key]
            return {
                "preview_result": preview,
                "target_lock_result": {},
                "target_lock_snapshot": {},
                "selected_identity_id": None,
                "identity_evidence": None,
                "bridge_result": None,
                "read_only": True,
            }
        evidence = self.build_find_marvin_controller_state(
            now=now,
            require_fresh_gemini=True,
            read_only=True,
        )
        preview = evidence.get("preview_result") or {}
        if preview.get("reason") in {
            "Marvin was not found in the current camera frame.",
            "marvin_identity_not_confirmed",
        }:
            # Semantic absence is not an action frame: inference/proposal
            # latency can make its source stamp too old for a search turn.
            baseline = _valid_source_frame_stamp(
                preview.get("identity_source_frame_stamp_ns")
            )
            if baseline is None:
                baseline = _valid_source_frame_stamp(preview.get("source_frame_stamp_ns"))
            boundary = time.monotonic()
            guard = self._strict_v2_current_execution_guard()
            try:
                if guard is not None:
                    guard()
                frame = self._fetch_strict_v2_frame_after(
                    baseline,
                    execution_guard=guard,
                    minimum_received_monotonic_seconds=boundary,
                )
                if guard is not None:
                    guard()
                stamp = _valid_source_frame_stamp(frame.source_frame_stamp_ns)
                if (baseline is None or stamp is None
                        or stamp <= baseline
                        or type(frame.width) is not int or frame.width <= 0
                        or type(frame.height) is not int or frame.height <= 0):
                    raise ValueError("fresh_search_frame_unavailable")
                preview["identity_source_frame_stamp_ns"] = baseline
                preview["source_frame_stamp_ns"] = stamp
                preview["source_timestamp"] = frame.received_at
                preview["received_monotonic_seconds"] = frame.received_monotonic_seconds
            except Exception:
                preview["source_frame_stamp_ns"] = None
                preview["reason"] = "find_marvin_search_fresh_frame_unavailable"
        return evidence

    def reacquire_find_marvin_v2(self, *, minimum_source_frame_stamp_ns):
        """Start a new Gemini episode while the runtime holds the stopped robot."""
        self._clear_marvin_v2_tracker_episode()
        guard = self._strict_v2_current_execution_guard()
        # Discard the first relay sample after STOP; require camera advancement
        # as well as a local receipt after this recovery boundary.
        frame = self._fetch_strict_v2_frame_after(
            minimum_source_frame_stamp_ns, execution_guard=guard,
            minimum_received_monotonic_seconds=time.monotonic(),
        )
        preview = self.preview_find_object(
            self.MARVIN_SEMANTIC_TARGET, require_fresh_gemini=True,
            minimum_source_frame_stamp_ns=frame.source_frame_stamp_ns,
        )
        return {"preview_result": preview, "target_lock_result": {},
                "target_lock_snapshot": {}, "selected_identity_id": None}

    def mark_strict_v2_action_stopped(self, source_frame_stamp_ns, stopped_at):
        with self._marvin_v2_tracker_episode_lock:
            episode = self._marvin_v2_tracker_episode
            if (isinstance(episode, dict) and episode.get("post_action_pending") is True
                    and episode.get("post_action_source_frame_stamp_ns") == source_frame_stamp_ns):
                episode["post_action_stopped_monotonic_seconds"] = stopped_at

    def _fetch_strict_v2_frame_after(self, minimum_stamp, *, execution_guard=None,
                                     minimum_received_monotonic_seconds=None,
                                     deadline=None, fetch_frame=None, on_frame=None):
        """Wait boundedly for actual camera advancement, never restamp a JPEG.

        The relay may briefly return its cached latest frame. Polling is
        perception only, uses the existing new-frame timeout, and checks STOP
        around every fetch. Transport/invalid-stamp failures are not retried.
        """
        if deadline is None:
            deadline = time.monotonic() + self.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS
        if fetch_frame is None:
            fetch_frame = self.semantic_vision.fetch_frame
        if type(minimum_stamp) is not int or minimum_stamp < 0:
            raise ValueError("camera_source_stamp_invalid")
        # Establish the relay's current source-clock baseline *after* slow
        # semantics, then wait for advancement. This excludes a cached frame
        # produced during Gemini without comparing clocks across hosts.
        establish_source_floor = minimum_received_monotonic_seconds is not None
        while time.monotonic() < deadline:
            if execution_guard is not None:
                execution_guard()
            frame = fetch_frame()
            if execution_guard is not None:
                execution_guard()
            if on_frame is not None:
                on_frame(frame)
            stamp = _valid_source_frame_stamp(getattr(frame, "source_frame_stamp_ns", None))
            if stamp is None:
                raise ValueError("camera_source_stamp_invalid")
            if time.monotonic() >= deadline:
                break
            receipt = getattr(frame, "received_monotonic_seconds", None)
            if minimum_received_monotonic_seconds is not None and (
                type(receipt) not in (int, float) or not math.isfinite(receipt)
                or receipt < minimum_received_monotonic_seconds
                or receipt > time.monotonic()
            ):
                raise ValueError("camera_local_receipt_invalid")
            if establish_source_floor:
                minimum_stamp = max(minimum_stamp, stamp)
                establish_source_floor = False
                continue
            if stamp > minimum_stamp:
                return frame
            if execution_guard is not None:
                execution_guard()
            time.sleep(min(self.MARVIN_POST_TURN_FRAME_POLL_SECONDS,
                           max(0.0, deadline - time.monotonic())))
        raise TimeoutError("new_camera_source_frame_unavailable")

    def _strict_v2_current_execution_guard(self):
        """Bind mission perception to its current execution, not diagnostics."""
        provider = getattr(self, "execution_authorization_provider", None)
        if not callable(provider) or not self._execution_is_current():
            return None

        def check():
            if not self._execution_is_current():
                raise _SemanticPreempted()

        return check

    def mark_strict_v2_action_dispatched(self, source_frame_stamp_ns, action):
        """Require tracked evidence from a newer frame after V2 self-motion."""
        if type(source_frame_stamp_ns) is not int or source_frame_stamp_ns < 0:
            return False
        with self._marvin_v2_tracker_episode_lock:
            episode = self._marvin_v2_tracker_episode
            if (not isinstance(episode, dict)
                    or episode.get("last_tracker_source_frame_stamp_ns")
                    != source_frame_stamp_ns
                    or episode.get("identity_source")
                    != "gemini_marvin_candidate_selection"):
                return False
            episode["post_action_pending"] = True
            episode["post_action_source_frame_stamp_ns"] = source_frame_stamp_ns
            episode["post_action_kind"] = str(action)
            return True

    def _continue_strict_v2_tracker_after_action(self):
        """Continue only the locked tracker across known V2 self-motion.

        Up to three genuinely newer frames may supply one independent valid
        match during one bounded refresh window. Cached frames only wait.
        Static candidate-vs-pre-action-box IoU is intentionally not
        used across a commanded camera rotation/translation.  Failure clears
        the episode so the next ordinary observation must reacquire via Gemini.
        """
        with self._marvin_v2_tracker_episode_lock:
            episode = self._marvin_v2_tracker_episode
            if not isinstance(episode, dict) or episode.get("post_action_pending") is not True:
                return None
            if episode.get("post_action_in_progress") is True:
                return {
                    "found": False,
                    "reason": "post_action_tracker_continuity_in_progress",
                }
            episode = dict(episode)
            self._marvin_v2_tracker_episode["post_action_in_progress"] = True
        tracker = episode.get("marvin_tracker")
        action_stamp = episode.get("post_action_source_frame_stamp_ns")
        previous_stamp = episode.get("last_tracker_source_frame_stamp_ns")
        previous_time = episode.get("last_tracker_received_at")
        if (tracker is None or type(action_stamp) is not int
                or type(previous_stamp) is not int or action_stamp != previous_stamp
                or self.semantic_vision is None):
            self._clear_marvin_v2_tracker_episode()
            return {"found": False, "reason": "post_action_tracker_episode_invalid"}
        started_at = time.monotonic()
        stopped_at = episode.get("post_action_stopped_monotonic_seconds")
        diagnostics = {
            "pre_action_tracker_bbox": episode.get("tracker_bbox"),
            "pre_action_tracker": episode.get("last_tracker_diagnostics"),
            "pre_action_source_frame_stamp_ns": action_stamp,
            "pre_action_received_at": previous_time,
            "pre_action_received_monotonic_seconds": episode.get("last_tracker_received_monotonic_seconds"),
            "action_kind": episode.get("post_action_kind"),
            "action_stopped_monotonic_seconds": stopped_at,
            "refresh_started_monotonic_seconds": started_at,
            "stop_to_refresh_seconds": started_at - stopped_at if stopped_at is not None else None,
        }
        guard = self._strict_v2_current_execution_guard()
        confirmed = self._confirm_marvin_local_tracker_frames(
            tracker,
            minimum_timestamp=previous_time,
            minimum_source_frame_stamp_ns=action_stamp,
            fetch_frame=self.semantic_vision.fetch_frame,
            check_current=guard, diagnostics=diagnostics, post_action=True,
        )
        diagnostics["refresh_elapsed_seconds"] = time.monotonic() - started_at
        frames = diagnostics.get("frames", [])
        last_frame = frames[-1] if frames else {}
        last_opencv = last_frame.get("opencv_tracker")
        post_bbox = last_frame.get("tracker_bbox") or last_frame.get("tracker_candidate_bbox")
        pre_bbox = episode.get("tracker_bbox")
        diagnostics["post_action_candidate_bbox"] = post_bbox
        diagnostics["pre_to_post_iou"] = self._target_bbox_iou(
            {"bbox": pre_bbox}, {"bbox": post_bbox},
        ) if post_bbox else None
        if (self._target_bbox({"bbox": pre_bbox}) is not None
                and self._target_bbox({"bbox": post_bbox}) is not None):
            diagnostics["translation_pixels"] = {
                axis: (post_bbox[axis + "1"] + post_bbox[axis + "2"]
                       - pre_bbox[axis + "1"] - pre_bbox[axis + "2"]) / 2.0
                for axis in ("x", "y")
            }
            diagnostics["bbox_scale_ratios"] = {
                axis: (post_bbox[axis + "2"] - post_bbox[axis + "1"])
                / (pre_bbox[axis + "2"] - pre_bbox[axis + "1"])
                for axis in ("x", "y")
                if pre_bbox[axis + "2"] > pre_bbox[axis + "1"]
            }
        diagnostics["pre_to_post_iou_used_for_admission"] = False
        opencv = confirmed.get("opencv_tracker") if isinstance(confirmed, dict) else None
        stamp = opencv.get("source_frame_stamp_ns") if isinstance(opencv, dict) else None
        bbox = opencv.get("bbox") if isinstance(opencv, dict) else None
        quality, threshold = (
            (opencv.get("quality"), opencv.get("threshold"))
            if isinstance(opencv, dict) else (None, None)
        )
        valid = bool(
            isinstance(confirmed, dict) and type(stamp) is int
            and stamp > action_stamp and isinstance(bbox, dict)
            and opencv.get("active") is True and opencv.get("matched") is True
            and isinstance(quality, (int, float)) and not isinstance(quality, bool)
            and isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(float(quality)) and math.isfinite(float(threshold))
            and quality >= threshold
        )
        if not valid:
            diagnostics["failure_reason"] = diagnostics.get("failure_reason") or "post_action_tracker_evidence_invalid"
            self._clear_marvin_v2_tracker_episode()
            reason = ("find_marvin_post_action_camera_new_frame_timeout"
                      if diagnostics["failure_reason"] == "find_marvin_post_action_camera_new_frame_timeout"
                      else "post_action_tracker_continuity_lost")
            return {
                "found": False,
                "source_frame_stamp_ns": stamp if stamp is not None else last_frame.get("source_frame_stamp_ns"),
                "post_action_tracker_diagnostics": diagnostics,
                "opencv_tracker": opencv or last_opencv or self._empty_opencv_tracker_diagnostic(
                    reason=reason,
                ),
                "reason": reason,
                "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                    active=False, continued=True, initialized=False,
                    accepted=False, reason=reason,
                ),
            }
        episode.update(
            tracker_bbox=dict(bbox), last_tracker_source_frame_stamp_ns=stamp,
            last_tracker_diagnostics=dict(opencv),
            last_tracker_received_monotonic_seconds=confirmed.get("received_monotonic_seconds"),
            last_tracker_received_at=confirmed.get("source_timestamp"),
            post_action_pending=False, post_action_in_progress=False,
        )
        with self._marvin_v2_tracker_episode_lock:
            # Do not resurrect an episode that was invalidated concurrently.
            current = self._marvin_v2_tracker_episode
            if (not isinstance(current, dict)
                    or current.get("post_action_source_frame_stamp_ns") != action_stamp
                    or current.get("post_action_pending") is not True
                    or current.get("post_action_in_progress") is not True
                    or current.get("marvin_tracker") is not tracker):
                self._marvin_v2_tracker_episode = None
                return {"found": False, "reason": "post_action_tracker_episode_superseded"}
            self._marvin_v2_tracker_episode = episode
        identity_stamp = episode.get("identity_source_frame_stamp_ns")
        confirmed.update(
            post_action_tracker_diagnostics=diagnostics,
            identity_confirmed=True,
            identity_source="marvin_locked_tracker_continuity",
            identity_source_frame_stamp_ns=identity_stamp,
            motion_authorized_marvin_candidate=True,
            strict_tracker_episode=self._strict_v2_tracker_diagnostic(
                active=True, continued=True, initialized=False,
                accepted=True, reason="post_action_tracker_continuity_matched",
            ),
            marvin_tracking_episode={
                "episode_id": episode.get("episode_id"),
                "identity_source": episode.get("identity_source"),
                "identity_source_frame_stamp_ns": identity_stamp,
                "post_action_source_frame_stamp_ns": action_stamp,
                "post_action_kind": episode.get("post_action_kind"),
                "tracker_initialized": True,
                "tracker_source_frame_stamp_ns": stamp,
                "tracker_quality": quality,
                "tracker_bbox": dict(bbox),
                "state": "POST_ACTION_TRACKED",
            },
            post_action_tracker_continuity=True,
            post_action_source_frame_stamp_ns=action_stamp,
        )
        return confirmed

    @staticmethod
    def _marvin_target_lock_snapshot_is_locked(snapshot):
        return (
            isinstance(snapshot, dict)
            and str(snapshot.get("tracking_mode") or "").strip().upper()
            == TargetLock.MODE_LOCKED
            and bool(str(snapshot.get("locked_identity_id") or "").strip())
        )

    def build_marvin_identity_episode_diagnostic(self, *, now=None):
        """Read Marvin episode evidence without refresh, resolution, or action.

        This is intentionally separate from the controller-state builder:
        diagnostic callers may inspect the existing lock and World Model data,
        but cannot refresh an entity or ask TargetLock to resolve one.
        """
        base = {
            "preview_result": None,
            "preview_marvin_continuity": None,
            "entity_id": None,
            "identity_id": None,
        }
        target_lock = getattr(self, "target_lock", None)
        snapshot = getattr(target_lock, "snapshot", None)
        if target_lock is None or not callable(snapshot):
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_target_lock_unavailable",
            )
        try:
            lock_snapshot = snapshot()
        except Exception:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_target_lock_snapshot_unavailable",
            )
        if not isinstance(lock_snapshot, dict):
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_target_lock_snapshot_malformed",
            )
        entity_id = str(lock_snapshot.get("locked_entity_id") or "").strip() or None
        identity_id = (
            str(lock_snapshot.get("locked_identity_id") or "").strip() or None
        )
        base.update(entity_id=entity_id, identity_id=identity_id)
        if lock_snapshot.get("tracking_mode") != TargetLock.MODE_LOCKED:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_target_lock_not_locked",
            )
        if entity_id is None or identity_id is None:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_target_lock_incomplete",
            )

        world_model = getattr(self, "world_model", None)
        get_entities = getattr(world_model, "get_entities", None)
        if not callable(get_entities):
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_world_model_read_unavailable",
            )
        try:
            entities = get_entities()
        except Exception:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_world_model_read_failed",
            )
        matches = [
            entity for entity in entities
            if isinstance(entity, dict)
            and entity.get("entity_id") == entity_id
            and str(entity.get("label") or "").strip().lower() == "marvin"
            and isinstance(entity.get("attributes"), dict)
            and str(entity["attributes"].get("identity_id") or "").strip()
            == identity_id
        ] if isinstance(entities, list) else []
        if not matches:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_confirmed_entity_missing_or_mismatched",
            )
        if len(matches) != 1:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_confirmed_entity_ambiguous",
            )
        attributes = matches[0]["attributes"]
        operator_confirmed = (
            attributes.get("operator_confirmed") is True
            and attributes.get("identity_confirmation_source")
            == "marvin_local_tracker_preview"
            and bool(attributes.get("identity_confirmation_timestamp"))
        )
        if not operator_confirmed:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_operator_confirmation_missing",
            )
        previous_confirmation = {
            "ok": True,
            "confirmed": True,
            "identity_confirmed": True,
            "entity_id": entity_id,
            "identity_id": identity_id,
            "confirmation_timestamp": attributes.get(
                "identity_confirmation_timestamp"
            ),
            "preview_candidate": attributes.get("preview_candidate"),
        }
        try:
            preview = self.preview_find_object(self.MARVIN_SEMANTIC_TARGET)
        except Exception:
            return self._marvin_identity_episode_diagnostic_failure(
                base, "marvin_preview_unavailable",
            )
        continuity = self._marvin_preview_continuity_metadata(preview)
        base.update(
            preview_result=preview,
            preview_marvin_continuity=continuity,
        )
        episode = evaluate_marvin_identity_episode(
            previous_confirmation,
            preview,
            now=now,
        )
        return dict(
            base,
            identity_episode_continuity=episode,
            accepted=episode.get("identity_continuity") is True,
            reason=episode.get("reason"),
        )

    @staticmethod
    def _marvin_preview_continuity_metadata(preview):
        if not isinstance(preview, dict):
            return None
        value = preview.get("marvin_continuity")
        if not isinstance(value, dict):
            observation = preview.get("target_observation")
            value = observation.get("marvin_continuity") if isinstance(observation, dict) else None
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _marvin_identity_episode_diagnostic_failure(base, reason):
        episode = {
            "ok": False,
            "identity_continuity": False,
            "reason": reason,
            "selected_identity_id": base["identity_id"],
            "entity_id": base["entity_id"],
            "episode_valid": False,
            "marvin_continuity": {
                "available": False,
                "tracker_id": None,
                "tracker_source": None,
            },
        }
        return dict(
            base,
            identity_episode_continuity=episode,
            accepted=False,
            reason=reason,
        )

    def execute_find_marvin_controller(
        self, state_provider, *, max_actions=FIND_MARVIN_CONTROLLER_MAX_ACTIONS,
        now=None, dry_run=False, stop_after_action=None,
        require_fresh_gemini=False,
    ):
        """Run a finite sequence of fresh, one-step Marvin search/pursuit actions.

        ``state_provider`` must return the current Preview and TargetLock
        evidence for one decision.  This controller owns neither transport nor
        scan planning, or low-level motion.  Its only motion-capable
        delegations are the existing one-step search and pursuit boundaries.
        """
        base = {
            "ok": False,
            "completed": False,
            "arrived_at_marvin": False,
            "arrival": None,
            "selected_identity_id": None,
            "reason": None,
            "max_actions": max_actions,
            "actions_executed": 0,
            "stale_replans": 0,
            "maximum_nonphysical_stale_replans": (
                self.MAX_NONPHYSICAL_STALE_REPLANS
            ),
            "history": [],
            "arrival_observations_confirmed": 0,
        }
        if (
            not isinstance(max_actions, int)
            or isinstance(max_actions, bool)
            or max_actions <= 0
        ):
            return dict(base, reason="invalid_find_marvin_action_limit")
        if not callable(state_provider):
            return dict(base, reason="find_marvin_state_provider_unavailable")
        if not isinstance(dry_run, bool):
            return dict(base, reason="invalid_find_marvin_dry_run")
        if not isinstance(require_fresh_gemini, bool):
            return dict(base, reason="invalid_find_marvin_identity_policy")
        if stop_after_action is not None and not callable(stop_after_action):
            return dict(base, reason="find_marvin_stop_callback_unavailable")

        def stop_completed_action(history_entry):
            """Stop before obtaining the next mandatory fresh observation."""
            if stop_after_action is None:
                return True
            try:
                stop_result = stop_after_action()
            except Exception as exc:
                history_entry["stop_result"] = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                return False
            history_entry["stop_result"] = stop_result
            return isinstance(stop_result, dict) and stop_result.get("ok") is True

        selected_identity_id = None
        arrival_candidate_timestamp = None
        local_progress_completed = False
        last_local_progress_result = None
        search_history = []
        local_scan_turn_index = 0
        # ``actions_executed`` counts physical or delivery-uncertain actions.
        # A verified pre-transport stale veto can replan once without spending
        # that budget, but this loop remains finite through its separate cap.
        # A successful local-progress handoff must always be followed by one
        # fresh identity/perception observation, even when its nested actions
        # consumed the final outer action slots.
        while (
            base["actions_executed"] < max_actions
            or local_progress_completed
        ):
            try:
                evidence = state_provider()
            except Exception as exc:
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_state_provider_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            if not isinstance(evidence, dict):
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_state_evidence_malformed",
                )
            if evidence.get("post_turn_frame_status") in {
                "timeout", "unavailable", "preempted",
                "preview_not_bound_to_new_frame",
            }:
                return dict(
                    base, ok=True, completed=False,
                    history=list(base["history"]),
                    reason="find_marvin_post_turn_frame_" + str(
                        evidence["post_turn_frame_status"]
                    ),
                    post_turn_frame=evidence,
                )
            preview = evidence.get("preview_result")
            lock_result = evidence.get("target_lock_result")
            lock_snapshot = evidence.get("target_lock_snapshot")
            if require_fresh_gemini and not self._marvin_v2_preview_is_verified(preview):
                pursuit = {
                    "state": "SEARCHING",
                    "pursuit_authorized": False,
                    "reason": "find_marvin_v2_fresh_identity_required",
                }
            else:
                try:
                    pursuit = evaluate_marvin_pursuit_state(
                        preview,
                        lock_result,
                        lock_snapshot,
                        selected_identity_id=evidence.get("selected_identity_id"),
                        identity_evidence=evidence.get("identity_evidence"),
                        bridge_result=evidence.get("bridge_result"),
                        now=now,
                    )
                except Exception as exc:
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="find_marvin_pursuit_evaluation_exception",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
            if not isinstance(pursuit, dict):
                pursuit = {
                    "state": "INSUFFICIENT_EVIDENCE",
                    "pursuit_authorized": False,
                    "reason": "marvin_pursuit_result_malformed",
                }
            current_identity_id = pursuit.get("selected_identity_id")
            history_entry = {
                "pursuit_state": pursuit.get("state"),
                "pursuit_authorized": (
                    pursuit.get("pursuit_authorized") is True
                ),
                "selected_identity_id": current_identity_id,
                "entity_id": pursuit.get("entity_id"),
                "fresh": pursuit.get("fresh"),
                "geometry_usable": pursuit.get("geometry_usable"),
                "arrival": None,
                "selected_action": "no_motion",
                "route": "none",
                "executed_primitive": None,
                "replan_required": False,
                "search_action": None,
                "search_step_result": None,
                "pursuit_step_result": None,
                "action_index": len(base["history"]) + 1,
                "action_budget_consumed": False,
            }
            if selected_identity_id is None:
                selected_identity_id = current_identity_id
            elif current_identity_id != selected_identity_id:
                base["history"].append(history_entry)
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_identity_changed",
                )

            scan = self._room_scan_snapshot()
            if scan and scan.get("scan_active") is True:
                source_stamp = _valid_source_frame_stamp(
                    preview.get("source_frame_stamp_ns")
                    if isinstance(preview, dict) else None
                )
                if scan.get("scan_turn_index", 0) == 0 and scan.get(
                    "last_seen_source_frame_stamp_ns"
                ) is None and source_stamp is not None:
                    self._room_scan_update(
                        last_seen_source_frame_stamp_ns=source_stamp,
                    )
                if pursuit.get("pursuit_authorized") is True and pursuit.get(
                    "state"
                ) in {VISUAL_READY_TO_ALIGN, VISUAL_READY_TO_APPROACH}:
                    self._room_scan_update(
                        scan_active=False,
                        scan_target_acquired=True,
                        scan_transition_pending=True,
                        scan_exhausted=False,
                    )

            visual_session = pursuit.get("state") in {
                VISUAL_READY_TO_ALIGN, VISUAL_READY_TO_APPROACH,
            }
            try:
                arrival = (
                    evaluate_marvin_visual_arrival(preview, now=now)
                    if visual_session else evaluate_marvin_arrival(
                        lock_result,
                        lock_snapshot,
                        selected_identity_id=current_identity_id,
                        now=now,
                    )
                )
            except Exception as exc:
                base["history"].append(history_entry)
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="marvin_arrival_evaluation_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            if (
                not isinstance(arrival, dict)
                or arrival.get("ok") is not True
                or not isinstance(arrival.get("arrived_at_marvin"), bool)
                or (
                    not visual_session
                    and arrival.get("selected_identity_id") != current_identity_id
                )
            ):
                base["history"].append(history_entry)
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="marvin_arrival_evaluation_failed",
                )

            normalized_preview = normalize_marvin_preview(preview)
            observation_timestamp = (
                normalized_preview.get("source_timestamp")
                or normalized_preview.get("vision_timestamp")
                if isinstance(normalized_preview, dict)
                else None
            )

            # Persistent TargetLock arrival remains identity-authoritative,
            # but each arrival confirmation must also be independently
            # supported by the current fresh, semantic Marvin Preview.
            if arrival.get("arrived_at_marvin") and not visual_session:
                try:
                    preview_arrival = evaluate_marvin_visual_arrival(
                        preview, now=now,
                    )
                except Exception as exc:
                    base["history"].append(history_entry)
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="marvin_arrival_preview_confirmation_failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                history_entry["preview_arrival_confirmation"] = preview_arrival
                if (
                    not isinstance(preview_arrival, dict)
                    or preview_arrival.get("ok") is not True
                    or preview_arrival.get("arrived_at_marvin") is not True
                    or preview_arrival.get("fresh") is not True
                    or preview_arrival.get("geometry_valid") is not True
                    or preview_arrival.get("visual_session_authorized") is not True
                ):
                    arrival = dict(
                        arrival,
                        arrived_at_marvin=False,
                        reason="marvin_preview_arrival_not_confirmed",
                    )

            # Once an arrival candidate is pending, the next decision must
            # be based on a distinct source frame. A cached/repeated frame is
            # not a confirmation and cannot be reused to plan motion.
            if arrival_candidate_timestamp is not None and (
                not isinstance(observation_timestamp, str)
                or not observation_timestamp
                or observation_timestamp == arrival_candidate_timestamp
            ):
                history_entry.update(
                    route="arrival_confirmation",
                    selected_action="arrival_confirmation_rejected",
                    arrival_confirmation_reason="observation_not_independent",
                    arrival_candidate_timestamp=arrival_candidate_timestamp,
                    confirmation_observation_timestamp=observation_timestamp,
                )
                base["history"].append(history_entry)
                base["arrival_observations_confirmed"] = 0
                if not stop_completed_action(history_entry):
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="find_marvin_arrival_confirmation_stop_failed",
                    )
                return dict(
                    base,
                    ok=True,
                    completed=False,
                    arrived_at_marvin=False,
                    history=list(base["history"]),
                    reason="find_marvin_arrival_confirmation_not_independent",
                    dry_run=dry_run,
                    next_route="none",
                )

            history_entry["arrival"] = arrival
            if arrival["arrived_at_marvin"]:
                if not (
                    (
                        pursuit.get("state") == "READY_TO_APPROACH"
                        or visual_session
                    )
                    and pursuit.get("pursuit_authorized") is True
                    and (
                        arrival.get("identity_authorized") is True
                        or arrival.get("visual_session_authorized") is True
                    )
                    and arrival.get("fresh") is True
                    and arrival.get("geometry_valid") is True
                ):
                    base["history"].append(history_entry)
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="marvin_arrival_evaluation_inconsistent",
                    )
                if arrival_candidate_timestamp is None:
                    if not isinstance(observation_timestamp, str) or not observation_timestamp:
                        base["history"].append(history_entry)
                        return dict(
                            base,
                            history=list(base["history"]),
                            reason="marvin_arrival_observation_timestamp_missing",
                        )
                    arrival_candidate_timestamp = observation_timestamp
                    history_entry.update(
                        route="arrival_confirmation",
                        selected_action="confirm_arrival",
                        arrival_confirmation_reason="first_qualifying_observation",
                        arrival_candidate_timestamp=arrival_candidate_timestamp,
                        arrival_observations_confirmed=1,
                    )
                    base["arrival_observations_confirmed"] = 1
                    base["history"].append(history_entry)
                    if not stop_completed_action(history_entry):
                        return dict(
                            base,
                            history=list(base["history"]),
                            reason="find_marvin_arrival_confirmation_stop_failed",
                        )
                    # No motion is permitted between the candidate and its
                    # confirmation. The next loop iteration obtains a new
                    # Preview/state bundle without consuming action budget.
                    continue

                history_entry.update(
                    route="arrival",
                    selected_action="arrived_at_marvin",
                    arrival_candidate_timestamp=arrival_candidate_timestamp,
                    confirmation_observation_timestamp=observation_timestamp,
                    arrival_observations_confirmed=2,
                )
                base["arrival_observations_confirmed"] = 2
                base["history"].append(history_entry)
                return dict(
                    base,
                    ok=True,
                    completed=True,
                    arrived_at_marvin=True,
                    arrival=arrival,
                    selected_identity_id=current_identity_id,
                    history=list(base["history"]),
                    reason="arrived_at_marvin",
                    dry_run=dry_run,
                    next_route="arrived",
                )

            if arrival_candidate_timestamp is not None:
                history_entry["arrival_candidate_reset"] = True
                history_entry["arrival_candidate_reset_reason"] = arrival.get(
                    "reason", "second_observation_did_not_arrive",
                )
                arrival_candidate_timestamp = None
                base["arrival_observations_confirmed"] = 0

            scan = self._room_scan_snapshot()
            if scan and scan.get("scan_transition_pending") is True:
                self._room_scan_update(scan_transition_pending=False)
                history_entry.update(
                    route="search",
                    selected_action="motion_authorized_marvin_acquired",
                )
                base["history"].append(history_entry)
                return dict(
                    base, ok=True, completed=False,
                    history=list(base["history"]),
                    reason="find_marvin_search_target_acquired",
                    scan_turn_index=scan.get("scan_turn_index"),
                )

            if local_progress_completed:
                history_entry.update(
                    route="post_progress_reassessment",
                    selected_action="perception_reassessment",
                    local_progress_result=last_local_progress_result,
                )
                base["history"].append(history_entry)
                return dict(
                    base,
                    ok=True,
                    completed=False,
                    arrived_at_marvin=False,
                    history=list(base["history"]),
                    reason="marvin_local_progress_complete",
                    local_progress_result=last_local_progress_result,
                    post_progress_pursuit_state=pursuit.get("state"),
                    selected_identity_id=current_identity_id,
                )

            state = pursuit.get("state")
            authorized = pursuit.get("pursuit_authorized") is True
            if state in {"SEARCHING", "REACQUIRE_REQUIRED"}:
                scan_state = self._room_scan_snapshot()
                if (
                    scan_state and scan_state.get("scan_active") is True
                    and _valid_source_frame_stamp(
                        preview.get("source_frame_stamp_ns")
                        if isinstance(preview, dict) else None
                    ) is None
                ):
                    base["history"].append(history_entry)
                    return dict(
                        base, ok=True, completed=False,
                        history=list(base["history"]),
                        reason="find_marvin_scan_source_frame_baseline_missing",
                    )
                history_entry["route"] = "search"
                history_entry["selected_action"] = "search_step"
                history_entry["pending_scan_turn_index"] = (
                    scan.get("scan_turn_index", 0) if scan else local_scan_turn_index
                )
                if dry_run:
                    history_entry["selected_action"] = "dry_run_search"
                    base["history"].append(history_entry)
                    return dict(
                        base, ok=True, dry_run=True, next_route="search",
                        history=list(base["history"]),
                        reason="find_marvin_dry_run",
                    )
                # An executor-side exception may occur after dispatch, so the
                # request consumes a shared bounded action opportunity first.
                base["actions_executed"] += 1
                # Once dispatch is attempted, delivery may be uncertain even
                # if the executor does not confirm motion. Record the shared
                # physical/delivery-uncertain action opportunity consistently.
                history_entry["action_budget_consumed"] = True
                try:
                    step = self.execute_marvin_search_step(
                        pursuit,
                        scan_turn_index=(
                            scan.get("scan_turn_index", 0)
                            if scan else local_scan_turn_index
                        ),
                        selected_identity_id=current_identity_id,
                        preview_result=preview,
                        target_lock_snapshot=lock_snapshot,
                        bridge_result=evidence.get("bridge_result"),
                        now=now,
                    )
                except Exception as exc:
                    base["history"].append(history_entry)
                    if not stop_completed_action(history_entry):
                        return dict(
                            base, history=list(base["history"]),
                            reason="find_marvin_post_action_stop_failed",
                        )
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="find_marvin_search_step_exception",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                history_entry["search_step_result"] = step
                if isinstance(step, dict):
                    planner = step.get("planner")
                    search_action = step.get("search_action")
                    if search_action is None and isinstance(planner, dict):
                        search_action = planner.get("selected_search_action")
                    history_entry["search_action"] = search_action
                    history_entry["selected_action"] = step.get(
                        "decision", "search_step",
                    )
                    history_entry["executed_primitive"] = step.get(
                        "executed_primitive"
                    )
                    history_entry["replan_required"] = (
                        step.get("replan_required") is True
                    )
                base["history"].append(history_entry)
                # An executor attempt is treated as potentially dispatched:
                # the autonomous runtime stops before it inspects the result
                # or obtains another Preview.
                if not stop_completed_action(history_entry):
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_post_action_stop_failed",
                    )
                if (
                    isinstance(step, dict)
                    and step.get("clearance_wait_timed_out") is True
                    and step.get("reason") == "find_marvin_clearance_wait_timeout"
                ):
                    # A pre-turn veto dispatched no motion. An active-monitor
                    # stop did dispatch a bounded turn, so retain that single
                    # attempted physical-action opportunity while passive
                    # clearance rechecks consume none.
                    attempted_turn = (
                        step.get("clearance_wait_origin") == "active_turn_monitor"
                        and step.get("interrupted_turn_detected") is True
                    )
                    if not attempted_turn:
                        base["actions_executed"] = max(
                            0, base["actions_executed"] - 1,
                        )
                    history_entry["action_budget_consumed"] = attempted_turn
                    history_entry["clearance_wait_timeout"] = True
                    return dict(
                        base,
                        ok=True,
                        completed=True,
                        arrived_at_marvin=False,
                        history=list(base["history"]),
                        reason="find_marvin_clearance_wait_timeout",
                        clearance_wait=step,
                    )
                if not isinstance(step, dict) or step.get("ok") is not True:
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_search_step_failed",
                    )
                if history_entry["search_action"] == "search_complete":
                    return dict(
                        base, ok=True, history=list(base["history"]),
                        reason="find_marvin_search_complete",
                    )
                if step.get("motion_executed") is not True:
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_search_step_no_motion",
                    )
                if step.get("replan_required") is not True:
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_search_step_replan_required",
                    )
                # The pure policy counts successful scan turns only.  Preview
                # checks and failed/non-motion requests never advance it.
                if history_entry["search_action"] not in {
                    "turn_left", "turn_right",
                }:
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_search_action_malformed",
                    )
                if self._room_scan_snapshot() is not None:
                    bridge_status = self.robot.status()
                    if not _bridge_status_zero(bridge_status):
                        return dict(
                            base, history=list(base["history"]),
                            reason="find_marvin_search_bridge_not_zero_after_stop",
                            bridge_status=bridge_status,
                        )
                    completed_index = self._room_scan_snapshot()
                    baseline_stamp = _valid_source_frame_stamp(
                        preview.get("source_frame_stamp_ns")
                        if isinstance(preview, dict) else None
                    )
                    if baseline_stamp is None:
                        return dict(
                            base, history=list(base["history"]),
                            reason="find_marvin_post_turn_source_frame_baseline_missing",
                        )
                    completed_turn = completed_index.get("scan_turn_index", 0) + 1
                    self._room_scan_update(
                        scan_turn_index=completed_turn,
                        last_completed_scan_turn=completed_turn,
                        pre_turn_source_frame_stamp_ns=baseline_stamp,
                        last_seen_source_frame_stamp_ns=baseline_stamp,
                        awaiting_new_source_frame=True,
                        scan_exhausted=(completed_turn >= completed_index.get(
                            "scan_max_turns", self.MARVIN_SCAN_MAX_TURNS
                        )),
                    )
                search_history.append({
                    "selected_search_action": history_entry["search_action"],
                })
                if self._room_scan_snapshot() is None:
                    local_scan_turn_index += 1
                continue

            if state not in {
                "READY_TO_APPROACH", VISUAL_READY_TO_ALIGN,
                VISUAL_READY_TO_APPROACH,
            } or not authorized:
                base["history"].append(history_entry)
                return dict(
                    base,
                    ok=True,
                    dry_run=dry_run,
                    next_route="none",
                    history=list(base["history"]),
                    reason=self._find_marvin_controller_pause_reason(
                        state, authorized,
                    ),
                )

            history_entry["route"] = "pursuit"
            history_entry["selected_action"] = "pursuit_step"
            if dry_run:
                history_entry["selected_action"] = "dry_run_pursuit"
                base["history"].append(history_entry)
                return dict(
                    base, ok=True, dry_run=True, next_route="pursuit",
                    history=list(base["history"]),
                    reason="find_marvin_dry_run",
                )
            # Count the one-step request before invoking it.  A transport-side
            # exception leaves delivery uncertain, so it still consumes this
            # bounded action opportunity and cannot be retried implicitly.
            base["actions_executed"] += 1
            try:
                action_budget_remaining = max_actions - (
                    base["actions_executed"] - 1
                )
                step = self.execute_marvin_pursuit_step(
                    preview,
                    lock_result,
                    lock_snapshot,
                    selected_identity_id=evidence.get("selected_identity_id"),
                    identity_evidence=evidence.get("identity_evidence"),
                    bridge_result=evidence.get("bridge_result"),
                    now=now,
                    local_progress_action_budget_remaining=(
                        action_budget_remaining
                    ),
                )
            except Exception as exc:
                base["history"].append(history_entry)
                if not stop_completed_action(history_entry):
                    return dict(
                        base, history=list(base["history"]),
                        reason="find_marvin_post_action_stop_failed",
                    )
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_pursuit_step_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            history_entry["pursuit_step_result"] = step
            nested_action_count = (
                step.get("local_progress_physical_actions")
                if isinstance(step, dict) else None
            )
            if (
                isinstance(step, dict)
                and step.get("local_progress_budget_blocked") is True
            ):
                base["actions_executed"] -= 1
                history_entry["action_budget_consumed"] = False
            if nested_action_count is not None:
                if (
                    not isinstance(nested_action_count, int)
                    or isinstance(nested_action_count, bool)
                    or not 0 <= nested_action_count <= 4
                    or nested_action_count > action_budget_remaining
                ):
                    base["history"].append(history_entry)
                    if not stop_completed_action(history_entry):
                        return dict(
                            base,
                            history=list(base["history"]),
                            reason="find_marvin_post_action_stop_failed",
                        )
                    return dict(
                        base,
                        history=list(base["history"]),
                        reason="find_marvin_local_progress_action_count_invalid",
                    )
                # Replace the controller's reserved one-step opportunity with
                # the actual nested physical-action count.
                base["actions_executed"] += nested_action_count - 1
                history_entry["action_budget_consumed"] = (
                    nested_action_count > 0
                )
                history_entry["action_budget_count"] = nested_action_count
            if isinstance(step, dict):
                history_entry["selected_action"] = step.get(
                    "decision", "pursuit_step",
                )
                history_entry["executed_primitive"] = step.get(
                    "executed_primitive"
                )
                history_entry["replan_required"] = (
                    step.get("replan_required") is True
                )
            base["history"].append(history_entry)
            # Treat a failed executor result as potentially dispatched too;
            # STOP is required before the controller can return or replan.
            if (
                (not isinstance(step, dict) or step.get("stop_required") is not False)
                and not stop_completed_action(history_entry)
            ):
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_post_action_stop_failed",
                )
            stale_replan = bool(
                isinstance(step, dict) and step.get("stale_replan") is True
            )
            budget_consumed = (
                step.get("local_progress_physical_actions", 0) > 0
                if isinstance(step, dict)
                and "local_progress_physical_actions" in step
                else False
                if isinstance(step, dict)
                and step.get("local_progress_budget_blocked") is True
                else not (
                    stale_replan
                    and isinstance(step, dict)
                    and step.get("action_budget_consumed") is False
                )
            )
            history_entry["action_budget_consumed"] = budget_consumed
            if (
                isinstance(step, dict)
                and step.get("local_progress_terminal")
                and step.get("local_progress_terminal")
                != "LOCAL_PROGRESS_COMPLETE"
            ):
                terminal = step["local_progress_terminal"]
                return dict(
                    base,
                    ok=True,
                    completed=False,
                    arrived_at_marvin=False,
                    history=list(base["history"]),
                    reason="marvin_local_progress_terminal",
                    local_progress_terminal=terminal,
                    local_progress_result=step.get("local_progress_result"),
                )
            if isinstance(step, dict) and step.get("local_progress_budget_blocked") is True:
                return dict(
                    base,
                    ok=True,
                    completed=False,
                    arrived_at_marvin=False,
                    history=list(base["history"]),
                    reason="marvin_local_progress_action_budget_insufficient",
                )
            if stale_replan:
                if not budget_consumed:
                    base["actions_executed"] -= 1
                    base["stale_replans"] += 1
                    if (
                        base["stale_replans"]
                        > self.MAX_NONPHYSICAL_STALE_REPLANS
                    ):
                        return dict(
                            base,
                            history=list(base["history"]),
                            reason="find_marvin_nonphysical_stale_replan_limit_reached",
                        )
                # STOP has completed.  The next loop iteration begins by
                # obtaining a new Preview/state bundle; it never reuses this
                # geometry or perception snapshot and never resends in-place.
                continue
            if not isinstance(step, dict) or step.get("ok") is not True:
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_pursuit_step_failed",
                )
            if (
                step.get("motion_executed") is not True
                and not (
                    stale_replan
                    and step.get("motion_possible") is True
                    and budget_consumed
                )
            ):
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_pursuit_step_no_motion",
                )
            if step.get("replan_required") is not True:
                return dict(
                    base,
                    history=list(base["history"]),
                    reason="find_marvin_pursuit_step_replan_required",
                )
            if (
                step.get("local_progress_terminal")
                == "LOCAL_PROGRESS_COMPLETE"
            ):
                local_progress_completed = True
                last_local_progress_result = step.get(
                    "local_progress_result",
                )

        return dict(
            base,
            ok=True,
            history=list(base["history"]),
            reason="find_marvin_action_limit_reached",
        )

    @staticmethod
    def _find_marvin_controller_pause_reason(state, authorized):
        """Map non-motion pursuit evidence to an explicit bounded outcome."""
        if state in {"REACQUIRE_REQUIRED", "SAME_IDENTITY_REACQUIRED"}:
            return "reacquire_required"
        if state == "INSUFFICIENT_EVIDENCE":
            return "insufficient_evidence"
        if state in {
            "READY_TO_APPROACH", VISUAL_READY_TO_ALIGN,
            VISUAL_READY_TO_APPROACH,
        } and not authorized:
            return "find_marvin_pursuit_not_authorized"
        return "paused_for_search"

    @staticmethod
    def _marvin_v2_preview_is_verified(preview):
        """Validate fresh semantic acquisition or locked post-action tracking."""
        if not isinstance(preview, dict):
            return False
        identity_stamp = _valid_source_frame_stamp(
            preview.get("identity_source_frame_stamp_ns")
        )
        tracker = preview.get("opencv_tracker")
        if not isinstance(tracker, dict):
            return False
        tracker_stamp = _valid_source_frame_stamp(
            tracker.get("source_frame_stamp_ns")
        )
        quality = tracker.get("quality")
        threshold = tracker.get("threshold")
        bbox = tracker.get("bbox")
        width = tracker.get("image_width")
        height = tracker.get("image_height")
        try:
            validated_bbox = MarvinLocalTracker._validate_bbox(
                bbox, width, height,
            )
        except (TypeError, ValueError):
            return False
        preview_bbox = preview.get("bbox")
        tracker_bbox_matches = bool(
            isinstance(preview_bbox, dict)
            and all(
                isinstance(preview_bbox.get(key), (int, float))
                and not isinstance(preview_bbox.get(key), bool)
                and math.isfinite(preview_bbox[key])
                and float(preview_bbox[key]) == float(bbox[key])
                for key in ("x1", "y1", "x2", "y2")
            )
            and preview.get("image_width") == width
            and preview.get("image_height") == height
        )
        common_current_tracker = bool(
            tracker.get("active") is True
            and tracker.get("matched") is True
            and isinstance(quality, (int, float))
            and not isinstance(quality, bool)
            and math.isfinite(quality)
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(threshold)
            and quality >= threshold
            and tracker_stamp is not None
            and validated_bbox is not None
            and tracker_bbox_matches
        )
        if (preview.get("identity_source") == "marvin_locked_tracker_continuity"
                and preview.get("post_action_tracker_continuity") is True):
            episode = preview.get("marvin_tracking_episode")
            return bool(
                preview.get("identity_confirmed") is True
                and isinstance(episode, dict)
                and episode.get("state") == "POST_ACTION_TRACKED"
                and type(identity_stamp) is int
                and tracker_stamp is not None and tracker_stamp > identity_stamp
                and type(episode.get("post_action_source_frame_stamp_ns")) is int
                and tracker_stamp > episode["post_action_source_frame_stamp_ns"]
                and common_current_tracker
            )
        return bool(
            preview.get("identity_confirmed") is True
            and preview.get("identity_source") == "gemini_marvin_candidate_selection"
            and identity_stamp is not None
            and tracker_stamp is not None
            and tracker_stamp > identity_stamp
            and common_current_tracker
        )

    def execute_marvin_pursuit_step(
        self,
        preview_result,
        target_lock_result,
        target_lock_snapshot,
        *,
        selected_identity_id=None,
        identity_evidence=None,
        bridge_result=None,
        now=None,
        local_progress_action_budget_remaining=None,
    ):
        """Execute one bounded Marvin decision from current identity evidence.

        This intentionally has no mission loop, search behavior, or cached
        authority. Every invocation evaluates current Preview/TargetLock
        evidence. Alignment remains its existing turn path; an authorized
        approach uses the runtime-owned local-progress handoff when installed.
        """
        pursuit = evaluate_marvin_pursuit_state(
            preview_result,
            target_lock_result,
            target_lock_snapshot,
            selected_identity_id=selected_identity_id,
            identity_evidence=identity_evidence,
            bridge_result=bridge_result,
            now=now,
        )
        pursuit_state = (
            pursuit.get("state")
            if isinstance(pursuit, dict)
            else "INSUFFICIENT_EVIDENCE"
        )
        base = {
            "ok": False,
            "pursuit_state": pursuit_state,
            "pursuit": pursuit if isinstance(pursuit, dict) else None,
            "decision": "no_motion",
            "executed_primitive": None,
            "motion_executed": False,
            "replan_required": False,
            "forward_safety": None,
            "avoidance_result": None,
            "reason": None,
        }
        if (
            not isinstance(pursuit, dict)
            or pursuit.get("state") not in {
                "READY_TO_APPROACH", VISUAL_READY_TO_ALIGN,
                VISUAL_READY_TO_APPROACH,
            }
            or pursuit.get("pursuit_authorized") is not True
        ):
            return dict(base, reason="marvin_pursuit_not_authorized")

        session = self._current_lidar_session()
        if session is None or self.world_model is None:
            return dict(base, reason="lidar_producer_session_unavailable")
        if pursuit_state == VISUAL_READY_TO_ALIGN:
            direction = (
                "LEFT" if pursuit.get("horizontal_error", 0.0) < 0.0 else "RIGHT"
            )
            duration = self._marvin_guarded_approach_turn_duration(
                pursuit.get("horizontal_error", 0.0)
            )
            if duration is None:
                return dict(base, reason="marvin_visual_alignment_geometry_inconsistent")
            try:
                turn = self._execute_target_directed_turn(
                    direction,
                    self.MARVIN_CENTERING_TURN_SPEED,
                    duration,
                    expected_lidar_session=session,
                    safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                )
            except Exception as exc:
                return dict(
                    base, decision="align_" + direction.lower(),
                    executed_primitive="guarded_turn_" + direction.lower(),
                    reason="marvin_visual_alignment_exception",
                    error=str(exc), error_type=type(exc).__name__,
                )
            turn_ok = bool(
                isinstance(turn, dict)
                and turn.get("ok") is True
                and turn.get("permitted") is True
            )
            return dict(
                base,
                ok=turn_ok,
                decision="align_" + direction.lower(),
                executed_primitive="guarded_turn_" + direction.lower(),
                motion_executed=turn_ok,
                replan_required=turn_ok,
                alignment_result=turn,
                reason=(
                    "marvin_visual_alignment_complete" if turn_ok
                    else (turn.get("reason", "marvin_visual_alignment_denied")
                          if isinstance(turn, dict) else "marvin_visual_alignment_denied")
                ),
            )
        local_progress_handler = getattr(
            self, "marvin_local_progress_with_avoidance_handler", None,
        )
        if callable(local_progress_handler):
            if (
                not isinstance(local_progress_action_budget_remaining, int)
                or isinstance(local_progress_action_budget_remaining, bool)
                or local_progress_action_budget_remaining < 4
            ):
                return dict(
                    base,
                    decision="local_progress_budget_blocked",
                    local_progress_budget_blocked=True,
                    action_budget_consumed=False,
                    stop_required=False,
                    reason="marvin_local_progress_action_budget_insufficient",
                )
            try:
                local_progress = local_progress_handler(
                    remaining_actions=local_progress_action_budget_remaining,
                )
            except Exception as exc:
                return dict(
                    base,
                    decision="local_progress_handoff",
                    local_progress_physical_actions=0,
                    local_progress_terminal="LOCAL_PROGRESS_EXECUTION_FAILED",
                    action_budget_consumed=False,
                    stop_required=True,
                    reason="marvin_local_progress_handoff_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            if not isinstance(local_progress, dict):
                return dict(
                    base,
                    decision="local_progress_handoff",
                    local_progress_physical_actions=0,
                    local_progress_terminal="LOCAL_PROGRESS_EXECUTION_FAILED",
                    action_budget_consumed=False,
                    stop_required=True,
                    reason="marvin_local_progress_handoff_result_malformed",
                )
            terminal = local_progress.get("terminal_state")
            physical_actions = local_progress.get("physical_actions")
            valid_count = (
                isinstance(physical_actions, int)
                and not isinstance(physical_actions, bool)
                and 0 <= physical_actions <= 4
                and physical_actions <= local_progress_action_budget_remaining
            )
            if not valid_count:
                return dict(
                    base,
                    decision="local_progress_handoff",
                    local_progress_result=local_progress,
                    local_progress_physical_actions=0,
                    local_progress_terminal="LOCAL_PROGRESS_EXECUTION_FAILED",
                    action_budget_consumed=False,
                    stop_required=True,
                    reason="marvin_local_progress_action_count_invalid",
                )
            if terminal == "LOCAL_PROGRESS_COMPLETE":
                if physical_actions < 1 or local_progress.get("bridge_stopped") is not True:
                    terminal = "LOCAL_PROGRESS_EXECUTION_FAILED"
                else:
                    return dict(
                        base,
                        ok=True,
                        decision="local_progress_handoff",
                        executed_primitive="local_progress_with_avoidance",
                        motion_executed=True,
                        replan_required=True,
                        local_progress_result=local_progress,
                        local_progress_physical_actions=physical_actions,
                        local_progress_terminal=terminal,
                        reason="marvin_local_progress_complete",
                    )
            terminal_reasons = {
                "LOCAL_PROGRESS_BLOCKED",
                "LOCAL_PROGRESS_SAFETY_VETO",
                "LOCAL_PROGRESS_EXECUTION_FAILED",
                "LOCAL_PROGRESS_OWNERSHIP_REJECTED",
                "LOCAL_PROGRESS_MAX_STEPS_REACHED",
            }
            if terminal in terminal_reasons:
                return dict(
                    base,
                    ok=True,
                    decision="local_progress_handoff",
                    executed_primitive=None,
                    motion_executed=physical_actions > 0,
                    replan_required=False,
                    local_progress_result=local_progress,
                    local_progress_physical_actions=physical_actions,
                    local_progress_terminal=terminal,
                    action_budget_consumed=physical_actions > 0,
                    stop_required=(
                        terminal != "LOCAL_PROGRESS_OWNERSHIP_REJECTED"
                        or physical_actions > 0
                    ),
                    reason=local_progress.get("reason", terminal.lower()),
                )
            return dict(
                base,
                decision="local_progress_handoff",
                local_progress_result=local_progress,
                local_progress_physical_actions=physical_actions,
                local_progress_terminal="LOCAL_PROGRESS_EXECUTION_FAILED",
                action_budget_consumed=physical_actions > 0,
                reason="marvin_local_progress_terminal_unknown",
            )
        try:
            lidar = self.world_model.get_lidar_obstacles(
                expected_session=session, now=now,
            )
            safety = evaluate_local_motion_safety(
                lidar,
                expected_session=session,
                linear_x=self.FIND_APPROACH_FORWARD_SPEED,
                duration=self.FIND_APPROACH_FORWARD_SECONDS,
                now=now,
            )
        except Exception as exc:
            return dict(
                base,
                reason="marvin_pursuit_lidar_read_or_evaluation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        base["forward_safety"] = safety
        if not self._marvin_pursuit_lidar_is_trusted(lidar, safety, session):
            if self._marvin_lidar_reason_is_stale(lidar, safety):
                return dict(
                    base,
                    ok=True,
                    decision="approach_forward",
                    executed_primitive=None,
                    stale_replan=True,
                    stale_replan_classification="NONPHYSICAL_STALE_REPLAN",
                    action_budget_consumed=False,
                    replan_required=True,
                    reason="marvin_pursuit_pre_dispatch_lidar_stale",
                )
            return dict(base, reason="marvin_pursuit_lidar_not_trusted")

        if safety.get("permitted") is True:
            # The initial geometry check above establishes the route.  Refresh
            # the same active producer-bound snapshot and interlock directly
            # before transport so an expiring authorization is never reused.
            dispatch = self._prepare_marvin_forward_dispatch(
                expected_session=session,
                now=now,
            )
            base["pre_dispatch_forward_safety"] = dispatch.get("safety")
            base["pre_dispatch_interlock"] = dispatch.get("interlock")
            if dispatch.get("status") == "nonphysical_stale_replan":
                return dict(
                    base,
                    ok=True,
                    decision="approach_forward",
                    executed_primitive=None,
                    stale_replan=True,
                    stale_replan_classification="NONPHYSICAL_STALE_REPLAN",
                    action_budget_consumed=False,
                    replan_required=True,
                    reason=dispatch.get("reason"),
                )
            if dispatch.get("status") != "ready":
                return dict(
                    base,
                    decision="approach_forward",
                    executed_primitive=None,
                    reason=dispatch.get(
                        "reason", "marvin_pursuit_pre_dispatch_not_authorized"
                    ),
                )
            try:
                forward = self.robot.move_forward(
                    speed=self.FIND_APPROACH_FORWARD_SPEED,
                    seconds=self.FIND_APPROACH_FORWARD_SECONDS,
                )
            except Exception as exc:
                return dict(
                    base,
                    decision="approach_forward",
                    executed_primitive="forward",
                    reason="marvin_pursuit_forward_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            forward = self._normalize_marvin_bounded_forward_result(
                forward,
                speed=self.FIND_APPROACH_FORWARD_SPEED,
                duration=self.FIND_APPROACH_FORWARD_SECONDS,
            )
            classification = self._classify_marvin_forward_result(forward)
            if classification == "NONPHYSICAL_STALE_REPLAN":
                return dict(
                    base,
                    ok=True,
                    decision="approach_forward",
                    executed_primitive="forward",
                    forward_result=forward,
                    stale_replan=True,
                    stale_replan_classification=classification,
                    action_budget_consumed=False,
                    replan_required=True,
                    reason="marvin_pursuit_nonphysical_stale_replan",
                )
            if classification == "PHYSICAL_OR_UNCERTAIN_STALE_REPLAN":
                transport = forward.get("transport_result") if isinstance(forward, dict) else None
                bridge_completed = self._is_canonical_marvin_bounded_forward_result(
                    transport,
                    speed=self.FIND_APPROACH_FORWARD_SPEED,
                    duration=self.FIND_APPROACH_FORWARD_SECONDS,
                )
                return dict(
                    base,
                    ok=True,
                    decision="approach_forward",
                    executed_primitive="forward",
                    forward_result=forward,
                    stale_replan=True,
                    stale_replan_classification=classification,
                    action_budget_consumed=True,
                    motion_executed=bridge_completed,
                    motion_possible=True,
                    replan_required=True,
                    reason="marvin_pursuit_physical_or_uncertain_stale_replan",
                )
            forward_ok = bool(
                isinstance(forward, dict)
                and forward.get("ok") is True
                and forward.get("executed") is True
            )
            if not forward_ok:
                return dict(
                    base,
                    decision="approach_forward",
                    executed_primitive="forward",
                    forward_result=forward,
                    reason=(
                        forward.get("reason", "marvin_pursuit_forward_failed")
                        if isinstance(forward, dict)
                        else "marvin_pursuit_forward_failed"
                    ),
                )
            return dict(
                base,
                ok=True,
                decision="approach_forward",
                executed_primitive="forward",
                motion_executed=True,
                replan_required=True,
                forward_result=forward,
                reason="marvin_pursuit_forward_complete",
            )

        # A trusted blocked forward path is an obstacle condition, never an
        # arrival inference.  The delegated step owns at most one primitive.
        try:
            avoidance = self.execute_local_obstacle_avoidance_step(
                expected_lidar_session=session,
                now=now,
            )
        except Exception as exc:
            return dict(
                base,
                decision="avoidance_required",
                reason="marvin_pursuit_avoidance_exception",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if not isinstance(avoidance, dict):
            return dict(
                base,
                decision="avoidance_required",
                reason="marvin_pursuit_avoidance_result_malformed",
            )
        executed = avoidance.get("motion_executed") is True
        return dict(
            base,
            ok=avoidance.get("ok") is True and executed,
            decision="avoidance_required",
            executed_primitive=avoidance.get("executed_primitive"),
            motion_executed=executed,
            replan_required=executed,
            avoidance_result=avoidance,
            reason=avoidance.get("reason", "marvin_pursuit_avoidance_failed"),
        )

    def execute_single_marvin_approach_step(
        self, *, expected_lidar_session, linear_speed, duration,
        target_tracker=None, camera_model=None, dispatch_guard=None, target_range_validator=None,
    ):
        """Run one bounded Marvin forward primitive, with no avoidance branch."""
        base = {"ok": False, "decision": "no_motion", "executed_primitive": None,
                "motion_executed": False, "forward_safety": None, "reason": None}
        if (linear_speed != self.FIND_APPROACH_FORWARD_SPEED
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or not 0 < duration <= self.FIND_APPROACH_FORWARD_SECONDS):
            return dict(base, reason="marvin_single_approach_parameters_invalid")
        if expected_lidar_session is None or self.world_model is None:
            return dict(base, reason="lidar_producer_session_unavailable")
        try:
            lidar = self.world_model.get_lidar_obstacles(expected_session=expected_lidar_session)
            if target_tracker is not None:
                from marvin_lidar_standoff import TARGET_STANDOFF_M, evaluate_marvin_lidar_standoff
                standoff = evaluate_marvin_lidar_standoff(
                    target_tracker, lidar, camera_model, expected_session=expected_lidar_session,
                )
                if target_range_validator is not None:
                    standoff = target_range_validator(target_tracker, lidar)
                base["target_standoff"] = standoff
                base["action_lidar_evidence"] = {
                    "producer_session": lidar.get("producer_session"),
                    "acquisition_sequence": lidar.get("acquisition_sequence"),
                    "source": lidar.get("source"),
                    "received_monotonic_seconds": lidar.get("received_monotonic_seconds"),
                    "effective_age_seconds": lidar.get("effective_age_seconds"),
                }
                if (standoff.get("ok") is not True or standoff.get("candidate_at_standoff") is True
                        or standoff.get("arrived_at_marvin") is True
                        or (target_range_validator is not None and
                            standoff.get("target_range_association_trusted") is not True)):
                    return dict(base, reason="marvin_single_approach_target_standoff_vetoed")
                duration = min(duration, (standoff["target_distance_m"] - TARGET_STANDOFF_M) / linear_speed)
            base["duration"] = duration
            safety = evaluate_local_motion_safety(
                lidar, expected_session=expected_lidar_session,
                linear_x=linear_speed, duration=duration,
            )
        except Exception as exc:
            return dict(base, reason="marvin_single_approach_lidar_read_or_evaluation_failed",
                        error=str(exc), error_type=type(exc).__name__)
        base["forward_safety"] = safety
        if not self._marvin_pursuit_lidar_is_trusted(lidar, safety, expected_lidar_session):
            return dict(base, reason="marvin_single_approach_lidar_not_trusted")
        if safety.get("permitted") is not True:
            return dict(base, reason="marvin_single_approach_translation_vetoed")
        try:
            if dispatch_guard is not None and dispatch_guard() is not True:
                return dict(base, reason="marvin_motion_observation_stale_or_preempted")
            self._emit_marvin_command_diagnostic(
                "start", start_monotonic_seconds=time.monotonic(), linear_x=linear_speed, linear_y=0.0,
                angular_z=0.0, duration=duration,
            )
            forward = self.robot.move_forward(
                speed=linear_speed,
                seconds=duration,
            )
            self._emit_marvin_command_diagnostic(
                "complete", completion_monotonic_seconds=time.monotonic(),
                bridge_acknowledgement=forward,
            )
        except Exception as exc:
            return dict(base, decision="approach_forward", executed_primitive="forward",
                        reason="marvin_single_approach_forward_exception",
                        error=str(exc), error_type=type(exc).__name__)
        forward = self._normalize_marvin_bounded_forward_result(
            forward,
            speed=linear_speed,
            duration=duration,
        )
        forward_ok = bool(isinstance(forward, dict) and forward.get("ok") is True
                          and forward.get("executed") is True)
        if not forward_ok:
            return dict(base, decision="approach_forward", executed_primitive="forward",
                        forward_result=forward,
                        reason=(forward.get("reason", "marvin_single_approach_forward_failed")
                                if isinstance(forward, dict) else "marvin_single_approach_forward_failed"))
        return dict(base, ok=True, decision="approach_forward", executed_primitive="forward",
                    motion_executed=True, forward_result=forward,
                    reason="marvin_single_approach_forward_complete")

    def execute_guarded_marvin_lateral_step(self, *, expected_lidar_session,
            linear_y, duration, dispatch_guard, selection_validator):
        """One pure lateral command through the existing Bridge health interlock."""
        from marvin_local_obstacle_avoidance import (
            LOCAL_AVOIDANCE_STRAFE_SPEED_MPS, LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS)
        base = {"ok": False, "motion_executed": False, "lateral_result": None}
        if (type(linear_y) not in (int, float) or not math.isfinite(linear_y)
                or abs(linear_y) != LOCAL_AVOIDANCE_STRAFE_SPEED_MPS
                or type(duration) not in (int, float) or not math.isfinite(duration)
                or not 0 < duration <= LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS):
            return dict(base, reason="marvin_lateral_parameters_invalid")
        interlock = getattr(self.robot, "forward_interlock", None)
        if interlock is None or not callable(getattr(self.robot, "move_lateral", None)):
            return dict(base, reason="marvin_lateral_interlock_unavailable")
        evidence = {}

        def final_guard():
            if not dispatch_guard():
                evidence["reason"] = "marvin_motion_observation_stale_or_preempted"
                return False
            try:
                lidar = self.world_model.get_lidar_obstacles(expected_session=expected_lidar_session)
                safe = evaluate_local_motion_safety(lidar, expected_session=expected_lidar_session,
                    linear_y=linear_y, duration=duration, lateral_swept_footprint=True)
                selection = selection_validator(lidar)
                evidence.update(lateral_safety=safe, local_detour=selection,
                    action_lidar_evidence={k: lidar.get(k) for k in (
                        "producer_session", "acquisition_sequence", "source", "received_monotonic_seconds",
                        "effective_age_seconds")})
                if not safe["permitted"] or not selection.get("accepted"):
                    evidence["reason"] = "marvin_local_detour_jit_veto"
                    return False
                permitted, reason = interlock.refresh()
                if not permitted:
                    evidence["reason"] = reason
                    return False
            except Exception as exc:
                evidence.update(reason="marvin_lateral_lidar_evaluation_failed", error=str(exc))
                return False
            return dispatch_guard()

        if not final_guard():
            return dict(base, **evidence)
        try:
            self._emit_marvin_command_diagnostic("start", start_monotonic_seconds=time.monotonic(),
                linear_x=0.0, linear_y=linear_y, angular_z=0.0, duration=duration)
            result = self.robot.move_lateral(speed=linear_y, seconds=duration, dispatch_guard=final_guard)
            self._emit_marvin_command_diagnostic("complete", completion_monotonic_seconds=time.monotonic(),
                bridge_acknowledgement=result)
        except Exception as exc:
            return dict(base, **dict(evidence, reason="marvin_lateral_transport_failed", error=str(exc)))
        canonical = (isinstance(result, dict) and result.get("ok") is True
            and result.get("action") == "motion" and result.get("mode") == "bounded"
            and result.get("automatic_stop") is True and result.get("returned_immediately") is False
            and result.get("linear_x") == 0 and result.get("angular_z") == 0
            and result.get("linear_y") == linear_y and result.get("duration") == duration
            and not result.get("error") and result.get("executed") is not False)
        return dict(dict(base, **evidence), ok=canonical, motion_executed=canonical,
            lateral_result=result, reason="marvin_lateral_step_complete" if canonical else
                ((result.get("reason") or result.get("error") or "marvin_lateral_step_failed")
                 if isinstance(result, dict) else "marvin_lateral_response_invalid"))

    def execute_guarded_local_forward(self, *, expected_lidar_session, now=None):
        """Run the established single bounded forward primitive for local use.

        This is deliberately a semantic-free delegate, not another motion
        implementation.  It uses the canonical Find Marvin speed and 0.50 s
        bound, local-motion envelope check, and forward-interlock dispatch
        gate owned by ``execute_single_marvin_approach_step``.
        """
        del now  # The delegated primitive obtains its own current snapshot.
        return self.execute_single_marvin_approach_step(
            expected_lidar_session=expected_lidar_session,
            linear_speed=self.FIND_APPROACH_FORWARD_SPEED,
            duration=self.FIND_APPROACH_FORWARD_SECONDS,
        )

    @staticmethod
    def _normalize_marvin_bounded_forward_result(result, *, speed, duration):
        """Normalize only a complete, accepted bounded Bridge forward result.

        ``RobotBridgeClient.move_forward`` returns the Bridge ``/motion``
        schema, while older local-forward callers returned ``executed``.
        A canonical bounded result is promoted to that internal field only
        after its exact speed, duration, non-streaming completion, and
        automatic-stop evidence have all been verified.
        """
        if not isinstance(result, dict) or result.get("ok") is not True:
            return result
        if result.get("error") or any(result.get(key) is False for key in (
            "forwarded", "confirmed_forwarded", "normal_completion",
            "transport_accepted", "transport_began", "transport_returned",
        )) or result.get("bounded_forward_invalidated") is True:
            return result
        if result.get("executed") is True:
            return result
        canonical = BehaviorManager._is_canonical_marvin_bounded_forward_result(
            result, speed=speed, duration=duration,
        )
        return dict(result, executed=True) if canonical else result

    @staticmethod
    def _is_canonical_marvin_bounded_forward_result(result, *, speed, duration):
        if not isinstance(result, dict) or result.get("ok") is not True:
            return False
        try:
            return bool(
                result.get("action") == "motion"
                and result.get("mode") == "bounded"
                and result.get("automatic_stop") is True
                and result.get("returned_immediately") is False
                and float(result.get("linear_x")) == float(speed)
                and float(result.get("angular_z")) == 0.0
                and float(result.get("duration")) == float(duration)
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _marvin_lidar_reason_is_stale(lidar, safety, interlock_reason=None):
        stale_reasons = {"stale", "stale_lidar", "not_fresh"}
        values = (
            lidar.get("reason") if isinstance(lidar, dict) else None,
            safety.get("reason") if isinstance(safety, dict) else None,
            interlock_reason,
        )
        return any(value in stale_reasons for value in values)

    @staticmethod
    def _classify_marvin_forward_result(result):
        """Classify only recognized stale outcomes; all others fail closed."""
        if not isinstance(result, dict):
            return "REAL_FAILURE"
        stale_reason = result.get("reason") or result.get("error")
        if stale_reason not in {"stale", "stale_lidar", "not_fresh"}:
            return "REAL_FAILURE"
        transport = result.get("transport_result")
        transport_started = bool(
            result.get("transport_attempted") is True
            or result.get("forwarded") is True
            or isinstance(transport, dict)
            or result.get("delivery_uncertain") is True
        )
        if (
            result.get("forwarded") is False
            and not transport_started
            and not isinstance(transport, dict)
        ):
            return "NONPHYSICAL_STALE_REPLAN"
        return "PHYSICAL_OR_UNCERTAIN_STALE_REPLAN"

    def _prepare_marvin_forward_dispatch(self, *, expected_session, now):
        """Refresh current LiDAR/interlock immediately before Marvin forward.

        This never acquires LiDAR itself.  It uses the active runtime worker's
        producer-bound World Model snapshot and the already configured
        interlock, preserving the normal freshness and geometry contracts.
        """
        result = {"status": "denied", "reason": None, "safety": None,
                  "interlock": None}
        if expected_session is None or self.world_model is None:
            return dict(result, reason="lidar_producer_session_unavailable")
        interlock = getattr(self.robot, "forward_interlock", None)
        refresh = getattr(interlock, "refresh", None)
        if not callable(refresh):
            return dict(result, reason="forward_interlock_refresh_unavailable")
        try:
            lidar = self.world_model.get_lidar_obstacles(
                expected_session=expected_session, now=now,
            )
            safety = evaluate_local_motion_safety(
                lidar,
                expected_session=expected_session,
                linear_x=self.FIND_APPROACH_FORWARD_SPEED,
                duration=self.FIND_APPROACH_FORWARD_SECONDS,
                now=now,
            )
        except Exception as exc:
            return dict(result, reason="marvin_pursuit_pre_dispatch_lidar_read_or_evaluation_failed",
                        error=str(exc), error_type=type(exc).__name__)
        result["safety"] = safety
        if not self._marvin_pursuit_lidar_is_trusted(lidar, safety, expected_session):
            if self._marvin_lidar_reason_is_stale(lidar, safety):
                return dict(result, status="nonphysical_stale_replan",
                            reason="marvin_pursuit_pre_dispatch_lidar_stale")
            return dict(result, reason="marvin_pursuit_pre_dispatch_lidar_not_trusted")
        if safety.get("permitted") is not True:
            return dict(result, reason="marvin_pursuit_pre_dispatch_translation_vetoed")
        try:
            permitted, reason = refresh()
        except Exception as exc:
            return dict(result, reason="forward_interlock_refresh_failed",
                        error=str(exc), error_type=type(exc).__name__)
        result["interlock"] = {"permitted": permitted, "reason": reason,
                               "producer_session": expected_session}
        if permitted is not True:
            if self._marvin_lidar_reason_is_stale(lidar, safety, reason):
                return dict(result, status="nonphysical_stale_replan",
                            reason="marvin_pursuit_pre_dispatch_interlock_stale")
            return dict(result, reason=reason or "forward_interlock_not_permitted")
        return dict(result, status="ready", reason="fresh_pre_dispatch_authorized")

    @staticmethod
    def _marvin_pursuit_lidar_is_trusted(lidar, safety, session):
        """Require the same validated producer-bound state used by safety."""
        return bool(
            isinstance(lidar, dict)
            and lidar.get("producer_session") == session
            and isinstance(safety, dict)
            and isinstance(safety.get("geometry"), dict)
            and safety["geometry"].get("valid") is True
        )

    def execute_local_obstacle_avoidance_step(
        self, *, expected_lidar_session=None, now=None,
        minimum_lidar_acquisition_sequence=None,
    ):
        """Plan and execute at most one existing bounded local primitive.

        This is deliberately not a mission handler.  A caller must obtain a
        fresh World Model snapshot again before asking for another step.
        """
        session = expected_lidar_session or self._current_lidar_session()
        base = {
            "ok": False,
            "planner": None,
            "selected_action": None,
            "executed_primitive": None,
            "motion_executed": False,
            "replan_required": False,
            "execution_result": None,
            "lidar_acquisition_sequence": None,
            "reason": None,
        }
        if session is None:
            return dict(base, reason="lidar_producer_session_unavailable")
        if self.world_model is None:
            return dict(base, reason="world_model_unavailable")
        try:
            state = self.world_model.get_lidar_obstacles(
                expected_session=session,
                now=now,
            )
        except Exception as exc:
            return dict(base, reason="local_avoidance_lidar_read_failed",
                        error=str(exc), error_type=type(exc).__name__)
        sequence = state.get("acquisition_sequence") if isinstance(state, dict) else None
        sequence_valid = (
            isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and sequence >= 0
        )
        minimum_valid = (
            isinstance(minimum_lidar_acquisition_sequence, int)
            and not isinstance(minimum_lidar_acquisition_sequence, bool)
            and minimum_lidar_acquisition_sequence >= 0
        )
        result = dict(
            base,
            lidar_acquisition_sequence=sequence if sequence_valid else None,
        )
        if minimum_lidar_acquisition_sequence is not None and (
            not minimum_valid
            or not sequence_valid
            or sequence <= minimum_lidar_acquisition_sequence
        ):
            return dict(result, reason="fresh_lidar_after_motion_unavailable")
        try:
            planner = plan_local_obstacle_avoidance(
                state, expected_session=session, now=now,
            )
        except Exception as exc:
            return dict(result, reason="local_avoidance_planner_error",
                        error=str(exc), error_type=type(exc).__name__)
        result["planner"] = planner
        if not isinstance(planner, dict):
            return dict(result, reason="invalid_local_avoidance_planner_result")
        selected = planner.get("selected_action")
        result["selected_action"] = selected
        if (
            planner.get("ok") is not True
            or planner.get("producer_session") != session
            or not isinstance(selected, str)
        ):
            return dict(result, reason=planner.get(
                "reason", "local_avoidance_not_permitted",
            ))
        evaluations = planner.get("candidate_evaluations")
        selected_evaluation = (
            evaluations.get(selected) if isinstance(evaluations, dict) else None
        )
        if (
            not isinstance(selected_evaluation, dict)
            or selected_evaluation.get("permitted") is not True
        ):
            return dict(result, reason="selected_candidate_not_permitted")

        mapping = {
            "forward": ("forward", None),
            "left_turn": ("left_turn", "LEFT"),
            "right_turn": ("right_turn", "RIGHT"),
            "forward_left": ("left_turn", "LEFT"),
            "forward_right": ("right_turn", "RIGHT"),
        }
        primitive = mapping.get(selected)
        if primitive is None:
            return dict(result, reason="unsupported_local_avoidance_action")
        executed_primitive, direction = primitive
        if direction is not None:
            turn_evaluation = evaluations.get(executed_primitive)
            if (
                not isinstance(turn_evaluation, dict)
                or turn_evaluation.get("permitted") is not True
            ):
                return dict(result, reason="mapped_turn_candidate_not_permitted")

        # Exactly one dispatch site follows.  There is intentionally no
        # fallback action or follow-up forward command in this invocation.
        try:
            if direction is None:
                execution = self.robot.local_forward()
                execution_ok = bool(
                    isinstance(execution, dict)
                    and execution.get("ok") is True
                    and execution.get("executed") is True
                )
            else:
                execution = self.execute_guarded_turn(
                    direction,
                    angular_speed=0.50,
                    duration=0.40,
                    expected_lidar_session=session,
                    now=now,
                )
                execution_ok = bool(
                    isinstance(execution, dict)
                    and execution.get("ok") is True
                    and execution.get("permitted") is True
                    and execution.get("confirmed_forwarded") is True
                )
        except Exception as exc:
            return dict(
                result,
                executed_primitive=executed_primitive,
                reason="local_avoidance_primitive_exception",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if not execution_ok:
            return dict(
                result,
                executed_primitive=executed_primitive,
                execution_result=execution,
                reason=(
                    execution.get("reason", "local_avoidance_primitive_failed")
                    if isinstance(execution, dict)
                    else "local_avoidance_primitive_failed"
                ),
            )
        reason = (
            "turned_for_forward_left_replan"
            if selected == "forward_left" else
            "turned_for_forward_right_replan"
            if selected == "forward_right" else
            "bounded_local_avoidance_primitive_complete"
        )
        return dict(
            result,
            ok=True,
            executed_primitive=executed_primitive,
            motion_executed=True,
            replan_required=True,
            execution_result=execution,
            reason=reason,
        )

    def execute_local_obstacle_avoidance_loop(
        self, *, expected_lidar_session=None, max_steps=3, now=None,
        lidar_wait_timeout_seconds=LOCAL_AVOIDANCE_LIDAR_WAIT_TIMEOUT_SECONDS,
        lidar_poll_interval_seconds=LOCAL_AVOIDANCE_LIDAR_POLL_INTERVAL_SECONDS,
    ):
        """Run a bounded sequence of coordinator steps with fresh LiDAR.

        This method intentionally contains no primitive or transport call.
        Every motion-capable operation remains inside the one-step coordinator.
        """
        base = {
            "ok": False,
            "completed": False,
            "reason": None,
            "producer_session": expected_lidar_session,
            "max_steps": max_steps,
            "steps_executed": 0,
            "steps": [],
            "freshness_waits": [],
        }
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            return dict(base, reason="invalid_local_avoidance_step_limit")
        session = expected_lidar_session or self._current_lidar_session()
        if session is None:
            return dict(base, reason="lidar_producer_session_unavailable")
        if not self._valid_local_avoidance_wait_bounds(
            lidar_wait_timeout_seconds, lidar_poll_interval_seconds,
        ):
            return dict(base, reason="invalid_lidar_freshness_wait_bounds")
        result = dict(base, producer_session=session)
        previous_sequence = None
        for _step_index in range(max_steps):
            try:
                step = self.execute_local_obstacle_avoidance_step(
                    expected_lidar_session=session,
                    now=now,
                    minimum_lidar_acquisition_sequence=previous_sequence,
                )
            except Exception as exc:
                step = {
                    "ok": False,
                    "motion_executed": False,
                    "reason": "local_avoidance_step_exception",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            result["steps"].append(step)
            if not isinstance(step, dict):
                return dict(result, reason="invalid_local_avoidance_step_result")
            if step.get("ok") is not True:
                return dict(result, reason=step.get(
                    "reason", "local_avoidance_step_failed",
                ))
            if step.get("motion_executed") is not True:
                return dict(result, reason="local_avoidance_motion_not_executed")
            sequence = step.get("lidar_acquisition_sequence")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
                or (previous_sequence is not None and sequence <= previous_sequence)
            ):
                return dict(result, reason="fresh_lidar_after_motion_unavailable")
            planner = step.get("planner")
            if (
                not isinstance(planner, dict)
                or planner.get("producer_session") != session
            ):
                return dict(result, reason="producer_session_mismatch")
            result["steps_executed"] += 1
            if step.get("replan_required") is not True:
                return dict(result, ok=True,
                            reason="local_avoidance_replan_not_required")
            previous_sequence = sequence
            if _step_index + 1 < max_steps:
                try:
                    freshness = self._wait_for_newer_lidar_snapshot(
                        expected_lidar_session=session,
                        previous_sequence=previous_sequence,
                        timeout_seconds=lidar_wait_timeout_seconds,
                        poll_interval_seconds=lidar_poll_interval_seconds,
                    )
                except Exception as exc:
                    freshness = {
                        "ok": False,
                        "reason": "local_avoidance_lidar_wait_exception",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "previous_acquisition_sequence": previous_sequence,
                    }
                wait_summary = {
                    key: value for key, value in freshness.items()
                    if key != "snapshot"
                }
                result["freshness_waits"].append(wait_summary)
                if freshness.get("ok") is not True:
                    return dict(result, reason=freshness.get(
                        "reason", "fresh_lidar_after_motion_unavailable",
                    ))
        return dict(
            result,
            ok=True,
            reason="local_avoidance_step_limit_reached",
        )

    @staticmethod
    def _valid_local_avoidance_wait_bounds(timeout_seconds, poll_interval_seconds):
        values = (timeout_seconds, poll_interval_seconds)
        if any(isinstance(value, bool) for value in values):
            return False
        try:
            timeout = float(timeout_seconds)
            interval = float(poll_interval_seconds)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(timeout)
            and math.isfinite(interval)
            and timeout > 0.0
            and interval > 0.0
            and interval <= timeout
        )

    def _wait_for_newer_lidar_snapshot(
        self, *, expected_lidar_session, previous_sequence,
        timeout_seconds=LOCAL_AVOIDANCE_LIDAR_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds=LOCAL_AVOIDANCE_LIDAR_POLL_INTERVAL_SECONDS,
    ):
        """Wait a bounded time for a newer World Model LiDAR acquisition."""
        base = {
            "ok": False,
            "snapshot": None,
            "acquisition_sequence": None,
            "previous_acquisition_sequence": previous_sequence,
            "last_acquisition_sequence": None,
            "producer_session": expected_lidar_session,
            "wait_elapsed_seconds": 0.0,
            "poll_count": 0,
            "reason": None,
        }
        if (
            not isinstance(expected_lidar_session, str)
            or not expected_lidar_session
            or not isinstance(previous_sequence, int)
            or isinstance(previous_sequence, bool)
            or previous_sequence < 0
            or not self._valid_local_avoidance_wait_bounds(
                timeout_seconds, poll_interval_seconds,
            )
        ):
            return dict(base, reason="invalid_lidar_freshness_wait_request")
        if self.world_model is None:
            return dict(base, reason="world_model_unavailable")

        timeout = float(timeout_seconds)
        interval = float(poll_interval_seconds)
        started = time.monotonic()
        deadline = started + timeout
        last_sequence = None

        def finish(reason, *, snapshot=None, sequence=None, ok=False):
            elapsed = max(0.0, time.monotonic() - started)
            return dict(
                base,
                ok=ok,
                snapshot=snapshot,
                acquisition_sequence=sequence,
                last_acquisition_sequence=last_sequence,
                wait_elapsed_seconds=elapsed,
                poll_count=poll_count,
                reason=reason,
            )

        poll_count = 0
        while True:
            if time.monotonic() >= deadline:
                return finish("fresh_lidar_after_motion_timeout")
            poll_count += 1
            try:
                snapshot = self.world_model.get_lidar_obstacles(
                    expected_session=expected_lidar_session,
                )
            except Exception as exc:
                result = finish("world_model_read_failed")
                result.update(error=str(exc), error_type=type(exc).__name__)
                return result
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                return finish("fresh_lidar_after_motion_timeout")
            if not isinstance(snapshot, dict):
                return finish("malformed_lidar_snapshot")
            session = snapshot.get("producer_session")
            if not isinstance(session, str) or not session:
                return finish("malformed_lidar_snapshot", snapshot=snapshot)
            if session != expected_lidar_session:
                return finish("lidar_producer_session_changed", snapshot=snapshot)
            sequence = snapshot.get("acquisition_sequence")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
            ):
                return finish("malformed_lidar_acquisition_sequence", snapshot=snapshot)
            last_sequence = sequence
            if sequence > previous_sequence:
                return finish(
                    "newer_lidar_snapshot_available",
                    snapshot=snapshot,
                    sequence=sequence,
                    ok=True,
                )
            if sequence < previous_sequence:
                return finish(
                    "lidar_acquisition_sequence_regressed",
                    snapshot=snapshot,
                    sequence=sequence,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return finish("fresh_lidar_after_motion_timeout")
            time.sleep(min(interval, remaining))

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
            "marvin_tracker": None,
        }
        self._semantic_episode = episode
        try:
            if getattr(mission, "marvin_guarded_approach_test", False) is True:
                outcome = self._execute_marvin_guarded_approach_test(target_name)
            elif getattr(mission, "marvin_centering_test", False) is True:
                outcome = self._execute_marvin_centering_test(target_name)
            elif getattr(mission, "marvin_one_step_test", False) is True:
                outcome = self._execute_marvin_one_step_test(target_name)
            else:
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

    def _execute_marvin_centering_test(self, target_name):
        """Acquire Marvin, issue at most one guarded turn, then reacquire."""
        base = {
            "ok": False, "completed": True, "executed": False,
            "behavior": "FIND_OBJECT", "mode": "marvin_centering_test",
            "target": "marvin", "target_found": False,
            "identity_confirmed": False, "tracking_confirmed": False,
            "alignment": None, "centering_direction": None,
            "pre_turn_yolo_horizontal_error_pixels": None,
            "pre_turn_tracker_horizontal_error_pixels": None,
            "post_turn_yolo_horizontal_error_pixels": None,
            "post_turn_tracker_horizontal_error_pixels": None,
            "pre_turn_absolute_error": None, "post_turn_absolute_error": None,
            "centering_improved": False, "post_turn_alignment": None,
            "turn_chunks_attempted": 0, "turn_chunks_completed": 0,
            "centering_turn_chunks_attempted": 0,
            "centering_turn_chunks_completed": 0,
            "avoidance_attempted": 0, "forward_calls": 0,
            "turn_speed": self.MARVIN_CENTERING_TURN_SPEED,
            "turn_duration": self.MARVIN_CENTERING_TURN_DURATION,
            "post_step_stop_result": None,
        }
        outcome = None
        turn_dispatched = False
        episode = self._semantic_episode

        def finish(**fields):
            result = dict(base)
            result.update(fields)
            return result

        try:
            if target_name != self.MARVIN_SEMANTIC_TARGET or episode is None:
                outcome = finish(state="MARVIN_CENTERING_BLOCKED", reason="marvin_centering_target_invalid")
            else:
                self._semantic_check_current(episode)
                episode["used"] = True
                first_geometry = {}

                def capture_first(bbox, width, _height):
                    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
                    first_geometry.update(
                        yolo_seed_bbox=dict(bbox),
                        yolo_center_x=center_x,
                        yolo_horizontal_error_pixels=center_x - width / 2.0,
                    )

                acquired = self._acquire_marvin_proposal_tracker_observation(
                    execution_guard=lambda: self._semantic_check_current(episode),
                    episode=episode,
                    before_tracker_initialization=capture_first,
                )
                self._semantic_check_current(episode)
                if not isinstance(acquired, dict) or acquired.get("source") != "marvin_local_tracker":
                    outcome = finish(state="MARVIN_CENTERING_BLOCKED", reason="marvin_local_tracker_confirmation_required")
                else:
                    tracker_error = float(acquired["cx"]) - float(acquired["image_width"]) / 2.0
                    yolo_error = float(first_geometry["yolo_horizontal_error_pixels"])
                    common = {
                        "target_found": True,
                        "identity_confirmed": True,
                        "tracking_confirmed": True,
                        "authority_source": "marvin_local_tracker",
                        "semantic_source": acquired.get("identity_source"),
                        "identity_source": acquired.get("identity_source"),
                        "proposal_label": acquired.get("proposal_label"),
                        "proposal_confidence": acquired.get("proposal_confidence"),
                        "proposal_support": acquired.get("proposal_support"),
                        "geometry_source": acquired.get("geometry_source"),
                        "yolo_seed_bbox": acquired.get("yolo_seed_bbox"),
                        "tracker_seed_bbox": acquired.get("tracker_seed_bbox"),
                        "tracker_seed_source": acquired.get("tracker_seed_source"),
                        "confirmation_diagnostics": acquired.get("confirmation_diagnostics"),
                        "bbox": acquired.get("bbox"),
                        "tracker_bbox": acquired.get("bbox"),
                        "tracker_cx": acquired.get("cx"), "tracker_cy": acquired.get("cy"),
                        "tracker_area": acquired.get("area"),
                        "image_width": acquired.get("image_width"),
                        "image_height": acquired.get("image_height"),
                        "yolo_center_x": first_geometry["yolo_center_x"],
                        "yolo_horizontal_error_pixels": yolo_error,
                        "pre_turn_yolo_horizontal_error_pixels": yolo_error,
                        "pre_turn_tracker_horizontal_error_pixels": tracker_error,
                        "pre_turn_absolute_error": abs(yolo_error),
                        "horizontal_error_pixels": tracker_error,
                        "steering_direction": "CENTER" if abs(yolo_error) <= self.FIND_CENTER_TOLERANCE_PIXELS else ("LEFT" if yolo_error < 0 else "RIGHT"),
                        "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
                    }
                    if abs(yolo_error) <= self.FIND_CENTER_TOLERANCE_PIXELS:
                        outcome = finish(
                            state="MARVIN_CENTERING_ALREADY_ALIGNED",
                            reason="Marvin is already within centering tolerance.",
                            alignment="CENTERED", post_turn_alignment="CENTERED",
                            **common,
                        )
                    else:
                        direction = "LEFT" if yolo_error < 0 else "RIGHT"
                        common["alignment"] = "OFF_CENTER"
                        common["centering_direction"] = direction
                        self._semantic_check_current(episode)
                        session = self._current_lidar_session()
                        if session is None:
                            outcome = finish(state="MARVIN_CENTERING_BLOCKED", reason="turn_guard_unavailable", **common)
                        else:
                            turn = self._execute_target_directed_turn(
                                direction, self.MARVIN_CENTERING_TURN_SPEED,
                                self.MARVIN_CENTERING_TURN_DURATION,
                                expected_lidar_session=session,
                                safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                            )
                            common["turn_chunks_attempted"] = 1
                            common["centering_turn_chunks_attempted"] = 1
                            common["turn_chunks_completed"] = int(isinstance(turn, dict) and turn.get("ok") is True)
                            common["centering_turn_chunks_completed"] = common["turn_chunks_completed"]
                            if not isinstance(turn, dict) or turn.get("ok") is not True:
                                outcome = finish(state="MARVIN_CENTERING_BLOCKED", reason="turn_guard_denied", turn_result=turn, **common)
                            else:
                                turn_dispatched = True
                                self.robot.stop()
                                self._semantic_check_current(episode)
                                post_geometry = {}

                                def capture_post(bbox, width, _height):
                                    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
                                    post_geometry["yolo_horizontal_error_pixels"] = center_x - width / 2.0

                                post = self._acquire_marvin_proposal_tracker_observation(
                                    execution_guard=lambda: self._semantic_check_current(episode),
                                    episode=episode,
                                    before_tracker_initialization=capture_post,
                                )
                                post_yolo_error = float(post_geometry["yolo_horizontal_error_pixels"])
                                post_tracker_error = float(post["cx"]) - float(post["image_width"]) / 2.0
                                outcome = finish(
                                    ok=True, executed=True,
                                    state="MARVIN_CENTERING_STEP_COMPLETE",
                                    reason="One bounded Marvin centering turn completed.",
                                    post_turn_yolo_horizontal_error_pixels=post_yolo_error,
                                    post_turn_tracker_horizontal_error_pixels=post_tracker_error,
                                    post_turn_absolute_error=abs(post_yolo_error),
                                    centering_improved=abs(post_yolo_error) < abs(yolo_error),
                                    post_turn_alignment=("CENTERED" if abs(post_yolo_error) <= self.FIND_CENTER_TOLERANCE_PIXELS else "OFF_CENTER"),
                                    turn_result=turn, **common,
                                )
        except _SemanticPreempted:
            outcome = finish(state="PREEMPTED", reason="FIND_OBJECT execution was preempted.")
        except Exception as exc:
            outcome = finish(
                state=("MARVIN_CENTERING_REACQUISITION_FAILED" if turn_dispatched else "MARVIN_CENTERING_BLOCKED"),
                reason=("marvin_centering_reacquisition_failed" if turn_dispatched else "marvin_centering_error"),
                error_type=type(exc).__name__,
            )
        finally:
            try:
                stop_result = self.robot.stop()
            except Exception as exc:
                stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            if outcome is None:
                outcome = finish(state="MARVIN_CENTERING_BLOCKED", reason="marvin_centering_no_result")
            outcome["post_step_stop_result"] = stop_result
            self._publish_tracking_state(outcome)
        return outcome

    def _marvin_one_step_forward_guard(self, episode):
        """Run the exact bounded forward guard used by Marvin motion tests."""
        interlock = getattr(self.robot, "forward_interlock", None)
        session = self._current_lidar_session()
        if interlock is None or session is None or self.world_model is None:
            return {"authorized": False, "reason": "forward_guard_unavailable"}

        permitted = False
        interlock_reason = None
        lidar = None
        guards = {}
        stale_identity = None
        stale_retry_count = 0
        initial_reason = None
        initial_sequence = None
        guard_authorized = False
        while True:
            if stale_retry_count > 0:
                self._semantic_check_current(episode)
                time.sleep(self.MARVIN_ONE_STEP_LIDAR_REFRESH_POLL_SECONDS)
                self._semantic_check_current(episode)
            permitted, interlock_reason = interlock.refresh()
            lidar = self.world_model.get_lidar_obstacles(expected_session=session)
            front = lidar.get("sectors", {}).get("front", {}) if isinstance(lidar, dict) else {}
            lidar_ok = bool(
                isinstance(lidar, dict)
                and lidar.get("producer_session") == session
                and lidar.get("available") is True
                and lidar.get("valid") is True
                and lidar.get("reason") == "fresh"
                and front.get("state") == "CLEAR"
            )
            lidar_reason = lidar.get("reason") if isinstance(lidar, dict) else None
            acquisition_sequence = lidar.get("acquisition_sequence") if isinstance(lidar, dict) else None
            if stale_retry_count == 0:
                initial_reason = lidar_reason or interlock_reason
                initial_sequence = acquisition_sequence
            guards = {
                "lidar_guard": lidar,
                "forward_interlock_result": {
                    "permitted": permitted, "reason": interlock_reason,
                    "producer_session": session,
                },
            }
            stale_reasons = {"stale", "stale_lidar", "not_fresh"}
            lidar_stale = lidar_reason in stale_reasons
            interlock_stale = interlock_reason in stale_reasons
            front_state = front.get("state") if isinstance(front, dict) else None
            stale_denial = (
                (lidar_stale or interlock_stale)
                and not (
                    lidar_reason not in (None, "fresh") and not lidar_stale
                )
                and not (
                    interlock_reason not in (None, "fresh_clear") and not interlock_stale
                )
                and not (front_state is not None and front_state != "CLEAR")
            )
            if permitted is True and interlock_reason == "fresh_clear" and lidar_ok:
                current_identity = (
                    lidar.get("producer_session"), acquisition_sequence
                ) if isinstance(lidar, dict) else None
                if stale_identity is None or current_identity != stale_identity:
                    guard_authorized = True
                    break
                stale_denial = True
            if not stale_denial:
                break
            stale_identity = (
                lidar.get("producer_session"), acquisition_sequence
            ) if isinstance(lidar, dict) else None
            if stale_retry_count + 1 >= self.MARVIN_ONE_STEP_LIDAR_REFRESH_MAX_ATTEMPTS:
                break
            stale_retry_count += 1
        refresh = {
            "lidar_refresh_attempted": stale_retry_count > 0,
            "lidar_refresh_attempt_count": stale_retry_count,
            "lidar_refresh_succeeded": (
                stale_retry_count > 0 and guard_authorized
                and permitted is True and interlock_reason == "fresh_clear"
                and lidar_ok
            ),
            "lidar_refresh_initial_reason": initial_reason,
            "lidar_refresh_final_reason": (
                interlock_reason if interlock_reason != "fresh_clear"
                else lidar.get("reason") if isinstance(lidar, dict) else None
            ),
            "lidar_refresh_initial_acquisition_sequence": initial_sequence,
            "lidar_refresh_final_acquisition_sequence": (
                lidar.get("acquisition_sequence") if isinstance(lidar, dict) else None
            ),
        }
        return {
            "authorized": (
                guard_authorized and permitted is True
                and interlock_reason == "fresh_clear" and lidar_ok
            ),
            "reason": "fresh_clear" if guard_authorized else "forward_guard_denied",
            "guards": guards,
            "refresh": refresh,
        }

    def _marvin_guarded_approach_turn_duration(self, horizontal_error):
        """Select the bounded approach correction duration by pixel error."""
        magnitude = abs(float(horizontal_error))
        if magnitude <= self.FIND_CENTER_TOLERANCE_PIXELS:
            return None
        return 0.50

    def _execute_marvin_guarded_approach_test(self, target_name):
        """Run a bounded acquire/align/approach/reacquire test."""
        base = {
            "ok": False, "completed": True, "executed": False,
            "behavior": "FIND_OBJECT", "mode": "marvin_guarded_approach_test",
            "target": "marvin", "target_found": False,
            "identity_confirmed": False, "tracking_confirmed": False,
            "max_turns": self.MARVIN_GUARDED_APPROACH_MAX_TURNS,
            "max_consecutive_alignment_turns": self.MARVIN_GUARDED_APPROACH_MAX_TURNS,
            "max_forward_steps": self.MARVIN_GUARDED_APPROACH_MAX_FORWARD_STEPS,
            "max_motion_actions": self.MARVIN_GUARDED_APPROACH_MAX_MOTION_ACTIONS,
            "turn_chunks_attempted": 0, "turn_chunks_completed": 0,
            "centering_turn_chunks_attempted": 0,
            "centering_turn_chunks_completed": 0,
            "approach_chunks_attempted": 0, "approach_chunks_completed": 0,
            "motion_actions_attempted": 0, "motion_actions_completed": 0,
            "forward_speed": self.FIND_APPROACH_FORWARD_SPEED,
            "forward_duration": self.FIND_APPROACH_FORWARD_SECONDS,
            "turn_speed": self.MARVIN_CENTERING_TURN_SPEED,
            "turn_duration": self.MARVIN_CENTERING_TURN_DURATION,
            "avoidance_attempted": 0, "current_alignment": None,
            "current_horizontal_error_pixels": None,
            "current_yolo_horizontal_error_pixels": None,
            "approach_cycle_results": [], "post_step_stop_result": None,
        }
        episode = self._semantic_episode
        outcome = None
        turns = 0
        forwards = 0
        actions = 0
        consecutive_alignment_turns = 0
        recorded_cycles = set()

        def record_cycle(cycle):
            """Retain the cycle even when its terminal action fails."""
            marker = id(cycle)
            if marker not in recorded_cycles:
                recorded_cycles.add(marker)
                base["approach_cycle_results"].append(cycle)

        def finish(**fields):
            result = dict(base)
            result.update(
                turn_chunks_attempted=turns,
                turn_chunks_completed=sum(bool(c.get("turn_completed")) for c in result["approach_cycle_results"]),
                approach_chunks_attempted=forwards,
                approach_chunks_completed=sum(bool(c.get("forward_completed")) for c in result["approach_cycle_results"]),
                motion_actions_attempted=actions,
                motion_actions_completed=sum(
                    bool(c.get("turn_completed")) or bool(c.get("forward_completed"))
                    for c in result["approach_cycle_results"]
                ),
            )
            result["consecutive_alignment_turns"] = consecutive_alignment_turns
            result["executed"] = bool(
                result["executed"]
                or result["motion_actions_completed"] > 0
            )
            result.update(fields)
            return result

        try:
            if target_name != self.MARVIN_SEMANTIC_TARGET or episode is None:
                outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="marvin_guarded_approach_target_invalid")
            else:
                episode["used"] = True
                while True:
                    self._semantic_check_current(episode)
                    geometry = {}

                    def capture(bbox, width, _height):
                        center_x = (bbox["x1"] + bbox["x2"]) / 2.0
                        geometry.update(
                            yolo_seed_bbox=dict(bbox),
                            yolo_horizontal_error_pixels=center_x - width / 2.0,
                        )

                    acquired = self._acquire_marvin_proposal_tracker_observation(
                        execution_guard=lambda: self._semantic_check_current(episode),
                        episode=episode,
                        before_tracker_initialization=capture,
                    )
                    self._semantic_check_current(episode)
                    if not isinstance(acquired, dict) or acquired.get("source") != "marvin_local_tracker":
                        raise ValueError("marvin_local_tracker_confirmation_required")
                    yolo_error = float(geometry["yolo_horizontal_error_pixels"])
                    tracker_error = float(acquired["cx"]) - float(acquired["image_width"]) / 2.0
                    alignment = "CENTERED" if abs(yolo_error) <= self.FIND_CENTER_TOLERANCE_PIXELS else "OFF_CENTER"
                    cycle = {
                        "cycle_index": actions + 1,
                        "acquisition_confirmed": True,
                        "proposal_label": acquired.get("proposal_label"),
                        "proposal_support": acquired.get("proposal_support"),
                        "authority_source": acquired.get("source"),
                        "yolo_seed_bbox": acquired.get("yolo_seed_bbox"),
                        "tracker_bbox": acquired.get("bbox"),
                        "yolo_horizontal_error_pixels": yolo_error,
                        "tracker_horizontal_error_pixels": tracker_error,
                        "alignment": alignment,
                        "selected_action": None, "selected_direction": None,
                        "turn_attempted": False, "turn_completed": False,
                        "forward_attempted": False, "forward_completed": False,
                        "stop_ok": None,
                    }
                    base_update = {
                        "target_found": True, "identity_confirmed": True,
                        "tracking_confirmed": True,
                        "authority_source": acquired.get("source"),
                        "identity_source": acquired.get("identity_source"),
                        "proposal_label": acquired.get("proposal_label"),
                        "proposal_support": acquired.get("proposal_support"),
                        "yolo_seed_bbox": acquired.get("yolo_seed_bbox"),
                        "tracker_bbox": acquired.get("bbox"),
                        "current_alignment": alignment,
                        "current_horizontal_error_pixels": tracker_error,
                        "current_yolo_horizontal_error_pixels": yolo_error,
                        "consecutive_alignment_turns": consecutive_alignment_turns,
                    }
                    if alignment == "OFF_CENTER":
                        if actions >= self.MARVIN_GUARDED_APPROACH_MAX_MOTION_ACTIONS:
                            record_cycle(cycle)
                            outcome = finish(
                                state="MARVIN_GUARDED_APPROACH_MOTION_LIMIT",
                                reason="Maximum guarded Marvin motion actions completed.",
                                **base_update,
                            )
                            break
                        if consecutive_alignment_turns >= self.MARVIN_GUARDED_APPROACH_MAX_TURNS:
                            record_cycle(cycle)
                            outcome = finish(
                                state="MARVIN_GUARDED_APPROACH_ALIGNMENT_LIMIT",
                                reason="Marvin remains off-center after the bounded turn limit.",
                                **base_update,
                            )
                            break
                        direction = "LEFT" if yolo_error < 0 else "RIGHT"
                        turn_duration = self._marvin_guarded_approach_turn_duration(yolo_error)
                        cycle["selected_action"] = "turn"
                        cycle["selected_direction"] = direction
                        cycle["selected_turn_speed"] = self.MARVIN_CENTERING_TURN_SPEED
                        cycle["selected_turn_duration"] = turn_duration
                        self._semantic_check_current(episode)
                        session = self._current_lidar_session()
                        if session is None:
                            record_cycle(cycle)
                            outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="turn_guard_unavailable", **base_update)
                            break
                        cycle["turn_attempted"] = True
                        self._semantic_check_current(episode)
                        try:
                            turn = self._execute_target_directed_turn(
                                direction, self.MARVIN_CENTERING_TURN_SPEED,
                                turn_duration,
                                expected_lidar_session=session,
                                safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                            )
                        except Exception:
                            turn = {"ok": False}
                        turns += 1
                        actions += 1
                        cycle["turn_completed"] = isinstance(turn, dict) and turn.get("ok") is True
                        if not cycle["turn_completed"]:
                            record_cycle(cycle)
                            outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="turn_guard_denied", **base_update)
                            break
                        consecutive_alignment_turns += 1
                        try:
                            turn_stop = self.robot.stop()
                        except Exception:
                            turn_stop = {"ok": False}
                        cycle["stop_ok"] = bool(
                            isinstance(turn_stop, dict) and turn_stop.get("ok") is True
                        )
                        record_cycle(cycle)
                        if not cycle["stop_ok"]:
                            outcome = finish(
                                state="MARVIN_GUARDED_APPROACH_BLOCKED",
                                reason="turn_stop_failed",
                                **base_update,
                            )
                            break
                        self._semantic_check_current(episode)
                        continue

                    if forwards >= self.MARVIN_GUARDED_APPROACH_MAX_FORWARD_STEPS:
                        outcome = finish(
                            ok=True, executed=True,
                            state="MARVIN_GUARDED_APPROACH_COMPLETE",
                            reason="Maximum guarded Marvin approach steps completed.",
                            **base_update,
                        )
                        break
                    if actions >= self.MARVIN_GUARDED_APPROACH_MAX_MOTION_ACTIONS:
                        record_cycle(cycle)
                        outcome = finish(
                            state="MARVIN_GUARDED_APPROACH_MOTION_LIMIT",
                            reason="Maximum guarded Marvin motion actions completed.",
                            **base_update,
                        )
                        break
                    guard = self._marvin_one_step_forward_guard(episode)
                    cycle["selected_action"] = "forward"
                    cycle.update({
                        key: guard.get("refresh", {}).get(key)
                        for key in (
                            "lidar_refresh_attempted", "lidar_refresh_attempt_count",
                            "lidar_refresh_succeeded", "lidar_refresh_initial_reason",
                            "lidar_refresh_final_reason",
                            "lidar_refresh_initial_acquisition_sequence",
                            "lidar_refresh_final_acquisition_sequence",
                        )
                    })
                    cycle["forward_interlock_reason"] = guard.get("guards", {}).get("forward_interlock_result", {}).get("reason")
                    if not guard.get("authorized"):
                        record_cycle(cycle)
                        outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="forward_guard_denied", **base_update)
                        break
                    self._semantic_check_current(episode)
                    cycle["forward_attempted"] = True
                    try:
                        forward = self.robot.move_forward(
                            speed=self.FIND_APPROACH_FORWARD_SPEED,
                            seconds=self.FIND_APPROACH_FORWARD_SECONDS,
                        )
                    except Exception:
                        forward = {"ok": False}
                    forwards += 1
                    actions += 1
                    cycle["forward_completed"] = isinstance(forward, dict) and forward.get("ok") is True
                    try:
                        forward_stop = self.robot.stop()
                    except Exception:
                        forward_stop = {"ok": False}
                    cycle["stop_ok"] = bool(
                        isinstance(forward_stop, dict) and forward_stop.get("ok") is True
                    )
                    record_cycle(cycle)
                    if not cycle["stop_ok"]:
                        outcome = finish(
                            state="MARVIN_GUARDED_APPROACH_BLOCKED",
                            reason="forward_stop_failed",
                            **base_update,
                        )
                        break
                    if not cycle["forward_completed"]:
                        outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="marvin_forward_failed", **base_update)
                        break
                    if forwards >= self.MARVIN_GUARDED_APPROACH_MAX_FORWARD_STEPS:
                        outcome = finish(
                            ok=True, executed=True,
                            state="MARVIN_GUARDED_APPROACH_COMPLETE",
                            reason="Maximum guarded Marvin approach steps completed.",
                            **base_update,
                        )
                        break
                    consecutive_alignment_turns = 0
                    self._semantic_check_current(episode)
        except _SemanticPreempted:
            outcome = finish(state="PREEMPTED", reason="FIND_OBJECT execution was preempted.")
        except Exception as exc:
            outcome = finish(state="MARVIN_GUARDED_APPROACH_REACQUISITION_FAILED", reason="marvin_guarded_approach_reacquisition_failed", error_type=type(exc).__name__)
        finally:
            try:
                stop_result = self.robot.stop()
            except Exception as exc:
                stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            if outcome is None:
                outcome = finish(state="MARVIN_GUARDED_APPROACH_BLOCKED", reason="marvin_guarded_approach_no_result")
            outcome["post_step_stop_result"] = stop_result
            outcome["final_stop_result"] = stop_result
            self._publish_tracking_state(outcome)
        return outcome

    def _execute_marvin_one_step_test(self, target_name):
        """Run the isolated, one-forward Marvin tracker validation path."""
        base = {
            "ok": False, "completed": True, "executed": False,
            "behavior": "FIND_OBJECT", "mode": "marvin_one_step_test",
            "target": "marvin", "target_found": False,
            "authority_source": None, "semantic_source": None,
            "marvin_local_tracker_confirmed": False,
            "steering_direction": None, "horizontal_error_pixels": None,
            "center_tolerance_pixels": self.FIND_CENTER_TOLERANCE_PIXELS,
            "lidar_guard": None, "forward_interlock_result": None,
            "approach_forward_speed": self.FIND_APPROACH_FORWARD_SPEED,
            "approach_forward_duration": self.FIND_APPROACH_FORWARD_SECONDS,
            "approach_chunks_attempted": 0, "approach_chunks_completed": 0,
            "forward_result": None, "post_step_stop_result": None,
            "lidar_refresh_attempted": False,
            "lidar_refresh_attempt_count": 0,
            "lidar_refresh_succeeded": False,
            "lidar_refresh_initial_reason": None,
            "lidar_refresh_final_reason": None,
            "lidar_refresh_initial_acquisition_sequence": None,
            "lidar_refresh_final_acquisition_sequence": None,
            "turn_chunks_attempted": 0, "turn_chunks_completed": 0,
            "centering_turn_chunks_attempted": 0,
            "centering_turn_chunks_completed": 0,
            "avoidance_attempted": 0,
        }
        outcome = None
        episode = self._semantic_episode

        def finish(**fields):
            value = dict(base)
            value.update(fields)
            return value

        try:
            if target_name != self.MARVIN_SEMANTIC_TARGET or episode is None:
                outcome = finish(state="MARVIN_ONE_STEP_REJECTED", reason="marvin_one_step_target_invalid")
            elif self.semantic_vision is None:
                outcome = finish(state="MARVIN_ONE_STEP_BLOCKED", reason="semantic_vision_unavailable")
            else:
                self._semantic_check_current(episode)
                episode["used"] = True
                telemetry = episode["telemetry"]
                telemetry["semantic_reacquisition_attempted"] = True
                proposal_geometry = {}

                def require_centered_yolo_bbox(bbox, width, _height):
                    center_x = (bbox["x1"] + bbox["x2"]) / 2.0
                    horizontal_error = center_x - width / 2.0
                    proposal_geometry.update(
                        yolo_seed_bbox=dict(bbox),
                        yolo_center_x=center_x,
                        yolo_horizontal_error_pixels=horizontal_error,
                    )
                    if abs(horizontal_error) > self.FIND_CENTER_TOLERANCE_PIXELS:
                        raise ValueError("marvin_yolo_proposal_not_centered")

                try:
                    acquired = self._acquire_marvin_proposal_tracker_observation(
                        execution_guard=lambda: self._semantic_check_current(episode),
                        episode=episode,
                        before_tracker_initialization=require_centered_yolo_bbox,
                    )
                except ValueError as exc:
                    if str(exc) != "marvin_yolo_proposal_not_centered":
                        raise
                    raise _MarvinProposalNotCentered(proposal_geometry) from exc
                telemetry.update(
                    semantic_reacquisition_completed=True,
                    semantic_reacquisition_found=True,
                    semantic_reacquisition_direction=None,
                    semantic_reacquisition_result=None,
                    marvin_proposal_acquisition=True,
                )
                confirmed = acquired
                if not (
                    isinstance(confirmed, dict)
                    and confirmed.get("source") == "marvin_local_tracker"
                    and self._vision_timestamp_is_iso(confirmed.get("source_timestamp"))
                    and self._target_is_fresh_and_acquired(confirmed)
                ):
                    outcome = finish(
                        state="MARVIN_ONE_STEP_TRACKER_UNCONFIRMED",
                        reason="marvin_local_tracker_confirmation_required",
                    )
                else:
                    horizontal_error = (
                        float(confirmed["cx"])
                        - float(confirmed["image_width"]) / 2.0
                    )
                    steering = (
                        "CENTERED"
                        if abs(horizontal_error) <= self.FIND_CENTER_TOLERANCE_PIXELS
                        else ("LEFT" if horizontal_error < 0 else "RIGHT")
                    )
                    common = {
                        "target_found": True,
                        "authority_source": "marvin_local_tracker",
                        "semantic_source": confirmed.get("identity_source"),
                        "identity_source": confirmed.get("identity_source"),
                        "identity_confirmed": confirmed.get("identity_confirmed"),
                        "proposal_label": confirmed.get("proposal_label"),
                        "proposal_confidence": confirmed.get("proposal_confidence"),
                        "proposal_support": confirmed.get("proposal_support"),
                        "geometry_source": confirmed.get("geometry_source"),
                        "yolo_seed_bbox": confirmed.get("yolo_seed_bbox"),
                        "tracker_seed_bbox": confirmed.get("tracker_seed_bbox"),
                        "tracker_seed_source": confirmed.get("tracker_seed_source"),
                        "confirmation_diagnostics": confirmed.get("confirmation_diagnostics"),
                        "marvin_local_tracker_confirmed": True,
                        "tracker_source_timestamp": confirmed["source_timestamp"],
                        "tracker_bbox": confirmed["bbox"],
                        "bbox": confirmed["bbox"],
                        "tracker_cx": confirmed["cx"], "tracker_cy": confirmed["cy"],
                        "tracker_area": confirmed["area"],
                        "target_label": "marvin",
                        "target_center_x": confirmed["cx"],
                        "target_center_y": confirmed["cy"],
                        "target_area": confirmed["area"],
                        "image_width": confirmed["image_width"],
                        "image_height": confirmed["image_height"],
                        "steering_direction": steering,
                        "horizontal_error_pixels": horizontal_error,
                        **proposal_geometry,
                    }
                    if steering != "CENTERED":
                        outcome = finish(
                            state="MARVIN_ONE_STEP_NOT_CENTERED",
                            reason="marvin_local_tracker_not_centered",
                            **common,
                        )
                    else:
                        interlock = getattr(self.robot, "forward_interlock", None)
                        session = self._current_lidar_session()
                        if interlock is None or session is None or self.world_model is None:
                            outcome = finish(
                                state="MARVIN_ONE_STEP_BLOCKED",
                                reason="forward_guard_unavailable", **common,
                            )
                        else:
                            guard = self._marvin_one_step_forward_guard(episode)
                            guards = guard.get("guards", {})
                            outcome_lidar_refresh = guard.get("refresh", {})
                            if not guard.get("authorized"):
                                outcome = finish(
                                    state="MARVIN_ONE_STEP_BLOCKED",
                                    reason="forward_guard_denied", **common, **guards,
                                    **outcome_lidar_refresh,
                                )
                            elif not self._execution_is_current():
                                outcome = finish(
                                    state="PREEMPTED",
                                    reason="FIND_OBJECT execution was preempted.",
                                    **common, **guards, **outcome_lidar_refresh,
                                )
                            else:
                                self._semantic_check_current(episode)
                                forward_result = None
                                try:
                                    forward_result = self.robot.move_forward(
                                        speed=self.FIND_APPROACH_FORWARD_SPEED,
                                        seconds=self.FIND_APPROACH_FORWARD_SECONDS,
                                    )
                                except Exception as exc:
                                    forward_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
                                attempted = 1
                                completed = int(isinstance(forward_result, dict) and forward_result.get("ok") is True)
                                success = completed == 1 and self._execution_is_current()
                                outcome = finish(
                                    ok=success,
                                    executed=True,
                                    state=("MARVIN_ONE_STEP_COMPLETE" if success else "MARVIN_ONE_STEP_FAILED"),
                                    reason=("One bounded Marvin forward step completed." if success else "marvin_one_step_forward_failed_or_preempted"),
                                    approach_chunks_attempted=attempted,
                                    approach_chunks_completed=completed,
                                    forward_result=forward_result,
                                    **common, **guards, **outcome_lidar_refresh,
                                )
        except _MarvinProposalNotCentered as exc:
            geometry = exc.geometry
            error = geometry["yolo_horizontal_error_pixels"]
            outcome = finish(
                state="MARVIN_ONE_STEP_NOT_CENTERED",
                reason="marvin_yolo_proposal_not_centered",
                yolo_seed_bbox=geometry["yolo_seed_bbox"],
                yolo_center_x=geometry["yolo_center_x"],
                yolo_horizontal_error_pixels=error,
                horizontal_error_pixels=error,
                steering_direction=("LEFT" if error < 0 else "RIGHT"),
            )
        except _SemanticPreempted:
            outcome = finish(state="PREEMPTED", reason="FIND_OBJECT execution was preempted.")
        except Exception as exc:
            outcome = finish(state="MARVIN_ONE_STEP_BLOCKED", reason="marvin_one_step_error", error_type=type(exc).__name__)
        finally:
            try:
                stop_result = self.robot.stop()
            except Exception as exc:
                stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            if outcome is None:
                outcome = finish(state="MARVIN_ONE_STEP_BLOCKED", reason="marvin_one_step_no_result")
            outcome["post_step_stop_result"] = stop_result
            self._publish_tracking_state(outcome)
        return outcome

    def preview_find_object(
        self, target_name, *, minimum_source_frame_stamp_ns=None,
        require_fresh_gemini=False,
    ):
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
        if normalized_target == self.MARVIN_SEMANTIC_TARGET:
            base["source_frame_stamp_ns"] = None
            base["opencv_tracker"] = self._empty_opencv_tracker_diagnostic()
        if not isinstance(require_fresh_gemini, bool):
            return dict(base, reason="find_marvin_v2_identity_policy_invalid")
        if not normalized_target:
            return dict(
                base,
                reason="FIND_OBJECT preview requires a target.",
            )

        # Marvin preview uses YOLO only for geometry, Gemini only for identity,
        # and the confirmed local tracker for the published preview geometry.
        if normalized_target == self.MARVIN_SEMANTIC_TARGET:
            try:
                observation_options = {
                    "minimum_source_frame_stamp_ns": minimum_source_frame_stamp_ns,
                }
                if require_fresh_gemini:
                    observation_options["require_fresh_gemini"] = True
                observation = self._preview_marvin_yolo_identity_observation(
                    **observation_options,
                )
            except Exception as exc:
                negative_preview = dict(
                    base,
                    reason=(
                        "Marvin tracker preview unavailable: "
                        + (
                            "ValueError"
                            if isinstance(exc, ValueError)
                            else type(exc).__name__
                        )
                        + (": " + str(exc) if str(exc) else "")
                    ),
                )
                if require_fresh_gemini and isinstance(exc, _MarvinLocalTrackerConfirmationRequired):
                    negative_preview["reason"] = "find_marvin_post_semantic_tracker_refresh_failed"
                tracker_diagnostics = getattr(exc, "opencv_tracker", None)
                if isinstance(tracker_diagnostics, dict):
                    negative_preview["opencv_tracker"] = tracker_diagnostics
                else:
                    negative_preview["opencv_tracker"] = (
                        self._empty_opencv_tracker_diagnostic(
                            reason=self._opencv_tracker_failure_reason(exc)
                        )
                    )
                if isinstance(exc, _MarvinProposalGeometryInvalid):
                    negative_preview["source_frame_stamp_ns"] = (
                        _valid_source_frame_stamp(exc.source_frame_stamp_ns)
                    )
                    negative_preview["motion_authorized_marvin_candidate"] = False
                return negative_preview
            if observation is None:
                return dict(base, reason="Marvin was not found in the current camera frame.")
            if isinstance(observation, dict) and observation.get("found") is False:
                negative_preview = dict(
                    base,
                    source_frame_stamp_ns=observation.get("source_frame_stamp_ns"),
                    identity_source_frame_stamp_ns=observation.get("identity_source_frame_stamp_ns"),
                    motion_authorized_marvin_candidate=False,
                    reason=observation.get("reason", "Marvin was not found in the current camera frame."),
                )
                if isinstance(observation.get("opencv_tracker"), dict):
                    negative_preview["opencv_tracker"] = observation["opencv_tracker"]
                if isinstance(observation.get("strict_tracker_episode"), dict):
                    negative_preview["strict_tracker_episode"] = observation[
                        "strict_tracker_episode"
                    ]
                return negative_preview
            result = self._build_find_object_preview(
                normalized_target,
                observation,
                source="marvin_local_tracker",
                authoritative=False,
            )
            for key in (
                "detector_target", "geometry_source", "identity_source",
                "identity_confirmed", "yolo_seed_bbox", "detector_confidence",
                "proposal_label", "proposal_confidence", "proposal_support",
                "motion_authorized_marvin_candidate",
                "tracker_seed_bbox", "tracker_seed_source",
                "tracker_horizontal_padding_fraction",
                "tracker_vertical_padding_fraction", "confirmation_diagnostics",
                "track_id", "tracker_source", "marvin_continuity",
                "opencv_tracker", "identity_source_frame_stamp_ns",
                "marvin_tracking_episode", "strict_tracker_episode",
            ):
                if key in observation:
                    result[key] = observation[key]
            return result

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

    def confirm_marvin_identity_from_preview(
        self,
        preview_result=None,
        *,
        now=None,
    ):
        """Serialize explicit confirmation requests against duplicate writes."""
        with self._marvin_identity_confirmation_lock:
            return self._confirm_marvin_identity_from_preview(
                preview_result,
                now=now,
            )

    def _confirm_marvin_identity_from_preview(
        self,
        preview_result=None,
        *,
        now=None,
    ):
        """Persist an explicitly operator-confirmed, fresh Marvin preview.

        Preview itself remains non-authoritative. This method is the narrow
        operator-confirmation boundary: it validates the current local-tracker
        observation, writes a canonical World Model person entity carrying a
        PersonIdentityManager ID, then asks the ordinary TargetLock resolver
        to acquire that entity. It never dispatches motion.
        """
        if self.world_model is None or self.target_lock is None:
            return self._marvin_identity_confirmation_failure(
                "world_model_or_target_lock_unavailable"
            )
        if preview_result is None:
            try:
                preview_result = self.preview_find_object("marvin")
            except Exception as exc:
                return self._marvin_identity_confirmation_failure(
                    "preview_acquisition_failed",
                    error=str(exc),
                )
        if not isinstance(preview_result, dict):
            return self._marvin_identity_confirmation_failure(
                "preview_result_malformed"
            )

        observation = preview_result.get("target_observation")
        if not isinstance(observation, dict):
            observation = preview_result
        diagnostics = preview_result.get("confirmation_diagnostics")
        if (
            preview_result.get("ok") is not True
            or preview_result.get("target_found") is not True
            or preview_result.get("target") != "marvin"
            or preview_result.get("source") != "marvin_local_tracker"
            or preview_result.get("identity_confirmed") is not True
            or preview_result.get("authoritative") is not False
            or observation.get("found") is not True
            or observation.get("target", "marvin") != "marvin"
            or observation.get("label", "marvin") != "marvin"
            or observation.get("stale") is True
            or observation.get("source", "marvin_local_tracker")
            != "marvin_local_tracker"
            or preview_result.get("identity_ambiguous") is True
            or observation.get("identity_ambiguous") is True
            or not isinstance(diagnostics, dict)
            or diagnostics.get("confirmation_status") != "target_confirmed"
            or diagnostics.get("qualified_support_reached") is not True
            or diagnostics.get("identity_ambiguous") is True
        ):
            return self._marvin_identity_confirmation_failure(
                "preview_not_confirmed_current_marvin"
            )

        timestamp = preview_result.get("source_timestamp") or observation.get(
            "source_timestamp"
        )
        if not timestamp:
            return self._marvin_identity_confirmation_failure(
                "preview_timestamp_missing_or_malformed"
            )
        parsed_timestamp = self._parse_marvin_confirmation_timestamp(
            timestamp
        )
        current_time = self._parse_marvin_confirmation_timestamp(now)
        if parsed_timestamp is None or current_time is None:
            return self._marvin_identity_confirmation_failure(
                "preview_timestamp_missing_or_malformed"
            )
        age_seconds = max(
            0.0,
            (current_time - parsed_timestamp).total_seconds(),
        )
        if age_seconds > self.TARGET_MAX_AGE_SECONDS:
            return self._marvin_identity_confirmation_failure(
                "preview_stale",
                preview_age_seconds=age_seconds,
            )

        bbox = self._validated_marvin_confirmation_geometry(
            preview_result,
            observation,
        )
        if bbox is None:
            return self._marvin_identity_confirmation_failure(
                "preview_geometry_invalid"
            )
        image_width = float(
            preview_result.get("image_width")
            or observation.get("image_width")
        )
        image_height = float(
            preview_result.get("image_height")
            or observation.get("image_height")
        )
        cx = (bbox["x1"] + bbox["x2"]) / 2.0
        cy = (bbox["y1"] + bbox["y2"]) / 2.0
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        try:
            confidence = float(
                preview_result.get("target_confidence")
                or observation.get("confidence")
                or preview_result.get("detector_confidence")
                or 0.0
            )
        except (TypeError, ValueError):
            return self._marvin_identity_confirmation_failure(
                "preview_confidence_malformed"
            )
        if not math.isfinite(confidence) or confidence < 0.0:
            return self._marvin_identity_confirmation_failure(
                "preview_confidence_malformed"
            )

        try:
            self.world_model.reload()
            lock_snapshot = self.target_lock.snapshot()
        except Exception as exc:
            return self._marvin_identity_confirmation_failure(
                "identity_state_unavailable",
                error=str(exc),
            )
        if not isinstance(lock_snapshot, dict):
            return self._marvin_identity_confirmation_failure(
                "target_lock_snapshot_malformed"
            )
        selected_identity_id = str(
            lock_snapshot.get("locked_identity_id") or ""
        ).strip() or None
        if (
            lock_snapshot.get("tracking_mode") == TargetLock.MODE_LOCKED
            and not selected_identity_id
        ):
            return self._marvin_identity_confirmation_failure(
                "conflicting_selected_identity"
            )
        preview_identity = str(
            preview_result.get("identity_id") or observation.get("identity_id") or ""
        ).strip() or None
        if (
            selected_identity_id
            and preview_identity
            and selected_identity_id != preview_identity
        ):
            return self._marvin_identity_confirmation_failure(
                "conflicting_selected_identity"
            )

        existing_entities = [
            entity
            for entity in self.world_model.entities.values()
            if self.world_model._normalize_entity_label(entity.label) == "marvin"
        ]
        distinct_identity_ids = {
            str(entity.attributes.get("identity_id") or "").strip()
            for entity in existing_entities
            if str(entity.attributes.get("identity_id") or "").strip()
        }
        if len(distinct_identity_ids) > 1:
            return self._marvin_identity_confirmation_failure(
                "conflicting_persistent_marvin_identities"
            )
        if selected_identity_id and selected_identity_id not in distinct_identity_ids:
            return self._marvin_identity_confirmation_failure(
                "conflicting_selected_identity"
            )
        if preview_identity and distinct_identity_ids and (
            preview_identity not in distinct_identity_ids
        ):
            return self._marvin_identity_confirmation_failure(
                "conflicting_persistent_marvin_identity"
            )
        if preview_identity and not distinct_identity_ids:
            return self._marvin_identity_confirmation_failure(
                "preview_identity_not_world_model_authoritative"
            )

        existing_entity = None
        if existing_entities:
            existing_entity = max(
                existing_entities,
                key=lambda entity: entity.last_seen,
            )
            entity_age = self.world_model._timestamp_age_seconds(
                existing_entity.last_seen
            )
            latest = self.world_model.find_latest_entity_by_label(
                "marvin", max_age_seconds=self.TARGET_MAX_AGE_SECONDS,
                refresh=False,
            )
            if (
                entity_age is None
                or entity_age > self.TARGET_MAX_AGE_SECONDS
                or not isinstance(latest, dict)
                or latest.get("found") is not True
                or latest.get("identity_ambiguous") is True
                or not self._target_observations_match(
                    {
                        "label": "marvin", "bbox": bbox, "cx": cx, "cy": cy,
                        "area": area, "image_width": image_width,
                        "image_height": image_height,
                    },
                    {
                        "label": "marvin", "bbox": latest.get("bbox"),
                        "cx": latest.get("cx"), "cy": latest.get("cy"),
                        "area": latest.get("area"),
                        "image_width": latest.get("image_width"),
                        "image_height": latest.get("image_height"),
                    },
                )
            ):
                return self._marvin_identity_confirmation_failure(
                    "existing_marvin_identity_not_unambiguously_current"
                )
            if selected_identity_id and latest.get("identity_id") != selected_identity_id:
                return self._marvin_identity_confirmation_failure(
                    "conflicting_selected_identity"
                )

        identity_reused = bool(existing_entity and distinct_identity_ids)
        entity_id = existing_entity.entity_id if existing_entity is not None else None
        if identity_reused:
            identity_id = next(iter(distinct_identity_ids))
            identity_status = "MATCHED"
        else:
            # An isolated canonical manager intentionally prevents matching
            # this operator-confirmed Marvin to unrelated/stale human records.
            identity_assignment = PersonIdentityManager().assign_identity({
                "label": "person",
                "cx": cx,
                "cy": cy,
                "area": area,
                "image_width": image_width,
                "image_height": image_height,
                "bbox": dict(bbox),
                "confidence": confidence,
            })
            identity_id = str(identity_assignment.get("identity_id") or "").strip()
            identity_status = identity_assignment.get("identity_status")
            if (
                not identity_id
                or identity_assignment.get("identity_ambiguous") is True
                or identity_status != "NEW"
            ):
                return self._marvin_identity_confirmation_failure(
                    "persistent_identity_assignment_failed"
                )

        confirmation_time = current_time.isoformat().replace("+00:00", "Z")
        attributes = {
            "identity_id": identity_id,
            "identity_status": identity_status,
            "identity_match_score": 1.0 if identity_reused else 0.0,
            "identity_ambiguous": False,
            "bbox": dict(bbox),
            "area": area,
            "image_width": image_width,
            "image_height": image_height,
            "targetable": True,
            "source_timestamp": timestamp,
            "operator_confirmed": True,
            "identity_confirmation_source": "marvin_local_tracker_preview",
            "identity_confirmation_timestamp": confirmation_time,
            "preview_candidate": {
                "source": "marvin_local_tracker",
                "source_timestamp": timestamp,
                "bbox": dict(bbox),
            },
        }
        try:
            location = {"cx": cx, "cy": cy, "frame": "camera"}
            if entity_id is None:
                entity_id = EntityRegistry(self.world_model).register_observation(
                    label="marvin",
                    entity_type="person",
                    confidence=confidence,
                    source="operator_confirmed_marvin_preview",
                    location=location,
                    attributes=attributes,
                )
            else:
                self.world_model.update_entity(
                    entity_id=entity_id,
                    label="marvin",
                    entity_type="person",
                    confidence=confidence,
                    source="operator_confirmed_marvin_preview",
                    location=location,
                    attributes=attributes,
                )
        except Exception as exc:
            return self._marvin_identity_confirmation_failure(
                "world_model_write_failed",
                error=str(exc),
            )
        try:
            resolved = self.target_lock.resolve(
                mission_id=self.target_lock.mission_id,
                target_label="marvin",
            )
            resolved_snapshot = self.target_lock.snapshot()
        except Exception as exc:
            return self._marvin_identity_confirmation_failure(
                "target_lock_resolution_failed",
                entity_id=entity_id,
                identity_id=identity_id,
                error=str(exc),
            )
        if (
            not isinstance(resolved, dict)
            or not isinstance(resolved_snapshot, dict)
            or resolved_snapshot.get("tracking_mode") != TargetLock.MODE_LOCKED
            or resolved_snapshot.get("locked_identity_id") != identity_id
            or resolved.get("identity_id") != identity_id
            or resolved.get("found") is not True
            or resolved.get("stale") is True
            or resolved.get("identity_ambiguous") is True
        ):
            return self._marvin_identity_confirmation_failure(
                "target_lock_resolution_failed",
                entity_id=entity_id,
                identity_id=identity_id,
                target_lock_mode=(
                    resolved_snapshot.get("tracking_mode")
                    if isinstance(resolved_snapshot, dict)
                    else None
                ),
            )

        return {
            "ok": True,
            "confirmed": True,
            "identity_confirmed": True,
            "reason": "marvin_identity_confirmed",
            "entity_id": entity_id,
            "identity_id": identity_id,
            "identity_created": not identity_reused,
            "identity_reused": identity_reused,
            "source_preview": {
                "source": "marvin_local_tracker",
                "candidate_id": preview_result.get("candidate_id"),
                "identity_source": preview_result.get("identity_source"),
                "proposal_label": preview_result.get("proposal_label"),
                "source_timestamp": timestamp,
                "bbox": dict(bbox),
            },
            "confirmation_timestamp": confirmation_time,
            "target_lock_mode": resolved_snapshot["tracking_mode"],
            "locked_identity_id": resolved_snapshot["locked_identity_id"],
            "motion_executed": False,
        }

    @staticmethod
    def _marvin_identity_confirmation_failure(reason, **details):
        return {
            "ok": False,
            "confirmed": False,
            "identity_confirmed": False,
            "reason": reason,
            "motion_executed": False,
            **details,
        }

    @staticmethod
    def _parse_marvin_confirmation_timestamp(value):
        from datetime import datetime, timezone

        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None:
            from datetime import timezone
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _validated_marvin_confirmation_geometry(preview, observation):
        def finite(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        bbox = (
            preview.get("bbox")
            if "bbox" in preview
            else observation.get("bbox")
        )
        width = (
            preview.get("image_width")
            if "image_width" in preview
            else observation.get("image_width")
        )
        height = (
            preview.get("image_height")
            if "image_height" in preview
            else observation.get("image_height")
        )
        if (
            not isinstance(bbox, dict)
            or not all(finite(bbox.get(key)) for key in ("x1", "y1", "x2", "y2"))
            or not finite(width)
            or not finite(height)
            or width <= 0
            or height <= 0
        ):
            return None
        normalized = {key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")}
        if (
            normalized["x1"] < 0
            or normalized["y1"] < 0
            or normalized["x2"] > float(width)
            or normalized["y2"] > float(height)
            or normalized["x2"] <= normalized["x1"]
            or normalized["y2"] <= normalized["y1"]
        ):
            return None
        return normalized

    def _preview_marvin_yolo_identity_observation(
        self, *, minimum_source_frame_stamp_ns=None,
        require_fresh_gemini=False,
    ):
        """Acquire Marvin perception for read-only Preview."""
        return self._acquire_marvin_proposal_tracker_observation(
            minimum_source_frame_stamp_ns=minimum_source_frame_stamp_ns,
            require_fresh_gemini=require_fresh_gemini,
            execution_guard=(self._strict_v2_current_execution_guard()
                             if require_fresh_gemini else None),
        )

    def _acquire_marvin_proposal_tracker_observation(
        self,
        *,
        execution_guard=None,
        episode=None,
        before_tracker_initialization=None,
        minimum_source_frame_stamp_ns=None,
        require_fresh_gemini=False,
    ):
        """Acquire Marvin using proposals, identity selection, and tracking.

        This core contains perception only.  Callers may provide an execution
        guard for mission-bound preemption checks; Preview intentionally does
        not provide one.
        """
        if execution_guard is not None:
            execution_guard()
        semantic_vision = self.semantic_vision
        if semantic_vision is None or not callable(
            getattr(semantic_vision, "select_marvin_candidate", None)
        ):
            raise ValueError("marvin_candidate_selection_unavailable")
        candidates, status, diagnostics = (
            self._confirm_marvin_proposal_candidates_with_status(
                execution_guard=execution_guard,
                minimum_source_frame_stamp_ns=minimum_source_frame_stamp_ns,
            )
        )
        if not candidates:
            if require_fresh_gemini:
                self._clear_marvin_v2_tracker_episode()
            source_stamp = diagnostics.get("latest_source_frame_stamp_ns")
            if (
                type(source_stamp) is int
                and (minimum_source_frame_stamp_ns is None
                     or source_stamp > minimum_source_frame_stamp_ns)
            ):
                return {
                    "found": False,
                    "source_frame_stamp_ns": source_stamp,
                    "reason": "Marvin was not found in the current camera frame.",
                }
            if status == "person_proposals_rejected":
                raise ValueError("marvin_person_proposal_rejected")
            raise ValueError("marvin_yolo_proposal_" + str(status))
        proposal_candidates = candidates
        candidates = self._filter_marvin_proposal_geometry(
            candidates, diagnostics,
        )
        if not candidates:
            if require_fresh_gemini:
                self._clear_marvin_v2_tracker_episode()
            source_stamps = [
                candidate.get("source_frame_stamp_ns")
                for candidate in proposal_candidates
                if isinstance(candidate, dict)
                and type(candidate.get("source_frame_stamp_ns")) is int
                and candidate.get("source_frame_stamp_ns") >= 0
            ]
            raise _MarvinProposalGeometryInvalid(
                max(source_stamps) if source_stamps else None
            )

        continuity_candidate = (
            None if require_fresh_gemini
            else self._marvin_preview_continuity_candidate(candidates)
        )
        if continuity_candidate is not None:
            try:
                return self._acquire_marvin_tracker_observation_from_candidate(
                    continuity_candidate,
                    diagnostics,
                    identity_source="marvin_session_continuity",
                    execution_guard=execution_guard,
                    episode=episode,
                    before_tracker_initialization=(
                        before_tracker_initialization
                    ),
                )
            except Exception:
                # Fresh tracker confirmation remains mandatory.  A failed
                # continuity attempt cannot retain semantic authority.
                self._clear_marvin_preview_continuity()

        if execution_guard is not None:
            execution_guard()
        frame = (self._fetch_strict_v2_frame_after(
            minimum_source_frame_stamp_ns, execution_guard=execution_guard,
        ) if require_fresh_gemini and minimum_source_frame_stamp_ns is not None
            else semantic_vision.fetch_frame())
        if execution_guard is not None:
            execution_guard()
        if any(
            frame.width != int(candidate["image_width"])
            or frame.height != int(candidate["image_height"])
            for candidate in candidates
        ):
            raise ValueError("marvin_yolo_frame_dimensions_changed")
        candidates = candidates[: self.MARVIN_PREVIEW_MAX_SEMANTIC_CANDIDATES]
        if execution_guard is not None:
            execution_guard()
        if require_fresh_gemini:
            self._emit_marvin_semantic_frame_diagnostic(frame)
        identity = semantic_vision.select_marvin_candidate(frame, candidates)
        if execution_guard is not None:
            execution_guard()
        if not isinstance(identity, dict) or identity.get("confirmed") is not True:
            if require_fresh_gemini:
                self._clear_marvin_v2_tracker_episode()
            # The source stamp identifies the camera observation, not the
            # semantic result. Preserve it on a negative preview without
            # promoting the candidate or granting any motion authority.
            return {
                "found": False,
                "source_frame_stamp_ns": diagnostics.get(
                    "latest_source_frame_stamp_ns"
                ),
                "identity_source_frame_stamp_ns": _valid_source_frame_stamp(
                    getattr(frame, "source_frame_stamp_ns", None)
                ),
                "reason": "marvin_identity_not_confirmed",
            }
        identity_source = identity.get(
            "source", "gemini_marvin_candidate_selection",
        )
        identity_source_stamp = _valid_source_frame_stamp(
            getattr(frame, "source_frame_stamp_ns", None)
        )
        if require_fresh_gemini and (
            identity_source != "gemini_marvin_candidate_selection"
            or identity_source_stamp is None
        ):
            self._clear_marvin_v2_tracker_episode()
            return {
                "found": False,
                "source_frame_stamp_ns": diagnostics.get(
                    "latest_source_frame_stamp_ns"
                ),
                "identity_source": identity_source,
                "identity_source_frame_stamp_ns": identity_source_stamp,
                "reason": "find_marvin_v2_fresh_identity_frame_unavailable",
            }
        selected_index = identity.get("candidate_index")
        if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
            raise ValueError("marvin_candidate_selection_index_invalid")
        yolo_candidate = candidates[selected_index]
        if require_fresh_gemini:
            self._emit_marvin_semantic_frame_diagnostic(frame, yolo_candidate)
            return self._acquire_strict_v2_tracker_observation_from_candidate(
                yolo_candidate, diagnostics,
                identity_source=identity_source,
                identity_source_frame_stamp_ns=identity_source_stamp,
                execution_guard=execution_guard,
                frame=frame,
                action_frame_minimum_stamp_ns=identity_source_stamp,
                action_frame_minimum_received_monotonic_seconds=time.monotonic(),
            )
        result = self._acquire_marvin_tracker_observation_from_candidate(
            yolo_candidate,
            diagnostics,
            identity_source=identity_source,
            identity_source_frame_stamp_ns=identity_source_stamp,
            require_fresh_gemini=require_fresh_gemini,
            execution_guard=execution_guard,
            episode=episode,
            before_tracker_initialization=before_tracker_initialization,
            frame=frame,
        )
        if not require_fresh_gemini:
            self._set_marvin_preview_continuity(result)
        return result

    MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU = 0.70

    def _clear_marvin_v2_tracker_episode(self):
        with self._marvin_v2_tracker_episode_lock:
            self._marvin_v2_tracker_episode = None

    def _strict_v2_tracker_diagnostic(self, *, active, continued, initialized,
                                      accepted, iou=None, reason=None):
        return {
            "active": bool(active),
            "continued_existing_tracker": bool(continued),
            "initialized_this_observation": bool(initialized),
            "semantic_association": "accepted" if accepted else "failed",
            "association_metric": "iou",
            "association_iou": iou,
            "association_threshold": self.MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU,
            "reset_reason": reason,
        }

    def _acquire_strict_v2_tracker_observation_from_candidate(
        self, yolo_candidate, diagnostics, *, identity_source,
        identity_source_frame_stamp_ns, execution_guard, frame,
        action_frame_minimum_stamp_ns=None,
        action_frame_minimum_received_monotonic_seconds=None,
    ):
        """Keep strict V2 geometry bound to one semantically rechecked tracker.

        Fresh Gemini selection remains mandatory on every invocation.  The
        selected proposal may continue an existing episode only when it has
        strict IoU agreement with that episode's last confirmed tracker box.
        A disagreement clears the episode; it never seeds a replacement in
        the same observation.
        """
        if (identity_source != "gemini_marvin_candidate_selection"
                or _valid_source_frame_stamp(identity_source_frame_stamp_ns) is None):
            self._clear_marvin_v2_tracker_episode()
            return {
                "found": False, "identity_source": identity_source,
                "identity_source_frame_stamp_ns": identity_source_frame_stamp_ns,
                "reason": "marvin_v2_fresh_identity_invalid",
                "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                    active=False, continued=False, initialized=False,
                    accepted=False, reason="fresh_identity_invalid",
                ),
            }
        candidate_bbox = self._target_bbox(yolo_candidate)
        if candidate_bbox is None:
            self._clear_marvin_v2_tracker_episode()
            return {
                "found": False, "identity_source": identity_source,
                "identity_source_frame_stamp_ns": identity_source_frame_stamp_ns,
                "reason": "marvin_v2_candidate_bbox_invalid",
                "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                    active=False, continued=False, initialized=False,
                    accepted=False, reason="candidate_bbox_invalid",
                ),
            }
        try:
            candidate_tracker_seed_bbox = self._expand_marvin_tracker_seed_bbox(
                candidate_bbox,
                int(yolo_candidate["image_width"]),
                int(yolo_candidate["image_height"]),
            )
        except (KeyError, TypeError, ValueError):
            self._clear_marvin_v2_tracker_episode()
            return {
                "found": False, "identity_source": identity_source,
                "identity_source_frame_stamp_ns": identity_source_frame_stamp_ns,
                "reason": "marvin_v2_candidate_seed_bbox_invalid",
                "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                    active=False, continued=False, initialized=False,
                    accepted=False, reason="candidate_seed_bbox_invalid",
                ),
            }
        with self._marvin_v2_tracker_episode_lock:
            episode = self._marvin_v2_tracker_episode
            episode = dict(episode) if isinstance(episode, dict) else None
            association_iou = None
            if episode is None:
                episode = {"marvin_tracker": None, "tracker_bbox": None,
                           "last_tracker_source_frame_stamp_ns": None}
                initialized = True
                continued = False
            else:
                initialized = False
                continued = True
                tracker_bbox = episode.get("tracker_bbox")
                association_iou = self._target_bbox_iou(
                    {"bbox": candidate_tracker_seed_bbox}, {"bbox": tracker_bbox},
                )
                if association_iou < self.MARVIN_V2_TRACKER_ASSOCIATION_MIN_IOU:
                    self._marvin_v2_tracker_episode = None
                    return {
                        "found": False, "identity_source": identity_source,
                        "identity_source_frame_stamp_ns": identity_source_frame_stamp_ns,
                        "reason": "marvin_v2_semantic_tracker_association_failed",
                        "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                            active=False, continued=True, initialized=False,
                            accepted=False, iou=association_iou,
                            reason="semantic_tracker_iou_below_threshold",
                        ),
                    }
            try:
                result = self._acquire_marvin_tracker_observation_from_candidate(
                    yolo_candidate, diagnostics, identity_source=identity_source,
                    identity_source_frame_stamp_ns=identity_source_frame_stamp_ns,
                    require_fresh_gemini=True, execution_guard=execution_guard,
                    episode=episode, frame=frame,
                    existing_tracker=(episode.get("marvin_tracker") if continued else None),
                    minimum_source_frame_stamp_ns=(
                        action_frame_minimum_stamp_ns
                        if action_frame_minimum_stamp_ns is not None
                        else identity_source_frame_stamp_ns
                    ),
                    minimum_received_monotonic_seconds=action_frame_minimum_received_monotonic_seconds,
                )
            except Exception:
                self._marvin_v2_tracker_episode = None
                raise
            tracker = result.get("opencv_tracker") if isinstance(result, dict) else None
            stamp = tracker.get("source_frame_stamp_ns") if isinstance(tracker, dict) else None
            bbox = tracker.get("bbox") if isinstance(tracker, dict) else None
            previous_stamp = episode.get("last_tracker_source_frame_stamp_ns")
            if (
                type(stamp) is not int or stamp < 0 or not isinstance(bbox, dict)
                or (continued and (type(previous_stamp) is not int or stamp <= previous_stamp))
                or (action_frame_minimum_stamp_ns is not None
                    and stamp <= action_frame_minimum_stamp_ns)
            ):
                self._marvin_v2_tracker_episode = None
                return {
                    "found": False, "identity_source": identity_source,
                    "identity_source_frame_stamp_ns": identity_source_frame_stamp_ns,
                    "reason": "marvin_v2_tracker_source_stamp_invalid",
                    "strict_tracker_episode": self._strict_v2_tracker_diagnostic(
                        active=False, continued=continued, initialized=initialized,
                        accepted=False, reason="tracker_source_stamp_invalid",
                    ),
                }
            episode.update(
                marvin_tracker=episode.get("marvin_tracker"),
                tracker_bbox=dict(bbox),
                last_tracker_diagnostics=dict(tracker),
                last_tracker_source_frame_stamp_ns=stamp,
                last_tracker_received_at=result.get("source_timestamp"),
                last_tracker_received_monotonic_seconds=result.get("received_monotonic_seconds"),
                identity_source=identity_source,
                identity_source_frame_stamp_ns=identity_source_frame_stamp_ns,
                episode_id=(episode.get("episode_id") or
                            "marvin-v2-" + str(identity_source_frame_stamp_ns)),
            )
            self._marvin_v2_tracker_episode = episode
            result["strict_tracker_episode"] = self._strict_v2_tracker_diagnostic(
                active=True, continued=continued, initialized=initialized,
                accepted=True,
                # This records the exact pre-update comparison that admitted
                # continuation, rather than a post-update tracker box.
                iou=(1.0 if initialized else association_iou),
            )
            return result

    @staticmethod
    def _empty_opencv_tracker_diagnostic(*, reason="tracker_not_initialized"):
        return {
            "active": False,
            "matched": False,
            "quality": None,
            "threshold": MarvinLocalTracker.MIN_MATCH_QUALITY,
            "bbox": None,
            "center_x": None,
            "center_y": None,
            "horizontal_error": None,
            "image_width": None,
            "image_height": None,
            "source_frame_stamp_ns": None,
            "reason": reason,
        }

    @staticmethod
    def _opencv_tracker_failure_reason(exc):
        message = str(exc).casefold()
        if "bbox" in message:
            return "invalid_bbox"
        if "seed_invalid" in message or "dependencies_unavailable" in message:
            return "template_unavailable"
        if "frame" in message or "decode" in message:
            return "frame_unavailable"
        return "tracker_not_initialized"

    @staticmethod
    def _opencv_tracker_diagnostic(tracker, frame, bbox=None):
        diagnostic_method = getattr(tracker, "preview_diagnostics", None)
        if callable(diagnostic_method):
            diagnostic = diagnostic_method()
        else:
            diagnostic = {
                "active": True,
                "matched": bbox is not None,
                "quality": getattr(tracker, "last_quality", None),
                "threshold": getattr(
                    tracker, "MIN_MATCH_QUALITY",
                    MarvinLocalTracker.MIN_MATCH_QUALITY,
                ),
                "bbox": dict(bbox) if isinstance(bbox, dict) else None,
                "center_x": None,
                "center_y": None,
                "horizontal_error": None,
                "image_width": getattr(
                    frame, "width", getattr(tracker, "last_image_width", None)
                ),
                "image_height": getattr(
                    frame, "height", getattr(tracker, "last_image_height", None)
                ),
                "source_frame_stamp_ns": _valid_source_frame_stamp(
                    getattr(
                        frame, "source_frame_stamp_ns",
                        getattr(tracker, "last_source_frame_stamp_ns", None),
                    )
                ),
                "reason": "matched" if bbox is not None else "frame_unavailable",
            }
        diagnostic = dict(diagnostic) if isinstance(diagnostic, dict) else {}
        diagnostic["active"] = True
        diagnostic["matched"] = bbox is not None
        diagnostic["bbox"] = dict(bbox) if isinstance(bbox, dict) else None
        diagnostic["image_width"] = getattr(
            frame, "width", getattr(tracker, "last_image_width", None)
        )
        diagnostic["image_height"] = getattr(
            frame, "height", getattr(tracker, "last_image_height", None)
        )
        diagnostic["source_frame_stamp_ns"] = _valid_source_frame_stamp(
            getattr(
                frame, "source_frame_stamp_ns",
                getattr(tracker, "last_source_frame_stamp_ns", None),
            )
        )
        # Bind receipt to this exact update frame, never to the older seed or
        # to tracker metadata retained from an earlier frame.
        diagnostic["received_monotonic_seconds"] = getattr(frame, "received_monotonic_seconds", None)
        diagnostic["received_monotonic_clock"] = "local_process_relative"
        if bbox is not None:
            center_x = (bbox["x1"] + bbox["x2"]) / 2.0
            center_y = (bbox["y1"] + bbox["y2"]) / 2.0
            diagnostic["center_x"] = center_x
            diagnostic["center_y"] = center_y
            diagnostic["horizontal_error"] = MarvinLocalTracker.horizontal_error(
                center_x, diagnostic["image_width"],
            )
            diagnostic["reason"] = "matched"
        else:
            diagnostic["center_x"] = None
            diagnostic["center_y"] = None
            diagnostic["horizontal_error"] = None
            diagnostic["reason"] = getattr(
                tracker, "last_reason", diagnostic.get("reason", "frame_unavailable")
            )
        return diagnostic

    def _acquire_marvin_tracker_observation_from_candidate(
        self,
        yolo_candidate,
        diagnostics,
        *,
        identity_source,
        execution_guard=None,
        episode=None,
        before_tracker_initialization=None,
        frame=None,
        identity_source_frame_stamp_ns=None,
        require_fresh_gemini=False,
        existing_tracker=None,
        minimum_source_frame_stamp_ns=None,
        minimum_received_monotonic_seconds=None,
    ):
        """Confirm fresh local-tracker geometry for one selected proposal."""
        semantic_vision = self.semantic_vision
        if frame is None:
            if execution_guard is not None:
                execution_guard()
            frame = semantic_vision.fetch_frame()
        if not isinstance(yolo_candidate, dict):
            raise ValueError("marvin_yolo_candidate_invalid")
        if (
            frame.width != int(yolo_candidate["image_width"])
            or frame.height != int(yolo_candidate["image_height"])
        ):
            raise ValueError("marvin_yolo_frame_dimensions_changed")
        bbox = MarvinLocalTracker._validate_bbox(
            yolo_candidate.get("bbox"),
            int(yolo_candidate["image_width"]),
            int(yolo_candidate["image_height"]),
        )
        yolo_bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox))
        tracker_seed_bbox = self._expand_marvin_tracker_seed_bbox(
            yolo_bbox,
            int(yolo_candidate["image_width"]),
            int(yolo_candidate["image_height"]),
        )
        if before_tracker_initialization is not None:
            before_tracker_initialization(
                yolo_bbox,
                int(yolo_candidate["image_width"]),
                int(yolo_candidate["image_height"]),
            )
        if execution_guard is not None:
            execution_guard()
        temporary_episode = episode is None
        if temporary_episode:
            previous_episode = self._semantic_episode
            episode = {
                "used": True,
                "turn_attempts": 0,
                "turn_completions": 0,
                "telemetry": {},
                "marvin_tracker": None,
            }
            self._semantic_episode = episode
        if existing_tracker is None:
            episode["marvin_tracker"] = self.marvin_local_tracker_factory(
                frame, tracker_seed_bbox,
            )
        else:
            episode["marvin_tracker"] = existing_tracker
        current_frame_floor = minimum_source_frame_stamp_ns

        def fetch_tracker_frame():
            nonlocal current_frame_floor
            if require_fresh_gemini and current_frame_floor is not None:
                current = self._fetch_strict_v2_frame_after(
                    current_frame_floor, execution_guard=execution_guard,
                    minimum_received_monotonic_seconds=minimum_received_monotonic_seconds,
                )
                current_frame_floor = current.source_frame_stamp_ns
                return current
            return semantic_vision.fetch_frame()

        try:
            confirmed = self._confirm_marvin_local_tracker_frames(
                episode["marvin_tracker"],
                minimum_timestamp=frame.received_at,
                minimum_source_frame_stamp_ns=minimum_source_frame_stamp_ns,
                fetch_frame=fetch_tracker_frame,
                check_current=execution_guard,
            )
        finally:
            if temporary_episode:
                self._semantic_episode = previous_episode
        if execution_guard is not None:
            execution_guard()
        if confirmed is None:
            raise _MarvinLocalTrackerConfirmationRequired(
                self._opencv_tracker_diagnostic(
                    episode["marvin_tracker"], None,
                )
            )
        # Preserve only provider-supplied tracker diagnostics. They remain
        # non-authoritative metadata and never alter local tracker behavior.
        tracker_metadata = {}
        raw_proposal = yolo_candidate.get("raw_detection")
        if not isinstance(raw_proposal, dict):
            raw_proposal = {}
        for key in ("track_id", "tracker_source"):
            value = yolo_candidate.get(key, raw_proposal.get(key))
            if value is not None:
                tracker_metadata[key] = value
        marvin_continuity = self._marvin_continuity_metadata(
            yolo_candidate, raw_proposal,
        )
        if marvin_continuity is not None:
            tracker_metadata["marvin_continuity"] = marvin_continuity
        result = dict(
            confirmed,
            label="marvin",
            target="marvin",
            source="marvin_local_tracker",
            proposal_label=yolo_candidate.get("proposal_label"),
            proposal_confidence=yolo_candidate.get("confidence"),
            proposal_support=yolo_candidate.get("proposal_support"),
            # Identity came from the semantic seed; motion belongs only to
            # the latest confirmed local tracker frame, never that seed.
            source_frame_stamp_ns=(
                (confirmed.get("opencv_tracker") or {}).get("source_frame_stamp_ns")
                if require_fresh_gemini else yolo_candidate.get("source_frame_stamp_ns")
            ),
            detector_confidence=yolo_candidate.get("confidence"),
            geometry_source="yolo_proposal",
            identity_source=identity_source,
            identity_confirmed=True,
            motion_authorized_marvin_candidate=(
                self._marvin_motion_authorized_candidate(
                    yolo_candidate, confirmed,
                    require_fresh_gemini=require_fresh_gemini,
                    identity_source=identity_source,
                    identity_source_frame_stamp_ns=(
                        identity_source_frame_stamp_ns
                    ),
                )
            ),
            identity_source_frame_stamp_ns=identity_source_frame_stamp_ns,
            yolo_seed_bbox=yolo_bbox,
            tracker_seed_bbox=tracker_seed_bbox,
            tracker_seed_source="bounded_yolo_proposal_expansion",
            tracker_horizontal_padding_fraction=(
                self.MARVIN_TRACKER_HORIZONTAL_PADDING_FRACTION
            ),
            tracker_vertical_padding_fraction=(
                self.MARVIN_TRACKER_VERTICAL_PADDING_FRACTION
            ),
            confirmation_diagnostics=diagnostics,
            opencv_tracker=confirmed.get("opencv_tracker"),
            **tracker_metadata,
        )
        if require_fresh_gemini:
            tracker = result.get("opencv_tracker")
            tracker = tracker if isinstance(tracker, dict) else {}
            result["marvin_tracking_episode"] = {
                "episode_id": (
                    "marvin-v2-" + str(identity_source_frame_stamp_ns)
                ),
                "identity_confirmed_at": getattr(frame, "received_at", None),
                "identity_source": identity_source,
                "identity_source_frame_stamp_ns": (
                    identity_source_frame_stamp_ns
                ),
                "tracker_initialized": True,
                "tracker_source_frame_stamp_ns": tracker.get(
                    "source_frame_stamp_ns"
                ),
                "tracker_quality": tracker.get("quality"),
                "tracker_bbox": tracker.get("bbox"),
                "last_verified_at": result.get("source_timestamp"),
                "last_verified_source_frame_stamp_ns": tracker.get(
                    "source_frame_stamp_ns"
                ),
                "state": "TRACKING",
            }
        return result

    def _marvin_motion_authorized_candidate(
        self, candidate, confirmed, *, require_fresh_gemini=False,
        identity_source=None, identity_source_frame_stamp_ns=None,
    ):
        """Return Marvin visual-session motion compatibility, fail-closed.

        Semantic selection and local tracking remain visible through the
        non-authoritative Preview. Directing Marvin visual-session motion
        additionally requires the documented detector alias.
        """
        if require_fresh_gemini:
            tracker = (
                confirmed.get("opencv_tracker")
                if isinstance(confirmed, dict) else None
            )
            if not isinstance(tracker, dict):
                return False
            quality = tracker.get("quality")
            threshold = tracker.get("threshold")
            return bool(
                identity_source == "gemini_marvin_candidate_selection"
                and _valid_source_frame_stamp(
                    identity_source_frame_stamp_ns
                ) is not None
                and tracker.get("active") is True
                and tracker.get("matched") is True
                and isinstance(quality, (int, float))
                and not isinstance(quality, bool)
                and math.isfinite(quality)
                and isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                and math.isfinite(threshold)
                and quality >= threshold
                and _valid_source_frame_stamp(
                    tracker.get("source_frame_stamp_ns")
                ) is not None
                and tracker.get("source_frame_stamp_ns")
                > identity_source_frame_stamp_ns
                and isinstance(tracker.get("bbox"), dict)
                and self._target_bbox(confirmed) is not None
                and self._target_is_fresh_and_acquired(confirmed)
            )
        alias = str(self.MARVIN_DETECTOR_ALIAS or "").strip().casefold()
        label = str(
            candidate.get("proposal_label", candidate.get("label", ""))
            if isinstance(candidate, dict) else ""
        ).strip().casefold()
        return bool(
            alias == "teddy bear"
            and label == alias
            and isinstance(confirmed, dict)
            and confirmed.get("found") is True
            and confirmed.get("stale") is not True
            and confirmed.get("source") == "marvin_local_tracker"
            and confirmed.get("identity_ambiguous") is not True
            and self._target_bbox(confirmed) is not None
            and self._target_is_fresh_and_acquired(confirmed)
        )

    def _marvin_preview_continuity_candidate(self, candidates):
        """Return one current proposal only when it matches session authority."""
        with self._marvin_preview_continuity_lock:
            authority = self._marvin_preview_continuity
            authority = dict(authority) if isinstance(authority, dict) else None
        if authority is None:
            return None
        matches = []
        for candidate in candidates:
            continuity = self._marvin_continuity_metadata(candidate)
            if (
                continuity == authority
                and candidate.get("identity_ambiguous") is not True
            ):
                matches.append(candidate)
        # More than one current proposal carrying the authority is ambiguous.
        if len(matches) == 1:
            return matches[0]
        # A missing, changed, or ambiguous provider continuity record ends
        # this session.  The caller must re-establish semantic identity.
        self._clear_marvin_preview_continuity()
        return None

    def _set_marvin_preview_continuity(self, observation):
        continuity = self._marvin_continuity_metadata(observation)
        if continuity is None:
            self._clear_marvin_preview_continuity()
            return
        with self._marvin_preview_continuity_lock:
            self._marvin_preview_continuity = continuity

    def _clear_marvin_preview_continuity(self):
        with self._marvin_preview_continuity_lock:
            self._marvin_preview_continuity = None

    @staticmethod
    def _marvin_continuity_metadata(*sources):
        """Return validated, provider-supplied Marvin continuity metadata.

        This intentionally does not derive an identifier from geometry, a
        label, or any identity field.  Invalid or incomplete provider data is
        omitted so consumers fail closed.
        """
        for source in sources:
            if not isinstance(source, dict):
                continue
            value = source.get("marvin_continuity")
            if not isinstance(value, dict):
                continue
            tracker_id = value.get("tracker_id")
            tracker_source = value.get("tracker_source")
            tracker_generation = value.get("tracker_generation")
            if (
                isinstance(tracker_id, int)
                and not isinstance(tracker_id, bool)
                and tracker_id >= 0
                and isinstance(tracker_source, str)
                and tracker_source.strip()
                and isinstance(tracker_generation, str)
                and tracker_generation.strip()
            ):
                return {
                    "tracker_id": tracker_id,
                    "tracker_source": tracker_source.strip(),
                    "tracker_generation": tracker_generation.strip(),
                }
        return None

    @classmethod
    def _filter_marvin_proposal_geometry(cls, candidates, diagnostics=None):
        """Reject implausibly wide Marvin proposals without using labels."""
        before = len(candidates)
        plausible = []
        for candidate in candidates:
            bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
            try:
                width = float(bbox["x2"]) - float(bbox["x1"])
                height = float(bbox["y2"]) - float(bbox["y1"])
                plausible_geometry = (
                    math.isfinite(width)
                    and math.isfinite(height)
                    and width > 0.0
                    and height > 0.0
                    and width / height <= cls.MARVIN_PROPOSAL_MAX_WIDTH_TO_HEIGHT_RATIO
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                plausible_geometry = False
            if plausible_geometry:
                plausible.append(candidate)
        if isinstance(diagnostics, dict):
            diagnostics.update({
                "marvin_geometry_filter_applied": True,
                "marvin_geometry_max_width_height_ratio": (
                    cls.MARVIN_PROPOSAL_MAX_WIDTH_TO_HEIGHT_RATIO
                ),
                "marvin_geometry_candidates_before": before,
                "marvin_geometry_candidates_after": len(plausible),
                "marvin_geometry_candidates_rejected": before - len(plausible),
            })
        return plausible

    @classmethod
    def _expand_marvin_tracker_seed_bbox(
        cls, bbox, image_width, image_height
    ):
        """Expand only the Marvin preview tracker envelope deterministically."""
        if (
            type(image_width) is not int
            or type(image_height) is not int
            or image_width <= 0
            or image_height <= 0
        ):
            raise ValueError("marvin_tracker_seed_dimensions_invalid")
        values = MarvinLocalTracker._validate_bbox(
            bbox, image_width, image_height,
        )
        x1, y1, x2, y2 = values
        width = x2 - x1
        height = y2 - y1
        horizontal_margin = width * cls.MARVIN_TRACKER_HORIZONTAL_PADDING_FRACTION
        vertical_margin = height * cls.MARVIN_TRACKER_VERTICAL_PADDING_FRACTION
        expanded = {
            "x1": max(0, min(image_width, int(round(x1 - horizontal_margin)))),
            "y1": max(0, min(image_height, int(round(y1 - vertical_margin)))),
            "x2": max(0, min(image_width, int(round(x2 + horizontal_margin)))),
            "y2": max(0, min(image_height, int(round(y2 + vertical_margin)))),
        }
        if expanded["x2"] <= expanded["x1"] or expanded["y2"] <= expanded["y1"]:
            raise ValueError("marvin_tracker_seed_geometry_invalid")
        return expanded

    def _confirm_marvin_proposal_candidates_with_status(
        self, *, execution_guard=None, minimum_source_frame_stamp_ns=None
    ):
        """Cluster class-agnostic proposals by fresh geometry only."""
        fetch = getattr(self.vision, "fetch_detection_proposals", None)
        normalize = getattr(self.vision, "normalize_detection", None)
        diagnostics = {
            "confirmation_status": None,
            "elapsed_seconds": 0.0,
            "fetch_attempts": 0,
            "distinct_fresh_timestamps": 0,
            "evidence_frames_evaluated": 0,
            "actionable_frames": 0,
            "maximum_frames": self.TARGET_CONFIRMATION_MAX_FRAMES,
            "minimum_support": self.TARGET_CONFIRMATION_MIN_SUPPORT,
            "confirmation_window_seconds": self.MARVIN_PREVIEW_CONFIRMATION_WINDOW_SECONDS,
            "poll_seconds": self.TARGET_CONFIRMATION_POLL_SECONDS,
            "minimum_timestamp": None,
            "attempts": [],
            "terminal_reason": None,
            "qualified_support_reached": False,
            "detector_target": None,
            "marvin_person_proposals_rejected": 0,
            "latest_source_frame_stamp_ns": None,
        }
        started = time.monotonic()
        seen_timestamps = set()
        person_timestamps = set()
        last_timestamp = None
        clusters = []
        actionable_seen = False
        person_proposal_seen = False

        def finish(candidates, status, reason=None):
            diagnostics["confirmation_status"] = status
            diagnostics["elapsed_seconds"] = round(
                max(0.0, time.monotonic() - started), 6
            )
            diagnostics["distinct_fresh_timestamps"] = len(seen_timestamps)
            diagnostics["terminal_reason"] = reason or (
                "support_reached" if status == "target_confirmed"
                else "insufficient_temporal_or_geometric_support"
            )
            return candidates, status, diagnostics

        if not callable(fetch) or not callable(normalize):
            return finish([], "target_lost", "proposal_source_unavailable")

        while (
            diagnostics["evidence_frames_evaluated"]
            < self.TARGET_CONFIRMATION_MAX_FRAMES
            and time.monotonic() - started
            <= self.MARVIN_PREVIEW_CONFIRMATION_WINDOW_SECONDS
        ):
            if execution_guard is not None:
                execution_guard()
            diagnostics["fetch_attempts"] += 1
            try:
                payload = fetch()
            except Exception as exc:
                return finish(
                    [],
                    "target_reconfirmation_failed" if actionable_seen else "target_lost",
                    type(exc).__name__,
                )
            attempt = {
                "attempt_index": diagnostics["fetch_attempts"],
                "response_timestamp": (
                    payload.get("timestamp")
                    if isinstance(payload, dict) else None
                ),
                "actionable_candidate_count": 0,
                "candidate_labels": [],
                "cluster_support_counts": [],
            }
            diagnostics["attempts"].append(attempt)
            if not isinstance(payload, dict) or payload.get("camera_running") is not True:
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue
            source_frame_stamp_ns = payload.get("source_frame_stamp_ns")
            if type(source_frame_stamp_ns) is not int or source_frame_stamp_ns < 0:
                source_frame_stamp_ns = None
            # Track the latest observation's identity exactly. A malformed or
            # missing stamp must clear an earlier value rather than silently
            # falling back to an older camera frame.
            diagnostics["latest_source_frame_stamp_ns"] = source_frame_stamp_ns
            if (
                minimum_source_frame_stamp_ns is not None
                and (source_frame_stamp_ns is None
                     or source_frame_stamp_ns <= minimum_source_frame_stamp_ns)
            ):
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue
            timestamp = payload.get("timestamp")
            if (
                not isinstance(timestamp, str)
                or not timestamp.strip()
                or timestamp in seen_timestamps
                or not self._vision_timestamp_is_newer(timestamp, last_timestamp)
            ):
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue
            seen_timestamps.add(timestamp)
            last_timestamp = timestamp
            detections = payload.get("detections")
            if not isinstance(detections, list):
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue

            observations = []
            for raw_detection in detections:
                if not isinstance(raw_detection, dict):
                    continue
                try:
                    normalized = normalize(raw_detection)
                except Exception:
                    continue
                if not isinstance(normalized, dict):
                    continue
                # A detector-labeled person cannot provide geometry authority
                # for Marvin. Filter before the class-agnostic temporal
                # clustering below, so person boxes cannot add support to or
                # alter a compatible proposal's representative geometry.
                if str(normalized.get("label") or "").strip().casefold() == "person":
                    person_proposal_seen = True
                    person_timestamps.add(timestamp)
                    diagnostics["marvin_person_proposals_rejected"] += 1
                    continue
                normalized.update({
                    "found": True,
                    "stale": False,
                    "target": self.MARVIN_SEMANTIC_TARGET,
                    "source_timestamp": timestamp,
                    "source_frame_stamp_ns": source_frame_stamp_ns,
                    "proposal_label": normalized.get("label"),
                    "raw_detection": dict(raw_detection),
                })
                if (
                    self._target_is_fresh_and_acquired(normalized)
                    and self._target_bbox(normalized) is not None
                ):
                    observations.append(normalized)
                    actionable_seen = True
            if not observations:
                if len(person_timestamps) >= self.TARGET_CONFIRMATION_MAX_FRAMES:
                    break
                time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)
                continue

            diagnostics["actionable_frames"] += 1
            diagnostics["evidence_frames_evaluated"] += 1
            attempt["actionable_candidate_count"] = len(observations)
            attempt["candidate_labels"] = [
                observation.get("label") for observation in observations
            ]
            for observation in observations:
                matching = []
                for index, cluster in enumerate(clusters):
                    if timestamp in cluster["timestamps"]:
                        continue
                    if any(
                        self._target_observation_match_details(
                            observation, member, ignore_label=True
                        )["matched"]
                        for member in cluster["observations"]
                    ):
                        matching.append(index)
                if matching:
                    cluster = clusters[matching[0]]
                    cluster["observations"].append(observation)
                    cluster["timestamps"].add(timestamp)
                else:
                    clusters.append({
                        "observations": [observation],
                        "timestamps": {timestamp},
                    })
            attempt["cluster_support_counts"] = [
                len(cluster["timestamps"]) for cluster in clusters
            ]
            if any(
                support >= self.TARGET_CONFIRMATION_MIN_SUPPORT
                for support in attempt["cluster_support_counts"]
            ):
                diagnostics["qualified_support_reached"] = True
            time.sleep(self.TARGET_CONFIRMATION_POLL_SECONDS)

        eligible = [
            cluster for cluster in clusters
            if len(cluster["timestamps"]) >= self.TARGET_CONFIRMATION_MIN_SUPPORT
        ]
        if not eligible:
            if not actionable_seen and person_proposal_seen:
                return finish(
                    [],
                    "person_proposals_rejected",
                    "marvin_person_proposal_rejected",
                )
            return finish(
                [],
                "target_reconfirmation_failed" if actionable_seen else "target_lost",
            )

        representatives = []
        for cluster in eligible:
            representative = max(
                cluster["observations"],
                key=lambda item: (
                    float(item.get("area") or 0.0),
                    float(item.get("confidence") or 0.0),
                ),
            )
            representative = dict(representative)
            representative["proposal_support"] = len(cluster["timestamps"])
            representatives.append(representative)
        representatives.sort(
            key=lambda item: (
                -int(item["proposal_support"]),
                -float(item.get("area") or 0.0),
                item.get("label") or "",
                tuple(
                    item["bbox"].get(key)
                    for key in ("x1", "y1", "x2", "y2")
                ),
            )
        )
        return finish(representatives, "target_confirmed")

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
            MarvinLocalTracker.horizontal_error(cx, image_width)
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
        if target_name == self.MARVIN_SEMANTIC_TARGET:
            result["received_monotonic_seconds"] = observation.get("received_monotonic_seconds")
            result["source_frame_stamp_ns"] = observation.get(
                "source_frame_stamp_ns"
            )
        # Keep the timestamp attached to the exact observation selected for
        # this preview. Marvin's temporary tracker has no persistent identity
        # key today, so do not synthesize one from its seed or track state.
        source_timestamp = observation.get("source_timestamp")
        if source_timestamp is not None:
            result["source_timestamp"] = source_timestamp
            result["vision_timestamp"] = source_timestamp
        for key in (
            "entity_id", "identity_id", "identity_status",
            "identity_ambiguous", "identity_match_score",
        ):
            if key in observation:
                result[key] = observation[key]
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
    def _target_observation_match_details(cls, first, second, *, ignore_label=False):
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
        if not ignore_label and (
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
        detector_target=None,
        return_diagnostics=False,
        confirmation_window_seconds=None,
    ):
        """Confirm a target and optionally return bounded diagnostics."""
        semantic_target = str(target_name or "").strip().lower()
        detector_target = (
            self._detector_target_label(semantic_target)
            if detector_target is None
            else str(detector_target).strip().lower()
        )
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
            "detector_target": detector_target,
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
            return max(
                winning["observations"],
                key=lambda item: (
                    float(item.get("confidence") or 0.0),
                    float(item.get("area") or 0.0),
                ),
            )

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
                payload = fetch(detector_target)
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
                if label.casefold() != detector_target.casefold():
                    continue
                try:
                    normalized = normalize(raw_detection)
                except Exception:
                    continue
                if not isinstance(normalized, dict):
                    continue
                normalized["found"] = True
                normalized["stale"] = False
                normalized["target"] = semantic_target
                normalized["detector_target"] = detector_target
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
        if isinstance(target, dict) and target.get("source") == "marvin_local_tracker":
            return target if self._target_is_fresh_and_acquired(target) else None
        processor = getattr(self.vision, "process_detection_frame", None)
        if not callable(processor):
            return None
        raw_detection = target.get("raw_detection")
        if not isinstance(raw_detection, dict):
            return None
        try:
            processor([dict(raw_detection)])
            promoted = self._get_target_observation(
                target.get("detector_target", target.get("target", ""))
            )
        except Exception:
            return None
        return promoted if self._target_is_fresh_and_acquired(promoted) else None

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
        local_progress_calls = 0
        local_progress_physical_actions = 0
        max_local_progress_physical_actions = (
            self.FIND_APPROACH_MAX_CHUNKS
            + self.FIND_AVOIDANCE_MAX_TURN_CHUNKS
        )
        local_progress_history = []
        current = observation

        def target_telemetry(target, previous=None):
            value = dict(previous or {})
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
                "target_observation": target,
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
                "local_progress_calls": local_progress_calls,
                "local_progress_physical_actions": (
                    local_progress_physical_actions
                ),
                "physical_actions": local_progress_physical_actions,
                "maximum_local_progress_physical_actions": (
                    max_local_progress_physical_actions
                ),
                "local_progress_history": list(local_progress_history),
            })
            value.update(fields)
            self._publish_tracking_state(value)
            return value

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

            local_progress_handler = getattr(
                self, "local_progress_with_avoidance_handler", None
            )
            if callable(local_progress_handler):
                if local_progress_physical_actions >= max_local_progress_physical_actions:
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_BLOCKED",
                        reason="local_progress_physical_action_budget_exhausted",
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

                local_progress_calls += 1
                approach_attempted += 1
                try:
                    local_progress_result = local_progress_handler()
                except Exception as exc:
                    local_progress_result = {
                        "ok": False,
                        "terminal_state": "LOCAL_PROGRESS_EXECUTION_FAILED",
                        "reason": "local_progress_handoff_exception",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "physical_actions": 0,
                    }
                if not isinstance(local_progress_result, dict):
                    local_progress_result = {
                        "ok": False,
                        "terminal_state": "LOCAL_PROGRESS_EXECUTION_FAILED",
                        "reason": "local_progress_handoff_result_malformed",
                        "physical_actions": 0,
                    }
                nested_actions = local_progress_result.get("physical_actions")
                if (not isinstance(nested_actions, int)
                        or isinstance(nested_actions, bool)
                        or nested_actions < 0):
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_FAILED",
                        reason="local_progress_physical_action_count_invalid",
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                local_progress_physical_actions += nested_actions
                if nested_actions > self.FIND_APPROACH_MAX_CHUNKS:
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_FAILED",
                        reason="local_progress_exceeded_single_episode_action_limit",
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                if local_progress_physical_actions > max_local_progress_physical_actions:
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_FAILED",
                        reason="local_progress_exceeded_find_object_action_budget",
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                progress_record = {
                    "request_index": local_progress_calls,
                    "terminal_state": local_progress_result.get("terminal_state"),
                    "mode": local_progress_result.get("mode"),
                    "physical_actions": nested_actions,
                    "reason": local_progress_result.get("reason"),
                    "nested_result": local_progress_result,
                }
                local_progress_history.append(progress_record)
                step = {
                    "step_index": approach_attempted,
                    "local_progress_result": local_progress_result,
                    "physical_actions": nested_actions,
                    "post_motion_confirmation_status": None,
                    "post_motion_confirmation_diagnostics": None,
                }
                approach_steps.append(step)
                approach_result = local_progress_result
                terminal = local_progress_result.get("terminal_state")
                if terminal != "LOCAL_PROGRESS_COMPLETE":
                    state = {
                        "LOCAL_PROGRESS_BLOCKED": "APPROACH_BLOCKED",
                        "LOCAL_PROGRESS_SAFETY_VETO": "APPROACH_BLOCKED",
                        "LOCAL_PROGRESS_EXECUTION_FAILED": "APPROACH_FAILED",
                        "LOCAL_PROGRESS_OWNERSHIP_REJECTED": "APPROACH_BLOCKED",
                        "LOCAL_PROGRESS_MAX_STEPS_REACHED": "APPROACH_BLOCKED",
                    }.get(terminal, "APPROACH_FAILED")
                    return result(
                        ok=False,
                        completed=True,
                        state=state,
                        reason=local_progress_result.get(
                            "reason", terminal or "local_progress_failed"
                        ),
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                if nested_actions <= 0:
                    return result(
                        ok=False,
                        completed=True,
                        state="APPROACH_FAILED",
                        reason="local_progress_completed_without_physical_action",
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                approach_completed += 1

                post_cutoff = post_motion_cutoff(current)
                if self._vision_timestamp_is_iso(post_cutoff):
                    post_cutoff = datetime.now(timezone.utc).isoformat()
                confirmed, confirmation_status, confirmation_diagnostics = (
                    self._confirm_find_target_with_semantic(
                        target_name,
                        minimum_timestamp=post_cutoff,
                        return_diagnostics=True,
                        confirmation_window_seconds=(
                            self.FIND_POST_MOTION_CONFIRMATION_WINDOW_SECONDS
                        ),
                    )
                )
                step["post_motion_confirmation_status"] = confirmation_status
                step["post_motion_confirmation_diagnostics"] = confirmation_diagnostics
                if confirmed is None:
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="TARGET_LOST_AFTER_APPROACH",
                        reason="Target was not freshly re-confirmed after local progress.",
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                promoted = self._promote_confirmed_target(confirmed)
                if promoted is None:
                    return result(
                        ok=False,
                        completed=True,
                        target_found=False,
                        state="TARGET_LOST_AFTER_APPROACH",
                        reason="Target promotion failed after local progress.",
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        local_progress_result=local_progress_result,
                        **telemetry,
                    )
                current = promoted
                telemetry = target_telemetry(current, telemetry)

                # A bounded episode owns its complete movement budget.  Return
                # to the caller after fresh target perception; never add a
                # fifth action or start another avoidance episode here.
                if local_progress_result.get("mode") == "BOUNDED_AVOIDANCE":
                    return result(
                        ok=True,
                        completed=False,
                        state="APPROACH_AVOIDANCE_COMPLETE",
                        reason="Bounded local avoidance completed and target was freshly re-confirmed; Find Object continues with perception.",
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        local_progress_result=local_progress_result,
                        approach_result=local_progress_result,
                        **telemetry,
                    )
                if approach_completed >= self.FIND_APPROACH_MAX_CHUNKS:
                    return result(
                        ok=True,
                        completed=True,
                        state="APPROACH_SEQUENCE_COMPLETE",
                        reason="Bounded approach sequence completed and target was freshly re-confirmed.",
                        confirmation_status=confirmation_status,
                        confirmation_diagnostics=confirmation_diagnostics,
                        local_progress_result=local_progress_result,
                        approach_result=local_progress_result,
                        **telemetry,
                    )
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
            try:
                permitted, reason = refresh()
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
                    safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
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
        semantic_target = str(target_name or "").strip().lower()
        detector_target = self._detector_target_label(semantic_target)
        if (
            self.world_model is not None
            and hasattr(
                self.world_model,
                "find_latest_entity_by_label",
            )
        ):
            return self.world_model.find_latest_entity_by_label(
                semantic_target,
                max_age_seconds=self.TARGET_MAX_AGE_SECONDS,
                refresh=True,
            )

        if (
            self.vision is not None
            and hasattr(self.vision, "find_target")
        ):
            return self.vision.find_target(detector_target)

        raise RuntimeError(
            "No World Model perception source is available."
        )

    @classmethod
    def _detector_target_label(cls, target_name):
        """Map only the dedicated Marvin identity to its detector alias."""
        target = str(target_name or "").strip().lower()
        return (
            cls.MARVIN_DETECTOR_ALIAS
            if target == cls.MARVIN_SEMANTIC_TARGET
            else target
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
