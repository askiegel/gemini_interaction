#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from behavior_manager import (
    BehaviorManager, FIND_MARVIN_FORWARD_SPEED_MPS,
    LOCAL_AVOIDANCE_LIDAR_POLL_INTERVAL_SECONDS,
)
from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT
from marvin_arrival_policy import evaluate_marvin_visual_arrival
from marvin_lidar_standoff import TARGET_STANDOFF_M, evaluate_marvin_lidar_standoff
from marvin_target_range_association import MarvinTargetRangeAssociation
from marvin_route_obstruction import route_progress, evaluate_route_progress
from marvin_local_bypass import plan_local_bypass, evaluate_avoidance_progress
from marvin_progress_diagnostics import MarvinProgressDiagnostics
from marvin_local_obstacle_avoidance import (
    MAX_LOCAL_AVOIDANCE_ACTIONS, TURN_SPEED, TURN_DURATION, select_marvin_detour,
    select_marvin_escape_action, LOCAL_AVOIDANCE_STRAFE_SPEED_MPS, LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
)
from marvin_search_policy import MAX_SCAN_TURNS, SCAN_DIRECTION
from marvin_pursuit_state import (
    FIND_CENTER_TOLERANCE_PIXELS,
    VISUAL_READY_TO_ALIGN,
    VISUAL_READY_TO_APPROACH,
    evaluate_marvin_pursuit_state,
)
from config import load_config
from lidar_perception import (
    LidarPerceptionWorker, MAXIMUM_EFFECTIVE_AGE_SECONDS, unavailable_state,
)
from robot_bridge.forward_interlock import (
    ForwardMotionInterlock,
    evaluate_lidar_state,
)
from mission_manager import MissionManager
from provider_factory import create_provider
from robot_bridge.client import RobotBridgeClient
from tracking_state import build_tracking_state, empty_tracking_state
from tony2_localization_facade import Tony2LocalizationFacade
from vision_adapter import VisionAdapter
from semantic_vision import SemanticVisionClient
from world_model import WorldModel
from local_reactive_obstacle_avoidance import (
    FORWARD_CLEAR,
    STOP_BLOCKED,
    TURN_LEFT,
    TURN_RIGHT,
    decide_forward_reaction,
)
from local_motion_safety_envelope import (
    MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR,
    OCTANT_SECTORS,
    evaluate_local_motion_safety,
)


def _bounded_alignment_number(value, *, maximum):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 < float(value) <= maximum
    )


def _marvin_alignment_lidar_is_current(value, session):
    return (
        isinstance(value, dict)
        and value.get("available") is True
        and value.get("valid") is True
        and value.get("reason") == "fresh"
        and value.get("producer_session") == session
        and isinstance(value.get("local_motion_geometry"), dict)
        and value["local_motion_geometry"].get("valid") is True
    )


@dataclass
class _MarvinLiveProofOwner:
    """Exclusive proof lease, never a MissionManager mission."""
    mission_id: str  # Diagnostic correlation ID used by the shared loop.
    generation: int
    thread_id: int
    dispatch_opportunities: int = 0


@dataclass(frozen=True)
class _MarvinLiveProofContinuation:
    """Process-local planning checkpoint; contains no dispatch authority.

    Old geometry is comparison history only. The tracker stays in its existing
    BehaviorManager episode, and every invocation observes and validates again.
    """
    generation: int
    producer_session: str
    avoidance: dict
    previous_selection: Optional[dict]
    previous_clearances: Optional[dict]
    avoidance_history: list
    avoidance_lidar_refresh_history: list
    action_history: list
    last_action_state: str
    previous_stamp: int
    camera_floor_stamp: int
    action_finished_monotonic_seconds: float
    action_lidar_evidence: tuple
    post_lidar_sequence: int
    identity_source_frame_stamp_ns: int
    acquired: bool
    search_turns: int
    reacquisition_attempts: int
    consecutive_reacquisition_failures: int
    range_association: dict
    completion: dict


def _marvin_proof_selection_history(selection):
    """Retain selector inputs, never old permission, options or JIT results."""
    if selection is None:
        return None
    return copy.deepcopy({key: selection[key] for key in (
        "action_type", "direction", "route", "left_clearance_m", "right_clearance_m",
        "ineffective_action_types", "first_post_action_bypass_progress",
    ) if key in selection})


def _marvin_proof_checkpoint_digest(checkpoint):
    return hashlib.sha256(json.dumps(asdict(checkpoint), sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class CognitiveRuntime:
    """
    Persistent cognitive runtime for Mini Pupper 2.

    This class owns the long-lived instances of:

        Provider
        MissionManager
        WorldModel
        VisionAdapter
        RobotBridgeClient
        BehaviorManager

    Mission submission and mission execution remain separate operations.

    The runtime executes one active mission at a time. When that mission
    finishes, MissionManager automatically activates the next queued mission.
    """

    LOOP_INTERVAL_SECONDS = 0.03
    FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS = 6
    FIND_MARVIN_MAX_EPISODES = 4
    FIND_MARVIN_SCAN_MAX_EPISODES = 6
    MAX_CONSECUTIVE_MARVIN_REACQUISITION_FAILURES = 3
    # One recovery per existing full-search turn allowance is a secondary
    # mission bound; successes reset failures, but cannot recover forever.
    MAX_MARVIN_REACQUISITION_EPISODES = MAX_SCAN_TURNS
    MAX_CONSECUTIVE_MARVIN_LIDAR_INTERRUPTION_FAILURES = 3
    MAX_LOCAL_AVOIDANCE_ACTIONS = MAX_LOCAL_AVOIDANCE_ACTIONS
    MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS = 1.0
    # LD06 acquisition polls every 0.08 s with a 0.25 s request deadline.
    # Allow two freshness windows for a new publication, without relaxing age.
    MARVIN_NEW_LIDAR_TIMEOUT_SECONDS = 2 * MAXIMUM_EFFECTIVE_AGE_SECONDS
    MARVIN_NEW_LIDAR_POLL_SECONDS = LOCAL_AVOIDANCE_LIDAR_POLL_INTERVAL_SECONDS
    MAX_ACTIVE_LOCALIZATION_TURNS = 6
    ACTIVE_LOCALIZATION_TURN_SPEED = 0.25
    ACTIVE_LOCALIZATION_TURN_DURATION = 0.50
    MAX_LOCAL_REACTIVE_ACTIONS_PER_STEP = 1
    MAX_REACTIVE_STEPS = 4

    def __init__(
        self,
        provider=None,
        mission_manager=None,
        world_model=None,
        vision_adapter=None,
        robot_client=None,
        behavior_manager=None,
        loop_interval=None,
        lidar_worker_factory=None,
        semantic_vision=None,
        localization_facade=None,
        marvin_camera_model=None,
    ):
        self.config = None

        if provider is None:
            self.config = load_config()
            provider = create_provider(self.config)

        self.provider = provider
        self.mission_manager = mission_manager or MissionManager()
        self.world_model = world_model or WorldModel()

        self.vision_adapter = vision_adapter or VisionAdapter(
            world_model=self.world_model,
        )

        if semantic_vision is None and self.config is not None:
            semantic_vision = SemanticVisionClient.from_config(self.config)

        self.robot_client = robot_client or RobotBridgeClient(
            timeout=15.0,
        )
        self.localization_facade = (
            localization_facade or Tony2LocalizationFacade()
        )

        self.behavior_manager = behavior_manager or BehaviorManager(
            robot_client=self.robot_client,
            vision_adapter=self.vision_adapter,
            world_model=self.world_model,
            semantic_vision=semantic_vision,
        )
        # Measured camera intrinsics/extrinsics only; absence blocks V2 approach.
        # See marvin_lidar_standoff.py for the calibration JSON fields.
        if marvin_camera_model is None:
            try:
                marvin_camera_model = json.loads(os.getenv("MARVIN_CAMERA_LIDAR_CALIBRATION", "null"))
            except (ValueError, TypeError):
                marvin_camera_model = {}
        self.marvin_camera_model = marvin_camera_model
        self._marvin_last_action_lidar_evidence = None
        self._marvin_target_range_association = MarvinTargetRangeAssociation()
        self.marvin_progress_diagnostics = MarvinProgressDiagnostics()
        self.behavior_manager.marvin_command_diagnostic_callback = (
            lambda *args: self._retain_marvin_diagnostic("command_event", *args)
        )
        self.behavior_manager.marvin_perception_diagnostic_callback = (
            lambda *args: self._retain_marvin_diagnostic("perception_event", *args)
        )

        self.loop_interval = (
            float(loop_interval)
            if loop_interval is not None
            else self.LOOP_INTERVAL_SECONDS
        )

        self.running = False
        self.started_at = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.tracking_state: Dict[str, Any] = empty_tracking_state()
        self._state_lock = threading.RLock()
        # Alignment authorization is observation-scoped: a strict V2 tracker
        # frame may admit at most one physical turn, never one per process.
        self._marvin_alignment_observation = None
        self._marvin_alignment_consensus = []
        self._marvin_alignment_geometry_history = []
        self._marvin_alignment_consumed_source_frame_stamps = set()
        self._marvin_approach_step_consumed = False
        self._marvin_autonomous_run_consumed = False
        self._marvin_live_proof_consumed = False
        self._marvin_live_proof_owner = None
        self._marvin_live_proof_state = "UNINITIALIZED"
        self._marvin_live_proof_continuation = None
        self._marvin_live_proof_checkpoint_digest = None
        self._marvin_live_proof_invalidation_reason = None
        self._marvin_controller_lock = threading.RLock()
        self._active_localization_lock = threading.Lock()
        self._local_reactive_step_lock = threading.Lock()
        self._bounded_local_reactive_avoidance_lock = threading.Lock()
        # This lock serializes only high-level routing.  Lower coordinators
        # retain their own ownership locks while they perform physical work.
        self._local_progress_with_avoidance_lock = threading.Lock()
        self._physical_action_lock = threading.Lock()
        self._last_runtime_state = None
        self._control_generation = 0
        self._behavior_execution_generation = None
        self._behavior_execution_thread_id = None
        self._find_object_progress_context = threading.local()
        self._find_marvin_progress_context = threading.local()
        self._lidar_lifecycle_lock = threading.RLock()
        self._lidar_started = False
        self._lidar_stopped = False
        self._lidar_error = None
        self.lidar_worker = None
        self.forward_interlock = None
        self.behavior_manager.tracking_state_callback = (
            self._publish_behavior_tracking
        )
        self.behavior_manager.execution_authorization_provider = (
            self._behavior_execution_is_current
        )
        self.behavior_manager.local_progress_with_avoidance_handler = (
            self._run_find_object_local_progress
        )
        self.behavior_manager.marvin_local_progress_with_avoidance_handler = (
            self._run_find_marvin_local_progress
        )
        try:
            factory = lidar_worker_factory or LidarPerceptionWorker
            self.lidar_worker = factory(
                self.world_model,
                base_url=getattr(self.robot_client, "base_url", None),
            )
            try:
                self.lidar_worker.diagnostic_sample_callback = lambda sample: self._retain_marvin_diagnostic("record_lidar", sample)
            except Exception:
                pass  # Optional retention cannot disable the safety producer.
            self.forward_interlock = ForwardMotionInterlock(
                self.world_model.get_lidar_obstacles,
                expected_session=self.lidar_worker.session,
                stop_callback=self.robot_client.stop,
            )
            self.behavior_manager.lidar_session_provider = (
                lambda: self.lidar_worker.session
            )
            configure = getattr(self.robot_client, "configure_forward_interlock", None)
            if callable(configure):
                configure(self.forward_interlock)
        except Exception as exc:
            self._lidar_error = str(exc)
            try:
                self.world_model.publish_lidar_obstacles(unavailable_state("worker_creation_failed"))
            except Exception:
                pass

    def _retain_marvin_diagnostic(self, method, *args, **kwargs):
        """Diagnostics have no return path into control or safety admission."""
        try:
            return getattr(self.marvin_progress_diagnostics, method)(*args, **kwargs)
        except Exception:
            return None

    def _start_lidar(self):
        with self._lidar_lifecycle_lock:
            if self._lidar_started or self._lidar_stopped or self.lidar_worker is None:
                return
            self._lidar_started = True
            try:
                self.lidar_worker.start()
            except Exception as exc:
                self._lidar_error = str(exc)
                self._stop_lidar()

    def _publish_behavior_tracking(self, result):
        """Publish in-progress visual behavior telemetry safely.

        BehaviorManager calls this only for transient FIND_OBJECT states.
        The execution generation check prevents a late callback from
        overwriting an operator STOP state.
        """
        with self._state_lock:
            if (
                self._behavior_execution_generation is None
                or self._behavior_execution_generation != self._control_generation
            ):
                return
            self.tracking_state = build_tracking_state(
                result,
                previous=self.tracking_state,
            )

    def _behavior_execution_is_current(self):
        """Report whether the active behavior generation is still valid."""
        with self._state_lock:
            return (
                self._behavior_execution_generation is not None
                and self._behavior_execution_generation
                == self._control_generation
            )

    def build_find_marvin_controller_state(self, *, require_fresh_gemini=False):
        """Return one fresh Marvin controller evidence bundle without action."""
        builder = getattr(
            self.behavior_manager,
            "build_find_marvin_controller_state",
            None,
        )
        if not callable(builder):
            raise RuntimeError("find_marvin_state_provider_unavailable")
        if require_fresh_gemini:
            return builder(require_fresh_gemini=True)
        return builder()

    def dry_run_find_marvin_controller(self, *, execute=False):
        """Evaluate one Find-Marvin controller decision with action disabled."""
        if execute is not False:
            return {
                "ok": False,
                "target": "marvin",
                "execution_authorized": False,
                "motion_executed": False,
                "reason": "find_marvin_execution_not_authorized",
            }
        controller = getattr(
            self.behavior_manager,
            "execute_find_marvin_controller",
            None,
        )
        if not callable(controller):
            return {
                "ok": False,
                "target": "marvin",
                "execution_authorized": False,
                "motion_executed": False,
                "reason": "find_marvin_controller_unavailable",
            }
        try:
            result = controller(
                self.build_find_marvin_controller_state,
                max_actions=1,
                dry_run=True,
            )
        except Exception as exc:
            return {
                "ok": False,
                "target": "marvin",
                "execution_authorized": False,
                "motion_executed": False,
                "reason": "find_marvin_dry_run_exception",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        if not isinstance(result, dict):
            return {
                "ok": False,
                "target": "marvin",
                "execution_authorized": False,
                "motion_executed": False,
                "reason": "find_marvin_dry_run_result_malformed",
            }
        history = result.get("history")
        latest = history[-1] if isinstance(history, list) and history else {}
        return dict(
            result,
            target="marvin",
            controller_ready=result.get("ok") is True,
            pursuit_state=(
                latest.get("pursuit_state")
                if isinstance(latest, dict) else None
            ),
            next_route=result.get("next_route", "none"),
            execution_authorized=False,
            motion_executed=False,
        )

    def observe_find_marvin_v2(self):
        # Diagnostics must not supersede a mission-owned observation.
        lock = getattr(self, "_marvin_controller_lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            return {"ok": False, "read_only": True, "executed": False,
                    "reason": "find_marvin_controller_owned"}
        try:
            return self._observe_find_marvin_v2()
        finally:
            if lock is not None:
                lock.release()

    def _observe_find_marvin_v2(self, *, reacquisition_source_floor=None):
        """Return one strict V2 decision without controller or motion side effects."""
        base = {
            "ok": False,
            "read_only": True,
            "authoritative": False,
            "executed": False,
            "fresh_gemini_required": True,
            "session_continuity_used": False,
            "identity_confirmed": False,
            "identity_source": None,
            "proposal_label": None,
            "proposal_confidence": None,
            "opencv_tracker": None,
            "strict_tracker_episode": None,
            "controller": {
                "state": "INSUFFICIENT_EVIDENCE",
                "decision": "REVERIFY_REQUIRED",
                "reason": None,
                "center_tolerance_pixels": FIND_CENTER_TOLERANCE_PIXELS,
            },
        }
        # A V2 observation supersedes any prior alignment authorization even
        # when it is non-authorizing or fails closed.  This prevents a later
        # POST from using an older strict frame after the current scene has
        # changed.  A qualifying observation is installed below only after
        # the current V2 result has been evaluated.
        with self._state_lock:
            self._marvin_alignment_observation = None
        observer = getattr(self.behavior_manager, "observe_find_marvin_v2", None)
        if not callable(observer):
            self._reset_marvin_alignment_consensus()
            return dict(base, reason="find_marvin_v2_observer_unavailable")
        try:
            if reacquisition_source_floor is None:
                evidence = observer()
            else:
                evidence = self.behavior_manager.reacquire_find_marvin_v2(
                    minimum_source_frame_stamp_ns=reacquisition_source_floor,
                )
        except Exception as exc:
            self._reset_marvin_alignment_consensus()
            return dict(base, reason="find_marvin_v2_observation_failed",
                        error=str(exc), error_type=type(exc).__name__)
        if not isinstance(evidence, dict):
            self._reset_marvin_alignment_consensus()
            return dict(base, reason="find_marvin_v2_observation_malformed")
        preview = evidence.get("preview_result")
        if not isinstance(preview, dict):
            self._reset_marvin_alignment_consensus()
            return dict(base, reason="find_marvin_v2_preview_malformed")
        tracker = preview.get("opencv_tracker")
        identity_source = preview.get("identity_source")
        common = dict(
            # Tony2 process-relative diagnostic, not a portable timestamp.
            received_monotonic_seconds=(tracker.get("received_monotonic_seconds")
                if isinstance(tracker, dict) and tracker.get("active") is True
                else preview.get("received_monotonic_seconds")),
            received_monotonic_clock="local_process_relative",
            source_frame_stamp_ns=(tracker.get("source_frame_stamp_ns")
                                   if isinstance(tracker, dict) and tracker.get("active") is True
                                   else preview.get("source_frame_stamp_ns")),
            target_found=preview.get("target_found") is True,
            perception_reason=preview.get("reason"),
            identity_confirmed=preview.get("identity_confirmed") is True,
            identity_source=identity_source,
            identity_source_frame_stamp_ns=preview.get("identity_source_frame_stamp_ns"),
            proposal_label=preview.get("proposal_label"),
            proposal_confidence=preview.get("proposal_confidence"),
            opencv_tracker=dict(tracker) if isinstance(tracker, dict) else None,
            strict_tracker_episode=(
                dict(preview["strict_tracker_episode"])
                if isinstance(preview.get("strict_tracker_episode"), dict)
                else None
            ),
            session_continuity_used=(identity_source == "marvin_session_continuity"),
            marvin_tracking_episode=preview.get("marvin_tracking_episode"),
            post_action_tracker_diagnostics=preview.get("post_action_tracker_diagnostics"),
        )
        post_action_continuity = (
            preview.get("post_action_tracker_continuity") is True
            and identity_source == "marvin_locked_tracker_continuity"
        )
        if post_action_continuity:
            common["fresh_gemini_required"] = False
            common["post_action_tracker_continuity"] = True
            common["post_action_source_frame_stamp_ns"] = preview.get(
                "post_action_source_frame_stamp_ns"
            )
        # V2 verification is deliberately repeated here as a response gate:
        # generic Gemini proposal labels are accepted, continuity is not.
        verifier = getattr(self.behavior_manager, "_marvin_v2_preview_is_verified", None)
        verified = callable(verifier) and verifier(preview) is True
        if not verified:
            self._reset_marvin_alignment_consensus()
            return dict(base, **common, reason="find_marvin_v2_fresh_identity_required",
                        controller=dict(base["controller"], state="SEARCHING",
                                        decision="REVERIFY_REQUIRED",
                                        reason="find_marvin_v2_fresh_identity_required"))
        pursuit = evaluate_marvin_pursuit_state(
            preview, evidence.get("target_lock_result"),
            evidence.get("target_lock_snapshot"),
            selected_identity_id=evidence.get("selected_identity_id"),
            identity_evidence=evidence.get("identity_evidence"),
            bridge_result=evidence.get("bridge_result"),
        )
        state = pursuit.get("state", "INSUFFICIENT_EVIDENCE") if isinstance(pursuit, dict) else "INSUFFICIENT_EVIDENCE"
        reason = pursuit.get("reason") if isinstance(pursuit, dict) else "pursuit_result_malformed"
        visual_arrival = evaluate_marvin_visual_arrival(preview)
        arrival = {"ok": False, "arrived_at_marvin": False,
                   "authority": "target_bearing_lidar", "reason": "marvin_not_centered"}
        if state == VISUAL_READY_TO_ALIGN:
            # Seed/retain range across pure turns; range never authorizes alignment.
            arrival = dict(self._marvin_v2_lidar_arrival(tracker, commit_range=True),
                           arrived_at_marvin=False)  # Centering is mandatory even at close range.
            error = pursuit.get("horizontal_error")
            decision = "TURN_LEFT" if error < 0 else "TURN_RIGHT" if error > 0 else "BLOCKED"
        elif state == VISUAL_READY_TO_APPROACH:
            arrival = self._marvin_v2_lidar_arrival(tracker, commit_range=True)
            reason = arrival["reason"]
            if arrival.get("ok") is not True:
                state, decision = "BLOCKED", "BLOCKED"
            elif arrival.get("target_range_association_trusted") is not True:
                if (arrival.get("direct_path_blocked") is True
                        or arrival.get("route_to_marvin_obstructed") is True):
                    decision = "AVOID"
                else:
                    state, decision = "BLOCKED", "BLOCKED"
            elif arrival.get("arrived_at_marvin") is True:
                state, decision = "ARRIVED", "ARRIVED"
            elif arrival.get("route_to_marvin_obstructed") is True:
                decision = "AVOID"
            else:
                decision = "FORWARD"
        elif state == "SEARCHING":
            decision = "SEARCH"
        else:
            decision = "BLOCKED"
        result = dict(base, **common, ok=True, reason=reason,
                      controller={"state": state, "decision": decision,
                                  "reason": reason,
                                  "distance_state": ("ARRIVED" if decision == "ARRIVED" else
                                                     "APPROACH" if decision == "FORWARD" else "UNKNOWN"),
                                  "center_tolerance_pixels": FIND_CENTER_TOLERANCE_PIXELS},
                      arrival=arrival, visual_arrival=visual_arrival)
        result.update({key: arrival.get(key) for key in (
            "nearest_forward_obstacle_distance_m", "candidate_target_return_distance_m",
            "verified_marvin_distance_m", "target_range_association_trusted",
            "target_range_association_reason", "direct_path_blocked", "route_to_marvin_obstructed",
            "blocking_obstacle_distance_m", "blocking_obstacle_bearing_deg",
            "blocking_obstacle_x_m", "blocking_obstacle_y_m")})
        if decision == "AVOID":
            result["controller"]["path_state"] = "DIRECT_PATH_BLOCKED"
        geometry_continuity = self._update_marvin_alignment_geometry_continuity(
            result, tracker, state, decision,
        )
        result["geometry_continuity"] = geometry_continuity
        # A current strict tracker observation authorizes at most one bounded
        # action.  The former three-observation turn consensus made normal
        # closed-loop pursuit brittle: one transient tracker-quality miss
        # discarded otherwise current Marvin evidence.  Geometry continuity
        # still gates turns, while a centered strict observation independently
        # authorizes one guarded forward step.  In both cases the exact source
        # stamp is consumed before physical dispatch below.
        action_observation = None
        if geometry_continuity["accepted"] is True:
            action_observation = self._marvin_v2_action_observation(
                result, tracker, state, decision,
            )
        elif state == VISUAL_READY_TO_APPROACH and decision in {"FORWARD", "AVOID"}:
            action_observation = self._marvin_v2_action_observation(
                result, tracker, state, decision,
            )
        with self._state_lock:
            self._marvin_alignment_consensus = []
            self._marvin_alignment_observation = action_observation
        return result

    def _marvin_v2_lidar_arrival(self, tracker, lidar=None, *, commit_range=False, current_action_jit=False):
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if lidar is None and getattr(worker, "running", False) is True:
            reader = getattr(getattr(self, "world_model", None), "get_lidar_obstacles", None)
            if callable(reader):
                try:
                    lidar = reader(expected_session=session)
                except Exception:
                    lidar = None
        result = evaluate_marvin_lidar_standoff(
            tracker, lidar, getattr(self, "marvin_camera_model", None), expected_session=session,
        )
        prior = getattr(self, "_marvin_last_action_lidar_evidence", None)
        if (result.get("ok") is True and prior is not None and session == prior[0]
                and (result["acquisition_sequence"] < prior[1] if current_action_jit
                     else result["acquisition_sequence"] <= prior[1])):
            return dict(result, ok=False, arrived_at_marvin=False,
                        reason="target_lidar_newer_observation_required")
        with self._state_lock:
            if not hasattr(self, "_marvin_target_range_association"):
                self._marvin_target_range_association = MarvinTargetRangeAssociation()
            association = self._marvin_target_range_association.evaluate(
                result, lidar, tracker, expected_session=session,
                commit=(commit_range and self._marvin_motion_stamp_is_fresh(
                    tracker.get("source_frame_stamp_ns"),
                    tracker.get("received_monotonic_seconds"))))
        route = association.get("route") or {}
        if route.get("valid") and route.get("route_to_marvin_obstructed"):
            # Advisory evidence on the EXACT association scan. Later reporting
            # must not mix this route with geometry from another generation.
            association["local_bypass_candidates"] = {side: plan_local_bypass(
                lidar, association, route, expected_session=session, side=side)
                for side in ("LEFT", "RIGHT")}
        return association

    MARVIN_ALIGNMENT_GEOMETRY_HISTORY_WINDOW = 3
    MARVIN_ALIGNMENT_MAX_SEED_CENTER_DELTA_PIXELS = 30.0

    def _reset_marvin_alignment_consensus(self):
        """Clear all pending alignment evidence after a failed V2 observation."""
        with self._state_lock:
            self._marvin_alignment_observation = None
            self._marvin_alignment_consensus = []
            self._marvin_alignment_geometry_history = []

    def _marvin_motion_stamp_is_fresh(self, stamp, received_monotonic_seconds=None):
        """Remote stamp identifies a frame; only local receipt measures age."""
        receipt = received_monotonic_seconds
        now = time.monotonic()
        return (type(stamp) is int and stamp >= 0
                and type(receipt) in (int, float)
                and 0 <= receipt <= now and math.isfinite(receipt)
                and 0 <= now - receipt
                <= self.MARVIN_MOTION_OBSERVATION_MAX_AGE_SECONDS)

    @staticmethod
    def _marvin_alignment_consensus_sample(result, tracker, state, decision):
        """Extract one strictly valid current observation for turn consensus."""
        if (
            not isinstance(result, dict)
            or result.get("identity_confirmed") is not True
            or result.get("identity_source") not in {
                "gemini_marvin_candidate_selection",
                "marvin_locked_tracker_continuity",
            }
            or (result.get("identity_source") == "marvin_locked_tracker_continuity"
                and (result.get("post_action_tracker_continuity") is not True
                     or type(result.get("post_action_source_frame_stamp_ns")) is not int
                     or not isinstance(tracker, dict)
                     or type(tracker.get("source_frame_stamp_ns")) is not int
                     or tracker["source_frame_stamp_ns"]
                     <= result["post_action_source_frame_stamp_ns"]))
            or state != VISUAL_READY_TO_ALIGN
            or decision not in {"TURN_LEFT", "TURN_RIGHT"}
            or not isinstance(tracker, dict)
            or tracker.get("active") is not True
            or tracker.get("matched") is not True
        ):
            return None
        stamp = tracker.get("source_frame_stamp_ns")
        quality = tracker.get("quality")
        threshold = tracker.get("threshold")
        bbox = tracker.get("bbox")
        if (
            type(stamp) is not int
            or stamp < 0
            or not isinstance(quality, (int, float))
            or isinstance(quality, bool)
            or not math.isfinite(float(quality))
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or float(quality) < float(threshold)
            or not isinstance(bbox, dict)
        ):
            return None
        try:
            x1, y1, x2, y2 = (
                bbox[key] for key in ("x1", "y1", "x2", "y2")
            )
        except KeyError:
            return None
        values = (x1, y1, x2, y2)
        if (
            any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) for value in values)
            or float(x2) <= float(x1)
            or float(y2) <= float(y1)
        ):
            return None
        return {
            "source_frame_stamp_ns": stamp,
            "direction": decision,
            "identity_source": result["identity_source"],
            "center_x": (float(x1) + float(x2)) / 2.0,
        }

    @staticmethod
    def _marvin_v2_action_observation(result, tracker, state, decision):
        """Return the one current strict V2 observation eligible for action.

        This is deliberately current-frame scoped, not a temporal-motion
        consensus.  It accepts only the existing two controller states that
        map to bounded local actions and preserves every identity/tracker
        validation used by the older alignment gate.
        """
        if (
            not isinstance(result, dict)
            or result.get("identity_confirmed") is not True
            or result.get("identity_source") not in {
                "gemini_marvin_candidate_selection",
                "marvin_locked_tracker_continuity",
            }
            or (result.get("identity_source") == "marvin_locked_tracker_continuity"
                and (result.get("post_action_tracker_continuity") is not True
                     or type(result.get("post_action_source_frame_stamp_ns")) is not int
                     or not isinstance(tracker, dict)
                     or type(tracker.get("source_frame_stamp_ns")) is not int
                     or tracker["source_frame_stamp_ns"]
                     <= result["post_action_source_frame_stamp_ns"]))
            or not isinstance(tracker, dict)
            or tracker.get("active") is not True
            or tracker.get("matched") is not True
            or state not in {VISUAL_READY_TO_ALIGN, VISUAL_READY_TO_APPROACH}
            or (state == VISUAL_READY_TO_ALIGN and decision not in {"TURN_LEFT", "TURN_RIGHT"})
            or (state == VISUAL_READY_TO_APPROACH and decision not in {"FORWARD", "AVOID"})
        ):
            return None
        arrival = result.get("arrival")
        if isinstance(arrival, dict) and arrival.get("arrived_at_marvin") is True:
            return None
        if state == VISUAL_READY_TO_APPROACH and (
            not isinstance(arrival, dict) or arrival.get("ok") is not True
            or arrival.get("authority") != "target_bearing_lidar"
            or not isinstance(arrival.get("target_distance_m"), (int, float))
            or isinstance(arrival.get("target_distance_m"), bool)
            or not math.isfinite(arrival["target_distance_m"])
            or (decision == "FORWARD" and (
                arrival.get("target_range_association_trusted") is not True
                or arrival["target_distance_m"] <= TARGET_STANDOFF_M))
            or (decision == "AVOID" and (
                arrival.get("direct_path_blocked") is not True
                and arrival.get("route_to_marvin_obstructed") is not True))
        ):
            return None
        stamp = tracker.get("source_frame_stamp_ns")
        quality = tracker.get("quality")
        threshold = tracker.get("threshold")
        bbox = tracker.get("bbox")
        if (
            type(stamp) is not int or stamp < 0
            or not isinstance(quality, (int, float)) or isinstance(quality, bool)
            or not math.isfinite(float(quality))
            or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or float(quality) < float(threshold)
            or not isinstance(bbox, dict)
        ):
            return None
        try:
            x1, y1, x2, y2 = (bbox[key] for key in ("x1", "y1", "x2", "y2"))
        except KeyError:
            return None
        values = (x1, y1, x2, y2)
        if (
            any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) for value in values)
            or float(x2) <= float(x1) or float(y2) <= float(y1)
        ):
            return None
        return {
            "source_frame_stamp_ns": stamp,
            "received_monotonic_seconds": tracker.get("received_monotonic_seconds"),
            "identity_confirmed": True,
            "identity_source": result["identity_source"],
            "post_action_tracker_continuity": (
                result.get("post_action_tracker_continuity") is True
            ),
            "post_action_source_frame_stamp_ns": result.get(
                "post_action_source_frame_stamp_ns"
            ),
            "opencv_tracker": dict(tracker),
            "controller_state": state,
            "controller_decision": decision,
            "target_standoff": result.get("arrival"),
        }

    def _update_marvin_alignment_geometry_continuity(
        self, result, tracker, state, decision,
    ):
        """Admit only locally continuous fresh tracker geometry to consensus.

        Strict V2 GET observations intentionally create a new local tracker
        from a new Gemini-selected seed.  Its match score validates that one
        short episode, not continuity with the prior GET.  Keep a tiny
        in-memory history of accepted tracker centers so a grossly different
        newly seeded box cannot immediately become motion evidence.
        """
        sample = self._marvin_alignment_consensus_sample(
            result, tracker, state, decision,
        )
        diagnostic = {
            "accepted": False,
            "reason": None,
            "history_length": 0,
            "center_delta_px": None,
        }
        with self._state_lock:
            previous = list(getattr(self, "_marvin_alignment_geometry_history", []))
            if sample is None:
                self._marvin_alignment_geometry_history = []
                diagnostic["reason"] = "not_strict_alignment_geometry"
            elif not previous:
                self._marvin_alignment_geometry_history = [sample]
                diagnostic.update(
                    accepted=True,
                    reason="geometry_baseline_established",
                    history_length=1,
                )
            else:
                last = previous[-1]
                if sample["identity_source"] != last.get("identity_source"):
                    self._marvin_alignment_geometry_history = []
                    diagnostic["reason"] = "geometry_identity_source_changed"
                elif sample["source_frame_stamp_ns"] <= last.get(
                    "source_frame_stamp_ns", -1,
                ):
                    self._marvin_alignment_geometry_history = []
                    diagnostic["reason"] = "geometry_source_stamp_non_monotonic"
                else:
                    centers = sorted(entry["center_x"] for entry in previous)
                    midpoint = len(centers) // 2
                    reference_center = (
                        centers[midpoint]
                        if len(centers) % 2
                        else (centers[midpoint - 1] + centers[midpoint]) / 2.0
                    )
                    center_delta = abs(sample["center_x"] - reference_center)
                    diagnostic["center_delta_px"] = center_delta
                    if center_delta > self.MARVIN_ALIGNMENT_MAX_SEED_CENTER_DELTA_PIXELS:
                        # Do not turn the rejected seed into a new baseline:
                        # recovery needs a wholly new stable sequence.
                        self._marvin_alignment_geometry_history = []
                        diagnostic["reason"] = "geometry_center_discontinuity"
                    else:
                        history = (previous + [sample])[(-self.MARVIN_ALIGNMENT_GEOMETRY_HISTORY_WINDOW):]
                        self._marvin_alignment_geometry_history = history
                        diagnostic.update(
                            accepted=True,
                            reason="geometry_continuous",
                            history_length=len(history),
                        )
            if diagnostic["accepted"] is not True:
                self._marvin_alignment_consensus = []
                self._marvin_alignment_observation = None
            return diagnostic

    def execute_bounded_find_marvin_autonomous(self, *, max_actions):
        """Run one explicitly-authorized, finite Marvin controller episode.

        The controller obtains a new Preview after every action.  Forward
        safety remains inside the active BehaviorManager immediately before
        dispatch; this method deliberately does not cache or pre-authorize a
        LiDAR snapshot. This public test endpoint retains its one-shot guard.
        """
        return self._execute_bounded_find_marvin_episode(
            max_actions=max_actions,
            consume_one_shot=True,
            require_fresh_gemini=True,
        )

    def _invalidate_marvin_live_proof(self, reason):
        """An unrelated owner or failed gate cannot start a fresh proof episode."""
        if not getattr(self, "_marvin_live_proof_consumed", False):
            return
        with self._state_lock:
            self._marvin_live_proof_continuation = None
            self._marvin_live_proof_checkpoint_digest = None
            self._marvin_live_proof_state = "FAILED_LOCKED"
            self._marvin_live_proof_invalidation_reason = reason

    def _marvin_live_proof_idle(self):
        return (self.running is True and self._marvin_live_proof_owner is None
            and self._behavior_execution_generation is None
            and self.mission_manager.get_active_mission() is None
            and not self.mission_manager.get_queue()
            and self.world_model.robot_state.get("runtime_state") == "IDLE")

    def _marvin_live_proof_checkpoint_valid(self):
        """Verify the sealed checkpoint and its stopped completion certificate.

        Historical freshness is checked at completion, never used as current
        motion authority. Rearm and the next step independently read sensors.
        """
        c = self._marvin_live_proof_continuation
        try:
            if (type(c) is not _MarvinLiveProofContinuation
                    or _marvin_proof_checkpoint_digest(c) != self._marvin_live_proof_checkpoint_digest
                    or c.generation != self._control_generation
                    or c.producer_session != getattr(self.lidar_worker, "session", None)
                    or type(c.previous_stamp) is not int or type(c.camera_floor_stamp) is not int
                    or not 0 <= c.previous_stamp < c.camera_floor_stamp
                    or c.previous_stamp not in self._marvin_alignment_consumed_source_frame_stamps
                    or type(c.post_lidar_sequence) is not int
                    or c.action_lidar_evidence[0] != c.producer_session
                    or type(c.action_lidar_evidence[1]) is not int
                    or c.post_lidar_sequence <= c.action_lidar_evidence[1]
                    or c.acquired is not True
                    or type(c.identity_source_frame_stamp_ns) is not int
                    or not 0 <= c.identity_source_frame_stamp_ns < c.camera_floor_stamp
                    or not 0 <= c.avoidance["local_avoidance_actions"] <= self.MAX_LOCAL_AVOIDANCE_ACTIONS
                    or not 0 <= c.avoidance["local_bypass_actions"] <= c.avoidance["local_avoidance_actions"]):
                return False
            proof = c.completion
            if (proof["state"] != "PROOF_COMPLETE" or proof["stop_confirmed"] is not True
                    or proof["full_step_completed"] is not True
                    or proof["actions_executed"] != 1 or proof["dispatch_opportunities"] != 1
                    or proof["physical_dispatches_or_uncertain"] != 1
                    or proof["delivery_uncertain"] is not False
                    or proof["lidar_valid"] is not True or proof["lidar_reason"] != "fresh"
                    or proof["tracker_matched"] is not True
                    or proof["identity_confirmed"] is not True
                    or proof["tracker_quality"] < max(.80, proof["tracker_threshold"])
                    or proof["camera_received_monotonic_seconds"] <= c.action_finished_monotonic_seconds):
                return False
            episode = getattr(self.behavior_manager, "_marvin_v2_tracker_episode", None)
            return (isinstance(episode, dict)
                and episode.get("last_tracker_source_frame_stamp_ns") == c.camera_floor_stamp
                and episode.get("identity_source_frame_stamp_ns") == c.identity_source_frame_stamp_ns)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError, RecursionError):
            return False

    def rearm_find_marvin_live_proof(self, *, rearm):
        """Grant one later continuation invocation; never execute or observe."""
        base = {"ok": False, "action": "find_marvin_live_proof_rearm",
                "max_physical_actions": 1, "motion_executed": False}
        if rearm is not True:
            return dict(base, reason="marvin_live_proof_rearm_invalid")
        if not self._marvin_controller_lock.acquire(blocking=False):
            return dict(base, reason="marvin_live_proof_owner_busy")
        physical_acquired = False
        try:
            with self._state_lock:
                if not self._marvin_live_proof_idle():
                    return dict(base, reason="marvin_live_proof_runtime_not_idle")
                if self._marvin_live_proof_state != "COMPLETE_DISARMED":
                    return dict(base, reason="marvin_live_proof_not_complete_disarmed",
                                proof_state=self._marvin_live_proof_state)
                if not self._marvin_live_proof_checkpoint_valid():
                    self._invalidate_marvin_live_proof("marvin_live_proof_continuation_invalid")
                    return dict(base, reason="marvin_live_proof_continuation_invalid")
                if not self._physical_action_lock.acquire(blocking=False):
                    return dict(base, reason="marvin_live_proof_physical_owner_busy")
                physical_acquired = True
                bridge = self._marvin_bridge_ready_and_stopped()
                if bridge.get("ok") is not True or bridge.get("status") != "READY":
                    self._invalidate_marvin_live_proof("marvin_live_proof_bridge_not_stopped")
                    return dict(base, reason="marvin_live_proof_bridge_not_stopped")
                session, lidar = self._active_localization_lidar_is_current()
                c = self._marvin_live_proof_continuation
                if (session != c.producer_session or lidar is None
                        or type(lidar.get("acquisition_sequence")) is not int
                        or lidar["acquisition_sequence"] < c.post_lidar_sequence):
                    self._invalidate_marvin_live_proof("marvin_live_proof_lidar_discontinuity")
                    return dict(base, reason="marvin_live_proof_lidar_discontinuity")
                self._marvin_live_proof_state = "ARMED"
                return dict(base, ok=True, reason="marvin_live_proof_rearmed", proof_state="ARMED")
        finally:
            if physical_acquired:
                self._physical_action_lock.release()
            self._marvin_controller_lock.release()

    def execute_find_marvin_live_proof_step(self, *, max_physical_actions):
        """One explicitly armed step of the real loop, with no queued mission.

        The explicit lease shares generation/thread admission and all existing
        dispatch locks. It never impersonates or registers an active mission.
        """
        base = {"ok": False, "action": "find_marvin_live_proof_step",
                "execution_authorized": False, "motion_executed": False,
                "actions_executed": 0, "max_physical_actions": 1}
        if type(max_physical_actions) is not int or max_physical_actions != 1:
            return dict(base, reason="marvin_live_proof_limit_invalid")
        lock = self._marvin_controller_lock
        if not lock.acquire(blocking=False):
            return dict(base, reason="marvin_live_proof_owner_busy")
        owner = None
        consumed_before = set()
        try:
            with self._state_lock:
                if self._marvin_live_proof_state not in {"UNINITIALIZED", "ARMED"}:
                    return dict(base, reason="marvin_live_proof_already_consumed",
                                proof_state=self._marvin_live_proof_state)
                if (self.running is not True or self._marvin_live_proof_owner is not None
                        or self._behavior_execution_generation is not None
                        or self.mission_manager.get_active_mission() is not None
                        or self.mission_manager.get_queue()
                        or self.world_model.robot_state.get("runtime_state") != "IDLE"):
                    return dict(base, reason="marvin_live_proof_runtime_not_idle")
                continuation = None
                if self._marvin_live_proof_state == "ARMED":
                    if not self._marvin_live_proof_checkpoint_valid():
                        self._invalidate_marvin_live_proof("marvin_live_proof_continuation_invalid")
                        return dict(base, reason="marvin_live_proof_continuation_invalid")
                    continuation = copy.deepcopy(self._marvin_live_proof_continuation)
                if self._physical_action_lock.locked():
                    return dict(base, reason="marvin_live_proof_physical_owner_busy")
                self._marvin_live_proof_consumed = True
                self._marvin_live_proof_state = "EXECUTING"  # Consume before observation/dispatch.
                self._marvin_live_proof_continuation = None
                self._marvin_live_proof_checkpoint_digest = None
                consumed_before = set(self._marvin_alignment_consumed_source_frame_stamps)
                owner = _MarvinLiveProofOwner("marvin-proof-" + uuid.uuid4().hex,
                    self._control_generation, threading.get_ident())
                self._marvin_live_proof_owner = owner
                self._behavior_execution_generation = owner.generation
                self._behavior_execution_thread_id = owner.thread_id
                self._set_runtime_state("EXECUTING_PROOF")
            base["execution_authorized"] = True
            try:
                result = self._execute_normal_marvin_find_mission_locked(
                    owner, control_generation=owner.generation,
                    proof_max_physical_actions=1, proof_continuation=continuation)
            except Exception as exc:
                try:
                    stopped = self.robot_client.stop()
                except Exception as stop_exc:
                    stopped = {"ok": False, "error": str(stop_exc)}
                result = dict(base, reason="marvin_live_proof_exception", error=str(exc),
                    stop_result=stopped, bridge_after_stop=self._marvin_bridge_ready_and_stopped(),
                    proof={"max_physical_actions": 1, "action_complete": False,
                        "dispatch_opportunities": owner.dispatch_opportunities,
                        "physical_dispatches_or_uncertain": int(bool(
                            self._marvin_alignment_consumed_source_frame_stamps - consumed_before)),
                        "post_action_evidence": None,
                        "source_stamps": [{"source_frame_stamp_ns": stamp, "consumed": True}
                            for stamp in sorted(self._marvin_alignment_consumed_source_frame_stamps - consumed_before)]})
            with self._state_lock:
                if (result.get("state") == "PROOF_COMPLETE"
                        and self._marvin_live_proof_checkpoint_valid()):
                    self._marvin_live_proof_state = "COMPLETE_DISARMED"
                else:
                    self._invalidate_marvin_live_proof(result.get("reason") or "marvin_live_proof_failed")
                    if result.get("state") == "ARRIVED" and result.get("ok") is True:
                        self._marvin_live_proof_state = "ARRIVED_DISARMED"
                    clear = getattr(self.behavior_manager, "_clear_marvin_v2_tracker_episode", None)
                    if callable(clear):
                        clear()
                if owner.generation == self._control_generation:
                    self.last_result = result
                    self.tracking_state = build_tracking_state(result, previous=self.tracking_state)
                    self.tracking_state["active"] = False
            return dict(base, ok=result.get("ok") is True,
                actions_executed=result.get("actions_executed", 0),
                motion_executed=result.get("actions_executed", 0) > 0,
                reason=result.get("reason"), controller_result=result,
                proof_state=self._marvin_live_proof_state,
                continuation_available=self._marvin_live_proof_continuation is not None)
        finally:
            try:
                if owner is not None:
                    try:
                        clear_scan = getattr(self.behavior_manager, "clear_find_marvin_room_scan", None)
                        if callable(clear_scan):
                            clear_scan(owner.mission_id)
                    finally:
                        with self._state_lock:
                            if self._marvin_live_proof_state == "EXECUTING":
                                self._invalidate_marvin_live_proof("marvin_live_proof_exception")
                            self._marvin_live_proof_owner = None
                            if self._behavior_execution_thread_id == owner.thread_id:
                                self._behavior_execution_generation = None
                                self._behavior_execution_thread_id = None
                            if owner.generation == self._control_generation:
                                self._set_runtime_state("IDLE")
            finally:
                lock.release()

    def _marvin_live_proof_owner_is_current(self, owner=None):
        """A proof lease is valid only in its owning thread, while idle otherwise."""
        current = getattr(self, "_marvin_live_proof_owner", None)
        return bool(current is not None and (owner is None or current is owner)
            and self.running is True and current.generation == self._control_generation
            and current.thread_id == threading.get_ident()
            and self._behavior_execution_generation == current.generation
            and self._behavior_execution_thread_id == current.thread_id
            and self.mission_manager.get_active_mission() is None
            and not self.mission_manager.get_queue())

    def _marvin_detour_owner_is_current(self):
        """Require a normal mission owner or the exclusive one-action proof lease."""
        return (self._marvin_live_proof_owner_is_current()
                or (self.mission_manager.get_active_mission() is not None
                    and self._marvin_motion_owner_is_current()))

    def _execute_bounded_find_marvin_episode(
        self, *, max_actions, consume_one_shot,
        require_fresh_gemini=False,
    ):
        """Execute exactly one existing bounded controller episode.

        Normal Marvin mission continuation calls this private boundary with
        ``consume_one_shot=False``; the dedicated autonomous test endpoint
        continues to call the public one-shot wrapper above.
        """
        base = {
            "ok": False,
            "action": "bounded_find_marvin_autonomous_run",
            "execution_authorized": False,
            "motion_executed": False,
            "actions_executed": 0,
            "max_actions": max_actions,
            "controller_result": None,
            "reason": None,
        }
        if (
            not isinstance(max_actions, int)
            or isinstance(max_actions, bool)
            or not 0 < max_actions <= self.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS
        ):
            return dict(base, reason="marvin_autonomous_action_limit_invalid")
        if self.running is not True:
            return dict(base, reason="marvin_autonomous_runtime_not_running")
        behavior = getattr(self, "behavior_manager", None)
        controller = getattr(behavior, "execute_find_marvin_controller", None)
        robot = getattr(behavior, "robot", None)
        stop = getattr(robot, "stop", None)
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if not callable(controller) or not callable(stop):
            return dict(base, reason="marvin_autonomous_controller_or_stop_unavailable")
        if worker is None or worker.running is not True or not isinstance(session, str) or not session:
            return dict(base, reason="marvin_autonomous_lidar_session_unavailable")
        controller_lock = getattr(self, "_marvin_controller_lock", None)
        if controller_lock is None or not controller_lock.acquire(blocking=False):
            return dict(base, reason="marvin_autonomous_controller_already_running")
        try:
            if getattr(self, "_marvin_live_proof_owner", None) is not None:
                return dict(base, reason="marvin_autonomous_controller_already_running")
            if consume_one_shot:
                with self._state_lock:
                    if self._marvin_autonomous_run_consumed:
                        return dict(base, reason="marvin_autonomous_run_already_consumed")
                    self._marvin_autonomous_run_consumed = True
            self._invalidate_marvin_live_proof("another_physical_behavior")
            base["execution_authorized"] = True
            state_provider = self.build_find_marvin_controller_state
            if require_fresh_gemini:
                state_provider = lambda: self.build_find_marvin_controller_state(
                    require_fresh_gemini=True,
                )
            try:
                result = controller(
                    state_provider,
                    max_actions=max_actions,
                    dry_run=False,
                    stop_after_action=stop,
                    require_fresh_gemini=require_fresh_gemini,
                )
            except Exception as exc:
                return dict(
                    base,
                    reason="marvin_autonomous_controller_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            if not isinstance(result, dict):
                return dict(base, reason="marvin_autonomous_controller_result_malformed")
            history = result.get("history")
            moved = bool(isinstance(history, list) and any(
                isinstance(entry, dict)
                and (
                    isinstance(entry.get("pursuit_step_result"), dict)
                    and entry["pursuit_step_result"].get("motion_executed") is True
                    or isinstance(entry.get("search_step_result"), dict)
                    and entry["search_step_result"].get("motion_executed") is True
                )
                for entry in history
            ))
            return dict(
                base,
                ok=result.get("ok") is True,
                motion_executed=moved,
                actions_executed=result.get("actions_executed", 0),
                controller_result=result,
                reason=result.get("reason", "marvin_autonomous_controller_complete"),
            )
        finally:
            controller_lock.release()

    @staticmethod
    def _is_normal_marvin_find_mission(mission):
        """Identify only the normal MissionManager FIND_OBJECT Marvin route."""
        return bool(
            getattr(mission, "mission_type", None) == "FIND_OBJECT"
            and str(getattr(mission, "target", "") or "").strip().lower()
            == "marvin"
        )

    def _execute_normal_marvin_find_mission(
        self, mission, *, control_generation=None,
    ):
        controller_lock = getattr(self, "_marvin_controller_lock", None)
        if controller_lock is None or not controller_lock.acquire(blocking=False):
            return {
                "ok": False,
                "completed": True,
                "behavior": "FIND_OBJECT",
                "target": "marvin",
                "mission_route": "marvin_v2_closed_loop",
                "mission_id": getattr(mission, "mission_id", None),
                "arrived_at_marvin": False,
                "mission_outcome": "safe_failure",
                "state": "FIND_MARVIN_FAILED",
                "reason": "marvin_autonomous_controller_already_running",
            }
        try:
            return self._execute_normal_marvin_find_mission_locked(
                mission,
                control_generation=control_generation,
            )
        finally:
            clear_scan = getattr(
                getattr(self, "behavior_manager", None),
                "clear_find_marvin_room_scan", None,
            )
            if callable(clear_scan):
                clear_scan(getattr(mission, "mission_id", None))
            controller_lock.release()

    def _wait_for_new_marvin_lidar_evidence(
        self, *, expected_session, previous_sequence, execution_guard,
        timeout_seconds=None, allow_transient_stale=False,
    ):
        """Poll World Model telemetry while stopped; never acquire or mint scans."""
        timeout = (self.MARVIN_NEW_LIDAR_TIMEOUT_SECONDS if timeout_seconds is None
                   else timeout_seconds)
        started = time.monotonic()
        polls = 0
        last = None

        def finish(reason, *, ok=False):
            return {"ok": ok, "reason": reason, "snapshot": last,
                    "producer_session": expected_session,
                    "previous_acquisition_sequence": previous_sequence,
                    "poll_count": polls,
                    "wait_elapsed_seconds": max(0.0, time.monotonic() - started)}

        if (not isinstance(expected_session, str) or not expected_session
                or type(previous_sequence) is not int or previous_sequence < 0
                or not _bounded_alignment_number(timeout, maximum=1.0)
                or not callable(execution_guard)):
            return finish("find_marvin_lidar_wait_request_invalid")
        deadline = started + timeout
        while True:
            if not execution_guard():
                return finish("find_marvin_mission_preempted")
            worker = self.lidar_worker
            if worker.session != expected_session:
                return finish("find_marvin_lidar_producer_session_changed")
            if worker.running is not True:
                return finish("find_marvin_lidar_not_current")
            if time.monotonic() >= deadline:
                return finish("find_marvin_new_lidar_evidence_timeout")
            polls += 1
            try:
                last = self.world_model.get_lidar_obstacles(expected_session=expected_session)
            except Exception:
                if not execution_guard():
                    return finish("find_marvin_mission_preempted")
                return finish("find_marvin_lidar_read_failed")
            if not execution_guard():
                return finish("find_marvin_mission_preempted")
            if (worker.session != expected_session or isinstance(last, dict)
                    and last.get("producer_session") != expected_session):
                return finish("find_marvin_lidar_producer_session_changed")
            if worker.running is not True or not isinstance(last, dict):
                return finish("find_marvin_lidar_not_current")
            sequence = last.get("acquisition_sequence")
            if type(sequence) is not int or sequence < previous_sequence:
                return finish("find_marvin_lidar_acquisition_sequence_invalid")
            if time.monotonic() >= deadline:
                return finish("find_marvin_new_lidar_evidence_timeout")
            if sequence > previous_sequence:
                age = last.get("effective_age_seconds")
                fresh = (_marvin_alignment_lidar_is_current(last, expected_session)
                         and type(age) in (int, float) and math.isfinite(age)
                         and 0 <= age <= MAXIMUM_EFFECTIVE_AGE_SECONDS)
                if fresh:
                    return finish("find_marvin_new_lidar_evidence_received", ok=True)
                if not (allow_transient_stale and last.get("reason") in {"stale", "stale_lidar"}):
                    return finish("find_marvin_lidar_not_current")
            # The old acquisition may age out while stopped. It cannot release
            # this wait; only a new scan's freshness/validity can admit progress.
            # STOP/preemption is checked on both sides of every poll and sleep.
            if not execution_guard():
                return finish("find_marvin_mission_preempted")
            time.sleep(min(self.MARVIN_NEW_LIDAR_POLL_SECONDS,
                           max(0.0, deadline - time.monotonic())))

    def _refresh_marvin_avoidance_plan(self, *, tracker, expected_session,
                                      blocked_sequence, execution_guard):
        """Rebuild a stopped avoidance decision from one newer World Model scan.

        This is planning evidence only. The existing executor still consumes
        the camera stamp and independently obtains its final JIT safety scan.
        """
        wait = self._wait_for_new_marvin_lidar_evidence(
            expected_session=expected_session, previous_sequence=blocked_sequence,
            execution_guard=execution_guard)
        result = {
            "ok": False, "decision": None, "wait": wait,
            "blocked_forward_lidar_sequence": blocked_sequence,
            "avoidance_planning_lidar_sequence": None,
            "avoidance_lidar_refresh_required": True,
            "avoidance_lidar_refresh_wait_seconds": wait["wait_elapsed_seconds"],
            "avoidance_lidar_refresh_result": wait["reason"],
        }
        if wait["ok"] is not True:
            reason = ("find_marvin_avoidance_new_lidar_required"
                      if wait["reason"] == "find_marvin_new_lidar_evidence_timeout"
                      else wait["reason"])
            return dict(result, reason=reason, avoidance_lidar_refresh_result=reason)
        lidar = wait["snapshot"]
        result["avoidance_planning_lidar_sequence"] = lidar["acquisition_sequence"]
        if not execution_guard():
            return dict(result, reason="find_marvin_mission_preempted")
        if not self._marvin_motion_stamp_is_fresh(
                tracker.get("source_frame_stamp_ns"), tracker.get("received_monotonic_seconds")):
            return dict(result, reason="marvin_motion_observation_stale")
        association = self._marvin_v2_lidar_arrival(tracker, lidar, commit_range=True)
        result.update(lidar=lidar, association=association)
        if association.get("ok") is not True:
            return dict(result, reason=association.get("reason"))
        route = association.get("route") or {}
        if route.get("valid") is not True:
            return dict(result, reason=route.get("reason") or "marvin_route_geometry_unavailable")
        if (association.get("authority") == "target_bearing_lidar"
                and association.get("target_range_association_trusted") is True
                and association.get("arrived_at_marvin") is True
                and association.get("target_distance_m", float("inf")) <= TARGET_STANDOFF_M):
            return dict(result, ok=True, decision="ARRIVED", reason="arrived_at_marvin")
        duration = (min(0.50, (association["target_distance_m"] - TARGET_STANDOFF_M)
                    / FIND_MARVIN_FORWARD_SPEED_MPS)
                    if association.get("target_range_association_trusted") is True else 0.50)
        direct = evaluate_local_motion_safety(
            lidar, expected_session=expected_session,
            linear_x=FIND_MARVIN_FORWARD_SPEED_MPS, duration=duration)
        result.update(direct=direct, forward_duration=duration)
        if direct.get("reason") not in {"protected_region_clear", "translation_protected_region_violated"}:
            return dict(result, reason=direct.get("reason"))
        if direct.get("permitted") and not route.get("route_to_marvin_obstructed"):
            if association.get("target_range_association_trusted") is not True:
                return dict(result, reason=association["target_range_association_reason"])
            return dict(result, ok=True, decision="FORWARD", reason="direct_path_restored")
        return dict(result, ok=True, decision="AVOID", reason="marvin_route_obstructed")

    def _execute_normal_marvin_find_mission_locked(
        self, mission, *, control_generation=None, proof_max_physical_actions=None,
        proof_continuation=None,
    ):
        """Own the complete V2 observe/action/STOP loop for one mission."""
        if proof_max_physical_actions is not None and (
                type(proof_max_physical_actions) is not int or proof_max_physical_actions != 1
                or not self._marvin_live_proof_owner_is_current(mission)):
            return {"ok": False, "reason": "marvin_live_proof_owner_or_limit_invalid"}
        proof_dispatches = 0
        proof_post_action_evidence = None
        behavior = self.behavior_manager
        if control_generation is None:
            control_generation = self._control_generation
        history = []
        reacquisition_history = []
        tracker_loss_history = []
        lidar_wait_history = []
        lidar_recovery_history = []
        avoidance_history = []
        avoidance_lidar_refresh_history = []
        avoidance = {"local_avoidance_active": False, "local_avoidance_actions": 0,
                     "last_detour_direction": None, "left_clearance_m": None,
                     "right_clearance_m": None, "direct_path_blocked": False,
                     "avoidance_reason": None, "last_detour_improved_direct_path": None,
                     "route_to_marvin_obstructed": False, "previous_action_type": None,
                     "progress_improved": None, "selected_action_type": None,
                     "blocked_forward_lidar_sequence": None,
                     "avoidance_planning_lidar_sequence": None,
                     "avoidance_lidar_refresh_required": False,
                     "avoidance_lidar_refresh_wait_seconds": None,
                     "avoidance_lidar_refresh_result": None,
                     "local_bypass_active": False, "local_bypass_side": None,
                     "local_bypass_target_x_m": None, "local_bypass_target_y_m": None,
                     "local_bypass_actions": 0, "local_bypass_reason": None}
        previous_clearances = None
        previous_selection = None
        if proof_continuation is None:
            self._marvin_target_range_association = MarvinTargetRangeAssociation()
        consecutive_lidar_interruptions = 0
        reacquisition_attempts = 0
        consecutive_reacquisition_failures = 0
        search_turns = 0
        acquired = False
        observation = None
        previous_stamp = None
        action_finished_monotonic_seconds = None
        prior_action_state = None
        camera_floor_stamp = None
        expected_identity_stamp = None
        if proof_continuation is not None:
            c = proof_continuation
            avoidance = copy.deepcopy(c.avoidance)
            previous_selection = copy.deepcopy(c.previous_selection)
            previous_clearances = copy.deepcopy(c.previous_clearances)
            avoidance_history = copy.deepcopy(c.avoidance_history)
            avoidance_lidar_refresh_history = copy.deepcopy(c.avoidance_lidar_refresh_history)
            previous_stamp = c.previous_stamp
            camera_floor_stamp = c.camera_floor_stamp
            expected_identity_stamp = c.identity_source_frame_stamp_ns
            action_finished_monotonic_seconds = c.action_finished_monotonic_seconds
            prior_action_state = c.last_action_state
            acquired = c.acquired
            search_turns = c.search_turns
            reacquisition_attempts = c.reacquisition_attempts
            consecutive_reacquisition_failures = c.consecutive_reacquisition_failures
            self._marvin_target_range_association.__dict__.update(copy.deepcopy(c.range_association))
            self._marvin_last_action_lidar_evidence = c.action_lidar_evidence
        retain = self._retain_marvin_diagnostic
        retain("begin", mission.mission_id,
               expected_session=getattr(self.lidar_worker, "session", None),
               camera_model=self.marvin_camera_model)
        self._reset_marvin_alignment_consensus()
        clear_episode = getattr(behavior, "_clear_marvin_v2_tracker_episode", None)
        if proof_continuation is None and callable(clear_episode):
            clear_episode()  # A new mission requires fresh semantic acquisition.

        def current():
            return self._marvin_mission_context_is_current(mission, control_generation)

        def clear_bypass(reason):
            avoidance.update(local_bypass_active=False, local_bypass_side=None,
                local_bypass_target_x_m=None, local_bypass_target_y_m=None, local_bypass_reason=reason,
                bypass_target_x_m=None, bypass_target_y_m=None, bypass_distance_m=None,
                bypass_bearing_deg=None, route_to_bypass_obstructed=None,
                bypass_corridor_occupancy=None, bypass_corridor_overlap_m=None, bypass_forward_permitted=False)

        def record_avoidance_reassessment(route, association):
            # Capture the first valid reassessment of the last physical detour.
            # Later alignment or planning must not overwrite that action's outcome.
            if (not previous_selection or
                    (history[-1]["state"] if history else prior_action_state) != "AVOIDING"
                    or not route.get("valid") or not avoidance_history
                    or avoidance_history[-1].get("actual_route_occupancy") is not None):
                return
            bypass = (association.get("local_bypass_candidates") or {}).get(previous_selection.get("direction"))
            progress = evaluate_avoidance_progress(previous_selection, route, bypass)
            avoidance_history[-1]["post_action_bypass"] = bypass
            improved = progress["meaningful_progress"]
            if previous_selection.get("action_type") == "BYPASS_FORWARD":
                # Freeze this action's first actual outcome before any ordinary
                # alignment changes the coordinate frame. Later turns cannot
                # manufacture longitudinal passage for an ineffective bypass.
                previous_selection["first_post_action_bypass_progress"] = dict(progress)
            avoidance_history[-1].update(post_action_route=route, progress_improved=improved,
                post_action_target_association=association, actual_route_occupancy=route["route_occupancy"],
                actual_max_overlap_m=route["corridor_overlap_m"],
                actual_blocker_centerline_clearance_m=route.get("blocking_obstacle_centerline_clearance_m"),
                meaningful_progress=improved, meaningful_progress_reason=progress["meaningful_progress_reason"],
                actual_route_progress=progress)
            retain("avoidance_reassessment", route, association, improved, progress)

        def finish(state, reason):
            # Capture planning state before terminal reporting clears bypass
            # telemetry. No returned mutable result is the checkpoint authority.
            planning_avoidance = copy.deepcopy(avoidance)
            clear_bypass(reason)
            avoidance["local_avoidance_active"] = False
            self._reset_marvin_alignment_consensus()
            try:
                stop = self.robot_client.stop()
            except Exception as exc:
                stop = {"ok": False, "error": str(exc)}
            stop_completed_monotonic_seconds = time.monotonic()
            zero = self._marvin_bridge_ready_and_stopped()
            safe = (isinstance(stop, dict) and stop.get("ok") is True
                    and zero.get("ok") is True and zero.get("status") == "READY")
            if not safe:
                state, reason = "BLOCKED", "find_marvin_stop_or_bridge_failed"
            resumable = (proof_max_physical_actions is not None and safe
                         and state == "PROOF_COMPLETE" and acquired)
            if not resumable and callable(clear_episode):
                clear_episode()
            retain("mission_stop", stop, zero, stop_completed_monotonic_seconds)
            retain("terminal", state, reason)
            result = {
                "ok": safe, "completed": True, "behavior": "FIND_OBJECT",
                "target": "marvin", "mission_id": mission.mission_id,
                "mission_route": "marvin_v2_closed_loop", "state": state,
                "reason": reason, "arrived_at_marvin": state == "ARRIVED",
                "mission_outcome": ("arrived_at_marvin" if state == "ARRIVED"
                                    else "safe_incomplete" if safe else "safe_failure"),
                "search_turns": search_turns, "max_search_turns": MAX_SCAN_TURNS,
                "actions_executed": sum(row.get("motion_executed") is True for row in history),
                "completed_bypass_forward_actions": sum(row["result"].get("action_type") == "BYPASS_FORWARD"
                    and row["result"].get("full_step_completed") is True for row in history),
                "interrupted_bypass_forward_attempts": sum(row["result"].get("action_type") == "BYPASS_FORWARD"
                    and row["result"].get("interrupted") is True for row in history),
                "completed_forward_actions": sum(row["state"] == "ADVANCING"
                    and row["result"].get("full_step_completed") is True for row in history),
                "interrupted_forward_attempts": sum(row["state"] == "ADVANCING"
                    and row["result"].get("interrupted") is True for row in history),
                "completed_strafe_actions": sum((row["result"].get("action_type") or "").startswith("STRAFE")
                    and row["result"].get("full_step_completed") is True for row in history),
                "interrupted_strafe_attempts": sum((row["result"].get("action_type") or "").startswith("STRAFE")
                    and row["result"].get("interrupted") is True for row in history),
                "consecutive_lidar_interruptions": consecutive_lidar_interruptions,
                "max_consecutive_lidar_interruptions": self.MAX_CONSECUTIVE_MARVIN_LIDAR_INTERRUPTION_FAILURES,
                "max_lidar_recovery_episodes": self.MAX_MARVIN_REACQUISITION_EPISODES,
                "lidar_recovery_history": lidar_recovery_history,
                **avoidance, "max_local_avoidance_actions": self.MAX_LOCAL_AVOIDANCE_ACTIONS,
                "local_avoidance_history": avoidance_history,
                "avoidance_lidar_refresh_history": avoidance_lidar_refresh_history,
                "history": history, "stop_result": stop, "bridge_after_stop": zero,
                "reacquisition_attempts": reacquisition_attempts,
                "consecutive_reacquisition_failures": consecutive_reacquisition_failures,
                "max_consecutive_reacquisition_failures": self.MAX_CONSECUTIVE_MARVIN_REACQUISITION_FAILURES,
                "max_reacquisition_episodes": self.MAX_MARVIN_REACQUISITION_EPISODES,
                "max_reacquisition_attempts": (self.MAX_MARVIN_REACQUISITION_EPISODES
                                              * self.MAX_CONSECUTIVE_MARVIN_REACQUISITION_FAILURES),
                "reacquisition_history": reacquisition_history,
                "tracker_loss_history": tracker_loss_history,
                "lidar_wait_history": lidar_wait_history,
                "final_observation": observation,
                "progress_diagnostics": retain("snapshot"),
            }
            if proof_max_physical_actions is not None:
                consumed = self._marvin_alignment_consumed_source_frame_stamps
                result["proof"] = {
                    "max_physical_actions": 1, "physical_dispatches_or_uncertain": proof_dispatches,
                    "dispatch_opportunities": mission.dispatch_opportunities,
                    "source_stamps": [{"source_frame_stamp_ns": row["source_frame_stamp_ns"],
                        "consumed": row["source_frame_stamp_ns"] in consumed} for row in history],
                    "post_action_evidence": proof_post_action_evidence,
                    "action_complete": safe and state == "PROOF_COMPLETE",
                }
                if resumable:
                    evidence = proof_post_action_evidence
                    post = evidence["observation"]
                    tracker = post["opencv_tracker"]
                    snapshot = evidence["lidar_wait"]["snapshot"]
                    action_result = history[-1]["result"]
                    transport = ((action_result.get("lateral_step") or {}).get("lateral_result") or
                        (action_result.get("approach_result") or {}).get("forward_result") or {})
                    # Retain side memory, not a target or permission from an
                    # old corridor. The existing selector rebuilds both sides.
                    for key in list(planning_avoidance):
                        if key.startswith("bypass_") or key in {
                                "local_bypass_active", "local_bypass_target_x_m", "local_bypass_target_y_m"}:
                            planning_avoidance[key] = False if key.endswith(("active", "permitted")) else None
                    retained_avoidance_history = []
                    for row in avoidance_history:
                        item = {key: copy.deepcopy(row[key]) for key in (
                            "source_frame_stamp_ns", "previous_clearances", "previous_action_type",
                            "last_detour_improved_direct_path", "dispatched", "physical_dispatch_confirmed",
                            "motion_executed", "action_lidar_evidence", "action_type",
                            "actual_route_occupancy", "actual_max_overlap_m",
                            "actual_blocker_centerline_clearance_m", "meaningful_progress",
                            "meaningful_progress_reason", "post_action_route", "actual_route_progress",
                            "progress_improved",
                        ) if key in row}
                        item["selection"] = _marvin_proof_selection_history(row.get("selection"))
                        retained_avoidance_history.append(item)
                    action_history = list(proof_continuation.action_history) if proof_continuation else []
                    action_history.append({"state": history[-1]["state"],
                        "source_frame_stamp_ns": previous_stamp,
                        "action_type": action_result.get("action_type"),
                        "action_lidar_evidence": self._marvin_last_action_lidar_evidence})
                    checkpoint = _MarvinLiveProofContinuation(
                        generation=control_generation,
                        producer_session=snapshot["producer_session"],
                        avoidance=planning_avoidance,
                        previous_selection=_marvin_proof_selection_history(previous_selection),
                        previous_clearances=copy.deepcopy(previous_clearances),
                        avoidance_history=retained_avoidance_history,
                        avoidance_lidar_refresh_history=[{key: copy.deepcopy(value)
                            for key, value in row.items() if not isinstance(value, (dict, list))}
                            for row in avoidance_lidar_refresh_history],
                        action_history=copy.deepcopy(action_history), last_action_state=history[-1]["state"],
                        previous_stamp=previous_stamp, camera_floor_stamp=post["source_frame_stamp_ns"],
                        action_finished_monotonic_seconds=action_finished_monotonic_seconds,
                        action_lidar_evidence=self._marvin_last_action_lidar_evidence,
                        post_lidar_sequence=snapshot["acquisition_sequence"],
                        identity_source_frame_stamp_ns=post["identity_source_frame_stamp_ns"],
                        acquired=acquired, search_turns=search_turns,
                        reacquisition_attempts=reacquisition_attempts,
                        consecutive_reacquisition_failures=consecutive_reacquisition_failures,
                        range_association=copy.deepcopy(vars(self._marvin_target_range_association)),
                        completion={"state": state, "stop_confirmed": safe,
                            "full_step_completed": action_result.get("full_step_completed",
                                action_result.get("ok") is True) is True,
                            "actions_executed": result["actions_executed"],
                            "dispatch_opportunities": mission.dispatch_opportunities,
                            "physical_dispatches_or_uncertain": proof_dispatches,
                            "delivery_uncertain": bool(action_result.get("delivery_uncertain")
                                or transport.get("delivery_uncertain")),
                            "lidar_valid": snapshot.get("valid"), "lidar_reason": snapshot.get("reason"),
                            "tracker_matched": tracker.get("active") is True and tracker.get("matched") is True,
                            "identity_confirmed": post.get("identity_confirmed"),
                            "tracker_quality": tracker.get("quality"), "tracker_threshold": tracker.get("threshold"),
                            "camera_received_monotonic_seconds": post.get("received_monotonic_seconds")})
                    with self._state_lock:
                        if current():
                            self._marvin_live_proof_continuation = checkpoint
                            self._marvin_live_proof_checkpoint_digest = _marvin_proof_checkpoint_digest(checkpoint)
                    result["proof"]["cumulative_actions_completed"] = len(action_history)
            return result

        while current():
            bridge = self._marvin_bridge_ready_and_stopped()
            if bridge.get("ok") is not True or bridge.get("status") != "READY":
                return finish("BLOCKED", "find_marvin_bridge_not_ready_or_stopped")
            prior = self._marvin_last_action_lidar_evidence
            if prior is not None:
                wait = self._wait_for_new_marvin_lidar_evidence(
                    expected_session=prior[0], previous_sequence=prior[1], execution_guard=current,
                )
                lidar_wait_history.append(wait)
                if isinstance(wait.get("snapshot"), dict):
                    retain("record_lidar", wait["snapshot"])
                if wait["ok"] is not True:
                    return finish("STOPPED" if not current() else "BLOCKED", wait["reason"])
            # No camera/semantic decision is made before the stopped LiDAR wait.
            observation = self.observe_find_marvin_v2()
            retain("observe", observation)
            if not current():
                return finish("STOPPED", "find_marvin_mission_preempted")
            tracker = observation.get("opencv_tracker") or {}
            stamp = observation.get("source_frame_stamp_ns")
            if proof_continuation is not None and (
                    observation.get("identity_source") == "marvin_locked_tracker_continuity"
                    and observation.get("identity_source_frame_stamp_ns")
                        != expected_identity_stamp):
                return finish("REVERIFY_REQUIRED", "marvin_live_proof_identity_discontinuity")
            strict_episode = observation.get("strict_tracker_episode")
            if (proof_continuation is not None and not history
                    and observation.get("identity_confirmed") is True
                    and isinstance(strict_episode, dict)
                    and strict_episode.get("continued_existing_tracker") is not True):
                return finish("REVERIFY_REQUIRED", "marvin_live_proof_identity_discontinuity")
            if (acquired and observation.get("identity_confirmed") is not True
                    and observation.get("perception_reason") == "post_action_tracker_continuity_lost"):
                # The mission owns recovery. No search or action can occur
                # until new Gemini identity AND a post-Gemini action frame pass
                # the normal freshness, LiDAR and motion admission gates below.
                clear_bypass("marvin_identity_reacquisition")
                if previous_selection:
                    previous_selection = dict(previous_selection, local_bypass=None)
                    # Expire the local target through semantic recovery. Fresh
                    # geometry may re-establish it; retain side and action history.
                loss = observation
                tracker_loss_history.append(loss)
                if len(tracker_loss_history) > self.MAX_MARVIN_REACQUISITION_EPISODES:
                    return finish("REVERIFY_REQUIRED", "find_marvin_recovery_backstop_exhausted")
                self._reset_marvin_alignment_consensus()
                if callable(clear_episode):
                    clear_episode()
                floor = max(value for value in (previous_stamp, stamp)
                            if type(value) is int)
                recovered = False
                while consecutive_reacquisition_failures < self.MAX_CONSECUTIVE_MARVIN_REACQUISITION_FAILURES:
                    if not current():
                        return finish("STOPPED", "find_marvin_mission_preempted")
                    try:
                        stop = self.robot_client.stop()
                    except Exception:
                        return finish("BLOCKED", "find_marvin_reacquisition_stop_failed")
                    bridge = self._marvin_bridge_ready_and_stopped()
                    if (not isinstance(stop, dict) or stop.get("ok") is not True
                            or bridge.get("ok") is not True or bridge.get("status") != "READY"):
                        return finish("BLOCKED", "find_marvin_reacquisition_stop_failed")
                    reacquisition_attempts += 1  # Cumulative telemetry, not the failure budget.
                    self._publish_behavior_tracking({
                        "behavior": "FIND_OBJECT", "target": "marvin", "state": "REACQUIRE",
                        "reacquisition_attempt": reacquisition_attempts,
                        "post_action_tracker_diagnostics": loss.get("post_action_tracker_diagnostics"),
                    })
                    if not current():
                        return finish("STOPPED", "find_marvin_mission_preempted")
                    observation = self._observe_find_marvin_v2(reacquisition_source_floor=floor)
                    retain("observe", observation, reacquisition=True)
                    reacquisition_history.append({
                        "state": "REACQUIRE", "attempt": reacquisition_attempts,
                        "loss_observation": loss, "source_floor": floor,
                        "observation": observation, "motion_executed": False,
                    })
                    if not current():
                        return finish("STOPPED", "find_marvin_mission_preempted")
                    identity_stamp = observation.get("identity_source_frame_stamp_ns")
                    action_stamp = observation.get("source_frame_stamp_ns")
                    if (observation.get("ok") is True
                            and observation.get("identity_confirmed") is True
                            and observation.get("identity_source") == "gemini_marvin_candidate_selection"
                            and type(identity_stamp) is int and identity_stamp > floor
                            and type(action_stamp) is int and action_stamp > identity_stamp):
                        recovered = True
                        consecutive_reacquisition_failures = 0
                        reacquisition_history[-1].update(succeeded=True, consecutive_failures_after=0)
                        break
                    consecutive_reacquisition_failures += 1
                    reacquisition_history[-1].update(
                        succeeded=False, consecutive_failures_after=consecutive_reacquisition_failures)
                    # Neither an old identity nor a failed action observation
                    # may survive into the next attempt.
                    self._reset_marvin_alignment_consensus()
                    if callable(clear_episode):
                        clear_episode()
                    floor = max([floor] + [value for value in (identity_stamp, action_stamp)
                                           if type(value) is int])
                if not recovered:
                    return finish("REVERIFY_REQUIRED", "find_marvin_semantic_reacquisition_exhausted")
                tracker = observation.get("opencv_tracker") or {}
                stamp = observation.get("source_frame_stamp_ns")
            if acquired and observation.get("identity_confirmed") is not True:
                return finish("REVERIFY_REQUIRED", observation.get("perception_reason")
                              or observation.get("reason") or "marvin_identity_lost")
            if observation.get("perception_reason") in {
                "find_marvin_post_semantic_tracker_refresh_failed",
                "find_marvin_search_fresh_frame_unavailable",
            }:
                return finish("REVERIFY_REQUIRED", observation["perception_reason"])
            receipt = observation.get("received_monotonic_seconds")
            if not self._marvin_motion_stamp_is_fresh(stamp, receipt):
                return finish("BLOCKED", "marvin_motion_observation_stale")
            if (previous_stamp is not None and stamp <= previous_stamp
                    or camera_floor_stamp is not None and stamp <= camera_floor_stamp
                    or action_finished_monotonic_seconds is not None
                    and receipt <= action_finished_monotonic_seconds):
                return finish("REVERIFY_REQUIRED", "find_marvin_new_camera_frame_required")
            session, lidar = self._active_localization_lidar_is_current()
            if lidar is None:
                return finish("BLOCKED", "find_marvin_lidar_not_current")
            prior = getattr(self, "_marvin_last_action_lidar_evidence", None)
            sequence = lidar.get("acquisition_sequence")
            if prior is not None and session != prior[0]:
                return finish("BLOCKED", "find_marvin_lidar_producer_session_changed")
            if type(sequence) is not int or sequence < 0 or (prior is not None and sequence <= prior[1]):
                return finish("BLOCKED", "find_marvin_lidar_acquisition_sequence_invalid")
            if (proof_continuation is not None
                    and sequence < proof_continuation.post_lidar_sequence):
                return finish("BLOCKED", "marvin_live_proof_lidar_discontinuity")
            if observation.get("identity_confirmed") is True:
                expected_identity_stamp = observation.get("identity_source_frame_stamp_ns")
            decision = (observation.get("controller") or {}).get("decision")
            observed_route = (observation.get("arrival") or {}).get("route") or {}
            if observed_route.get("valid") and not observed_route.get("route_to_marvin_obstructed"):
                clear_bypass("direct_path_restored")
            if previous_selection and observation.get("identity_confirmed") is True:
                new_association = observation.get("arrival") or {}
                new_route = new_association.get("route") or {}
                record_avoidance_reassessment(new_route, new_association)
            if proof_max_physical_actions is not None and proof_dispatches >= 1:
                # The ordinary stopped LiDAR wait, strict tracker/reacquisition,
                # fresh frame, exact stamp and session checks above run first.
                # Return before evaluating or dispatching action number two.
                proof_post_action_evidence = {"lidar_wait": lidar_wait_history[-1],
                    "observation": observation, "current_lidar": lidar,
                    "action_lidar_evidence": history[-1]["action_lidar_evidence"],
                    "action_source_frame_stamp_ns": previous_stamp}
                if acquired and (observation.get("ok") is not True
                        or tracker.get("active") is not True or tracker.get("matched") is not True
                        or type(tracker.get("quality")) not in (int, float)
                        or type(tracker.get("threshold")) not in (int, float)
                        or not math.isfinite(tracker["quality"])
                        or not math.isfinite(tracker["threshold"])
                        or tracker["quality"] < max(.80, tracker["threshold"])
                        or tracker.get("source_frame_stamp_ns") != stamp):
                    return finish("REVERIFY_REQUIRED", "find_marvin_live_proof_post_action_tracker_invalid")
                complete = history[-1]["result"].get("full_step_completed",
                    history[-1]["result"].get("ok") is True) is True
                action_result = history[-1]["result"]
                transport = ((action_result.get("lateral_step") or {}).get("lateral_result") or
                    (action_result.get("approach_result") or {}).get("forward_result") or {})
                complete = (complete and action_result.get("interrupted") is not True
                    and action_result.get("delivery_uncertain") is not True
                    and transport.get("delivery_uncertain") is not True)
                return finish("PROOF_COMPLETE" if complete else "PROOF_INTERRUPTED",
                    "find_marvin_live_proof_action_complete" if complete else
                    "find_marvin_live_proof_action_interrupted")
            if observation.get("identity_confirmed") is not True:
                # Only a fresh, explicit negative perception result permits
                # initial search. Camera/semantic errors are not "no target".
                absent = observation.get("perception_reason") in {
                    "Marvin was not found in the current camera frame.",
                    "marvin_identity_not_confirmed",
                }
                if acquired or not absent:
                    return finish("REVERIFY_REQUIRED", observation.get("perception_reason")
                                  or observation.get("reason") or "marvin_identity_lost")
                if search_turns >= MAX_SCAN_TURNS:
                    return finish("SEARCH_EXHAUSTED", "find_marvin_search_exhausted")
                state = "SEARCHING"
                action = lambda: self._execute_marvin_v2_search_turn(stamp, receipt)
            else:
                acquired = True
                if observation.get("ok") is not True:
                    return finish("REVERIFY_REQUIRED", observation.get("reason"))
                if decision == "ARRIVED":
                    arrival = observation.get("arrival") or {}
                    if (arrival.get("ok") is True
                            and arrival.get("authority") == "target_bearing_lidar"
                            and arrival.get("target_range_association_trusted") is True
                            and arrival.get("arrived_at_marvin") is True
                            and arrival.get("target_distance_m", float("inf")) <= TARGET_STANDOFF_M):
                        return finish("ARRIVED", "arrived_at_marvin")
                    return finish("BLOCKED", "find_marvin_arrival_evidence_invalid")
                if decision in {"TURN_LEFT", "TURN_RIGHT"}:
                    state = "ALIGNING"
                    action = lambda: self.execute_single_marvin_alignment(
                        direction="LEFT" if decision == "TURN_LEFT" else "RIGHT",
                        angular_speed=0.25, duration=0.50, source_frame_stamp_ns=stamp)
                elif decision in {"FORWARD", "AVOID"}:
                    state = "ADVANCING"
                    action = lambda: self.execute_single_marvin_approach(
                        linear_speed=FIND_MARVIN_FORWARD_SPEED_MPS,
                        duration=0.50, source_frame_stamp_ns=stamp)
                    # Use the same producer-bound World Model snapshot for the
                    # direct-path check and both advisory escape candidates.
                    standoff = self._marvin_v2_lidar_arrival(tracker, lidar)
                    if standoff.get("ok") is not True:
                        return finish("BLOCKED", standoff.get("reason"))
                    if not standoff.get("arrived_at_marvin"):
                        forward_duration = (min(0.50, (standoff["target_distance_m"] -
                            TARGET_STANDOFF_M) / FIND_MARVIN_FORWARD_SPEED_MPS)
                            if standoff.get("target_range_association_trusted") is True else 0.50)
                        direct = evaluate_local_motion_safety(
                            lidar, expected_session=session,
                            linear_x=FIND_MARVIN_FORWARD_SPEED_MPS, duration=forward_duration)
                        route = standoff.get("route") or {}
                        if direct.get("reason") not in {"protected_region_clear", "translation_protected_region_violated"}:
                            return finish("BLOCKED", direct.get("reason"))
                        route_blocked = route.get("valid") and route.get("route_to_marvin_obstructed")
                        if direct.get("permitted") and not route_blocked:
                            if decision == "AVOID" or standoff.get("target_range_association_trusted") is not True:
                                return finish("BLOCKED", standoff["target_range_association_reason"])
                            if avoidance["local_avoidance_active"]:
                                avoidance.update(local_avoidance_active=False, direct_path_blocked=False,
                                    route_to_marvin_obstructed=False, avoidance_reason="direct_path_restored",
                                    last_detour_improved_direct_path=True, progress_improved=True)
                                previous_clearances = None
                                previous_selection = None  # End episode, retain mission budget.
                        else:
                            stopped = self.robot_client.stop()
                            zero = self._marvin_bridge_ready_and_stopped()
                            if not current():
                                return finish("STOPPED", "find_marvin_mission_preempted")
                            if (not isinstance(stopped, dict) or stopped.get("ok") is not True
                                    or zero.get("ok") is not True or zero.get("status") != "READY"):
                                return finish("BLOCKED", "find_marvin_local_avoidance_stop_failed")
                            # The blocked scan justified STOP, never the action
                            # after the STOP/status round trips. Refresh first.
                            refreshed = self._refresh_marvin_avoidance_plan(
                                tracker=tracker, expected_session=session,
                                blocked_sequence=sequence, execution_guard=current)
                            avoidance_lidar_refresh_history.append(refreshed)
                            avoidance.update({key: refreshed[key] for key in (
                                "blocked_forward_lidar_sequence", "avoidance_planning_lidar_sequence",
                                "avoidance_lidar_refresh_required", "avoidance_lidar_refresh_wait_seconds",
                                "avoidance_lidar_refresh_result")})
                            if not current():
                                return finish("STOPPED", "find_marvin_mission_preempted")
                            if refreshed["ok"] is not True:
                                history.append({"state": "ADVANCING", "decision_only": True,
                                    "source_frame_stamp_ns": stamp, "observation": observation,
                                    "motion_executed": False, "action_lidar_evidence": None,
                                    "result": {"ok": False, "motion_executed": False,
                                        "execution_authorized": False, "actions_executed": 0,
                                        "source_stamp_consumed": False, "full_step_completed": False,
                                        "reason": refreshed["reason"],
                                        "approach_result": {"forward_safety": direct}}})
                                return finish("BLOCKED", refreshed["reason"])
                            lidar, standoff = refreshed["lidar"], refreshed["association"]
                            decision = refreshed["decision"]
                            route = standoff["route"]
                            observation = dict(observation, arrival=standoff,
                                reason=standoff["reason"],
                                controller=dict(observation["controller"], decision=decision,
                                    state="ARRIVED" if decision == "ARRIVED" else VISUAL_READY_TO_APPROACH,
                                    reason=standoff["reason"],
                                    distance_state="ARRIVED" if decision == "ARRIVED" else
                                        "APPROACH" if decision == "FORWARD" else "UNKNOWN",
                                    path_state="DIRECT_PATH_BLOCKED" if decision == "AVOID" else "DIRECT_PATH_CLEAR"),
                                **{key: standoff.get(key) for key in (
                                    "nearest_forward_obstacle_distance_m", "candidate_target_return_distance_m",
                                    "verified_marvin_distance_m", "target_range_association_trusted",
                                    "target_range_association_reason", "direct_path_blocked",
                                    "route_to_marvin_obstructed", "blocking_obstacle_distance_m",
                                    "blocking_obstacle_bearing_deg", "blocking_obstacle_x_m", "blocking_obstacle_y_m")})
                            retain("observe", observation)
                            if decision == "ARRIVED":
                                avoidance.update(local_avoidance_active=False, direct_path_blocked=False,
                                    route_to_marvin_obstructed=False, avoidance_reason="arrived_at_marvin")
                                return finish("ARRIVED", "arrived_at_marvin")
                            # Rebind the existing, unconsumed visual reference
                            # to the new route decision without new Gemini work.
                            # Strict identity and stamp gates remain authoritative.
                            action_observation = self._marvin_v2_action_observation(
                                observation, tracker, VISUAL_READY_TO_APPROACH, decision)
                            with self._state_lock:
                                self._marvin_alignment_observation = action_observation
                            direct = refreshed["direct"]
                            forward_duration = refreshed["forward_duration"]
                            route_blocked = route["route_to_marvin_obstructed"]
                            if decision == "FORWARD":
                                avoidance.update(local_avoidance_active=False, direct_path_blocked=False,
                                    route_to_marvin_obstructed=False, avoidance_reason="direct_path_restored",
                                    selected_action_type=None, left_clearance_m=None, right_clearance_m=None,
                                    last_detour_improved_direct_path=True, progress_improved=True)
                                previous_clearances = None
                                previous_selection = None
                            else:
                                previous_direction = (avoidance["last_detour_direction"]
                                    if avoidance["local_avoidance_active"] else None)
                                avoidance.update(local_avoidance_active=True,
                                    direct_path_blocked=direct.get("permitted") is not True,
                                    route_to_marvin_obstructed=bool(route_blocked),
                                    last_detour_improved_direct_path=route_progress(
                                        (previous_selection or {}).get("route"), route) if previous_selection else None)
                                if avoidance["local_avoidance_actions"] >= self.MAX_LOCAL_AVOIDANCE_ACTIONS:
                                    avoidance["avoidance_reason"] = "find_marvin_local_avoidance_exhausted"
                                    return finish("BLOCKED", avoidance["avoidance_reason"])
                                allow_strafe = (zero.get("motion_capabilities") or {}).get("linear_y") is True
                                # Legacy Bridge clients cannot strafe. Preserve their
                                # existing guarded-turn policy; the new route case
                                # can still evaluate four options with strafe denied.
                                four_primitives = allow_strafe or direct.get("permitted") is True
                                if four_primitives:
                                    detour = select_marvin_escape_action(lidar, standoff, expected_session=session,
                                        allow_strafe=allow_strafe, previous_selection=previous_selection)
                                else:
                                    detour = select_marvin_detour(lidar, expected_session=session,
                                        forward_speed=FIND_MARVIN_FORWARD_SPEED_MPS,
                                        forward_duration=forward_duration, previous_direction=previous_direction,
                                        previous_clearances=previous_clearances)
                                    detour["action_type"] = "TURN_" + detour["direction"] if detour["direction"] else None
                                    detour["route"] = route
                                bypass = detour.get("local_bypass") or {}
                                avoidance.update({key: bypass.get(key) for key in (
                                    "local_bypass_side", "local_bypass_target_x_m",
                                    "local_bypass_target_y_m", "local_bypass_reason", "route_to_bypass_obstructed",
                                    "bypass_corridor_occupancy", "bypass_corridor_overlap_m", "bypass_forward_permitted",
                    "bypass_target_x_m", "bypass_target_y_m", "bypass_distance_m", "bypass_bearing_deg")})
                                avoidance["local_bypass_active"] = detour.get("action_type") == "BYPASS_FORWARD"
                                avoidance.update(left_clearance_m=detour["left_clearance_m"],
                                    right_clearance_m=detour["right_clearance_m"], avoidance_reason=detour["reason"],
                                    selected_action_type=detour["action_type"],
                                    progress_improved=detour.get("progress_improved"))
                                if previous_selection:
                                    record_avoidance_reassessment(route, standoff)
                                avoidance_history.append({"source_frame_stamp_ns": stamp,
                                    "selection": detour, "previous_clearances": previous_clearances,
                                    "previous_action_type": (previous_selection or {}).get("action_type"),
                                    "last_detour_improved_direct_path": avoidance["last_detour_improved_direct_path"]})
                                if detour["direction"] is None:
                                    history.append({"state": "ADVANCING", "decision_only": True,
                                        "source_frame_stamp_ns": stamp, "observation": observation,
                                        "motion_executed": False, "action_lidar_evidence": None,
                                        "result": {"ok": False, "motion_executed": False,
                                            "execution_authorized": False, "actions_executed": 0,
                                            "source_stamp_consumed": False, "full_step_completed": False,
                                            "reason": detour["reason"], "approach_result": {"forward_safety": direct}}})
                                    return finish("BLOCKED", detour["reason"])
                                context = {"previous_direction": previous_direction,
                                    "previous_clearances": previous_clearances,
                                    "four_primitives": four_primitives, "previous_selection": previous_selection,
                                    "selected_action_type": detour["action_type"], "allow_strafe": allow_strafe,
                                    "selected_direction": detour["direction"]}
                                state = "AVOIDING"
                                if detour["action_type"] == "BYPASS_FORWARD":
                                    action = lambda: self._dispatch_marvin_observation_action(
                                        self._execute_single_marvin_approach,
                                        linear_speed=FIND_MARVIN_FORWARD_SPEED_MPS, duration=.50,
                                        source_frame_stamp_ns=stamp, local_detour_context=context)
                                elif detour["action_type"].startswith("STRAFE"):
                                    action = lambda: self._dispatch_marvin_observation_action(
                                        self._execute_single_marvin_strafe, source_frame_stamp_ns=stamp,
                                        local_detour_context=context)
                                else:
                                    action = lambda: self._dispatch_marvin_observation_action(
                                        self._execute_single_marvin_alignment, direction=detour["direction"],
                                        angular_speed=TURN_SPEED, duration=TURN_DURATION,
                                        source_frame_stamp_ns=stamp, local_detour_context=context)
                else:
                    return finish("BLOCKED", observation.get("reason") or "find_marvin_unexpected_controller_state")
            self._publish_behavior_tracking({
                "behavior": "FIND_OBJECT", "target": "marvin", "state": state,
                "opencv_tracker": tracker, "source_frame_stamp_ns": stamp,
                **avoidance,
                **{key: (observation.get("arrival") or {}).get(key) for key in (
                    "nearest_forward_obstacle_distance_m", "candidate_target_return_distance_m",
                    "verified_marvin_distance_m", "target_range_association_trusted",
                    "target_range_association_reason", "route_to_marvin_obstructed",
                    "blocking_obstacle_distance_m", "blocking_obstacle_bearing_deg",
                    "blocking_obstacle_x_m", "blocking_obstacle_y_m")},
            })
            if not current():
                return finish("STOPPED", "find_marvin_mission_preempted")
            try:
                if state == "AVOIDING":
                    observation = dict(observation, local_avoidance_action=detour["action_type"],
                        local_avoidance_selection=detour)
                retain("prepare_action", state, observation)
                result = action()
            except Exception as exc:
                if proof_max_physical_actions is not None:
                    consumed = stamp in self._marvin_alignment_consumed_source_frame_stamps
                    proof_dispatches += int(consumed)  # Delivery cannot be ruled out.
                    history.append({"state": state, "source_frame_stamp_ns": stamp,
                        "observation": observation, "action_lidar_evidence": self._marvin_last_action_lidar_evidence,
                        "motion_executed": False, "result": {"ok": False,
                            "source_stamp_consumed": consumed, "delivery_uncertain": consumed,
                            "full_step_completed": False, "error": str(exc)}})
                return finish("BLOCKED", "find_marvin_action_exception: " + str(exc))
            if not isinstance(result, dict):
                if proof_max_physical_actions is not None:
                    consumed = stamp in self._marvin_alignment_consumed_source_frame_stamps
                    proof_dispatches += int(consumed)
                    history.append({"state": state, "source_frame_stamp_ns": stamp,
                        "observation": observation, "action_lidar_evidence": self._marvin_last_action_lidar_evidence,
                        "motion_executed": False, "result": {"ok": False,
                            "source_stamp_consumed": consumed, "delivery_uncertain": consumed,
                            "full_step_completed": False}})
                return finish("BLOCKED", "find_marvin_action_result_malformed")
            history.append({"state": state, "source_frame_stamp_ns": stamp,
                            "observation": observation, "result": result,
                            "action_lidar_evidence": self._marvin_last_action_lidar_evidence,
                            "motion_executed": result.get("motion_executed") is True})
            if proof_max_physical_actions is not None:
                forward = ((result.get("lateral_step") or {}).get("lateral_result") or
                           (result.get("approach_result") or {}).get("forward_result") or {})
                proof_dispatches += int(result.get("motion_executed") is True
                    or (result.get("turn_result") or {}).get("confirmed_forwarded") is True
                    or forward.get("transport_attempted") is True
                    or forward.get("delivery_uncertain") is True
                    or result.get("reason") == "marvin_alignment_turn_exception"
                    or (result.get("approach_result") or {}).get("reason")
                        == "marvin_single_approach_forward_exception"
                    or (result.get("lateral_step") or {}).get("reason") == "marvin_lateral_transport_failed")
            retain("action_result", result)
            if state == "AVOIDING":
                dispatched = bool(result.get("execution_authorized") is True)
                lateral = ((result.get("lateral_step") or {}).get("lateral_result") or
                    (result.get("approach_result") or {}).get("forward_result") or {})
                physical_dispatch = (result.get("motion_executed") is True
                    or (result.get("turn_result") or {}).get("confirmed_forwarded") is True
                    or (lateral.get("transport_attempted") is True and
                        (lateral.get("transport_result") or {}).get("ok") is True))
                if physical_dispatch:
                    avoidance["local_avoidance_actions"] += 1
                    if detour["action_type"] == "BYPASS_FORWARD":
                        avoidance["local_bypass_actions"] += 1
                    avoidance["last_detour_direction"] = detour["direction"]
                    jit_detour = result["local_detour"]
                    previous_selection = jit_detour
                    avoidance["previous_action_type"] = detour["action_type"]
                    previous_clearances = {"LEFT": jit_detour["left_clearance_m"],
                                           "RIGHT": jit_detour["right_clearance_m"]}
                    avoidance.update(left_clearance_m=jit_detour["left_clearance_m"],
                                     right_clearance_m=jit_detour["right_clearance_m"])
                avoidance_history[-1].update(dispatched=dispatched,
                    physical_dispatch_confirmed=physical_dispatch,
                    motion_executed=result.get("motion_executed") is True,
                    action_lidar_evidence=self._marvin_last_action_lidar_evidence)
                prediction = ((result.get("local_detour") or detour).get("options") or {}).get(
                    detour["action_type"], {})
                avoidance_history[-1].update(action_type=detour["action_type"],
                    predicted_route_occupancy=prediction.get("predicted_route_occupancy"),
                    predicted_max_overlap_m=prediction.get("predicted_max_overlap_m"),
                    predicted_blocker_centerline_clearance_m=prediction.get("predicted_blocker_centerline_clearance_m"),
                    actual_route_occupancy=None, actual_max_overlap_m=None,
                    actual_blocker_centerline_clearance_m=None,
                    meaningful_progress=None, meaningful_progress_reason=None)
            if not current():
                return finish("STOPPED", "find_marvin_mission_preempted")
            if result.get("ok") is not True or result.get("motion_executed") is not True:
                forward = ((result.get("lateral_step") or {}).get("lateral_result") or
                    (result.get("approach_result") or {}).get("forward_result") or {})
                recoverable = (state in {"ADVANCING", "AVOIDING"} and result.get("interrupted") is True
                    and result.get("source_stamp_consumed") is True
                    and (forward.get("bounded_forward_invalidated") is True or
                         forward.get("bounded_lateral_invalidated") is True)
                    and forward.get("reason") in {"stale", "stale_lidar"}
                    and (forward.get("transport_result") or {}).get("ok") is True)
                if not recoverable:
                    return finish("BLOCKED", result.get("reason") or "find_marvin_guarded_action_failed")
                bridge = self._marvin_bridge_ready_and_stopped()
                if (forward.get("bounded_lateral_invalidated") is True
                        and result.get("bridge_stop_confirmed") is not True):
                    return finish("BLOCKED", "find_marvin_lidar_recovery_stop_unconfirmed")
                if (forward.get("interlock_stop_succeeded") is not True
                        or (result.get("stop_result") or {}).get("ok") is not True
                        or bridge.get("ok") is not True or bridge.get("status") != "READY"):
                    return finish("BLOCKED", "find_marvin_lidar_recovery_stop_unconfirmed")
                if not current():
                    return finish("STOPPED", "find_marvin_mission_preempted")
                consecutive_lidar_interruptions += 1
                recovery = {"source_frame_stamp_ns": stamp, "motion_executed": False,
                            "consecutive_interruptions": consecutive_lidar_interruptions}
                lidar_recovery_history.append(recovery)
                if (consecutive_lidar_interruptions >= self.MAX_CONSECUTIVE_MARVIN_LIDAR_INTERRUPTION_FAILURES
                        or len(lidar_recovery_history) > self.MAX_MARVIN_REACQUISITION_EPISODES):
                    return finish("BLOCKED", "find_marvin_lidar_recovery_exhausted")
                action_finished_monotonic_seconds = time.monotonic()
                previous_stamp = stamp
                self._reset_marvin_alignment_consensus()
                retain("action_result", result, action_finished_monotonic_seconds)
                mark_stopped = getattr(behavior, "mark_strict_v2_action_stopped", None)
                if callable(mark_stopped):
                    mark_stopped(stamp, action_finished_monotonic_seconds)
                baseline = self._marvin_last_action_lidar_evidence
                invalidating = (forward.get("interlock_dispatch_outcome") or {}).get("invalidating_lidar_evidence") or {}
                expired_age = invalidating.get("effective_age_seconds")
                if (baseline is None or baseline[0] != session or type(baseline[1]) is not int
                        or invalidating.get("producer_session") != session
                        or type(invalidating.get("acquisition_sequence")) is not int
                        or invalidating["acquisition_sequence"] < baseline[1]
                        or type(expired_age) not in (int, float) or not math.isfinite(expired_age)
                        or expired_age <= MAXIMUM_EFFECTIVE_AGE_SECONDS):
                    return finish("BLOCKED", "find_marvin_lidar_recovery_evidence_invalid")
                # Neither the action's safety scan nor the scan that expired
                # can release recovery. Only a genuinely newer fresh scan can.
                self._marvin_last_action_lidar_evidence = (
                    session, max(baseline[1], invalidating["acquisition_sequence"]))
                self._publish_behavior_tracking({"behavior": "FIND_OBJECT", "target": "marvin",
                    "state": "WAITING_FOR_FRESH_LIDAR", "source_frame_stamp_ns": stamp})
                wait = self._wait_for_new_marvin_lidar_evidence(
                    expected_session=session, previous_sequence=self._marvin_last_action_lidar_evidence[1],
                    execution_guard=current, allow_transient_stale=True)
                recovery["wait"] = wait
                if isinstance(wait.get("snapshot"), dict):
                    retain("record_lidar", wait["snapshot"])
                if wait["ok"] is not True:
                    return finish("STOPPED" if not current() else "BLOCKED", wait["reason"])
                # Do not replay/resume the command. The next cycle observes a
                # new camera frame, checks tracker/identity and replans with JIT.
                continue
            consecutive_lidar_interruptions = 0
            bridge = self._marvin_bridge_ready_and_stopped()
            if bridge.get("ok") is not True or bridge.get("status") != "READY":
                return finish("BLOCKED", "find_marvin_bridge_not_stopped_after_action")
            action_finished_monotonic_seconds = time.monotonic()
            retain("action_result", result, action_finished_monotonic_seconds)
            previous_stamp = stamp
            mark_stopped = getattr(behavior, "mark_strict_v2_action_stopped", None)
            if callable(mark_stopped):
                mark_stopped(stamp, action_finished_monotonic_seconds)
            # Retain the actual JIT authorization acquisition. Reading again
            # here would discard a usable N+1 and unnecessarily require N+2.
            baseline = self._marvin_last_action_lidar_evidence
            if (baseline is None or baseline[0] != session
                    or type(baseline[1]) is not int):
                return finish("BLOCKED", "find_marvin_action_lidar_evidence_invalid")
            if state == "SEARCHING":
                search_turns += 1
        return finish("STOPPED", "find_marvin_mission_preempted")

    @staticmethod
    def _marvin_search_exhaustion_is_safe(controller):
        """Recognize only a completed, stopped bounded search-plan result."""
        history = controller.get("history")
        if (
            controller.get("completed") is not False
            or controller.get("arrived_at_marvin") is not False
            or not isinstance(history, list)
            or not history
        ):
            return False
        terminal = history[-1]
        search_step = (
            terminal.get("search_step_result")
            if isinstance(terminal, dict) else None
        )
        planner = (
            search_step.get("planner")
            if isinstance(search_step, dict) else None
        )
        if not (
            terminal.get("route") == "search"
            and terminal.get("selected_action") == "search_complete"
            and isinstance(terminal.get("stop_result"), dict)
            and terminal["stop_result"].get("ok") is True
            and isinstance(search_step, dict)
            and search_step.get("ok") is True
            and search_step.get("decision") == "search_complete"
            and search_step.get("motion_executed") is False
            and isinstance(planner, dict)
            and planner.get("completed") is True
            and planner.get("selected_search_action") == "search_complete"
        ):
            return False
        for entry in history:
            if not isinstance(entry, dict):
                return False
            if entry.get("action_budget_consumed") is True:
                stop_result = entry.get("stop_result")
                if not (
                    isinstance(stop_result, dict)
                    and stop_result.get("ok") is True
                ):
                    return False
        return True

    @staticmethod
    def _marvin_clearance_timeout_is_safe(controller):
        """Accept only a stopped no-motion room-scan clearance timeout."""
        history = controller.get("history")
        if (
            controller.get("completed") is not True
            or controller.get("arrived_at_marvin") is not False
            or not isinstance(history, list)
            or not history
        ):
            return False
        terminal = history[-1]
        step = terminal.get("search_step_result") if isinstance(terminal, dict) else None
        wait = controller.get("clearance_wait")
        bridge = wait.get("bridge_status") if isinstance(wait, dict) else None
        motion = bridge.get("motion") if isinstance(bridge, dict) else None
        origin = wait.get("clearance_wait_origin") if isinstance(wait, dict) else None
        step_origin = step.get("clearance_wait_origin") if isinstance(step, dict) else None
        interrupted = step.get("interrupted_turn_detected") is True if isinstance(step, dict) else False
        action_budget_consumed = (
            terminal.get("action_budget_consumed") if isinstance(terminal, dict) else None
        )
        accounting_valid = (
            origin == step_origin == "pre_turn" and action_budget_consumed is False
        ) or (
            origin == step_origin == "active_turn_monitor"
            and interrupted
            and step.get("interrupted_turn_monitor_reason")
            == "rotational_protected_region_violated"
            and action_budget_consumed is True
        )
        return bool(
            isinstance(terminal, dict)
            and terminal.get("route") == "search"
            and accounting_valid
            and isinstance(terminal.get("stop_result"), dict)
            and terminal["stop_result"].get("ok") is True
            and isinstance(step, dict)
            and step.get("clearance_wait_timed_out") is True
            and step.get("motion_executed") is False
            and step.get("reason") == "find_marvin_clearance_wait_timeout"
            and isinstance(wait, dict)
            and terminal.get("pending_scan_turn_index") == wait.get("pending_scan_turn_index")
            and wait.get("decision") == "clearance_wait_timeout"
            and wait.get("clearance_wait_origin") in {
                "pre_turn", "active_turn_monitor",
            }
            and isinstance(bridge, dict)
            and bridge.get("ok") is True
            and bridge.get("ros_ready") is True
            and isinstance(motion, dict)
            and motion.get("linear_x") == 0
            and motion.get("linear_y", 0.0) == 0
            and motion.get("angular_z") == 0
            and motion.get("streaming") is False
        )

    def _marvin_mission_context_is_current(self, mission, control_generation):
        with self._state_lock:
            if isinstance(mission, _MarvinLiveProofOwner):
                return (control_generation == mission.generation
                        and self._marvin_live_proof_owner_is_current(mission))
            active = self.mission_manager.get_active_mission()
            return bool(
                self.running is True
                and control_generation == self._control_generation
                and self._is_normal_marvin_find_mission(mission)
                and active is not None
                and active.mission_id == getattr(mission, "mission_id", None)
            )

    @staticmethod
    def _marvin_episode_is_safe_action_limit(episode, controller):
        if (
            episode.get("ok") is not True
            or episode.get("execution_authorized") is not True
            or controller.get("ok") is not True
            or controller.get("completed") is not False
            or controller.get("arrived_at_marvin") is not False
            or controller.get("reason") != "find_marvin_action_limit_reached"
            or controller.get("actions_executed") != CognitiveRuntime.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS
        ):
            return False
        history = controller.get("history")
        if not isinstance(history, list) or not history:
            return False
        counted = 0
        for entry in history:
            if not isinstance(entry, dict):
                return False
            attempted = entry.get("action_budget_consumed") is True
            nested_count = entry.get("action_budget_count")
            if nested_count is not None:
                if (
                    not isinstance(nested_count, int)
                    or isinstance(nested_count, bool)
                    or not 0 <= nested_count <= CognitiveRuntime.MAX_REACTIVE_STEPS
                    or attempted != (nested_count > 0)
                ):
                    return False
                counted += nested_count
                if nested_count > 0:
                    step = entry.get("pursuit_step_result")
                    progress = (
                        step.get("local_progress_result")
                        if isinstance(step, dict) else None
                    )
                    if not isinstance(progress, dict) or progress.get("bridge_stopped") is not True:
                        return False
                    nested = progress.get("nested_result")
                    nested_history = (
                        nested.get("history") if isinstance(nested, dict) else None
                    )
                    if progress.get("mode") == "BOUNDED_AVOIDANCE":
                        if (
                            not isinstance(nested_history, list)
                            or sum(
                                1 for cycle in nested_history
                                if isinstance(cycle, dict)
                                and cycle.get("action_executed") is True
                            ) != nested_count
                            or any(
                                not isinstance(cycle, dict)
                                or cycle.get("bridge_stopped") is not True
                                for cycle in nested_history
                            )
                        ):
                            return False
                candidate_stop = entry.get("selected_action") == "confirm_arrival"
                if nested_count > 0 or candidate_stop:
                    stop = entry.get("stop_result")
                    if not isinstance(stop, dict) or stop.get("ok") is not True:
                        return False
                continue
            candidate_stop = entry.get("selected_action") == "confirm_arrival"
            if attempted:
                counted += 1
            if attempted or candidate_stop:
                stop = entry.get("stop_result")
                if not isinstance(stop, dict) or stop.get("ok") is not True:
                    return False
        return counted == CognitiveRuntime.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS

    def _marvin_bridge_ready_and_stopped(self):
        status_reader = getattr(self.robot_client, "status", None)
        if not callable(status_reader):
            return {"ok": False, "reason": "bridge_status_unavailable"}
        try:
            status = status_reader()
        except Exception as exc:
            return {"ok": False, "reason": "bridge_status_error", "error": str(exc)}
        if not isinstance(status, dict):
            return {"ok": False, "reason": "bridge_status_malformed"}
        motion = status.get("motion")
        if not isinstance(motion, dict):
            return {"ok": False, "reason": "bridge_motion_state_missing"}
        ready = (
            status.get("ok") is True
            and status.get("ros_ready") is True
            and motion.get("linear_x") == 0
            and motion.get("linear_y", 0.0) == 0
            and motion.get("angular_z") == 0
            and motion.get("streaming") is False
        )
        return status if ready else {
            "ok": False,
            "reason": "bridge_not_ready_or_not_stopped",
            "status": status,
        }

    @staticmethod
    def _active_localization_pose_is_trusted(status):
        """Require the same canonical Tony2 authority used by navigation."""
        if not isinstance(status, dict):
            return False
        navigation = status.get("navigation")
        telemetry = status.get("telemetry")
        pose = telemetry.get("pose") if isinstance(telemetry, dict) else None
        position = pose.get("position") if isinstance(pose, dict) else None
        values = (
            position.get("x") if isinstance(position, dict) else None,
            position.get("y") if isinstance(position, dict) else None,
            pose.get("yaw_radians") if isinstance(pose, dict) else None,
            telemetry.get("age_seconds") if isinstance(telemetry, dict) else None,
        )
        return bool(
            status.get("ok") is True
            and status.get("authoritative") is True
            and status.get("read_only") is True
            and status.get("source") == "tony2_navigation_amcl"
            and isinstance(navigation, dict)
            and navigation.get("localization_validated") is True
            and navigation.get("transform_ready") is True
            and isinstance(telemetry, dict)
            and telemetry.get("available") is True
            and isinstance(pose, dict)
            and pose.get("frame_id") == "map"
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in values
            )
            and 0.0 <= float(values[-1]) < 3.0
        )

    @staticmethod
    def _active_localization_is_recoverable(status):
        navigation = status.get("navigation") if isinstance(status, dict) else None
        return bool(
            isinstance(navigation, dict)
            and navigation.get("localization_state")
            == "ACTIVE_LOCALIZATION_REQUIRED"
            and navigation.get("localization_validated") is False
            and navigation.get("goal_submission_enabled") is False
            and navigation.get("goal_active") is False
        )

    def _active_localization_lidar_is_current(self):
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if (
            worker is None
            or worker.running is not True
            or not isinstance(session, str)
            or not session
        ):
            return None, None
        try:
            state = self.world_model.get_lidar_obstacles(
                expected_session=session,
            )
        except Exception:
            return session, None
        return (
            session,
            state if _marvin_alignment_lidar_is_current(state, session) else None,
        )

    def _active_localization_has_behavior_owner(self):
        """Do not overlap an active mission or another behavior execution."""
        with self._state_lock:
            active = self.mission_manager.get_active_mission()
            authorized_mission_id = getattr(
                getattr(self, "_find_object_progress_context", None),
                "mission_id",
                None,
            )
            authorized_find_object = bool(
                authorized_mission_id is not None
                and active is not None
                and getattr(active, "mission_id", None) == authorized_mission_id
                and getattr(active, "mission_type", None) == "FIND_OBJECT"
                and getattr(self, "_behavior_execution_generation", None)
                == getattr(self, "_control_generation", None)
                and getattr(self, "_behavior_execution_thread_id", None)
                == threading.get_ident()
            )
            marvin_context = getattr(
                self, "_find_marvin_progress_context", None,
            )
            marvin_mission_id = getattr(
                marvin_context, "mission_id", None,
            )
            authorized_find_marvin = bool(
                marvin_mission_id is not None
                and active is not None
                and self._is_normal_marvin_find_mission(active)
                and getattr(active, "mission_id", None) == marvin_mission_id
                and getattr(self, "_behavior_execution_generation", None)
                == getattr(self, "_control_generation", None)
                and getattr(self, "_behavior_execution_thread_id", None)
                == threading.get_ident()
            )
            if authorized_find_object or authorized_find_marvin:
                return False
            return (
                active is not None
                or self._behavior_execution_generation is not None
            )

    def _run_find_object_local_progress(self):
        """Route the active Find Object owner through the generic coordinator."""
        with self._state_lock:
            active = self.mission_manager.get_active_mission()
            authorized = bool(
                active is not None
                and getattr(active, "mission_type", None) == "FIND_OBJECT"
                and self._behavior_execution_generation == self._control_generation
                and self._behavior_execution_thread_id == threading.get_ident()
            )
            mission_id = getattr(active, "mission_id", None) if authorized else None
        if not authorized:
            return {
                "ok": False,
                "action": "local_progress_with_avoidance",
                "terminal_state": "LOCAL_PROGRESS_OWNERSHIP_REJECTED",
                "mode": None,
                "reason": "find_object_behavior_ownership_unavailable",
                "physical_actions": 0,
                "max_physical_actions": self.MAX_REACTIVE_STEPS,
                "bridge_stopped": None,
                "nested_result": None,
            }
        previous = getattr(self._find_object_progress_context, "mission_id", None)
        self._find_object_progress_context.mission_id = mission_id
        try:
            return self.run_local_progress_with_avoidance()
        finally:
            if previous is None:
                try:
                    del self._find_object_progress_context.mission_id
                except AttributeError:
                    pass
            else:
                self._find_object_progress_context.mission_id = previous

    def _run_find_marvin_local_progress(self, *, remaining_actions):
        """Authorize one handoff only for the active normal Find Marvin owner."""
        base = {
            "ok": False,
            "action": "local_progress_with_avoidance",
            "terminal_state": "LOCAL_PROGRESS_OWNERSHIP_REJECTED",
            "mode": None,
            "reason": "find_marvin_behavior_ownership_unavailable",
            "physical_actions": 0,
            "max_physical_actions": self.MAX_REACTIVE_STEPS,
            "bridge_stopped": None,
            "nested_result": None,
        }
        with self._state_lock:
            active = self.mission_manager.get_active_mission()
            authorized = bool(
                self._is_normal_marvin_find_mission(active)
                and getattr(self, "_behavior_execution_generation", None)
                == getattr(self, "_control_generation", None)
                and getattr(self, "_behavior_execution_thread_id", None)
                == threading.get_ident()
            )
            mission_id = getattr(active, "mission_id", None) if authorized else None
        if not authorized:
            return base
        if (
            not isinstance(remaining_actions, int)
            or isinstance(remaining_actions, bool)
            or remaining_actions < self.MAX_REACTIVE_STEPS
        ):
            return dict(
                base,
                terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                reason="marvin_local_progress_action_budget_insufficient",
            )
        context = self._find_marvin_progress_context
        previous_mission_id = getattr(context, "mission_id", None)
        previous_remaining_actions = getattr(
            context, "remaining_actions", None,
        )
        context.mission_id = mission_id
        context.remaining_actions = remaining_actions
        try:
            return self.run_local_progress_with_avoidance()
        finally:
            if previous_mission_id is None:
                try:
                    del context.mission_id
                except AttributeError:
                    pass
            else:
                context.mission_id = previous_mission_id
            if previous_remaining_actions is None:
                try:
                    del context.remaining_actions
                except AttributeError:
                    pass
            else:
                context.remaining_actions = previous_remaining_actions

    def _local_reactive_navigation_goal_active(self):
        """Read only the goal-active flag; localization itself is irrelevant.

        A local step does not require AMCL, a map, or a valid global pose, but
        it must not overlap a navigation goal controlled by Tony2.
        """
        facade = getattr(self, "localization_facade", None)
        reader = getattr(facade, "get_localization_status", None)
        if not callable(reader):
            return None
        try:
            status = reader()
        except Exception:
            return None
        navigation = status.get("navigation") if isinstance(status, dict) else None
        if not isinstance(navigation, dict):
            return None
        return navigation.get("goal_active") is True

    def _local_reactive_lidar_state(self):
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if (
            worker is None
            or worker.running is not True
            or not isinstance(session, str)
            or not session
        ):
            return None, None
        reader = getattr(getattr(self, "world_model", None), "get_lidar_obstacles", None)
        if not callable(reader):
            return session, None
        try:
            return session, reader(expected_session=session)
        except Exception:
            return session, None

    def run_local_reactive_step(self):
        """Sense, decide, execute no more than one local guarded action, stop.

        This is intentionally not a loop.  The next action always requires a
        separate invocation and fresh local LiDAR state.
        """
        base = {
            "ok": False,
            "action": "local_reactive_step",
            "max_physical_actions": self.MAX_LOCAL_REACTIVE_ACTIONS_PER_STEP,
            "decision": None,
            "action_attempted": False,
            "action_executed": False,
            "reason": None,
            "decision_evidence": None,
            "executor_result": None,
            "stop_result": None,
            "bridge_stopped": False,
        }
        step_lock = getattr(self, "_local_reactive_step_lock", None)
        physical_lock = getattr(self, "_physical_action_lock", None)
        if step_lock is None or physical_lock is None:
            return dict(base, reason="local_reactive_locks_unavailable")
        if not step_lock.acquire(blocking=False):
            return dict(base, reason="local_reactive_step_already_running")
        physical_acquired = False
        try:
            if self.running is not True:
                return dict(base, reason="cognitive_runtime_not_running")
            if self._active_localization_has_behavior_owner():
                return dict(base, reason="physical_behavior_already_active")
            if not physical_lock.acquire(blocking=False):
                return dict(base, reason="physical_behavior_already_active")
            physical_acquired = True
            self._invalidate_marvin_live_proof("another_physical_behavior")
            if self._active_localization_lock.locked():
                return dict(base, reason="active_localization_already_running")
            navigation_goal_active = self._local_reactive_navigation_goal_active()
            if navigation_goal_active is None:
                return dict(base, reason="navigation_goal_state_unavailable")
            if navigation_goal_active:
                return dict(base, reason="navigation_goal_active")
            bridge_before = self._marvin_bridge_ready_and_stopped()
            if bridge_before.get("ok") is not True:
                return dict(base, reason="bridge_not_ready_or_not_stopped",
                            bridge_before=bridge_before)
            session, lidar = self._local_reactive_lidar_state()
            if session is None or lidar is None:
                return dict(base, reason="lidar_producer_session_or_state_unavailable")
            decision = decide_forward_reaction(
                lidar, expected_session=session,
                forward_linear_speed=FIND_MARVIN_FORWARD_SPEED_MPS,
            )
            if not isinstance(decision, dict):
                return dict(base, reason="local_reactive_decision_malformed")
            selected = decision.get("decision")
            result = dict(base, decision=selected, decision_evidence=decision)
            behavior = getattr(self, "behavior_manager", None)
            if selected == STOP_BLOCKED:
                return self._finish_local_reactive_step(
                    result, reason="local_reactive_stop_blocked",
                )
            if selected not in {FORWARD_CLEAR, TURN_LEFT, TURN_RIGHT}:
                return self._finish_local_reactive_step(
                    result, reason="local_reactive_decision_not_actionable",
                )
            if selected == FORWARD_CLEAR:
                executor = getattr(behavior, "execute_guarded_local_forward", None)
                if not callable(executor):
                    return self._finish_local_reactive_step(
                        result, reason="guarded_local_forward_unavailable",
                    )
                try:
                    execution = executor(expected_lidar_session=session)
                except Exception as exc:
                    execution = {"ok": False, "motion_executed": False,
                                 "reason": "guarded_local_forward_exception",
                                 "error": str(exc)}
                action_executed = bool(
                    isinstance(execution, dict)
                    and execution.get("motion_executed") is True
                )
            else:
                executor = getattr(behavior, "execute_guarded_turn", None)
                if not callable(executor):
                    return self._finish_local_reactive_step(
                        result, reason="guarded_turn_unavailable",
                    )
                direction = "LEFT" if selected == TURN_LEFT else "RIGHT"
                try:
                    execution = executor(
                        direction,
                        self.ACTIVE_LOCALIZATION_TURN_SPEED,
                        self.ACTIVE_LOCALIZATION_TURN_DURATION,
                        expected_lidar_session=session,
                        safety_mode="ROTATIONAL_SWEPT_FOOTPRINT",
                    )
                except Exception as exc:
                    execution = {"ok": False, "motion_executed": False,
                                 "reason": "guarded_turn_exception",
                                 "error": str(exc)}
                action_executed = bool(
                    isinstance(execution, dict)
                    and execution.get("ok") is True
                    and execution.get("permitted") is True
                    and execution.get("confirmed_forwarded") is True
                )
            result.update(
                action_attempted=True,
                action_executed=action_executed,
                executor_result=execution,
            )
            return self._finish_local_reactive_step(
                result,
                reason=(
                    "local_reactive_action_complete"
                    if action_executed else "local_reactive_executor_vetoed"
                ),
            )
        finally:
            if physical_acquired:
                physical_lock.release()
            step_lock.release()

    def _finish_local_reactive_step(self, result, *, reason):
        """Issue STOP and require the canonical Bridge-zero proof."""
        try:
            stop_result = self.robot_client.stop()
        except Exception as exc:
            stop_result = {"ok": False, "error": str(exc)}
        bridge_after = self._marvin_bridge_ready_and_stopped()
        bridge_stopped = bridge_after.get("ok") is True
        return dict(
            result,
            ok=bool(result.get("action_executed") is True
                    and isinstance(stop_result, dict)
                    and stop_result.get("ok") is True
                    and bridge_stopped),
            reason=(reason if bridge_stopped else "bridge_not_stopped_after_local_reactive_step"),
            stop_result=stop_result,
            bridge_stopped=bridge_stopped,
            bridge_after_stop=bridge_after,
        )

    def run_bounded_local_reactive_avoidance(self):
        """Run at most four fully stopped, LiDAR-re-sensed local reactions.

        This method deliberately owns no motion implementation.  Each cycle
        delegates to ``run_local_reactive_step()``, which remains responsible
        for sensing, deciding, guarded execution, STOP, and Bridge-zero
        verification.  The episode lock only prevents overlapping bounded
        episodes; it never holds the physical-action lock across cycles.
        """
        base = {
            "ok": False,
            "action": "bounded_local_reactive_avoidance",
            "terminal_state": None,
            "reason": None,
            "steps_attempted": 0,
            "physical_actions": 0,
            "max_steps": self.MAX_REACTIVE_STEPS,
            "turns_executed": 0,
            "forward_steps_executed": 0,
            "bridge_stopped": False,
            "history": [],
        }
        episode_lock = getattr(self, "_bounded_local_reactive_avoidance_lock", None)
        if episode_lock is None:
            return dict(base, terminal_state="EXECUTION_FAILED",
                        reason="bounded_reactive_episode_lock_unavailable")
        if not episode_lock.acquire(blocking=False):
            return dict(base, terminal_state="OWNERSHIP_REJECTED",
                        reason="bounded_reactive_avoidance_already_running")

        ownership_reasons = {
            "physical_behavior_already_active",
            "active_localization_already_running",
            "navigation_goal_active",
            "local_reactive_step_already_running",
        }
        try:
            if self.running is not True:
                return dict(base, terminal_state="OWNERSHIP_REJECTED",
                            reason="cognitive_runtime_not_running")

            for step_number in range(1, self.MAX_REACTIVE_STEPS + 1):
                step_result = self.run_local_reactive_step()
                if not isinstance(step_result, dict):
                    return dict(
                        base,
                        terminal_state="EXECUTION_FAILED",
                        reason="local_reactive_step_result_malformed",
                    )

                decision = step_result.get("decision")
                action_attempted = step_result.get("action_attempted") is True
                action_executed = step_result.get("action_executed") is True
                bridge_stopped = step_result.get("bridge_stopped") is True
                history_entry = {
                    "step": step_number,
                    "decision": decision,
                    "action_attempted": action_attempted,
                    "action_executed": action_executed,
                    "reason": step_result.get("reason"),
                    "bridge_stopped": bridge_stopped,
                }
                base["history"].append(history_entry)
                base["steps_attempted"] = step_number
                base["bridge_stopped"] = bridge_stopped

                # A pre-ownership rejection must not command STOP against the
                # legitimate current owner.  End this episode without a retry.
                if step_result.get("reason") in ownership_reasons:
                    return dict(base, terminal_state="OWNERSHIP_REJECTED",
                                reason=step_result.get("reason"))

                # Account for transport as soon as the single-step authority
                # reports it.  A later STOP/Bridge failure does not erase the
                # fact that this bounded physical action happened.
                if action_executed:
                    base["physical_actions"] += 1
                    if decision in {TURN_LEFT, TURN_RIGHT}:
                        base["turns_executed"] += 1
                    elif decision == FORWARD_CLEAR:
                        base["forward_steps_executed"] += 1

                # No next sense/action is permitted until the preceding
                # single-step owner has completed STOP and Bridge-zero proof.
                if not bridge_stopped:
                    return dict(base, terminal_state="EXECUTION_FAILED",
                                reason="bridge_not_stopped_after_reactive_cycle")

                if decision == STOP_BLOCKED:
                    return dict(base, ok=True, terminal_state="BLOCKED",
                                reason=step_result.get("reason"))

                if action_attempted and not action_executed:
                    terminal_state = (
                        "SAFETY_VETO"
                        if step_result.get("reason") == "local_reactive_executor_vetoed"
                        else "EXECUTION_FAILED"
                    )
                    return dict(base, terminal_state=terminal_state,
                                reason=step_result.get("reason"))

                if not action_executed:
                    return dict(base, terminal_state="EXECUTION_FAILED",
                                reason="local_reactive_action_not_executed")

                # A successful first forward action is sufficient.  Likewise,
                # after one or more avoidance turns, a fresh FORWARD_CLEAR
                # decision proves a newly clear local path and ends the episode.
                if decision == FORWARD_CLEAR:
                    return dict(base, ok=True, terminal_state="PATH_CLEAR",
                                reason="guarded_forward_completed")

            # Every successful single step already issued STOP and verified the
            # Bridge before this bounded budget can be exhausted.
            return dict(base, ok=True, terminal_state="MAX_STEPS_REACHED",
                        reason="reactive_step_budget_exhausted")
        finally:
            episode_lock.release()

    def run_local_progress_with_avoidance(self):
        """Attempt one local forward-progress unit without adding motion paths.

        The initial LiDAR decision is a routing decision only.  A clear path
        delegates once to the existing single-step coordinator; a turnable
        obstruction delegates once to the existing four-step bounded episode.
        Neither branch carries cached motion authorization into its delegate.
        """
        base = {
            "ok": False,
            "action": "local_progress_with_avoidance",
            "terminal_state": None,
            "mode": None,
            "reason": None,
            "initial_decision": None,
            "initial_decision_evidence": None,
            "physical_actions": 0,
            "max_physical_actions": self.MAX_REACTIVE_STEPS,
            "bridge_stopped": None,
            "nested_result": None,
        }
        handoff_lock = getattr(self, "_local_progress_with_avoidance_lock", None)
        if handoff_lock is None:
            return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                        reason="local_progress_handoff_lock_unavailable")
        if not handoff_lock.acquire(blocking=False):
            return dict(base, terminal_state="LOCAL_PROGRESS_OWNERSHIP_REJECTED",
                        reason="local_progress_with_avoidance_already_running")

        ownership_reasons = {
            "physical_behavior_already_active",
            "active_localization_already_running",
            "navigation_goal_active",
            "local_reactive_step_already_running",
            "bounded_reactive_avoidance_already_running",
        }
        try:
            if self.running is not True:
                return dict(base, terminal_state="LOCAL_PROGRESS_OWNERSHIP_REJECTED",
                            reason="cognitive_runtime_not_running")

            producer_session, lidar_state = self._local_reactive_lidar_state()
            if producer_session is None or lidar_state is None:
                return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            reason="lidar_producer_session_or_state_unavailable")
            decision_result = decide_forward_reaction(
                lidar_state,
                expected_session=producer_session,
                forward_linear_speed=FIND_MARVIN_FORWARD_SPEED_MPS,
            )
            if not isinstance(decision_result, dict):
                return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            reason="local_progress_decision_malformed")

            decision = decision_result.get("decision")
            base.update(
                initial_decision=decision,
                initial_decision_evidence=decision_result,
            )
            if decision == STOP_BLOCKED:
                return dict(
                    base,
                    ok=True,
                    terminal_state="LOCAL_PROGRESS_BLOCKED",
                    mode="NO_MOTION",
                    reason=decision_result.get("reason", "local_progress_stop_blocked"),
                )

            if decision == FORWARD_CLEAR:
                step_result = self.run_local_reactive_step()
                if not isinstance(step_result, dict):
                    return dict(
                        base,
                        terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                        mode="DIRECT_FORWARD",
                        reason="local_reactive_step_result_malformed",
                    )
                executed = step_result.get("action_executed") is True
                bridge_stopped = step_result.get("bridge_stopped") is True
                direct = dict(
                    base,
                    mode="DIRECT_FORWARD",
                    nested_result=step_result,
                    physical_actions=1 if executed else 0,
                    bridge_stopped=bridge_stopped,
                )
                if step_result.get("reason") in ownership_reasons:
                    return dict(direct,
                                terminal_state="LOCAL_PROGRESS_OWNERSHIP_REJECTED",
                                reason=step_result.get("reason"))
                if not bridge_stopped:
                    return dict(direct,
                                terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                                reason="bridge_not_stopped_after_direct_local_progress")
                if step_result.get("decision") == STOP_BLOCKED:
                    return dict(direct, ok=True,
                                terminal_state="LOCAL_PROGRESS_BLOCKED",
                                reason=step_result.get("reason"))
                if step_result.get("action_attempted") is True and not executed:
                    terminal = (
                        "LOCAL_PROGRESS_SAFETY_VETO"
                        if step_result.get("reason") == "local_reactive_executor_vetoed"
                        else "LOCAL_PROGRESS_EXECUTION_FAILED"
                    )
                    return dict(direct, terminal_state=terminal,
                                reason=step_result.get("reason"))
                if executed and step_result.get("decision") == FORWARD_CLEAR:
                    return dict(direct, ok=True,
                                terminal_state="LOCAL_PROGRESS_COMPLETE",
                                reason="guarded_forward_completed")
                return dict(direct, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            reason="direct_local_progress_not_completed")

            if decision not in {TURN_LEFT, TURN_RIGHT}:
                return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            reason="local_progress_decision_unknown")

            bounded_result = self.run_bounded_local_reactive_avoidance()
            if not isinstance(bounded_result, dict):
                return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            mode="BOUNDED_AVOIDANCE",
                            reason="bounded_reactive_result_malformed")
            physical_actions = bounded_result.get("physical_actions")
            if (not isinstance(physical_actions, int)
                    or isinstance(physical_actions, bool)
                    or not 0 <= physical_actions <= self.MAX_REACTIVE_STEPS):
                return dict(base, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            mode="BOUNDED_AVOIDANCE",
                            nested_result=bounded_result,
                            reason="bounded_reactive_physical_action_count_invalid")
            bounded = dict(
                base,
                mode="BOUNDED_AVOIDANCE",
                nested_result=bounded_result,
                physical_actions=physical_actions,
                bridge_stopped=bounded_result.get("bridge_stopped") is True,
            )
            terminal_map = {
                "PATH_CLEAR": ("LOCAL_PROGRESS_COMPLETE", True),
                "BLOCKED": ("LOCAL_PROGRESS_BLOCKED", True),
                "SAFETY_VETO": ("LOCAL_PROGRESS_SAFETY_VETO", False),
                "MAX_STEPS_REACHED": ("LOCAL_PROGRESS_MAX_STEPS_REACHED", True),
                "OWNERSHIP_REJECTED": ("LOCAL_PROGRESS_OWNERSHIP_REJECTED", False),
                "EXECUTION_FAILED": ("LOCAL_PROGRESS_EXECUTION_FAILED", False),
            }
            mapped = terminal_map.get(bounded_result.get("terminal_state"))
            if mapped is None:
                return dict(bounded, terminal_state="LOCAL_PROGRESS_EXECUTION_FAILED",
                            reason="bounded_reactive_terminal_state_unknown")
            terminal_state, ok = mapped
            return dict(bounded, ok=ok, terminal_state=terminal_state,
                        reason=bounded_result.get("reason"))
        finally:
            handoff_lock.release()

    def run_bounded_active_localization(self):
        """Obtain LiDAR viewpoints through at most six guarded in-place turns.

        Tony2 remains the sole authority for AMCL lifecycle and canonical map
        pose validation.  This method owns only the bounded physical scan and
        never calls a Home seed or a navigation-goal API.
        """
        base = {
            "ok": False,
            "action": "bounded_active_localization",
            "turns_executed": 0,
            "turn_history": [],
            "last_turn_direction": None,
            "last_localization_result": None,
            "localized": False,
            "terminal_reason": None,
        }
        if not self._active_localization_lock.acquire(blocking=False):
            return dict(
                base,
                terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                reason="active_localization_already_running",
            )
        physical_lock = getattr(self, "_physical_action_lock", None)
        if physical_lock is None:
            physical_lock = threading.Lock()
            self._physical_action_lock = physical_lock
        if not physical_lock.acquire(blocking=False):
            self._active_localization_lock.release()
            return dict(
                base,
                terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                reason="physical_behavior_already_active",
            )
        try:
            if self.running is not True:
                return dict(
                    base,
                    terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                    reason="cognitive_runtime_not_running",
                )
            if self._active_localization_has_behavior_owner():
                return dict(
                    base,
                    terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                    reason="physical_behavior_already_active",
                )
            self._invalidate_marvin_live_proof("another_physical_behavior")
            status = self.localization_facade.get_localization_status()
            base["last_localization_result"] = status
            if self._active_localization_pose_is_trusted(status):
                return dict(base, ok=True, localized=True,
                            terminal_reason="ACTIVE_LOCALIZATION_SUCCESS",
                            reason="canonical_localization_already_validated")
            if not self._active_localization_is_recoverable(status):
                return dict(
                    base,
                    terminal_reason="ACTIVE_LOCALIZATION_HARD_FAILURE",
                    reason="localization_not_recoverable",
                )

            for turn_index in range(self.MAX_ACTIVE_LOCALIZATION_TURNS):
                status = self.localization_facade.get_localization_status()
                base["last_localization_result"] = status
                if self._active_localization_pose_is_trusted(status):
                    return dict(base, ok=True, localized=True,
                                terminal_reason="ACTIVE_LOCALIZATION_SUCCESS",
                                reason="canonical_localization_validated")
                if not self._active_localization_is_recoverable(status):
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_HARD_FAILURE",
                        reason="localization_not_recoverable",
                    )
                if self._active_localization_has_behavior_owner():
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                        reason="physical_behavior_already_active",
                    )
                bridge_before = self._marvin_bridge_ready_and_stopped()
                if bridge_before.get("ok") is not True:
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_BRIDGE_NOT_STOPPED",
                        reason="bridge_not_ready_or_not_stopped",
                        bridge_status=bridge_before,
                    )
                session, lidar = self._active_localization_lidar_is_current()
                if lidar is None:
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_LIDAR_NOT_READY",
                        reason="lidar_not_fresh_or_geometry_invalid",
                    )

                direction = "LEFT" if turn_index % 2 == 0 else "RIGHT"
                base["last_turn_direction"] = direction
                try:
                    turn_result = self.behavior_manager.execute_guarded_turn(
                        direction,
                        self.ACTIVE_LOCALIZATION_TURN_SPEED,
                        self.ACTIVE_LOCALIZATION_TURN_DURATION,
                        expected_lidar_session=session,
                        safety_mode="ROTATIONAL_SWEPT_FOOTPRINT",
                    )
                except Exception as exc:
                    turn_result = {"ok": False, "error": str(exc)}
                base["turns_executed"] += 1
                turn_entry = {
                    "index": turn_index + 1,
                    "direction": direction,
                    "linear_x": 0.0,
                    "angular_speed": self.ACTIVE_LOCALIZATION_TURN_SPEED,
                    "duration": self.ACTIVE_LOCALIZATION_TURN_DURATION,
                    "turn_result": turn_result,
                }

                try:
                    stop_result = self.robot_client.stop()
                except Exception as exc:
                    stop_result = {"ok": False, "error": str(exc)}
                turn_entry["stop_result"] = stop_result
                base["turn_history"].append(turn_entry)
                if not isinstance(stop_result, dict) or stop_result.get("ok") is not True:
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_STOP_FAILED",
                        reason="stop_command_failed",
                    )
                bridge_after = self._marvin_bridge_ready_and_stopped()
                turn_entry["bridge_after_stop"] = bridge_after
                if bridge_after.get("ok") is not True:
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_STOP_FAILED",
                        reason="bridge_not_stopped_after_turn",
                    )
                if not (
                    isinstance(turn_result, dict)
                    and turn_result.get("ok") is True
                    and turn_result.get("permitted") is True
                    and turn_result.get("confirmed_forwarded") is True
                ):
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_TURN_BLOCKED",
                        reason="guarded_turn_failed",
                    )

                retry = self.localization_facade.retry_global_localization()
                base["last_localization_result"] = retry
                status = self.localization_facade.get_localization_status()
                base["last_localization_result"] = status
                if self._active_localization_pose_is_trusted(status):
                    return dict(base, ok=True, localized=True,
                                terminal_reason="ACTIVE_LOCALIZATION_SUCCESS",
                                reason="canonical_localization_validated",
                                localization_retry=retry)
                if not self._active_localization_is_recoverable(status):
                    return dict(
                        base,
                        terminal_reason="ACTIVE_LOCALIZATION_HARD_FAILURE",
                        reason="localization_retry_failed",
                        localization_retry=retry,
                    )

            return dict(
                base,
                terminal_reason="ACTIVE_LOCALIZATION_EXHAUSTED",
                reason="active_localization_turn_budget_exhausted",
            )
        finally:
            physical_lock.release()
            self._active_localization_lock.release()

    def execute_single_marvin_alignment(
        self, *, direction, angular_speed, duration, source_frame_stamp_ns,
    ):
        return self._dispatch_marvin_observation_action(
            self._execute_single_marvin_alignment,
            direction=direction, angular_speed=angular_speed, duration=duration,
            source_frame_stamp_ns=source_frame_stamp_ns,
        )

    def _marvin_motion_owner_is_current(self):
        if getattr(self, "_marvin_live_proof_owner", None) is not None:
            return self._marvin_live_proof_owner_is_current()
        manager = getattr(self, "mission_manager", None)
        active = manager.get_active_mission() if manager is not None else None
        generation = getattr(self, "_behavior_execution_generation", None)
        if active is not None or generation is not None:
            return (active is not None and self._is_normal_marvin_find_mission(active)
                    and generation == getattr(self, "_control_generation", None)
                    and getattr(self, "_behavior_execution_thread_id", None) == threading.get_ident())
        return getattr(self, "_last_runtime_state", None) != "STOPPED"

    def _dispatch_marvin_observation_action(self, callback, *,
                                            search_received_monotonic_seconds=None, **kwargs):
        rejected = {"ok": False, "execution_authorized": False,
                    "actions_executed": 0, "motion_executed": False,
                    "source_frame_stamp_ns": kwargs.get("source_frame_stamp_ns"),
                    "reason": "marvin_motion_ownership_conflict"}
        controller = getattr(self, "_marvin_controller_lock", None)
        physical = getattr(self, "_physical_action_lock", None)
        if controller is not None and not controller.acquire(blocking=False):
            return rejected
        physical_acquired = False
        try:
            if physical is not None:
                physical_acquired = physical.acquire(blocking=False)
                if not physical_acquired:
                    return rejected
            with self._state_lock:
                if not self._marvin_motion_owner_is_current():
                    return rejected
                proof_owner = getattr(self, "_marvin_live_proof_owner", None)
                if proof_owner is None:
                    self._invalidate_marvin_live_proof("another_physical_behavior")
                if proof_owner is not None:
                    if proof_owner.dispatch_opportunities >= 1:
                        return dict(rejected, reason="marvin_live_proof_action_limit_reached")
                    proof_owner.dispatch_opportunities += 1
                generation = getattr(self, "_control_generation", None)
                # Capture the receipt bound to the exact authorization before
                # one-shot consumption clears it. Callers cannot supply a
                # replacement receipt through the public motion API.
                observation = getattr(self, "_marvin_alignment_observation", None)
                receipt = None
                if (isinstance(observation, dict)
                        and observation.get("source_frame_stamp_ns") == kwargs["source_frame_stamp_ns"]):
                    tracker = observation.get("opencv_tracker")
                    if (isinstance(tracker, dict)
                            and tracker.get("source_frame_stamp_ns") == kwargs["source_frame_stamp_ns"]
                            and tracker.get("received_monotonic_seconds") == observation.get("received_monotonic_seconds")):
                        receipt = observation.get("received_monotonic_seconds")
                if callback == self._guarded_marvin_v2_search_turn:
                    receipt = search_received_monotonic_seconds

            def dispatch_guard():
                with self._state_lock:
                    return (generation == getattr(self, "_control_generation", None)
                            and self._marvin_motion_owner_is_current()
                            and self._marvin_motion_stamp_is_fresh(kwargs["source_frame_stamp_ns"], receipt))

            return callback(**kwargs, dispatch_guard=dispatch_guard)
        finally:
            if physical_acquired:
                physical.release()
            if controller is not None:
                controller.release()

    def _execute_marvin_v2_search_turn(self, stamp, receipt):
        return self._dispatch_marvin_observation_action(
            self._guarded_marvin_v2_search_turn, source_frame_stamp_ns=stamp,
            search_received_monotonic_seconds=receipt,
        )

    def _guarded_marvin_v2_search_turn(self, *, source_frame_stamp_ns, dispatch_guard):
        base = {"ok": False, "actions_executed": 0, "motion_executed": False}
        session, lidar = self._active_localization_lidar_is_current()
        if lidar is None:
            return dict(base, reason="find_marvin_search_lidar_not_current")
        with self._state_lock:
            if source_frame_stamp_ns in self._marvin_alignment_consumed_source_frame_stamps:
                return dict(base, reason="marvin_search_observation_already_consumed")
            if not dispatch_guard():
                return dict(base, reason="marvin_motion_observation_stale_or_preempted")
            self._marvin_alignment_consumed_source_frame_stamps.add(source_frame_stamp_ns)
            self._reset_marvin_alignment_consensus()
        try:
            turn = self.behavior_manager._execute_target_directed_turn(
                SCAN_DIRECTION, self.behavior_manager.MARVIN_SEARCH_TURN_SPEED,
                self.behavior_manager.MARVIN_SEARCH_TURN_SECONDS,
                expected_lidar_session=session, safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                dispatch_guard=dispatch_guard,
            )
        finally:
            self.robot_client.stop()
        moved = (isinstance(turn, dict) and turn.get("ok") is True
                 and turn.get("permitted") is True and turn.get("confirmed_forwarded") is True)
        if moved:
            evidence = turn.get("action_lidar_evidence", lidar)
            if not isinstance(evidence, dict):
                evidence = {}
            self._marvin_last_action_lidar_evidence = (
                evidence.get("producer_session"), evidence.get("acquisition_sequence"))
        return dict(base, ok=moved, actions_executed=1, motion_executed=moved,
                    turn_result=turn, reason="find_marvin_search_turn_complete" if moved
                    else "find_marvin_search_guarded_turn_vetoed")

    def _execute_single_marvin_strafe(self, *, source_frame_stamp_ns,
                                      local_detour_context, dispatch_guard):
        """Private mission-owned executor: new identity frame, one lateral step."""
        base = {"ok": False, "action": "single_marvin_local_strafe",
            "execution_authorized": False, "motion_executed": False, "actions_executed": 0,
            "source_stamp_consumed": False, "full_step_completed": False,
            "source_frame_stamp_ns": source_frame_stamp_ns,
            "requested_duration": LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS,
            "actual_confirmed_run_duration_seconds": None}
        kind = local_detour_context.get("selected_action_type")
        if kind not in {"STRAFE_LEFT", "STRAFE_RIGHT"}:
            return dict(base, reason="marvin_lateral_parameters_invalid")
        session, lidar = self._active_localization_lidar_is_current()
        if lidar is None or not self.running or not dispatch_guard():
            return dict(base, reason="marvin_lateral_lidar_or_execution_not_current")
        zero = self._marvin_bridge_ready_and_stopped()
        if (zero.get("ok") is not True or zero.get("status") != "READY"
                or (zero.get("motion_capabilities") or {}).get("linear_y") is not True
                or (zero.get("motion") or {}).get("linear_y") != 0):
            return dict(base, reason="bridge_lateral_support_unavailable")
        with self._state_lock:
            consumed = self._marvin_alignment_consumed_source_frame_stamps
            if source_frame_stamp_ns in consumed:
                return dict(base, reason="marvin_lateral_observation_already_consumed")
            observation = self._marvin_alignment_observation
            if (not isinstance(observation, dict) or observation.get("source_frame_stamp_ns") != source_frame_stamp_ns
                    or not self._marvin_detour_owner_is_current()):
                return dict(base, reason="marvin_lateral_ownership_or_observation_invalid")
            tracker = observation.get("opencv_tracker") or {}
            strict = self._marvin_v2_action_observation(dict(observation,
                arrival=observation.get("target_standoff")), tracker,
                observation.get("controller_state"), observation.get("controller_decision"))
            if strict is None or not dispatch_guard():
                return dict(base, reason="marvin_lateral_observation_not_authorized")

            duration = None

            def validate_selection(sample):
                association = self._marvin_v2_lidar_arrival(tracker, sample, current_action_jit=True)
                if association.get("ok") is not True or association.get("arrived_at_marvin"):
                    return {"accepted": False}
                selection = select_marvin_escape_action(sample, association, expected_session=session,
                    allow_strafe=True, previous_selection=local_detour_context.get("previous_selection"),
                    strafe_duration_limit=duration if duration is not None else LOCAL_AVOIDANCE_STRAFE_MAX_SECONDS)
                candidate_duration = (selection.get("options", {}).get(kind) or {}).get("requested_duration")
                accepted = (selection.get("action_type") == kind
                    and (duration is None or candidate_duration == duration))
                if accepted:
                    self._marvin_last_action_lidar_evidence = (session, sample["acquisition_sequence"])
                return dict(selection, accepted=accepted, target_association=association)

            selection = validate_selection(lidar)
            if not selection.get("accepted"):
                return dict(base, reason="marvin_local_detour_jit_veto", local_detour=selection)
            duration = selection["options"][kind]["requested_duration"]
            base["requested_duration"] = duration
            # Permanently consume before any transport. Diagnostics never seed this set.
            consumed.add(source_frame_stamp_ns)
            self._marvin_alignment_consensus = []
            self._marvin_alignment_geometry_history = []
        base.update(execution_authorized=True, source_stamp_consumed=True,
            action_type=kind, direction=kind.split("_")[1])
        if self.behavior_manager.mark_strict_v2_action_dispatched(source_frame_stamp_ns, "strafe") is not True:
            return dict(base, reason="marvin_lateral_tracker_episode_not_current")
        signed_speed = LOCAL_AVOIDANCE_STRAFE_SPEED_MPS * (1 if kind == "STRAFE_LEFT" else -1)
        try:
            step = self.behavior_manager.execute_guarded_marvin_lateral_step(
                expected_lidar_session=session, linear_y=signed_speed,
                duration=duration, dispatch_guard=dispatch_guard,
                selection_validator=validate_selection)
        except Exception as exc:
            step = {"ok": False, "motion_executed": False, "reason": "marvin_lateral_step_exception", "error": str(exc)}
        try:
            stopped = self.robot_client.stop()
        except Exception as exc:
            stopped = {"ok": False, "error": str(exc)}
        zero = self._marvin_bridge_ready_and_stopped()
        stop_ok = (isinstance(stopped, dict) and stopped.get("ok") is True
            and zero.get("ok") is True and zero.get("status") == "READY"
            and (zero.get("motion") or {}).get("linear_y") == 0)
        transport = step.get("lateral_result") or {}
        moved = step.get("ok") is True and step.get("motion_executed") is True
        interrupted = transport.get("bounded_lateral_invalidated") is True
        if moved or (transport.get("transport_result") or {}).get("ok") is True:
            self._marvin_target_range_association.record_forward_bound(
                LOCAL_AVOIDANCE_STRAFE_SPEED_MPS, duration)
        return dict(base, ok=moved and stop_ok, motion_executed=moved,
            full_step_completed=moved and stop_ok, actions_executed=int(moved), interrupted=interrupted,
            interruption_reason=transport.get("reason") if interrupted else None,
            lateral_step=step, stop_result=stopped, bridge_after_stop=zero,
            bridge_stop_confirmed=stop_ok,
            local_detour=step.get("local_detour") or selection,
            reason="marvin_lateral_step_complete" if moved and stop_ok else
                step.get("reason") if not moved else "marvin_lateral_stop_unconfirmed")

    def _execute_single_marvin_alignment(
        self, *, direction, angular_speed, duration, source_frame_stamp_ns, dispatch_guard,
        local_detour_context=None,
    ):
        """Execute one capped, guarded Marvin alignment turn and then stop.

        This is intentionally not a controller, mission, or general motion
        interface.  It delegates one angular-only request to the active
        BehaviorManager so its in-memory LiDAR producer session is preserved.
        """
        base = {
            "ok": False,
            "action": "single_marvin_alignment_turn",
            "execution_authorized": False,
            "motion_executed": False,
            "actions_executed": 0,
            "direction": None,
            "angular_speed": None,
            "duration": None,
            "source_frame_stamp_ns": None,
            "producer_session": None,
            "turn_result": None,
            "stop_result": None,
            "reason": None,
        }
        normalized_direction = (
            direction.strip().upper() if isinstance(direction, str) else None
        )
        if normalized_direction not in {"LEFT", "RIGHT"}:
            return dict(base, reason="marvin_alignment_direction_invalid")
        if not _bounded_alignment_number(angular_speed, maximum=0.25):
            return dict(base, reason="marvin_alignment_angular_speed_invalid")
        if not _bounded_alignment_number(duration, maximum=0.50):
            return dict(base, reason="marvin_alignment_duration_invalid")
        if (
            not isinstance(source_frame_stamp_ns, int)
            or isinstance(source_frame_stamp_ns, bool)
            or source_frame_stamp_ns < 0
        ):
            return dict(base, reason="marvin_alignment_source_frame_stamp_invalid")
        base.update(
            direction=normalized_direction,
            angular_speed=float(angular_speed),
            duration=float(duration),
            source_frame_stamp_ns=source_frame_stamp_ns,
        )
        if self.running is not True:
            return dict(base, reason="marvin_alignment_runtime_not_running")
        behavior = getattr(self, "behavior_manager", None)
        execute_turn = getattr(behavior, "_execute_target_directed_turn", None)
        robot = getattr(behavior, "robot", None)
        stop = getattr(robot, "stop", None)
        if not callable(execute_turn) or not callable(stop):
            return dict(base, reason="marvin_alignment_guarded_turn_unavailable")
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if worker is None or worker.running is not True or not isinstance(session, str) or not session:
            return dict(base, reason="marvin_alignment_lidar_session_unavailable")
        base["producer_session"] = session
        world_model = getattr(self, "world_model", None)
        get_lidar = getattr(world_model, "get_lidar_obstacles", None)
        if not callable(get_lidar):
            return dict(base, reason="marvin_alignment_lidar_read_unavailable")
        try:
            lidar = get_lidar(expected_session=session)
        except Exception as exc:
            return dict(base, reason="marvin_alignment_lidar_read_failed", error=str(exc))
        if not _marvin_alignment_lidar_is_current(lidar, session):
            return dict(base, reason="marvin_alignment_lidar_not_current", lidar=lidar)

        with self._state_lock:
            consumed = self._marvin_alignment_consumed_source_frame_stamps
            if source_frame_stamp_ns in consumed:
                return dict(base, reason="marvin_alignment_observation_already_consumed")
            observation = self._marvin_alignment_observation
            if not isinstance(observation, dict) or (
                observation.get("source_frame_stamp_ns") != source_frame_stamp_ns
            ):
                return dict(base, reason="marvin_alignment_observation_not_current")
            tracker = observation.get("opencv_tracker")
            detour_mode = local_detour_context is not None
            if detour_mode and (not isinstance(local_detour_context, dict)
                    or not self._marvin_detour_owner_is_current()):
                return dict(base, reason="marvin_local_detour_ownership_conflict")
            expected_direction = normalized_direction if detour_mode else (
                "LEFT" if observation.get("controller_decision") == "TURN_LEFT"
                else "RIGHT" if observation.get("controller_decision") == "TURN_RIGHT"
                else None
            )
            strict_observation = bool(
                observation.get("identity_confirmed") is True
                and (
                    observation.get("identity_source")
                    == "gemini_marvin_candidate_selection"
                    or (
                        observation.get("identity_source")
                        == "marvin_locked_tracker_continuity"
                        and observation.get("post_action_tracker_continuity") is True
                        and type(observation.get("post_action_source_frame_stamp_ns")) is int
                        and type(source_frame_stamp_ns) is int
                        and source_frame_stamp_ns
                        > observation["post_action_source_frame_stamp_ns"]
                    )
                )
                and observation.get("controller_state") == (
                    VISUAL_READY_TO_APPROACH if detour_mode else VISUAL_READY_TO_ALIGN)
                and (not detour_mode or observation.get("controller_decision") in {"FORWARD", "AVOID"})
                and expected_direction == normalized_direction
                and isinstance(tracker, dict)
                and tracker.get("active") is True
                and tracker.get("matched") is True
                and isinstance(tracker.get("quality"), (int, float))
                and not isinstance(tracker.get("quality"), bool)
                and isinstance(tracker.get("threshold"), (int, float))
                and not isinstance(tracker.get("threshold"), bool)
                and tracker.get("quality") >= tracker.get("threshold")
                and tracker.get("source_frame_stamp_ns") == source_frame_stamp_ns
                and isinstance(tracker.get("bbox"), dict)
            )
            if not strict_observation:
                return dict(base, reason="marvin_alignment_observation_not_authorized")
            if not dispatch_guard():
                self._reset_marvin_alignment_consensus()
                return dict(base, reason="marvin_motion_observation_stale_or_preempted")
            if detour_mode:
                # An advisory decision cannot authorize a turn. Revalidate
                # target range, blockage and the selected side from JIT LiDAR.
                standoff = self._marvin_v2_lidar_arrival(tracker, lidar)
                if standoff.get("ok") is not True or standoff.get("arrived_at_marvin") is not False:
                    return dict(base, reason="marvin_local_detour_target_standoff_veto")
                effective_duration = (min(0.50, (standoff["target_distance_m"] -
                    TARGET_STANDOFF_M) / FIND_MARVIN_FORWARD_SPEED_MPS)
                    if standoff.get("target_range_association_trusted") is True else 0.50)
                if local_detour_context.get("four_primitives"):
                    detour = select_marvin_escape_action(lidar, standoff, expected_session=session,
                        allow_strafe=local_detour_context["allow_strafe"],
                        previous_selection=local_detour_context.get("previous_selection"))
                    if detour.get("action_type") != local_detour_context["selected_action_type"]:
                        return dict(base, reason="marvin_local_detour_jit_veto", local_detour=detour)
                else:
                    detour = select_marvin_detour(lidar, expected_session=session,
                        forward_speed=FIND_MARVIN_FORWARD_SPEED_MPS, forward_duration=effective_duration,
                        previous_direction=local_detour_context.get("previous_direction"),
                        previous_clearances=local_detour_context.get("previous_clearances"))
                if detour["direction"] != normalized_direction:
                    return dict(base, reason="marvin_local_detour_jit_veto", local_detour=detour)
                base.update(action="single_marvin_local_detour_turn", local_detour=detour)
            # Consume before guarded dispatch: any later transport ambiguity or
            # JIT veto requires a genuinely new strict observation, preventing
            # an HTTP retry from duplicating a possible physical action.
            consumed.add(source_frame_stamp_ns)
            self._marvin_alignment_geometry_history = []
            self._marvin_alignment_consensus = []

            if type(lidar.get("acquisition_sequence")) is int:
                self._marvin_last_action_lidar_evidence = (session, lidar["acquisition_sequence"])

        mark_action = getattr(behavior, "mark_strict_v2_action_dispatched", None)
        if callable(mark_action) and mark_action(source_frame_stamp_ns, "turn") is not True:
            return dict(base, reason="marvin_alignment_tracker_episode_not_current")

        base["execution_authorized"] = True
        try:
            turn = execute_turn(
                normalized_direction,
                float(angular_speed),
                float(duration),
                expected_lidar_session=session,
                safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
                dispatch_guard=dispatch_guard,
            )
        except Exception as exc:
            try:
                stop_result = stop()
            except Exception as stop_exc:
                stop_result = {"ok": False, "error": str(stop_exc)}
            return dict(
                base, actions_executed=1, turn_result=None,
                stop_result=stop_result, reason="marvin_alignment_turn_exception",
                error=str(exc), error_type=type(exc).__name__,
            )
        try:
            stop_result = stop()
        except Exception as exc:
            stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        motion_executed = bool(
            isinstance(turn, dict)
            and turn.get("ok") is True
            and turn.get("permitted") is True
            and turn.get("confirmed_forwarded") is True
        )
        stop_ok = isinstance(stop_result, dict) and stop_result.get("ok") is True
        if motion_executed and isinstance(turn.get("action_lidar_evidence"), dict):
            evidence = turn["action_lidar_evidence"]
            self._marvin_last_action_lidar_evidence = (
                evidence.get("producer_session"), evidence.get("acquisition_sequence"))
        return dict(
            base,
            ok=motion_executed and stop_ok,
            motion_executed=motion_executed,
            actions_executed=1,
            turn_result=turn,
            stop_result=stop_result,
            reason=(
                "marvin_alignment_turn_complete" if motion_executed and stop_ok
                else "marvin_alignment_turn_or_stop_failed"
            ),
        )

    def execute_single_marvin_approach(
        self, *, linear_speed, duration, source_frame_stamp_ns,
    ):
        return self._dispatch_marvin_observation_action(
            self._execute_single_marvin_approach, linear_speed=linear_speed,
            duration=duration, source_frame_stamp_ns=source_frame_stamp_ns,
        )

    def _execute_single_marvin_approach(self, *, linear_speed, duration, source_frame_stamp_ns,
                                       dispatch_guard, local_detour_context=None):
        """Execute one current-observation-bound guarded Marvin forward step."""
        base = {"ok": False, "action": "single_marvin_approach_step",
                "execution_authorized": False, "motion_executed": False,
                "actions_executed": 0, "linear_speed": None, "duration": None,
                "source_frame_stamp_ns": None, "producer_session": None, "approach_result": None,
                "stop_result": None, "reason": None}
        if not _bounded_alignment_number(linear_speed, maximum=FIND_MARVIN_FORWARD_SPEED_MPS):
            return dict(base, reason="marvin_approach_linear_speed_invalid")
        if not _bounded_alignment_number(duration, maximum=0.50):
            return dict(base, reason="marvin_approach_duration_invalid")
        if float(linear_speed) != FIND_MARVIN_FORWARD_SPEED_MPS or float(duration) != 0.50:
            return dict(base, reason="marvin_approach_parameters_not_calibrated")
        if (
            not isinstance(source_frame_stamp_ns, int)
            or isinstance(source_frame_stamp_ns, bool)
            or source_frame_stamp_ns < 0
        ):
            return dict(base, reason="marvin_approach_source_frame_stamp_invalid")
        base.update(
            linear_speed=float(linear_speed), duration=float(duration),
            source_frame_stamp_ns=source_frame_stamp_ns,
        )
        bypass_mode = local_detour_context is not None
        if bypass_mode and (not isinstance(local_detour_context, dict)
                or local_detour_context.get("selected_action_type") != "BYPASS_FORWARD"
                or not self._marvin_detour_owner_is_current()):
            return dict(base, reason="marvin_local_bypass_ownership_invalid")
        if self.running is not True:
            return dict(base, reason="marvin_approach_runtime_not_running")
        if bypass_mode:
            zero = self._marvin_bridge_ready_and_stopped()
            if zero.get("ok") is not True or zero.get("status") != "READY":
                return dict(base, reason="marvin_local_bypass_bridge_not_stopped")
        behavior = getattr(self, "behavior_manager", None)
        approach = getattr(behavior, "execute_single_marvin_approach_step", None)
        robot = getattr(behavior, "robot", None)
        stop = getattr(robot, "stop", None)
        if not callable(approach) or not callable(stop):
            return dict(base, reason="marvin_approach_primitive_unavailable")
        worker = getattr(self, "lidar_worker", None)
        session = getattr(worker, "session", None)
        if worker is None or worker.running is not True or not isinstance(session, str) or not session:
            return dict(base, reason="marvin_approach_lidar_session_unavailable")
        base["producer_session"] = session
        get_lidar = getattr(getattr(self, "world_model", None), "get_lidar_obstacles", None)
        if not callable(get_lidar):
            return dict(base, reason="marvin_approach_lidar_read_unavailable")
        try:
            lidar = get_lidar(expected_session=session)
        except Exception as exc:
            return dict(base, reason="marvin_approach_lidar_read_failed", error=str(exc))
        if not _marvin_alignment_lidar_is_current(lidar, session):
            return dict(base, reason="marvin_approach_lidar_not_current", lidar=lidar)
        with self._state_lock:
            consumed = self._marvin_alignment_consumed_source_frame_stamps
            if source_frame_stamp_ns in consumed:
                return dict(base, reason="marvin_approach_observation_already_consumed")
            observation = self._marvin_alignment_observation
            tracker = (
                observation.get("opencv_tracker")
                if isinstance(observation, dict) else None
            )
            strict_observation = bool(
                isinstance(observation, dict)
                and observation.get("source_frame_stamp_ns") == source_frame_stamp_ns
                and observation.get("identity_confirmed") is True
                and (
                    observation.get("identity_source")
                    == "gemini_marvin_candidate_selection"
                    or (
                        observation.get("identity_source")
                        == "marvin_locked_tracker_continuity"
                        and observation.get("post_action_tracker_continuity") is True
                        and type(observation.get("post_action_source_frame_stamp_ns")) is int
                        and type(source_frame_stamp_ns) is int
                        and source_frame_stamp_ns
                        > observation["post_action_source_frame_stamp_ns"]
                    )
                )
                and observation.get("controller_state") == VISUAL_READY_TO_APPROACH
                and observation.get("controller_decision") == ("AVOID" if bypass_mode else "FORWARD")
                and isinstance(tracker, dict)
                and tracker.get("active") is True
                and tracker.get("matched") is True
                and isinstance(tracker.get("quality"), (int, float))
                and not isinstance(tracker.get("quality"), bool)
                and isinstance(tracker.get("threshold"), (int, float))
                and not isinstance(tracker.get("threshold"), bool)
                and tracker.get("quality") >= tracker.get("threshold")
                and tracker.get("source_frame_stamp_ns") == source_frame_stamp_ns
                and isinstance(tracker.get("bbox"), dict)
            )
            if not strict_observation:
                return dict(base, reason="marvin_approach_observation_not_authorized")
            if not dispatch_guard():
                self._reset_marvin_alignment_consensus()
                return dict(base, reason="marvin_motion_observation_stale_or_preempted")
            observed_standoff = observation.get("target_standoff")
            if (not isinstance(observed_standoff, dict)
                    or observed_standoff.get("ok") is not True
                    or observed_standoff.get("authority") != "target_bearing_lidar"
                    or (not bypass_mode and observed_standoff.get("target_range_association_trusted") is not True)
                    or observed_standoff.get("arrived_at_marvin") is not False):
                return dict(base, reason="marvin_approach_target_distance_not_authorized")
            # Recompute target range from current producer-bound points, not
            # from the cached arrival diagnostic. Local safety independently
            # evaluates the entire actual forward bound immediately afterward.
            standoff = self._marvin_v2_lidar_arrival(tracker, lidar)
            if (standoff.get("ok") is not True or standoff.get("arrived_at_marvin") is True
                    or (not bypass_mode and standoff.get("target_range_association_trusted") is not True)):
                return dict(base, reason="marvin_approach_target_standoff_veto", target_standoff=standoff)
            selection = None

            def validate_bypass(sample):
                with self._state_lock:
                    current_observation = self._marvin_alignment_observation or {}
                    current_tracker = current_observation.get("opencv_tracker") or {}
                    if (not dispatch_guard()
                            or current_observation.get("source_frame_stamp_ns") != source_frame_stamp_ns
                            or current_observation.get("identity_confirmed") is not True
                            or current_tracker.get("active") is not True
                            or current_tracker.get("matched") is not True
                            or current_tracker.get("source_frame_stamp_ns") != source_frame_stamp_ns
                            or type(current_tracker.get("quality")) not in (int, float)
                            or type(current_tracker.get("threshold")) not in (int, float)
                            or not math.isfinite(current_tracker["quality"])
                            or not math.isfinite(current_tracker["threshold"])
                            or current_tracker["quality"] < max(.80, current_tracker["threshold"])):
                        return {"accepted": False}
                association = self._marvin_v2_lidar_arrival(tracker, sample, current_action_jit=True)
                if association.get("ok") is not True or association.get("arrived_at_marvin"):
                    return {"accepted": False}
                selected = select_marvin_escape_action(sample, association, expected_session=session,
                    allow_strafe=local_detour_context["allow_strafe"],
                    previous_selection=local_detour_context.get("previous_selection"))
                accepted = (selected.get("action_type") == "BYPASS_FORWARD"
                    and selected.get("direction") == local_detour_context["selected_direction"]
                    and (selected.get("local_bypass") or {}).get("bypass_forward_permitted") is True)
                if accepted:
                    self._marvin_last_action_lidar_evidence = (session, sample["acquisition_sequence"])
                return dict(selected, accepted=accepted, target_association=association)

            if bypass_mode:
                selection = validate_bypass(lidar)
                if not selection.get("accepted"):
                    return dict(base, reason="marvin_local_bypass_jit_veto", local_detour=selection)
                effective_duration = float(duration)
                base.update(action_type="BYPASS_FORWARD", direction=selection["direction"], local_detour=selection)
            else:
                effective_duration = min(float(duration),
                    (standoff["target_distance_m"] - TARGET_STANDOFF_M) / float(linear_speed))
            base.update(requested_duration=float(duration), duration=effective_duration,
                        target_standoff=standoff)
            # As with a turn, consume immediately before the guarded dispatch.
            # An HTTP retry must never duplicate a possible forward movement.
            consumed.add(source_frame_stamp_ns)
            self._marvin_alignment_geometry_history = []
            self._marvin_alignment_consensus = []
            self._marvin_last_action_lidar_evidence = (session, standoff["acquisition_sequence"])
        mark_action = getattr(behavior, "mark_strict_v2_action_dispatched", None)
        if callable(mark_action) and mark_action(source_frame_stamp_ns, "forward") is not True:
            return dict(base, reason="marvin_approach_tracker_episode_not_current")
        base["execution_authorized"] = True
        try:
            approach_result = approach(expected_lidar_session=session,
                                       linear_speed=float(linear_speed), duration=effective_duration,
                                       target_tracker=None if bypass_mode else dict(tracker),
                                       **({"local_selection_validator": validate_bypass} if bypass_mode else {}),
                                       camera_model=getattr(self, "marvin_camera_model", None),
                                       target_range_validator=lambda tracker, sample: self._marvin_v2_lidar_arrival(
                                           tracker, sample, current_action_jit=True),
                                       dispatch_guard=dispatch_guard)
        except Exception as exc:
            approach_result = {"ok": False, "motion_executed": False, "error": str(exc)}
        try:
            stop_result = stop()
        except Exception as exc:
            stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        moved = bool(isinstance(approach_result, dict) and approach_result.get("motion_executed") is True)
        if isinstance(approach_result, dict):
            latest_standoff = approach_result.get("target_standoff")
            if isinstance(latest_standoff, dict) and latest_standoff.get("ok") is True:
                with self._state_lock:
                    self._marvin_last_action_lidar_evidence = (session, latest_standoff["acquisition_sequence"])
        stop_ok = isinstance(stop_result, dict) and stop_result.get("ok") is True
        if bypass_mode:
            zero = self._marvin_bridge_ready_and_stopped()
            stop_ok = stop_ok and zero.get("ok") is True and zero.get("status") == "READY"
            base.update(bridge_after_stop=zero, bridge_stop_confirmed=stop_ok,
                local_detour=approach_result.get("local_detour") or selection)
        forward = (approach_result or {}).get("forward_result") or {}
        if moved or (forward.get("transport_result") or {}).get("ok") is True:
            with self._state_lock:
                executed_bound = approach_result.get("duration", effective_duration)
                if (type(executed_bound) not in (int, float) or not math.isfinite(executed_bound)
                        or not 0 < executed_bound <= effective_duration):
                    executed_bound = effective_duration
                self._marvin_target_range_association.record_forward_bound(float(linear_speed), executed_bound)
        interrupted = forward.get("bounded_forward_invalidated") is True
        return dict(base, ok=moved and stop_ok, motion_executed=moved, actions_executed=int(moved),
                    interrupted=interrupted, interruption_reason=forward.get("reason") if interrupted else None,
                    source_stamp_consumed=source_frame_stamp_ns in consumed,
                    full_step_completed=moved and stop_ok,
                    actual_confirmed_run_duration_seconds=None,
                    approach_result=approach_result, stop_result=stop_result,
                    reason=("marvin_approach_step_complete" if moved and stop_ok else
                            (approach_result.get("reason", "marvin_approach_step_failed")
                             if isinstance(approach_result, dict) else "marvin_approach_step_failed")))

    def confirm_find_marvin_identity(self, *, confirm=False):
        """Explicitly confirm one fresh Marvin Preview without motion."""
        if confirm is not True:
            return {
                "ok": False,
                "confirmed": False,
                "reason": "explicit_identity_confirmation_required",
                "motion_executed": False,
            }
        confirmer = getattr(
            self.behavior_manager,
            "confirm_marvin_identity_from_preview",
            None,
        )
        if not callable(confirmer):
            return {
                "ok": False,
                "confirmed": False,
                "reason": "marvin_identity_confirmation_unavailable",
                "motion_executed": False,
            }
        try:
            result = confirmer()
        except Exception as exc:
            return {
                "ok": False,
                "confirmed": False,
                "reason": "marvin_identity_confirmation_exception",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "motion_executed": False,
            }
        if not isinstance(result, dict):
            return {
                "ok": False,
                "confirmed": False,
                "reason": "marvin_identity_confirmation_result_malformed",
                "motion_executed": False,
            }
        if result.get("motion_executed") is not False:
            return {
                "ok": False,
                "confirmed": False,
                "reason": "identity_confirmation_motion_invariant_failed",
                "motion_executed": False,
            }
        return result

    def _stop_lidar(self):
        with self._lidar_lifecycle_lock:
            if self._lidar_stopped:
                return
            self._lidar_stopped = True
            if self.lidar_worker is not None:
                try:
                    self.lidar_worker.stop()
                except Exception as exc:
                    self._lidar_error = str(exc)

    def _lidar_status(self):
        worker = self.lidar_worker
        session = worker.session if worker is not None else None
        state = self.world_model.get_lidar_obstacles(expected_session=session)
        running = bool(worker is not None and worker.running)
        if self._lidar_error or not running:
            state.update(available=False, valid=False)
            if state.get("reason") == "fresh":
                state["reason"] = "worker_not_running"
        geometry = state.get("local_motion_geometry")
        geometry_sectors = (
            geometry.get("sectors") if isinstance(geometry, dict) else None
        )
        required_sectors_valid = bool(
            isinstance(geometry, dict)
            and geometry.get("valid") is True
            and isinstance(geometry_sectors, dict)
            and all(
                isinstance(geometry_sectors.get(name), dict)
                and geometry_sectors[name].get("valid_sample_count", 0)
                >= MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
                for name, _, _ in OCTANT_SECTORS
            )
        )
        return {
            "running": running,
            "producer_session": session,
            "session_matches": bool(
                session and state.get("producer_session") == session
            ),
            "acquisition_sequence": worker.sequence if worker is not None else 0,
            "available": state.get("available", False),
            "valid": state.get("valid", False),
            "reason": state.get("reason"),
            "effective_age_seconds": state.get("effective_age_seconds"),
            "local_motion_geometry_valid": bool(
                isinstance(geometry, dict) and geometry.get("valid") is True
            ),
            "required_sectors_valid": required_sectors_valid,
            "front_state": (
                state.get("sectors", {}).get("front", {}).get("state", "UNKNOWN")
                if state.get("valid") else "UNKNOWN"
            ),
            "last_error": self._lidar_error or (worker.last_error if worker is not None else None),
        }

    def get_find_marvin_admission_snapshot(self):
        """Return one read-only, coherent safety/status view for Find Marvin.

        The forward-interlock health portion is evaluated against the exact
        LiDAR state copied for this snapshot. Its cached monitor result is
        included separately for diagnostics, but is not substituted for the
        current sample.
        """
        bridge_status = None
        bridge_error = None
        bridge_status_reader = getattr(self.robot_client, "status", None)
        if callable(bridge_status_reader):
            try:
                bridge_status = bridge_status_reader()
            except Exception as exc:
                bridge_error = f"{type(exc).__name__}: {exc}"
        else:
            bridge_error = "robot_bridge_status_unavailable"

        with self._state_lock:
            worker = self.lidar_worker
            expected_session = worker.session if worker is not None else None
            try:
                lidar_state = self.world_model.get_lidar_obstacles(
                    expected_session=expected_session,
                )
            except Exception as exc:
                lidar_state = unavailable_state(
                    "world_model_read_error", expected_session,
                    getattr(worker, "sequence", 0),
                )
                lidar_state["worker_error"] = f"{type(exc).__name__}: {exc}"

            worker_running = bool(worker is not None and worker.running)
            worker_error = (
                self._lidar_error
                or getattr(worker, "last_error", None)
                or lidar_state.get("worker_error")
            )
            if worker_error or not worker_running:
                lidar_state.update(available=False, valid=False)
                if lidar_state.get("reason") == "fresh":
                    lidar_state["reason"] = "worker_not_running"

            geometry = lidar_state.get("local_motion_geometry")
            geometry_sectors = (
                geometry.get("sectors") if isinstance(geometry, dict) else None
            )
            required_sectors_valid = bool(
                isinstance(geometry, dict)
                and geometry.get("valid") is True
                and isinstance(geometry_sectors, dict)
                and all(
                    isinstance(geometry_sectors.get(name), dict)
                    and geometry_sectors[name].get("valid_sample_count", 0)
                    >= MINIMUM_VALID_SAMPLES_PER_REQUIRED_SECTOR
                    for name, _, _ in OCTANT_SECTORS
                )
            )
            sectors = lidar_state.get("sectors")
            front = sectors.get("front") if isinstance(sectors, dict) else None
            front_state = (
                front.get("state", "UNKNOWN")
                if lidar_state.get("valid") is True and isinstance(front, dict)
                else "UNKNOWN"
            )
            lidar = {
                "running": worker_running,
                "available": lidar_state.get("available", False),
                "valid": lidar_state.get("valid", False),
                "reason": lidar_state.get("reason"),
                "age_seconds": lidar_state.get("effective_age_seconds"),
                "acquisition_sequence": lidar_state.get(
                    "acquisition_sequence", getattr(worker, "sequence", None),
                ),
                "producer_session": lidar_state.get("producer_session"),
                "expected_session": expected_session,
                "session_matches": bool(
                    expected_session
                    and lidar_state.get("producer_session") == expected_session
                ),
                "worker_error": worker_error,
                "front_state": front_state,
                "local_motion_geometry_valid": bool(
                    isinstance(geometry, dict) and geometry.get("valid") is True
                ),
                "required_sectors_valid": required_sectors_valid,
            }

            cached_interlock = (
                self.forward_interlock.status()
                if self.forward_interlock is not None
                else {"configured": False, "reason": "not_configured"}
            )
            interlock_session = cached_interlock.get("producer_session")
            interlock_session_matches = bool(
                expected_session
                and interlock_session == expected_session
                and lidar.get("session_matches") is True
            )
            lidar_allows_forward = False
            lidar_interlock_reason = "missing_lidar_state"
            if expected_session:
                lidar_allows_forward, lidar_interlock_reason = evaluate_lidar_state(
                    lidar_state, expected_session,
                )
            interlock_operational = bool(
                cached_interlock.get("configured") is True
                and cached_interlock.get("monitor_running") is True
                and interlock_session_matches
            )
            forward_permitted = interlock_operational and lidar_allows_forward
            if cached_interlock.get("configured") is not True:
                interlock_reason = "not_configured"
            elif cached_interlock.get("monitor_running") is not True:
                interlock_reason = "monitor_not_running"
            elif not interlock_session_matches:
                interlock_reason = "producer_session_mismatch"
            else:
                interlock_reason = lidar_interlock_reason
            interlock = {
                "configured": cached_interlock.get("configured") is True,
                "monitor_running": cached_interlock.get("monitor_running") is True,
                "age_seconds": lidar.get("age_seconds"),
                "reason": interlock_reason,
                "front_state": front_state,
                "forward_permitted": forward_permitted,
                "producer_session": interlock_session,
                "session_matches": interlock_session_matches,
                "active_forward": cached_interlock.get("active_forward"),
                "pending_forward": cached_interlock.get("pending_forward"),
                "monitor_reported_reason": cached_interlock.get("reason"),
                "monitor_reported_age_seconds": cached_interlock.get(
                    "effective_age_seconds"
                ),
                "monitor_reported_forward_permitted": cached_interlock.get(
                    "forward_permitted"
                ),
                "monitor_inhibited": cached_interlock.get("inhibited"),
                "last_stop_error": cached_interlock.get("last_stop_error"),
            }

            manager = self.mission_manager
            active = manager.get_active_mission()
            active_mission = active.to_dict() if active is not None else None
            queue = manager.get_queue()
            runtime_snapshot = {
                "running": self.running is True,
                "state": self.world_model.robot_state.get(
                    "runtime_state", "UNKNOWN",
                ),
                "last_error": self.last_error,
                "active_mission": active_mission,
                "queue": queue,
                "queue_count": len(queue),
            }
            bridge = bridge_status if isinstance(bridge_status, dict) else {}
            motion_value = bridge.get("motion")
            motion = motion_value if isinstance(motion_value, dict) else {}
            bridge_snapshot = {
                "connected": bridge.get("ok") is True,
                "status": bridge.get("status"),
                "ros_ready": bridge.get("ros_ready"),
                "linear_x": motion.get("linear_x"),
                "angular_z": motion.get("angular_z"),
                "streaming": motion.get("streaming"),
                "error": bridge_error or bridge.get("ros_error") or bridge.get("error"),
            }
            snapshot = {
                "ok": True,
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime_snapshot,
                "lidar": lidar,
                "forward_interlock": interlock,
                "bridge": bridge_snapshot,
            }
            snapshot["reasons"] = self._find_marvin_admission_reasons(snapshot)
            snapshot["admission_ready"] = not snapshot["reasons"]
            return snapshot

    @staticmethod
    def _find_marvin_admission_reasons(snapshot):
        """Fail-closed Find Marvin admission rules over one snapshot."""
        runtime = snapshot.get("runtime", {})
        lidar = snapshot.get("lidar", {})
        interlock = snapshot.get("forward_interlock", {})
        bridge = snapshot.get("bridge", {})
        reasons = []

        def stopped_number(value):
            return bool(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value == 0
            )

        requirements = (
            (runtime.get("running") is True, "runtime is not running"),
            (runtime.get("state") == "IDLE", "runtime is not IDLE"),
            (runtime.get("active_mission") is None, "another mission is active"),
            (runtime.get("queue_count") == 0, "mission queue is not empty"),
            (runtime.get("last_error") is None, "runtime reports an error"),
            (lidar.get("running") is True, "LiDAR worker is not running"),
            (lidar.get("available") is True, "LiDAR is unavailable"),
            (lidar.get("valid") is True, "LiDAR is invalid"),
            (lidar.get("reason") == "fresh", "LiDAR is not fresh"),
            (lidar.get("front_state") in {"CLEAR", "CAUTION", "BLOCKED"}, "LiDAR front sector is unavailable"),
            (lidar.get("session_matches") is True, "LiDAR producer session does not match"),
            (lidar.get("local_motion_geometry_valid") is True, "LiDAR local geometry is invalid"),
            (lidar.get("required_sectors_valid") is True, "LiDAR required sectors are incomplete"),
            (interlock.get("configured") is True, "forward interlock is not configured"),
            (interlock.get("monitor_running") is True, "forward interlock monitor is not running"),
            (interlock.get("forward_permitted") is True, "forward motion is not permitted"),
            (interlock.get("reason") == "fresh_clear", "forward interlock is not fresh_clear"),
            (interlock.get("session_matches") is True, "forward interlock LiDAR session does not match"),
            (interlock.get("active_forward") is False, "forward motion is active"),
            (interlock.get("pending_forward") is False, "forward motion is pending"),
            (interlock.get("last_stop_error") is None, "forward interlock STOP failed"),
            (bridge.get("connected") is True, "Robot Bridge status unavailable"),
            (bridge.get("status") == "READY", "Robot Bridge is not READY"),
            (bridge.get("ros_ready") is True, "Robot Bridge ROS is not ready"),
            (stopped_number(bridge.get("linear_x")), "Robot Bridge linear motion is not zero"),
            (stopped_number(bridge.get("angular_z")), "Robot Bridge angular motion is not zero"),
            (bridge.get("streaming") is False, "Robot Bridge streaming motion is active"),
        )
        reasons.extend(reason for passed, reason in requirements if not passed)
        return reasons

    def submit_text(self, user_text: str):
        """
        Convert natural-language text into an intent and submit its mission.

        This method preserves the existing provider-based cognitive pipeline.
        """
        command = str(user_text or "").strip()

        if not command:
            raise ValueError("Command text cannot be empty.")

        intent = self.provider.get_intent(command)
        mission = self.submit_intent(intent)

        return {
            "command": command,
            "intent": intent,
            "mission": mission.to_dict(),
        }

    def submit_intent(self, intent: Dict[str, Any]):
        """
        Submit a parsed intent to the persistent MissionManager.

        STOP is handled as an immediate runtime-level preemption. It cancels
        the active mission, clears queued missions, commands the Robot Bridge
        to stop, and prevents an in-flight behavior cycle from restoring the
        previous mission state.
        """
        if not isinstance(intent, dict):
            raise TypeError("Intent must be a dictionary.")

        intent_name = str(
            intent.get("intent", "UNKNOWN")
        ).strip().upper()

        with self._state_lock:
            if intent_name != "STOP" and getattr(self, "_marvin_live_proof_owner", None) is not None:
                raise ValueError("marvin_live_proof_owns_runtime")
            self._invalidate_marvin_live_proof("stop_intent" if intent_name == "STOP" else "normal_mission_submission")
            if intent_name == "STOP":
                self._control_generation += 1
                self._reset_marvin_alignment_consensus()

                mission = self.mission_manager.handle_intent(
                    intent
                )

                try:
                    robot_result = self.robot_client.stop()

                    stop_ok = bool(
                        robot_result.get("ok")
                    )

                    stop_error = None

                except Exception as exc:
                    robot_result = {
                        "ok": False,
                        "error": str(exc),
                    }

                    stop_ok = False
                    stop_error = str(exc)

                self.last_result = {
                    "ok": stop_ok,
                    "executed": True,
                    "completed": True,
                    "behavior": "STOP",
                    "state": "STOPPED",
                    "reason": (
                        "Robot stop command sent and all "
                        "missions were cancelled."
                    ),
                    "robot_result": robot_result,
                }

                self.last_error = stop_error
                self.tracking_state = empty_tracking_state(
                    state="STOPPED",
                )
                self._last_runtime_state = "STOPPED"

                self.world_model.update_robot_state(
                    runtime_state="STOPPED",
                    mission=None,
                    mission_queue=[],
                    last_behavior_result=self.last_result,
                    last_cancelled_mission=mission.to_dict(),
                )

                return mission

            mission = self.mission_manager.handle_intent(
                intent
            )

            self.world_model.update_robot_state(
                runtime_state="MISSION_ACCEPTED",
                mission=mission.to_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

            return mission

    def run_once(self):
        """
        Execute one active mission, if available.

        Returns None while idle. Otherwise returns the BehaviorManager result.
        """
        with self._state_lock:
            if getattr(self, "_marvin_live_proof_owner", None) is not None:
                return None
            mission = self.mission_manager.get_active_mission()

            if mission is None:
                mission = self.mission_manager.start_next_mission()

            if mission is None:
                self._set_runtime_state("IDLE")
                return None

            mission_id = mission.mission_id
            self._invalidate_marvin_live_proof("normal_mission_execution")
            control_generation = self._control_generation
            self._behavior_execution_generation = control_generation
            self._behavior_execution_thread_id = threading.get_ident()

            self.world_model.update_robot_state(
                runtime_state="EXECUTING",
                mission=mission.to_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

        execution_error = None

        try:
            if self._is_normal_marvin_find_mission(mission):
                result = self._execute_normal_marvin_find_mission(
                    mission,
                    control_generation=control_generation,
                )
            else:
                result = self.behavior_manager.execute(mission)

            if not isinstance(result, dict):
                raise TypeError(
                    "BehaviorManager.execute() must return a dictionary."
                )

        except Exception as exc:
            result = {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": str(exc),
            }

            execution_error = str(exc)

            try:
                self.robot_client.stop()
            except Exception:
                pass

        with self._state_lock:
            if control_generation != self._control_generation:
                # STOP arrived while this bounded behavior was executing.
                # submit_intent(STOP) already cancelled all missions, stopped
                # the robot, and persisted the authoritative stopped state.
                if self._behavior_execution_generation == control_generation:
                    self._behavior_execution_generation = None
                    self._behavior_execution_thread_id = None
                return self.last_result

            self.last_result = result
            self.last_error = execution_error
            self._behavior_execution_generation = None
            self._behavior_execution_thread_id = None
            self.tracking_state = build_tracking_state(
                result,
                previous=self.tracking_state,
            )

            if self._is_normal_marvin_find_mission(mission) and result.get("completed") is True:
                # Preserve terminal diagnostics, but the V2 episode is over.
                self.tracking_state["active"] = False

            active = self.mission_manager.get_active_mission()

            if active and active.mission_id == mission_id:
                behavior_completed = result.get("completed")

                # Backward compatibility:
                # Existing bounded behaviors may not include a completed
                # field. Those behaviors still complete after one execution.
                if behavior_completed is None:
                    behavior_completed = True

                if result.get("ok") and behavior_completed:
                    finished_mission = (
                        self.mission_manager.complete_active_mission()
                    )
                    # The mission result/history retain the terminal
                    # outcome; runtime_state describes lifecycle readiness.
                    runtime_state = "MISSION_ACTIVE"

                elif result.get("ok"):
                    # The behavior completed one safe bounded step, but the
                    # mission itself remains active. The persistent runtime
                    # will invoke it again during the next loop cycle.
                    finished_mission = None
                    runtime_state = "MISSION_ACTIVE"

                else:
                    finished_mission = (
                        self.mission_manager.cancel_active_mission(
                            speech=result.get(
                                "reason",
                                "Mission execution failed.",
                            )
                        )
                    )
                    runtime_state = "MISSION_FAILED"
            else:
                finished_mission = mission
                runtime_state = "MISSION_INTERRUPTED"

            next_mission = self.mission_manager.get_active_mission()

            if (
                next_mission is None
                and not self.mission_manager.get_queue()
                and execution_error is None
            ):
                runtime_state = "IDLE"
            elif next_mission is None and self.mission_manager.get_queue():
                # Queued work remains available; do not expose a transient
                # idle window before the existing scheduler starts it.
                runtime_state = "MISSION_ACCEPTED"

            world_updates = {
                "runtime_state": runtime_state,
                "mission": (
                    next_mission.to_dict()
                    if next_mission is not None
                    else None
                ),
                "mission_queue": self.mission_manager.get_queue(),
                "last_behavior_result": result,
                "tracking": self.tracking_state,
            }

            if finished_mission is not None:
                world_updates["last_completed_mission"] = (
                    finished_mission.to_dict()
                )

            self.world_model.update_robot_state(
                **world_updates
            )

        return result

    def run_forever(self):
        """
        Run the persistent mission-processing loop until shutdown.
        """
        self.running = True
        self.started_at = time.time()

        try:
            self.world_model.update_robot_state(
                runtime_state="STARTING",
                cognitive_runtime_running=True,
                mission=self._active_mission_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

            print("============================================")
            print(" Mini Pupper 2 Cognitive Runtime")
            print("============================================")
            print("State:   RUNNING")
            print("Mode:    Persistent mission processing")
            print("Stop:    Ctrl+C")
            print()

            self._start_lidar()
            self._retain_marvin_diagnostic("start")
            if self.forward_interlock is not None:
                self.forward_interlock.start()
            while self.running:
                self.run_once()
                time.sleep(self.loop_interval)

        finally:
            self.running = False

            try:
                self.robot_client.stop()
            except Exception:
                pass

            self._stop_lidar()
            self._retain_marvin_diagnostic("stop")
            if self.forward_interlock is not None:
                self.forward_interlock.stop()

            self.world_model.update_robot_state(
                runtime_state="STOPPED",
                cognitive_runtime_running=False,
                mission=self._active_mission_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

            print()
            print("Cognitive runtime stopped.")

    def stop(self):
        """
        Request a clean runtime shutdown.
        """
        self.running = False
        self._invalidate_marvin_live_proof("runtime_shutdown")
        self._retain_marvin_diagnostic("stop")
        self._stop_lidar()
        if self.forward_interlock is not None:
            self.forward_interlock.stop()

    def get_status_summary(self):
        """Dashboard view without copying retained history or diagnostics.

        Reporting is read-only. Admission and motion continue to use their
        existing independent evidence paths.
        """
        from runtime_reporting import status_summary

        with self._state_lock:
            active = self.mission_manager.get_active_mission()
            queue = self.mission_manager.mission_queue
            return status_summary({
                "ok": True,
                "service": "mini_pupper_cognitive_runtime",
                "running": self.running,
                "runtime_state": self.world_model.robot_state.get("runtime_state", "UNKNOWN"),
                "uptime_seconds": max(0.0, time.time() - self.started_at) if self.started_at is not None else None,
                "active_mission": active.to_dict() if active is not None else None,
                "queue": [mission.to_dict() for mission in queue[:20]],
                "queue_count": len(queue),
                "history_count": len(self.mission_manager.mission_history),
                "last_result": self.last_result,
                "tracking": self.tracking_state,
                "last_error": self.last_error,
                "lidar_perception": self._lidar_status(),
                "forward_interlock": self.forward_interlock.status() if self.forward_interlock is not None else {
                    "configured": False, "reason": "not_configured"},
            })

    def get_status(self):
        """
        Return a serializable runtime status snapshot.
        """
        with self._state_lock:
            uptime_seconds = None

            if self.started_at is not None:
                uptime_seconds = max(
                    0.0,
                    time.time() - self.started_at,
                )

            active = self.mission_manager.get_active_mission()

            return {
                "ok": True,
                "service": "mini_pupper_cognitive_runtime",
                "running": self.running,
                "runtime_state": self.world_model.robot_state.get(
                    "runtime_state",
                    "UNKNOWN",
                ),
                "uptime_seconds": uptime_seconds,
                "active_mission": (
                    active.to_dict()
                    if active is not None
                    else None
                ),
                "queue": self.mission_manager.get_queue(),
                "history_count": len(
                    self.mission_manager.mission_history
                ),
                "last_result": self.last_result,
                "marvin_progress_diagnostics": self._retain_marvin_diagnostic("snapshot"),
                "marvin_odometry_diagnostics": self._retain_marvin_diagnostic("odometry_snapshot"),
                "tracking": dict(self.tracking_state),
                "last_error": self.last_error,
                "lidar_perception": self._lidar_status(),
                "forward_interlock": (
                    self.forward_interlock.status()
                    if self.forward_interlock is not None
                    else {"configured": False, "reason": "not_configured"}
                ),
            }

    def _active_mission_dict(self):
        active = self.mission_manager.get_active_mission()

        if active is None:
            return None

        return active.to_dict()

    def _set_runtime_state(self, runtime_state: str):
        """
        Persist runtime state only when it changes.

        This prevents the idle loop from rewriting the World Model four times
        per second.
        """
        if runtime_state == self._last_runtime_state:
            return

        self._last_runtime_state = runtime_state

        self.world_model.update_robot_state(
            runtime_state=runtime_state,
            cognitive_runtime_running=self.running,
            mission=self._active_mission_dict(),
            mission_queue=self.mission_manager.get_queue(),
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the persistent Mini Pupper 2 cognitive mission runtime."
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one available mission cycle and exit.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the initial runtime status and exit.",
    )

    args = parser.parse_args()

    runtime = CognitiveRuntime()

    def handle_shutdown(signum, frame):
        del signum
        del frame
        runtime.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if args.status:
        print(json.dumps(runtime.get_status(), indent=2))
        return

    if args.once:
        result = runtime.run_once()
        print(json.dumps(result, indent=2))
        return

    runtime.run_forever()


if __name__ == "__main__":
    main()
