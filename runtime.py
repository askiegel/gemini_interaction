#!/usr/bin/env python3

import argparse
import json
import math
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from behavior_manager import BehaviorManager
from guarded_turn_policy import ROTATIONAL_SWEPT_FOOTPRINT
from marvin_arrival_policy import evaluate_marvin_visual_arrival
from marvin_pursuit_state import (
    FIND_CENTER_TOLERANCE_PIXELS,
    VISUAL_READY_TO_ALIGN,
    VISUAL_READY_TO_APPROACH,
    evaluate_marvin_pursuit_state,
)
from config import load_config
from lidar_perception import LidarPerceptionWorker, unavailable_state
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
            evidence = observer()
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
            identity_confirmed=preview.get("identity_confirmed") is True,
            identity_source=identity_source,
            proposal_label=preview.get("proposal_label"),
            proposal_confidence=preview.get("proposal_confidence"),
            opencv_tracker=dict(tracker) if isinstance(tracker, dict) else None,
            session_continuity_used=(identity_source == "marvin_session_continuity"),
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
        arrival = evaluate_marvin_visual_arrival(preview)
        if isinstance(arrival, dict) and arrival.get("arrived_at_marvin") is True:
            decision = "ARRIVED"
        elif state == VISUAL_READY_TO_ALIGN:
            error = pursuit.get("horizontal_error")
            decision = "TURN_LEFT" if error < 0 else "TURN_RIGHT" if error > 0 else "BLOCKED"
        elif state == VISUAL_READY_TO_APPROACH:
            decision = "FORWARD"
        elif state == "SEARCHING":
            decision = "SEARCH"
        else:
            decision = "BLOCKED"
        result = dict(base, **common, ok=True, reason=reason,
                      controller={"state": state, "decision": decision,
                                  "reason": reason,
                                  "center_tolerance_pixels": FIND_CENTER_TOLERANCE_PIXELS},
                      arrival=arrival)
        geometry_continuity = self._update_marvin_alignment_geometry_continuity(
            result, tracker, state, decision,
        )
        result["geometry_continuity"] = geometry_continuity
        if geometry_continuity["accepted"] is True:
            self._update_marvin_alignment_consensus(
                result, tracker, state, decision,
            )
        else:
            # A raw observation remains visible to the operator, but a seed
            # geometry discontinuity is never allowed to contribute to the
            # physical-alignment consensus window.
            with self._state_lock:
                self._marvin_alignment_consensus = []
                self._marvin_alignment_observation = None
        return result

    MARVIN_ALIGNMENT_CONSENSUS_WINDOW = 3
    MARVIN_ALIGNMENT_MAX_CENTER_SPREAD_PIXELS = 30.0
    MARVIN_ALIGNMENT_GEOMETRY_HISTORY_WINDOW = 3
    MARVIN_ALIGNMENT_MAX_SEED_CENTER_DELTA_PIXELS = 30.0

    def _reset_marvin_alignment_consensus(self):
        """Clear all pending alignment evidence after a failed V2 observation."""
        with self._state_lock:
            self._marvin_alignment_observation = None
            self._marvin_alignment_consensus = []
            self._marvin_alignment_geometry_history = []

    @staticmethod
    def _marvin_alignment_consensus_sample(result, tracker, state, decision):
        """Extract one strictly valid current observation for turn consensus."""
        if (
            not isinstance(result, dict)
            or result.get("identity_confirmed") is not True
            or result.get("identity_source")
            != "gemini_marvin_candidate_selection"
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

    def _update_marvin_alignment_consensus(self, result, tracker, state, decision):
        """Authorize only the newest sample of a stable strict turn window."""
        sample = self._marvin_alignment_consensus_sample(
            result, tracker, state, decision,
        )
        with self._state_lock:
            alignment_observation = None
            previous = list(getattr(self, "_marvin_alignment_consensus", []))
            if sample is None:
                consensus = []
            elif not previous:
                consensus = [sample]
            else:
                last = previous[-1]
                if (
                    sample["identity_source"] != last.get("identity_source")
                    or sample["direction"] != last.get("direction")
                    or sample["source_frame_stamp_ns"]
                    <= last.get("source_frame_stamp_ns", -1)
                ):
                    # A discontinuity is not evidence for a new window: the
                    # next authorization requires three entirely new samples.
                    consensus = []
                else:
                    consensus = (previous + [sample])[(-self.MARVIN_ALIGNMENT_CONSENSUS_WINDOW):]
                    centers = [entry["center_x"] for entry in consensus]
                    if (
                        len(consensus) == self.MARVIN_ALIGNMENT_CONSENSUS_WINDOW
                        and max(centers) - min(centers)
                        > self.MARVIN_ALIGNMENT_MAX_CENTER_SPREAD_PIXELS
                    ):
                        consensus = []
                    elif len(consensus) == self.MARVIN_ALIGNMENT_CONSENSUS_WINDOW:
                        alignment_observation = {
                            "source_frame_stamp_ns": sample["source_frame_stamp_ns"],
                            "identity_confirmed": result["identity_confirmed"],
                            "identity_source": result["identity_source"],
                            "opencv_tracker": dict(tracker),
                            "controller_state": state,
                            "controller_decision": decision,
                        }
            self._marvin_alignment_consensus = consensus
            self._marvin_alignment_observation = alignment_observation

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
            if consume_one_shot:
                with self._state_lock:
                    if self._marvin_autonomous_run_consumed:
                        return dict(base, reason="marvin_autonomous_run_already_consumed")
                    self._marvin_autonomous_run_consumed = True
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
                "mission_route": "bounded_marvin_autonomous",
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

    def _execute_normal_marvin_find_mission_locked(
        self, mission, *, control_generation=None,
    ):
        """Delegate a normal Find-Marvin mission to the reviewed controller.

        The controller remains the only pursuit implementation. This method
        owns bounded mission continuation, with a separate finite room-scan
        episode allowance and the existing pursuit episode allowance. Scan
        progress itself remains mission-scoped in BehaviorManager.
        """
        mission_id = getattr(mission, "mission_id", None)
        behavior = getattr(self, "behavior_manager", None)
        begin_scan = getattr(behavior, "begin_find_marvin_room_scan", None)
        scan_enabled = callable(begin_scan)
        if scan_enabled:
            try:
                begin_scan(mission_id)
            except Exception as exc:
                return {
                    "ok": False, "completed": True,
                    "arrived_at_marvin": False,
                    "mission_outcome": "safe_failure",
                    "state": "FIND_MARVIN_FAILED",
                    "mission_id": mission_id,
                    "reason": "find_marvin_scan_state_initialization_failed",
                    "error": str(exc),
                }
        if control_generation is None:
            with self._state_lock:
                control_generation = self._control_generation

        episode_results = []
        total_actions = 0
        total_stale_replans = 0
        any_motion = False
        last_controller = None

        max_episodes = (
            self.FIND_MARVIN_SCAN_MAX_EPISODES + self.FIND_MARVIN_MAX_EPISODES
            if scan_enabled else self.FIND_MARVIN_MAX_EPISODES
        )
        scan_episodes_used = 0
        pursuit_episodes_used = 0

        def result_base():
            return {
                "action": "bounded_find_marvin_autonomous_run",
                "execution_authorized": bool(episode_results),
                "behavior": "FIND_OBJECT",
                "target": "marvin",
                "mission_route": "bounded_marvin_autonomous",
                "mission_id": mission_id,
                "episodes_executed": len(episode_results),
                "max_episodes": max_episodes,
                "max_actions": self.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS,
                "total_actions_executed": total_actions,
                "actions_executed": total_actions,
                "stale_replans": total_stale_replans,
                "episode_results": list(episode_results),
                "controller_result": last_controller,
                "motion_executed": any_motion,
            }

        episode_number = 0
        while scan_episodes_used < self.FIND_MARVIN_SCAN_MAX_EPISODES or pursuit_episodes_used < self.FIND_MARVIN_MAX_EPISODES:
            scan_snapshot = (
                behavior._room_scan_snapshot()
                if scan_enabled and callable(getattr(behavior, "_room_scan_snapshot", None))
                else None
            )
            scanning = bool(scan_snapshot and scan_snapshot.get("scan_active") is True)
            if scanning and scan_episodes_used >= self.FIND_MARVIN_SCAN_MAX_EPISODES:
                return dict(
                    result_base(), ok=True, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_incomplete",
                    state="FIND_MARVIN_SAFE_INCOMPLETE",
                    reason="find_marvin_scan_episode_limit_reached",
                    completion_reason="find_marvin_scan_episode_limit_reached",
                )
            if not scanning and pursuit_episodes_used >= self.FIND_MARVIN_MAX_EPISODES:
                return dict(
                    result_base(), ok=True, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_incomplete",
                    state="FIND_MARVIN_SAFE_INCOMPLETE",
                    reason="find_marvin_mission_episode_limit_reached",
                    completion_reason="find_marvin_mission_episode_limit_reached",
                )
            episode_number += 1
            if not self._marvin_mission_context_is_current(
                mission, control_generation,
            ):
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="preempted",
                    state="FIND_MARVIN_PREEMPTED",
                    reason="find_marvin_mission_preempted",
                )

            episode = self._execute_bounded_find_marvin_episode(
                max_actions=self.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS,
                consume_one_shot=False,
                require_fresh_gemini=True,
            )
            if not isinstance(episode, dict):
                episode = {"ok": False, "reason": "marvin_episode_result_malformed"}
            controller = episode.get("controller_result")
            episode_record = {
                "episode": episode_number,
                "result": episode,
            }
            episode_results.append(episode_record)
            if scanning:
                scan_episodes_used += 1
            else:
                pursuit_episodes_used += 1
            last_controller = controller if isinstance(controller, dict) else None

            count = episode.get("actions_executed", 0)
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= self.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS:
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason="find_marvin_episode_action_count_invalid",
                )
            total_actions += count
            episode_stale_replans = episode.get("controller_result", {}).get(
                "stale_replans", 0,
            ) if isinstance(episode.get("controller_result"), dict) else 0
            if (
                not isinstance(episode_stale_replans, int)
                or isinstance(episode_stale_replans, bool)
                or episode_stale_replans < 0
            ):
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason="find_marvin_episode_stale_replan_count_invalid",
                )
            total_stale_replans += episode_stale_replans
            any_motion = any_motion or episode.get("motion_executed") is True

            if not isinstance(controller, dict) or episode.get("ok") is not True:
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason=episode.get("reason", "find_marvin_controller_failed"),
                    controller_reason=(controller.get("reason") if isinstance(controller, dict) else None),
                )

            if (
                controller.get("reason") == "arrived_at_marvin"
                and controller.get("arrived_at_marvin") is True
                and controller.get("completed") is True
            ):
                return dict(
                    result_base(), ok=True, completed=True,
                    arrived_at_marvin=True,
                    mission_outcome="arrived_at_marvin",
                    state="ARRIVED_AT_MARVIN",
                    reason="arrived_at_marvin",
                    completion_reason="arrived_at_marvin",
                )

            if controller.get("reason") != "find_marvin_action_limit_reached":
                if controller.get("reason") == "find_marvin_search_target_acquired":
                    bridge_status = self._marvin_bridge_ready_and_stopped()
                    if not isinstance(bridge_status, dict) or bridge_status.get("ok") is not True:
                        return dict(
                            result_base(), ok=False, completed=True,
                            arrived_at_marvin=False,
                            mission_outcome="safe_failure",
                            state="FIND_MARVIN_FAILED",
                            reason="find_marvin_acquisition_bridge_not_stopped",
                            bridge_status=bridge_status,
                        )
                    continue
                if controller.get("reason") == "find_marvin_post_turn_frame_preempted":
                    return dict(
                        result_base(), ok=False, completed=True,
                        arrived_at_marvin=False, mission_outcome="preempted",
                        state="FIND_MARVIN_PREEMPTED",
                        reason="find_marvin_mission_preempted",
                    )
                if (
                    str(controller.get("reason", "")).startswith(
                        "find_marvin_post_turn_frame_"
                    )
                    or controller.get("reason") in {
                        "find_marvin_scan_source_frame_baseline_missing",
                        "find_marvin_post_turn_source_frame_baseline_missing",
                    }
                ):
                    return dict(
                        result_base(), ok=True, completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason=controller.get("reason"),
                        completion_reason=controller.get("reason"),
                    )
                if (
                    controller.get("reason") == "find_marvin_clearance_wait_timeout"
                    and self._marvin_clearance_timeout_is_safe(controller)
                ):
                    return dict(
                        result_base(),
                        ok=True,
                        completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason="find_marvin_clearance_wait_timeout",
                        completion_reason="find_marvin_clearance_wait_timeout",
                        clearance_wait=controller.get("clearance_wait"),
                    )
                if (
                    controller.get("reason") == "find_marvin_search_complete"
                    and self._marvin_search_exhaustion_is_safe(controller)
                ):
                    return dict(
                        result_base(),
                        ok=True,
                        completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason="find_marvin_search_exhausted",
                        completion_reason="find_marvin_search_exhausted",
                    )
                if controller.get("reason") == "find_marvin_arrival_confirmation_not_independent":
                    return dict(
                        result_base(), ok=True, completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason=controller["reason"],
                        completion_reason=controller["reason"],
                    )
                if controller.get("reason") == "marvin_local_progress_terminal":
                    terminal = controller.get("local_progress_terminal")
                    progress = controller.get("local_progress_result")
                    common = dict(
                        result_base(),
                        completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        local_progress_terminal=terminal,
                        local_progress_result=progress,
                    )
                    if terminal == "LOCAL_PROGRESS_EXECUTION_FAILED":
                        return dict(
                            common,
                            ok=False,
                            mission_outcome="safe_failure",
                            state="FIND_MARVIN_FAILED",
                            reason=(
                                progress.get("reason", terminal.lower())
                                if isinstance(progress, dict) else terminal.lower()
                            ),
                        )
                    return dict(
                        common,
                        ok=True,
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason=(
                            progress.get("reason", terminal.lower())
                            if isinstance(progress, dict) else terminal.lower()
                        ),
                        completion_reason=terminal,
                    )
                if controller.get("reason") == "marvin_local_progress_action_budget_insufficient":
                    return dict(
                        result_base(),
                        ok=True,
                        completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason=controller["reason"],
                        completion_reason=controller["reason"],
                    )
                if controller.get("reason") == "marvin_local_progress_complete":
                    return dict(
                        result_base(),
                        ok=True,
                        completed=True,
                        arrived_at_marvin=False,
                        mission_outcome="safe_incomplete",
                        state="FIND_MARVIN_SAFE_INCOMPLETE",
                        reason=controller["reason"],
                        completion_reason="local_progress_reassessment_required",
                        post_progress_pursuit_state=controller.get(
                            "post_progress_pursuit_state",
                        ),
                        local_progress_result=controller.get(
                            "local_progress_result",
                        ),
                    )
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason="find_marvin_controller_terminal_result_unrecognized",
                    controller_reason=controller.get("reason"),
                )

            if not self._marvin_episode_is_safe_action_limit(episode, controller):
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason="find_marvin_action_limit_result_not_safely_stopped",
                )

            # The prior controller has already stopped after every executor
            # attempt. Verify that stop and the Bridge state before permitting
            # a new controller episode; the next call obtains a fresh Preview
            # and has an episode-local arrival/stale-replan state.
            if not self._marvin_mission_context_is_current(
                mission, control_generation,
            ):
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="preempted",
                    state="FIND_MARVIN_PREEMPTED",
                    reason="find_marvin_mission_preempted",
            )
            bridge_status = self._marvin_bridge_ready_and_stopped()
            if not isinstance(bridge_status, dict) or bridge_status.get("ok") is not True:
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="safe_failure",
                    state="FIND_MARVIN_FAILED",
                    reason="find_marvin_episode_bridge_not_ready_or_stopped",
                    bridge_status=bridge_status,
                )
            if not self._marvin_mission_context_is_current(
                mission, control_generation,
            ):
                return dict(
                    result_base(), ok=False, completed=True,
                    arrived_at_marvin=False, mission_outcome="preempted",
                    state="FIND_MARVIN_PREEMPTED",
                    reason="find_marvin_mission_preempted",
                )

        # The bounded loop always returns from an explicit terminal branch.
        return dict(
            result_base(), ok=False, completed=True,
            arrived_at_marvin=False, mission_outcome="safe_failure",
            state="FIND_MARVIN_FAILED",
            reason="find_marvin_mission_episode_loop_exited_unexpectedly",
        )

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
            and motion.get("angular_z") == 0
            and motion.get("streaming") is False
        )

    def _marvin_mission_context_is_current(self, mission, control_generation):
        with self._state_lock:
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
            expected_direction = (
                "LEFT" if observation.get("controller_decision") == "TURN_LEFT"
                else "RIGHT" if observation.get("controller_decision") == "TURN_RIGHT"
                else None
            )
            strict_observation = bool(
                observation.get("identity_confirmed") is True
                and observation.get("identity_source")
                == "gemini_marvin_candidate_selection"
                and observation.get("controller_state") == VISUAL_READY_TO_ALIGN
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
            # Consume before guarded dispatch: any later transport ambiguity or
            # JIT veto requires a genuinely new strict observation, preventing
            # an HTTP retry from duplicating a possible physical action.
            consumed.add(source_frame_stamp_ns)

        base["execution_authorized"] = True
        try:
            turn = execute_turn(
                normalized_direction,
                float(angular_speed),
                float(duration),
                expected_lidar_session=session,
                safety_mode=ROTATIONAL_SWEPT_FOOTPRINT,
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

    def execute_single_marvin_approach(self, *, linear_speed, duration):
        """Execute one capped active-runtime Marvin forward step, then stop."""
        base = {"ok": False, "action": "single_marvin_approach_step",
                "execution_authorized": False, "motion_executed": False,
                "actions_executed": 0, "linear_speed": None, "duration": None,
                "producer_session": None, "approach_result": None,
                "stop_result": None, "reason": None}
        if not _bounded_alignment_number(linear_speed, maximum=0.08):
            return dict(base, reason="marvin_approach_linear_speed_invalid")
        if not _bounded_alignment_number(duration, maximum=0.50):
            return dict(base, reason="marvin_approach_duration_invalid")
        if float(linear_speed) != 0.08 or float(duration) != 0.50:
            return dict(base, reason="marvin_approach_parameters_not_calibrated")
        base.update(linear_speed=float(linear_speed), duration=float(duration))
        if self.running is not True:
            return dict(base, reason="marvin_approach_runtime_not_running")
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
            if getattr(self, "_marvin_approach_step_consumed", False):
                return dict(base, reason="marvin_approach_step_already_consumed")
            self._marvin_approach_step_consumed = True
        base["execution_authorized"] = True
        try:
            approach_result = approach(expected_lidar_session=session,
                                       linear_speed=float(linear_speed), duration=float(duration))
        except Exception as exc:
            approach_result = {"ok": False, "motion_executed": False, "error": str(exc)}
        try:
            stop_result = stop()
        except Exception as exc:
            stop_result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        moved = bool(isinstance(approach_result, dict) and approach_result.get("motion_executed") is True)
        stop_ok = isinstance(stop_result, dict) and stop_result.get("ok") is True
        return dict(base, ok=moved and stop_ok, motion_executed=moved, actions_executed=1,
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
            if intent_name == "STOP":
                self._control_generation += 1

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
            mission = self.mission_manager.get_active_mission()

            if mission is None:
                mission = self.mission_manager.start_next_mission()

            if mission is None:
                self._set_runtime_state("IDLE")
                return None

            mission_id = mission.mission_id
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
        self._stop_lidar()
        if self.forward_interlock is not None:
            self.forward_interlock.stop()

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
