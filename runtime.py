#!/usr/bin/env python3

import argparse
import json
import math
import signal
import threading
import time
from typing import Any, Dict, Optional

from behavior_manager import BehaviorManager
from config import load_config
from lidar_perception import LidarPerceptionWorker, unavailable_state
from robot_bridge.forward_interlock import ForwardMotionInterlock
from mission_manager import MissionManager
from provider_factory import create_provider
from robot_bridge.client import RobotBridgeClient
from tracking_state import build_tracking_state, empty_tracking_state
from vision_adapter import VisionAdapter
from semantic_vision import SemanticVisionClient
from world_model import WorldModel


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
        self._marvin_alignment_step_consumed = False
        self._marvin_approach_step_consumed = False
        self._marvin_autonomous_run_consumed = False
        self._last_runtime_state = None
        self._control_generation = 0
        self._behavior_execution_generation = None
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

    def build_find_marvin_controller_state(self):
        """Return one fresh Marvin controller evidence bundle without action."""
        builder = getattr(
            self.behavior_manager,
            "build_find_marvin_controller_state",
            None,
        )
        if not callable(builder):
            raise RuntimeError("find_marvin_state_provider_unavailable")
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

    def execute_bounded_find_marvin_autonomous(self, *, max_actions):
        """Run one explicitly-authorized, finite Marvin controller episode.

        The controller obtains a new Preview after every action.  Forward
        safety remains inside the active BehaviorManager immediately before
        dispatch; this method deliberately does not cache or pre-authorize a
        LiDAR snapshot.
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
        with self._state_lock:
            if self._marvin_autonomous_run_consumed:
                return dict(base, reason="marvin_autonomous_run_already_consumed")
            self._marvin_autonomous_run_consumed = True
        base["execution_authorized"] = True
        try:
            result = controller(
                self.build_find_marvin_controller_state,
                max_actions=max_actions,
                dry_run=False,
                stop_after_action=stop,
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

    def execute_single_marvin_alignment(
        self, *, direction, angular_speed, duration,
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
        base.update(
            direction=normalized_direction,
            angular_speed=float(angular_speed),
            duration=float(duration),
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
            if self._marvin_alignment_step_consumed:
                return dict(base, reason="marvin_alignment_step_already_consumed")
            self._marvin_alignment_step_consumed = True

        base["execution_authorized"] = True
        try:
            turn = execute_turn(
                normalized_direction,
                float(angular_speed),
                float(duration),
                expected_lidar_session=session,
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
        return {
            "running": running,
            "producer_session": session,
            "acquisition_sequence": worker.sequence if worker is not None else 0,
            "available": state.get("available", False),
            "valid": state.get("valid", False),
            "reason": state.get("reason"),
            "effective_age_seconds": state.get("effective_age_seconds"),
            "front_state": (
                state.get("sectors", {}).get("front", {}).get("state", "UNKNOWN")
                if state.get("valid") else "UNKNOWN"
            ),
            "last_error": self._lidar_error or (worker.last_error if worker is not None else None),
        }

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

            self.world_model.update_robot_state(
                runtime_state="EXECUTING",
                mission=mission.to_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

        execution_error = None

        try:
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
                return self.last_result

            self.last_result = result
            self.last_error = execution_error
            self._behavior_execution_generation = None
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
