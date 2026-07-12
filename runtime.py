#!/usr/bin/env python3

import argparse
import json
import signal
import threading
import time
from typing import Any, Dict, Optional

from behavior_manager import BehaviorManager
from config import load_config
from mission_manager import MissionManager
from provider_factory import create_provider
from robot_bridge.client import RobotBridgeClient
from vision_adapter import VisionAdapter
from world_model import WorldModel


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

    LOOP_INTERVAL_SECONDS = 0.25

    def __init__(
        self,
        provider=None,
        mission_manager=None,
        world_model=None,
        vision_adapter=None,
        robot_client=None,
        behavior_manager=None,
        loop_interval=None,
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

        self.robot_client = robot_client or RobotBridgeClient(
            timeout=15.0,
        )

        self.behavior_manager = behavior_manager or BehaviorManager(
            robot_client=self.robot_client,
            vision_adapter=self.vision_adapter,
            world_model=self.world_model,
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
        self._state_lock = threading.RLock()
        self._last_runtime_state = None

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
        """
        if not isinstance(intent, dict):
            raise TypeError("Intent must be a dictionary.")

        with self._state_lock:
            mission = self.mission_manager.handle_intent(intent)

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

            self.world_model.update_robot_state(
                runtime_state="EXECUTING",
                mission=mission.to_dict(),
                mission_queue=self.mission_manager.get_queue(),
            )

        try:
            result = self.behavior_manager.execute(mission)

            if not isinstance(result, dict):
                raise TypeError(
                    "BehaviorManager.execute() must return a dictionary."
                )

            self.last_result = result
            self.last_error = None

        except Exception as exc:
            result = {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": str(exc),
            }

            self.last_result = result
            self.last_error = str(exc)

            try:
                self.robot_client.stop()
            except Exception:
                pass

        with self._state_lock:
            active = self.mission_manager.get_active_mission()

            if active and active.mission_id == mission_id:
                if result.get("ok"):
                    finished_mission = (
                        self.mission_manager.complete_active_mission()
                    )
                    runtime_state = "MISSION_COMPLETED"
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

            self.world_model.update_robot_state(
                runtime_state=runtime_state,
                mission=(
                    next_mission.to_dict()
                    if next_mission is not None
                    else None
                ),
                last_completed_mission=(
                    finished_mission.to_dict()
                    if finished_mission is not None
                    else None
                ),
                mission_queue=self.mission_manager.get_queue(),
                last_behavior_result=result,
            )

        return result

    def run_forever(self):
        """
        Run the persistent mission-processing loop until shutdown.
        """
        self.running = True
        self.started_at = time.time()

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

        try:
            while self.running:
                self.run_once()
                time.sleep(self.loop_interval)

        finally:
            self.running = False

            try:
                self.robot_client.stop()
            except Exception:
                pass

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
                "last_error": self.last_error,
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
