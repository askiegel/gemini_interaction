#!/usr/bin/env python3

import os
import shutil
import tempfile
import threading

from mission_manager import MissionManager
from runtime import CognitiveRuntime
from world_model import WorldModel


class FakeProvider:
    def get_intent(self, text):
        normalized = str(text).strip().lower()

        if normalized == "stop":
            return {
                "intent": "STOP",
                "speech": "Stopping now.",
                "target": None,
            }

        if "backpack" in normalized:
            return {
                "intent": "FIND_OBJECT",
                "speech": "Looking for your backpack.",
                "target": "backpack",
            }

        return {
            "intent": "TURN_LEFT",
            "speech": "Turning left.",
            "target": None,
        }


class FakeRobotClient:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1

        return {
            "ok": True,
            "action": "stop",
            "stop_count": self.stop_count,
        }


class PersistentBehavior:
    def execute(self, mission):
        return {
            "ok": True,
            "executed": True,
            "completed": False,
            "behavior": mission.mission_type,
            "state": "SEARCHING",
            "reason": "Mission remains active.",
        }


class BlockingBehavior:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, mission):
        self.started.set()

        if not self.release.wait(timeout=5):
            raise RuntimeError(
                "Blocking behavior was not released."
            )

        return {
            "ok": True,
            "executed": True,
            "completed": False,
            "behavior": mission.mission_type,
            "state": "APPROACHING",
            "reason": "This result must not overwrite STOP.",
        }


def create_runtime(
    storage_path,
    robot_client,
    behavior_manager,
):
    return CognitiveRuntime(
        provider=FakeProvider(),
        mission_manager=MissionManager(),
        world_model=WorldModel(storage_path),
        vision_adapter=object(),
        robot_client=robot_client,
        behavior_manager=behavior_manager,
        loop_interval=0.01,
    )


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="runtime_stop_preemption_"
    )

    normal_storage = os.path.join(
        temporary_directory,
        "normal_stop.json",
    )

    race_storage = os.path.join(
        temporary_directory,
        "inflight_stop.json",
    )

    try:
        print("===== CANCEL ACTIVE AND QUEUED MISSIONS =====")

        robot = FakeRobotClient()

        runtime = create_runtime(
            storage_path=normal_storage,
            robot_client=robot,
            behavior_manager=PersistentBehavior(),
        )

        first = runtime.submit_text(
            "Find my backpack"
        )

        second = runtime.submit_text(
            "Turn left"
        )

        assert first["mission"]["status"] == "ACTIVE"
        assert second["mission"]["status"] == "QUEUED"

        stop_submission = runtime.submit_text(
            "Stop"
        )

        print(stop_submission)
        print(runtime.get_status())

        assert robot.stop_count == 1

        assert (
            runtime.mission_manager
            .get_active_mission()
            is None
        )

        assert (
            runtime.mission_manager
            .get_queue()
            == []
        )

        assert (
            stop_submission["mission"]["status"]
            == "CANCELLED"
        )

        history = (
            runtime.mission_manager
            .get_history()
        )

        queued_matches = [
            mission
            for mission in history
            if (
                mission["mission_id"]
                == second["mission"]["mission_id"]
            )
        ]

        assert len(queued_matches) == 1

        assert (
            queued_matches[0]["status"]
            == "CANCELLED"
        )

        status = runtime.get_status()

        assert status["runtime_state"] == "STOPPED"
        assert status["active_mission"] is None
        assert status["queue"] == []

        assert (
            status["last_result"]["behavior"]
            == "STOP"
        )

        assert (
            status["last_result"]["state"]
            == "STOPPED"
        )

        print()
        print("===== STOP DURING IN-FLIGHT BEHAVIOR =====")

        blocking_behavior = BlockingBehavior()
        race_robot = FakeRobotClient()

        race_runtime = create_runtime(
            storage_path=race_storage,
            robot_client=race_robot,
            behavior_manager=blocking_behavior,
        )

        race_runtime.submit_text(
            "Find my backpack"
        )

        thread_result = {}

        def execute_cycle():
            thread_result["result"] = (
                race_runtime.run_once()
            )

        behavior_thread = threading.Thread(
            target=execute_cycle,
            daemon=True,
        )

        behavior_thread.start()

        assert blocking_behavior.started.wait(
            timeout=3
        )

        race_runtime.submit_text("Stop")

        assert race_robot.stop_count == 1

        blocking_behavior.release.set()
        behavior_thread.join(timeout=5)

        assert not behavior_thread.is_alive()

        race_status = race_runtime.get_status()

        print(race_status)
        print(thread_result)

        assert race_status["runtime_state"] == "STOPPED"
        assert race_status["active_mission"] is None
        assert race_status["queue"] == []

        assert (
            race_status["last_result"]["behavior"]
            == "STOP"
        )

        assert (
            race_status["last_result"]["state"]
            == "STOPPED"
        )

        assert (
            thread_result["result"]["behavior"]
            == "STOP"
        )

        race_runtime.world_model.reload()

        assert (
            race_runtime.world_model.robot_state[
                "runtime_state"
            ]
            == "STOPPED"
        )

        assert (
            race_runtime.world_model.robot_state[
                "mission"
            ]
            is None
        )

        assert (
            race_runtime.world_model.robot_state[
                "mission_queue"
            ]
            == []
        )

        print()
        print("PASS: STOP commands Robot Bridge immediately")
        print("PASS: active mission is cancelled")
        print("PASS: queued missions are cancelled and cleared")
        print("PASS: runtime returns to STOPPED state")
        print("PASS: no queued mission resumes")
        print(
            "PASS: in-flight behavior cannot overwrite STOP"
        )
        print()
        print("Runtime STOP preemption test passed.")

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
