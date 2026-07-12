#!/usr/bin/env python3

import os
import shutil
import tempfile

from mission_manager import MissionManager
from runtime import CognitiveRuntime
from world_model import WorldModel


class FakeProvider:
    def get_intent(self, user_text):
        commands = {
            "move forward": {
                "intent": "MOVE_FORWARD",
                "speech": "Moving forward.",
                "target": None,
            },
            "turn left": {
                "intent": "TURN_LEFT",
                "speech": "Turning left.",
                "target": None,
            },
        }

        return commands.get(
            user_text.lower(),
            {
                "intent": "UNKNOWN",
                "speech": "Unknown command.",
                "target": None,
            },
        )


class FakeRobotClient:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1

        return {
            "ok": True,
            "action": "stop",
        }


class FakeBehaviorManager:
    def __init__(self):
        self.executed_missions = []

    def execute(self, mission):
        self.executed_missions.append(mission.mission_type)

        return {
            "ok": True,
            "executed": True,
            "completed": True,
            "behavior": mission.mission_type,
            "reason": "Offline runtime test behavior completed.",
        }


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="cognitive_runtime_test_"
    )

    world_model_path = os.path.join(
        temporary_directory,
        "world_model_state.json",
    )

    try:
        world_model = WorldModel(world_model_path)
        mission_manager = MissionManager()
        robot_client = FakeRobotClient()
        behavior_manager = FakeBehaviorManager()

        runtime = CognitiveRuntime(
            provider=FakeProvider(),
            mission_manager=mission_manager,
            world_model=world_model,
            vision_adapter=object(),
            robot_client=robot_client,
            behavior_manager=behavior_manager,
            loop_interval=0.01,
        )

        print("===== SUBMIT FIRST MISSION =====")
        first_submission = runtime.submit_text("move forward")
        print(first_submission)

        assert (
            first_submission["mission"]["status"]
            == "ACTIVE"
        )

        assert (
            mission_manager.get_active_mission().mission_type
            == "MOVE_FORWARD"
        )

        print()
        print("===== SUBMIT SECOND MISSION =====")
        second_submission = runtime.submit_text("turn left")
        print(second_submission)

        assert (
            second_submission["mission"]["status"]
            == "QUEUED"
        )

        assert len(mission_manager.get_queue()) == 1

        print()
        print("===== EXECUTE FIRST MISSION =====")
        first_result = runtime.run_once()
        print(first_result)

        assert first_result["ok"] is True

        assert behavior_manager.executed_missions == [
            "MOVE_FORWARD",
        ]

        active_after_first = (
            mission_manager.get_active_mission()
        )

        assert active_after_first is not None

        assert (
            active_after_first.mission_type
            == "TURN_LEFT"
        )

        assert active_after_first.status == "ACTIVE"

        print()
        print("===== EXECUTE SECOND MISSION =====")
        second_result = runtime.run_once()
        print(second_result)

        assert second_result["ok"] is True

        assert behavior_manager.executed_missions == [
            "MOVE_FORWARD",
            "TURN_LEFT",
        ]

        assert mission_manager.get_active_mission() is None
        assert mission_manager.get_queue() == []
        assert len(mission_manager.get_history()) == 2

        status = runtime.get_status()

        print()
        print("===== FINAL RUNTIME STATUS =====")
        print(status)

        assert status["active_mission"] is None
        assert status["queue"] == []
        assert status["history_count"] == 2
        assert status["last_result"]["ok"] is True

        robot_state = world_model.robot_state

        assert (
            robot_state["runtime_state"]
            == "MISSION_COMPLETED"
        )

        assert robot_state["mission"] is None
        assert robot_state["mission_queue"] == []

        print()
        print("PASS: persistent runtime owns one MissionManager")
        print("PASS: second mission remains queued")
        print("PASS: first completion activates next mission")
        print("PASS: queue becomes empty after both missions")
        print("PASS: World Model stores runtime state")
        print()
        print("Cognitive Runtime offline test passed.")

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
