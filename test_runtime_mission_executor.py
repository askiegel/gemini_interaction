#!/usr/bin/env python3

import os
import shutil
import tempfile

from mission_manager import MissionManager
from runtime import CognitiveRuntime
from world_model import WorldModel


class FakeProvider:
    def get_intent(self, user_text):
        del user_text

        return {
            "intent": "FOLLOW_PERSON",
            "speech": "Following the person.",
            "target": "person",
        }


class FakeRobotClient:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1

        return {
            "ok": True,
            "action": "stop",
        }


class PersistentFakeBehaviorManager:
    """
    Simulate a mission that needs three bounded execution cycles.

    Cycles 1 and 2 advance the mission but do not complete it.
    Cycle 3 reaches the terminal state.
    """

    def __init__(self):
        self.execution_count = 0
        self.mission_ids = []

    def execute(self, mission):
        self.execution_count += 1
        self.mission_ids.append(mission.mission_id)

        if self.execution_count < 3:
            return {
                "ok": True,
                "executed": True,
                "completed": False,
                "behavior": mission.mission_type,
                "state": "TRACKING",
                "reason": (
                    "Completed one bounded tracking step; "
                    "mission remains active."
                ),
            }

        return {
            "ok": True,
            "executed": True,
            "completed": True,
            "behavior": mission.mission_type,
            "state": "TARGET_REACHED",
            "reason": "Persistent mission reached its terminal state.",
        }


class LegacyBoundedBehaviorManager:
    """
    Existing bounded behaviors omit the completed field.

    They must continue to complete after one execution.
    """

    def __init__(self):
        self.execution_count = 0

    def execute(self, mission):
        self.execution_count += 1

        return {
            "ok": True,
            "executed": True,
            "behavior": mission.mission_type,
            "reason": "Legacy bounded behavior completed.",
        }


def create_runtime(
    storage_path,
    behavior_manager,
):
    world_model = WorldModel(storage_path)

    return CognitiveRuntime(
        provider=FakeProvider(),
        mission_manager=MissionManager(),
        world_model=world_model,
        vision_adapter=object(),
        robot_client=FakeRobotClient(),
        behavior_manager=behavior_manager,
        loop_interval=0.01,
    )


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="runtime_mission_executor_"
    )

    persistent_storage = os.path.join(
        temporary_directory,
        "persistent_world_model.json",
    )

    legacy_storage = os.path.join(
        temporary_directory,
        "legacy_world_model.json",
    )

    try:
        print("===== SUBMIT PERSISTENT MISSION =====")

        behavior = PersistentFakeBehaviorManager()

        runtime = create_runtime(
            storage_path=persistent_storage,
            behavior_manager=behavior,
        )

        submission = runtime.submit_text(
            "Follow me"
        )

        mission_id = submission["mission"]["mission_id"]

        print(submission)

        assert (
            runtime.mission_manager
            .get_active_mission()
            .mission_id
            == mission_id
        )

        print()
        print("===== EXECUTOR CYCLE 1 =====")

        first_result = runtime.run_once()
        print(first_result)

        first_active = (
            runtime.mission_manager
            .get_active_mission()
        )

        assert first_result["ok"] is True
        assert first_result["completed"] is False
        assert first_active is not None
        assert first_active.mission_id == mission_id

        runtime.world_model.reload()

        assert (
            runtime.world_model.robot_state[
                "runtime_state"
            ]
            == "MISSION_ACTIVE"
        )

        print()
        print("===== EXECUTOR CYCLE 2 =====")

        second_result = runtime.run_once()
        print(second_result)

        second_active = (
            runtime.mission_manager
            .get_active_mission()
        )

        assert second_result["completed"] is False
        assert second_active is not None
        assert second_active.mission_id == mission_id

        print()
        print("===== EXECUTOR CYCLE 3 =====")

        third_result = runtime.run_once()
        print(third_result)

        assert third_result["completed"] is True
        assert (
            runtime.mission_manager
            .get_active_mission()
            is None
        )

        assert behavior.execution_count == 3
        assert behavior.mission_ids == [
            mission_id,
            mission_id,
            mission_id,
        ]

        runtime.world_model.reload()

        assert (
            runtime.world_model.robot_state[
                "runtime_state"
            ]
            == "MISSION_COMPLETED"
        )

        assert (
            runtime.world_model.robot_state[
                "last_behavior_result"
            ]["state"]
            == "TARGET_REACHED"
        )

        print()
        print("===== LEGACY BOUNDED BEHAVIOR =====")

        legacy_behavior = (
            LegacyBoundedBehaviorManager()
        )

        legacy_runtime = create_runtime(
            storage_path=legacy_storage,
            behavior_manager=legacy_behavior,
        )

        legacy_runtime.submit_text(
            "Follow me"
        )

        legacy_result = legacy_runtime.run_once()
        print(legacy_result)

        assert legacy_result["ok"] is True

        assert (
            "completed"
            not in legacy_result
        )

        assert (
            legacy_runtime.mission_manager
            .get_active_mission()
            is None
        )

        assert (
            legacy_behavior.execution_count
            == 1
        )

        print()
        print(
            "PASS: incomplete mission remains active"
        )
        print(
            "PASS: same mission executes across cycles"
        )
        print(
            "PASS: terminal behavior completes mission"
        )
        print(
            "PASS: World Model reports active mission state"
        )
        print(
            "PASS: legacy bounded behaviors remain compatible"
        )
        print()
        print(
            "Persistent Mission Executor test passed."
        )

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
