#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeRobotClient:
    def __init__(self):
        self.actions = []

    def stop(self):
        self.actions.append(("stop",))
        return {
            "ok": True,
            "action": "stop",
        }

    def move_forward(self, speed, seconds):
        self.actions.append(
            ("move_forward", speed, seconds)
        )

        return {
            "ok": True,
            "action": "motion",
            "linear_x": speed,
            "angular_z": 0.0,
            "duration": seconds,
        }

    def turn_left(self, speed, seconds):
        self.actions.append(
            ("turn_left", speed, seconds)
        )

        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": speed,
            "duration": seconds,
        }

    def turn_right(self, speed, seconds):
        self.actions.append(
            ("turn_right", speed, seconds)
        )

        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": -speed,
            "duration": seconds,
        }


class FakeWorldModel:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        self.queries.append(
            {
                "label": label,
                "max_age_seconds": max_age_seconds,
                "refresh": refresh,
            }
        )

        if not self.results:
            raise RuntimeError(
                "No fake World Model results remain."
            )

        return self.results.pop(0)


class ForbiddenVisionAdapter:
    def find_target(self, target):
        raise AssertionError(
            "BehaviorManager must not query VisionAdapter "
            "when a World Model is available."
        )


def make_mission():
    return create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Looking for your backpack.",
    )


def main():
    print("===== WORLD MODEL ARRIVAL QUERY =====")

    robot = FakeRobotClient()

    world_model = FakeWorldModel(
        [
            {
                "found": True,
                "stale": False,
                "target": "backpack",
                "entity_id": "backpack-001",
                "confidence": 0.91,
                "cx": 339.0,
                "cy": 250.0,
                "area": 76000.0,
                "image_width": 640.0,
                "image_height": 480.0,
            }
        ]
    )

    behavior = BehaviorManager(
        robot_client=robot,
        vision_adapter=ForbiddenVisionAdapter(),
        world_model=world_model,
    )

    result = behavior._execute_find_object_cycle(
        target_name="backpack",
        cycle_number=1,
    )

    print(result)

    assert result["ok"] is True
    assert result["state"] == "ARRIVED"
    assert result["completed"] is True
    assert robot.actions == [("stop",)]

    assert world_model.queries == [
        {
            "label": "backpack",
            "max_age_seconds": (
                behavior.TARGET_MAX_AGE_SECONDS
            ),
            "refresh": True,
        }
    ]

    print()
    print("===== STALE OBSERVATION SEARCH SAFETY =====")

    robot = FakeRobotClient()

    stale_world_model = FakeWorldModel(
        [
            {
                "found": False,
                "stale": True,
                "target": "backpack",
                "entity_id": "backpack-001",
                "reason": (
                    "Latest backpack observation is stale."
                ),
            }
        ]
    )

    behavior = BehaviorManager(
        robot_client=robot,
        vision_adapter=ForbiddenVisionAdapter(),
        world_model=stale_world_model,
    )

    result = behavior._execute_find_object_cycle(
        target_name="backpack",
        cycle_number=1,
    )

    print(result)

    assert result["ok"] is True
    assert result["state"] == "SEARCHING"
    assert robot.actions == [
        (
            "turn_left",
            behavior.SEARCH_TURN_SPEED,
            behavior.SEARCH_TURN_SECONDS,
        )
    ]

    print()
    print("PASS: BehaviorManager reads shared World Model")
    print("PASS: direct Vision Adapter query is bypassed")
    print("PASS: fresh target geometry drives arrival")
    print("PASS: stale observations cannot drive approach")
    print()
    print(
        "BehaviorManager World Model test passed."
    )


if __name__ == "__main__":
    main()
