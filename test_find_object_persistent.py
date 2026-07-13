#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeRobot:
    def __init__(self):
        self.commands = []

    def move_forward(self, speed, seconds):
        self.commands.append(
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
        self.commands.append(
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
        self.commands.append(
            ("turn_right", speed, seconds)
        )

        return {
            "ok": True,
            "action": "motion",
            "linear_x": 0.0,
            "angular_z": -speed,
            "duration": seconds,
        }

    def stop(self):
        self.commands.append(
            ("stop",)
        )

        return {
            "ok": True,
            "action": "stop",
        }


class FakeWorldModel:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0
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

        result = self.sequence[
            min(
                self.index,
                len(self.sequence) - 1,
            )
        ]

        self.index += 1
        return dict(result)


SEARCH = {
    "found": False,
    "stale": False,
    "reason": "Target not visible.",
}

CENTER_LEFT = {
    "found": True,
    "stale": False,
    "target": "backpack",
    "cx": 120.0,
    "cy": 240.0,
    "area": 15000.0,
    "image_width": 640.0,
    "image_height": 480.0,
}

APPROACH = {
    "found": True,
    "stale": False,
    "target": "backpack",
    "cx": 320.0,
    "cy": 240.0,
    "area": 22000.0,
    "image_width": 640.0,
    "image_height": 480.0,
}

ARRIVED = {
    "found": True,
    "stale": False,
    "target": "backpack",
    "cx": 320.0,
    "cy": 240.0,
    "area": 82000.0,
    "image_width": 640.0,
    "image_height": 480.0,
}


def main():
    robot = FakeRobot()

    world_model = FakeWorldModel(
        [
            SEARCH,
            CENTER_LEFT,
            APPROACH,
            ARRIVED,
        ]
    )

    manager = BehaviorManager(
        robot_client=robot,
        world_model=world_model,
    )

    mission = create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Looking for your backpack.",
    )

    print("===== RUNTIME CYCLE 1: SEARCH =====")

    result = manager.execute(mission)
    print(result)

    assert result["ok"] is True
    assert result["state"] == "SEARCHING"
    assert result["completed"] is False
    assert result["cycle"] == 1
    assert robot.commands[-1][0] == "turn_left"
    assert world_model.index == 1

    print()
    print("===== RUNTIME CYCLE 2: CENTER LEFT =====")

    result = manager.execute(mission)
    print(result)

    assert result["ok"] is True
    assert result["state"] == "CENTERING_LEFT"
    assert result["completed"] is False
    assert result["cycle"] == 1
    assert robot.commands[-1][0] == "turn_left"
    assert world_model.index == 2

    print()
    print("===== RUNTIME CYCLE 3: APPROACH =====")

    result = manager.execute(mission)
    print(result)

    assert result["ok"] is True
    assert result["state"] == "APPROACHING"
    assert result["completed"] is False
    assert result["cycle"] == 1
    assert robot.commands[-1][0] == "move_forward"
    assert world_model.index == 3

    print()
    print("===== RUNTIME CYCLE 4: ARRIVED =====")

    result = manager.execute(mission)
    print(result)

    assert result["ok"] is True
    assert result["state"] == "ARRIVED"
    assert result["completed"] is True
    assert result["cycle"] == 1
    assert robot.commands[-1][0] == "stop"
    assert world_model.index == 4

    assert len(world_model.queries) == 4

    print()
    print("PASS: each execute call performs one cycle")
    print("PASS: searching remains active")
    print("PASS: centering remains active")
    print("PASS: approach remains active")
    print("PASS: arrival completes mission")
    print("PASS: internal FIND_OBJECT loop was removed")
    print()
    print("Persistent FIND_OBJECT behavior passed.")


if __name__ == "__main__":
    main()
