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
        return {"ok": True}

    def turn_left(self, speed, seconds):
        self.commands.append(
            ("turn_left", speed, seconds)
        )
        return {"ok": True}

    def turn_right(self, speed, seconds):
        self.commands.append(
            ("turn_right", speed, seconds)
        )
        return {"ok": True}

    def stop(self):
        self.commands.append(("stop",))
        return {"ok": True}


class FakeWorldModel:
    def __init__(self, observations):
        self.observations = list(observations)
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

        observation = self.observations[
            min(
                self.index,
                len(self.observations) - 1,
            )
        ]

        self.index += 1
        return dict(observation)


def person(cx, area):
    return {
        "found": True,
        "stale": False,
        "target": "person",
        "label": "person",
        "confidence": 0.92,
        "cx": float(cx),
        "cy": 240.0,
        "area": float(area),
        "image_width": 640.0,
        "image_height": 480.0,
    }


def main():
    observations = [
        {
            "found": False,
            "stale": False,
            "target": "person",
        },
        person(cx=120, area=20000),
        person(cx=520, area=22000),
        person(cx=320, area=30000),
        person(cx=320, area=70000),
    ]

    robot = FakeRobot()
    world_model = FakeWorldModel(observations)

    manager = BehaviorManager(
        robot_client=robot,
        world_model=world_model,
    )

    mission = create_mission(
        mission_type="FOLLOW_PERSON",
        target="person",
        speech="Following the person.",
    )

    print("===== CYCLE 1: SEARCH =====")
    result = manager.execute(mission)
    print(result)

    assert result["behavior"] == "FOLLOW_PERSON"
    assert result["state"] == "HOLDING_NO_PREDICTION"
    assert result["completed"] is False
    assert result["state"] == "HOLDING_NO_PREDICTION"
    assert result["commanded_linear_x"] == 0.0
    assert result["commanded_angular_z"] == 0.0

    print()
    print("===== CYCLE 2: CENTER LEFT =====")
    result = manager.execute(mission)
    print(result)

    assert result["state"] == "CENTERING_LEFT"
    assert result["completed"] is False
    assert robot.commands[-1][0] == "turn_left"

    print()
    print("===== CYCLE 3: CENTER RIGHT =====")
    result = manager.execute(mission)
    print(result)

    assert result["state"] == "CENTERING_RIGHT"
    assert result["completed"] is False
    assert robot.commands[-1][0] == "turn_right"

    print()
    print("===== CYCLE 4: APPROACH =====")
    result = manager.execute(mission)
    print(result)

    assert result["state"] == "APPROACHING"
    assert result["completed"] is False
    assert robot.commands[-1][0] == "move_forward"

    print()
    print("===== CYCLE 5: MAINTAIN DISTANCE =====")
    result = manager.execute(mission)
    print(result)

    assert result["state"] == "MAINTAINING_DISTANCE"

    # FOLLOW_PERSON remains persistent until STOP.
    assert result["completed"] is False
    assert robot.commands[-1][0] == "stop"

    assert [
        command[0]
        for command in robot.commands
    ] == [
        "stop",
        "turn_left",
        "turn_right",
        "move_forward",
        "stop",
    ]

    assert len(world_model.queries) == 5

    for query in world_model.queries:
        assert query["label"] == "person"
        assert query["refresh"] is True
        assert (
            query["max_age_seconds"]
            == manager.TARGET_MAX_AGE_SECONDS
        )

    print()
    print("PASS: missing person causes a safe observation hold")
    print("PASS: image-left person causes one left correction")
    print("PASS: image-right person causes one right correction")
    print("PASS: centered distant person causes one forward step")
    print("PASS: close person causes a safe stop")
    print("PASS: FOLLOW_PERSON remains active until STOP")
    print("PASS: World Model is the perception source")
    print()
    print("FOLLOW_PERSON behavior test passed.")


if __name__ == "__main__":
    main()
