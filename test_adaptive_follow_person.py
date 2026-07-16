#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeRobot:
    def __init__(self):
        self.commands = []

    def motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        duration=0.25,
    ):
        command = {
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "duration": float(duration),
        }

        self.commands.append(
            ("motion", command)
        )

        return {
            "ok": True,
            "action": "motion",
            **command,
            "automatic_stop": True,
        }

    def turn_left(self, speed, seconds):
        raise AssertionError(
            "Adaptive FOLLOW_PERSON should use motion()."
        )

    def turn_right(self, speed, seconds):
        raise AssertionError(
            "Adaptive FOLLOW_PERSON should use motion()."
        )

    def move_forward(self, speed, seconds):
        raise AssertionError(
            "Adaptive FOLLOW_PERSON should use motion()."
        )

    def stop(self):
        self.commands.append(("stop",))

        return {
            "ok": True,
            "action": "stop",
        }


class FakeWorldModel:
    def __init__(self, observations):
        self.observations = list(
            observations
        )

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        if not self.observations:
            raise AssertionError(
                "No observations remain."
            )

        return dict(
            self.observations.pop(0)
        )


def person(cx, area=25000):
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
    robot = FakeRobot()

    world_model = FakeWorldModel(
        [
            # Errors relative to image center 320:
            person(cx=215),  # -105, gentle left
            person(cx=40),   # -280, stronger left
            person(cx=425),  # +105, gentle right
            person(cx=600),  # +280, stronger right
            person(cx=270),  # -50, forward + left steer
            person(cx=370),  # +50, forward + right steer
            person(cx=320),  # centered forward
            person(cx=320, area=70000),  # stop
        ]
    )

    manager = BehaviorManager(
        robot_client=robot,
        world_model=world_model,
    )

    mission = create_mission(
        mission_type="FOLLOW_PERSON",
        target="person",
        speech="Following the person.",
    )

    print("===== GENTLE LEFT CORRECTION =====")
    gentle_left = manager.execute(mission)
    print(gentle_left)

    print()
    print("===== STRONG LEFT CORRECTION =====")
    strong_left = manager.execute(mission)
    print(strong_left)

    assert gentle_left["state"] == "CENTERING_LEFT"
    assert strong_left["state"] == "CENTERING_LEFT"

    assert (
        gentle_left["commanded_angular_z"]
        > 0.0
    )

    assert (
        strong_left["commanded_angular_z"]
        >
        gentle_left["commanded_angular_z"]
    )

    assert (
        strong_left["commanded_angular_z"]
        <= manager.FOLLOW_MAX_TURN_SPEED
    )

    print()
    print("===== GENTLE RIGHT CORRECTION =====")
    gentle_right = manager.execute(mission)
    print(gentle_right)

    print()
    print("===== STRONG RIGHT CORRECTION =====")
    strong_right = manager.execute(mission)
    print(strong_right)

    assert gentle_right["state"] == "CENTERING_RIGHT"
    assert strong_right["state"] == "CENTERING_RIGHT"

    assert (
        gentle_right["commanded_angular_z"]
        < 0.0
    )

    assert (
        abs(
            strong_right[
                "commanded_angular_z"
            ]
        )
        >
        abs(
            gentle_right[
                "commanded_angular_z"
            ]
        )
    )

    assert (
        abs(
            strong_right[
                "commanded_angular_z"
            ]
        )
        <= manager.FOLLOW_MAX_TURN_SPEED
    )

    print()
    print("===== FORWARD WITH LEFT STEERING =====")
    approach_left = manager.execute(mission)
    print(approach_left)

    assert approach_left["state"] == "APPROACHING"

    assert (
        approach_left["commanded_linear_x"]
        == manager.FOLLOW_FORWARD_SPEED
    )

    assert (
        approach_left["commanded_angular_z"]
        > 0.0
    )

    print()
    print("===== FORWARD WITH RIGHT STEERING =====")
    approach_right = manager.execute(mission)
    print(approach_right)

    assert approach_right["state"] == "APPROACHING"

    assert (
        approach_right["commanded_angular_z"]
        < 0.0
    )

    print()
    print("===== CENTERED FORWARD MOTION =====")
    centered = manager.execute(mission)
    print(centered)

    assert centered["state"] == "APPROACHING"
    assert centered["commanded_angular_z"] == 0.0

    print()
    print("===== MAINTAIN DISTANCE =====")
    close = manager.execute(mission)
    print(close)

    assert close["state"] == "MAINTAINING_DISTANCE"
    assert close["completed"] is False
    assert robot.commands[-1][0] == "stop"

    motion_commands = [
        command
        for command in robot.commands
        if command[0] == "motion"
    ]

    assert len(motion_commands) == 7

    for _, command in motion_commands:
        assert command["duration"] <= 0.45

    print()
    print("PASS: small errors produce gentle corrections")
    print("PASS: large errors produce stronger corrections")
    print("PASS: turn speed is safely clamped")
    print("PASS: forward approach includes small steering")
    print("PASS: centered approach has zero angular command")
    print("PASS: every motion remains bounded")
    print("PASS: close person causes a safe stop")
    print()
    print("Adaptive FOLLOW_PERSON test passed.")


if __name__ == "__main__":
    main()
