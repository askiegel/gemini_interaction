#!/usr/bin/env python3

from behavior_manager import BehaviorManager
from mission_types import create_mission


class FakeStreamingRobot:
    def __init__(self):
        self.commands = []

    def streaming_motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        watchdog_timeout=0.50,
    ):
        command = {
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "watchdog_timeout": float(
                watchdog_timeout
            ),
        }

        self.commands.append(
            ("streaming_motion", command)
        )

        return {
            "ok": True,
            "action": "motion",
            "mode": "streaming",
            "returned_immediately": True,
            **command,
        }

    def motion(self, *args, **kwargs):
        raise AssertionError(
            "FOLLOW_PERSON must use streaming_motion()."
        )

    def turn_left(self, *args, **kwargs):
        raise AssertionError(
            "FOLLOW_PERSON must use streaming_motion()."
        )

    def turn_right(self, *args, **kwargs):
        raise AssertionError(
            "FOLLOW_PERSON must use streaming_motion()."
        )

    def move_forward(self, *args, **kwargs):
        raise AssertionError(
            "FOLLOW_PERSON must use streaming_motion()."
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
        "confidence": 0.93,
        "cx": float(cx),
        "cy": 240.0,
        "area": float(area),
        "image_width": 640.0,
        "image_height": 480.0,
    }


def main():
    robot = FakeStreamingRobot()

    world_model = FakeWorldModel(
        [
            {
                "found": False,
                "stale": False,
                "target": "person",
            },
            person(cx=100),
            person(cx=540),
            person(cx=285),
            person(cx=355),
            person(cx=320, area=70000),
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

    print("===== CONTINUOUS SEARCH =====")

    search = manager.execute(mission)
    print(search)

    assert search["state"] == "HOLDING_NO_PREDICTION"
    assert search["streaming"] is True
    assert search["commanded_linear_x"] == 0.0
    assert search["state"] == "HOLDING_NO_PREDICTION"
    assert search["commanded_linear_x"] == 0.0
    assert search["commanded_angular_z"] == 0.0

    print()
    print("===== STREAMING LEFT CENTERING =====")

    left = manager.execute(mission)
    print(left)

    assert left["state"] == "CENTERING_LEFT"
    assert left["streaming"] is True
    assert left["commanded_angular_z"] > 0.0

    print()
    print("===== STREAMING RIGHT CENTERING =====")

    right = manager.execute(mission)
    print(right)

    assert right["state"] == "CENTERING_RIGHT"
    assert right["streaming"] is True
    assert right["commanded_angular_z"] < 0.0

    print()
    print("===== STREAMING APPROACH LEFT =====")

    approach_left = manager.execute(mission)
    print(approach_left)

    assert approach_left["state"] == "APPROACHING"
    assert approach_left["streaming"] is True
    assert approach_left["commanded_linear_x"] > 0.0
    assert approach_left["commanded_angular_z"] > 0.0

    print()
    print("===== STREAMING APPROACH RIGHT =====")

    approach_right = manager.execute(mission)
    print(approach_right)

    assert approach_right["state"] == "APPROACHING"
    assert approach_right["streaming"] is True
    assert approach_right["commanded_linear_x"] > 0.0
    assert approach_right["commanded_angular_z"] < 0.0

    print()
    print("===== CLOSE PERSON STOPS STREAMING =====")

    close = manager.execute(mission)
    print(close)

    assert close["state"] == "MAINTAINING_DISTANCE"
    assert close["completed"] is False
    assert robot.commands[-1][0] == "stop"

    streaming_commands = [
        command
        for command in robot.commands
        if command[0] == "streaming_motion"
    ]

    assert len(streaming_commands) == 5

    for _, command in streaming_commands:
        assert (
            command["watchdog_timeout"]
            == manager.FOLLOW_STREAM_WATCHDOG_SECONDS
        )

    print()
    print("PASS: missing person causes a safe observation hold")
    print("PASS: centering refreshes streaming turn commands")
    print("PASS: approach refreshes combined streaming motion")
    print("PASS: bridge watchdog timeout is always supplied")
    print("PASS: close person explicitly stops streaming")
    print()
    print("Streaming FOLLOW_PERSON test passed.")


if __name__ == "__main__":
    main()
