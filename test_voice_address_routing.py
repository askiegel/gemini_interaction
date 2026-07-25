#!/usr/bin/env python3

from unittest.mock import patch

import voice_command
from robot_addressing import RobotAddressParser
from robot_fleet import load_robot_fleet
from robot_identity import get_robot_identity


class FakeProvider:
    def __init__(self):
        self.received_text = []

    def get_intent(self, user_text):
        self.received_text.append(user_text)

        return {
            "intent": "FOLLOW_PERSON",
            "speech": "Okay, I will follow you.",
            "target": "person",
        }


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

    print(f"PASS: {message}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)

    print(f"PASS: {message}")


def main():
    print("==========================================")
    print("VOICE ADDRESS ROUTING TEST")
    print("==========================================")

    identity = get_robot_identity()
    fleet = load_robot_fleet(
        local_identity=identity,
    )

    parser = RobotAddressParser(
        local_identity=identity,
        known_identities=fleet.remote_identities,
    )

    print()
    print("===== FLEET LOAD =====")

    assert_equal(
        identity.id,
        "mayday",
        "local robot is Mayday",
    )

    assert_equal(
        fleet.get("pypper").name,
        "Pypper",
        "Pypper is loaded from fleet configuration",
    )

    print()
    print("===== DIRECT PARSER ROUTING =====")

    addressed = parser.parse(
        "Mayday, follow me."
    )

    assert_equal(
        addressed.addressed_robot_id,
        "mayday",
        "Mayday address is recognized",
    )

    assert_equal(
        addressed.command_text,
        "follow me",
        "Mayday name is removed",
    )

    remote = parser.parse(
        "Pypper, follow me."
    )

    assert_equal(
        remote.addressed_robot_id,
        "pypper",
        "Pypper address is recognized",
    )

    assert_true(
        not remote.is_for("mayday"),
        "Pypper command is not for Mayday",
    )

    print()
    print("===== MAYDAY COMMAND PIPELINE =====")

    mayday_provider = FakeProvider()

    with patch(
        "voice_command.create_provider",
        return_value=mayday_provider,
    ):
        result = voice_command.run_command(
            "Mayday, follow me.",
            execute=False,
            submit_runtime=False,
        )

    assert_equal(
        mayday_provider.received_text,
        ["follow me"],
        "Gemini receives command text without robot name",
    )

    assert_true(
        result["ok"],
        "Mayday command completes dry-run processing",
    )

    print()
    print("===== PYPPER COMMAND PIPELINE =====")

    pypper_provider = FakeProvider()

    with patch(
        "voice_command.create_provider",
        return_value=pypper_provider,
    ):
        result = voice_command.run_command(
            "Pypper, follow me.",
            execute=False,
            submit_runtime=False,
        )

    assert_equal(
        pypper_provider.received_text,
        [],
        "Gemini is not called for a Pypper command",
    )

    assert_true(
        result["ignored"],
        "Mayday safely ignores Pypper command",
    )

    print()
    print("===== UNADDRESSED LOCAL COMMAND =====")

    local_provider = FakeProvider()

    with patch(
        "voice_command.create_provider",
        return_value=local_provider,
    ):
        result = voice_command.run_command(
            "Follow me.",
            execute=False,
            submit_runtime=False,
        )

    assert_equal(
        local_provider.received_text,
        ["follow me"],
        "unaddressed command remains local",
    )

    print()
    print("===== BROADCAST STOP ROUTING =====")

    stop = parser.parse("Everybody stop.")

    assert_true(
        stop.broadcast,
        "Everybody stop is broadcast",
    )

    assert_true(
        stop.is_for("mayday"),
        "broadcast stop applies to Mayday",
    )

    print()
    print("All Voice Address Routing tests passed.")
    print("No Runtime API, ROS, or robot motion command was sent.")


if __name__ == "__main__":
    main()
