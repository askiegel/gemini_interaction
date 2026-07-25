#!/usr/bin/env python3

from robot_addressing import RobotAddressParser
from robot_identity import RobotIdentity


def make_identity(
    robot_id,
    name,
    aliases,
    role,
    hostname,
):
    return RobotIdentity(
        robot_id=robot_id,
        display_name=name,
        voice_aliases=tuple(aliases),
        model="Mini Pupper 2",
        role=role,
        hostname=hostname,
        platform_version="1.1.0",
    )


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
    print("ROBOT ADDRESSING TEST")
    print("==========================================")

    mayday = make_identity(
        robot_id="mayday",
        name="Mayday",
        aliases=[
            "mayday",
            "may day",
        ],
        role="primary",
        hostname="minipupper",
    )

    pypper = make_identity(
        robot_id="pypper",
        name="Pypper",
        aliases=[
            "pypper",
            "pyper",
            "piper",
        ],
        role="secondary",
        hostname="minipupper2",
    )

    parser = RobotAddressParser(
        local_identity=mayday,
        known_identities=[pypper],
    )

    print()
    print("===== MAYDAY COMMAND =====")

    result = parser.parse(
        "Mayday, follow me."
    )

    assert_equal(
        result.addressed_robot_id,
        "mayday",
        "Mayday command routes to Mayday",
    )

    assert_equal(
        result.command_text,
        "follow me",
        "Mayday name is removed before intent parsing",
    )

    assert_true(
        result.is_for("mayday"),
        "Mayday accepts its addressed command",
    )

    assert_true(
        not result.is_for("pypper"),
        "Pypper rejects a command addressed to Mayday",
    )

    print()
    print("===== MAY DAY TRANSCRIPTION =====")

    result = parser.parse(
        "May day turn left"
    )

    assert_equal(
        result.addressed_robot_id,
        "mayday",
        "May Day transcription routes to Mayday",
    )

    assert_equal(
        result.command_text,
        "turn left",
        "May Day alias is removed",
    )

    print()
    print("===== PYPPER COMMAND =====")

    result = parser.parse(
        "Piper, find my backpack."
    )

    assert_equal(
        result.addressed_robot_id,
        "pypper",
        "Piper transcription routes to Pypper",
    )

    assert_equal(
        result.command_text,
        "find my backpack",
        "Pypper alias is removed before intent parsing",
    )

    assert_true(
        not result.is_for("mayday"),
        "Mayday rejects a command addressed to Pypper",
    )

    print()
    print("===== UNADDRESSED LOCAL COMMAND =====")

    result = parser.parse(
        "Turn right."
    )

    assert_equal(
        result.addressed_robot_id,
        None,
        "unaddressed command has no explicit robot ID",
    )

    assert_equal(
        result.command_text,
        "turn right",
        "unaddressed command text is preserved",
    )

    assert_true(
        result.is_for("mayday"),
        "local robot accepts an unaddressed command",
    )

    print()
    print("===== GLOBAL EMERGENCY STOP =====")

    result = parser.parse("Stop!")

    assert_true(
        result.broadcast,
        "plain emergency stop is broadcast",
    )

    assert_true(
        result.emergency_stop,
        "plain stop is classified as an emergency stop",
    )

    assert_true(
        result.is_for("mayday"),
        "broadcast stop applies to Mayday",
    )

    assert_true(
        result.is_for("pypper"),
        "broadcast stop applies to Pypper",
    )

    print()
    print("===== EXPLICIT FLEET STOP =====")

    result = parser.parse(
        "Everybody stop."
    )

    assert_true(
        result.broadcast,
        "Everybody creates a broadcast command",
    )

    assert_equal(
        result.command_text,
        "stop",
        "broadcast address is removed",
    )

    print()
    print("All Robot Addressing tests passed.")


if __name__ == "__main__":
    main()
