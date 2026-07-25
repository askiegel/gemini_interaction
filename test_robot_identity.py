#!/usr/bin/env python3

from config.config_manager import ConfigurationManager
from robot_identity import (
    RobotIdentity,
    RobotIdentityError,
    get_robot_identity,
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


def assert_raises(
    expected_exception,
    function,
    message,
):
    try:
        function()
    except expected_exception:
        print(f"PASS: {message}")
        return

    raise AssertionError(
        f"{message}\n"
        f"Expected {expected_exception.__name__}."
    )


def main():
    print("==========================================")
    print("ROBOT IDENTITY TEST")
    print("==========================================")

    config = ConfigurationManager().get_config()
    identity = get_robot_identity(config)

    assert_equal(
        identity.id,
        "mayday",
        "Mayday has a stable robot ID",
    )

    assert_equal(
        identity.name,
        "Mayday",
        "Mayday has a human-readable display name",
    )

    assert_true(
        identity.matches_alias("May Day"),
        "Mayday recognizes a speech-recognition alias",
    )

    assert_equal(
        identity.role,
        "primary",
        "Mayday is configured as the primary robot",
    )

    serialized = identity.to_dict()

    assert_equal(
        serialized["id"],
        "mayday",
        "serialized identity exposes id",
    )

    assert_equal(
        serialized["name"],
        "Mayday",
        "serialized identity exposes display name",
    )

    assert_true(
        isinstance(
            serialized["voice_aliases"],
            list,
        ),
        "serialized aliases are JSON-compatible",
    )

    assert_raises(
        RobotIdentityError,
        lambda: RobotIdentity.from_config(
            {
                "robot": {
                    "name": "Missing ID",
                    "model": "Mini Pupper 2",
                    "hostname": "robot",
                }
            }
        ),
        "missing robot ID is rejected",
    )

    print()
    print("All Robot Identity tests passed.")


if __name__ == "__main__":
    main()
