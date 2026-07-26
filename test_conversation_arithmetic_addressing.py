#!/usr/bin/env python3

from conversation_manager import ConversationManager
from conversation_service import ConversationService
from robot_addressing import RobotAddressParser
from robot_fleet import load_robot_fleet
from robot_identity import get_robot_identity


class ForbiddenProvider:
    def __init__(self):
        self.call_count = 0

    def get_conversation_decision(self, user_text, history):
        del user_text
        del history
        self.call_count += 1
        raise AssertionError(
            "Gemini must not be called for recognized arithmetic."
        )


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected!r}\nActual: {actual!r}"
        )

    print(f"PASS: {message}")


def main():
    identity = get_robot_identity()
    fleet = load_robot_fleet(
        local_identity=identity,
    )
    parser = RobotAddressParser(
        local_identity=identity,
        known_identities=fleet.remote_identities,
    )

    parser_cases = [
        (
            "What is 17 * 23?",
            "what is 17 * 23",
        ),
        (
            "What is 100 / 4?",
            "what is 100 / 4",
        ),
        (
            "What is 12 + 8?",
            "what is 12 + 8",
        ),
        (
            "Calculate (2 + 3) * 4.",
            "calculate (2 + 3) * 4",
        ),
        (
            "What is 7.5 * 2?",
            "what is 7.5 * 2",
        ),
        (
            "Mayday, what is 17 * 23?",
            "what is 17 * 23",
        ),
    ]

    for spoken, expected in parser_cases:
        result = parser.parse(spoken)

        assert_equal(
            result.command_text,
            expected,
            f"addressing preserves arithmetic in {spoken}",
        )

    provider = ForbiddenProvider()
    manager = ConversationManager(
        provider=provider,
        max_history_turns=4,
    )
    service = ConversationService(
        conversation_manager=manager,
        local_identity=identity,
        address_parser=parser,
        mission_submitter=lambda **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    f"Math must not submit a mission: {kwargs}"
                )
            )
        ),
    )

    unaddressed = service.process_text(
        "What is 17 * 23?"
    )
    assert_equal(
        unaddressed.reply,
        "The answer is 391.",
        "unaddressed symbolic math reaches deterministic calculator",
    )

    addressed = service.process_text(
        "Mayday, what is 100 / 4?"
    )
    assert_equal(
        addressed.reply,
        "The answer is 25.",
        "addressed symbolic math reaches deterministic calculator",
    )

    assert_equal(
        provider.call_count,
        0,
        "symbolic arithmetic bypasses Gemini after addressing",
    )
    assert_equal(
        unaddressed.mission_submitted,
        False,
        "unaddressed math creates no mission",
    )
    assert_equal(
        addressed.mission_submitted,
        False,
        "addressed math creates no mission",
    )

    stop = parser.parse("Stop!")
    assert_equal(
        stop.emergency_stop,
        True,
        "emergency STOP normalization remains intact",
    )

    print()
    print("Conversation arithmetic addressing test passed.")


if __name__ == "__main__":
    main()
