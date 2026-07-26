#!/usr/bin/env python3

from conversation_manager import ConversationManager


class ForbiddenProvider:
    def __init__(self):
        self.call_count = 0

    def get_conversation_decision(self, user_text, history):
        del user_text
        del history
        self.call_count += 1
        raise AssertionError(
            "Gemini must not be called for recognized calculus."
        )


def main():
    provider = ForbiddenProvider()
    manager = ConversationManager(
        provider=provider,
        max_history_turns=4,
    )

    result = manager.process(
        "Find the integral of natural log of x."
    )

    expected = (
        "The integral is x times natural log of x "
        "minus x, plus C."
    )

    if result.reply != expected:
        raise AssertionError(
            f"Unexpected calculus reply: {result.reply!r}"
        )

    if result.decision_type != "CONVERSATION":
        raise AssertionError(
            "Calculus must return a conversation decision."
        )

    if result.has_mission:
        raise AssertionError(
            "Calculus must not create a robot mission."
        )

    if provider.call_count != 0:
        raise AssertionError(
            "Recognized calculus did not bypass Gemini."
        )

    expected_history = [
        {
            "role": "user",
            "text": "Find the integral of natural log of x.",
        },
        {
            "role": "assistant",
            "text": expected,
        },
    ]

    if manager.get_history() != expected_history:
        raise AssertionError(
            "Calculus exchange was not stored in history."
        )

    print("PASS: deterministic calculus bypasses Gemini")
    print("PASS: deterministic calculus creates no mission")
    print("PASS: calculus exchange enters conversation history")
    print()
    print("Conversation symbolic math test passed.")


if __name__ == "__main__":
    main()
