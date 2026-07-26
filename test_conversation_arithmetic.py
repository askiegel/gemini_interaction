#!/usr/bin/env python3

from arithmetic_service import answer_arithmetic_question
from conversation_manager import ConversationManager


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
    cases = [
        ("What is 17 * 23?", "The answer is 391."),
        ("What is 12 plus 8?", "The answer is 20."),
        ("What is 100 divided by 4?", "The answer is 25."),
        ("Calculate (2 + 3) * 4", "The answer is 20."),
        ("How much is 9 minus 14?", "The answer is -5."),
        ("What is 7.5 times 2?", "The answer is 15."),
    ]

    for question, expected in cases:
        assert_equal(
            answer_arithmetic_question(question),
            expected,
            f"safe arithmetic works for {question}",
        )

    assert_equal(
        answer_arithmetic_question("What is 10 divided by 0?"),
        "I cannot divide by zero.",
        "division by zero is handled conversationally",
    )

    assert_equal(
        answer_arithmetic_question("What is 2 ** 10?"),
        (
            "I can calculate numbers using addition, subtraction, "
            "multiplication, division, and parentheses."
        ),
        "unsupported operators are rejected",
    )

    assert_equal(
        answer_arithmetic_question(
            "What is __import__('os').system('echo unsafe')?"
        ),
        (
            "I can calculate numbers using addition, subtraction, "
            "multiplication, division, and parentheses."
        ),
        "code execution input is rejected",
    )

    assert_equal(
        answer_arithmetic_question("Tell me about robots."),
        None,
        "non-math conversation is left for Gemini",
    )

    provider = ForbiddenProvider()
    manager = ConversationManager(
        provider=provider,
        max_history_turns=4,
    )

    result = manager.process("What is 17 times 23?")

    assert_equal(
        result.reply,
        "The answer is 391.",
        "ConversationManager returns deterministic math answer",
    )
    assert_equal(
        result.decision_type,
        "CONVERSATION",
        "math answer is a conversation decision",
    )
    assert_equal(
        result.has_mission,
        False,
        "math answer creates no mission",
    )
    assert_equal(
        provider.call_count,
        0,
        "recognized arithmetic bypasses Gemini",
    )
    assert_equal(
        manager.get_history(),
        [
            {
                "role": "user",
                "text": "What is 17 times 23?",
            },
            {
                "role": "assistant",
                "text": "The answer is 391.",
            },
        ],
        "deterministic math exchange enters conversation history",
    )

    print()
    print("Deterministic conversational arithmetic test passed.")


if __name__ == "__main__":
    main()
