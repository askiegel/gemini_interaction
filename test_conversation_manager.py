#!/usr/bin/env python3

from conversation_manager import (
    ConversationError,
    ConversationManager,
)


class ScriptedConversationProvider:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def get_conversation_decision(self, user_text, history):
        self.calls.append(
            {
                "user_text": user_text,
                "history": history,
            }
        )

        if not self.decisions:
            raise RuntimeError("No scripted decision remains.")

        decision = self.decisions.pop(0)

        if isinstance(decision, Exception):
            raise decision

        return decision


class LegacyIntentOnlyProvider:
    def get_intent(self, user_text):
        return {
            "intent": "UNKNOWN",
            "speech": "Unknown.",
            "target": None,
        }


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected!r}\nActual:   {actual!r}"
        )

    print(f"PASS: {message}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)

    print(f"PASS: {message}")


def assert_raises(
    expected_exception,
    function,
    expected_message,
    message,
):
    try:
        function()
    except expected_exception as exc:
        if expected_message not in str(exc):
            raise AssertionError(
                f"{message}\n"
                f"Expected error containing: {expected_message!r}\n"
                f"Actual error: {str(exc)!r}"
            ) from exc

        print(f"PASS: {message}")
        return

    raise AssertionError(
        f"{message}\nExpected {expected_exception.__name__}."
    )


def main():
    print("==========================================")
    print("CONVERSATION MANAGER V1 TEST")
    print("==========================================")

    provider = ScriptedConversationProvider(
        [
            {
                "reply": "Hello! What can I help you with?",
                "decision_type": "CONVERSATION",
            },
            {
                "reply": "Sure, I'll follow you.",
                "decision_type": "MISSION",
                "mission_type": "FOLLOW_PERSON",
                "target": "person",
            },
            {
                "reply": "I'll look for your backpack.",
                "decision_type": "MISSION",
                "mission_type": "FIND_OBJECT",
                "target": "backpack",
            },
        ]
    )

    manager = ConversationManager(
        provider=provider,
        max_history_turns=2,
    )

    print()
    print("===== NATURAL CONVERSATION =====")

    greeting = manager.process("  Hi  ")

    assert_equal(
        greeting.reply,
        "Hello! What can I help you with?",
        "ordinary conversation produces a reply",
    )
    assert_equal(
        greeting.decision_type,
        "CONVERSATION",
        "ordinary conversation creates no mission decision",
    )
    assert_true(
        not greeting.has_mission,
        "ordinary conversation contains no mission",
    )
    assert_equal(
        provider.calls[0]["user_text"],
        "Hi",
        "user text is normalized before provider use",
    )
    assert_equal(
        provider.calls[0]["history"],
        [],
        "first provider call starts with empty history",
    )

    print()
    print("===== FOLLOW PERSON REQUEST =====")

    follow = manager.process("Could you follow me?")

    assert_equal(
        follow.reply,
        "Sure, I'll follow you.",
        "mission request still produces a spoken reply",
    )
    assert_equal(
        follow.decision_type,
        "MISSION",
        "action request produces a mission decision",
    )
    assert_equal(
        follow.mission_type,
        "FOLLOW_PERSON",
        "FOLLOW_PERSON mission is preserved",
    )
    assert_equal(
        follow.target,
        "person",
        "FOLLOW_PERSON target is preserved",
    )
    assert_true(
        follow.has_mission,
        "FOLLOW_PERSON result advertises a mission",
    )
    assert_equal(
        provider.calls[1]["history"],
        [
            {
                "role": "user",
                "text": "Hi",
            },
            {
                "role": "assistant",
                "text": "Hello! What can I help you with?",
            },
        ],
        "next interaction receives prior conversation context",
    )

    print()
    print("===== FIND OBJECT REQUEST =====")

    find_object = manager.process(
        "I can't remember where I left my backpack."
    )

    assert_equal(
        find_object.mission_type,
        "FIND_OBJECT",
        "natural language can produce FIND_OBJECT",
    )
    assert_equal(
        find_object.target,
        "backpack",
        "FIND_OBJECT preserves the requested object",
    )

    expected_history = [
        {
            "role": "user",
            "text": "Could you follow me?",
        },
        {
            "role": "assistant",
            "text": "Sure, I'll follow you.",
        },
        {
            "role": "user",
            "text": "I can't remember where I left my backpack.",
        },
        {
            "role": "assistant",
            "text": "I'll look for your backpack.",
        },
    ]

    assert_equal(
        manager.get_history(),
        expected_history,
        "conversation history is bounded to configured turns",
    )

    print()
    print("===== HISTORY RESET =====")

    manager.clear_history()

    assert_equal(
        manager.get_history(),
        [],
        "conversation history can be reset",
    )

    print()
    print("===== DEFAULT FOLLOW TARGET =====")

    default_target_manager = ConversationManager(
        ScriptedConversationProvider(
            [
                {
                    "reply": "Okay, I'll follow you.",
                    "decision_type": "MISSION",
                    "mission_type": "FOLLOW_PERSON",
                }
            ]
        )
    )

    default_target_result = default_target_manager.process(
        "Stay with me."
    )

    assert_equal(
        default_target_result.target,
        "person",
        "FOLLOW_PERSON defaults to the person target",
    )

    print()
    print("===== SAFETY VALIDATION =====")

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider([])
        ).process("   "),
        "User text cannot be empty.",
        "empty user text is rejected",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider(
                [
                    {
                        "decision_type": "CONVERSATION",
                    }
                ]
            )
        ).process("Hello"),
        "non-empty reply",
        "provider cannot omit the robot reply",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider(
                [
                    {
                        "reply": "Executing.",
                        "decision_type": "MISSION",
                        "mission_type": "FLY",
                    }
                ]
            )
        ).process("Fly"),
        "Unsupported mission_type",
        "unknown missions are rejected before dispatch",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider(
                [
                    {
                        "reply": "Looking.",
                        "decision_type": "MISSION",
                        "mission_type": "FIND_OBJECT",
                    }
                ]
            )
        ).process("Find it"),
        "FIND_OBJECT requires a target.",
        "FIND_OBJECT cannot execute without a target",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider(
                [
                    {
                        "reply": "Hello.",
                        "decision_type": "CONVERSATION",
                        "mission_type": "MOVE_FORWARD",
                    }
                ]
            )
        ).process("Hello"),
        "Only MISSION decisions",
        "conversation responses cannot hide a mission",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            LegacyIntentOnlyProvider()
        ).process("Hello"),
        "does not implement get_conversation_decision",
        "legacy command provider is not silently treated as conversational",
    )

    assert_raises(
        ConversationError,
        lambda: ConversationManager(
            ScriptedConversationProvider(
                [
                    RuntimeError("temporary provider outage")
                ]
            )
        ).process("Hello"),
        "Conversational provider failed",
        "provider errors are wrapped consistently",
    )

    print()
    print("===== SERIALIZATION =====")

    serialized = ConversationManager(
        ScriptedConversationProvider(
            [
                {
                    "reply": "Stopping now.",
                    "decision_type": "MISSION",
                    "mission_type": "STOP",
                    "requires_confirmation": False,
                }
            ]
        )
    ).process("Stop").to_dict()

    assert_equal(
        serialized,
        {
            "reply": "Stopping now.",
            "decision_type": "MISSION",
            "mission_type": "STOP",
            "target": None,
            "requires_confirmation": False,
        },
        "conversation results serialize to provider-independent data",
    )

    print()
    print("All Conversation Manager v1 tests passed.")
    print("No microphone, runtime, ROS, or robot command was used.")


if __name__ == "__main__":
    main()
