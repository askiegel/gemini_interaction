#!/usr/bin/env python3

import json

from conversation_manager import ConversationManager
from providers.gemini import GeminiProvider


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


def create_test_provider(response_text):
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.client = FakeClient(response_text)
    provider.model = "test-gemini-model"

    return provider


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
    print("GEMINI CONVERSATION PROVIDER TEST")
    print("==========================================")

    print()
    print("===== NATURAL CONVERSATION =====")

    provider = create_test_provider(
        json.dumps(
            {
                "reply": "Hello! What can I help you with?",
                "decision_type": "CONVERSATION",
                "mission_type": None,
                "target": None,
                "requires_confirmation": False,
            }
        )
    )

    decision = provider.get_conversation_decision(
        "Hello",
        [],
    )

    assert_equal(
        decision["decision_type"],
        "CONVERSATION",
        "Gemini conversation output is parsed as structured JSON",
    )
    assert_equal(
        decision["reply"],
        "Hello! What can I help you with?",
        "natural reply is preserved",
    )

    call = provider.client.models.calls[0]

    assert_equal(
        call["model"],
        "test-gemini-model",
        "configured Gemini model is used",
    )
    assert_equal(
        len(call["contents"]),
        3,
        "conversation request contains prompt, history, and current message",
    )
    assert_true(
        "Current human message: Hello" in call["contents"][2],
        "current user message is sent separately",
    )

    print()
    print("===== HISTORY PROPAGATION =====")

    history = [
        {
            "role": "user",
            "text": "Hi",
        },
        {
            "role": "assistant",
            "text": "Hello!",
        },
    ]

    history_provider = create_test_provider(
        json.dumps(
            {
                "reply": "Sure, I'll follow you.",
                "decision_type": "MISSION",
                "mission_type": "FOLLOW_PERSON",
                "target": "person",
                "requires_confirmation": False,
            }
        )
    )

    history_provider.get_conversation_decision(
        "Could you follow me?",
        history,
    )

    history_contents = history_provider.client.models.calls[0]["contents"][1]

    assert_true(
        '"role": "user"' in history_contents,
        "user history is included in the Gemini request",
    )
    assert_true(
        '"role": "assistant"' in history_contents,
        "assistant history is included in the Gemini request",
    )
    assert_true(
        '"text": "Hello!"' in history_contents,
        "prior assistant response is preserved",
    )

    print()
    print("===== CONVERSATION MANAGER INTEGRATION =====")

    integration_provider = create_test_provider(
        """```json
{
  "reply": "I'll look for your backpack.",
  "decision_type": "MISSION",
  "mission_type": "FIND_OBJECT",
  "target": "backpack",
  "requires_confirmation": false
}
```"""
    )

    manager = ConversationManager(
        provider=integration_provider,
        max_history_turns=4,
    )

    result = manager.process(
        "I can't remember where I left my backpack."
    )

    assert_equal(
        result.reply,
        "I'll look for your backpack.",
        "ConversationManager accepts Gemini conversational reply",
    )
    assert_equal(
        result.mission_type,
        "FIND_OBJECT",
        "ConversationManager accepts Gemini mission request",
    )
    assert_equal(
        result.target,
        "backpack",
        "ConversationManager accepts Gemini mission target",
    )
    assert_equal(
        manager.get_history(),
        [
            {
                "role": "user",
                "text": "I can't remember where I left my backpack.",
            },
            {
                "role": "assistant",
                "text": "I'll look for your backpack.",
            },
        ],
        "validated Gemini exchange is added to conversation history",
    )

    print()
    print("===== HISTORY VALIDATION =====")

    assert_raises(
        ValueError,
        lambda: provider.get_conversation_decision(
            "Hello",
            "not-a-list",
        ),
        "must be a list",
        "non-list history is rejected",
    )

    assert_raises(
        ValueError,
        lambda: provider.get_conversation_decision(
            "Hello",
            [
                {
                    "role": "robot",
                    "text": "Hello",
                }
            ],
        ),
        "invalid role",
        "invalid conversation roles are rejected",
    )

    assert_raises(
        ValueError,
        lambda: provider.get_conversation_decision(
            "Hello",
            [
                {
                    "role": "user",
                    "text": "   ",
                }
            ],
        ),
        "requires text",
        "empty history messages are rejected",
    )

    print()
    print("===== COMMAND INTERFACE PRESERVED =====")

    assert_true(
        callable(getattr(GeminiProvider, "get_intent", None)),
        "legacy get_intent method remains available",
    )
    assert_true(
        callable(
            getattr(
                GeminiProvider,
                "get_conversation_decision",
                None,
            )
        ),
        "new conversational method is available separately",
    )

    print()
    print("All Gemini conversation provider tests passed.")
    print("No Gemini API request, runtime request, or robot command was sent.")


if __name__ == "__main__":
    main()
