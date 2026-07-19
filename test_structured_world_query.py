#!/usr/bin/env python3

from conversation_manager import (
    ConversationError,
    ConversationManager,
)
from conversation_service import ConversationService
from world_query_service import WorldQueryResult


class ScriptedProvider:
    def __init__(self, decision):
        self.decision = decision

    def get_conversation_decision(self, user_text, history):
        return dict(self.decision)


class RecordingWorldQueryService:
    def __init__(self):
        self.calls = []

    def execute(self, query_type, target=None):
        self.calls.append(
            {
                "query_type": query_type,
                "target": target,
            }
        )

        return WorldQueryResult(
            ok=True,
            query_type=query_type,
            target=target,
            reply=f"Structured query executed: {query_type}.",
            data={
                "source": "test",
            },
        )

    def execute_text(self, user_text):
        raise AssertionError(
            "execute_text() must not be used by ConversationService."
        )


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

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


def create_manager(decision):
    return ConversationManager(
        provider=ScriptedProvider(decision),
    )


def main():
    print("==========================================")
    print("STRUCTURED WORLD QUERY TEST")
    print("==========================================")

    print()
    print("===== MANAGER VALIDATION =====")

    manager = create_manager(
        {
            "reply": "I'll check what I remember.",
            "decision_type": "WORLD_QUERY",
            "mission_type": None,
            "query_type": "latest_entity",
            "target": "backpack",
            "requires_confirmation": False,
        }
    )

    result = manager.process(
        "Where did you last see my backpack?"
    )

    assert_equal(
        result.decision_type,
        "WORLD_QUERY",
        "WORLD_QUERY decision is preserved",
    )
    assert_equal(
        result.query_type,
        "LATEST_ENTITY",
        "query_type is normalized and validated",
    )
    assert_equal(
        result.target,
        "backpack",
        "LATEST_ENTITY target is preserved",
    )
    assert_equal(
        result.has_mission,
        False,
        "world query does not advertise a mission",
    )

    print()
    print("===== SERVICE DIRECT EXECUTION =====")

    world_query_service = RecordingWorldQueryService()

    service = ConversationService(
        conversation_manager=create_manager(
            {
                "reply": "I'll check my World Model.",
                "decision_type": "WORLD_QUERY",
                "mission_type": None,
                "query_type": "LATEST_ENTITY",
                "target": "backpack",
                "requires_confirmation": False,
            }
        ),
        mission_submitter=lambda **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "World queries must not submit missions."
                )
            )
        ),
        world_query_service=world_query_service,
    )

    service_result = service.process_text(
        "Where did you last see my backpack?"
    )

    assert_equal(
        world_query_service.calls,
        [
            {
                "query_type": "LATEST_ENTITY",
                "target": "backpack",
            }
        ],
        "ConversationService calls execute() with structured fields",
    )
    assert_equal(
        service_result.reply,
        "Structured query executed: LATEST_ENTITY.",
        "deterministic WorldQueryService reply replaces acknowledgement",
    )
    assert_equal(
        service_result.query_type,
        "LATEST_ENTITY",
        "service result exposes the executed query type",
    )
    assert_equal(
        service_result.target,
        "backpack",
        "service result exposes the world-query target",
    )
    assert_equal(
        service_result.mission_submitted,
        False,
        "structured world query does not submit a mission",
    )

    print()
    print("===== INVALID QUERY VALIDATION =====")

    assert_raises(
        ConversationError,
        lambda: create_manager(
            {
                "reply": "Checking.",
                "decision_type": "WORLD_QUERY",
                "query_type": "DELETE_WORLD_MODEL",
            }
        ).process("Delete your memory."),
        "Unsupported query_type",
        "unsupported world query types are rejected",
    )

    assert_raises(
        ConversationError,
        lambda: create_manager(
            {
                "reply": "Checking.",
                "decision_type": "WORLD_QUERY",
            }
        ).process("What do you see?"),
        "WORLD_QUERY decisions require query_type",
        "WORLD_QUERY requires a structured query type",
    )

    assert_raises(
        ConversationError,
        lambda: create_manager(
            {
                "reply": "Checking.",
                "decision_type": "WORLD_QUERY",
                "query_type": "LATEST_ENTITY",
            }
        ).process("Have you seen it?"),
        "LATEST_ENTITY requires a target",
        "LATEST_ENTITY requires an explicit target",
    )

    assert_raises(
        ConversationError,
        lambda: create_manager(
            {
                "reply": "Checking.",
                "decision_type": "WORLD_QUERY",
                "query_type": "VISION_STATUS",
                "target": "backpack",
            }
        ).process("Is vision working?"),
        "Only LATEST_ENTITY",
        "non-entity world queries cannot hide a target",
    )

    assert_raises(
        ConversationError,
        lambda: create_manager(
            {
                "reply": "Hello.",
                "decision_type": "CONVERSATION",
                "query_type": "VISION_STATUS",
            }
        ).process("Hello."),
        "Only WORLD_QUERY decisions",
        "conversation decisions cannot hide a world query",
    )

    print()
    print("All structured World Query tests passed.")
    print("No runtime, ROS, Gemini API, or robot command was used.")


if __name__ == "__main__":
    main()
