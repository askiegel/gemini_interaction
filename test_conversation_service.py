#!/usr/bin/env python3

from conversation_manager import ConversationResult
from conversation_service import ConversationService


class FakeConversationManager:
    def __init__(self, results):
        self.results = list(results)
        self.received_text = []
        self.history = [
            {
                "role": "user",
                "text": "Earlier message",
            }
        ]
        self.clear_count = 0

    def process(self, user_text):
        self.received_text.append(user_text)

        if not self.results:
            raise AssertionError(
                "FakeConversationManager has no result available."
            )

        return self.results.pop(0)

    def clear_history(self):
        self.clear_count += 1
        self.history = []

    def get_history(self):
        return list(self.history)


class FakeMissionSubmitter:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {
            "ok": True,
            "accepted": True,
            "mission": {
                "mission_id": "mission-test1234",
                "mission_type": "FOLLOW_PERSON",
                "status": "ACTIVE",
            },
        }

    def __call__(
        self,
        user_text,
        intent,
        runtime_url,
    ):
        self.calls.append(
            {
                "user_text": user_text,
                "intent": dict(intent),
                "runtime_url": runtime_url,
            }
        )

        return dict(self.response)


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
                f"Expected error containing: "
                f"{expected_message!r}\n"
                f"Actual error: {str(exc)!r}"
            ) from exc

        print(f"PASS: {message}")
        return

    raise AssertionError(
        f"{message}\n"
        f"Expected {expected_exception.__name__}."
    )


def main():
    print("==========================================")
    print("CONVERSATION SERVICE V1 TEST")
    print("==========================================")

    print()
    print("===== ORDINARY CONVERSATION =====")

    conversation_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Hello! How can I help?",
                decision_type="CONVERSATION",
            )
        ]
    )
    submitter = FakeMissionSubmitter()

    service = ConversationService(
        conversation_manager=conversation_manager,
        runtime_url="http://127.0.0.1:8770/",
        mission_submitter=submitter,
    )

    result = service.process_text("  Hello  ")

    assert_equal(
        result.reply,
        "Hello! How can I help?",
        "ordinary conversation preserves the robot reply",
    )
    assert_equal(
        result.decision_type,
        "CONVERSATION",
        "ordinary conversation preserves decision type",
    )
    assert_true(
        not result.mission_submitted,
        "ordinary conversation does not submit a mission",
    )
    assert_equal(
        submitter.calls,
        [],
        "runtime submitter is not called for conversation",
    )
    assert_equal(
        service.runtime_url,
        "http://127.0.0.1:8770",
        "runtime URL is normalized",
    )

    print()
    print("===== MISSION SUBMISSION =====")

    mission_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Sure, I'll follow you.",
                decision_type="MISSION",
                mission_type="FOLLOW_PERSON",
                target="person",
                requires_confirmation=False,
            )
        ]
    )
    mission_submitter = FakeMissionSubmitter()

    mission_service = ConversationService(
        conversation_manager=mission_manager,
        runtime_url="http://127.0.0.1:8770",
        mission_submitter=mission_submitter,
    )

    mission_result = mission_service.process_text(
        "  Could you follow me?  "
    )

    assert_true(
        mission_result.mission_submitted,
        "validated mission is submitted to the runtime",
    )
    assert_equal(
        mission_result.mission_type,
        "FOLLOW_PERSON",
        "mission type is preserved",
    )
    assert_equal(
        mission_result.target,
        "person",
        "mission target is preserved",
    )
    assert_equal(
        len(mission_submitter.calls),
        1,
        "runtime submitter is called exactly once",
    )
    assert_equal(
        mission_submitter.calls[0],
        {
            "user_text": "Could you follow me?",
            "intent": {
                "intent": "FOLLOW_PERSON",
                "speech": "Sure, I'll follow you.",
                "target": "person",
            },
            "runtime_url": "http://127.0.0.1:8770",
        },
        "conversation mission becomes a provider-independent intent",
    )
    assert_equal(
        mission_result.mission_submission["mission"][
            "mission_id"
        ],
        "mission-test1234",
        "runtime mission response is preserved",
    )

    print()
    print("===== FIND OBJECT MISSION =====")

    find_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Okay, I'll look for your backpack.",
                decision_type="MISSION",
                mission_type="FIND_OBJECT",
                target="backpack",
            )
        ]
    )
    find_submitter = FakeMissionSubmitter(
        {
            "ok": True,
            "accepted": True,
            "mission": {
                "mission_id": "mission-find001",
                "mission_type": "FIND_OBJECT",
                "status": "ACTIVE",
                "target": "backpack",
            },
        }
    )

    find_service = ConversationService(
        conversation_manager=find_manager,
        mission_submitter=find_submitter,
    )

    find_result = find_service.process_text(
        "Find my backpack."
    )

    assert_equal(
        find_submitter.calls[0]["intent"],
        {
            "intent": "FIND_OBJECT",
            "speech": "Okay, I'll look for your backpack.",
            "target": "backpack",
        },
        "FIND_OBJECT intent includes its target",
    )
    assert_true(
        find_result.mission_submitted,
        "FIND_OBJECT is submitted after validation",
    )

    print()
    print("===== DRY-RUN MISSION SAFETY =====")

    manager = FakeConversationManager(
        [
            ConversationResult(
                reply="I would follow you.",
                decision_type="MISSION",
                mission_type="FOLLOW_PERSON",
                target="person",
                requires_confirmation=False,
            )
        ]
    )

    submitter = FakeMissionSubmitter()

    service = ConversationService(
        conversation_manager=manager,
        runtime_url="http://127.0.0.1:8770",
        mission_submitter=submitter,
    )

    result = service.process_text(
        "Follow me",
        submit_missions=False,
    )

    assert_equal(
        result.mission_type,
        "FOLLOW_PERSON",
        "dry-run preserves the interpreted mission",
    )
    assert_true(
        not result.mission_submitted,
        "dry-run does not submit the mission",
    )
    assert_equal(
        len(submitter.calls),
        0,
        "dry-run never contacts the runtime submitter",
    )

    print()
    print("===== CONFIRMATION SAFETY =====")

    confirmation_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Should I move forward?",
                decision_type="MISSION",
                mission_type="MOVE_FORWARD",
                target=None,
                requires_confirmation=True,
            )
        ]
    )
    confirmation_submitter = FakeMissionSubmitter()

    confirmation_service = ConversationService(
        conversation_manager=confirmation_manager,
        mission_submitter=confirmation_submitter,
    )

    confirmation_result = confirmation_service.process_text(
        "Maybe move somewhere."
    )

    assert_true(
        confirmation_result.requires_confirmation,
        "confirmation requirement is preserved",
    )
    assert_true(
        not confirmation_result.mission_submitted,
        "unconfirmed mission is not submitted",
    )
    assert_equal(
        confirmation_submitter.calls,
        [],
        "runtime is never called for an unconfirmed mission",
    )

    print()
    print("===== WORLD QUERY =====")

    world_manager = FakeConversationManager(
        [
            ConversationResult(
                reply=(
                    "I'll check what I remember about "
                    "your backpack."
                ),
                decision_type="WORLD_QUERY",
            )
        ]
    )
    world_submitter = FakeMissionSubmitter()

    world_service = ConversationService(
        conversation_manager=world_manager,
        mission_submitter=world_submitter,
    )

    world_result = world_service.process_text(
        "Have you seen my backpack?"
    )

    assert_equal(
        world_result.decision_type,
        "WORLD_QUERY",
        "WORLD_QUERY response is preserved",
    )
    assert_true(
        not world_result.mission_submitted,
        "WORLD_QUERY does not create a robot mission",
    )
    assert_equal(
        world_submitter.calls,
        [],
        "WORLD_QUERY does not contact the runtime",
    )

    print()
    print("===== HISTORY CONTROL =====")

    history_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Hello.",
                decision_type="CONVERSATION",
            )
        ]
    )

    history_service = ConversationService(
        conversation_manager=history_manager,
        mission_submitter=FakeMissionSubmitter(),
    )

    assert_equal(
        history_service.get_history(),
        [
            {
                "role": "user",
                "text": "Earlier message",
            }
        ],
        "service exposes ConversationManager history",
    )

    history_service.clear_history()

    assert_equal(
        history_service.get_history(),
        [],
        "service clears ConversationManager history",
    )
    assert_equal(
        history_manager.clear_count,
        1,
        "history clear is delegated exactly once",
    )

    print()
    print("===== FAILURE VALIDATION =====")

    rejected_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Okay, I'll turn left.",
                decision_type="MISSION",
                mission_type="TURN_LEFT",
            )
        ]
    )
    rejected_submitter = FakeMissionSubmitter(
        {
            "ok": False,
            "accepted": False,
            "error": "Runtime unavailable.",
        }
    )

    rejected_service = ConversationService(
        conversation_manager=rejected_manager,
        mission_submitter=rejected_submitter,
    )

    assert_raises(
        RuntimeError,
        lambda: rejected_service.process_text(
            "Turn left."
        ),
        "did not accept",
        "runtime rejection is surfaced",
    )

    invalid_response_manager = FakeConversationManager(
        [
            ConversationResult(
                reply="Okay, I'll stop.",
                decision_type="MISSION",
                mission_type="STOP",
            )
        ]
    )

    invalid_response_service = ConversationService(
        conversation_manager=invalid_response_manager,
        mission_submitter=lambda **kwargs: "not-a-dictionary",
    )

    assert_raises(
        RuntimeError,
        lambda: invalid_response_service.process_text(
            "Stop."
        ),
        "must return a dictionary",
        "invalid runtime response is rejected",
    )

    assert_raises(
        ValueError,
        lambda: ConversationService(
            conversation_manager=None,
        ),
        "requires a ConversationManager",
        "missing ConversationManager is rejected",
    )

    assert_raises(
        ValueError,
        lambda: ConversationService(
            conversation_manager=history_manager,
            runtime_url="   ",
        ),
        "cannot be empty",
        "empty runtime URL is rejected",
    )

    print()
    print("===== SERIALIZATION =====")

    serialized = mission_result.to_dict()

    assert_equal(
        serialized["reply"],
        "Sure, I'll follow you.",
        "service result serializes the spoken reply",
    )
    assert_true(
        serialized["mission_submitted"],
        "service result serializes mission status",
    )
    assert_equal(
        serialized["mission_submission"]["mission"][
            "mission_id"
        ],
        "mission-test1234",
        "service result serializes runtime response",
    )

    print()
    print("All Conversation Service v1 tests passed.")
    print(
        "No Gemini request, Runtime API request, ROS command, "
        "or robot command was sent."
    )


if __name__ == "__main__":
    main()
