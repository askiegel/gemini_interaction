#!/usr/bin/env python3

from conversation_manager import ConversationResult
from conversation_service import ConversationService
from robot_addressing import RobotAddressParser
from robot_fleet import load_robot_fleet
from robot_identity import get_robot_identity


class FakeConversationManager:
    def __init__(self):
        self.received_text = []

    def process(self, user_text):
        self.received_text.append(user_text)

        return ConversationResult(
            reply="Okay, I will follow you.",
            decision_type="MISSION",
            mission_type="FOLLOW_PERSON",
            target="person",
            requires_confirmation=False,
        )

    def clear_history(self):
        pass

    def get_history(self):
        return []


class IdentityMissionSubmitter:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        user_text,
        intent,
        runtime_url,
        robot_id=None,
        addressing=None,
    ):
        self.calls.append(
            {
                "user_text": user_text,
                "intent": dict(intent),
                "runtime_url": runtime_url,
                "robot_id": robot_id,
                "addressing": dict(addressing or {}),
            }
        )

        return {
            "ok": True,
            "accepted": True,
            "robot_id": robot_id,
            "addressing": addressing,
            "mission": {
                "mission_id": "mission-conversation-addressing",
                "mission_type": intent["intent"],
                "status": "ACTIVE",
                "target": intent.get("target"),
            },
        }


class LegacyMissionSubmitter:
    def __init__(self):
        self.calls = []

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

        return {
            "ok": True,
            "accepted": True,
            "mission": {
                "mission_id": "mission-legacy",
                "mission_type": intent["intent"],
                "status": "ACTIVE",
            },
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


def create_service(manager, submitter):
    identity = get_robot_identity()
    fleet = load_robot_fleet(
        local_identity=identity,
    )

    parser = RobotAddressParser(
        local_identity=identity,
        known_identities=fleet.remote_identities,
    )

    return ConversationService(
        conversation_manager=manager,
        mission_submitter=submitter,
        local_identity=identity,
        address_parser=parser,
    )


def main():
    print("==========================================")
    print("CONVERSATION ADDRESSING TEST")
    print("==========================================")

    print()
    print("===== MAYDAY CONVERSATION MISSION =====")

    manager = FakeConversationManager()
    submitter = IdentityMissionSubmitter()
    service = create_service(manager, submitter)

    result = service.process_text(
        "Mayday, follow me."
    )

    assert_equal(
        manager.received_text,
        ["follow me"],
        "conversation provider receives text without Mayday address",
    )

    assert_equal(
        submitter.calls[0]["user_text"],
        "Mayday, follow me.",
        "runtime preserves original spoken text",
    )

    assert_equal(
        submitter.calls[0]["robot_id"],
        "mayday",
        "runtime submission includes Mayday robot ID",
    )

    assert_equal(
        submitter.calls[0]["addressing"]["command_text"],
        "follow me",
        "runtime submission includes addressing metadata",
    )

    assert_equal(
        result.robot_id,
        "mayday",
        "conversation result includes local robot ID",
    )

    assert_true(
        result.mission_submitted,
        "Mayday mission is submitted",
    )

    assert_true(
        not result.ignored,
        "Mayday mission is not ignored",
    )

    print()
    print("===== PYPPER CONVERSATION COMMAND =====")

    remote_manager = FakeConversationManager()
    remote_submitter = IdentityMissionSubmitter()
    remote_service = create_service(
        remote_manager,
        remote_submitter,
    )

    remote_result = remote_service.process_text(
        "Pypper, follow me."
    )

    assert_equal(
        remote_manager.received_text,
        [],
        "conversation provider is not called for Pypper",
    )

    assert_equal(
        remote_submitter.calls,
        [],
        "runtime is not called for Pypper",
    )

    assert_true(
        remote_result.ignored,
        "Mayday reports Pypper command as ignored",
    )

    assert_true(
        not remote_result.accepted,
        "Pypper command is not accepted locally",
    )

    assert_equal(
        remote_result.addressing["addressed_robot_id"],
        "pypper",
        "ignored result preserves Pypper ownership",
    )

    print()
    print("===== UNADDRESSED LOCAL CONVERSATION =====")

    local_manager = FakeConversationManager()
    local_submitter = IdentityMissionSubmitter()
    local_service = create_service(
        local_manager,
        local_submitter,
    )

    local_result = local_service.process_text(
        "Follow me."
    )

    assert_equal(
        local_manager.received_text,
        ["follow me"],
        "unaddressed command remains local",
    )

    assert_equal(
        local_submitter.calls[0]["robot_id"],
        "mayday",
        "unaddressed mission is owned by Mayday",
    )

    assert_true(
        local_result.accepted,
        "unaddressed local command is accepted",
    )

    print()
    print("===== LEGACY SUBMITTER COMPATIBILITY =====")

    legacy_manager = FakeConversationManager()
    legacy_submitter = LegacyMissionSubmitter()
    legacy_service = create_service(
        legacy_manager,
        legacy_submitter,
    )

    legacy_result = legacy_service.process_text(
        "Mayday, follow me."
    )

    assert_equal(
        len(legacy_submitter.calls),
        1,
        "legacy submitter is still called",
    )

    assert_equal(
        legacy_submitter.calls[0],
        {
            "user_text": "Mayday, follow me.",
            "intent": {
                "intent": "FOLLOW_PERSON",
                "speech": "Okay, I will follow you.",
                "target": "person",
            },
            "runtime_url": "http://127.0.0.1:8770",
        },
        "legacy submitter receives its original argument contract",
    )

    assert_true(
        legacy_result.mission_submitted,
        "legacy mission submission still succeeds",
    )

    print()
    print("===== DRY-RUN ADDRESSING =====")

    dry_manager = FakeConversationManager()
    dry_submitter = IdentityMissionSubmitter()
    dry_service = create_service(
        dry_manager,
        dry_submitter,
    )

    dry_result = dry_service.process_text(
        "Mayday, follow me.",
        submit_missions=False,
    )

    assert_equal(
        dry_manager.received_text,
        ["follow me"],
        "dry-run provider receives stripped command text",
    )

    assert_equal(
        dry_submitter.calls,
        [],
        "dry-run does not contact runtime",
    )

    assert_equal(
        dry_result.robot_id,
        "mayday",
        "dry-run result preserves robot identity",
    )

    print()
    print("All Conversation Addressing tests passed.")
    print("No Gemini, Runtime API, ROS, or robot command was sent.")


if __name__ == "__main__":
    main()
