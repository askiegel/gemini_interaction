#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from conversation_manager import ConversationResult
from conversation_service import ConversationService
from world_query_service import WorldQueryService


def assert_true(value, message):
    if not value:
        raise AssertionError(message)

    print(f"PASS: {message}")


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}: expected {expected!r}, got {actual!r}"
        )

    print(f"PASS: {message}")


class FakeConversationManager:
    """
    Simulate already-validated structured Gemini decisions.

    Natural-language interpretation belongs upstream of
    ConversationService. This integration test verifies that the service
    dispatches query_type and target directly to WorldQueryService.
    """

    def __init__(self):
        self.history = []

    def process(self, user_text):
        self.history.append(
            {
                "role": "user",
                "text": user_text,
            }
        )

        normalized = " ".join(
            user_text.strip().lower().split()
        )

        if "backpack" in normalized:
            query_type = "LATEST_ENTITY"
            target = "backpack"
        elif "objects" in normalized:
            query_type = "LIST_ENTITIES"
            target = None
        elif "vision" in normalized:
            query_type = "VISION_STATUS"
            target = None
        elif "mission" in normalized:
            query_type = "CURRENT_MISSION"
            target = None
        else:
            # Simulate a valid structured query whose deterministic
            # execution finds no matching entity.
            query_type = "LATEST_ENTITY"
            target = "room color"

        return ConversationResult(
            reply="Let me check my World Model.",
            decision_type="WORLD_QUERY",
            mission_type=None,
            query_type=query_type,
            target=target,
            requires_confirmation=False,
        )

    def clear_history(self):
        self.history.clear()

    def get_history(self):
        return list(self.history)


class FakeWorldModel:
    def __init__(self):
        now = datetime.now(timezone.utc)

        self.load_calls = 0
        self.robot_state = {
            "mission": None,
        }
        self.environment = {
            "vision": {
                "vision_status": "RUNNING",
                "camera_running": True,
                "last_error": None,
            },
        }
        self.entities = [
            {
                "entity_id": "backpack-001",
                "label": "backpack",
                "entity_type": "object",
                "last_seen": (
                    now - timedelta(seconds=8)
                ).isoformat(),
                "confidence": 0.88,
                "attributes": {
                    "position": "right",
                    "tracking": False,
                },
                "history": [],
            },
        ]

    def load(self):
        self.load_calls += 1

    def get_entities(self):
        return list(self.entities)


runtime_calls = []


def forbidden_runtime_submitter(**kwargs):
    runtime_calls.append(kwargs)

    raise AssertionError(
        "WORLD_QUERY must never contact the Runtime API."
    )


world_model = FakeWorldModel()

service = ConversationService(
    conversation_manager=FakeConversationManager(),
    runtime_url="http://127.0.0.1:8770",
    mission_submitter=forbidden_runtime_submitter,
    world_query_service=WorldQueryService(
        world_model=world_model,
    ),
)

print("===== ENTITY LOCATION QUERY =====")

result = service.process_text(
    "Where did you last see my backpack?",
    submit_missions=True,
)

assert_equal(
    result.decision_type,
    "WORLD_QUERY",
    "WORLD_QUERY decision is preserved",
)
assert_equal(
    result.query_type,
    "LATEST_ENTITY",
    "service result exposes structured query type",
)
assert_equal(
    result.target,
    "backpack",
    "service result exposes structured query target",
)
assert_equal(
    result.mission_submitted,
    False,
    "WORLD_QUERY does not submit a mission",
)
assert_equal(
    runtime_calls,
    [],
    "WORLD_QUERY never contacts Runtime API",
)
assert_true(
    result.world_query is not None,
    "structured World Query result is returned",
)
assert_equal(
    result.world_query["query_type"],
    "LATEST_ENTITY",
    "entity question dispatches LATEST_ENTITY",
)
assert_equal(
    result.world_query["target"],
    "backpack",
    "structured entity target is preserved",
)
assert_true(
    "backpack" in result.reply.lower(),
    "final reply comes from World Query Service",
)
assert_true(
    "right" in result.reply.lower(),
    "final reply includes stored position",
)

print()
print("===== ENTITY LIST QUERY =====")

result = service.process_text(
    "What objects can you see?",
    submit_missions=True,
)

assert_equal(
    result.query_type,
    "LIST_ENTITIES",
    "manager supplies LIST_ENTITIES query type",
)
assert_equal(
    result.world_query["query_type"],
    "LIST_ENTITIES",
    "service dispatches LIST_ENTITIES",
)
assert_true(
    "backpack" in result.reply.lower(),
    "entity-list reply contains known object",
)

print()
print("===== VISION STATUS QUERY =====")

result = service.process_text(
    "Is your vision working?",
    submit_missions=True,
)

assert_equal(
    result.query_type,
    "VISION_STATUS",
    "manager supplies VISION_STATUS query type",
)
assert_equal(
    result.world_query["query_type"],
    "VISION_STATUS",
    "service dispatches VISION_STATUS",
)
assert_true(
    "running" in result.reply.lower(),
    "vision status reply uses World Model state",
)

print()
print("===== CURRENT MISSION QUERY =====")

result = service.process_text(
    "What is your current mission?",
    submit_missions=True,
)

assert_equal(
    result.query_type,
    "CURRENT_MISSION",
    "manager supplies CURRENT_MISSION query type",
)
assert_equal(
    result.world_query["query_type"],
    "CURRENT_MISSION",
    "service dispatches CURRENT_MISSION",
)
assert_true(
    "do not currently have an active mission" in (
        result.reply.lower()
    ),
    "no-active-mission reply is deterministic",
)

print()
print("===== UNKNOWN ENTITY QUERY =====")

result = service.process_text(
    "What color is the room?",
    submit_missions=True,
)

assert_equal(
    result.world_query["query_type"],
    "LATEST_ENTITY",
    "valid structured unknown-entity query is executed",
)
assert_equal(
    result.world_query["target"],
    "room color",
    "unknown entity target is preserved",
)
assert_equal(
    result.world_query["data"]["found"],
    False,
    "unknown entity returns found=false",
)
assert_true(
    "do not have a recorded observation" in (
        result.reply.lower()
    ),
    "unknown entity gets deterministic safe response",
)
assert_equal(
    runtime_calls,
    [],
    "unknown entity query still does not contact runtime",
)

assert_true(
    world_model.load_calls >= 5,
    "shared World Model is refreshed for every query",
)

print()
print("==================================================")
print("PASS: Conversation World Query integration suite")
print("==================================================")
