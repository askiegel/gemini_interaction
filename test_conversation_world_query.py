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
    def __init__(self):
        self.history = []

    def process(self, user_text):
        self.history.append(
            {
                "role": "user",
                "text": user_text,
            }
        )

        return ConversationResult(
            reply="Let me check my World Model.",
            decision_type="WORLD_QUERY",
            mission_type=None,
            target=None,
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
    "entity question routes to LATEST_ENTITY",
)
assert_equal(
    result.world_query["target"],
    "backpack",
    "spoken possessive target is normalized",
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
    result.world_query["query_type"],
    "LIST_ENTITIES",
    "object-list question routes to LIST_ENTITIES",
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
    result.world_query["query_type"],
    "VISION_STATUS",
    "vision question routes to VISION_STATUS",
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
    result.world_query["query_type"],
    "CURRENT_MISSION",
    "mission question routes to CURRENT_MISSION",
)
assert_true(
    "do not currently have an active mission" in (
        result.reply.lower()
    ),
    "no-active-mission reply is deterministic",
)

print()
print("===== UNSUPPORTED WORLD QUERY =====")

result = service.process_text(
    "What color is the room?",
    submit_missions=True,
)

assert_equal(
    result.world_query["ok"],
    False,
    "unsupported World Query returns safe failure",
)
assert_equal(
    runtime_calls,
    [],
    "unsupported World Query still does not contact runtime",
)
assert_true(
    "could not determine" in result.reply.lower(),
    "unsupported World Query gets safe spoken response",
)

assert_true(
    world_model.load_calls >= 4,
    "shared World Model is refreshed for supported queries",
)

print()
print("==================================================")
print("PASS: Conversation World Query integration suite")
print("==================================================")
