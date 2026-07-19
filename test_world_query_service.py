#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from world_query_service import (
    WorldQueryError,
    WorldQueryService,
)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}: expected {expected!r}, got {actual!r}"
        )

    print(f"PASS: {message}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)

    print(f"PASS: {message}")


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
                    now - timedelta(seconds=12)
                ).isoformat(),
                "confidence": 0.88,
                "attributes": {
                    "position": "left",
                    "tracking": False,
                },
                "history": [],
            },
            {
                "entity_id": "person-001",
                "label": "person",
                "entity_type": "human",
                "last_seen": now.isoformat(),
                "confidence": 0.91,
                "attributes": {
                    "position": "center",
                    "tracking": True,
                },
                "history": [],
            },
        ]

    def load(self):
        self.load_calls += 1

    def get_entities(self):
        return list(self.entities)


world = FakeWorldModel()
service = WorldQueryService(world)

print("===== LATEST ENTITY =====")

result = service.execute(
    "LATEST_ENTITY",
    target="backpack",
)

assert_true(result.ok, "latest-entity query succeeds")
assert_equal(
    result.query_type,
    "LATEST_ENTITY",
    "query type is preserved",
)
assert_equal(
    result.target,
    "backpack",
    "target is preserved",
)
assert_true(
    result.data["found"],
    "backpack is found",
)
assert_equal(
    result.data["position"],
    "left",
    "backpack position is returned",
)
assert_true(
    "last saw the backpack" in result.reply.lower(),
    "reply describes the last backpack observation",
)
assert_true(
    "left" in result.reply.lower(),
    "reply includes backpack position",
)

print()
print("===== CURRENTLY TRACKED PERSON =====")

person_result = service.execute(
    "LATEST_ENTITY",
    target="person",
)

assert_true(
    "currently tracking" in person_result.reply.lower(),
    "tracked person is described as currently tracked",
)
assert_true(
    "center" in person_result.reply.lower(),
    "tracked person's position is included",
)

print()
print("===== MISSING ENTITY =====")

missing_result = service.execute(
    "LATEST_ENTITY",
    target="coffee mug",
)

assert_equal(
    missing_result.data["found"],
    False,
    "missing entity reports found=false",
)
assert_true(
    "do not have a recorded observation" in (
        missing_result.reply.lower()
    ),
    "missing entity gets deterministic answer",
)

print()
print("===== LIST ENTITIES =====")

list_result = service.execute("LIST_ENTITIES")

assert_equal(
    list_result.data["count"],
    2,
    "entity count is returned",
)
assert_equal(
    list_result.data["labels"],
    {
        "backpack": 1,
        "person": 1,
    },
    "entity label counts are returned",
)
assert_true(
    "backpack" in list_result.reply.lower(),
    "entity-list reply includes backpack",
)
assert_true(
    "person" in list_result.reply.lower(),
    "entity-list reply includes person",
)

print()
print("===== CURRENT MISSION =====")

mission_result = service.execute("CURRENT_MISSION")

assert_true(
    "do not currently have an active mission" in (
        mission_result.reply.lower()
    ),
    "no active mission is reported",
)

world.robot_state["mission"] = {
    "mission_type": "FOLLOW_PERSON",
    "target": "person",
    "status": "ACTIVE",
}

mission_result = service.execute("CURRENT_MISSION")

assert_true(
    "follow person" in mission_result.reply.lower(),
    "active mission type is reported",
)
assert_true(
    "active" in mission_result.reply.lower(),
    "active mission status is reported",
)

print()
print("===== VISION STATUS =====")

vision_result = service.execute("VISION_STATUS")

assert_true(
    "running" in vision_result.reply.lower(),
    "vision status is reported",
)

print()
print("===== VALIDATION =====")

try:
    service.execute("LATEST_ENTITY")
except WorldQueryError as exc:
    assert_true(
        "requires a target" in str(exc).lower(),
        "LATEST_ENTITY rejects missing target",
    )
else:
    raise AssertionError(
        "LATEST_ENTITY should reject a missing target."
    )

try:
    service.execute("DELETE_ENTITY")
except WorldQueryError as exc:
    assert_true(
        "unsupported world query type" in str(exc).lower(),
        "unsupported query is rejected",
    )
else:
    raise AssertionError(
        "Unsupported query should have been rejected."
    )

assert_true(
    world.load_calls >= 1,
    "World Model is refreshed before queries",
)

print()
print("==================================================")
print("PASS: World Query Service regression suite")
print("==================================================")
