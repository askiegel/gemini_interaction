"""Offline contracts for the read-only Marvin episode diagnostic."""

from copy import deepcopy
from types import SimpleNamespace

from behavior_manager import BehaviorManager
from runtime_api import RuntimeAPIHandler


NOW = "2026-09-27T02:30:02Z"
STAMP = "2026-09-27T02:30:01Z"
CONFIRMATION_STAMP = "2026-09-27T02:30:00Z"
ENTITY_ID = "marvin-001"
IDENTITY_ID = "marvin-identity-1"
EPISODE_ID = "marvin-episode-1"


def preview(**overrides):
    value = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target": "marvin",
        "source": "marvin_local_tracker",
        "source_timestamp": STAMP,
        "tracker_episode_id": EPISODE_ID,
        "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 400},
        "image_width": 640,
        "image_height": 480,
        "marvin_continuity": {
            "tracker_id": 1,
            "tracker_source": "marvin_continuity_botsort",
        },
    }
    value.update(overrides)
    return value


def entity(**attribute_overrides):
    attributes = {
        "identity_id": IDENTITY_ID,
        "operator_confirmed": True,
        "identity_confirmation_source": "marvin_local_tracker_preview",
        "identity_confirmation_timestamp": CONFIRMATION_STAMP,
        "preview_candidate": {"tracker_episode_id": EPISODE_ID},
    }
    attributes.update(attribute_overrides)
    return {
        "entity_id": ENTITY_ID,
        "label": "marvin",
        "entity_type": "person",
        "attributes": attributes,
    }


class ReadOnlyWorldModel:
    def __init__(self, entities):
        self.entities = deepcopy(entities)
        self.get_entities_calls = 0

    def get_entities(self):
        self.get_entities_calls += 1
        return deepcopy(self.entities)

    def update_entity(self, **_kwargs):
        raise AssertionError("diagnostic wrote World Model")

    def reload(self):
        raise AssertionError("diagnostic reloaded World Model")

    def save(self):
        raise AssertionError("diagnostic saved World Model")


class ReadOnlyTargetLock:
    def __init__(self, state=None):
        self.state = state or {
            "tracking_mode": "LOCKED",
            "locked_entity_id": ENTITY_ID,
            "locked_identity_id": IDENTITY_ID,
        }
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return deepcopy(self.state)

    def resolve(self, **_kwargs):
        raise AssertionError("diagnostic resolved TargetLock")


class ForbiddenRobot:
    def __getattr__(self, name):
        raise AssertionError(f"diagnostic touched robot.{name}")


def manager(world=None, lock=None, preview_value=None):
    instance = object.__new__(BehaviorManager)
    instance.world_model = world or ReadOnlyWorldModel([entity()])
    instance.target_lock = lock or ReadOnlyTargetLock()
    instance.robot = ForbiddenRobot()
    instance.preview_find_object = lambda _target: deepcopy(
        preview() if preview_value is None else preview_value
    )
    return instance


def test_read_only_diagnostic_evaluates_confirmed_episode_without_mutation():
    world = ReadOnlyWorldModel([entity()])
    lock = ReadOnlyTargetLock()
    result = manager(world, lock).build_marvin_identity_episode_diagnostic(now=NOW)

    assert result["accepted"] is True
    assert result["reason"] == "same_confirmed_tracker_episode"
    assert result["entity_id"] == ENTITY_ID
    assert result["identity_id"] == IDENTITY_ID
    assert result["preview_marvin_continuity"] == {
        "tracker_id": 1,
        "tracker_source": "marvin_continuity_botsort",
    }
    assert result["identity_episode_continuity"]["marvin_continuity"] == {
        "available": True,
        "tracker_id": 1,
        "tracker_source": "marvin_continuity_botsort",
    }
    assert world.get_entities_calls == 1
    assert lock.snapshot_calls == 1


def test_missing_or_malformed_continuity_metadata_fails_closed_without_mutation():
    missing = preview()
    missing.pop("marvin_continuity")
    malformed = preview(marvin_continuity={
        "tracker_id": "not-a-real-tracker-id",
        "tracker_source": "marvin_continuity_botsort",
    })
    for candidate in (missing, malformed):
        result = manager(
            preview_value=candidate
        ).build_marvin_identity_episode_diagnostic(now=NOW)

        assert result["accepted"] is False
        assert result["reason"] == "preview_marvin_continuity_missing_or_invalid"


def test_missing_lock_and_mismatched_entity_fail_closed_before_preview():
    no_lock = ReadOnlyTargetLock({
        "tracking_mode": "UNLOCKED",
        "locked_entity_id": None,
        "locked_identity_id": None,
    })
    instance = manager(lock=no_lock)
    instance.preview_find_object = lambda _target: (_ for _ in ()).throw(
        AssertionError("preview should not run without a lock")
    )
    result = instance.build_marvin_identity_episode_diagnostic(now=NOW)
    assert result["accepted"] is False
    assert result["reason"] == "marvin_target_lock_not_locked"

    mismatched = ReadOnlyWorldModel([entity(identity_id="other-identity")])
    result = manager(world=mismatched).build_marvin_identity_episode_diagnostic(
        now=NOW
    )
    assert result["accepted"] is False
    assert result["reason"] == "marvin_confirmed_entity_missing_or_mismatched"


def test_get_route_exposes_only_read_only_diagnostic_result():
    episode = {
        "ok": True,
        "identity_continuity": True,
        "reason": "same_confirmed_tracker_episode",
        "selected_identity_id": IDENTITY_ID,
        "entity_id": ENTITY_ID,
        "episode_valid": True,
        "marvin_continuity": {
            "available": True,
            "tracker_id": 1,
            "tracker_source": "marvin_continuity_botsort",
        },
    }
    expected = {
        "preview_marvin_continuity": {
            "tracker_id": 1,
            "tracker_source": "marvin_continuity_botsort",
        },
        "entity_id": ENTITY_ID,
        "identity_id": IDENTITY_ID,
        "identity_episode_continuity": episode,
        "accepted": True,
        "reason": "same_confirmed_tracker_episode",
    }

    class Behavior:
        def build_marvin_identity_episode_diagnostic(self):
            return expected

        def __getattr__(self, name):
            raise AssertionError(f"GET invoked behavior.{name}")

    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/identity-episode"
    handler.server = SimpleNamespace(
        runtime=SimpleNamespace(behavior_manager=Behavior())
    )
    responses = []
    handler.send_json = lambda status, payload: responses.append((status, payload))

    handler.do_GET()

    assert responses == [(200, expected)]
