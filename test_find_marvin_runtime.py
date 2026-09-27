"""Offline runtime plumbing contracts for the dry-run Find-Marvin adapter."""

from copy import deepcopy
from datetime import datetime, timezone
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from marvin_arrival_policy import evaluate_marvin_arrival
from marvin_pursuit_state import READY_TO_APPROACH, evaluate_marvin_pursuit_state
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


STAMP = datetime.now(timezone.utc).isoformat()
IDENTITY = "marvin-identity"


def preview():
    box = {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0}
    return {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "target_found": True,
        "identity_confirmed": True, "identity_id": IDENTITY,
        "source_timestamp": STAMP, "bbox": dict(box),
        "tracking": {"target_label": "marvin", "bbox": dict(box),
                     "identity_id": IDENTITY},
    }


def locked(**updates):
    value = {
        "found": True, "stale": False, "tracking_mode": "LOCKED",
        "identity_id": IDENTITY, "locked_identity_id": IDENTITY,
        "entity_id": "entity-1", "last_seen": STAMP,
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0},
        "image_width": 640.0, "image_height": 480.0,
        "cx": 194.0,
    }
    value.update(updates)
    return value


def snapshot(mode="LOCKED", identity=IDENTITY):
    return {"tracking_mode": mode, "locked_identity_id": identity}


def continuity_preview():
    value = preview()
    value.update({
        "source": "marvin_local_tracker",
        "entity_id": "entity-1",
    })
    value["target_observation"] = {
        "found": True,
        "stale": False,
        "source_timestamp": STAMP,
        "entity_id": "entity-1",
        "identity_id": IDENTITY,
        "bbox": dict(value["bbox"]),
        "image_width": 640.0,
        "image_height": 480.0,
        "identity_ambiguous": False,
    }
    return value


def confirmed_world_entity(identity=IDENTITY):
    return {
        "entity_id": "entity-1",
        "label": "marvin",
        "entity_type": "person",
        "attributes": {
            "identity_id": identity,
            "operator_confirmed": True,
            "identity_confirmation_source": "marvin_local_tracker_preview",
            "identity_confirmation_timestamp": STAMP,
        },
    }


class FakeWorldModel:
    def __init__(self, entities):
        self.entities = deepcopy(entities)
        self.update_calls = []

    def get_entities(self):
        return deepcopy(self.entities)

    def get_entity(self, entity_id):
        return next(
            (item for item in self.entities if item["entity_id"] == entity_id),
            None,
        )

    def update_entity(self, **kwargs):
        self.update_calls.append(deepcopy(kwargs))
        entity = next(
            (item for item in self.entities if item["entity_id"] == kwargs["entity_id"]),
            None,
        )
        if entity is not None:
            entity["label"] = kwargs["label"]
            entity["entity_type"] = kwargs["entity_type"]
            entity["confidence"] = kwargs["confidence"]
            entity["last_seen"] = STAMP
            entity.setdefault("attributes", {}).update(deepcopy(kwargs["attributes"]))
            entity.setdefault("history", []).append({
                "timestamp": STAMP,
                "source": kwargs["source"],
                "confidence": kwargs["confidence"],
                "location": deepcopy(kwargs["location"]),
                "attributes": deepcopy(kwargs["attributes"]),
            })


class FakeTargetLock:
    def __init__(self, result=None, state=None, on_resolve=None):
        self.result = result or locked()
        self.state = state or snapshot()
        self.mission_id = None
        self.target_label = "marvin"
        self.calls = []
        self.on_resolve = on_resolve

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_resolve is not None:
            self.on_resolve()
        return deepcopy(self.result)

    def snapshot(self):
        return deepcopy(self.state)


def manager(monkeypatch, target_lock=None, preview_value=None, world_model=None):
    value = preview() if preview_value is None else preview_value
    instance = BehaviorManager(robot_client=object())
    instance.target_lock = target_lock or FakeTargetLock()
    instance.world_model = world_model
    monkeypatch.setattr(instance, "preview_find_object", lambda _target: deepcopy(value))
    return instance


def test_state_provider_is_fresh_and_preserves_authoritative_bundle(monkeypatch):
    lock = FakeTargetLock()
    instance = manager(monkeypatch, lock)
    first = instance.build_find_marvin_controller_state(now=STAMP)
    second = instance.build_find_marvin_controller_state(now=STAMP)
    assert len(lock.calls) == 2 and first == second
    assert first["selected_identity_id"] == IDENTITY
    assert first["target_lock_snapshot"]["tracking_mode"] == "LOCKED"
    assert first["preview_result"]["authoritative"] is False
    assert first["target_lock_result"]["bbox"]["y2"] == 364.0
    assert first["identity_evidence"] == first["target_lock_result"]


def test_valid_continuity_refreshes_existing_entity_before_targetlock_resolve(monkeypatch):
    entity = confirmed_world_entity()
    world = FakeWorldModel([entity])
    lock = FakeTargetLock(on_resolve=lambda: world.update_calls or pytest.fail(
        "TargetLock.resolve must run after the authorized World Model refresh"
    ))
    instance = manager(
        monkeypatch, lock, continuity_preview(), world_model=world,
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)

    assert bundle["identity_continuity"] == {
        "ok": True,
        "allow_refresh": True,
        "reason": "same_confirmed_identity_observed",
        "entity_id": "entity-1",
        "identity_id": IDENTITY,
    }
    assert bundle["identity_refresh"]["allow_refresh"] is True
    assert bundle["identity_refresh"]["observation_update"] == {
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0},
        "image_width": 640.0,
        "image_height": 480.0,
        "source_timestamp": STAMP,
        "source": "marvin_local_tracker",
    }
    assert len(world.update_calls) == 1
    write = world.update_calls[0]
    assert write["entity_id"] == "entity-1"
    assert write["label"] == "marvin" and write["entity_type"] == "person"
    assert "identity_id" not in write["attributes"]
    assert "entity_id" not in write["attributes"]
    assert world.entities[0]["entity_id"] == "entity-1"
    assert world.entities[0]["attributes"]["identity_id"] == IDENTITY
    assert world.entities[0]["attributes"]["operator_confirmed"] is True
    assert lock.snapshot() == snapshot()
    assert len(lock.calls) == 1


def test_valid_refresh_dry_run_invokes_no_motion_executor(monkeypatch):
    world = FakeWorldModel([confirmed_world_entity()])
    instance = manager(
        monkeypatch,
        FakeTargetLock(),
        continuity_preview(),
        world_model=world,
    )
    executor_calls = []
    monkeypatch.setattr(
        instance,
        "execute_marvin_search_step",
        lambda *args, **kwargs: executor_calls.append("search"),
    )
    monkeypatch.setattr(
        instance,
        "execute_marvin_pursuit_step",
        lambda *args, **kwargs: executor_calls.append("pursuit"),
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "state": "SEARCHING",
            "pursuit_authorized": False,
            "selected_identity_id": IDENTITY,
            "reason": "no_current_lock",
        },
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_arrival",
        lambda *args, **kwargs: {
            "ok": True,
            "arrived_at_marvin": False,
            "selected_identity_id": IDENTITY,
            "reason": "not_at_standoff",
        },
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)
    result = instance.execute_find_marvin_controller(
        lambda: bundle, max_actions=1, dry_run=True,
    )

    assert bundle["identity_refresh"]["allow_refresh"] is True
    assert result["dry_run"] is True
    assert result["actions_executed"] == 0
    assert result["next_route"] == "search"
    assert executor_calls == []


def test_failed_continuity_is_read_only_and_dry_run_calls_no_executor(monkeypatch):
    waiting = FakeTargetLock(
        locked(found=False, tracking_mode="WAITING_FOR_IDENTITY"),
        snapshot("WAITING_FOR_IDENTITY"),
    )
    original_lock = waiting.snapshot()
    world = FakeWorldModel([confirmed_world_entity()])
    original_entities = deepcopy(world.entities)
    instance = manager(monkeypatch, waiting, preview(), world_model=world)
    executor_calls = []
    monkeypatch.setattr(
        instance, "execute_marvin_search_step",
        lambda *args, **kwargs: executor_calls.append("search"),
    )
    monkeypatch.setattr(
        instance, "execute_marvin_pursuit_step",
        lambda *args, **kwargs: executor_calls.append("pursuit"),
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "state": "REACQUIRE_REQUIRED",
            "pursuit_authorized": False,
            "selected_identity_id": IDENTITY,
            "reason": "waiting_for_identity",
        },
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_arrival",
        lambda *args, **kwargs: {
            "ok": True,
            "arrived_at_marvin": False,
            "selected_identity_id": IDENTITY,
            "reason": "target_lock_not_locked",
        },
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)
    assert bundle["identity_continuity"]["allow_refresh"] is False
    assert waiting.snapshot() == original_lock
    assert world.entities == original_entities
    assert world.update_calls == []

    result = instance.execute_find_marvin_controller(
        lambda: bundle, max_actions=1, dry_run=True,
    )
    assert result["next_route"] == "search"
    assert result["actions_executed"] == 0
    assert executor_calls == []


def test_provider_bundle_feeds_pursuit_and_arrival_policies(monkeypatch):
    bundle = manager(monkeypatch).build_find_marvin_controller_state(now=STAMP)
    pursuit = evaluate_marvin_pursuit_state(
        bundle["preview_result"], bundle["target_lock_result"],
        bundle["target_lock_snapshot"], selected_identity_id=IDENTITY,
        identity_evidence=bundle["identity_evidence"],
        bridge_result=bundle["bridge_result"], now=STAMP,
    )
    arrival = evaluate_marvin_arrival(
        bundle["target_lock_result"], bundle["target_lock_snapshot"],
        selected_identity_id=IDENTITY, now=STAMP,
    )
    assert pursuit["state"] == READY_TO_APPROACH
    assert arrival["arrived_at_marvin"] is True


def test_unlocked_target_lock_uses_read_only_visual_session_builder(monkeypatch):
    waiting = FakeTargetLock(
        locked(found=False, tracking_mode="WAITING_FOR_IDENTITY"),
        snapshot("WAITING_FOR_IDENTITY"),
    )
    world = FakeWorldModel([confirmed_world_entity()])
    bundle = manager(monkeypatch, waiting, world_model=world).build_find_marvin_controller_state()
    assert bundle["selected_identity_id"] is None
    assert bundle["target_lock_snapshot"]["tracking_mode"] == "WAITING_FOR_IDENTITY"
    assert waiting.calls == [] and world.update_calls == []
    broken = manager(monkeypatch)
    monkeypatch.setattr(broken, "preview_find_object", lambda _target: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        broken.build_find_marvin_controller_state()


def test_malformed_target_lock_result_fails_closed(monkeypatch):
    instance = manager(monkeypatch, FakeTargetLock(result="invalid"))
    with pytest.raises(RuntimeError, match="target_lock_result_malformed"):
        instance.build_find_marvin_controller_state()


def test_runtime_dry_run_never_authorizes_execution_or_motion():
    runtime = object.__new__(CognitiveRuntime)
    calls = []

    class Behavior:
        def execute_find_marvin_controller(self, provider, **kwargs):
            calls.append((provider(), kwargs))
            return {"ok": True, "completed": False, "actions_executed": 0,
                    "next_route": "search", "history": [{"pursuit_state": "SEARCHING"}]}

    runtime.behavior_manager = Behavior()
    runtime.build_find_marvin_controller_state = lambda: {"fresh": True}
    value = runtime.dry_run_find_marvin_controller(execute=False)
    assert value["controller_ready"] is True and value["execution_authorized"] is False
    assert value["motion_executed"] is False and value["next_route"] == "search"
    assert calls[0][1] == {"max_actions": 1, "dry_run": True}
    rejected = runtime.dry_run_find_marvin_controller(execute=True)
    assert rejected["ok"] is False and rejected["motion_executed"] is False
    assert len(calls) == 1


def test_runtime_endpoint_accepts_only_explicit_false_execute():
    runtime = SimpleNamespace(
        dry_run_find_marvin_controller=Mock(return_value={"ok": True, "motion_executed": False})
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/controller"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={"execute": False})
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    assert responses == [(200, {"ok": True, "motion_executed": False})]
    runtime.dry_run_find_marvin_controller.assert_called_once_with(execute=False)
    handler.require_json_request = Mock(return_value={"execute": True})
    handler.do_POST()
    assert runtime.dry_run_find_marvin_controller.call_count == 1
    assert responses[-1][1]["ok"] is False


def test_runtime_path_has_no_direct_primitive_or_navigation_calls():
    runtime_source = inspect.getsource(CognitiveRuntime.dry_run_find_marvin_controller).lower()
    builder_source = inspect.getsource(BehaviorManager.build_find_marvin_controller_state).lower()
    for source in (runtime_source, builder_source):
        for forbidden in ("robot.local_forward", "execute_guarded_turn", "execute_marvin_search_step", "execute_marvin_pursuit_step", "nav2", "robot_bridge"):
            assert forbidden not in source
