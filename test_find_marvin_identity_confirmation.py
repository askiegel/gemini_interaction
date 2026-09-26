"""Offline contracts for explicit, stationary Marvin identity confirmation."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from behavior_manager import BehaviorManager
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler
from world_model import WorldModel


def stamp(offset_seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def preview(**updates):
    box = {"x1": 100.0, "y1": 80.0, "x2": 260.0, "y2": 400.0}
    value = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target": "marvin",
        "target_found": True,
        "source": "marvin_local_tracker",
        "identity_confirmed": True,
        "identity_ambiguous": False,
        "source_timestamp": stamp(),
        "bbox": dict(box),
        "image_width": 640,
        "image_height": 480,
        "target_confidence": 0.91,
        "confirmation_diagnostics": {
            "confirmation_status": "target_confirmed",
            "qualified_support_reached": True,
            "distinct_fresh_timestamps": 2,
            "minimum_support": 2,
            "marvin_geometry_filter_applied": True,
            "marvin_geometry_candidates_after": 1,
        },
        "target_observation": {
            "found": True,
            "stale": False,
            "target": "marvin",
            "label": "marvin",
            "source": "marvin_local_tracker",
            "source_timestamp": None,
            "bbox": dict(box),
            "image_width": 640,
            "image_height": 480,
            "identity_ambiguous": False,
        },
    }
    value["target_observation"]["source_timestamp"] = value["source_timestamp"]
    value.update(updates)
    return value


def manager(tmp_path, preview_value=None):
    world = WorldModel(str(tmp_path / "world-model.json"))
    behavior = BehaviorManager(robot_client=object(), world_model=world)
    current = preview() if preview_value is None else preview_value
    behavior.preview_find_object = lambda target: deepcopy(current)
    return behavior, world


def test_confirmed_fresh_preview_creates_world_model_identity_and_normal_lock(tmp_path):
    behavior, world = manager(tmp_path)
    source = behavior.preview_find_object("marvin")
    assert world.find_latest_entity_by_label("marvin", refresh=False).get("found") is not True

    result = behavior.confirm_marvin_identity_from_preview(
        source,
        now=source["source_timestamp"],
    )

    assert result["ok"] is True and result["confirmed"] is True
    assert result["identity_created"] is True and result["identity_reused"] is False
    entity = world.get_entity(result["entity_id"])
    assert entity is not None and entity.label == "marvin"
    assert entity.entity_type == "person"
    assert entity.attributes["identity_id"] == result["identity_id"]
    assert result["identity_id"].startswith("person-identity-")
    assert result["target_lock_mode"] == "LOCKED"
    assert result["locked_identity_id"] == result["identity_id"]
    assert behavior.target_lock.snapshot()["locked_identity_id"] == result["identity_id"]
    assert result["motion_executed"] is False


def test_confirmation_uses_current_clock_when_now_is_omitted(tmp_path):
    behavior, _world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    result = behavior.confirm_marvin_identity_from_preview(value)
    assert result["ok"] is True


def test_repeated_confirmation_reuses_same_current_marvin_identity(tmp_path):
    behavior, world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    first = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    second = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert first["ok"] is True and second["ok"] is True
    assert second["identity_reused"] is True
    assert second["identity_id"] == first["identity_id"]
    assert len([entity for entity in world.entities.values() if entity.label == "marvin"]) == 1


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"target_found": False}, "preview_not_confirmed_current_marvin"),
        ({"ok": False}, "preview_not_confirmed_current_marvin"),
        ({"target": "person"}, "preview_not_confirmed_current_marvin"),
        ({"source": "vision_candidate"}, "preview_not_confirmed_current_marvin"),
        ({"identity_confirmed": False}, "preview_not_confirmed_current_marvin"),
        ({"identity_ambiguous": True}, "preview_not_confirmed_current_marvin"),
        ({"authoritative": True}, "preview_not_confirmed_current_marvin"),
        ({"confirmation_diagnostics": None}, "preview_not_confirmed_current_marvin"),
        ({"source_timestamp": "not-a-time"}, "preview_timestamp_missing_or_malformed"),
        ({"bbox": None}, "preview_geometry_invalid"),
        ({"bbox": {"x1": 2, "y1": 2, "x2": 2, "y2": 3}}, "preview_geometry_invalid"),
        ({"bbox": {"x1": 2, "y1": 3, "x2": 4, "y2": 2}}, "preview_geometry_invalid"),
        ({"bbox": {"x1": -1, "y1": 3, "x2": 4, "y2": 8}}, "preview_geometry_invalid"),
        ({"image_width": 0}, "preview_geometry_invalid"),
        ({"image_height": -1}, "preview_geometry_invalid"),
    ],
)
def test_invalid_preview_evidence_fails_before_world_model_mutation(tmp_path, updates, reason):
    value = preview(**updates)
    behavior, world = manager(tmp_path, value)
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview(
        value,
        now=stamp(5) if updates.get("source_timestamp", "present") is None else stamp(),
    )
    assert result["ok"] is False and result["reason"] == reason
    assert world.get_entities() == before


def test_missing_preview_timestamp_fails_without_mutation(tmp_path):
    value = preview()
    value["source_timestamp"] = None
    value["target_observation"]["source_timestamp"] = None
    behavior, world = manager(tmp_path, value)
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview(value)
    assert result["reason"] == "preview_timestamp_missing_or_malformed"
    assert world.get_entities() == before


def test_malformed_preview_result_fails_without_mutation(tmp_path):
    behavior, world = manager(tmp_path)
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview("not-an-object")
    assert result["reason"] == "preview_result_malformed"
    assert world.get_entities() == before


def test_world_model_write_failure_fails_closed(tmp_path, monkeypatch):
    behavior, world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    monkeypatch.setattr(
        world,
        "update_entity",
        Mock(side_effect=OSError("read-only")),
    )
    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert result["ok"] is False
    assert result["reason"] == "world_model_write_failed"
    assert world.get_entities() == []


def test_targetlock_handoff_failure_fails_closed_after_successful_persistence(
    tmp_path, monkeypatch
):
    behavior, world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    assign = Mock(return_value={
        "identity_id": "person-identity-confirmed-test",
        "identity_status": "NEW",
        "identity_ambiguous": False,
    })
    identity_manager = SimpleNamespace(assign_identity=assign)
    monkeypatch.setattr(
        "behavior_manager.PersonIdentityManager",
        lambda: identity_manager,
    )
    original_update = world.update_entity
    world.update_entity = Mock(wraps=original_update)
    behavior.target_lock.resolve = Mock(return_value={"found": False, "stale": True})
    behavior.execute_marvin_search_step = Mock()
    behavior.execute_marvin_pursuit_step = Mock()

    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])

    assert world.update_entity.call_count == 1
    assign.assert_called_once()
    assert result["ok"] is False and result["reason"] == "target_lock_resolution_failed"
    assert result["identity_confirmed"] is False
    assert result.get("target_lock_mode") != "LOCKED"
    assert result.get("locked_identity_id") != "person-identity-confirmed-test"
    assert behavior.target_lock.snapshot()["tracking_mode"] == "UNLOCKED"
    assert len(world.get_entities()) == 1
    behavior.execute_marvin_search_step.assert_not_called()
    behavior.execute_marvin_pursuit_step.assert_not_called()


def test_stale_preview_fails_without_mutation(tmp_path):
    value = preview()
    behavior, world = manager(tmp_path, value)
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview(
        value,
        now=(datetime.fromisoformat(value["source_timestamp"]) + timedelta(seconds=5)).isoformat(),
    )
    assert result["reason"] == "preview_stale"
    assert world.get_entities() == before


def test_conflicting_selected_identity_fails_closed(tmp_path):
    behavior, world = manager(tmp_path)
    behavior.target_lock.snapshot = lambda: {
        "tracking_mode": "LOCKED",
        "locked_identity_id": "person-identity-someone-else",
    }
    value = behavior.preview_find_object("marvin")
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert result["reason"] == "conflicting_selected_identity"
    assert world.get_entities() == before


def test_unrelated_stale_human_identity_is_never_reused(tmp_path):
    behavior, world = manager(tmp_path)
    world.update_entity(
        "person-007", "person", "person", 0.8, "vision_server",
        location={"cx": 180.0, "cy": 240.0, "frame": "camera"},
        attributes={"identity_id": "person-identity-stale-human"},
    )
    value = behavior.preview_find_object("marvin")
    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert result["ok"] is True
    assert result["identity_id"] != "person-identity-stale-human"


def test_existing_marvin_record_with_conflicting_identity_is_not_overwritten(tmp_path):
    behavior, world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    world.update_entity(
        "marvin-001", "marvin", "person", 0.8, "previous_confirmation",
        location={"cx": 180.0, "cy": 240.0, "frame": "camera"},
        attributes={
            "identity_id": "person-identity-existing",
            "bbox": dict(value["bbox"]),
            "image_width": 640,
            "image_height": 480,
            "area": 51200,
        },
    )
    value["identity_id"] = "person-identity-different"
    before = deepcopy(world.get_entities())
    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert result["ok"] is False
    assert result["reason"] == "conflicting_persistent_marvin_identity"
    assert world.get_entities() == before


def test_confirmation_inputs_are_not_mutated_and_targetlock_is_not_patched(tmp_path):
    behavior, world = manager(tmp_path)
    value = behavior.preview_find_object("marvin")
    before = deepcopy(value)
    original_resolve = behavior.target_lock.resolve
    behavior.target_lock.resolve = Mock(wraps=original_resolve)
    result = behavior.confirm_marvin_identity_from_preview(value, now=value["source_timestamp"])
    assert result["ok"] is True
    assert value == before
    behavior.target_lock.resolve.assert_called_once()
    source = inspect.getsource(BehaviorManager._confirm_marvin_identity_from_preview)
    assert "locked_identity_id =" not in source
    assert "tracking_mode =" not in source
    assert "target_lock.resolve(" in source


def test_confirmation_runtime_api_requires_explicit_true_and_has_no_motion_path():
    behavior = SimpleNamespace(confirm_marvin_identity_from_preview=Mock(return_value={
        "ok": True, "confirmed": True, "motion_executed": False,
    }))
    runtime = object.__new__(CognitiveRuntime)
    runtime.behavior_manager = behavior
    assert runtime.confirm_find_marvin_identity(confirm=False)["ok"] is False
    behavior.confirm_marvin_identity_from_preview.assert_not_called()
    assert runtime.confirm_find_marvin_identity(confirm=True)["ok"] is True
    behavior.confirm_marvin_identity_from_preview.assert_called_once_with()

    source = inspect.getsource(CognitiveRuntime.confirm_find_marvin_identity).lower()
    for forbidden in ("local_forward", "guarded_turn", "nav2", "submit_mission", "robot_bridge"):
        assert forbidden not in source
    behavior_source = inspect.getsource(BehaviorManager._confirm_marvin_identity_from_preview).lower()
    for forbidden in ("local_forward", "guarded_turn", "execute_marvin_search_step", "execute_marvin_pursuit_step", "nav2", "robot_bridge"):
        assert forbidden not in behavior_source


def test_confirmation_endpoint_is_post_only_and_requires_exact_true_body():
    runtime = SimpleNamespace(
        confirm_find_marvin_identity=Mock(return_value={
            "ok": True, "confirmed": True, "motion_executed": False,
        })
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/confirm-identity"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={"confirm": True})
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    assert responses[0][0] == 200
    runtime.confirm_find_marvin_identity.assert_called_once_with(confirm=True)
    handler.require_json_request = Mock(return_value={"confirm": False})
    handler.do_POST()
    assert responses[-1][0] == 400
    assert runtime.confirm_find_marvin_identity.call_count == 1
