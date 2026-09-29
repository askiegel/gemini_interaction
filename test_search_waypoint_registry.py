"""Offline contracts for the dynamic, motion-free waypoint registry."""

import ast
import inspect
import math

import pytest

import search_waypoint_registry as registry_module
from search_waypoint_registry import (
    DuplicateWaypointError,
    SearchWaypointRegistry,
    UnknownWaypointError,
    WaypointValidationError,
)


def make_registry(tmp_path, clock=None):
    return SearchWaypointRegistry(tmp_path / "waypoints.json", clock=clock)


def add(registry, waypoint_id="office", **updates):
    values = {
        "waypoint_id": waypoint_id,
        "name": "Office",
        "x": 1.25,
        "y": -0.5,
        "yaw": 0.75,
    }
    values.update(updates)
    return registry.add_waypoint(**values)


def test_add_and_get_waypoint(tmp_path):
    registry = make_registry(tmp_path, clock=lambda: "created")
    waypoint = add(registry)

    assert waypoint == registry.get_waypoint("office")
    assert waypoint.active is True
    assert waypoint.created_at == waypoint.updated_at == "created"


def test_multiple_waypoints_preserve_deterministic_insertion_order(tmp_path):
    registry = make_registry(tmp_path)
    add(registry, "office")
    add(registry, "studio", name="Studio", x=2, y=3, yaw=-1)

    assert [waypoint.waypoint_id for waypoint in registry.list_waypoints()] == [
        "office", "studio",
    ]


def test_active_list_excludes_deactivated_waypoint_and_reactivation_restores_it(tmp_path):
    registry = make_registry(tmp_path)
    add(registry, "office")
    add(registry, "studio", name="Studio", x=2, y=3, yaw=0)

    deactivated = registry.deactivate_waypoint("office")
    assert deactivated.active is False
    assert [waypoint.waypoint_id for waypoint in registry.list_active_waypoints()] == ["studio"]

    reactivated = registry.reactivate_waypoint("office")
    assert reactivated.active is True
    assert [waypoint.waypoint_id for waypoint in registry.list_active_waypoints()] == [
        "office", "studio",
    ]


def test_update_changes_fields_and_refreshes_only_updated_at(tmp_path):
    timestamps = iter(["created", "updated"])
    registry = make_registry(tmp_path, clock=lambda: next(timestamps))
    original = add(registry)

    updated = registry.update_waypoint(
        "office", name="Office Desk", x=2.0, y=1.0, yaw=-0.25,
    )

    assert updated.name == "Office Desk"
    assert (updated.x, updated.y, updated.yaw) == (2.0, 1.0, -0.25)
    assert updated.created_at == original.created_at == "created"
    assert updated.updated_at == "updated"


def test_remove_deletes_waypoint_and_removed_id_is_explicitly_unknown(tmp_path):
    registry = make_registry(tmp_path)
    added = add(registry)

    assert registry.remove_waypoint("office") == added
    assert registry.list_waypoints() == []
    with pytest.raises(UnknownWaypointError):
        registry.get_waypoint("office")


def test_duplicate_ids_are_rejected_without_replacement(tmp_path):
    registry = make_registry(tmp_path)
    original = add(registry)

    with pytest.raises(DuplicateWaypointError):
        add(registry, name="Replacement")
    assert registry.get_waypoint("office") == original


@pytest.mark.parametrize("field, value", [
    ("waypoint_id", ""), ("waypoint_id", " office"),
    ("name", ""), ("name", " Office"),
    ("x", "1"), ("y", object()), ("yaw", True),
    ("x", math.nan), ("y", math.inf), ("yaw", -math.inf),
    ("active", 1), ("active", "true"),
])
def test_invalid_waypoint_fields_are_rejected(tmp_path, field, value):
    registry = make_registry(tmp_path)
    with pytest.raises(WaypointValidationError):
        add(registry, **{field: value})
    assert registry.list_waypoints() == []


@pytest.mark.parametrize("field, value", [
    ("name", ""), ("x", "1"), ("y", math.nan),
    ("yaw", math.inf), ("active", "false"),
])
def test_invalid_updates_are_rejected_without_mutating_waypoint(tmp_path, field, value):
    registry = make_registry(tmp_path)
    original = add(registry)

    with pytest.raises(WaypointValidationError):
        registry.update_waypoint("office", **{field: value})
    assert registry.get_waypoint("office") == original


@pytest.mark.parametrize("operation", [
    lambda registry: registry.get_waypoint("missing"),
    lambda registry: registry.update_waypoint("missing", name="Other"),
    lambda registry: registry.deactivate_waypoint("missing"),
    lambda registry: registry.reactivate_waypoint("missing"),
    lambda registry: registry.remove_waypoint("missing"),
])
def test_unknown_waypoint_operations_fail_explicitly(tmp_path, operation):
    with pytest.raises(UnknownWaypointError):
        operation(make_registry(tmp_path))


def test_deactivation_and_removal_survive_registry_reconstruction(tmp_path):
    path = tmp_path / "waypoints.json"
    registry = SearchWaypointRegistry(path)
    add(registry, "office")
    add(registry, "studio", name="Studio", x=2, y=3, yaw=0)
    registry.deactivate_waypoint("office")
    registry.remove_waypoint("studio")

    reconstructed = SearchWaypointRegistry(path)
    assert reconstructed.get_waypoint("office").active is False
    assert reconstructed.list_active_waypoints() == []
    with pytest.raises(UnknownWaypointError):
        reconstructed.get_waypoint("studio")


def test_registry_module_has_no_navigation_robot_or_marvin_behavior_dependency():
    tree = ast.parse(inspect.getsource(registry_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not imported.intersection({
        "behavior_manager", "runtime", "runtime_api", "robot_bridge",
        "rclpy", "world_model", "voice_relay",
    })
