"""Offline contracts for saving a trusted current pose as a search waypoint."""

import ast
import inspect

import pytest

import current_localized_pose as pose_module
import search_waypoint_capture as capture_module
from current_localized_pose import (
    CurrentLocalizedPoseProvider,
    LocalizedPoseUnavailableError,
)
from search_waypoint_capture import SearchWaypointCaptureService
from search_waypoint_registry import (
    DuplicateWaypointError,
    SearchWaypointRegistry,
    WaypointValidationError,
)


def valid_status(*, pose=None, age_seconds=0.1):
    if pose is None:
        pose = {
            "frame_id": "map",
            "position": {"x": 1.25, "y": -0.5},
            "yaw_radians": 0.75,
        }
    return 200, {
        "ok": True,
        "authoritative": True,
        "read_only": True,
        "source": "tony2_navigation_amcl",
        "navigation": {
            "running": True,
            "localization_enabled": True,
            "transform_ready": True,
            "localization_validated": True,
        },
        "telemetry": {
            "available": True,
            "age_seconds": age_seconds,
            "received_at": "2026-09-28T12:00:00+00:00",
            "pose": pose,
        },
    }


class Reader:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


def make_service(tmp_path, result=None):
    reader = Reader(result or valid_status())
    registry = SearchWaypointRegistry(tmp_path / "waypoints.json")
    provider = CurrentLocalizedPoseProvider(reader)
    return SearchWaypointCaptureService(registry, provider), registry, reader


def test_valid_localized_pose_is_saved_exactly_once_with_provider_geometry(tmp_path):
    service, registry, reader = make_service(tmp_path)

    saved = service.save_current_pose_as_waypoint(
        waypoint_id="office", name="Office"
    )

    assert (saved.x, saved.y, saved.yaw) == (1.25, -0.5, 0.75)
    assert saved == registry.get_waypoint("office")
    assert reader.calls == 1


def test_capture_persists_through_registry_reconstruction(tmp_path):
    path = tmp_path / "waypoints.json"
    reader = Reader(valid_status())
    service = SearchWaypointCaptureService(
        SearchWaypointRegistry(path), CurrentLocalizedPoseProvider(reader)
    )
    service.save_current_pose_as_waypoint(waypoint_id="office", name="Office")

    restored = SearchWaypointRegistry(path).get_waypoint("office")
    assert (restored.x, restored.y, restored.yaw) == (1.25, -0.5, 0.75)


@pytest.mark.parametrize("mutation", [
    lambda payload: payload[1]["telemetry"]["pose"].update(frame_id="odom"),
    lambda payload: payload[1]["navigation"].update(localization_validated=False),
    lambda payload: payload[1]["navigation"].update(transform_ready=False),
    lambda payload: payload[1]["telemetry"].update(age_seconds=3.0),
    lambda payload: payload[1]["telemetry"].update(pose=None),
])
def test_untrusted_or_unavailable_pose_fails_closed_without_writing(tmp_path, mutation):
    result = valid_status()
    mutation(result)
    service, registry, _ = make_service(tmp_path, result)

    with pytest.raises(LocalizedPoseUnavailableError):
        service.save_current_pose_as_waypoint(waypoint_id="office", name="Office")
    assert registry.list_waypoints() == []


@pytest.mark.parametrize("kwargs", [
    {"waypoint_id": "", "name": "Office"},
    {"waypoint_id": "office", "name": ""},
])
def test_invalid_capture_request_fails_before_reading_pose_or_writing(tmp_path, kwargs):
    service, registry, reader = make_service(tmp_path)

    with pytest.raises(WaypointValidationError):
        service.save_current_pose_as_waypoint(**kwargs)
    assert reader.calls == 0
    assert registry.list_waypoints() == []


def test_duplicate_id_fails_without_reading_pose_or_modifying_original(tmp_path):
    service, registry, reader = make_service(tmp_path)
    original = registry.add_waypoint(
        waypoint_id="office", name="Original", x=9, y=8, yaw=7
    )

    with pytest.raises(DuplicateWaypointError):
        service.save_current_pose_as_waypoint(waypoint_id="office", name="Office")
    assert reader.calls == 0
    assert registry.get_waypoint("office") == original


def test_provider_rejects_missing_required_pose_authority_fields():
    for field in ("ok", "authoritative", "read_only", "source"):
        result = valid_status()
        result[1].pop(field)
        with pytest.raises(LocalizedPoseUnavailableError):
            CurrentLocalizedPoseProvider(Reader(result)).get_current_pose()


def test_capture_modules_have_no_navigation_ros_robot_or_process_dependency():
    forbidden = {
        "rclpy", "subprocess", "requests", "urllib", "socket",
        "robot_bridge", "behavior_manager", "runtime", "voice_relay",
    }
    for module in (pose_module, capture_module):
        tree = ast.parse(inspect.getsource(module))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not imports.intersection(forbidden)
