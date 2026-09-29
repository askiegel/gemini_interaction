"""Offline contracts for the zero-motion remember-current-location command."""

import ast
import inspect
from pathlib import Path

import pytest

import remember_current_location as intent_module
import remember_current_location_service as command_module
from conversation_manager import ConversationResult
from conversation_service import ConversationService
from current_localized_pose import LocalizedPoseUnavailableError
from current_localized_pose import CurrentLocalizedPoseProvider
from remember_current_location import (
    RememberLocationNameError,
    parse_remember_current_location,
)
from remember_current_location_service import RememberCurrentLocationService
from search_waypoint_registry import DuplicateWaypointError, SearchWaypoint
from search_waypoint_registry import SearchWaypointRegistry
from search_waypoint_capture import SearchWaypointCaptureService


class NeverCalledConversationManager:
    def __init__(self):
        self.calls = []

    def process(self, text):
        self.calls.append(text)
        raise AssertionError("remember command must not invoke conversation provider")

    def clear_history(self):
        pass

    def get_history(self):
        return []


class StaticConversationManager:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def process(self, text):
        self.calls.append(text)
        return self.result

    def clear_history(self):
        pass

    def get_history(self):
        return []


class FakeCaptureService:
    def __init__(self, result=None, error=None):
        self.result = result or SearchWaypoint(
            waypoint_id="office", name="Office", x=1.25, y=-0.5, yaw=0.75,
            active=True, created_at="created", updated_at="created",
        )
        self.error = error
        self.calls = []

    def save_current_pose_as_waypoint(self, *, waypoint_id, name):
        self.calls.append({"waypoint_id": waypoint_id, "name": name})
        if self.error is not None:
            raise self.error
        return self.result


class NeverCalledSubmitter:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("remember command must not submit a runtime mission")


def make_service(capture=None, manager=None, submitter=None):
    return ConversationService(
        conversation_manager=manager or NeverCalledConversationManager(),
        mission_submitter=submitter or NeverCalledSubmitter(),
        remember_location_service=RememberCurrentLocationService(
            capture or FakeCaptureService()
        ),
    )


def test_explicit_remember_phrases_parse_to_distinct_intent_with_deterministic_id():
    office = parse_remember_current_location("Remember this place as Office")
    guest_bedroom = parse_remember_current_location(
        "Remember this location as Guest Bedroom"
    )
    charging_area = parse_remember_current_location("Save this place as Charging Area")

    assert (office.intent, office.name, office.waypoint_id) == (
        "REMEMBER_CURRENT_LOCATION", "Office", "office",
    )
    assert (guest_bedroom.name, guest_bedroom.waypoint_id) == (
        "Guest Bedroom", "guest_bedroom",
    )
    assert charging_area.waypoint_id == "charging_area"


def test_remember_command_calls_capture_once_and_returns_stored_waypoint():
    capture = FakeCaptureService()
    submitter = NeverCalledSubmitter()
    service = make_service(capture=capture, submitter=submitter)

    result = service.process_text("Remember this place as Office")

    assert capture.calls == [{"waypoint_id": "office", "name": "Office"}]
    assert submitter.calls == []
    assert result.reply == "Okay, I remembered this location as Office."
    assert result.command_result == {
        "ok": True,
        "intent": "REMEMBER_CURRENT_LOCATION",
        "reason": None,
        "waypoint": {
            "waypoint_id": "office", "name": "Office",
            "x": 1.25, "y": -0.5, "yaw": 0.75, "frame": "map",
        },
    }


def test_addressed_command_preserves_human_name_case():
    capture = FakeCaptureService()
    result = make_service(capture=capture).process_text(
        "Mayday, Remember this location as Guest Bedroom"
    )

    assert result.command_result["ok"] is True
    assert capture.calls == [{
        "waypoint_id": "guest_bedroom", "name": "Guest Bedroom",
    }]


@pytest.mark.parametrize("command, reason", [
    ("Remember this place", "NO_NAME"),
    ("Remember this location as North/West", "INVALID_NAME"),
])
def test_missing_or_malformed_name_fails_before_capture(command, reason):
    capture = FakeCaptureService()
    result = make_service(capture=capture).process_text(command)

    assert result.command_result["reason"] == reason
    assert capture.calls == []


@pytest.mark.parametrize("error, reason", [
    (DuplicateWaypointError("duplicate"), "DUPLICATE_WAYPOINT"),
    (LocalizedPoseUnavailableError("not ready"), "LOCALIZATION_NOT_READY"),
    (LocalizedPoseUnavailableError("stale", reason="STALE_POSE"), "STALE_POSE"),
])
def test_capture_failures_are_safe_and_do_not_submit_missions(error, reason):
    capture = FakeCaptureService(error=error)
    submitter = NeverCalledSubmitter()
    result = make_service(capture=capture, submitter=submitter).process_text(
        "Remember this place as Office"
    )

    assert result.command_result["ok"] is False
    assert result.command_result["reason"] == reason
    assert len(capture.calls) == 1
    assert submitter.calls == []


def test_stale_command_capture_does_not_partially_write_registry(tmp_path):
    status = (200, {
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
            "age_seconds": 3.0,
            "received_at": "now",
            "pose": {
                "frame_id": "map",
                "position": {"x": 1.0, "y": 2.0},
                "yaw_radians": 0.5,
            },
        },
    })
    registry = SearchWaypointRegistry(tmp_path / "waypoints.json")
    capture = SearchWaypointCaptureService(
        registry, CurrentLocalizedPoseProvider(lambda: status),
    )
    result = make_service(capture=capture).process_text(
        "Remember this place as Office"
    )

    assert result.command_result["reason"] == "STALE_POSE"
    assert registry.list_waypoints() == []


def test_command_api_accepts_only_name_and_derived_id_not_coordinates():
    parameters = inspect.signature(
        RememberCurrentLocationService.execute
    ).parameters
    assert tuple(parameters) == ("self", "waypoint_id", "name")


def test_normal_find_object_conversation_route_is_unchanged():
    manager = StaticConversationManager(ConversationResult(
        reply="Okay, I will find Marvin.",
        decision_type="MISSION",
        mission_type="FIND_OBJECT",
        target="marvin",
    ))

    class Submitter:
        def __init__(self):
            self.calls = []

        def __call__(self, user_text, intent, runtime_url):
            self.calls.append(intent)
            return {"ok": True, "accepted": True, "mission": {"mission_id": "m"}}

    submitter = Submitter()
    result = make_service(manager=manager, submitter=submitter).process_text("Find Marvin")

    assert result.mission_submitted is True
    assert submitter.calls == [{
        "intent": "FIND_OBJECT", "speech": "Okay, I will find Marvin.",
        "target": "marvin",
    }]


def test_name_normalization_rejects_unsafe_characters_without_ambiguity():
    with pytest.raises(RememberLocationNameError):
        parse_remember_current_location("Save this place as Office_2")


def test_command_modules_have_no_ros_navigation_robot_or_process_dependency():
    forbidden = {
        "rclpy", "subprocess", "requests", "urllib", "socket",
        "robot_bridge", "behavior_manager", "runtime", "voice_relay",
    }
    for module in (intent_module, command_module):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not imported.intersection(forbidden)


def test_browser_service_wires_only_the_existing_read_only_pose_status_reader():
    source = (
        Path(__file__).resolve().parent / "voice_relay" / "server.py"
    ).read_text(encoding="utf-8")
    assert "localized_pose_status_reader" in source
    assert "get_tony2_navigation_runtime().live_pose_status()" in source
