"""Offline contracts for the normal MissionManager Find-Marvin route."""

import pytest

from intent_parser import validate_intent
from mission_manager import MissionManager
from runtime import CognitiveRuntime
from world_model import WorldModel


class FakeProvider:
    def get_intent(self, text):
        normalized = str(text).strip().lower()
        target = "backpack" if "backpack" in normalized else "Marvin"
        return validate_intent({
            "intent": "FIND_OBJECT",
            "target": target,
            "speech": f"Finding {target}.",
        })


class FakeRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return {"ok": True, "action": "stop"}


class FakeLidarWorker:
    session = "test-active-lidar-session"
    running = True

    def __init__(self, *_args, **_kwargs):
        self.sequence = 0
        self.last_error = None

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


class FakeBehavior:
    def __init__(self, robot, controller_result):
        self.robot = robot
        self.controller_result = controller_result
        self.controller_calls = []
        self.generic_calls = []

    def execute_find_marvin_controller(
        self,
        state_provider,
        *,
        max_actions,
        dry_run,
        stop_after_action,
    ):
        self.controller_calls.append({
            "state_provider": state_provider,
            "max_actions": max_actions,
            "dry_run": dry_run,
            "stop_after_action": stop_after_action,
        })
        return dict(self.controller_result)

    def execute(self, mission):
        self.generic_calls.append(mission)
        return {
            "ok": True,
            "completed": True,
            "behavior": mission.mission_type,
            "state": "GENERIC_FIND_COMPLETE",
            "reason": "generic_find_completed",
        }


def controller_result(*, reason, completed, arrived):
    return {
        "ok": True,
        "completed": completed,
        "arrived_at_marvin": arrived,
        "reason": reason,
        "actions_executed": 2,
        "history": [{
            "pursuit_step_result": {"motion_executed": True},
            "stop_result": {"ok": True},
        }],
    }


def make_runtime(tmp_path, result):
    robot = FakeRobot()
    behavior = FakeBehavior(robot, result)
    runtime = CognitiveRuntime(
        provider=FakeProvider(),
        mission_manager=MissionManager(),
        world_model=WorldModel(str(tmp_path / "world.json")),
        vision_adapter=object(),
        robot_client=robot,
        behavior_manager=behavior,
        lidar_worker_factory=FakeLidarWorker,
    )
    runtime.running = True
    return runtime, behavior


@pytest.mark.parametrize("phrase", ["Find Marvin", "Find marvin", "Go find Marvin"])
def test_find_marvin_text_routes_through_mission_to_existing_bounded_controller(tmp_path, phrase):
    runtime, behavior = make_runtime(
        tmp_path,
        controller_result(
            reason="arrived_at_marvin", completed=True, arrived=True,
        ),
    )

    submitted = runtime.submit_text(phrase)
    assert submitted["intent"]["intent"] == "FIND_OBJECT"
    assert submitted["intent"]["target"] == "marvin"
    assert submitted["mission"]["mission_type"] == "FIND_OBJECT"
    assert submitted["mission"]["target"] == "marvin"

    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    call = behavior.controller_calls[0]
    assert call["max_actions"] == 6
    assert call["dry_run"] is False
    assert callable(call["stop_after_action"])
    assert behavior.generic_calls == []
    assert result["mission_route"] == "bounded_marvin_autonomous"
    assert result["mission_outcome"] == "arrived_at_marvin"
    assert result["arrived_at_marvin"] is True
    assert runtime.mission_manager.get_active_mission() is None


def test_non_marvin_find_keeps_existing_generic_find_behavior(tmp_path):
    runtime, behavior = make_runtime(
        tmp_path,
        controller_result(
            reason="arrived_at_marvin", completed=True, arrived=True,
        ),
    )

    runtime.submit_text("Find backpack")
    result = runtime.run_once()

    assert behavior.controller_calls == []
    assert len(behavior.generic_calls) == 1
    assert result["state"] == "GENERIC_FIND_COMPLETE"


def test_action_limit_is_safe_incomplete_not_arrival_success(tmp_path):
    runtime, behavior = make_runtime(
        tmp_path,
        controller_result(
            reason="find_marvin_action_limit_reached",
            completed=False,
            arrived=False,
        ),
    )

    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["ok"] is True and result["completed"] is True
    assert result["mission_outcome"] == "safe_incomplete"
    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert runtime.mission_manager.get_active_mission() is None


def test_invalid_or_failed_controller_result_fails_closed(tmp_path):
    runtime, behavior = make_runtime(
        tmp_path,
        {"ok": False, "reason": "marvin_autonomous_lidar_session_unavailable"},
    )

    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["mission_outcome"] == "safe_failure"
    assert result["reason"] == "marvin_autonomous_lidar_session_unavailable"
    assert runtime.mission_manager.get_active_mission() is None


def test_stop_preempts_marvin_route_without_a_second_controller_execution(tmp_path):
    result = controller_result(
        reason="arrived_at_marvin", completed=True, arrived=True,
    )
    runtime, behavior = make_runtime(tmp_path, result)
    original = behavior.execute_find_marvin_controller

    def preempting_controller(*args, **kwargs):
        runtime.submit_intent({"intent": "STOP", "speech": "Stopping.", "target": None})
        return original(*args, **kwargs)

    behavior.execute_find_marvin_controller = preempting_controller
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert result["behavior"] == "STOP"
    assert runtime.mission_manager.get_active_mission() is None
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert len(behavior.controller_calls) == 1
    assert runtime.robot_client.stop_calls >= 1


def test_mission_route_leaves_reviewed_safety_limits_owned_by_runtime():
    assert CognitiveRuntime.FIND_MARVIN_AUTONOMOUS_MAX_ACTIONS == 6
