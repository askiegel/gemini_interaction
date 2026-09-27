"""Offline contracts for the normal MissionManager Find-Marvin route."""

from copy import deepcopy

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

    def status(self):
        return {
            "ok": True,
            "ros_ready": True,
            "status": "READY",
            "motion": {
                "linear_x": 0,
                "angular_z": 0,
                "streaming": False,
            },
        }


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
        self.controller_results = (
            list(controller_result)
            if isinstance(controller_result, list)
            else [controller_result]
        )
        self.controller_calls = []
        self.generic_calls = []
        self.state_provider_calls = 0
        self.provided_states = []
        self.on_controller_call = None

    def build_find_marvin_controller_state(self):
        self.state_provider_calls += 1
        return {"observation_sequence": self.state_provider_calls}

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
        self.provided_states.append(state_provider())
        if self.on_controller_call is not None:
            self.on_controller_call(len(self.controller_calls))
        index = min(len(self.controller_calls) - 1, len(self.controller_results) - 1)
        return deepcopy(self.controller_results[index])

    def execute(self, mission):
        self.generic_calls.append(mission)
        return {
            "ok": True,
            "completed": True,
            "behavior": mission.mission_type,
            "state": "GENERIC_FIND_COMPLETE",
            "reason": "generic_find_completed",
        }


def controller_result(*, reason, completed, arrived, actions=2, arrival_observations_confirmed=0):
    return {
        "ok": True,
        "completed": completed,
        "arrived_at_marvin": arrived,
        "reason": reason,
        "actions_executed": actions,
        "arrival_observations_confirmed": arrival_observations_confirmed,
        "history": [
            {
                "action_budget_consumed": True,
                "pursuit_step_result": {"motion_executed": True},
                "stop_result": {"ok": True},
            }
            for _ in range(actions)
        ],
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
        [controller_result(
            reason="find_marvin_action_limit_reached",
            completed=False,
            arrived=False,
            actions=6,
        )] * CognitiveRuntime.FIND_MARVIN_MAX_EPISODES,
    )

    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 3
    assert result["ok"] is True and result["completed"] is True
    assert result["mission_outcome"] == "safe_incomplete"
    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_mission_episode_limit_reached"
    assert result["episodes_executed"] == 3
    assert result["total_actions_executed"] == 18
    assert result["max_episodes"] == 3
    assert result["max_actions"] == 6
    assert [entry["episode"] for entry in result["episode_results"]] == [1, 2, 3]
    assert all(
        entry["result"]["controller_result"]["reason"]
        == "find_marvin_action_limit_reached"
        for entry in result["episode_results"]
    )
    assert runtime.mission_manager.get_active_mission() is None


def test_unconfirmed_arrival_candidate_completes_only_as_safe_incomplete(tmp_path):
    runtime, _behavior = make_runtime(
        tmp_path,
        controller_result(
            reason="find_marvin_arrival_confirmation_not_independent",
            completed=False,
            arrived=False,
        ),
    )

    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert result["mission_outcome"] == "safe_incomplete"
    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_arrival_confirmation_not_independent"


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
    assert CognitiveRuntime.FIND_MARVIN_MAX_EPISODES == 3


def test_safe_action_limit_continues_with_fresh_episode_and_arrival_terminates(tmp_path):
    runtime, behavior = make_runtime(
        tmp_path,
        [
            controller_result(
                reason="find_marvin_action_limit_reached",
                completed=False, arrived=False, actions=6,
                arrival_observations_confirmed=0,
            ),
            controller_result(
                reason="arrived_at_marvin",
                completed=True, arrived=True, actions=0,
                arrival_observations_confirmed=2,
            ),
        ],
    )
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 2
    assert behavior.state_provider_calls == 2
    assert behavior.provided_states == [
        {"observation_sequence": 1},
        {"observation_sequence": 2},
    ]
    assert all(call["max_actions"] == 6 for call in behavior.controller_calls)
    assert result["episodes_executed"] == 2
    assert result["total_actions_executed"] == 6
    assert result["max_actions"] == 6 and result["max_episodes"] == 3
    assert result["arrived_at_marvin"] is True
    assert result["mission_outcome"] == "arrived_at_marvin"
    assert result["reason"] == "arrived_at_marvin"
    assert runtime._marvin_autonomous_run_consumed is False
    assert runtime.mission_manager.get_active_mission() is None


def test_two_action_limited_episodes_then_arrival_succeeds_on_third(tmp_path):
    limit = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
    )
    arrival = controller_result(
        reason="arrived_at_marvin",
        completed=True, arrived=True, actions=0,
        arrival_observations_confirmed=2,
    )
    runtime, behavior = make_runtime(tmp_path, [limit, limit, arrival])
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 3
    assert behavior.state_provider_calls == 3
    assert result["episodes_executed"] == 3
    assert result["total_actions_executed"] == 12
    assert result["max_actions"] == 6 and result["max_episodes"] == 3
    assert result["arrived_at_marvin"] is True
    assert result["mission_outcome"] == "arrived_at_marvin"


def test_stop_between_episodes_prevents_continuation(tmp_path):
    limit = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
    )
    runtime, behavior = make_runtime(tmp_path, [limit, limit])

    def stop_after_first_episode(call_number):
        if call_number == 1:
            runtime.submit_intent({
                "intent": "STOP", "speech": "Stopping.", "target": None,
            })

    behavior.on_controller_call = stop_after_first_episode
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["behavior"] == "STOP"
    assert runtime.mission_manager.get_active_mission() is None
    assert runtime.get_status()["runtime_state"] == "STOPPED"
    assert runtime.robot_client.stop_calls >= 1


@pytest.mark.parametrize(
    "reason",
    [
        "marvin_pursuit_forward_failed",
        "find_marvin_nonphysical_stale_replan_limit_reached",
    ],
)
def test_fatal_and_stale_replan_terminal_results_do_not_continue(tmp_path, reason):
    failed = controller_result(
        reason=reason, completed=False, arrived=False, actions=1,
    )
    runtime, behavior = make_runtime(tmp_path, [failed, controller_result(
        reason="arrived_at_marvin", completed=True, arrived=True,
    )])
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["mission_outcome"] == "safe_failure"
    assert result["arrived_at_marvin"] is False


def test_episode_arrival_candidate_is_not_carried_into_next_episode(tmp_path):
    first = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
        arrival_observations_confirmed=0,
    )
    # A controller may retain episode-local diagnostics in its full result;
    # the mission route must not feed that state to a new controller call.
    first["arrival_candidate_timestamp"] = "episode-1-only"
    second = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
        arrival_observations_confirmed=0,
    )
    third = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
        arrival_observations_confirmed=0,
    )
    runtime, behavior = make_runtime(tmp_path, [first, second, third])
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 3
    assert behavior.state_provider_calls == 3
    assert result["episodes_executed"] == 3
    assert result["mission_outcome"] == "safe_incomplete"
    assert result["total_actions_executed"] == 18
    assert result["max_actions"] == 6 and result["max_episodes"] == 3
    assert result["reason"] == "find_marvin_mission_episode_limit_reached"
    assert result["episode_results"][0]["result"]["controller_result"][
        "arrival_candidate_timestamp"
    ] == "episode-1-only"
    assert all(
        "arrival_candidate_timestamp" not in call
        for call in behavior.controller_calls
    )


def test_bridge_failure_between_episodes_prevents_continuation(tmp_path):
    limit = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
    )
    runtime, behavior = make_runtime(tmp_path, [limit, controller_result(
        reason="arrived_at_marvin", completed=True, arrived=True, actions=0,
    )])
    runtime.robot_client.status = lambda: {
        "ok": True,
        "ros_ready": False,
        "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
    }
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["mission_outcome"] == "safe_failure"
    assert result["reason"] == "find_marvin_episode_bridge_not_ready_or_stopped"


def test_action_limit_with_missing_stop_evidence_does_not_continue(tmp_path):
    unsafe_limit = controller_result(
        reason="find_marvin_action_limit_reached",
        completed=False, arrived=False, actions=6,
    )
    unsafe_limit["history"][-1]["stop_result"] = {"ok": False}
    runtime, behavior = make_runtime(tmp_path, [unsafe_limit, controller_result(
        reason="arrived_at_marvin", completed=True, arrived=True, actions=0,
    )])
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()

    assert len(behavior.controller_calls) == 1
    assert result["mission_outcome"] == "safe_failure"
    assert result["reason"] == "find_marvin_action_limit_result_not_safely_stopped"
