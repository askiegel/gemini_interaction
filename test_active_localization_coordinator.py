import threading

from runtime import CognitiveRuntime


def _recoverable_status():
    return {
        "ok": False,
        "authoritative": True,
        "read_only": True,
        "source": "tony2_navigation_amcl",
        "navigation": {
            "localization_state": "ACTIVE_LOCALIZATION_REQUIRED",
            "localization_validated": False,
            "goal_submission_enabled": False,
            "goal_active": False,
        },
        "telemetry": {"available": False, "pose": None, "age_seconds": None},
    }


def _trusted_status():
    return {
        "ok": True,
        "authoritative": True,
        "read_only": True,
        "source": "tony2_navigation_amcl",
        "navigation": {
            "localization_state": "LOCALIZED",
            "localization_validated": True,
            "transform_ready": True,
            "goal_submission_enabled": True,
            "goal_active": False,
        },
        "telemetry": {
            "available": True,
            "age_seconds": 0.1,
            "pose": {
                "frame_id": "map",
                "position": {"x": 1.0, "y": 2.0},
                "yaw_radians": 0.3,
            },
        },
    }


class _Facade:
    def __init__(self, statuses, events=None):
        self.statuses = list(statuses)
        self.status_calls = 0
        self.retry_calls = 0
        self.events = events

    def get_localization_status(self):
        self.status_calls += 1
        if self.statuses:
            return self.statuses.pop(0)
        return _recoverable_status()

    def retry_global_localization(self):
        self.retry_calls += 1
        if self.events is not None:
            self.events.append("retry")
        return {"ok": False, "reason": "ACTIVE_LOCALIZATION_REQUIRED"}


class _Robot:
    def __init__(self, *, bridge_ok=True, stop_ok=True, events=None):
        self.bridge_ok = bridge_ok
        self.stop_ok = stop_ok
        self.stop_calls = 0
        self.events = events

    def status(self):
        if not self.bridge_ok:
            return {"ok": False}
        return {
            "ok": True,
            "ros_ready": True,
            "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
        }

    def stop(self):
        self.stop_calls += 1
        if self.events is not None:
            self.events.append("stop")
        return {"ok": self.stop_ok}


class _Behavior:
    def __init__(self, *, succeeds=True):
        self.succeeds = succeeds
        self.calls = []

    def execute_guarded_turn(self, direction, angular_speed, duration, *, expected_lidar_session,
                             safety_mode="LEGACY_BROAD_SIDE"):
        self.calls.append((direction, angular_speed, duration, expected_lidar_session, safety_mode))
        return {
            "ok": self.succeeds,
            "permitted": self.succeeds,
            "confirmed_forwarded": self.succeeds,
        }


class _World:
    def __init__(self, valid=True):
        self.valid = valid

    def get_lidar_obstacles(self, *, expected_session):
        return {
            "available": True,
            "valid": self.valid,
            "reason": "fresh" if self.valid else "stale",
            "producer_session": expected_session,
            "local_motion_geometry": {"valid": self.valid},
        }


class _MissionManager:
    def __init__(self, active=None):
        self.active = active

    def get_active_mission(self):
        return self.active


def _runtime(statuses, *, bridge_ok=True, stop_ok=True, turn_ok=True, lidar_ok=True, events=None):
    runtime = CognitiveRuntime.__new__(CognitiveRuntime)
    runtime.localization_facade = _Facade(statuses, events=events)
    runtime.robot_client = _Robot(bridge_ok=bridge_ok, stop_ok=stop_ok, events=events)
    runtime.behavior_manager = _Behavior(succeeds=turn_ok)
    runtime.world_model = _World(valid=lidar_ok)
    runtime.lidar_worker = type("Worker", (), {"session": "lidar-session", "running": True})()
    runtime.mission_manager = _MissionManager()
    runtime._state_lock = threading.RLock()
    runtime._active_localization_lock = threading.Lock()
    runtime._behavior_execution_generation = None
    runtime.running = True
    return runtime


def test_already_localized_performs_zero_turns():
    runtime = _runtime([_trusted_status()])
    result = runtime.run_bounded_active_localization()
    assert result["ok"] is True
    assert result["turns_executed"] == 0
    assert runtime.behavior_manager.calls == []
    assert runtime.localization_facade.retry_calls == 0


def test_recoverable_state_turns_left_stops_then_retries_to_success():
    runtime = _runtime([_recoverable_status(), _recoverable_status(), _trusted_status()])
    result = runtime.run_bounded_active_localization()
    assert result["ok"] is True
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_SUCCESS"
    assert runtime.behavior_manager.calls == [
        ("LEFT", 0.25, 0.50, "lidar-session", "ROTATIONAL_SWEPT_FOOTPRINT")
    ]
    assert runtime.robot_client.stop_calls == 1
    assert runtime.localization_facade.retry_calls == 1
    assert result["turn_history"][0]["linear_x"] == 0.0


def test_turn_directions_alternate_left_then_right():
    statuses = [_recoverable_status()] * 4 + [_trusted_status()]
    runtime = _runtime(statuses)
    result = runtime.run_bounded_active_localization()
    assert result["ok"] is True
    assert [call[0] for call in runtime.behavior_manager.calls] == ["LEFT", "RIGHT"]
    assert all(call[1:3] == (0.25, 0.50) for call in runtime.behavior_manager.calls)


def test_six_turn_budget_exhausts_without_a_seventh_turn():
    runtime = _runtime([_recoverable_status()] * 20)
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_EXHAUSTED"
    assert result["turns_executed"] == 6
    assert len(runtime.behavior_manager.calls) == 6
    assert runtime.localization_facade.retry_calls == 6
    assert runtime.robot_client.stop_calls == 6
    assert [call[0] for call in runtime.behavior_manager.calls] == [
        "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT",
    ]


def test_bridge_not_stopped_blocks_without_turn():
    runtime = _runtime([_recoverable_status()] * 2, bridge_ok=False)
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_BRIDGE_NOT_STOPPED"
    assert runtime.behavior_manager.calls == []


def test_guarded_turn_failure_still_stops_and_never_retries():
    runtime = _runtime([_recoverable_status()] * 2, turn_ok=False)
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_TURN_BLOCKED"
    assert runtime.robot_client.stop_calls == 1
    assert runtime.localization_facade.retry_calls == 0


def test_stale_lidar_blocks_without_any_physical_turn():
    runtime = _runtime([_recoverable_status()] * 2, lidar_ok=False)
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_LIDAR_NOT_READY"
    assert runtime.behavior_manager.calls == []
    assert runtime.robot_client.stop_calls == 0


def test_retry_happens_only_after_stop():
    events = []
    runtime = _runtime(
        [_recoverable_status(), _recoverable_status(), _trusted_status()],
        events=events,
    )
    result = runtime.run_bounded_active_localization()
    assert result["ok"] is True
    assert events == ["stop", "retry"]


def test_stop_failure_prevents_localization_retry():
    runtime = _runtime([_recoverable_status()] * 2, stop_ok=False)
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_STOP_FAILED"
    assert runtime.localization_facade.retry_calls == 0


def test_hard_localization_failure_never_turns():
    hard_failure = _recoverable_status()
    hard_failure["navigation"]["localization_state"] = "UNLOCALIZED"
    runtime = _runtime([hard_failure])
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_HARD_FAILURE"
    assert runtime.behavior_manager.calls == []


def test_existing_physical_mission_blocks_active_localization():
    runtime = _runtime([_recoverable_status()])
    runtime.mission_manager = _MissionManager(active=object())
    result = runtime.run_bounded_active_localization()
    assert result["terminal_reason"] == "ACTIVE_LOCALIZATION_TURN_BLOCKED"
    assert runtime.behavior_manager.calls == []
