from behavior_manager import BehaviorManager

class Robot:
    def turn_left(self, *args, **kwargs): raise AssertionError("legacy left")
    def turn_right(self, *args, **kwargs): raise AssertionError("legacy right")

def test_explicit_turns_bind_current_session_and_propagate_guarded_result(monkeypatch):
    manager=BehaviorManager(robot_client=Robot())
    manager.lidar_session_provider=lambda: "session-1"
    calls=[]
    def guarded(direction, angular_speed, duration, *, expected_lidar_session):
        calls.append((direction, angular_speed, duration, expected_lidar_session))
        return {"ok": True, "confirmed_forwarded": True, "angular_z": angular_speed if direction == "LEFT" else -angular_speed, "linear_x": 0.0}
    monkeypatch.setattr(manager, "execute_guarded_turn", guarded)
    left=manager._execute_turn_left(None); right=manager._execute_turn_right(None)
    assert calls == [("LEFT", .5, .4, "session-1"), ("RIGHT", .5, .4, "session-1")]
    assert left["executed"] and left["angular_z"] > 0 and left["linear_x"] == 0.0
    assert right["executed"] and right["angular_z"] < 0 and right["linear_x"] == 0.0

def test_missing_session_fails_closed_without_guarded_motion(monkeypatch):
    manager=BehaviorManager(robot_client=Robot()); manager.lidar_session_provider=lambda: None
    monkeypatch.setattr(manager, "execute_guarded_turn", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    result=manager._execute_turn_left(None)
    assert not result["ok"] and not result["executed"] and result["reason"] == "lidar_producer_session_unavailable"

def test_denied_guarded_result_is_not_success(monkeypatch):
    manager=BehaviorManager(robot_client=Robot()); manager.lidar_session="s"
    monkeypatch.setattr(manager, "execute_guarded_turn", lambda *a, **k: {"ok": False, "confirmed_forwarded": False, "reason": "stale_lidar"})
    result=manager._execute_turn_right(None)
    assert not result["ok"] and not result["executed"] and result["reason"] == "stale_lidar"
