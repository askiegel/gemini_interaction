from types import SimpleNamespace

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from marvin_pursuit_state import VISUAL_READY_TO_APPROACH


class _TargetLock:
    target_label = "marvin"

    def snapshot(self):
        return {"tracking_mode": "UNLOCKED", "locked_identity_id": None}


def test_scan_state_is_explicit_and_resets_at_mission_boundary():
    manager = BehaviorManager(robot_client=object())
    first = manager.begin_find_marvin_room_scan("mission-a", source_frame_stamp_ns=100)
    assert first["scan_turn_index"] == 0
    assert first["scan_max_turns"] == 26
    assert first["scan_direction"] == "LEFT"
    assert first["last_seen_source_frame_stamp_ns"] == 100
    assert first["scan_active"] is True and first["scan_exhausted"] is False
    manager._room_scan_update(scan_turn_index=11)
    manager.begin_find_marvin_room_scan("mission-b", source_frame_stamp_ns=900)
    second = manager._room_scan_snapshot()
    assert second["mission_id"] == "mission-b"
    assert second["scan_turn_index"] == 0
    assert second["last_seen_source_frame_stamp_ns"] == 900
    manager.clear_find_marvin_room_scan("mission-a")
    assert manager._room_scan_snapshot() is not None
    manager.clear_find_marvin_room_scan("mission-b")
    assert manager._room_scan_snapshot() is None


def test_post_turn_frame_wait_accepts_only_strictly_newer_source_stamp(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.target_lock = _TargetLock()
    manager.vision = SimpleNamespace(fetch_detection_proposals=lambda: {
        "source_frame_stamp_ns": 101,
    })
    manager.begin_find_marvin_room_scan("mission", source_frame_stamp_ns=100)
    manager._room_scan_update(
        scan_turn_index=1, last_completed_scan_turn=1,
        pre_turn_source_frame_stamp_ns=100,
        awaiting_new_source_frame=True,
    )
    preview_calls = []
    monkeypatch.setattr(manager, "preview_find_object", lambda target, **kwargs: (
        preview_calls.append(target) or {"ok": False, "source_frame_stamp_ns": 101}
    ))
    result = manager.build_find_marvin_controller_state()
    assert preview_calls == ["marvin"]
    assert result["preview_result"]["source_frame_stamp_ns"] == 101
    assert manager._room_scan_snapshot()["last_seen_source_frame_stamp_ns"] == 101
    assert manager._room_scan_snapshot()["pre_turn_source_frame_stamp_ns"] is None


def test_null_preview_stamp_is_rejected_even_after_camera_poll_observed_new_frame(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.target_lock = _TargetLock()
    manager.vision = SimpleNamespace(fetch_detection_proposals=lambda: {
        "source_frame_stamp_ns": 101,
    })
    manager.begin_find_marvin_room_scan("mission", source_frame_stamp_ns=100)
    manager._room_scan_update(
        scan_turn_index=1, last_completed_scan_turn=1,
        pre_turn_source_frame_stamp_ns=100,
        awaiting_new_source_frame=True,
    )
    monkeypatch.setattr(manager, "preview_find_object", lambda target, **kwargs: {
        "ok": False, "source_frame_stamp_ns": None,
    })
    result = manager.build_find_marvin_controller_state()
    assert result["post_turn_frame_status"] == "preview_not_bound_to_new_frame"


def test_no_target_preview_preserves_observed_source_stamp_for_initial_baseline(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    received = []

    def no_target(*, minimum_source_frame_stamp_ns=None):
        received.append(minimum_source_frame_stamp_ns)
        return {
            "found": False, "source_frame_stamp_ns": 455,
            "reason": "no proposal matched",
        }

    monkeypatch.setattr(manager, "_preview_marvin_yolo_identity_observation", no_target)
    result = manager.preview_find_object("marvin")
    assert result["ok"] is False
    assert result["source_frame_stamp_ns"] == 455
    assert received == [None]


def test_post_turn_equal_old_or_missing_stamp_times_out_without_preview(monkeypatch):
    for stamp in (100, 99, None, "101", True):
        manager = BehaviorManager(robot_client=object())
        manager.target_lock = _TargetLock()
        manager.vision = SimpleNamespace(fetch_detection_proposals=lambda s=stamp: {
            "source_frame_stamp_ns": s,
        })
        manager.MARVIN_POST_TURN_FRAME_TIMEOUT_SECONDS = 0.001
        manager.MARVIN_POST_TURN_FRAME_POLL_SECONDS = 0
        manager.begin_find_marvin_room_scan("mission", source_frame_stamp_ns=100)
        manager._room_scan_update(
            scan_turn_index=1, last_completed_scan_turn=1,
            pre_turn_source_frame_stamp_ns=100,
            awaiting_new_source_frame=True,
        )
        preview_calls = []
        monkeypatch.setattr(manager, "preview_find_object", lambda target, **kwargs: preview_calls.append(target))
        result = manager.build_find_marvin_controller_state()
        assert result["post_turn_frame_status"] == "timeout"
        assert preview_calls == []
        assert manager._room_scan_snapshot()["scan_turn_index"] == 1


def test_authorized_acquisition_at_late_scan_indices_stops_before_another_turn(monkeypatch):
    for acquire_after in (0, 1, 6, 7, 20, 25, 26):
        manager = BehaviorManager(robot_client=SimpleNamespace(status=lambda: {
            "ok": True, "ros_ready": True,
            "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
        }))
        manager.begin_find_marvin_room_scan(
            f"mission-{acquire_after}", source_frame_stamp_ns=100,
        )
        executed_turns = []

        def fake_search(_pursuit, *, scan_turn_index, **_kwargs):
            executed_turns.append(scan_turn_index)
            return {
                "ok": True, "decision": "search_turn", "search_action": "turn_left",
                "planner": {"selected_search_action": "turn_left"},
                "motion_executed": True, "replan_required": True,
                "executed_primitive": "guarded_turn_left",
            }

        monkeypatch.setattr(manager, "execute_marvin_search_step", fake_search)
        monkeypatch.setattr(
            behavior_manager_module, "evaluate_marvin_pursuit_state",
            lambda preview, *_args, **_kwargs: ({
                "state": VISUAL_READY_TO_APPROACH,
                "pursuit_authorized": True,
                "selected_identity_id": None,
                "fresh": True, "geometry_usable": True,
            } if preview.get("motion_authorized_marvin_candidate") is True else {
                "state": "SEARCHING", "pursuit_authorized": False,
                "selected_identity_id": None,
            }),
        )
        monkeypatch.setattr(
            behavior_manager_module, "evaluate_marvin_visual_arrival",
            lambda *_args, **_kwargs: {"ok": True, "arrived_at_marvin": False},
        )
        monkeypatch.setattr(
            behavior_manager_module, "evaluate_marvin_arrival",
            lambda *_args, **_kwargs: {
                "ok": True, "arrived_at_marvin": False,
                "selected_identity_id": None,
            },
        )
        provider_count = 0

        def provider():
            nonlocal provider_count
            provider_count += 1
            authorized = manager._room_scan_snapshot()["scan_turn_index"] >= acquire_after
            return {"preview_result": {
                "motion_authorized_marvin_candidate": authorized,
                "source_frame_stamp_ns": 100 + provider_count,
            }}

        # Resume bounded episodes until the requested acquisition boundary.
        for _ in range(7):
            result = manager.execute_find_marvin_controller(
                provider, max_actions=6,
                stop_after_action=lambda: {"ok": True},
            )
            if result["reason"] == "find_marvin_search_target_acquired":
                break
            assert result["reason"] == "find_marvin_action_limit_reached"

        assert result["reason"] == "find_marvin_search_target_acquired"
        assert manager._room_scan_snapshot()["scan_active"] is False
        assert manager._room_scan_snapshot()["scan_target_acquired"] is True
        assert manager._room_scan_snapshot()["scan_transition_pending"] is False
        assert manager._room_scan_snapshot()["scan_turn_index"] == acquire_after
        assert len(executed_turns) == acquire_after
        pursuit_calls = []
        monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *args, **kwargs: (
            pursuit_calls.append(True) or {
                "ok": True, "motion_executed": True,
                "replan_required": True,
            }
        ))
        pursuit_result = manager.execute_find_marvin_controller(
            provider, max_actions=1,
            stop_after_action=lambda: {"ok": True},
        )
        assert pursuit_calls == [True]
        assert pursuit_result["history"][0]["route"] == "pursuit"
        assert len(executed_turns) == acquire_after


def test_missing_initial_source_stamp_does_not_dispatch_a_scan_turn(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.begin_find_marvin_room_scan("mission")
    turns = []
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_pursuit_state",
        lambda *_args, **_kwargs: {
            "state": "SEARCHING", "pursuit_authorized": False,
            "selected_identity_id": None,
        },
    )
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_arrival",
        lambda *_args, **_kwargs: {
            "ok": True, "arrived_at_marvin": False,
            "selected_identity_id": None,
        },
    )
    monkeypatch.setattr(
        manager, "execute_marvin_search_step",
        lambda *_args, **_kwargs: turns.append(True),
    )
    result = manager.execute_find_marvin_controller(
        lambda: {"preview_result": {"source_frame_stamp_ns": None}},
        max_actions=1,
    )
    assert result["reason"] == "find_marvin_scan_source_frame_baseline_missing"
    assert turns == []
    assert manager._room_scan_snapshot()["scan_turn_index"] == 0


def test_rejected_scan_turn_does_not_advance_mission_index(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.begin_find_marvin_room_scan("mission", source_frame_stamp_ns=100)
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_pursuit_state",
        lambda *_args, **_kwargs: {
            "state": "SEARCHING", "pursuit_authorized": False,
            "selected_identity_id": None,
        },
    )
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_arrival",
        lambda *_args, **_kwargs: {
            "ok": True, "arrived_at_marvin": False,
            "selected_identity_id": None,
        },
    )
    monkeypatch.setattr(manager, "execute_marvin_search_step", lambda *_args, **_kwargs: {
        "ok": False, "decision": "search_turn", "search_action": "turn_left",
        "planner": {"selected_search_action": "turn_left"},
        "motion_executed": False, "replan_required": False,
    })
    result = manager.execute_find_marvin_controller(
        lambda: {"preview_result": {"source_frame_stamp_ns": 100}},
        max_actions=1, stop_after_action=lambda: {"ok": True},
    )
    assert result["reason"] == "find_marvin_search_step_failed"
    assert manager._room_scan_snapshot()["scan_turn_index"] == 0
