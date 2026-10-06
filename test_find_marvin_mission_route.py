"""Normal MissionManager routing for the unified V2 closed loop (offline)."""

from unittest.mock import Mock

import pytest

from intent_parser import validate_intent
from test_find_marvin_closed_loop import make_runtime, motions, run


class Provider:
    def get_intent(self, text):
        return validate_intent({
            "intent": "FIND_OBJECT", "target": "Marvin" if "marvin" in text.lower() else "backpack",
            "speech": "Finding the target.",
        })


def test_one_operator_text_starts_full_v2_mission_without_endpoint_interaction(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8), (0, .5)])
    runtime.provider = Provider()
    runtime.submit_text("Find Marvin")
    result = runtime.run_once()
    assert result["mission_route"] == "marvin_v2_closed_loop"
    assert result["state"] == "ARRIVED"
    assert len(motions(events)) == 1
    assert len(behavior.stamps) == 2
    assert runtime.mission_manager.get_active_mission() is None


def test_non_marvin_find_keeps_generic_behavior_route(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [])
    runtime.provider = Provider()
    behavior.execute = Mock(return_value={"ok": True, "completed": True, "behavior": "FIND_OBJECT"})
    runtime.submit_text("Find backpack")
    runtime.run_once()
    behavior.execute.assert_called_once()
    assert events == []


@pytest.mark.parametrize("scan_turns", [0, 7, 25, 26])
def test_acquisition_at_any_search_index_stops_search_immediately(tmp_path, monkeypatch, scan_turns):
    runtime, behavior, _, events, _ = make_runtime(
        tmp_path, monkeypatch, ["absent"] * scan_turns + [(0, .5)])
    result = run(runtime)
    assert result["state"] == "ARRIVED"
    assert result["search_turns"] == scan_turns
    assert len(motions(events)) == scan_turns
    assert len(behavior.stamps) == scan_turns + 1


def test_new_mission_reacquires_identity_without_process_lifetime_latch(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(
        tmp_path, monkeypatch, [(0, .8), (0, .5), (120, .8), (0, .5)])
    assert run(runtime)["state"] == "ARRIVED"
    assert run(runtime)["state"] == "ARRIVED"
    assert behavior.identity_sources == [
        "gemini_marvin_candidate_selection", "marvin_locked_tracker_continuity",
        "gemini_marvin_candidate_selection", "marvin_locked_tracker_continuity",
    ]
    assert len(motions(events)) == 2


def test_direct_path_blocked_never_hands_off_to_obstacle_avoidance(tmp_path, monkeypatch):
    runtime, behavior, _, events, _ = make_runtime(tmp_path, monkeypatch, [(0, .8)])
    behavior.unsafe_forward = True
    behavior.execute_local_obstacle_avoidance_step = Mock(side_effect=AssertionError("avoidance forbidden"))
    runtime.run_local_progress_with_avoidance = Mock(side_effect=AssertionError("avoidance forbidden"))
    result = run(runtime)
    assert result["state"] == "BLOCKED"
    assert motions(events) == []
    behavior.execute_local_obstacle_avoidance_step.assert_not_called()
    runtime.run_local_progress_with_avoidance.assert_not_called()
