"""Offline bounded-controller contracts for Find-Marvin routing."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
import pytest


def evidence(identity="marvin-1", preview_result=None):
    return {"preview_result": preview_result or {"preview": "fresh"}, "target_lock_result": {"lock": "fresh"},
            "target_lock_snapshot": {"snapshot": "fresh"}, "selected_identity_id": identity}


def arrival_preview(height_fraction, timestamp, *, ambiguous=False, confirmed=True):
    image_height = 480.0
    bbox_height = round(float(height_fraction) * image_height)
    bbox_y1 = max(1.0, (image_height - bbox_height) / 2.0)
    return {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "target": "marvin",
        "target_found": True,
        "source": "marvin_local_tracker",
        "identity_confirmed": confirmed,
        "motion_authorized_marvin_candidate": True,
        "ambiguous": ambiguous,
        "source_timestamp": timestamp,
        "image_width": 640.0,
        "image_height": image_height,
        "bbox": {
            "x1": 200.0,
            "y1": bbox_y1,
            "x2": 400.0,
            "y2": bbox_y1 + bbox_height,
        },
    }


def pursuit(state="READY_TO_APPROACH", authorized=True, identity="marvin-1"):
    return {"ok": True, "state": state, "pursuit_authorized": authorized,
            "selected_identity_id": identity, "entity_id": "entity-1", "fresh": True,
            "geometry_usable": True}


def successful_pursuit():
    return {"ok": True, "decision": "approach_forward", "executed_primitive": "forward",
            "motion_executed": True, "replan_required": True}


def nonphysical_stale_replan():
    return {
        "ok": True, "decision": "approach_forward", "executed_primitive": None,
        "motion_executed": False, "replan_required": True,
        "stale_replan": True, "stale_replan_classification": "NONPHYSICAL_STALE_REPLAN",
        "action_budget_consumed": False,
    }


def physical_stale_replan():
    return {
        "ok": True, "decision": "approach_forward", "executed_primitive": "forward",
        "motion_executed": True, "motion_possible": True, "replan_required": True,
        "stale_replan": True,
        "stale_replan_classification": "PHYSICAL_OR_UNCERTAIN_STALE_REPLAN",
        "action_budget_consumed": True,
    }


def successful_search(action="turn_left"):
    return {"ok": True, "decision": "search_turn", "search_action": action,
            "executed_primitive": "guarded_turn_left", "motion_executed": True,
            "replan_required": True, "planner": {"selected_search_action": action}}


def not_arrived(identity="marvin-1"):
    return {"ok": True, "arrived_at_marvin": False,
            "selected_identity_id": identity}


def arrived(identity="marvin-1"):
    return {"ok": True, "arrived_at_marvin": True,
            "selected_identity_id": identity, "identity_authorized": True,
            "fresh": True, "geometry_valid": True,
            "reason": "marvin_visual_standoff_reached"}


def invoke(monkeypatch, states, *, search_steps=None, pursuit_steps=None, max_actions=6,
           identities=None, arrivals=None, dry_run=False, stop_after_action=None):
    manager = BehaviorManager(robot_client=object())
    provider_calls, evaluator_calls, arrival_calls, search_calls, pursuit_calls = [], [], [], [], []
    state_values = iter(states)
    identity_values = iter(identities or ["marvin-1"] * len(states))
    search_values = iter(search_steps or [successful_search()] * max_actions)
    pursuit_values = iter(pursuit_steps or [successful_pursuit()] * max_actions)
    arrival_values = iter(arrivals) if arrivals is not None else None

    def provider():
        provider_calls.append(True)
        stamp = (
            datetime(2026, 9, 26, 16, 0, 9, tzinfo=timezone.utc)
            + timedelta(microseconds=len(provider_calls))
        ).isoformat()
        preview_result = arrival_preview(0.56, stamp)
        return evidence(next(identity_values), preview_result)

    def evaluate(*args, **kwargs):
        evaluator_calls.append((args, kwargs))
        value = next(state_values)
        if isinstance(value, Exception):
            raise value
        return value

    def search(*args, **kwargs):
        search_calls.append((args, kwargs))
        value = next(search_values)
        if isinstance(value, Exception):
            raise value
        return value

    def pursuit_step(*args, **kwargs):
        pursuit_calls.append((args, kwargs))
        value = next(pursuit_values)
        if isinstance(value, Exception):
            raise value
        return value

    def arrival(*args, **kwargs):
        arrival_calls.append((args, kwargs))
        if arrival_values is None:
            return not_arrived(kwargs.get("selected_identity_id"))
        value = next(arrival_values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", evaluate)
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_arrival", arrival)
    monkeypatch.setattr(manager, "execute_marvin_search_step", search)
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", pursuit_step)
    result = manager.execute_find_marvin_controller(
        provider, max_actions=max_actions, now="2026-09-26T16:00:10+00:00", dry_run=dry_run,
        stop_after_action=stop_after_action)
    return result, provider_calls, evaluator_calls, arrival_calls, search_calls, pursuit_calls


def test_searching_and_reacquire_route_only_to_search(monkeypatch):
    for state in ("SEARCHING", "REACQUIRE_REQUIRED"):
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit(state, False)], max_actions=1)
        assert len(searches) == result["actions_executed"] == 1
        assert pursuits == [] and result["reason"] == "find_marvin_action_limit_reached"


def test_search_dispatch_attempt_consumes_budget_and_requires_stop(monkeypatch):
    stops = []
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit("SEARCHING", False)],
        max_actions=1,
        stop_after_action=lambda: stops.append(True) or {"ok": False},
    )
    assert len(searches) == 1 and pursuits == []
    assert result["actions_executed"] == 1
    assert result["history"][0]["action_budget_consumed"] is True
    assert result["history"][0]["stop_result"] == {"ok": False}
    assert stops == [True]
    assert result["reason"] == "find_marvin_post_action_stop_failed"


def test_ready_authorized_routes_only_to_pursuit(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, [pursuit()], max_actions=1)
    assert searches == [] and len(pursuits) == result["actions_executed"] == 1


def test_visual_session_routes_to_one_pursuit_step_without_persistent_identity(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    evidence_value = {
        "preview_result": {"fresh": "marvin"},
        "target_lock_result": {},
        "target_lock_snapshot": {"tracking_mode": "UNLOCKED"},
        "selected_identity_id": None,
    }
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "ok": True, "state": "VISUAL_READY_TO_ALIGN",
            "pursuit_authorized": True, "selected_identity_id": None,
            "entity_id": None, "fresh": True, "geometry_usable": True,
        },
    )
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_visual_arrival",
        lambda *args, **kwargs: {
            "ok": True, "arrived_at_marvin": False,
            "visual_session_authorized": True,
        },
    )
    monkeypatch.setattr(
        manager, "execute_marvin_pursuit_step",
        lambda *args, **kwargs: calls.append((args, kwargs)) or successful_pursuit(),
    )
    result = manager.execute_find_marvin_controller(lambda: evidence_value, max_actions=1)
    assert result["actions_executed"] == 1 and result["history"][0]["route"] == "pursuit"
    assert len(calls) == 1


def test_visual_arrival_stops_without_pursuit_or_identity(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    evidence_values = iter((
        {
            "preview_result": arrival_preview(0.56, "2026-09-26T16:00:08+00:00"),
            "target_lock_result": {},
            "target_lock_snapshot": {"tracking_mode": "UNLOCKED"},
            "selected_identity_id": None,
        },
        {
            "preview_result": arrival_preview(0.57, "2026-09-26T16:00:09+00:00"),
            "target_lock_result": {},
            "target_lock_snapshot": {"tracking_mode": "UNLOCKED"},
            "selected_identity_id": None,
        },
    ))
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "ok": True, "state": "VISUAL_READY_TO_APPROACH",
            "pursuit_authorized": True, "selected_identity_id": None,
            "entity_id": None, "fresh": True, "geometry_usable": True,
        },
    )
    monkeypatch.setattr(
        behavior_manager_module, "evaluate_marvin_visual_arrival",
        lambda *args, **kwargs: {
            "ok": True, "arrived_at_marvin": True,
            "visual_session_authorized": True, "fresh": True,
            "geometry_valid": True,
        },
    )
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
    result = manager.execute_find_marvin_controller(
        lambda: next(evidence_values), max_actions=1,
        now="2026-09-26T16:00:10+00:00",
    )
    assert result["arrived_at_marvin"] is True and result["actions_executed"] == 0
    assert calls == []


def run_visual_preview_sequence(monkeypatch, previews, *, max_actions=2):
    manager = BehaviorManager(robot_client=object())
    events = []
    preview_values = iter(previews)
    pursuit_calls = []
    search_calls = []

    def provider():
        value = next(preview_values)
        events.append(("preview", value["source_timestamp"]))
        return {
            "preview_result": value,
            "target_lock_result": {},
            "target_lock_snapshot": {"tracking_mode": "UNLOCKED"},
            "selected_identity_id": None,
        }

    def pursuit_step(*_args, **_kwargs):
        events.append(("pursuit", None))
        pursuit_calls.append(True)
        return successful_pursuit()

    def search_step(*_args, **_kwargs):
        events.append(("search", None))
        search_calls.append(True)
        return successful_search()

    def stop():
        events.append(("stop", None))
        return {"ok": True, "stopped": True}

    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", pursuit_step)
    monkeypatch.setattr(manager, "execute_marvin_search_step", search_step)
    result = manager.execute_find_marvin_controller(
        provider,
        max_actions=max_actions,
        now="2026-09-26T16:00:10+00:00",
        stop_after_action=stop,
    )
    return result, events, pursuit_calls, search_calls


def test_live_bbox_jump_29583_65417_29375_does_not_complete_arrival(monkeypatch):
    previews = [
        arrival_preview(0.29583, "2026-09-26T16:00:08+00:00"),
        arrival_preview(0.65417, "2026-09-26T16:00:09+00:00"),
        arrival_preview(0.29375, "2026-09-26T16:00:09.500000+00:00"),
    ]
    result, events, pursuits, _searches = run_visual_preview_sequence(
        monkeypatch, previews, max_actions=2,
    )

    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert result["actions_executed"] == 2
    assert result["history"][1]["selected_action"] == "confirm_arrival"
    assert result["history"][1]["arrival"]["arrived_at_marvin"] is True
    assert result["history"][2]["arrival_candidate_reset"] is True
    assert result["history"][2]["arrival"]["arrived_at_marvin"] is False
    assert len(pursuits) == 2
    candidate_index = events.index(("preview", "2026-09-26T16:00:09+00:00"))
    confirm_index = events.index(("preview", "2026-09-26T16:00:09.500000+00:00"))
    assert events[candidate_index + 1:confirm_index] == [("stop", None)]


def test_two_consecutive_independent_arrival_previews_complete_without_motion(monkeypatch):
    result, events, pursuits, searches = run_visual_preview_sequence(
        monkeypatch,
        [
            arrival_preview(0.56, "2026-09-26T16:00:08+00:00"),
            arrival_preview(0.57, "2026-09-26T16:00:09+00:00"),
        ],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is result["completed"] is True
    assert result["actions_executed"] == 0
    assert result["arrival_observations_confirmed"] == 2
    assert pursuits == searches == []
    assert [event[0] for event in events] == ["preview", "stop", "preview"]


def test_far_unclipped_preview_is_not_arrival(monkeypatch):
    result, _events, _pursuits, _searches = run_visual_preview_sequence(
        monkeypatch,
        [arrival_preview(0.3167, "2026-09-26T16:00:08+00:00")],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["history"][0]["arrival"]["arrival_geometry_valid"] is True
    assert result["history"][0]["arrival"]["reason"] == "marvin_visual_standoff_not_reached"


def test_two_consecutive_clipped_oversized_previews_cannot_complete_arrival(monkeypatch):
    previews = [
        arrival_preview(0.78125, "2026-09-26T16:00:08+00:00"),
        arrival_preview(0.81667, "2026-09-26T16:00:09+00:00"),
    ]
    for preview in previews:
        bbox_height = preview["bbox"]["y2"] - preview["bbox"]["y1"]
        preview["bbox"]["y1"] = 0.0
        preview["bbox"]["y2"] = bbox_height
    result, events, pursuits, _searches = run_visual_preview_sequence(
        monkeypatch, previews, max_actions=2,
    )
    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert result["actions_executed"] == 2
    assert all(
        entry["arrival"]["arrived_at_marvin"] is False
        and entry["arrival"]["reason"] == "preview_arrival_geometry_clipped"
        for entry in result["history"]
    )
    assert len(pursuits) == 2
    assert [event[0] for event in events] == ["preview", "pursuit", "stop", "preview", "pursuit", "stop"]


def test_unclipped_candidate_followed_by_clipped_frame_resets_without_arrival(monkeypatch):
    clipped = arrival_preview(0.57, "2026-09-26T16:00:09+00:00")
    bbox_height = clipped["bbox"]["y2"] - clipped["bbox"]["y1"]
    clipped["bbox"]["y1"] = 0.0
    clipped["bbox"]["y2"] = bbox_height
    result, events, pursuits, _searches = run_visual_preview_sequence(
        monkeypatch,
        [arrival_preview(0.56, "2026-09-26T16:00:08+00:00"), clipped],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["history"][0]["selected_action"] == "confirm_arrival"
    assert result["history"][1]["arrival_candidate_reset"] is True
    assert result["history"][1]["arrival"]["reason"] == "preview_arrival_geometry_clipped"
    assert result["actions_executed"] == len(pursuits) == 1
    assert [event[0] for event in events] == ["preview", "stop", "preview", "pursuit", "stop"]


def test_arrival_candidate_then_nonarrival_resets_and_replans_from_second_preview(monkeypatch):
    result, events, pursuits, _searches = run_visual_preview_sequence(
        monkeypatch,
        [
            arrival_preview(0.56, "2026-09-26T16:00:08+00:00"),
            arrival_preview(0.40, "2026-09-26T16:00:09+00:00"),
        ],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["history"][1]["arrival_candidate_reset"] is True
    assert result["actions_executed"] == len(pursuits) == 1
    assert [event[0] for event in events] == ["preview", "stop", "preview", "pursuit", "stop"]


@pytest.mark.parametrize("second", [
    arrival_preview(0.56, "2026-09-26T16:00:00+00:00"),
    arrival_preview(0.56, "2026-09-26T16:00:09+00:00", ambiguous=True),
    arrival_preview(0.56, "2026-09-26T16:00:09+00:00", confirmed=False),
])
def test_stale_or_ambiguous_confirmation_cannot_complete_arrival(monkeypatch, second):
    result, _events, _pursuits, _searches = run_visual_preview_sequence(
        monkeypatch,
        [arrival_preview(0.56, "2026-09-26T16:00:08+00:00"), second],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert result["history"][1]["arrival_candidate_reset"] is True


def test_malformed_confirmation_geometry_resets_arrival_candidate(monkeypatch):
    malformed = arrival_preview(0.56, "2026-09-26T16:00:09+00:00")
    malformed["bbox"]["x2"] = malformed["bbox"]["x1"]
    result, _events, _pursuits, _searches = run_visual_preview_sequence(
        monkeypatch,
        [arrival_preview(0.56, "2026-09-26T16:00:08+00:00"), malformed],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["history"][1]["arrival_candidate_reset"] is True


def test_same_source_timestamp_is_not_two_arrival_confirmations(monkeypatch):
    stamp = "2026-09-26T16:00:09+00:00"
    result, events, pursuits, searches = run_visual_preview_sequence(
        monkeypatch,
        [arrival_preview(0.56, stamp), arrival_preview(0.57, stamp)],
        max_actions=1,
    )
    assert result["arrived_at_marvin"] is False
    assert result["completed"] is False
    assert result["reason"] == "find_marvin_arrival_confirmation_not_independent"
    assert result["actions_executed"] == 0
    assert pursuits == searches == []
    assert [event[0] for event in events] == ["preview", "stop", "preview", "stop"]


def test_arrival_stops_before_any_search_or_pursuit_executor(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit()], arrivals=[arrived(), arrived()], max_actions=1)
    assert result["ok"] is result["completed"] is result["arrived_at_marvin"] is True
    assert result["reason"] == "arrived_at_marvin"
    assert result["actions_executed"] == 0
    assert len(providers) == len(evaluations) == len(arrivals) == 2
    assert searches == pursuits == []
    assert result["history"][-1]["route"] == "arrival"
    assert result["history"][-2]["route"] == "arrival_confirmation"
    assert result["arrival_observations_confirmed"] == 2


def test_arrival_claim_in_searching_state_fails_closed_without_motion(monkeypatch):
    result, _providers, _evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False)], arrivals=[arrived()], max_actions=1)
    assert len(arrivals) == 1 and searches == pursuits == []
    assert result["reason"] == "marvin_arrival_evaluation_inconsistent"
    assert result["actions_executed"] == 0


def test_nonrouting_states_execute_no_executor(monkeypatch):
    for state in ("CANDIDATE_SEEN", "MARVIN_LOCKED", "SAME_IDENTITY_REACQUIRED", "INSUFFICIENT_EVIDENCE"):
        result, providers, evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit(state=state, authorized=False)])
        assert result["actions_executed"] == 0 and len(providers) == len(evaluations) == 1
        assert searches == pursuits == []


def test_search_then_ready_routes_one_executor_per_fresh_iteration(monkeypatch):
    stops = []
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit()], max_actions=2,
        stop_after_action=lambda: stops.append(True) or {"ok": True})
    assert len(providers) == len(evaluations) == 2
    assert len(searches) == len(pursuits) == 1
    assert [entry["route"] for entry in result["history"]] == ["search", "pursuit"]
    assert result["actions_executed"] == 2
    assert len(stops) == 2
    assert result["history"][0]["action_budget_consumed"] is True
    assert result["history"][0]["stop_result"]["ok"] is True


def test_nested_preview_schema_search_reacquires_then_returns_to_pursuit(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    manager.lidar_session = "offline-search-session"
    stamp_one = "2026-09-26T16:00:08+00:00"
    stamp_two = "2026-09-26T16:00:09+00:00"
    absent = {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "identity_confirmed": False,
        "source_timestamp": stamp_one,
        "tracking": {
            "active": False, "target_label": "marvin",
            "source": "marvin_local_tracker", "vision_timestamp": stamp_one,
            "image_width": 640, "image_height": 480, "bbox": None,
        },
    }
    reacquired = {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "identity_confirmed": True,
        "motion_authorized_marvin_candidate": True,
        "source_timestamp": stamp_two,
        "tracking": {
            "active": True, "target_label": "marvin",
            "source": "marvin_local_tracker", "vision_timestamp": stamp_two,
            "image_width": 640, "image_height": 480,
            "bbox": {"x1": 247, "y1": 239, "x2": 387, "y2": 395},
            "identity_ambiguous": False,
        },
    }
    evidence_values = iter((absent, reacquired))
    events = []
    pursuit_inputs = []

    def provider():
        value = next(evidence_values)
        events.append(("preview", value["source_timestamp"]))
        return {
            "preview_result": value,
            "target_lock_result": {},
            "target_lock_snapshot": {"tracking_mode": "UNLOCKED"},
            "selected_identity_id": None,
        }

    def guarded_turn(*args, **kwargs):
        events.append(("guarded_turn", args[0]))
        return {"ok": True, "permitted": True, "confirmed_forwarded": True}

    def pursuit_step(preview_value, *_args, **_kwargs):
        pursuit_inputs.append(preview_value)
        events.append(("pursuit", None))
        return successful_pursuit()

    def stop():
        events.append(("stop", None))
        return {"ok": True, "stopped": True}

    monkeypatch.setattr(manager, "execute_guarded_turn", guarded_turn)
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", pursuit_step)
    result = manager.execute_find_marvin_controller(
        provider, max_actions=2, now="2026-09-26T16:00:10+00:00",
        stop_after_action=stop,
    )

    assert result["reason"] == "find_marvin_action_limit_reached"
    assert [entry["route"] for entry in result["history"]] == ["search", "pursuit"]
    assert result["history"][0]["search_step_result"]["planner"]["preview_status"] == "no_target"
    assert result["history"][0]["search_action"] == "turn_left"
    assert result["history"][1]["pursuit_state"] == "VISUAL_READY_TO_APPROACH"
    assert result["history"][1]["pursuit_authorized"] is True
    assert pursuit_inputs == [reacquired]
    assert events == [
        ("preview", stamp_one), ("guarded_turn", "LEFT"), ("stop", None),
        ("preview", stamp_two), ("pursuit", None), ("stop", None),
    ]


def test_arrival_after_pursuit_uses_fresh_second_iteration_without_extra_action(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit(), pursuit()],
        arrivals=[not_arrived(), arrived(), arrived()], max_actions=2)
    assert len(providers) == len(evaluations) == len(arrivals) == 3
    assert searches == [] and len(pursuits) == result["actions_executed"] == 1
    assert result["completed"] is result["arrived_at_marvin"] is True


def test_arrival_after_search_then_pursuit_stops_before_next_action(monkeypatch):
    result, _providers, _evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit(), pursuit(), pursuit()],
        arrivals=[not_arrived(), not_arrived(), arrived(), arrived()], max_actions=3)
    assert len(arrivals) == 4 and len(searches) == len(pursuits) == 1
    assert result["actions_executed"] == 2 and result["reason"] == "arrived_at_marvin"


def test_pursuit_then_reacquire_routes_to_search(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit("REACQUIRE_REQUIRED", False)], max_actions=2)
    assert len(searches) == len(pursuits) == 1
    assert [entry["route"] for entry in result["history"]] == ["pursuit", "search"]


def test_candidate_or_bridge_only_stops_without_follow_on_motion(monkeypatch):
    for state in ("CANDIDATE_SEEN", "SAME_IDENTITY_REACQUIRED"):
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [pursuit("SEARCHING", False), pursuit(state, False)], max_actions=3)
        assert len(searches) == 1 and pursuits == []
        assert result["actions_executed"] == 1


def test_identity_change_between_search_and_pursuit_fails_closed(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit(identity="marvin-2")],
        identities=["marvin-1", "marvin-2"], max_actions=3)
    assert result["reason"] == "find_marvin_identity_changed"
    assert len(searches) == 1 and pursuits == []


def test_global_budget_is_shared_across_search_and_pursuit(monkeypatch):
    states = [pursuit("SEARCHING", False), pursuit("SEARCHING", False), pursuit(), pursuit()]
    result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, states, max_actions=4)
    assert result["reason"] == "find_marvin_action_limit_reached"
    assert len(searches) == len(pursuits) == 2
    assert result["actions_executed"] == len(searches) + len(pursuits) == 4


def test_autonomous_stop_callback_runs_after_every_action_before_fresh_preview(monkeypatch):
    stops = []
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit()] * 6,
        max_actions=6,
        stop_after_action=lambda: stops.append(True) or {"ok": True},
    )
    assert result["actions_executed"] == 6
    assert len(providers) == len(evaluations) == len(stops) == 6
    assert searches == [] and len(pursuits) == 6
    assert all(entry["stop_result"] == {"ok": True} for entry in result["history"])


def test_arrival_before_sixth_action_stops_without_exhausting_six_action_budget(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit()] * 7,
        arrivals=[not_arrived()] * 5 + [arrived(), arrived()],
        max_actions=6,
        stop_after_action=lambda: {"ok": True},
    )
    assert result["completed"] is result["arrived_at_marvin"] is True
    assert result["actions_executed"] == len(pursuits) == 5
    assert searches == [] and len(providers) == len(evaluations) == len(arrivals) == 7


def test_failed_post_action_stop_prevents_a_new_preview_or_action(monkeypatch):
    result, providers, _evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit()], max_actions=2,
        stop_after_action=lambda: {"ok": False},
    )
    assert result["reason"] == "find_marvin_post_action_stop_failed"
    assert result["actions_executed"] == 1
    assert len(providers) == len(pursuits) == 1 and searches == []


def test_nonphysical_stale_replan_stops_then_requires_fresh_state_without_spending_budget(monkeypatch):
    stops = []
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit(), pursuit()],
        pursuit_steps=[nonphysical_stale_replan(), successful_pursuit()],
        max_actions=1,
        stop_after_action=lambda: stops.append(True) or {"ok": True},
    )
    assert result["ok"] is True and result["reason"] == "find_marvin_action_limit_reached"
    assert result["actions_executed"] == 1 and result["stale_replans"] == 1
    assert len(providers) == len(evaluations) == len(pursuits) == len(stops) == 2
    assert searches == []
    assert result["history"][0]["action_budget_consumed"] is False
    assert result["history"][1]["action_budget_consumed"] is True


def test_physical_or_uncertain_stale_replan_consumes_budget_without_pursuit_failure(monkeypatch):
    stops = []
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit()], pursuit_steps=[physical_stale_replan()], max_actions=1,
        stop_after_action=lambda: stops.append(True) or {"ok": True},
    )
    assert result["ok"] is True and result["reason"] == "find_marvin_action_limit_reached"
    assert result["actions_executed"] == 1 and result["stale_replans"] == 0
    assert len(providers) == len(evaluations) == len(pursuits) == len(stops) == 1
    assert searches == [] and result["history"][0]["action_budget_consumed"] is True


def test_second_nonphysical_stale_replan_terminates_with_finite_separate_cap(monkeypatch):
    stops = []
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(
        monkeypatch,
        [pursuit(), pursuit()],
        pursuit_steps=[nonphysical_stale_replan(), nonphysical_stale_replan()],
        max_actions=1,
        stop_after_action=lambda: stops.append(True) or {"ok": True},
    )
    assert result["ok"] is False
    assert result["reason"] == "find_marvin_nonphysical_stale_replan_limit_reached"
    assert result["actions_executed"] == 0 and result["stale_replans"] == 2
    assert len(providers) == len(evaluations) == len(pursuits) == len(stops) == 2
    assert searches == []


def test_default_budget_stops_an_alternating_stream_at_six(monkeypatch):
    states = [pursuit("SEARCHING", False), pursuit()] * 3
    result, providers, evaluations, _arrivals, searches, pursuits = invoke(monkeypatch, states)
    assert result["max_actions"] == result["actions_executed"] == 6
    assert len(providers) == len(evaluations) == len(searches) + len(pursuits) == 6


def test_arrival_check_does_not_consume_budget_or_add_final_observation(monkeypatch):
    result, providers, evaluations, arrivals, searches, pursuits = invoke(
        monkeypatch, [pursuit(), pursuit()], max_actions=1)
    assert result["actions_executed"] == len(pursuits) == 1 and searches == []
    assert len(providers) == len(evaluations) == len(arrivals) == 1
    assert result["reason"] == "find_marvin_action_limit_reached"


def test_executor_exceptions_or_failures_have_no_fallback(monkeypatch):
    cases = (
        (pursuit("SEARCHING", False), [RuntimeError("offline")], None, "find_marvin_search_step_exception"),
        (pursuit("SEARCHING", False), [{"ok": False}], None, "find_marvin_search_step_failed"),
        (pursuit(), None, [RuntimeError("offline")], "find_marvin_pursuit_step_exception"),
        (pursuit(), None, [{"ok": False}], "find_marvin_pursuit_step_failed"),
    )
    for state, search_steps, pursuit_steps, reason in cases:
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [state], search_steps=search_steps, pursuit_steps=pursuit_steps)
        assert result["reason"] == reason and result["actions_executed"] == 1
        assert len(searches) + len(pursuits) == 1


def test_arrival_evaluator_exception_or_malformed_result_stops_before_motion(monkeypatch):
    cases = (
        [RuntimeError("offline")], [None],
        [{"ok": False, "arrived_at_marvin": False, "selected_identity_id": "marvin-1"}],
        [{"ok": True, "arrived_at_marvin": True, "selected_identity_id": "other"}],
    )
    for arrivals in cases:
        result, _providers, _evaluations, arrival_calls, searches, pursuits = invoke(
            monkeypatch, [pursuit()], arrivals=arrivals, max_actions=1)
        assert len(arrival_calls) == 1 and searches == pursuits == []
        assert result["actions_executed"] == 0
        assert result["reason"] in {"marvin_arrival_evaluation_failed", "marvin_arrival_evaluation_inconsistent"}


def test_arrival_receives_no_lidar_authority(monkeypatch):
    _result, _providers, _evaluations, calls, _searches, _pursuits = invoke(
        monkeypatch, [pursuit()], max_actions=1)
    assert len(calls) == 1
    assert set(calls[0][1]) == {"selected_identity_id", "now"}


def test_dry_run_reports_existing_routes_without_any_executor(monkeypatch):
    cases = (
        (pursuit("SEARCHING", False), "search"),
        (pursuit("REACQUIRE_REQUIRED", False), "search"),
        (pursuit(), "pursuit"),
    )
    for state, route in cases:
        result, providers, evaluations, arrivals, searches, pursuits = invoke(
            monkeypatch, [state], max_actions=1, dry_run=True)
        assert result["dry_run"] is True and result["next_route"] == route
        assert result["actions_executed"] == 0
        assert len(providers) == len(evaluations) == len(arrivals) == 1
        assert searches == pursuits == []


def test_no_motion_replan_false_and_search_complete_stop(monkeypatch):
    cases = (
        (pursuit("SEARCHING", False), [{"ok": True, "motion_executed": False}], None, "find_marvin_search_step_no_motion"),
        (pursuit("SEARCHING", False), [{"ok": True, "search_action": "turn_left", "motion_executed": True, "replan_required": False}], None, "find_marvin_search_step_replan_required"),
        (pursuit("SEARCHING", False), [{"ok": True, "search_action": "search_complete", "motion_executed": False}], None, "find_marvin_search_complete"),
        (pursuit(), None, [{"ok": True, "motion_executed": False}], "find_marvin_pursuit_step_no_motion"),
        (pursuit(), None, [{"ok": True, "motion_executed": True, "replan_required": False}], "find_marvin_pursuit_step_replan_required"),
    )
    for state, search_steps, pursuit_steps, reason in cases:
        result, _providers, _evaluations, _arrivals, searches, pursuits = invoke(
            monkeypatch, [state], search_steps=search_steps, pursuit_steps=pursuit_steps)
        assert result["reason"] == reason and len(searches) + len(pursuits) == 1


def test_invalid_limits_provider_and_evaluator_fail_closed(monkeypatch):
    manager = BehaviorManager(robot_client=object())
    calls = []
    monkeypatch.setattr(manager, "execute_marvin_search_step", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: calls.append(True))
    for limit in (0, -1, True, 1.5, "6", None):
        assert manager.execute_find_marvin_controller(lambda: evidence(), max_actions=limit)["reason"] == "invalid_find_marvin_action_limit"
    assert manager.execute_find_marvin_controller(None)["reason"] == "find_marvin_state_provider_unavailable"
    assert manager.execute_find_marvin_controller(lambda: None)["reason"] == "find_marvin_state_evidence_malformed"
    assert manager.execute_find_marvin_controller(
        lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )["reason"] == "find_marvin_state_provider_exception"
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert manager.execute_find_marvin_controller(lambda: evidence())["reason"] == "find_marvin_pursuit_evaluation_exception"
    assert calls == []


def test_search_history_advances_only_for_successful_turns(monkeypatch):
    result, _providers, _evaluations, _arrivals, searches, _pursuits = invoke(
        monkeypatch, [pursuit("SEARCHING", False), pursuit("SEARCHING", False)], max_actions=2)
    assert searches[0][1]["prior_search_history"] == []
    assert searches[1][1]["prior_search_history"] == [{"selected_search_action": "turn_left"}]
    assert result["history"][0]["search_action"] == "turn_left"


def test_deterministic_history_and_inputs_not_mutated(monkeypatch):
    fixture = evidence()
    before = deepcopy(fixture)
    manager = BehaviorManager(robot_client=object())
    monkeypatch.setattr(behavior_manager_module, "evaluate_marvin_pursuit_state", lambda *a, **k: pursuit())
    monkeypatch.setattr(manager, "execute_marvin_pursuit_step", lambda *a, **k: successful_pursuit())
    first = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    second = manager.execute_find_marvin_controller(lambda: fixture, max_actions=1)
    assert first == second and fixture == before


def test_controller_source_delegates_arrival_without_geometry_or_motion_logic():
    source = open("behavior_manager.py", encoding="utf-8").read()
    start = source.index("    def execute_find_marvin_controller(")
    end = source.index("    def execute_marvin_pursuit_step(", start)
    controller = source[start:end]
    for forbidden in ("robot.local_forward", "execute_guarded_turn", "execute_local_obstacle_avoidance_step",
                      "execute_local_obstacle_avoidance_loop", "0.545833", "0.160339", "bbox_width",
                      "height_fraction", "area_fraction", "lidar", "while true",
                      "self._execute_find", "self.execute_behavior"):
        assert forbidden not in controller.lower()
    assert controller.count("self.execute_marvin_search_step(") == 1
    assert controller.count("self.execute_marvin_pursuit_step(") == 1
    assert controller.count("evaluate_marvin_arrival(") == 1
