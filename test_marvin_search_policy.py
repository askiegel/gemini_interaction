"""Offline contracts for pure bounded Marvin local-scan planning."""

from copy import deepcopy
import inspect

import pytest

from marvin_search_policy import (
    DEFAULT_LOCAL_SCAN_PLAN,
    DEFAULT_MAX_SEARCH_ACTIONS,
    plan_marvin_search_step,
)
from marvin_pursuit_state import evaluate_marvin_pursuit_state


STAMP = "2026-09-25T12:00:00+00:00"


def state(name="SEARCHING", identity=None):
    return {"state": name, "selected_identity_id": identity}


def preview(**updates):
    value = {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "target_found": True,
        "source": "marvin_local_tracker", "identity_confirmed": True,
        "motion_authorized_marvin_candidate": True,
        "source_timestamp": STAMP, "image_width": 640, "image_height": 480,
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
    }
    value.update(updates)
    return value


def plan(current, **kwargs):
    return plan_marvin_search_step(current, now=STAMP, **kwargs)


def test_searching_fresh_candidate_hands_back_without_scan_motion():
    result = plan(state(), preview_result=preview(proposal_label="teddy bear"))
    assert result["selected_search_action"] == "preview_only"
    assert result["candidate_available"] is True and result["reacquired"] is False
    assert result["preview_status"] == "fresh_candidate"


def test_search_accepts_dashboard_style_nested_tracking_preview():
    live_shape = {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "identity_confirmed": True,
        "motion_authorized_marvin_candidate": True,
        "source_timestamp": STAMP, "source": "marvin_local_tracker",
        "tracking": {
            "active": True, "target_label": "marvin",
            "vision_timestamp": STAMP, "source": "marvin_local_tracker",
            "image_width": 640, "image_height": 480,
            "bbox": {"x1": 247, "y1": 239, "x2": 387, "y2": 395},
            "identity_ambiguous": False,
        },
    }
    result = plan(state(), preview_result=live_shape)
    assert result["reason"] == "marvin_candidate_requires_identity_confirmation"
    assert result["preview_status"] == "fresh_candidate"
    assert result["candidate_available"] is True
    assert result["selected_search_action"] == "preview_only"


def test_searching_empty_preview_plans_first_then_next_deterministic_scan():
    first = plan(state(), preview_result=preview(target_found=False, identity_confirmed=False))
    second = plan(state(), preview_result=preview(target_found=False, identity_confirmed=False), prior_search_history=[first])
    assert first["selected_search_action"] == DEFAULT_LOCAL_SCAN_PLAN[0]
    assert second["selected_search_action"] == DEFAULT_LOCAL_SCAN_PLAN[1]
    assert first["search_actions_used"] == 1 and second["search_actions_used"] == 2


def test_scan_plan_exhausts_without_another_action():
    history = [{"selected_search_action": action} for action in DEFAULT_LOCAL_SCAN_PLAN]
    result = plan(state(), preview_result=None, prior_search_history=history)
    assert result["completed"] is True
    assert result["selected_search_action"] == "search_complete"


def test_max_one_action_never_plans_a_second():
    first = plan(state(), preview_result=None, max_search_actions=1)
    second = plan(state(), preview_result=None, max_search_actions=1, prior_search_history=[first])
    assert first["selected_search_action"] == "turn_left"
    assert second["selected_search_action"] == "search_complete"


def test_invalid_limits_and_history_fail_closed():
    for limit in (0, -1, True, 1.5, "4", None):
        assert plan(state(), max_search_actions=limit)["reason"] == "invalid_marvin_search_action_limit"
    assert plan(state(), prior_search_history="bad")["selected_search_action"] == "fail_closed"


def test_reacquire_requires_actual_locked_same_identity():
    waiting = plan(state("REACQUIRE_REQUIRED", "marvin-1"), selected_identity_id="marvin-1", preview_result=preview())
    restored = plan(
        state("REACQUIRE_REQUIRED", "marvin-1"), selected_identity_id="marvin-1",
        preview_result=preview(), target_lock_snapshot={"tracking_mode": "LOCKED", "locked_identity_id": "marvin-1"},
    )
    assert waiting["reacquired"] is False and waiting["selected_search_action"] == "preview_only"
    assert restored["reacquired"] is True and restored["selected_search_action"] == "reacquired"


def test_different_or_ambiguous_identity_never_claims_reacquisition():
    for classification in ("different_identity", "ambiguous_identity"):
        result = plan(
            state("REACQUIRE_REQUIRED", "marvin-1"), selected_identity_id="marvin-1",
            bridge_result={"classification": classification}, preview_result=preview(),
        )
        assert result["reacquired"] is False
        assert result["selected_search_action"] == "fail_closed"


def test_stale_preview_is_not_accepted_and_malformed_preview_fails_closed():
    stale = plan(state(), preview_result=preview(source_timestamp="2026-09-25T11:59:56+00:00"))
    malformed = plan(state(), preview_result=preview(bbox=None))
    assert stale["candidate_available"] is False and stale["selected_search_action"] == "turn_left"
    assert stale["preview_status"] == "stale"
    assert malformed["selected_search_action"] == "fail_closed"
    assert malformed["preview_status"] == "malformed"


def test_no_target_stale_ambiguous_and_unconfirmed_are_bounded_noncandidate_states():
    cases = (
        (preview(target_found=False, identity_confirmed=False), "no_target"),
        (preview(ambiguous=True), "ambiguous"),
        (preview(identity_confirmed=False), "unconfirmed"),
        (preview(source_timestamp="2026-09-25T11:59:56+00:00"), "stale"),
    )
    for candidate, expected_status in cases:
        result = plan(state(), preview_result=candidate)
        assert result["preview_status"] == expected_status
        assert result["candidate_available"] is False
        assert result["selected_search_action"] == "turn_left"


@pytest.mark.parametrize("label", ("laptop", "chair", "suitcase"))
def test_incompatible_confirmed_preview_does_not_suppress_bounded_search(label):
    candidate = preview(
        proposal_label=label,
        motion_authorized_marvin_candidate=False,
        confirmation_diagnostics={"confirmation_status": "target_confirmed"},
        tracking={"active": True, "target_label": "marvin"},
    )
    result = plan(state(), preview_result=candidate)
    assert result["preview_status"] == "no_target"
    assert result["candidate_available"] is False
    assert result["selected_search_action"] == "turn_left"
    assert result["selected_search_action"] != "preview_only"


def test_missing_motion_authority_field_is_no_target_and_does_not_suppress_search():
    legacy = preview(proposal_label="teddy bear")
    legacy.pop("motion_authorized_marvin_candidate")
    result = plan(state(), preview_result=legacy)
    assert result["preview_status"] == "no_target"
    assert result["candidate_available"] is False
    assert result["selected_search_action"] == "turn_left"


def test_fresh_reacquired_nested_preview_returns_to_normal_visual_pursuit():
    live_shape = {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "identity_confirmed": True,
        # This nested dashboard shape models a valid reacquired teddy-bear
        # Marvin Preview emitted by BehaviorManager.
        "motion_authorized_marvin_candidate": True,
        "source_timestamp": STAMP, "source": "marvin_local_tracker",
        "tracking": {
            "active": True, "target_label": "marvin",
            "vision_timestamp": STAMP, "source": "marvin_local_tracker",
            "image_width": 640, "image_height": 480,
            "bbox": {"x1": 247, "y1": 239, "x2": 387, "y2": 395},
            "identity_ambiguous": False,
        },
    }
    search = plan(state(), preview_result=live_shape)
    pursuit = evaluate_marvin_pursuit_state(
        live_shape, {}, {"tracking_mode": "UNLOCKED"}, now=STAMP,
    )
    assert search["selected_search_action"] == "preview_only"
    assert pursuit["state"] == "VISUAL_READY_TO_APPROACH"
    assert pursuit["pursuit_authorized"] is True


def test_selected_identity_is_preserved_and_outputs_are_deterministic_without_mutation():
    current = state("REACQUIRE_REQUIRED", "marvin-1")
    history = [{"selected_search_action": "turn_left"}]
    candidate = preview(target_found=False, identity_confirmed=False)
    before = deepcopy((current, history, candidate))
    first = plan(current, selected_identity_id="marvin-1", preview_result=candidate, prior_search_history=history)
    second = plan(current, selected_identity_id="marvin-1", preview_result=candidate, prior_search_history=history)
    assert first == second and first["selected_identity_id"] == "marvin-1"
    assert (current, history, candidate) == before


def test_policy_is_pure_bounded_and_has_no_motion_dependencies():
    import marvin_search_policy
    source = inspect.getsource(marvin_search_policy).lower()
    for forbidden in ("random", "while true", "robot_client", "cmd_vel", "rospy", "rclpy", "stanford", "nav2", "requests."):
        assert forbidden not in source
    assert DEFAULT_MAX_SEARCH_ACTIONS == 4
