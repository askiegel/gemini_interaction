"""Offline contracts for pure Marvin Preview/TargetLock state orchestration."""

from copy import deepcopy
import inspect

from marvin_pursuit_state import (
    CANDIDATE_SEEN, INSUFFICIENT_EVIDENCE, MARVIN_LOCKED, REACQUIRE_REQUIRED,
    READY_TO_APPROACH, SAME_IDENTITY_REACQUIRED, SEARCHING,
    VISUAL_READY_TO_ALIGN, VISUAL_READY_TO_APPROACH,
    evaluate_marvin_pursuit_state,
)


STAMP = "2026-09-25T12:00:00+00:00"


def preview(**updates):
    value = {"ok": True, "preview": True, "authoritative": False,
             "target": "marvin", "target_found": True, "source": "marvin_local_tracker",
             "identity_confirmed": True, "source_timestamp": STAMP,
             "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
             "tracking": {"target_label": "marvin", "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}}}
    value.update(updates)
    return value


def snapshot(identity=None, mode="UNLOCKED"):
    return {"locked_identity_id": identity, "tracking_mode": mode}


def locked(identity="marvin-1", **updates):
    value = {"found": True, "stale": False, "tracking_mode": "LOCKED",
             "identity_id": identity, "locked_identity_id": identity,
             "entity_id": "entity-1", "last_seen": STAMP,
             "cx": 320.0, "image_width": 640.0}
    value.update(updates)
    return value


def evaluate(candidate=None, result=None, state=None, **kwargs):
    return evaluate_marvin_pursuit_state(candidate, result, state, now=STAMP, **kwargs)


def test_no_preview_or_lock_is_searching():
    assert evaluate(None, None, snapshot())["state"] == SEARCHING


def test_preview_without_image_dimensions_cannot_authorize_visual_session():
    value = evaluate(preview(), None, snapshot())
    assert value["pursuit_authorized"] is False


def test_fresh_unambiguous_preview_without_target_lock_authorizes_visual_session_only():
    value = evaluate(
        preview(
            image_width=640.0, image_height=480.0,
            bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 220.0},
        ), None, snapshot(),
    )
    assert value["state"] == VISUAL_READY_TO_ALIGN
    assert value["pursuit_authorized"] is True
    assert value["selected_identity_id"] is value["entity_id"] is None
    assert value["horizontal_error"] == 130.0

    centered = evaluate(
        preview(
            image_width=640.0, image_height=480.0,
            bbox={"x1": 270.0, "y1": 20.0, "x2": 370.0, "y2": 220.0},
            marvin_continuity={"tracker_id": 7, "tracker_source": "diagnostic"},
        ), None, snapshot(),
    )
    assert centered["state"] == VISUAL_READY_TO_APPROACH
    assert centered["selected_identity_id"] is None


def test_visual_session_fails_closed_on_ambiguous_preview():
    value = evaluate(
        preview(
            image_width=640.0, image_height=480.0,
            target_observation={"identity_ambiguous": True},
        ), None, snapshot(),
    )
    assert value["pursuit_authorized"] is False


def live_preview(*, centered=False, **updates):
    """Exact public Preview shape captured during 2026-09-27 preflight."""
    tracking_bbox = (
        {"x1": 270, "y1": 200, "x2": 370, "y2": 466}
        if centered else {"x1": 281, "y1": 200, "x2": 497, "y2": 466}
    )
    value = {
        "ok": True, "preview": True, "authoritative": False,
        "source": "marvin_local_tracker", "target": "marvin",
        "identity_confirmed": True,
        "source_timestamp": STAMP, "vision_timestamp": STAMP,
        "marvin_continuity": {
            "tracker_id": 13, "tracker_source": "marvin_continuity_botsort",
        },
        "tracking": {
            "active": True, "behavior": "FIND_OBJECT", "state": "PREVIEW",
            "source": "marvin_local_tracker", "target_label": "marvin",
            "bbox": tracking_bbox, "image_width": 640, "image_height": 480,
            "horizontal_error": 0 if centered else 69,
            "identity_ambiguous": False,
        },
    }
    value.update(updates)
    return value


def test_exact_live_nested_tracking_preview_authorizes_alignment_without_identity():
    value = evaluate(live_preview(), None, snapshot())
    assert value["state"] == VISUAL_READY_TO_ALIGN
    assert value["horizontal_error"] == 69.0
    assert value["selected_identity_id"] is value["entity_id"] is None


def test_exact_live_nested_tracking_preview_centered_authorizes_approach():
    value = evaluate(live_preview(centered=True), None, snapshot())
    assert value["state"] == VISUAL_READY_TO_APPROACH


def test_live_nested_tracking_preview_fails_closed_for_missing_stale_ambiguous_or_unconfirmed():
    cases = (
        live_preview(tracking={"active": True, "target_label": "marvin"}),
        live_preview(source_timestamp="2026-09-25T11:59:56+00:00"),
        live_preview(tracking={**live_preview()["tracking"], "identity_ambiguous": True}),
        live_preview(identity_confirmed=False),
    )
    assert all(not evaluate(item, None, snapshot())["pursuit_authorized"] for item in cases)


def test_waiting_lock_requires_reacquisition():
    value = evaluate(preview(), {"found": False, "tracking_mode": "WAITING_FOR_IDENTITY"}, snapshot("marvin-1", "WAITING_FOR_IDENTITY"))
    assert value["state"] == REACQUIRE_REQUIRED


def test_bridge_same_identity_while_waiting_never_authorizes():
    value = evaluate(preview(identity_id="marvin-1"), {"found": False, "tracking_mode": "WAITING_FOR_IDENTITY"}, snapshot("marvin-1", "WAITING_FOR_IDENTITY"))
    assert value["state"] == SAME_IDENTITY_REACQUIRED
    assert value["pursuit_authorized"] is False


def test_restored_fresh_target_lock_with_geometry_is_ready():
    value = evaluate(None, locked(), snapshot("marvin-1", "LOCKED"))
    assert value["state"] == READY_TO_APPROACH
    assert value["pursuit_authorized"] is True
    assert value["geometry_usable"] is True


def test_fresh_lock_missing_geometry_is_not_ready():
    value = evaluate(None, locked(cx=None), snapshot("marvin-1", "LOCKED"))
    assert value["state"] == MARVIN_LOCKED
    assert value["pursuit_authorized"] is False


def test_stale_lock_requires_reacquisition():
    value = evaluate(None, locked(last_seen="2026-09-25T11:59:56+00:00"), snapshot("marvin-1", "LOCKED"))
    assert value["state"] == REACQUIRE_REQUIRED


def test_different_or_ambiguous_identity_fails_closed():
    assert evaluate(None, locked(identity="other"), snapshot("marvin-1", "LOCKED"))["state"] == INSUFFICIENT_EVIDENCE
    assert evaluate(None, locked(identity_ambiguous=True), snapshot("marvin-1", "LOCKED"))["state"] == INSUFFICIENT_EVIDENCE


def test_malformed_inputs_and_missing_freshness_fail_closed():
    assert evaluate(preview(bbox=None), None, snapshot())["state"] == INSUFFICIENT_EVIDENCE
    assert evaluate(preview(source_timestamp=None), None, snapshot())["state"] == INSUFFICIENT_EVIDENCE
    assert evaluate(None, "not-a-result", snapshot("marvin-1"))["state"] == INSUFFICIENT_EVIDENCE
    assert evaluate(None, locked(last_seen=None), snapshot("marvin-1", "LOCKED"))["state"] == INSUFFICIENT_EVIDENCE


def test_stale_preview_cannot_create_candidate_or_reacquisition_authority():
    value = evaluate(preview(source_timestamp="2026-09-25T11:59:56+00:00"), None, snapshot())
    assert value["state"] == SEARCHING


def test_semantic_bbox_overlap_and_tracker_reuse_never_authorize():
    semantic = evaluate(preview(), None, snapshot())
    overlap = evaluate(preview(), {"found": False}, snapshot("marvin-1", "WAITING_FOR_IDENTITY"))
    reused = evaluate(preview(tracking={"target_label": "marvin", "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}, "track_id": "reused"}), None, snapshot())
    assert all(not item["pursuit_authorized"] for item in (semantic, overlap, reused))


def test_waiting_preserves_selected_identity_and_bridge_must_be_consistent():
    waiting = evaluate(None, {"found": False, "tracking_mode": "WAITING_FOR_IDENTITY"}, snapshot("marvin-1", "WAITING_FOR_IDENTITY"))
    assert waiting["selected_identity_id"] == "marvin-1"
    bad_bridge = {"classification": "same_identity_reacquired", "selected_identity_id": "marvin-1", "candidate_identity_id": "marvin-1"}
    value = evaluate(preview(), None, snapshot("marvin-1"), bridge_result=bad_bridge)
    assert value["state"] == INSUFFICIENT_EVIDENCE


def test_deterministic_and_inputs_not_mutated():
    values = (preview(identity_id="marvin-1"), locked(), snapshot("marvin-1", "LOCKED"))
    before = deepcopy(values)
    assert evaluate(*values) == evaluate(*values)
    assert values == before


def test_policy_has_no_motion_ros_or_robot_dependencies():
    import marvin_pursuit_state
    source = inspect.getsource(marvin_pursuit_state).lower()
    for forbidden in ("robot_client", "cmd_vel", "rospy", "rclpy", "stanford", "nav2", "guarded_turn", "local_forward"):
        assert forbidden not in source
