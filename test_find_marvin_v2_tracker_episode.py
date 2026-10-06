"""Strict V2 tracker-episode association contracts."""

import threading
from types import SimpleNamespace

import pytest

from behavior_manager import BehaviorManager


LOW = {"x1": 212, "y1": 0, "x2": 640, "y2": 384}
LOW_NEAR = {"x1": 221, "y1": 0, "x2": 619, "y2": 376}
LOW_NEAR_2 = {"x1": 207, "y1": 0, "x2": 640, "y2": 387}
HIGH = {"x1": 355, "y1": 16, "x2": 640, "y2": 403}
HIGH_NEAR = {"x1": 364, "y1": 4, "x2": 640, "y2": 414}
HIGH_NEAR_2 = {"x1": 362, "y1": 6, "x2": 640, "y2": 407}


def _candidate(bbox):
    return {"bbox": dict(bbox), "image_width": 640, "image_height": 480}


@pytest.mark.parametrize(("initial", "associated", "competing", "competing_near", "competing_near_2"), [
    (LOW, LOW_NEAR, HIGH, HIGH_NEAR, HIGH_NEAR_2),
    (HIGH, HIGH_NEAR, LOW, LOW_NEAR, LOW_NEAR_2),
])
def test_strict_v2_reuses_tracker_for_associated_proposal_and_rejects_mode_switch(
    initial, associated, competing, competing_near, competing_near_2,
):
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = None
    created = []
    continued = []

    def acquire(candidate, _diagnostics, *, episode, existing_tracker=None, **_kwargs):
        if existing_tracker is None:
            tracker = object()
            created.append(tracker)
            episode["marvin_tracker"] = tracker
        else:
            tracker = existing_tracker
            continued.append(tracker)
        # Model production: the raw semantic proposal is expanded/clamped
        # before seeding, and the confirmed tracker publishes that seed's
        # geometry rather than the raw proposal unchanged.
        bbox = manager._expand_marvin_tracker_seed_bbox(
            candidate["bbox"], candidate["image_width"], candidate["image_height"],
        )
        return {"opencv_tracker": {
            "source_frame_stamp_ns": 100 + len(created) + len(continued),
            "bbox": bbox,
        }}

    manager._acquire_marvin_tracker_observation_from_candidate = acquire
    frame = SimpleNamespace(received_at="2026-10-04T00:00:00+00:00")
    first = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(initial), {}, identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=1, execution_guard=None, frame=frame,
    )
    second = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(associated), {}, identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=2, execution_guard=None, frame=frame,
    )

    assert first["strict_tracker_episode"]["initialized_this_observation"] is True
    assert first["opencv_tracker"]["bbox"] != initial
    assert second["strict_tracker_episode"]["continued_existing_tracker"] is True
    assert len(created) == 1
    assert continued == [created[0]]

    rejected = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(competing), {}, identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=3, execution_guard=None, frame=frame,
    )
    if competing is HIGH:
        # This proposal is large enough to meet the existing visual-arrival
        # height threshold if it reached arrival evaluation.  Association is
        # intentionally checked first, so ARRIVED cannot reseed an unrelated
        # strict episode.
        assert competing["y2"] - competing["y1"] > 480.0 * 0.545833
    assert rejected["found"] is False
    assert rejected["reason"] == "marvin_v2_semantic_tracker_association_failed"
    # The association must compare the expanded/clamped current semantic
    # seed to the prior confirmed tracker box, never raw proposal geometry.
    expanded_competing = manager._expand_marvin_tracker_seed_bbox(
        competing, 640, 480,
    )
    expected_iou = manager._target_bbox_iou(
        {"bbox": expanded_competing}, {"bbox": second["opencv_tracker"]["bbox"]},
    )
    assert rejected["strict_tracker_episode"]["association_iou"] == expected_iou
    assert rejected["strict_tracker_episode"]["association_iou"] < 0.70
    assert manager._marvin_v2_tracker_episode is None
    assert len(created) == 1

    # A rejected competing proposal is not silently reseeded in the same
    # observation.  Only the following request can explicitly establish a
    # new episode, after which later associated requests reuse its object.
    reacquired = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(competing_near), {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=4, execution_guard=None, frame=frame,
    )
    resumed = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(competing_near_2), {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=5, execution_guard=None, frame=frame,
    )
    assert reacquired["strict_tracker_episode"]["initialized_this_observation"] is True
    assert resumed["strict_tracker_episode"]["continued_existing_tracker"] is True
    assert len(created) == 2
    assert continued[-1] is created[-1]


@pytest.mark.parametrize("action", ["turn", "forward"])
def test_controlled_action_continues_same_locked_tracker_on_new_camera_geometry(monkeypatch, action):
    manager = object.__new__(BehaviorManager)
    tracker = object()
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": tracker,
        "tracker_bbox": {"x1": 200, "y1": 20, "x2": 600, "y2": 400},
        "last_tracker_source_frame_stamp_ns": 500,
        "last_tracker_received_at": "2026-10-04T00:00:00+00:00",
        "identity_source": "gemini_marvin_candidate_selection",
        "identity_source_frame_stamp_ns": 450,
        "episode_id": "marvin-v2-450",
    }
    manager.semantic_vision = SimpleNamespace(fetch_frame=object())
    calls = []
    moved_bbox = {"x1": 35, "y1": 24, "x2": 435, "y2": 404}
    opencv = {
        "active": True, "matched": True, "quality": 0.94, "threshold": 0.80,
        "bbox": moved_bbox, "image_width": 640, "image_height": 480,
        "center_x": 235.0, "center_y": 214.0, "horizontal_error": -85.0,
        "source_frame_stamp_ns": 501,
    }
    monkeypatch.setattr(
        manager, "_confirm_marvin_local_tracker_frames",
        lambda received_tracker, **kwargs: calls.append((received_tracker, kwargs)) or {
            "found": True, "bbox": moved_bbox, "cx": 235.0, "cy": 214.0,
            "area": 152000, "image_width": 640, "image_height": 480,
            "source_timestamp": "2026-10-04T00:00:01+00:00",
            "opencv_tracker": opencv,
        },
    )
    assert manager.mark_strict_v2_action_dispatched(500, action) is True

    evidence = manager.observe_find_marvin_v2()
    preview = evidence["preview_result"]
    assert len(calls) == 1
    assert calls[0][0] is tracker
    assert calls[0][1]["minimum_source_frame_stamp_ns"] == 500
    assert preview["identity_source"] == "marvin_locked_tracker_continuity"
    assert preview["identity_confirmed"] is True
    assert preview["post_action_tracker_continuity"] is True
    assert preview["opencv_tracker"]["source_frame_stamp_ns"] == 501
    assert preview["bbox"] == moved_bbox
    assert preview["marvin_tracking_episode"]["state"] == "POST_ACTION_TRACKED"
    assert BehaviorManager._marvin_v2_preview_is_verified(preview) is True
    assert manager._marvin_v2_tracker_episode["marvin_tracker"] is tracker
    assert manager._marvin_v2_tracker_episode["last_tracker_source_frame_stamp_ns"] == 501


def test_failed_post_action_tracker_continuity_clears_lock_and_requires_reverify(monkeypatch):
    manager = object.__new__(BehaviorManager)
    tracker = object()
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": tracker, "tracker_bbox": LOW,
        "last_tracker_source_frame_stamp_ns": 600,
        "last_tracker_received_at": "2026-10-04T00:00:00+00:00",
        "identity_source": "gemini_marvin_candidate_selection",
        "identity_source_frame_stamp_ns": 550,
        "episode_id": "marvin-v2-550",
    }
    manager.semantic_vision = SimpleNamespace(fetch_frame=object())
    monkeypatch.setattr(manager, "_confirm_marvin_local_tracker_frames", lambda *_a, **_k: None)
    assert manager.mark_strict_v2_action_dispatched(600, "turn") is True
    evidence = manager.observe_find_marvin_v2()
    preview = evidence["preview_result"]
    assert preview["identity_confirmed"] is False
    assert preview["reason"] == "post_action_tracker_continuity_lost"
    assert BehaviorManager._marvin_v2_preview_is_verified(preview) is False
    assert manager._marvin_v2_tracker_episode is None


def test_explicit_non_gemini_identity_cannot_inherit_existing_episode():
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": object(), "tracker_bbox": LOW,
        "last_tracker_source_frame_stamp_ns": 700,
    }
    result = manager._acquire_strict_v2_tracker_observation_from_candidate(
        _candidate(LOW_NEAR), {}, identity_source="explicit_identity_mismatch",
        identity_source_frame_stamp_ns=701, execution_guard=None,
        frame=SimpleNamespace(received_at="2026-10-04T00:00:01+00:00"),
    )
    assert result["found"] is False
    assert result["reason"] == "marvin_v2_fresh_identity_invalid"
    assert manager._marvin_v2_tracker_episode is None


def test_post_action_continuity_rejects_non_new_source_stamp(monkeypatch):
    manager = object.__new__(BehaviorManager)
    tracker = object()
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": tracker, "tracker_bbox": LOW,
        "last_tracker_source_frame_stamp_ns": 800,
        "last_tracker_received_at": "2026-10-04T00:00:00+00:00",
        "identity_source": "gemini_marvin_candidate_selection",
        "identity_source_frame_stamp_ns": 750,
        "episode_id": "marvin-v2-750",
    }
    manager.semantic_vision = SimpleNamespace(fetch_frame=object())
    stale = {
        "active": True, "matched": True, "quality": 0.99, "threshold": 0.80,
        "bbox": dict(LOW), "source_frame_stamp_ns": 800,
    }
    monkeypatch.setattr(
        manager, "_confirm_marvin_local_tracker_frames",
        lambda *_a, **_k: {"source_timestamp": "2026-10-04T00:00:01+00:00",
                           "opencv_tracker": stale},
    )
    assert manager.mark_strict_v2_action_dispatched(800, "forward") is True
    result = manager._continue_strict_v2_tracker_after_action()
    assert result["found"] is False
    assert manager._marvin_v2_tracker_episode is None


def test_concurrent_post_action_observation_fails_closed_without_discarding_episode():
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = {
        "post_action_pending": True,
        "post_action_in_progress": True,
        "post_action_source_frame_stamp_ns": 900,
        "marvin_tracker": object(),
    }
    result = manager._continue_strict_v2_tracker_after_action()
    assert result["found"] is False
    assert result["reason"] == "post_action_tracker_continuity_in_progress"
    assert manager._marvin_v2_tracker_episode["post_action_pending"] is True
