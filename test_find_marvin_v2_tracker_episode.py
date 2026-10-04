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
