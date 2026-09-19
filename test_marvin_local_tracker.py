"""Synthetic, production-dependency tests for the local Marvin tracker."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from marvin_local_tracker import MarvinLocalTracker


WIDTH = 640
HEIGHT = 480
SEED_BBOX = {"x1": 184, "y1": 84, "x2": 420, "y2": 403}


def _frame(*, left=184, top=84, width=WIDTH, height=HEIGHT, include_target=True):
    image = np.full((height, width), 18, dtype=np.uint8)
    target_width = SEED_BBOX["x2"] - SEED_BBOX["x1"]
    target_height = SEED_BBOX["y2"] - SEED_BBOX["y1"]
    if include_target:
        rows, columns = np.indices((target_height, target_width))
        pattern = ((rows * 17 + columns * 31 + (rows // 9) * 53) % 220) + 20
        image[top:top + target_height, left:left + target_width] = pattern.astype(np.uint8)
        cv2.circle(image, (left + 42, top + 51), 21, 255, 3)
        cv2.line(image, (left + 9, top + 12), (left + 211, top + 289), 5, 180)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return SimpleNamespace(data=encoded.tobytes(), width=width, height=height)


def test_valid_seed_and_identical_frame_are_tracked_deterministically():
    frame = _frame()
    first = MarvinLocalTracker(frame, SEED_BBOX)
    second = MarvinLocalTracker(frame, SEED_BBOX)

    assert first.update(frame) == SEED_BBOX
    assert second.update(frame) == SEED_BBOX
    assert first.last_quality == pytest.approx(1.0)
    assert first.last_search_roi == second.last_search_roi


@pytest.mark.parametrize("offset", [(15, 0), (0, 18)])
def test_modest_translation_is_tracked(offset):
    tracker = MarvinLocalTracker(_frame(), SEED_BBOX)
    dx, dy = offset
    result = tracker.update(_frame(left=SEED_BBOX["x1"] + dx, top=SEED_BBOX["y1"] + dy))

    assert result == {
        "x1": SEED_BBOX["x1"] + dx,
        "y1": SEED_BBOX["y1"] + dy,
        "x2": SEED_BBOX["x2"] + dx,
        "y2": SEED_BBOX["y2"] + dy,
    }
    assert 0 <= result["x1"] < result["x2"] <= WIDTH
    assert 0 <= result["y1"] < result["y2"] <= HEIGHT


@pytest.mark.parametrize(
    "bbox",
    [
        {"x1": 184, "y1": 84, "x2": 420},
        {"x1": 420, "y1": 84, "x2": 184, "y2": 403},
        {"x1": -1, "y1": 84, "x2": 420, "y2": 403},
        {"x1": 184, "y1": 84, "x2": 641, "y2": 403},
    ],
)
def test_invalid_seed_bbox_is_rejected(bbox):
    with pytest.raises(ValueError, match="bbox_invalid"):
        MarvinLocalTracker(_frame(), bbox)


def test_changed_dimensions_and_corrupt_jpeg_fail_closed():
    tracker = MarvinLocalTracker(_frame(), SEED_BBOX)

    assert tracker.update(_frame(width=320, height=240, include_target=False)) is None
    assert tracker.update(SimpleNamespace(data=b"not-a-jpeg", width=WIDTH, height=HEIGHT)) is None


def test_unrelated_image_and_poor_match_quality_report_loss():
    tracker = MarvinLocalTracker(_frame(), SEED_BBOX)

    assert tracker.update(_frame(include_target=False)) is None
    assert tracker.last_quality is not None
    assert tracker.last_quality < tracker.MIN_MATCH_QUALITY
