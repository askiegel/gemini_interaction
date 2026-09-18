#!/usr/bin/env python3

import math

from tracking_state import (
    build_tracking_state,
    empty_tracking_state,
)


def test_centering_left():
    result = {
        "ok": True,
        "executed": True,
        "completed": False,
        "behavior": "FOLLOW_PERSON",
        "state": "CENTERING_LEFT",
        "target": "person",
        "cx": 182,
        "image_width": 640,
        "area": 47290,
        "confidence": 0.91,
        "center_tolerance_pixels": 95,
        "vision_timestamp": (
            "2026-07-15T23:00:00Z"
        ),
    }

    tracking = build_tracking_state(
        result
    )

    assert tracking["active"] is True
    assert tracking["behavior"] == "FOLLOW_PERSON"
    assert tracking["state"] == "CENTERING_LEFT"
    assert tracking["target_label"] == "person"
    assert tracking["target_center_x"] == 182.0
    assert tracking["image_center_x"] == 320.0
    assert tracking["horizontal_error"] == -138.0
    assert tracking["target_area"] == 47290.0
    assert tracking["steering_direction"] == "LEFT"


def test_centering_right():
    result = {
        "behavior": "FIND_OBJECT",
        "state": "CENTERING_RIGHT",
        "target_label": "backpack",
        "target_center_x": 500,
        "image_width": 640,
        "target_area": 22000,
    }

    tracking = build_tracking_state(
        result
    )

    assert tracking["horizontal_error"] == 180.0
    assert tracking["steering_direction"] == "RIGHT"


def test_centered_target():
    result = {
        "behavior": "FOLLOW_PERSON",
        "state": "APPROACHING",
        "target": {
            "label": "person",
            "center_x": 326,
            "area": 30000,
            "confidence": 0.88,
            "image_width": 640,
        },
        "center_tolerance_pixels": 95,
    }

    tracking = build_tracking_state(
        result
    )

    assert tracking["horizontal_error"] == 6.0
    assert tracking["steering_direction"] == "CENTER"
    assert tracking["distance_state"] == "TOO_FAR"


def test_target_reached_maps_to_existing_at_distance_state():
    tracking = build_tracking_state({
        "behavior": "FIND_OBJECT",
        "state": "TARGET_REACHED",
        "target": "backpack",
        "target_area": 90000,
    })
    assert tracking["distance_state"] == "AT_DISTANCE"

    arrived = build_tracking_state({
        "behavior": "FIND_OBJECT",
        "state": "ARRIVED",
        "target": "backpack",
    })
    assert arrived["distance_state"] == "AT_DISTANCE"


def test_stop_clears_tracking():
    previous = {
        **empty_tracking_state(),
        "active": True,
        "state": "APPROACHING",
        "target_label": "person",
    }

    tracking = build_tracking_state(
        {
            "behavior": "STOP",
            "state": "STOPPED",
        },
        previous=previous,
    )

    assert tracking["active"] is False
    assert tracking["state"] == "STOPPED"
    assert tracking["target_label"] is None


def test_nonvisual_behavior_preserves_tracking():
    previous = {
        **empty_tracking_state(),
        "active": True,
        "behavior": "FOLLOW_PERSON",
        "state": "CENTERING_LEFT",
        "target_label": "person",
    }

    tracking = build_tracking_state(
        {
            "behavior": "MOVE_FORWARD",
            "state": "MOVING",
        },
        previous=previous,
    )

    assert tracking == previous


def test_find_object_bbox_precedence_and_fallbacks():
    top_level = {
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_observation": {
            "bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
            "location": {"bbox": {"x1": 5, "y1": 6, "x2": 35, "y2": 45}},
        },
    }
    assert build_tracking_state(top_level)["bbox"] == {
        "x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0,
    }

    location = {
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_observation": {
            "bbox": None,
            "location": {"bbox": {"x1": 5, "y1": 6, "x2": 35, "y2": 45}},
        },
    }
    assert build_tracking_state(location)["bbox"]["x1"] == 5.0

    raw = {
        "behavior": "FIND_OBJECT",
        "state": "CENTERED",
        "target_observation": {
            "bbox": None,
            "location": {},
            "attributes": {
                "raw_detection": {"x1": 10, "y1": 11, "x2": 50, "y2": 60},
            },
        },
    }
    assert build_tracking_state(raw)["bbox"]["x2"] == 50.0


def test_find_object_invalid_bbox_does_not_propagate():
    result = {
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_observation": {
            "bbox": {"x1": 1, "y1": 2, "x2": math.inf, "y2": 40},
            "location": {},
        },
    }
    assert build_tracking_state(result)["bbox"] is None


def test_find_object_failure_retains_previous_valid_bbox():
    previous = {
        **empty_tracking_state(),
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 120.0},
    }
    result = {
        "behavior": "FIND_OBJECT",
        "state": "TARGET_RECONFIRMATION_FAILED",
        "target_observation": {"bbox": None},
    }
    tracking = build_tracking_state(result, previous=previous)
    assert tracking["bbox"] == previous["bbox"]


def main():
    test_centering_left()
    test_centering_right()
    test_centered_target()
    test_target_reached_maps_to_existing_at_distance_state()
    test_stop_clears_tracking()
    test_nonvisual_behavior_preserves_tracking()
    test_find_object_bbox_precedence_and_fallbacks()
    test_find_object_invalid_bbox_does_not_propagate()
    test_find_object_failure_retains_previous_valid_bbox()

    print("PASS: runtime tracking state tests")


if __name__ == "__main__":
    main()
