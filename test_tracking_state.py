#!/usr/bin/env python3

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


def main():
    test_centering_left()
    test_centering_right()
    test_centered_target()
    test_stop_clears_tracking()
    test_nonvisual_behavior_preserves_tracking()

    print("PASS: runtime tracking state tests")


if __name__ == "__main__":
    main()
