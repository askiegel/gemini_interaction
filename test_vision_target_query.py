#!/usr/bin/env python3

import tempfile
from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


class OfflineVisionAdapter(VisionAdapter):
    def __init__(self, world_model, detections, payload=None):
        super().__init__(
            world_model=world_model,
            vision_url="http://offline.invalid/detections/latest",
        )
        self._offline_detections = detections
        self._offline_payload = payload or {}

    def fetch_detections(self):
        self.last_payload = {
            "vision_status": "DETECTIONS_AVAILABLE",
            "image_width": 640,
            "image_height": 480,
            "detections": self._offline_detections,
            **self._offline_payload,
        }

        self.update_vision_health(self.last_payload)
        return self._offline_detections


def assert_close(actual, expected, tolerance=0.001):
    assert abs(actual - expected) <= tolerance, (
        f"Expected {expected}, got {actual}"
    )


def test_backpack_found_from_bbox():
    with tempfile.TemporaryDirectory() as directory:
        world = WorldModel(
            storage_path=str(Path(directory) / "world.json")
        )

        adapter = OfflineVisionAdapter(
            world_model=world,
            detections=[
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox": [40, 30, 200, 450],
                },
                {
                    "label": "backpack",
                    "confidence": 0.88,
                    "bbox": [300, 120, 500, 420],
                },
            ],
        )

        result = adapter.find_target("backpack")

        assert result["found"] is True
        assert result["target"] == "backpack"
        assert result["label"] == "backpack"
        assert_close(result["confidence"], 0.88)
        assert_close(result["cx"], 400.0)
        assert_close(result["cy"], 270.0)
        assert_close(result["area"], 60000.0)
        assert_close(result["image_width"], 640.0)
        assert_close(result["image_height"], 480.0)

        labels = [entity["label"] for entity in world.get_entities()]
        assert "person" in labels
        assert "backpack" in labels


def test_backpack_alias():
    with tempfile.TemporaryDirectory() as directory:
        world = WorldModel(
            storage_path=str(Path(directory) / "world.json")
        )

        adapter = OfflineVisionAdapter(
            world_model=world,
            detections=[
                {
                    "name": "back pack",
                    "conf": 0.74,
                    "cx": 318,
                    "cy": 244,
                    "area": 42000,
                }
            ],
        )

        result = adapter.find_target("backpack")

        assert result["found"] is True
        assert result["label"] == "backpack"
        assert_close(result["cx"], 318.0)
        assert_close(result["area"], 42000.0)


def test_best_matching_detection():
    with tempfile.TemporaryDirectory() as directory:
        world = WorldModel(
            storage_path=str(Path(directory) / "world.json")
        )

        adapter = OfflineVisionAdapter(
            world_model=world,
            detections=[
                {
                    "label": "backpack",
                    "confidence": 0.55,
                    "bbox": [10, 10, 110, 110],
                },
                {
                    "label": "backpack",
                    "confidence": 0.93,
                    "bbox": [200, 100, 500, 450],
                },
            ],
        )

        result = adapter.find_target("backpack")

        assert result["found"] is True
        assert_close(result["confidence"], 0.93)
        assert_close(result["cx"], 350.0)
        assert_close(result["area"], 105000.0)


def test_target_not_found():
    with tempfile.TemporaryDirectory() as directory:
        world = WorldModel(
            storage_path=str(Path(directory) / "world.json")
        )

        adapter = OfflineVisionAdapter(
            world_model=world,
            detections=[
                {
                    "label": "chair",
                    "confidence": 0.82,
                    "bbox": [100, 100, 300, 400],
                }
            ],
        )

        result = adapter.find_target("backpack")

        assert result == {
            "found": False,
            "target": "backpack",
            "vision_status": "DETECTIONS_AVAILABLE",
        }


if __name__ == "__main__":
    tests = [
        test_backpack_found_from_bbox,
        test_backpack_alias,
        test_best_matching_detection,
        test_target_not_found,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print()
    print("All Vision Target Query tests passed.")
    print("No commands were sent to the physical robot.")
