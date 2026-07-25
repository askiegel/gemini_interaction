#!/usr/bin/env python3

import tempfile
from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


class DuplicatePersonVisionAdapter(VisionAdapter):
    def fetch_vision_payload(self):
        return {
            "timestamp": "2026-07-25T22:30:00Z",
            "camera_running": True,
            "image_width": 640,
            "image_height": 480,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.94,
                    "bbox": {
                        "x1": 220,
                        "y1": 80,
                        "x2": 380,
                        "y2": 430,
                    },
                },
                {
                    "label": "person",
                    "confidence": 0.72,
                    "bbox": {
                        "x1": 225,
                        "y1": 82,
                        "x2": 382,
                        "y2": 428,
                    },
                },
            ],
        }


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        world_model = WorldModel(
            storage_path=str(
                Path(temp_dir)
                / "world_model_state.json"
            )
        )

        adapter = DuplicatePersonVisionAdapter(
            world_model
        )

        processed = (
            adapter.process_detection_frame(
                adapter.fetch_detections()
            )
        )

        people = [
            detection
            for detection in processed
            if detection.get("label") == "person"
        ]

        identities = (
            adapter.identity_manager.get_identities()
        )

        print("processed people:", people)
        print("identities:", identities)

        assert len(people) == 1
        assert people[0]["entity_id"] == "person-001"
        assert people[0]["confidence"] == 0.94

        assert len(identities) == 1
        assert (
            identities[0]["transient_entity_id"]
            == "person-001"
        )

        assert (
            people[0]["identity_id"]
            == identities[0]["identity_id"]
        )

        assert (
            people[0]["identity_status"]
            != "NEW_FRAME_CONFLICT"
        )

        print()
        print(
            "PASS: Duplicate YOLO person boxes mapped "
            "to one World Model entity."
        )
        print(
            "PASS: One entity received exactly one "
            "persistent identity per frame."
        )


if __name__ == "__main__":
    main()
