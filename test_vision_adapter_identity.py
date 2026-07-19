#!/usr/bin/env python3

from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


STATE_FILE = Path("test_vision_adapter_identity_state.json")


class SequencedVisionAdapter(VisionAdapter):
    def __init__(self, world_model):
        super().__init__(world_model)

        self.payloads = [
            {
                "timestamp": "2026-07-18T02:00:00Z",
                "camera_running": True,
                "image_width": 640,
                "image_height": 480,
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.94,
                        "entity_id": "person-001",
                        "bbox": {
                            "x1": 220,
                            "y1": 80,
                            "x2": 380,
                            "y2": 430,
                        },
                    },
                    {
                        "label": "backpack",
                        "confidence": 0.88,
                        "entity_id": "backpack-001",
                        "bbox": {
                            "x1": 450,
                            "y1": 200,
                            "x2": 550,
                            "y2": 340,
                        },
                    },
                ],
            },
            {
                "timestamp": "2026-07-18T02:00:01Z",
                "camera_running": True,
                "image_width": 640,
                "image_height": 480,
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.95,
                        "entity_id": "person-027",
                        "bbox": {
                            "x1": 230,
                            "y1": 82,
                            "x2": 390,
                            "y2": 432,
                        },
                    },
                    {
                        "label": "backpack",
                        "confidence": 0.89,
                        "entity_id": "backpack-001",
                        "bbox": {
                            "x1": 448,
                            "y1": 198,
                            "x2": 552,
                            "y2": 342,
                        },
                    },
                ],
            },
        ]

    def fetch_vision_payload(self):
        if not self.payloads:
            raise RuntimeError("No test payloads remain.")

        return self.payloads.pop(0)


def cleanup():
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def main():
    cleanup()

    try:
        world_model = WorldModel(str(STATE_FILE))
        adapter = SequencedVisionAdapter(world_model)

        print("===== FIRST DETECTION FRAME =====")

        first_frame = adapter.fetch_detections()
        first_person = first_frame[0]
        first_backpack = first_frame[1]

        print(first_person)
        print(first_backpack)

        first_identity_id = first_person.get("identity_id")

        assert first_identity_id
        assert first_person["identity_status"] == "NEW"
        assert first_person["identity_ambiguous"] is False
        assert first_person["image_width"] == 640
        assert first_person["image_height"] == 480

        # Non-person objects must not be modified by identity assignment.
        assert "identity_id" not in first_backpack
        assert "identity_status" not in first_backpack

        print()
        print("===== SECOND DETECTION FRAME =====")

        second_frame = adapter.fetch_detections()
        second_person = second_frame[0]
        second_backpack = second_frame[1]

        print(second_person)
        print(second_backpack)

        assert second_person["entity_id"] == "person-027"
        assert second_person["entity_id"] != first_person["entity_id"]

        assert second_person["identity_id"] == first_identity_id
        assert second_person["identity_status"] == "MATCHED"
        assert second_person["identity_match_score"] > 0.0
        assert second_person["identity_ambiguous"] is False

        assert "identity_id" not in second_backpack
        assert "identity_status" not in second_backpack

        print()
        print("===== NORMALIZED PERSON DETECTION =====")

        normalized = adapter.normalize_detection(second_person)
        print(normalized)

        assert normalized["label"] == "person"
        assert normalized["entity_id"] == "person-027"
        assert normalized["identity_id"] == first_identity_id
        assert normalized["identity_status"] == "MATCHED"
        assert normalized["identity_match_score"] > 0.0
        assert normalized["identity_ambiguous"] is False

        assert (
            normalized["attributes"]["identity_id"]
            == first_identity_id
        )

        print()
        print(
            "PASS: VisionAdapter assigns persistent identities "
            "to person detections"
        )
        print(
            "PASS: persistent identity survives transient "
            "detector-ID changes"
        )
        print(
            "PASS: non-person detections remain unchanged"
        )
        print(
            "PASS: normalize_detection preserves identity fields"
        )
        print()
        print("VisionAdapter identity integration test passed.")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
