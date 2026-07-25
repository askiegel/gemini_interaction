#!/usr/bin/env python3

import tempfile
from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


class LivePayloadVisionAdapter(VisionAdapter):
    """
    Reproduce the real YOLO contract: detections do not arrive with a
    World Model entity_id.
    """

    def fetch_vision_payload(self):
        return {
            "timestamp": "2026-07-25T22:00:00Z",
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
            ],
        }


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = (
            Path(temp_dir)
            / "world_model_state.json"
        )

        world_model = WorldModel(
            storage_path=str(state_file)
        )

        adapter = LivePayloadVisionAdapter(
            world_model=world_model
        )

        print(
            "===== PROCESS REALISTIC YOLO FRAME ====="
        )

        entity_ids = adapter.process_once()

        print("entity_ids:", entity_ids)

        assert entity_ids == ["person-001"]

        entity = world_model.get_entity(
            "person-001"
        )

        assert entity is not None
        assert len(entity.history) == 1

        identities = (
            adapter.identity_manager.get_identities()
        )

        assert len(identities) == 1

        identity = identities[0]

        print("identity:", identity)
        print("entity attributes:", entity.attributes)

        assert entity.attributes.get("identity_id")

        # This is the live integration requirement. It currently fails because
        # identity assignment runs before EntityRegistry returns person-001.
        assert (
            identity["transient_entity_id"]
            == "person-001"
        ), (
            "PersonIdentityManager did not receive the "
            "Entity Registry ID before matching."
        )

        assert (
            entity.attributes["identity_id"]
            == identity["identity_id"]
        )

        assert len(entity.history) == 1

        print()
        print(
            "PASS: Registry entity ID reached "
            "PersonIdentityManager before matching."
        )
        print(
            "PASS: World Model stored exactly one "
            "observation for the frame."
        )


if __name__ == "__main__":
    main()
