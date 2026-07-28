#!/usr/bin/env python3

import tempfile
from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


def location(cx, cy, area, x1, y1, x2, y2):
    return {
        "frame": "camera",
        "cx": float(cx),
        "cy": float(cy),
        "area": float(area),
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
        "image_width": 640.0,
        "image_height": 480.0,
    }


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        world = WorldModel(
            storage_path=str(
                Path(temp_dir)
                / "world_model_state.json"
            )
        )

        # Reproduce the live polluted registry: two historical entities occupy
        # the locations selected by the large and nested YOLO person boxes.
        world.update_entity(
            entity_id="person-001",
            label="person",
            entity_type="human",
            confidence=0.90,
            source="test",
            location=location(
                360, 349, 34713,
                294, 218, 427, 479,
            ),
            attributes={"targetable": True},
        )

        world.update_entity(
            entity_id="person-002",
            label="person",
            entity_type="human",
            confidence=0.80,
            source="test",
            location=location(
                390, 281, 9525,
                352, 217, 427, 345,
            ),
            attributes={"targetable": True},
        )

        adapter = VisionAdapter(world)

        before_counts = {
            entity_id: len(entity.history)
            for entity_id, entity
            in world.entities.items()
        }

        detections = [
            {
                "label": "person",
                "confidence": 0.568,
                "x1": 294,
                "y1": 218,
                "x2": 427,
                "y2": 479,
                "center_x": 360,
                "center_y": 349,
                "area": 34713,
                "image_width": 640,
                "image_height": 480,
            },
            {
                "label": "person",
                "confidence": 0.466,
                "x1": 352,
                "y1": 217,
                "x2": 427,
                "y2": 345,
                "center_x": 390,
                "center_y": 281,
                "area": 9525,
                "image_width": 640,
                "image_height": 480,
            },
        ]

        processed = adapter.process_detection_frame(
            detections
        )

        people = [
            detection
            for detection in processed
            if detection.get("label") == "person"
        ]

        updated_entities = [
            entity_id
            for entity_id, entity
            in world.entities.items()
            if (
                len(entity.history)
                > before_counts.get(entity_id, 0)
            )
        ]

        identities = (
            adapter.identity_manager.get_identities()
        )

        print("processed_people:", people)
        print("updated_entities:", updated_entities)
        print("identities:", identities)

        assert len(people) == 1, (
            "Nested boxes reached identity assignment separately."
        )

        assert updated_entities == ["person-001"], (
            "Nested box updated a second historical registry entity."
        )

        assert len(identities) == 1, (
            "Nested boxes received separate persistent identities."
        )

        assert people[0]["confidence"] == 0.568
        assert people[0]["area"] == 34713

        print()
        print(
            "PASS: Nested boxes were collapsed before registry matching."
        )


if __name__ == "__main__":
    main()
