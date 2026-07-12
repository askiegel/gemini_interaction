#!/usr/bin/env python3

import os
import shutil
import tempfile

from world_model import WorldModel


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="world_model_target_query_"
    )

    storage_path = os.path.join(
        temporary_directory,
        "world_model_state.json",
    )

    try:
        print("===== CREATE INDEPENDENT READ INSTANCE =====")

        reader = WorldModel(storage_path)
        writer = WorldModel(storage_path)

        print("Reader initialized before vision observation.")

        print()
        print("===== WRITE VISION OBSERVATION =====")

        writer.update_entity(
            entity_id="backpack-query-test",
            label="backpack",
            entity_type="object",
            confidence=0.91,
            source="vision_service_test",
            location={
                "cx": 438,
                "cy": 265,
            },
            attributes={
                "targetable": True,
                "raw_detection": {
                    "label": "backpack",
                    "confidence": 0.91,
                    "bbox": [
                        388,
                        190,
                        488,
                        340,
                    ],
                    "image_width": 640,
                    "image_height": 480,
                },
            },
        )

        print("Vision observation stored by writer instance.")

        print()
        print("===== QUERY THROUGH SECOND INSTANCE =====")

        result = reader.find_latest_entity_by_label(
            "back pack",
            max_age_seconds=5.0,
            refresh=True,
        )

        print(result)

        assert result["found"] is True
        assert result["stale"] is False
        assert (
            result["entity_id"]
            == "backpack-query-test"
        )
        assert result["target"] == "backpack"
        assert result["confidence"] == 0.91
        assert result["cx"] == 438.0
        assert result["cy"] == 265.0
        assert result["area"] == 15000.0
        assert result["image_width"] == 640.0
        assert result["image_height"] == 480.0

        print()
        print("===== MISSING TARGET QUERY =====")

        missing = reader.find_latest_entity_by_label(
            "bottle",
            max_age_seconds=5.0,
            refresh=True,
        )

        print(missing)

        assert missing["found"] is False
        assert missing["stale"] is False
        assert missing["target"] == "bottle"

        print()
        print("===== STALE TARGET PROTECTION =====")

        writer.reload()

        stale_entity = writer.get_entity(
            "backpack-query-test"
        )

        stale_entity.last_seen = (
            "2000-01-01T00:00:00Z"
        )

        writer.save()

        stale = reader.find_latest_entity_by_label(
            "backpack",
            max_age_seconds=1.0,
            refresh=True,
        )

        print(stale)

        assert stale["found"] is False
        assert stale["stale"] is True
        assert stale["entity_id"] == (
            "backpack-query-test"
        )
        assert stale["age_seconds"] > 1.0

        print()
        print(
            "PASS: independent reader reloads shared state"
        )
        print(
            "PASS: spoken target aliases are normalized"
        )
        print(
            "PASS: target geometry is reconstructed"
        )
        print(
            "PASS: missing targets return safely"
        )
        print(
            "PASS: stale observations cannot drive motion"
        )
        print()
        print(
            "World Model target query test passed."
        )

    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
