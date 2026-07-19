#!/usr/bin/env python3

import tempfile
from pathlib import Path

from vision_adapter import VisionAdapter
from world_model import WorldModel


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = (
            Path(temp_dir)
            / "world_model_identity_test.json"
        )

        world_model = WorldModel(
            storage_path=str(storage_path)
        )

        adapter = VisionAdapter(
            world_model=world_model
        )

        detection = {
            "label": "person",
            "confidence": 0.94,
            "x1": 100,
            "y1": 80,
            "x2": 260,
            "y2": 380,
            "center_x": 180,
            "center_y": 230,
            "area": 48000,
            "image_width": 640,
            "image_height": 480,
            "identity_id": (
                "person-identity-alpha"
            ),
            "identity_match_score": 0.91,
            "identity_status": "MATCHED",
            "identity_ambiguous": False,
            "identity_diagnostics": {
                "best_score": 0.91,
                "second_score": 0.37,
                "score_margin": 0.54,
                "ambiguous": False,
                "decision": "MATCHED",
            },
            "attributes": {
                "appearance_source": "test",
            },
        }

        print(
            "===== PROCESS IDENTITY DETECTION ====="
        )

        entity_id = adapter.process_detection(
            detection
        )

        print("entity_id:", entity_id)

        assert entity_id

        entity = world_model.get_entity(
            entity_id
        )

        assert entity is not None

        print()
        print("===== ENTITY ATTRIBUTES =====")
        print(entity.attributes)

        assert (
            entity.attributes["identity_id"]
            == "person-identity-alpha"
        )
        assert (
            entity.attributes[
                "identity_match_score"
            ]
            == 0.91
        )
        assert (
            entity.attributes["identity_status"]
            == "MATCHED"
        )
        assert (
            entity.attributes[
                "identity_ambiguous"
            ]
            is False
        )

        diagnostics = entity.attributes[
            "identity_diagnostics"
        ]

        assert diagnostics["best_score"] == 0.91
        assert diagnostics["second_score"] == 0.37
        assert diagnostics["score_margin"] == 0.54
        assert diagnostics["decision"] == "MATCHED"

        assert (
            entity.attributes[
                "appearance_source"
            ]
            == "test"
        )

        latest = entity.history[-1]

        print()
        print(
            "===== LATEST OBSERVATION ATTRIBUTES ====="
        )
        print(latest.attributes)

        assert (
            latest.attributes["identity_id"]
            == "person-identity-alpha"
        )
        assert (
            latest.attributes[
                "identity_match_score"
            ]
            == 0.91
        )
        assert (
            latest.attributes[
                "identity_diagnostics"
            ]["decision"]
            == "MATCHED"
        )

        world_model.save()

        reloaded = WorldModel(
            storage_path=str(storage_path)
        )

        persisted = reloaded.get_entity(
            entity_id
        )

        print()
        print(
            "===== RELOADED ENTITY ATTRIBUTES ====="
        )
        print(persisted.attributes)

        assert (
            persisted.attributes["identity_id"]
            == "person-identity-alpha"
        )
        assert (
            persisted.attributes[
                "identity_diagnostics"
            ]["score_margin"]
            == 0.54
        )

        normalized = adapter.normalize_detection(
            detection
        )

        print()
        print(
            "===== NORMALIZED TARGET RESULT ====="
        )
        print(normalized)

        assert (
            normalized["identity_id"]
            == "person-identity-alpha"
        )
        assert (
            normalized[
                "identity_diagnostics"
            ]["decision"]
            == "MATCHED"
        )

        print()
        print(
            "PASS: VisionAdapter stores identity "
            "on the World Model entity"
        )
        print(
            "PASS: observation history stores "
            "identity telemetry"
        )
        print(
            "PASS: persistent reload preserves "
            "identity telemetry"
        )
        print(
            "PASS: normalized target includes "
            "identity diagnostics"
        )
        print()
        print(
            "Vision identity World Model test passed."
        )


if __name__ == "__main__":
    main()
