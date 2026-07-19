#!/usr/bin/env python3

from dataclasses import dataclass

from target_lock import TargetLock


@dataclass
class FakeObservation:
    timestamp: str
    source: str
    confidence: float
    location: dict
    attributes: dict


@dataclass
class FakeEntity:
    entity_id: str
    label: str
    last_seen: str
    confidence: float
    attributes: dict
    history: list


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.entity_queries = []

        self.entity = FakeEntity(
            entity_id="person-003",
            label="person",
            last_seen="2099-01-01T00:00:00Z",
            confidence=0.94,
            attributes={
                "bbox": {
                    "x1": 100,
                    "y1": 80,
                    "x2": 260,
                    "y2": 380,
                },
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
            },
            history=[
                FakeObservation(
                    timestamp=(
                        "2099-01-01T00:00:00Z"
                    ),
                    source="vision",
                    confidence=0.94,
                    location={
                        "cx": 180,
                        "cy": 230,
                    },
                    attributes={},
                )
            ],
        )

    def find_latest_entity_by_label(
        self,
        label,
        max_age_seconds=None,
        refresh=True,
    ):
        self.label_queries += 1

        return {
            "found": True,
            "stale": False,
            "target": label,
            "entity_id": "person-003",
            "label": "person",
            "confidence": 0.94,
            "cx": 180.0,
            "cy": 230.0,
            "area": 48000.0,
            "bbox": {
                "x1": 100,
                "y1": 80,
                "x2": 260,
                "y2": 380,
            },
            "image_width": 640.0,
            "image_height": 480.0,
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
        }

    def reload(self):
        return None

    def get_entity(self, entity_id):
        self.entity_queries.append(entity_id)

        if entity_id == self.entity.entity_id:
            return self.entity

        return None


def assert_identity(result):
    assert (
        result["identity_id"]
        == "person-identity-alpha"
    )
    assert result["identity_match_score"] == 0.91
    assert result["identity_status"] == "MATCHED"
    assert result["identity_ambiguous"] is False

    diagnostics = result["identity_diagnostics"]

    assert diagnostics["best_score"] == 0.91
    assert diagnostics["second_score"] == 0.37
    assert diagnostics["score_margin"] == 0.54
    assert diagnostics["decision"] == "MATCHED"


def main():
    world_model = FakeWorldModel()

    lock = TargetLock(
        world_model=world_model,
        max_age_seconds=3.0,
    )

    print("===== ACQUISITION CYCLE =====")

    first = lock.resolve(
        mission_id="mission-follow-identity",
        target_label="person",
    )

    print(first)

    assert first["found"] is True
    assert first["entity_id"] == "person-003"
    assert_identity(first)

    print()
    print("===== LOCKED ENTITY CYCLE =====")

    second = lock.resolve(
        mission_id="mission-follow-identity",
        target_label="person",
    )

    print(second)

    assert second["found"] is True
    assert second["entity_id"] == "person-003"
    assert_identity(second)

    assert world_model.label_queries == 1
    assert world_model.entity_queries == [
        "person-003",
    ]

    print()
    print(
        "PASS: acquisition includes persistent identity"
    )
    print(
        "PASS: locked entity query preserves identity"
    )
    print(
        "PASS: diagnostic scores survive later cycles"
    )
    print(
        "PASS: no second label acquisition occurred"
    )
    print()
    print(
        "TargetLock identity telemetry test passed."
    )


if __name__ == "__main__":
    main()
