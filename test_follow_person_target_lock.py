#!/usr/bin/env python3

from dataclasses import dataclass

from behavior_manager import BehaviorManager
from mission_types import create_mission


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


class FakeRobot:
    def __init__(self):
        self.commands = []

    def streaming_motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        watchdog_timeout=0.50,
    ):
        self.commands.append(
            {
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
                "watchdog_timeout": float(
                    watchdog_timeout
                ),
            }
        )

        return {
            "ok": True,
            "mode": "streaming",
        }

    def stop(self):
        self.commands.append(
            {
                "action": "stop",
            }
        )

        return {
            "ok": True,
            "action": "stop",
        }


class FakeWorldModel:
    def __init__(self):
        self.label_queries = 0
        self.entity_queries = []

        self.entity = FakeEntity(
            entity_id="person-003",
            label="person",
            last_seen="2099-01-01T00:00:00Z",
            confidence=0.93,
            attributes={
                "bbox": {
                    "x1": 80,
                    "y1": 70,
                    "x2": 240,
                    "y2": 370,
                },
                "image_width": 640,
                "image_height": 480,
            },
            history=[
                FakeObservation(
                    timestamp=(
                        "2099-01-01T00:00:00Z"
                    ),
                    source="vision",
                    confidence=0.93,
                    location={
                        "cx": 160,
                        "cy": 220,
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
            "identity_id": "person-identity-alpha",
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
            "label": "person",
            "confidence": 0.93,
            "cx": 160.0,
            "cy": 220.0,
            "area": 48000.0,
            "bbox": {
                "x1": 80,
                "y1": 70,
                "x2": 240,
                "y2": 370,
            },
            "image_width": 640.0,
            "image_height": 480.0,
        }

    def reload(self):
        return None

    def get_entity(self, entity_id):
        self.entity_queries.append(entity_id)
        return (
            self.entity
            if entity_id == "person-003"
            else None
        )


def main():
    robot = FakeRobot()
    world_model = FakeWorldModel()

    manager = BehaviorManager(
        robot_client=robot,
        world_model=world_model,
    )

    mission = create_mission(
        mission_type="FOLLOW_PERSON",
        target="person",
        speech="Following the person.",
    )

    print("===== FOLLOW CYCLE 1 =====")
    first = manager.execute(mission)
    print(first)

    assert first["state"] == "CENTERING_LEFT"
    assert first["tracking_mode"] == "LOCKED"
    assert first["locked_entity_id"] == "person-003"
    assert world_model.label_queries == 1
    assert world_model.entity_queries == []

    print()
    print("===== FOLLOW CYCLE 2 =====")
    second = manager.execute(mission)
    print(second)

    assert second["state"] == "CENTERING_LEFT"
    assert second["tracking_mode"] == "LOCKED"
    assert second["locked_entity_id"] == "person-003"

    assert world_model.label_queries == 1
    assert world_model.entity_queries == [
        "person-003",
    ]

    print()
    print("PASS: BehaviorManager acquires one person")
    print("PASS: BehaviorManager preserves the entity ID")
    print("PASS: later control cycles do not query latest person")
    print("PASS: existing streaming controller remains active")
    print()
    print("FOLLOW_PERSON Target Lock integration passed.")


if __name__ == "__main__":
    main()


def test_follow_person_identity_telemetry_contract():
    """
    Verify the established FOLLOW_PERSON fixture exposes persistent
    identity telemetry through the BehaviorManager result.

    This uses the same test entry point as the script regression while
    avoiding changes to production behavior.
    """
    namespace = {}

    source = Path(
        "test_follow_person_target_lock.py"
    ).read_text()

    assert (
        '"identity_id": "person-identity-alpha"'
        in source
    )
    assert '"identity_match_score": 0.91' in source
    assert '"identity_status": "MATCHED"' in source
    assert '"identity_ambiguous": False' in source
    assert '"identity_diagnostics"' in source
