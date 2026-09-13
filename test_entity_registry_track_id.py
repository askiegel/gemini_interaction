from types import SimpleNamespace
import pytest

from entity_registry import EntityRegistry
from world_model import WorldModel


def _entity(attributes):
    return SimpleNamespace(
        attributes=attributes,
        history=[],
    )


def test_track_id_does_not_change_generic_match_score():
    registry = EntityRegistry(WorldModel("/tmp/entity_registry_score.json"))
    entity = _entity({"track_id": 7, "raw_detection": {"track_id": 7}})
    location = {"frame": "camera", "cx": 10, "cy": 10}

    with_id = registry._score_match(
        entity,
        location,
        {"track_id": 42, "raw_detection": {"track_id": 42}},
    )
    without_id = registry._score_match(entity, location, {})

    assert with_id == without_id == 0.4


def test_equal_track_id_alone_cannot_match_unrelated_objects(tmp_path):
    registry = EntityRegistry(WorldModel(str(tmp_path / "state.json")))
    first = registry.register_observation(
        label="backpack",
        location={"frame": "camera", "cx": 10, "cy": 10},
        attributes={"track_id": 99, "raw_detection": {"track_id": 99}},
    )
    second = registry.register_observation(
        label="backpack",
        location={"frame": "camera", "cx": 500, "cy": 400},
        attributes={"track_id": 99, "raw_detection": {"track_id": 99}},
    )

    assert first != second


def test_detector_metadata_remains_stored(tmp_path):
    world_model = WorldModel(str(tmp_path / "state.json"))
    registry = EntityRegistry(world_model)
    raw_detection = {"label": "backpack", "track_id": 12}
    entity_id = registry.register_observation(
        label="backpack",
        location={"frame": "camera", "cx": 10, "cy": 10},
        attributes={"track_id": 12, "raw_detection": raw_detection},
    )

    entity = world_model.entities[entity_id]
    assert entity.attributes["track_id"] == 12
    assert entity.attributes["raw_detection"] == raw_detection
    assert entity.history[-1].attributes["track_id"] == 12
    assert entity.history[-1].attributes["raw_detection"] == raw_detection


def test_raw_detection_does_not_affect_semantic_score(tmp_path):
    registry = EntityRegistry(WorldModel(str(tmp_path / "state.json")))
    entity = _entity({"raw_detection": {"bbox": [1, 2, 3, 4]}})
    location = {"frame": "camera", "cx": 10, "cy": 10}

    changed_raw = registry._score_match(
        entity,
        location,
        {"raw_detection": {"bbox": [100, 200, 300, 400]}},
    )
    absent_raw = registry._score_match(entity, location, {})

    assert changed_raw == absent_raw == 0.4


def test_person_identity_id_matching_remains_semantic(tmp_path):
    registry = EntityRegistry(WorldModel(str(tmp_path / "state.json")))
    entity = _entity({"identity_id": "person-alpha"})
    location = {"frame": "camera", "cx": 10, "cy": 10}

    score = registry._score_match(
        entity,
        location,
        {"identity_id": "person-alpha"},
    )

    assert score == pytest.approx(0.6)
