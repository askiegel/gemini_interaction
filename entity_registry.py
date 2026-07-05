from typing import Dict, Any, Optional, List
from math import sqrt
from world_model import WorldModel


class EntityRegistry:
    """
    Converts raw observations into persistent world entities.

    This is the beginning of the robotics knowledge graph layer.
    It links repeated detections to the same entity when possible.
    """

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def register_observation(
        self,
        label: str,
        entity_type: str = "object",
        confidence: float = 0.0,
        source: str = "unknown",
        location: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        attributes = attributes or {}

        entity_id = self._match_existing_entity(
            label=label,
            entity_type=entity_type,
            location=location,
            attributes=attributes
        )

        if entity_id is None:
            entity_id = self._create_entity_id(label)

        self.world_model.update_entity(
            entity_id=entity_id,
            label=label,
            entity_type=entity_type,
            confidence=confidence,
            source=source,
            location=location,
            attributes=attributes
        )

        return entity_id

    def _match_existing_entity(
        self,
        label: str,
        entity_type: str,
        location: Optional[Dict[str, Any]],
        attributes: Dict[str, Any]
    ) -> Optional[str]:
        candidates = []

        for entity in self.world_model.entities.values():
            if entity.label != label:
                continue

            if entity.entity_type != entity_type:
                continue

            score = self._score_match(entity, location, attributes)

            if score >= 0.65:
                candidates.append((score, entity.entity_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _score_match(
        self,
        entity,
        location: Optional[Dict[str, Any]],
        attributes: Dict[str, Any]
    ) -> float:
        score = 0.4

        if entity.attributes:
            shared_keys = set(entity.attributes.keys()) & set(attributes.keys())

            for key in shared_keys:
                if entity.attributes.get(key) == attributes.get(key):
                    score += 0.2

        if location and entity.history:
            last_location = entity.history[-1].location

            if last_location:
                distance_score = self._location_similarity(last_location, location)
                score += distance_score * 0.4

        return min(score, 1.0)

    def _location_similarity(
        self,
        old_location: Dict[str, Any],
        new_location: Dict[str, Any]
    ) -> float:
        if old_location.get("frame") != new_location.get("frame"):
            return 0.0

        if "cx" not in old_location or "cx" not in new_location:
            return 0.0

        if "cy" not in old_location or "cy" not in new_location:
            return 0.0

        dx = old_location["cx"] - new_location["cx"]
        dy = old_location["cy"] - new_location["cy"]
        distance = sqrt(dx * dx + dy * dy)

        if distance < 50:
            return 1.0
        if distance < 150:
            return 0.6
        if distance < 300:
            return 0.3

        return 0.0

    def _create_entity_id(self, label: str) -> str:
        existing = [
            entity_id for entity_id in self.world_model.entities.keys()
            if entity_id.startswith(f"{label}-")
        ]

        next_number = len(existing) + 1
        return f"{label}-{next_number:03d}"

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        return [
            entity.to_dict()
            for entity in self.world_model.entities.values()
            if entity.entity_type == entity_type
        ]

    def get_entities_by_label(self, label: str) -> List[Dict[str, Any]]:
        return [
            entity.to_dict()
            for entity in self.world_model.entities.values()
            if entity.label == label
        ]
