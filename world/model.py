from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os


WORLD_MODEL_FILE = "world_model_state.json"


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class EntityObservation:
    timestamp: str
    source: str
    confidence: float
    location: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldEntity:
    entity_id: str
    label: str
    entity_type: str
    first_seen: str
    last_seen: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    history: List[EntityObservation] = field(default_factory=list)

    def to_dict(self):
        data = asdict(self)
        data["history"] = [asdict(h) for h in self.history]
        return data

    @staticmethod
    def from_dict(data):
        entity = WorldEntity(
            entity_id=data["entity_id"],
            label=data["label"],
            entity_type=data.get("entity_type", "unknown"),
            first_seen=data["first_seen"],
            last_seen=data["last_seen"],
            confidence=data.get("confidence", 0.0),
            attributes=data.get("attributes", {}),
            history=[]
        )

        for item in data.get("history", []):
            entity.history.append(EntityObservation(**item))

        return entity


class WorldModel:
    def __init__(self, storage_path: str = WORLD_MODEL_FILE):
        self.storage_path = storage_path

        self.robot_state = {
            "battery": None,
            "mission": None,
            "navigation_state": "UNKNOWN",
            "updated_at": now_iso()
        }

        self.environment = {}
        self.entities: Dict[str, WorldEntity] = {}
        self.recent_events: List[Dict[str, Any]] = []

        self.load()

    def set_robot_state(self, key: str, value: Any):
        self.robot_state[key] = value
        self.robot_state["updated_at"] = now_iso()
        self.add_event("robot_state_updated", {"key": key, "value": value})
        self.save()

    def add_event(self, event_type: str, data: Dict[str, Any]):
        event = {
            "timestamp": now_iso(),
            "type": event_type,
            "data": data
        }

        self.recent_events.append(event)
        self.recent_events = self.recent_events[-100:]

    def update_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str = "object",
        confidence: float = 0.0,
        source: str = "unknown",
        location: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        attributes = attributes or {}
        timestamp = now_iso()

        observation = EntityObservation(
            timestamp=timestamp,
            source=source,
            confidence=confidence,
            location=location,
            attributes=attributes
        )

        if entity_id not in self.entities:
            self.entities[entity_id] = WorldEntity(
                entity_id=entity_id,
                label=label,
                entity_type=entity_type,
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=confidence,
                attributes=attributes,
                history=[observation]
            )

            self.add_event("entity_created", {
                "entity_id": entity_id,
                "label": label,
                "entity_type": entity_type
            })

        else:
            entity = self.entities[entity_id]
            entity.label = label
            entity.entity_type = entity_type
            entity.last_seen = timestamp
            entity.confidence = confidence
            entity.attributes.update(attributes)
            entity.history.append(observation)
            entity.history = entity.history[-50:]

            self.add_event("entity_updated", {
                "entity_id": entity_id,
                "label": label,
                "confidence": confidence
            })

        self.save()

    def get_entity(self, entity_id: str) -> Optional[WorldEntity]:
        return self.entities.get(entity_id)

    def get_entities(self) -> List[Dict[str, Any]]:
        return [entity.to_dict() for entity in self.entities.values()]

    def get_recent_events(self) -> List[Dict[str, Any]]:
        return self.recent_events

    def get_context(self) -> Dict[str, Any]:
        return {
            "robot_state": self.robot_state,
            "environment": self.environment,
            "entities": self.get_entities(),
            "recent_events": self.recent_events
        }

    def update_robot_state(self, **updates):
        """
        Compatibility interface for updating multiple robot-state fields.

        The World Model remains the single source of truth. Each call stores
        all supplied values and persists the resulting state.
        """
        if not updates:
            return self.robot_state

        self.robot_state.update(updates)
        self.robot_state["updated_at"] = now_iso()

        self.add_event(
            "robot_state_updated",
            {"updates": updates},
        )

        self.save()
        return self.robot_state

    def update_from_detections(self, detections, source="vision"):
        """
        Compatibility interface for importing normalized vision detections.

        Each detection is converted into a WorldEntity observation using the
        existing update_entity() implementation.
        """
        updated_entity_ids = []

        for index, detection in enumerate(detections):
            label = str(detection.get("label", "unknown")).strip().lower()
            entity_id = detection.get("entity_id")

            if not entity_id:
                entity_id = f"{label}-{index + 1:03d}"

            entity_type = detection.get("entity_type")
            if not entity_type:
                entity_type = "human" if label == "person" else "object"

            location = detection.get("location")

            if location is None:
                location = {
                    key: detection[key]
                    for key in (
                        "cx",
                        "cy",
                        "center_x",
                        "center_y",
                        "distance_m",
                        "position",
                    )
                    if key in detection
                }

            attributes = dict(detection.get("attributes", {}))

            for key in (
                "distance_m",
                "position",
                "tracking",
                "bbox",
                "reid",
                "image_width",
                "image_height",
            ):
                if key in detection:
                    attributes[key] = detection[key]

            self.update_entity(
                entity_id=entity_id,
                label=label,
                entity_type=entity_type,
                confidence=float(detection.get("confidence", 0.0)),
                source=source,
                location=location or None,
                attributes=attributes,
            )

            updated_entity_ids.append(entity_id)

        return updated_entity_ids

    def snapshot(self) -> Dict[str, Any]:
        """
        Return the compatibility snapshot consumed by the cognitive interface.
        """
        robot = dict(self.robot_state)

        if "battery_percent" not in robot:
            robot["battery_percent"] = robot.get("battery")

        robot.setdefault("name", "Mini Pupper 2")
        robot.setdefault("mission", None)
        robot.setdefault("navigation_state", "UNKNOWN")
        robot.setdefault("front_clear", None)
        robot.setdefault("nearest_obstacle_m", None)

        entities = []

        for entity in self.entities.values():
            data = entity.to_dict()
            location = {}

            if entity.history:
                location = entity.history[-1].location or {}

            attributes = entity.attributes or {}

            data["distance_m"] = attributes.get(
                "distance_m",
                location.get("distance_m"),
            )
            data["position"] = attributes.get(
                "position",
                location.get("position"),
            )
            data["tracking"] = attributes.get("tracking", False)

            entities.append(data)

        return {
            "robot": robot,
            "environment": self.environment,
            "entities": entities,
            "recent_events": list(self.recent_events),
        }

    def save(self):
        data = {
            "robot_state": self.robot_state,
            "environment": self.environment,
            "entities": {
                entity_id: entity.to_dict()
                for entity_id, entity in self.entities.items()
            },
            "recent_events": self.recent_events
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.storage_path):
            return

        with open(self.storage_path, "r") as f:
            data = json.load(f)

        self.robot_state.update(data.get("robot_state", {}))
        self.environment = data.get("environment", {})
        self.recent_events = data.get("recent_events", [])

        self.entities = {
            entity_id: WorldEntity.from_dict(entity_data)
            for entity_id, entity_data in data.get("entities", {}).items()
        }
