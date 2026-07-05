from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4


def now():
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class WorldEntity:
    entity_id: str
    label: str
    entity_type: str
    confidence: Optional[float] = None
    distance_m: Optional[float] = None
    position: Optional[str] = None
    tracking: bool = False
    last_seen: Optional[str] = None
    source: str = "unknown"

    def to_dict(self):
        return asdict(self)


class WorldModel:
    def __init__(self):
        self.robot = {
            "name": "Mini Pupper 2",
            "battery_percent": 84,
            "mission": "IDLE",
            "navigation_state": "STANDBY",
            "front_clear": True,
            "nearest_obstacle_m": 1.2,
        }

        self.entities: Dict[str, WorldEntity] = {}
        self.events = []

    def update_robot_state(self, **kwargs):
        self.robot.update(kwargs)

    def update_from_detections(self, detections, source="vision"):
        for detection in detections:
            label = str(detection.get("label", "unknown")).lower()
            entity_type = self._classify_entity(label)
            entity_id = self._entity_id(label, entity_type)

            self.entities[entity_id] = WorldEntity(
                entity_id=entity_id,
                label=label,
                entity_type=entity_type,
                confidence=detection.get("confidence"),
                distance_m=detection.get("distance_m"),
                position=detection.get("position"),
                tracking=detection.get("tracking", False),
                last_seen=now(),
                source=source,
            )

    def add_event(self, event):
        self.events.append(event.to_dict())

    def snapshot(self):
        return {
            "timestamp": now(),
            "robot": self.robot,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "recent_events": self.events[-10:],
        }

    def format_for_prompt(self):
        snap = self.snapshot()
        lines = []

        robot = snap["robot"]

        lines.append("World Model")
        lines.append("-----------")
        lines.append(f"Robot: {robot.get('name')}")
        lines.append(f"Mission: {robot.get('mission')}")
        lines.append(f"Navigation: {robot.get('navigation_state')}")
        lines.append(f"Battery: {robot.get('battery_percent')}%")
        lines.append(f"Front Clear: {robot.get('front_clear')}")
        lines.append(f"Nearest Obstacle: {robot.get('nearest_obstacle_m')} m")
        lines.append("")

        lines.append("Entities:")

        if not snap["entities"]:
            lines.append("- none")
        else:
            for entity in snap["entities"]:
                lines.append(
                    f"- {entity['label']} "
                    f"type={entity['entity_type']} "
                    f"confidence={entity['confidence']} "
                    f"distance={entity['distance_m']}m "
                    f"position={entity['position']} "
                    f"tracking={entity['tracking']} "
                    f"last_seen={entity['last_seen']}"
                )

        lines.append("")
        lines.append("Recent Events:")

        if not snap["recent_events"]:
            lines.append("- none")
        else:
            for event in snap["recent_events"]:
                lines.append(
                    f"- {event.get('event_type')} "
                    f"source={event.get('source')} "
                    f"payload={event.get('payload')}"
                )

        return "\n".join(lines)

    def _classify_entity(self, label):
        if label in {"person", "human", "tony"}:
            return "person"

        if label in {"backpack", "chair", "bottle", "book", "cup"}:
            return "object"

        return "unknown"

    def _entity_id(self, label, entity_type):
        clean = label.replace(" ", "_").lower()
        if clean:
            return f"{entity_type}-{clean}"
        return f"{entity_type}-{uuid4().hex[:8]}"
