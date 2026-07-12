from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import copy
import fcntl
import json
import os
import tempfile
import threading


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
        data["history"] = [
            asdict(observation)
            for observation in self.history
        ]
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
            history=[],
        )

        for item in data.get("history", []):
            entity.history.append(
                EntityObservation(**item)
            )

        return entity


class WorldModel:
    """
    Persistent shared World Model.

    Multiple processes may safely use separate WorldModel instances that point
    to the same storage file.

    Persistence guarantees:

        - Cross-process advisory file locking
        - Atomic JSON replacement
        - Top-level dirty-field merging
        - Entity-level merging
        - Pending-event merging
        - No partially written JSON files

    The World Model remains the single source of truth.
    """

    def __init__(self, storage_path: str = WORLD_MODEL_FILE):
        self.storage_path = str(storage_path)
        self.lock_path = f"{self.storage_path}.lock"

        self._thread_lock = threading.RLock()
        self._pending_events: List[Dict[str, Any]] = []

        self.robot_state = self._default_robot_state()
        self.environment: Dict[str, Any] = {}
        self.entities: Dict[str, WorldEntity] = {}
        self.recent_events: List[Dict[str, Any]] = []

        self._synced_robot_state: Dict[str, Any] = {}
        self._synced_environment: Dict[str, Any] = {}
        self._synced_entities: Dict[str, Dict[str, Any]] = {}

        self.load()

    @staticmethod
    def _default_robot_state():
        return {
            "battery": None,
            "mission": None,
            "navigation_state": "UNKNOWN",
            "updated_at": now_iso(),
        }

    def set_robot_state(self, key: str, value: Any):
        with self._thread_lock:
            self.robot_state[key] = value
            self.robot_state["updated_at"] = now_iso()

            self.add_event(
                "robot_state_updated",
                {
                    "key": key,
                    "value": value,
                },
            )

            self.save()

    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ):
        event = {
            "timestamp": now_iso(),
            "type": event_type,
            "data": data,
        }

        with self._thread_lock:
            self.recent_events.append(event)
            self.recent_events = self.recent_events[-100:]
            self._pending_events.append(event)

        return event

    def update_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str = "object",
        confidence: float = 0.0,
        source: str = "unknown",
        location: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        attributes = attributes or {}
        timestamp = now_iso()

        observation = EntityObservation(
            timestamp=timestamp,
            source=source,
            confidence=confidence,
            location=location,
            attributes=attributes,
        )

        with self._thread_lock:
            if entity_id not in self.entities:
                self.entities[entity_id] = WorldEntity(
                    entity_id=entity_id,
                    label=label,
                    entity_type=entity_type,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    confidence=confidence,
                    attributes=attributes,
                    history=[observation],
                )

                self.add_event(
                    "entity_created",
                    {
                        "entity_id": entity_id,
                        "label": label,
                        "entity_type": entity_type,
                    },
                )

            else:
                entity = self.entities[entity_id]
                entity.label = label
                entity.entity_type = entity_type
                entity.last_seen = timestamp
                entity.confidence = confidence
                entity.attributes.update(attributes)
                entity.history.append(observation)
                entity.history = entity.history[-50:]

                self.add_event(
                    "entity_updated",
                    {
                        "entity_id": entity_id,
                        "label": label,
                        "confidence": confidence,
                    },
                )

            self.save()

    def get_entity(
        self,
        entity_id: str,
    ) -> Optional[WorldEntity]:
        return self.entities.get(entity_id)

    def get_entities(self) -> List[Dict[str, Any]]:
        return [
            entity.to_dict()
            for entity in self.entities.values()
        ]

    def get_recent_events(self) -> List[Dict[str, Any]]:
        return list(self.recent_events)

    def get_context(self) -> Dict[str, Any]:
        return {
            "robot_state": dict(self.robot_state),
            "environment": copy.deepcopy(self.environment),
            "entities": self.get_entities(),
            "recent_events": list(self.recent_events),
        }

    def update_robot_state(self, **updates):
        """
        Update multiple robot-state fields and persist them safely.
        """
        if not updates:
            return self.robot_state

        with self._thread_lock:
            self.robot_state.update(updates)
            self.robot_state["updated_at"] = now_iso()

            self.add_event(
                "robot_state_updated",
                {
                    "updates": updates,
                },
            )

            self.save()

            return self.robot_state

    def update_from_detections(
        self,
        detections,
        source="vision",
    ):
        updated_entity_ids = []

        for index, detection in enumerate(detections):
            label = str(
                detection.get("label", "unknown")
            ).strip().lower()

            entity_id = detection.get("entity_id")

            if not entity_id:
                entity_id = f"{label}-{index + 1:03d}"

            entity_type = detection.get("entity_type")

            if not entity_type:
                entity_type = (
                    "human"
                    if label == "person"
                    else "object"
                )

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

            attributes = dict(
                detection.get("attributes", {})
            )

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
                confidence=float(
                    detection.get("confidence", 0.0)
                ),
                source=source,
                location=location or None,
                attributes=attributes,
            )

            updated_entity_ids.append(entity_id)

        return updated_entity_ids

    def snapshot(self) -> Dict[str, Any]:
        robot = dict(self.robot_state)

        if "battery_percent" not in robot:
            robot["battery_percent"] = robot.get("battery")

        robot.setdefault("name", "Mini Pupper 2")
        robot.setdefault("mission", None)
        robot.setdefault(
            "navigation_state",
            "UNKNOWN",
        )
        robot.setdefault("front_clear", None)
        robot.setdefault("nearest_obstacle_m", None)

        entities = []

        for entity in self.entities.values():
            data = entity.to_dict()
            location = {}

            if entity.history:
                location = (
                    entity.history[-1].location
                    or {}
                )

            attributes = entity.attributes or {}

            data["distance_m"] = attributes.get(
                "distance_m",
                location.get("distance_m"),
            )

            data["position"] = attributes.get(
                "position",
                location.get("position"),
            )

            data["tracking"] = attributes.get(
                "tracking",
                False,
            )

            entities.append(data)

        return {
            "robot": robot,
            "environment": copy.deepcopy(
                self.environment
            ),
            "entities": entities,
            "recent_events": list(
                self.recent_events
            ),
        }

    def save(self):
        """
        Merge this process's changed fields into the shared World Model.

        The merge and atomic file replacement occur while holding an exclusive
        cross-process lock.
        """
        with self._thread_lock:
            storage_path = Path(self.storage_path)
            storage_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            changed_robot_keys = self._changed_keys(
                self.robot_state,
                self._synced_robot_state,
            )

            changed_environment_keys = self._changed_keys(
                self.environment,
                self._synced_environment,
            )

            current_entities = self._serialize_entities()

            changed_entity_ids = {
                entity_id
                for entity_id, entity_data
                in current_entities.items()
                if (
                    self._synced_entities.get(entity_id)
                    != entity_data
                )
            }

            with self._open_lock_file() as lock_file:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX,
                )

                try:
                    disk_data = self._read_data_unlocked()

                    merged_robot_state = self._default_robot_state()
                    merged_robot_state.update(
                        disk_data.get(
                            "robot_state",
                            {},
                        )
                    )

                    for key in changed_robot_keys:
                        merged_robot_state[key] = copy.deepcopy(
                            self.robot_state[key]
                        )

                    merged_environment = copy.deepcopy(
                        disk_data.get(
                            "environment",
                            {},
                        )
                    )

                    for key in changed_environment_keys:
                        merged_environment[key] = copy.deepcopy(
                            self.environment[key]
                        )

                    merged_entities = copy.deepcopy(
                        disk_data.get(
                            "entities",
                            {},
                        )
                    )

                    for entity_id in changed_entity_ids:
                        merged_entities[entity_id] = copy.deepcopy(
                            current_entities[entity_id]
                        )

                    disk_events = list(
                        disk_data.get(
                            "recent_events",
                            [],
                        )
                    )

                    merged_events = (
                        disk_events
                        + copy.deepcopy(
                            self._pending_events
                        )
                    )[-100:]

                    merged_data = {
                        "robot_state": merged_robot_state,
                        "environment": merged_environment,
                        "entities": merged_entities,
                        "recent_events": merged_events,
                    }

                    self._atomic_write_unlocked(
                        merged_data
                    )

                    self._apply_loaded_data(
                        merged_data
                    )

                    self._pending_events = []
                    self._capture_synced_state()

                finally:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_UN,
                    )

    def load(self):
        """
        Load a consistent snapshot while holding a shared cross-process lock.
        """
        with self._thread_lock:
            storage_path = Path(self.storage_path)
            storage_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self._open_lock_file() as lock_file:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_SH,
                )

                try:
                    data = self._read_data_unlocked()
                    self._apply_loaded_data(data)
                    self._pending_events = []
                    self._capture_synced_state()

                finally:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_UN,
                    )

    def reload(self):
        """
        Refresh this instance from the shared persistent World Model.
        """
        self.load()
        return self.get_context()

    def _open_lock_file(self):
        lock_path = Path(self.lock_path)
        lock_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return open(
            lock_path,
            "a+",
            encoding="utf-8",
        )

    def _read_data_unlocked(self):
        storage_path = Path(self.storage_path)

        if not storage_path.exists():
            return {}

        raw_text = storage_path.read_text(
            encoding="utf-8"
        ).strip()

        if not raw_text:
            return {}

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError(
                "World Model storage must contain a JSON object."
            )

        return data

    def _atomic_write_unlocked(
        self,
        data: Dict[str, Any],
    ):
        storage_path = Path(self.storage_path)
        storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(storage_path.parent),
                prefix=f".{storage_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                json.dump(
                    data,
                    temporary_file,
                    indent=2,
                )

                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                storage_path,
            )

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def _apply_loaded_data(
        self,
        data: Dict[str, Any],
    ):
        robot_state = self._default_robot_state()
        robot_state.update(
            data.get(
                "robot_state",
                {},
            )
        )

        self.robot_state = robot_state
        self.environment = copy.deepcopy(
            data.get(
                "environment",
                {},
            )
        )

        self.recent_events = list(
            data.get(
                "recent_events",
                [],
            )
        )[-100:]

        self.entities = {
            entity_id: WorldEntity.from_dict(
                entity_data
            )
            for entity_id, entity_data
            in data.get(
                "entities",
                {},
            ).items()
        }

    def _capture_synced_state(self):
        self._synced_robot_state = copy.deepcopy(
            self.robot_state
        )

        self._synced_environment = copy.deepcopy(
            self.environment
        )

        self._synced_entities = copy.deepcopy(
            self._serialize_entities()
        )

    def _serialize_entities(self):
        return {
            entity_id: entity.to_dict()
            for entity_id, entity
            in self.entities.items()
        }

    @staticmethod
    def _changed_keys(
        current: Dict[str, Any],
        synced: Dict[str, Any],
    ):
        return {
            key
            for key, value in current.items()
            if synced.get(key) != value
        }
