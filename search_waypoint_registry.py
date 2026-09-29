"""Persistent, motion-free registry of named map-frame search waypoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, Dict, List, Optional


SEARCH_WAYPOINT_REGISTRY_FILE = "search_waypoint_registry.json"
_VERSION = 1
_UNSET = object()


class WaypointRegistryError(Exception):
    """Base error for waypoint-registry operations."""


class WaypointValidationError(WaypointRegistryError, ValueError):
    """Raised when a waypoint field is absent or invalid."""


class DuplicateWaypointError(WaypointRegistryError, ValueError):
    """Raised when adding an already-defined waypoint ID."""


class UnknownWaypointError(WaypointRegistryError, KeyError):
    """Raised when an operation references a missing waypoint ID."""


@dataclass(frozen=True)
class SearchWaypoint:
    """An immutable named pose in the fixed-map frame."""

    waypoint_id: str
    name: str
    x: float
    y: float
    yaw: float
    active: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SearchWaypointRegistry:
    """One authoritative durable registry of configured search locations.

    The registry deliberately contains definitions only. It does not import or
    call navigation, localization, Robot Bridge, the World Model, or Marvin
    pursuit code. Insertion order is retained as the deterministic order a
    future bounded room-search coordinator may consume.
    """

    def __init__(
        self,
        storage_path: str = SEARCH_WAYPOINT_REGISTRY_FILE,
        *,
        clock: Optional[Callable[[], str]] = None,
    ):
        self.storage_path = Path(storage_path)
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self._waypoints: Dict[str, SearchWaypoint] = {}
        self.reload()

    @staticmethod
    def _validate_text(field: str, value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise WaypointValidationError(
                f"{field} must be a non-empty string without surrounding whitespace."
            )
        return value

    @staticmethod
    def _validate_coordinate(field: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WaypointValidationError(f"{field} must be a finite number.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise WaypointValidationError(f"{field} must be finite.")
        return numeric

    @staticmethod
    def _validate_active(value: object) -> bool:
        if type(value) is not bool:
            raise WaypointValidationError("active must be a boolean.")
        return value

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, str) or not value:
            raise WaypointRegistryError("Waypoint clock returned an invalid timestamp.")
        return value

    @classmethod
    def _waypoint_from_data(cls, data: object) -> SearchWaypoint:
        if not isinstance(data, dict):
            raise WaypointRegistryError("Persisted waypoint record must be an object.")
        expected = {
            "waypoint_id", "name", "x", "y", "yaw", "active",
            "created_at", "updated_at",
        }
        if set(data) != expected:
            raise WaypointRegistryError("Persisted waypoint record has an invalid schema.")
        created_at = data["created_at"]
        updated_at = data["updated_at"]
        if not isinstance(created_at, str) or not created_at:
            raise WaypointRegistryError("Persisted waypoint created_at is invalid.")
        if not isinstance(updated_at, str) or not updated_at:
            raise WaypointRegistryError("Persisted waypoint updated_at is invalid.")
        return SearchWaypoint(
            waypoint_id=cls._validate_text("waypoint_id", data["waypoint_id"]),
            name=cls._validate_text("name", data["name"]),
            x=cls._validate_coordinate("x", data["x"]),
            y=cls._validate_coordinate("y", data["y"]),
            yaw=cls._validate_coordinate("yaw", data["yaw"]),
            active=cls._validate_active(data["active"]),
            created_at=created_at,
            updated_at=updated_at,
        )

    def reload(self) -> None:
        """Load the sole registry file; a missing file means an empty registry."""
        with self._lock:
            if not self.storage_path.exists():
                self._waypoints = {}
                return
            try:
                payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WaypointRegistryError("Waypoint registry could not be loaded.") from exc
            if not isinstance(payload, dict) or payload.get("version") != _VERSION:
                raise WaypointRegistryError("Waypoint registry has an unsupported schema.")
            records = payload.get("waypoints")
            if not isinstance(records, list):
                raise WaypointRegistryError("Waypoint registry waypoints must be a list.")
            loaded: Dict[str, SearchWaypoint] = {}
            for record in records:
                waypoint = self._waypoint_from_data(record)
                if waypoint.waypoint_id in loaded:
                    raise WaypointRegistryError("Waypoint registry contains duplicate IDs.")
                loaded[waypoint.waypoint_id] = waypoint
            self._waypoints = loaded

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "waypoints": [waypoint.to_dict() for waypoint in self._waypoints.values()],
        }
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.storage_path.parent),
                prefix=f".{self.storage_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.storage_path)
        except OSError as exc:
            raise WaypointRegistryError("Waypoint registry could not be saved.") from exc
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def _require_waypoint(self, waypoint_id: object) -> SearchWaypoint:
        identifier = self._validate_text("waypoint_id", waypoint_id)
        try:
            return self._waypoints[identifier]
        except KeyError as exc:
            raise UnknownWaypointError(f"Unknown waypoint ID: {identifier}") from exc

    def add_waypoint(
        self, *, waypoint_id: str, name: str, x: float, y: float, yaw: float,
        active: bool = True,
    ) -> SearchWaypoint:
        with self._lock:
            identifier = self._validate_text("waypoint_id", waypoint_id)
            if identifier in self._waypoints:
                raise DuplicateWaypointError(f"Waypoint ID already exists: {identifier}")
            timestamp = self._timestamp()
            waypoint = SearchWaypoint(
                waypoint_id=identifier,
                name=self._validate_text("name", name),
                x=self._validate_coordinate("x", x),
                y=self._validate_coordinate("y", y),
                yaw=self._validate_coordinate("yaw", yaw),
                active=self._validate_active(active),
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._waypoints[identifier] = waypoint
            self._save()
            return waypoint

    def get_waypoint(self, waypoint_id: str) -> SearchWaypoint:
        with self._lock:
            return self._require_waypoint(waypoint_id)

    def list_waypoints(self) -> List[SearchWaypoint]:
        with self._lock:
            return list(self._waypoints.values())

    def list_active_waypoints(self) -> List[SearchWaypoint]:
        with self._lock:
            return [waypoint for waypoint in self._waypoints.values() if waypoint.active]

    def update_waypoint(
        self, waypoint_id: str, *, name: object = _UNSET, x: object = _UNSET,
        y: object = _UNSET, yaw: object = _UNSET, active: object = _UNSET,
    ) -> SearchWaypoint:
        with self._lock:
            existing = self._require_waypoint(waypoint_id)
            if all(value is _UNSET for value in (name, x, y, yaw, active)):
                raise WaypointValidationError("At least one waypoint field must be updated.")
            waypoint = SearchWaypoint(
                waypoint_id=existing.waypoint_id,
                name=(existing.name if name is _UNSET else self._validate_text("name", name)),
                x=(existing.x if x is _UNSET else self._validate_coordinate("x", x)),
                y=(existing.y if y is _UNSET else self._validate_coordinate("y", y)),
                yaw=(existing.yaw if yaw is _UNSET else self._validate_coordinate("yaw", yaw)),
                active=(existing.active if active is _UNSET else self._validate_active(active)),
                created_at=existing.created_at,
                updated_at=self._timestamp(),
            )
            self._waypoints[waypoint.waypoint_id] = waypoint
            self._save()
            return waypoint

    def deactivate_waypoint(self, waypoint_id: str) -> SearchWaypoint:
        return self.update_waypoint(waypoint_id, active=False)

    def reactivate_waypoint(self, waypoint_id: str) -> SearchWaypoint:
        return self.update_waypoint(waypoint_id, active=True)

    def remove_waypoint(self, waypoint_id: str) -> SearchWaypoint:
        with self._lock:
            waypoint = self._require_waypoint(waypoint_id)
            del self._waypoints[waypoint.waypoint_id]
            self._save()
            return waypoint
