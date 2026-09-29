"""Non-motion command service for current-pose waypoint capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

from current_localized_pose import LocalizedPoseUnavailableError
from remember_current_location import REMEMBER_CURRENT_LOCATION
from search_waypoint_capture import SearchWaypointCaptureService
from search_waypoint_registry import (
    DuplicateWaypointError,
    WaypointRegistryError,
    WaypointValidationError,
)


@dataclass(frozen=True)
class RememberCurrentLocationResult:
    ok: bool
    intent: str
    reason: Optional[str]
    waypoint: Optional[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RememberCurrentLocationService:
    """Save only the capture service's trusted pose; never controls motion."""

    def __init__(self, capture_service: SearchWaypointCaptureService):
        self._capture_service = capture_service

    def execute(self, *, waypoint_id: str, name: str) -> RememberCurrentLocationResult:
        try:
            waypoint = self._capture_service.save_current_pose_as_waypoint(
                waypoint_id=waypoint_id,
                name=name,
            )
        except DuplicateWaypointError:
            return self._failure("DUPLICATE_WAYPOINT")
        except WaypointValidationError:
            return self._failure("INVALID_NAME")
        except LocalizedPoseUnavailableError as exc:
            reason = getattr(exc, "reason", "LOCALIZATION_NOT_READY")
            if reason not in {"LOCALIZATION_NOT_READY", "STALE_POSE"}:
                reason = "LOCALIZATION_NOT_READY"
            return self._failure(reason)
        except WaypointRegistryError:
            return self._failure("CAPTURE_FAILED")
        return RememberCurrentLocationResult(
            ok=True,
            intent=REMEMBER_CURRENT_LOCATION,
            reason=None,
            waypoint={
                "waypoint_id": waypoint.waypoint_id,
                "name": waypoint.name,
                "x": waypoint.x,
                "y": waypoint.y,
                "yaw": waypoint.yaw,
                "frame": "map",
            },
        )

    @staticmethod
    def _failure(reason: str) -> RememberCurrentLocationResult:
        return RememberCurrentLocationResult(
            ok=False,
            intent=REMEMBER_CURRENT_LOCATION,
            reason=reason,
            waypoint=None,
        )
