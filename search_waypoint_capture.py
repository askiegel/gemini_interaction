"""Save a read-only validated localized pose through the waypoint registry."""

from __future__ import annotations

from current_localized_pose import CurrentLocalizedPoseProvider
from search_waypoint_registry import SearchWaypoint, SearchWaypointRegistry


class SearchWaypointCaptureService:
    """Capture location definitions only; this class never controls a robot."""

    def __init__(
        self,
        registry: SearchWaypointRegistry,
        pose_provider: CurrentLocalizedPoseProvider,
    ):
        self._registry = registry
        self._pose_provider = pose_provider

    def save_current_pose_as_waypoint(
        self, *, waypoint_id: str, name: str,
    ) -> SearchWaypoint:
        """Persist the current authoritative map pose, never caller coordinates."""
        self._registry.validate_new_waypoint(
            waypoint_id=waypoint_id,
            name=name,
        )
        pose = self._pose_provider.get_current_pose()
        return self._registry.add_waypoint(
            waypoint_id=waypoint_id,
            name=name,
            x=pose.x,
            y=pose.y,
            yaw=pose.yaw,
        )
