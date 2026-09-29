"""Read-only adapter for the validated Tony2 AMCL map-pose contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Tuple


CANONICAL_LOCALIZED_POSE_SOURCE = "tony2_navigation_amcl"
MAX_LOCALIZED_POSE_AGE_SECONDS = 3.0


class LocalizedPoseUnavailableError(RuntimeError):
    """Raised when the existing localization authority cannot prove a pose safe."""


@dataclass(frozen=True)
class CurrentLocalizedPose:
    """One fresh, validated map-frame pose from the running Tony2 AMCL probe."""

    x: float
    y: float
    yaw: float
    timestamp: str
    frame: str = "map"


class CurrentLocalizedPoseProvider:
    """Adapt ``Tony2NavigationRuntime.live_pose_status`` without controlling it.

    ``status_reader`` is intentionally injected.  The production wiring can pass
    the existing already-running Tony2 runtime's read-only ``live_pose_status``
    method; this module neither creates that runtime nor imports ROS/Nav2 code.
    """

    def __init__(
        self,
        status_reader: Callable[[], Tuple[int, Mapping[str, object]]],
        *,
        max_pose_age_seconds: float = MAX_LOCALIZED_POSE_AGE_SECONDS,
    ):
        if (
            isinstance(max_pose_age_seconds, bool)
            or not isinstance(max_pose_age_seconds, (int, float))
            or not math.isfinite(float(max_pose_age_seconds))
            or float(max_pose_age_seconds) <= 0.0
        ):
            raise ValueError("max_pose_age_seconds must be a positive finite number.")
        self._status_reader = status_reader
        self._max_pose_age_seconds = float(max_pose_age_seconds)

    @staticmethod
    def _mapping(value: object, field: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise LocalizedPoseUnavailableError(f"{field} is unavailable.")
        return value

    @staticmethod
    def _finite_number(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LocalizedPoseUnavailableError(f"{field} is invalid.")
        result = float(value)
        if not math.isfinite(result):
            raise LocalizedPoseUnavailableError(f"{field} is invalid.")
        return result

    def get_current_pose(self) -> CurrentLocalizedPose:
        """Return only a fresh map pose backed by validated localization authority."""
        try:
            status_code, payload_value = self._status_reader()
        except Exception as exc:
            raise LocalizedPoseUnavailableError(
                "Current localized pose could not be read."
            ) from exc

        payload = self._mapping(payload_value, "localized pose status")
        if status_code != 200:
            raise LocalizedPoseUnavailableError("Localized pose status is unavailable.")
        if payload.get("ok") is not True:
            raise LocalizedPoseUnavailableError("Localized pose status is not ready.")
        if payload.get("authoritative") is not True:
            raise LocalizedPoseUnavailableError("Localized pose authority is not validated.")
        if payload.get("read_only") is not True:
            raise LocalizedPoseUnavailableError("Localized pose source is not read-only.")
        if payload.get("source") != CANONICAL_LOCALIZED_POSE_SOURCE:
            raise LocalizedPoseUnavailableError("Localized pose source is not canonical AMCL.")

        navigation = self._mapping(payload.get("navigation"), "navigation status")
        required_navigation = (
            "running",
            "localization_enabled",
            "transform_ready",
            "localization_validated",
        )
        if any(navigation.get(field) is not True for field in required_navigation):
            raise LocalizedPoseUnavailableError("Localization authority is not validated.")

        telemetry = self._mapping(payload.get("telemetry"), "localized pose telemetry")
        if telemetry.get("available") is not True:
            raise LocalizedPoseUnavailableError("Current localized pose is unavailable.")
        age_seconds = self._finite_number(
            telemetry.get("age_seconds"), "localized pose age_seconds"
        )
        if age_seconds < 0.0 or age_seconds >= self._max_pose_age_seconds:
            raise LocalizedPoseUnavailableError("Current localized pose is stale.")

        timestamp = telemetry.get("received_at")
        if not isinstance(timestamp, str) or not timestamp:
            raise LocalizedPoseUnavailableError("Current localized pose timestamp is invalid.")
        pose = self._mapping(telemetry.get("pose"), "current localized pose")
        if pose.get("frame_id") != "map":
            raise LocalizedPoseUnavailableError("Current localized pose is not in the map frame.")
        position = self._mapping(pose.get("position"), "current localized position")
        return CurrentLocalizedPose(
            x=self._finite_number(position.get("x"), "current localized pose x"),
            y=self._finite_number(position.get("y"), "current localized pose y"),
            yaw=self._finite_number(pose.get("yaw_radians"), "current localized pose yaw"),
            timestamp=timestamp,
        )
