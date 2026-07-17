#!/usr/bin/env python3

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class TargetLock:
    """
    Own the transient target identity used by a tracking behavior.

    The World Model remains the source of perception data. TargetLock only
    remembers which World Model entity the active behavior selected.

    Commit 1 responsibilities:

        - Acquire the best visible entity by label.
        - Remember its entity_id.
        - Query that entity by ID on later control cycles.
        - Reset when a different mission starts or STOP is requested.

    Lost-target timeout and directional recovery are intentionally deferred
    to the next feature commit.
    """

    MODE_UNLOCKED = "UNLOCKED"
    MODE_LOCKED = "LOCKED"

    def __init__(
        self,
        world_model,
        max_age_seconds: float = 3.0,
    ):
        self.world_model = world_model
        self.max_age_seconds = float(max_age_seconds)

        self.mission_id: Optional[str] = None
        self.target_label: Optional[str] = None
        self.locked_entity_id: Optional[str] = None
        self.locked_since: Optional[str] = None
        self.tracking_mode = self.MODE_UNLOCKED

    @staticmethod
    def _now_iso() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> Optional[datetime]:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc,
            )

        return parsed.astimezone(timezone.utc)

    @classmethod
    def _age_seconds(
        cls,
        timestamp: Any,
    ) -> Optional[float]:
        parsed = cls._parse_timestamp(timestamp)

        if parsed is None:
            return None

        return max(
            0.0,
            (
                datetime.now(timezone.utc) - parsed
            ).total_seconds(),
        )

    @staticmethod
    def _bbox_from_value(
        value: Any,
    ) -> Optional[Dict[str, float]]:
        if isinstance(value, dict):
            x1 = value.get("x1", value.get("left"))
            y1 = value.get("y1", value.get("top"))
            x2 = value.get("x2", value.get("right"))
            y2 = value.get("y2", value.get("bottom"))

            if None not in (x1, y1, x2, y2):
                return {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }

        if (
            isinstance(value, (list, tuple))
            and len(value) >= 4
        ):
            return {
                "x1": float(value[0]),
                "y1": float(value[1]),
                "x2": float(value[2]),
                "y2": float(value[3]),
            }

        return None

    def reset(self):
        self.mission_id = None
        self.target_label = None
        self.locked_entity_id = None
        self.locked_since = None
        self.tracking_mode = self.MODE_UNLOCKED

    def start_mission(
        self,
        mission_id: Optional[str],
        target_label: str,
    ):
        normalized_mission_id = str(
            mission_id or ""
        ).strip() or None

        normalized_label = str(
            target_label or ""
        ).strip().lower() or None

        if (
            self.mission_id != normalized_mission_id
            or self.target_label != normalized_label
        ):
            self.reset()
            self.mission_id = normalized_mission_id
            self.target_label = normalized_label

    def _lock_from_observation(
        self,
        observation: Dict[str, Any],
    ):
        entity_id = str(
            observation.get("entity_id") or ""
        ).strip()

        if not entity_id:
            return

        self.locked_entity_id = entity_id
        self.locked_since = self._now_iso()
        self.tracking_mode = self.MODE_LOCKED

    def _query_locked_entity(
        self,
    ) -> Dict[str, Any]:
        """
        Read the specifically locked World Model entity.

        This deliberately does not query by label. Therefore another person
        cannot replace the selected person merely because their detection is
        newer or has higher confidence.
        """
        if not self.locked_entity_id:
            return {
                "found": False,
                "target": self.target_label,
                "reason": "No entity is locked.",
            }

        if hasattr(self.world_model, "reload"):
            self.world_model.reload()

        if not hasattr(self.world_model, "get_entity"):
            raise RuntimeError(
                "World Model does not provide get_entity()."
            )

        entity = self.world_model.get_entity(
            self.locked_entity_id
        )

        if entity is None:
            return {
                "found": False,
                "target": self.target_label,
                "entity_id": self.locked_entity_id,
                "stale": False,
                "reason": (
                    f"Locked entity "
                    f"'{self.locked_entity_id}' is unavailable."
                ),
            }

        latest_observation = (
            entity.history[-1]
            if getattr(entity, "history", None)
            else None
        )

        location = dict(
            getattr(
                latest_observation,
                "location",
                None,
            )
            or {}
        )

        observation_attributes = dict(
            getattr(
                latest_observation,
                "attributes",
                None,
            )
            or {}
        )

        entity_attributes = dict(
            getattr(entity, "attributes", None)
            or {}
        )

        attributes = {
            **entity_attributes,
            **observation_attributes,
        }

        last_seen = getattr(
            entity,
            "last_seen",
            None,
        )

        age_seconds = self._age_seconds(
            last_seen
        )

        stale = (
            age_seconds is not None
            and age_seconds > self.max_age_seconds
        )

        bbox = self._bbox_from_value(
            attributes.get("bbox")
        )

        cx = location.get(
            "cx",
            location.get("center_x"),
        )

        cy = location.get(
            "cy",
            location.get("center_y"),
        )

        area = attributes.get("area")

        if bbox is not None:
            width = max(
                0.0,
                bbox["x2"] - bbox["x1"],
            )

            height = max(
                0.0,
                bbox["y2"] - bbox["y1"],
            )

            if cx is None:
                cx = bbox["x1"] + width / 2.0

            if cy is None:
                cy = bbox["y1"] + height / 2.0

            if area is None:
                area = width * height

        result = {
            "found": not stale,
            "stale": stale,
            "target": self.target_label,
            "entity_id": self.locked_entity_id,
            "label": getattr(
                entity,
                "label",
                self.target_label,
            ),
            "confidence": float(
                getattr(entity, "confidence", 0.0)
                or 0.0
            ),
            "cx": (
                float(cx)
                if cx is not None
                else None
            ),
            "cy": (
                float(cy)
                if cy is not None
                else None
            ),
            "area": (
                float(area)
                if area is not None
                else None
            ),
            "bbox": bbox,
            "image_width": attributes.get(
                "image_width"
            ),
            "image_height": attributes.get(
                "image_height"
            ),
            "last_seen": last_seen,
            "detection_age_ms": (
                int(round(age_seconds * 1000.0))
                if age_seconds is not None
                else None
            ),
        }

        if stale:
            result["reason"] = (
                f"Locked entity '{self.locked_entity_id}' "
                f"is stale."
            )

        return result

    def resolve(
        self,
        mission_id: Optional[str],
        target_label: str,
    ) -> Dict[str, Any]:
        """
        Return the observation for the active target lock.

        The first successful cycle acquires by label. Every later cycle reads
        only the locked entity ID.
        """
        self.start_mission(
            mission_id=mission_id,
            target_label=target_label,
        )

        if self.locked_entity_id:
            return self._query_locked_entity()

        observation = (
            self.world_model.find_latest_entity_by_label(
                target_label,
                max_age_seconds=self.max_age_seconds,
                refresh=True,
            )
        )

        if observation.get("found"):
            self._lock_from_observation(
                observation
            )

        return observation

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tracking_mode": self.tracking_mode,
            "locked_entity_id": self.locked_entity_id,
            "locked_since": self.locked_since,
        }
