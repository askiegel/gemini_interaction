#!/usr/bin/env python3

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from prediction_tracker import PredictionTracker


class TargetLock:
    """
    Preserve the World Model identity selected by an active tracking mission.

    Normal operation:

        1. Acquire an entity by label.
        2. Remember its entity_id.
        3. Query only that entity on later control cycles.

    Lost-target recovery:

        1. Retain the entity lock during a short visibility interruption.
        2. Remember whether the target was last seen left, right, or center.
        3. Supply that direction to BehaviorManager for recovery steering.
        4. Release the lock only after the recovery timeout expires.
        5. Permit a fresh label acquisition after the lock is released.
    """

    MODE_UNLOCKED = "UNLOCKED"
    MODE_LOCKED = "LOCKED"
    MODE_RECOVERING = "RECOVERING"
    MODE_WAITING_FOR_IDENTITY = "WAITING_FOR_IDENTITY"

    DIRECTION_LEFT = "LEFT"
    DIRECTION_RIGHT = "RIGHT"
    DIRECTION_CENTER = "CENTER"

    def __init__(
        self,
        world_model,
        max_age_seconds: float = 3.0,
        recovery_timeout_seconds: float = 2.0,
    ):
        self.world_model = world_model
        self.max_age_seconds = float(max_age_seconds)
        self.recovery_timeout_seconds = float(
            recovery_timeout_seconds
        )

        self.prediction_tracker = PredictionTracker(
            maximum_prediction_seconds=(
                self.recovery_timeout_seconds
            )
        )

        self.mission_id: Optional[str] = None
        self.target_label: Optional[str] = None

        self.locked_entity_id: Optional[str] = None
        self.locked_identity_id: Optional[str] = None
        self.last_entity_migration: Optional[Dict[str, Any]] = None
        self.locked_since: Optional[str] = None
        self.tracking_mode = self.MODE_UNLOCKED

        self.last_valid_observation: Optional[
            Dict[str, Any]
        ] = None
        self.last_visible_at: Optional[str] = None
        self.lost_since: Optional[str] = None
        self.waiting_since: Optional[str] = None
        self.waiting_entity_id: Optional[str] = None
        self.last_seen_direction: Optional[str] = None

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _now_iso(cls) -> str:
        return (
            cls._now()
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
                cls._now() - parsed
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

    @classmethod
    def _direction_from_observation(
        cls,
        observation: Dict[str, Any],
    ) -> Optional[str]:
        cx = observation.get("cx")
        image_width = observation.get("image_width")

        if cx is None or image_width is None:
            return None

        center = float(image_width) / 2.0
        horizontal_error = float(cx) - center

        center_band = max(
            40.0,
            float(image_width) * 0.10,
        )

        if horizontal_error < -center_band:
            return cls.DIRECTION_LEFT

        if horizontal_error > center_band:
            return cls.DIRECTION_RIGHT

        return cls.DIRECTION_CENTER

    def reset(self):
        self.mission_id = None
        self.target_label = None

        self.locked_entity_id = None
        self.locked_identity_id = None
        self.last_entity_migration = None
        self.locked_since = None
        self.tracking_mode = self.MODE_UNLOCKED

        self.last_valid_observation = None
        self.last_visible_at = None
        self.lost_since = None
        self.waiting_since = None
        self.waiting_entity_id = None
        self.last_seen_direction = None

        self.prediction_tracker.reset()

    def _release_lock(self):
        self.locked_entity_id = None
        self.locked_identity_id = None
        self.last_entity_migration = None
        self.locked_since = None
        self.tracking_mode = self.MODE_UNLOCKED

        self.last_valid_observation = None
        self.last_visible_at = None
        self.lost_since = None
        self.waiting_since = None
        self.waiting_entity_id = None
        self.last_seen_direction = None

        self.prediction_tracker.reset()

    def _enter_identity_wait(
        self,
    ) -> Dict[str, Any]:
        """
        Release the transient entity lock while preserving persistent identity.

        Detector entity IDs may disappear after an occlusion or tracker reset.
        The selected identity remains authoritative until the FOLLOW_PERSON
        mission is stopped, cancelled, or replaced.
        """
        expired_entity_id = self.locked_entity_id

        if self.waiting_since is None:
            self.waiting_since = self._now_iso()

        # Release the active transient lock so WAITING_FOR_IDENTITY remains
        # stationary and reports locked_entity_id=None. Preserve the expired
        # entity separately for exact-entity continuity checks only.
        self.waiting_entity_id = expired_entity_id
        self.locked_entity_id = None
        self.tracking_mode = self.MODE_WAITING_FOR_IDENTITY

        self.last_valid_observation = None
        self.lost_since = None
        self.last_seen_direction = None

        self.prediction_tracker.reset()

        return {
            "found": False,
            "stale": False,
            "target": self.target_label,
            "entity_id": None,
            "expired_entity_id": expired_entity_id,
            "identity_id": self.locked_identity_id,
            "locked_identity_id": self.locked_identity_id,
            "identity_lost": True,
            "identity_retained": bool(
                self.locked_identity_id
            ),
            "lock_expired": True,
            "reacquisition_blocked": True,
            "label_reacquisition_blocked": True,
            "identity_reacquisition_pending": True,
            "tracking_mode": (
                self.MODE_WAITING_FOR_IDENTITY
            ),
            "waiting_since": self.waiting_since,
            "waiting_entity_id": self.waiting_entity_id,
            "recovery_direction": None,
            "reason": (
                "Predictive recovery expired. The transient "
                "entity lock was released, but the persistent "
                "identity remains selected. Waiting for that "
                "same identity to become visible again."
            ),
        }

    def _waiting_result(
        self,
    ) -> Dict[str, Any]:
        """
        Report a stationary identity wait without acquiring by label.
        """
        waiting_age_seconds = (
            self._age_seconds(self.waiting_since)
            if self.waiting_since
            else 0.0
        )

        return {
            "found": False,
            "stale": False,
            "target": self.target_label,
            "entity_id": None,
            "identity_id": self.locked_identity_id,
            "locked_identity_id": self.locked_identity_id,
            "identity_lost": True,
            "identity_retained": bool(
                self.locked_identity_id
            ),
            "lock_expired": True,
            "reacquisition_blocked": True,
            "label_reacquisition_blocked": True,
            "identity_reacquisition_pending": True,
            "tracking_mode": (
                self.MODE_WAITING_FOR_IDENTITY
            ),
            "waiting_since": self.waiting_since,
            "waiting_age_seconds": (
                waiting_age_seconds or 0.0
            ),
            "recovery_direction": None,
            "reason": (
                "Waiting for the selected persistent identity. "
                "Label-based acquisition of another person is blocked."
            ),
        }

    def _identity_mismatch_result(
        self,
        observation: Dict[str, Any],
        observed_identity_id: str,
    ) -> Dict[str, Any]:
        """
        Block silent transfer of persistent mission ownership.
        """
        if (
            self.tracking_mode
            == self.MODE_WAITING_FOR_IDENTITY
        ):
            result = self._waiting_result()
        else:
            result = {
                **observation,
                "found": False,
                "entity_id": self.locked_entity_id,
            }

        return {
            **result,
            "identity_id": self.locked_identity_id,
            "locked_identity_id": (
                self.locked_identity_id
            ),
            "observed_identity_id": (
                observed_identity_id
            ),
            "identity_mismatch": True,
            "identity_refreshed": False,
            "reacquisition_blocked": True,
            "label_reacquisition_blocked": True,
            "reason": (
                "The currently observed entity reports a "
                "different persistent identity. Mission "
                "ownership remains with the originally "
                "selected identity."
            ),
        }

    def _resolve_waiting_identity(
        self,
    ) -> Dict[str, Any]:
        """
        Query only the selected persistent identity while waiting.

        Label-based acquisition is never used in this state. Tracking returns
        to LOCKED only when the World Model reports the same persistent
        identity with a fresh transient entity ID.
        """
        waiting = self._waiting_result()

        # Before searching by persistent identity, check continuity through
        # the exact transient entity that was selected before recovery
        # expired. This never performs label-based reacquisition.
        #
        # If that exact entity is fresh again, its current identity assignment
        # is authoritative for continuity. This handles identity refinement
        # such as old temporary identity -> stable identity while preventing a
        # different person from stealing the lock.
        if (
            self.waiting_entity_id
            and hasattr(self.world_model, "get_entity")
        ):
            waiting_entity_id = self.waiting_entity_id
            self.locked_entity_id = waiting_entity_id

            try:
                continuity_result = self._query_locked_entity()
            finally:
                self.locked_entity_id = None

            if continuity_result.get("found"):
                observed_identity_id = str(
                    continuity_result.get("identity_id") or ""
                ).strip() or None

                previous_identity_id = self.locked_identity_id
                identity_mismatch = bool(
                    previous_identity_id
                    and observed_identity_id
                    and observed_identity_id
                    != previous_identity_id
                )

                if identity_mismatch:
                    return self._identity_mismatch_result(
                        continuity_result,
                        observed_identity_id,
                    )

                # Adopt an identity only when the mission does not
                # already own one.
                identity_refreshed = bool(
                    observed_identity_id
                    and not previous_identity_id
                )

                if identity_refreshed:
                    self.locked_identity_id = observed_identity_id

                waiting_since = self.waiting_since
                waiting_age_seconds = (
                    self._age_seconds(waiting_since)
                    if waiting_since
                    else 0.0
                )

                self.locked_entity_id = waiting_entity_id
                self.waiting_entity_id = None
                self.tracking_mode = self.MODE_LOCKED
                self.waiting_since = None
                self.lost_since = None

                self._remember_visible_observation(
                    continuity_result
                )

                return {
                    **continuity_result,
                    "identity_id": (
                        observed_identity_id
                        or self.locked_identity_id
                    ),
                    "locked_identity_id": (
                        self.locked_identity_id
                    ),
                    "identity_lost": False,
                    "identity_retained": True,
                    "identity_refreshed": identity_refreshed,
                    "previous_identity_id": (
                        previous_identity_id
                        if identity_refreshed
                        else None
                    ),
                    "new_identity_id": (
                        self.locked_identity_id
                        if identity_refreshed
                        else None
                    ),
                    "lock_expired": False,
                    "reacquisition_blocked": False,
                    "label_reacquisition_blocked": True,
                    "identity_reacquisition_pending": False,
                    "identity_lookup_attempted": False,
                    "identity_reacquired": True,
                    "reacquired_entity_id": (
                        self.locked_entity_id
                    ),
                    "tracking_mode": self.MODE_LOCKED,
                    "previous_waiting_since": waiting_since,
                    "waiting_duration_seconds": (
                        waiting_age_seconds or 0.0
                    ),
                    "reason": (
                        "The exact previously selected entity became "
                        "visible again and tracking resumed."
                    ),
                }

        if not self.locked_identity_id:
            return {
                **waiting,
                "identity_reacquisition_pending": False,
                "reason": (
                    "Identity waiting cannot continue because "
                    "no persistent identity is selected."
                ),
            }

        if not hasattr(
            self.world_model,
            "find_latest_entity_by_identity",
        ):
            return {
                **waiting,
                "identity_lookup_available": False,
                "reason": (
                    "Waiting for the selected identity, but "
                    "the World Model does not provide identity lookup."
                ),
            }

        result = (
            self.world_model
            .find_latest_entity_by_identity(
                self.locked_identity_id,
                max_age_seconds=(
                    self.max_age_seconds
                ),
                refresh=True,
            )
        )

        resolved_identity_id = str(
            result.get("identity_id") or ""
        ).strip() or None

        if (
            resolved_identity_id
            and resolved_identity_id
            != self.locked_identity_id
        ):
            return {
                **waiting,
                "identity_lookup_attempted": True,
                "identity_mismatch": True,
                "resolved_identity_id": (
                    resolved_identity_id
                ),
                "reason": (
                    "Identity reacquisition was blocked because "
                    "the World Model returned a different identity."
                ),
            }

        new_entity_id = str(
            result.get("entity_id") or ""
        ).strip() or None

        if (
            not result.get("found")
            or not new_entity_id
        ):
            return {
                **waiting,
                "stale": bool(
                    result.get("stale")
                ),
                "identity_lookup_attempted": True,
                "identity_lookup_result": result,
                "reason": (
                    result.get("reason")
                    or waiting["reason"]
                ),
            }

        waiting_since = self.waiting_since
        waiting_age_seconds = (
            self._age_seconds(waiting_since)
            if waiting_since
            else 0.0
        )

        self.locked_entity_id = new_entity_id
        self.waiting_entity_id = None
        self.tracking_mode = self.MODE_LOCKED
        self.waiting_since = None
        self.lost_since = None

        self._remember_visible_observation(
            result
        )

        return {
            **result,
            "identity_id": self.locked_identity_id,
            "locked_identity_id": (
                self.locked_identity_id
            ),
            "identity_lost": False,
            "identity_retained": True,
            "lock_expired": False,
            "reacquisition_blocked": False,
            "label_reacquisition_blocked": True,
            "identity_reacquisition_pending": False,
            "identity_lookup_attempted": True,
            "identity_reacquired": True,
            "reacquired_entity_id": new_entity_id,
            "tracking_mode": self.MODE_LOCKED,
            "previous_waiting_since": waiting_since,
            "waiting_duration_seconds": (
                waiting_age_seconds or 0.0
            ),
            "reason": (
                "The selected persistent identity became "
                "visible again and tracking resumed."
            ),
        }

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

    def _remember_visible_observation(
        self,
        observation: Dict[str, Any],
    ):
        self.last_valid_observation = dict(
            observation
        )
        self.last_visible_at = self._now_iso()
        self.lost_since = None
        self.waiting_since = None

        direction = self._direction_from_observation(
            observation
        )

        if direction is not None:
            self.last_seen_direction = direction

        self.prediction_tracker.update(
            cx=observation.get("cx"),
            area=observation.get("area"),
            image_width=observation.get(
                "image_width"
            ),
            timestamp=(
                observation.get("last_seen")
                or self.last_visible_at
            ),
        )

    def _lock_from_observation(
        self,
        observation: Dict[str, Any],
    ):
        entity_id = str(
            observation.get("entity_id") or ""
        ).strip()

        if not entity_id:
            return

        identity_id = str(
            observation.get("identity_id") or ""
        ).strip() or None

        self.locked_entity_id = entity_id
        self.locked_identity_id = identity_id
        self.locked_since = self._now_iso()
        self.tracking_mode = self.MODE_LOCKED

        self._remember_visible_observation(
            observation
        )

    def _query_locked_target(
        self,
    ) -> Dict[str, Any]:
        """
        Resolve the active target through persistent identity when possible.

        A detector or tracker may assign a new transient entity_id to the
        same person. When the World Model supports identity lookup, the
        newest entity carrying locked_identity_id becomes the active entity.

        Label-based reacquisition is never performed while a lock exists.
        Older World Model test doubles that do not yet expose identity lookup
        continue using the legacy locked entity query.
        """
        # First preserve continuity through the currently locked transient
        # entity. The identity manager may refine or replace identity_id while
        # the same detector entity remains continuously visible.
        #
        # This is safe because label-based reacquisition is not involved:
        # the World Model must still return the exact locked entity_id and the
        # observation must be fresh.
        continuity_result = None

        if hasattr(self.world_model, "get_entity"):
            continuity_result = self._query_locked_entity()

        if (
            continuity_result is not None
            and continuity_result.get("found")
        ):
            observed_identity_id = str(
                continuity_result.get("identity_id") or ""
            ).strip() or None

            previous_identity_id = self.locked_identity_id
            identity_mismatch = bool(
                previous_identity_id
                and observed_identity_id
                and observed_identity_id
                != previous_identity_id
            )

            if identity_mismatch:
                return self._identity_mismatch_result(
                    continuity_result,
                    observed_identity_id,
                )

            # Adopt an identity only when the mission does not
            # already own one.
            identity_refreshed = bool(
                observed_identity_id
                and not previous_identity_id
            )

            if identity_refreshed:
                self.locked_identity_id = observed_identity_id

            return {
                **continuity_result,
                "identity_id": (
                    observed_identity_id
                    or self.locked_identity_id
                ),
                "locked_identity_id": self.locked_identity_id,
                "identity_refreshed": identity_refreshed,
                "previous_identity_id": (
                    previous_identity_id
                    if identity_refreshed
                    else None
                ),
                "new_identity_id": (
                    self.locked_identity_id
                    if identity_refreshed
                    else None
                ),
                "entity_migrated": False,
                "previous_entity_id": None,
                "new_entity_id": None,
            }

        if (
            self.locked_identity_id
            and hasattr(
                self.world_model,
                "find_latest_entity_by_identity",
            )
        ):
            previous_entity_id = self.locked_entity_id

            result = (
                self.world_model
                .find_latest_entity_by_identity(
                    self.locked_identity_id,
                    max_age_seconds=(
                        self.max_age_seconds
                    ),
                    refresh=True,
                )
            )

            resolved_identity_id = str(
                result.get("identity_id") or ""
            ).strip()

            if (
                resolved_identity_id
                and resolved_identity_id
                != self.locked_identity_id
            ):
                return {
                    "found": False,
                    "stale": False,
                    "target": self.target_label,
                    "entity_id": previous_entity_id,
                    "identity_id": (
                        self.locked_identity_id
                    ),
                    "identity_mismatch": True,
                    "reacquisition_blocked": True,
                    "reason": (
                        "World Model identity lookup returned "
                        "an entity belonging to a different "
                        "identity. Migration was blocked."
                    ),
                }

            new_entity_id = str(
                result.get("entity_id") or ""
            ).strip() or None

            entity_migrated = bool(
                new_entity_id
                and previous_entity_id
                and new_entity_id
                != previous_entity_id
            )

            if new_entity_id:
                self.locked_entity_id = new_entity_id

            if entity_migrated:
                self.last_entity_migration = {
                    "timestamp": self._now_iso(),
                    "identity_id": (
                        self.locked_identity_id
                    ),
                    "previous_entity_id": (
                        previous_entity_id
                    ),
                    "new_entity_id": new_entity_id,
                }

            return {
                **result,
                "identity_id": (
                    result.get("identity_id")
                    or self.locked_identity_id
                ),
                "entity_migrated": entity_migrated,
                "previous_entity_id": (
                    previous_entity_id
                    if entity_migrated
                    else None
                ),
                "new_entity_id": (
                    new_entity_id
                    if entity_migrated
                    else None
                ),
            }

        result = self._query_locked_entity()

        return {
            **result,
            "entity_migrated": False,
            "previous_entity_id": None,
            "new_entity_id": None,
        }

    def _query_locked_entity(
        self,
    ) -> Dict[str, Any]:
        """
        Read only the selected entity ID.

        A newer or higher-confidence person cannot replace the selected
        person while this lock is active.
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

        if area is None and bbox is not None:
            area = max(
                0.0,
                bbox["x2"] - bbox["x1"],
            ) * max(
                0.0,
                bbox["y2"] - bbox["y1"],
            )

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
            "identity_id": attributes.get(
                "identity_id"
            ),
            "identity_match_score": attributes.get(
                "identity_match_score"
            ),
            "identity_status": attributes.get(
                "identity_status"
            ),
            "identity_ambiguous": bool(
                attributes.get(
                    "identity_ambiguous",
                    False,
                )
            ),
            "identity_diagnostics": attributes.get(
                "identity_diagnostics"
            ),
        }

        if stale:
            result["reason"] = (
                f"Locked entity '{self.locked_entity_id}' "
                f"is stale."
            )

        return result

    def _recovery_result(
        self,
        query_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.lost_since is None:
            self.lost_since = self._now_iso()

        lost_age_seconds = (
            self._age_seconds(self.lost_since)
            or 0.0
        )

        if (
            lost_age_seconds
            > self.recovery_timeout_seconds
        ):
            waiting = self._enter_identity_wait()

            return {
                **waiting,
                "stale": bool(
                    query_result.get("stale")
                ),
                "lost_age_seconds": (
                    lost_age_seconds
                ),
            }

        self.tracking_mode = self.MODE_RECOVERING

        prediction = (
            self.prediction_tracker.predict()
        )

        recovery_direction = (
            prediction.get("predicted_direction")
            if prediction.get("available")
            else self.last_seen_direction
        )

        if recovery_direction == self.DIRECTION_CENTER:
            horizontal_velocity = prediction.get(
                "horizontal_velocity"
            )

            velocity_deadband = 15.0

            if (
                horizontal_velocity is not None
                and horizontal_velocity > velocity_deadband
            ):
                recovery_direction = (
                    self.DIRECTION_RIGHT
                )
            elif (
                horizontal_velocity is not None
                and horizontal_velocity < -velocity_deadband
            ):
                recovery_direction = (
                    self.DIRECTION_LEFT
                )
            else:
                recovery_direction = (
                    self.DIRECTION_CENTER
                )

        if (
            recovery_direction is None
            and self.last_seen_direction is not None
        ):
            recovery_direction = (
                self.last_seen_direction
            )

        return {
            "found": False,
            "stale": bool(
                query_result.get("stale")
            ),
            "target": self.target_label,
            "entity_id": self.locked_entity_id,
            "lock_expired": False,
            "recovery_direction": (
                recovery_direction
            ),
            "prediction": prediction,
            "predicted_cx": prediction.get(
                "predicted_cx"
            ),
            "predicted_area": prediction.get(
                "predicted_area"
            ),
            "predicted_direction": prediction.get(
                "predicted_direction"
            ),
            "horizontal_velocity": prediction.get(
                "horizontal_velocity"
            ),
            "prediction_horizon_seconds": (
                prediction.get(
                    "prediction_horizon_seconds"
                )
            ),
            "last_seen_direction": (
                self.last_seen_direction
            ),
            "last_visible_at": (
                self.last_visible_at
            ),
            "lost_since": self.lost_since,
            "lost_age_seconds": (
                lost_age_seconds
            ),
            "recovery_timeout_seconds": (
                self.recovery_timeout_seconds
            ),
            "reason": (
                "Locked target is temporarily unavailable. "
                + (
                    "Holding position near the predicted "
                    "target location."
                    if recovery_direction
                    == self.DIRECTION_CENTER
                    else (
                        "Recovering toward "
                        f"{recovery_direction.lower()}."
                        if recovery_direction
                        else
                        "No reliable recovery direction "
                        "is available."
                    )
                )
            ),
        }

    def resolve(
        self,
        mission_id: Optional[str],
        target_label: str,
    ) -> Dict[str, Any]:
        self.start_mission(
            mission_id=mission_id,
            target_label=target_label,
        )

        if (
            self.tracking_mode
            == self.MODE_WAITING_FOR_IDENTITY
        ):
            return self._resolve_waiting_identity()

        if self.locked_entity_id:
            observation = (
                self._query_locked_target()
            )

            if observation.get("found"):
                self.tracking_mode = self.MODE_LOCKED
                self._remember_visible_observation(
                    observation
                )
                return observation

            if observation.get("identity_mismatch"):
                self.tracking_mode = self.MODE_LOCKED
                return observation

            recovery = self._recovery_result(
                observation
            )

            return recovery

        acquisition = (
            self.world_model
            .find_latest_entity_by_label(
                self.target_label,
                max_age_seconds=(
                    self.max_age_seconds
                ),
                refresh=True,
            )
        )

        if acquisition.get("found"):
            self._lock_from_observation(
                acquisition
            )

        return acquisition

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tracking_mode": self.tracking_mode,
            "locked_entity_id": (
                self.locked_entity_id
            ),
            "locked_identity_id": (
                self.locked_identity_id
            ),
            "last_entity_migration": (
                dict(self.last_entity_migration)
                if self.last_entity_migration
                else None
            ),
            "locked_since": self.locked_since,
            "last_visible_at": (
                self.last_visible_at
            ),
            "lost_since": self.lost_since,
            "waiting_since": self.waiting_since,
            "waiting_age_seconds": (
                self._age_seconds(self.waiting_since)
                if self.waiting_since
                else None
            ),
            "last_seen_direction": (
                self.last_seen_direction
            ),
            "recovery_timeout_seconds": (
                self.recovery_timeout_seconds
            ),
            "prediction": (
                self.prediction_tracker.snapshot()
            ),
        }
