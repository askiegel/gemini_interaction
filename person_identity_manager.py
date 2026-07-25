#!/usr/bin/env python3

"""
Persistent person identity assignment for the Mini Pupper cognitive platform.

This module intentionally has no OpenCV, NumPy, Torch, or ONNX dependency.
The first implementation uses geometric and temporal continuity:

    - bounding-box overlap
    - center-point distance
    - apparent-size similarity
    - elapsed time since last observation

Later, appearance embeddings can be added without changing the public
identity-assignment interface.

Important distinction:

    entity_id
        A transient World Model or detector tracking identifier.

    identity_id
        A persistent human identity assigned by this manager.
"""

from dataclasses import asdict, dataclass
import json
import os
from datetime import datetime, timezone
from math import hypot
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from identity_diagnostics import (
    CandidateDiagnostic,
    MatchDiagnostic,
)


@dataclass
class PersonIdentity:
    identity_id: str
    created_at: str
    last_seen_at: str
    observation_count: int

    cx: Optional[float] = None
    cy: Optional[float] = None
    area: Optional[float] = None
    image_width: Optional[float] = None
    image_height: Optional[float] = None
    bbox: Optional[Dict[str, float]] = None

    transient_entity_id: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonIdentityManager:
    """
    Assign persistent identity IDs to person detections.

    Identity matching is conservative. A detection is matched only when the
    best candidate exceeds the minimum score and clearly beats the second-best
    candidate. Ambiguous detections receive a new identity instead of silently
    switching one person's identity to another.
    """

    DEFAULT_MAX_IDENTITY_AGE_SECONDS = 30.0
    DEFAULT_MINIMUM_MATCH_SCORE = 0.58
    DEFAULT_MINIMUM_SCORE_MARGIN = 0.08
    DEFAULT_MAX_CENTER_DISTANCE_RATIO = 0.38
    DEFAULT_MINIMUM_IOU = 0.05

    # A recently established identity may survive a brief score drop or
    # small ambiguity margin without creating another identity.
    DEFAULT_HYSTERESIS_MAX_AGE_SECONDS = 10.0
    DEFAULT_HYSTERESIS_MINIMUM_MATCH_SCORE = 0.52
    DEFAULT_HYSTERESIS_MINIMUM_OBSERVATIONS = 2

    def __init__(
        self,
        max_identity_age_seconds: float = (
            DEFAULT_MAX_IDENTITY_AGE_SECONDS
        ),
        minimum_match_score: float = (
            DEFAULT_MINIMUM_MATCH_SCORE
        ),
        minimum_score_margin: float = (
            DEFAULT_MINIMUM_SCORE_MARGIN
        ),
        max_center_distance_ratio: float = (
            DEFAULT_MAX_CENTER_DISTANCE_RATIO
        ),
        minimum_iou: float = DEFAULT_MINIMUM_IOU,
        hysteresis_max_age_seconds: float = (
            DEFAULT_HYSTERESIS_MAX_AGE_SECONDS
        ),
        hysteresis_minimum_match_score: float = (
            DEFAULT_HYSTERESIS_MINIMUM_MATCH_SCORE
        ),
        hysteresis_minimum_observations: int = (
            DEFAULT_HYSTERESIS_MINIMUM_OBSERVATIONS
        ),
    ):
        self.max_identity_age_seconds = float(
            max_identity_age_seconds
        )
        self.minimum_match_score = float(
            minimum_match_score
        )
        self.minimum_score_margin = float(
            minimum_score_margin
        )
        self.max_center_distance_ratio = float(
            max_center_distance_ratio
        )
        self.minimum_iou = float(minimum_iou)

        self.hysteresis_max_age_seconds = float(
            hysteresis_max_age_seconds
        )
        self.hysteresis_minimum_match_score = float(
            hysteresis_minimum_match_score
        )
        self.hysteresis_minimum_observations = int(
            hysteresis_minimum_observations
        )

        self.identities: Dict[str, PersonIdentity] = {}
        self.entity_bindings: Dict[str, str] = {}

        self.last_match_diagnostic: Optional[
            Dict[str, Any]
        ] = None

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
    def _float_or_none(
        value: Any,
    ) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_bbox(
        cls,
        value: Any,
    ) -> Optional[Dict[str, float]]:
        if isinstance(value, dict):
            x1 = value.get(
                "x1",
                value.get("left"),
            )
            y1 = value.get(
                "y1",
                value.get("top"),
            )
            x2 = value.get(
                "x2",
                value.get("right"),
            )
            y2 = value.get(
                "y2",
                value.get("bottom"),
            )

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
    def _normalize_detection(
        cls,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        bbox = cls._normalize_bbox(
            detection.get("bbox")
        )

        cx = cls._float_or_none(
            detection.get(
                "cx",
                detection.get("center_x"),
            )
        )

        cy = cls._float_or_none(
            detection.get(
                "cy",
                detection.get("center_y"),
            )
        )

        area = cls._float_or_none(
            detection.get("area")
        )

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

        label = str(
            detection.get("label") or ""
        ).strip().lower()

        return {
            "label": label,
            "cx": cx,
            "cy": cy,
            "area": area,
            "bbox": bbox,
            "image_width": cls._float_or_none(
                detection.get("image_width")
            ),
            "image_height": cls._float_or_none(
                detection.get("image_height")
            ),
            "confidence": float(
                detection.get("confidence") or 0.0
            ),
            "entity_id": (
                str(
                    detection.get("entity_id") or ""
                ).strip()
                or None
            ),
        }

    @staticmethod
    def _bbox_iou(
        first: Optional[Dict[str, float]],
        second: Optional[Dict[str, float]],
    ) -> float:
        if first is None or second is None:
            return 0.0

        intersection_x1 = max(
            first["x1"],
            second["x1"],
        )
        intersection_y1 = max(
            first["y1"],
            second["y1"],
        )
        intersection_x2 = min(
            first["x2"],
            second["x2"],
        )
        intersection_y2 = min(
            first["y2"],
            second["y2"],
        )

        intersection_width = max(
            0.0,
            intersection_x2 - intersection_x1,
        )
        intersection_height = max(
            0.0,
            intersection_y2 - intersection_y1,
        )
        intersection_area = (
            intersection_width
            * intersection_height
        )

        first_area = max(
            0.0,
            first["x2"] - first["x1"],
        ) * max(
            0.0,
            first["y2"] - first["y1"],
        )

        second_area = max(
            0.0,
            second["x2"] - second["x1"],
        ) * max(
            0.0,
            second["y2"] - second["y1"],
        )

        union_area = (
            first_area
            + second_area
            - intersection_area
        )

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def _area_similarity(
        first_area: Optional[float],
        second_area: Optional[float],
    ) -> float:
        if (
            first_area is None
            or second_area is None
            or first_area <= 0.0
            or second_area <= 0.0
        ):
            return 0.0

        return min(
            first_area,
            second_area,
        ) / max(
            first_area,
            second_area,
        )

    @staticmethod
    def _center_distance_ratio(
        detection: Dict[str, Any],
        identity: PersonIdentity,
    ) -> Optional[float]:
        if None in (
            detection.get("cx"),
            detection.get("cy"),
            identity.cx,
            identity.cy,
        ):
            return None

        image_width = (
            detection.get("image_width")
            or identity.image_width
        )
        image_height = (
            detection.get("image_height")
            or identity.image_height
        )

        if (
            image_width is None
            or image_height is None
            or image_width <= 0.0
            or image_height <= 0.0
        ):
            return None

        frame_diagonal = hypot(
            image_width,
            image_height,
        )

        if frame_diagonal <= 0.0:
            return None

        distance = hypot(
            detection["cx"] - identity.cx,
            detection["cy"] - identity.cy,
        )

        return distance / frame_diagonal

    def _match_diagnostic(
        self,
        detection: Dict[str, Any],
        identity: PersonIdentity,
    ) -> CandidateDiagnostic:
        """
        Calculate the existing match score and expose its components.

        The calculations and weights in this method intentionally match the
        original _match_score() implementation.
        """
        age = self._age_seconds(
            identity.last_seen_at
        )

        if (
            age is None
            or age > self.max_identity_age_seconds
        ):
            return CandidateDiagnostic(
                identity_id=identity.identity_id,
                age_seconds=age,
                center_distance_ratio=None,
                center_score=0.0,
                iou_score=0.0,
                area_score=0.0,
                temporal_score=0.0,
                transient_entity_bonus=0.0,
                weak_spatial_match_penalty_applied=False,
                final_score=0.0,
            )

        iou = self._bbox_iou(
            detection.get("bbox"),
            identity.bbox,
        )

        center_distance_ratio = (
            self._center_distance_ratio(
                detection,
                identity,
            )
        )

        if center_distance_ratio is None:
            center_score = 0.0
        elif (
            center_distance_ratio
            > self.max_center_distance_ratio
        ):
            center_score = 0.0
        else:
            center_score = max(
                0.0,
                1.0
                - (
                    center_distance_ratio
                    / self.max_center_distance_ratio
                ),
            )

        area_score = self._area_similarity(
            detection.get("area"),
            identity.area,
        )

        temporal_score = max(
            0.0,
            1.0
            - (
                age
                / self.max_identity_age_seconds
            ),
        )

        transient_entity_bonus = 0.0

        if (
            detection.get("entity_id")
            and identity.transient_entity_id
            and detection["entity_id"]
            == identity.transient_entity_id
        ):
            transient_entity_bonus = 0.15

        score = (
            0.38 * center_score
            + 0.27 * iou
            + 0.20 * area_score
            + 0.15 * temporal_score
            + transient_entity_bonus
        )

        weak_spatial_match_penalty_applied = (
            iou < self.minimum_iou
            and center_score < 0.60
        )

        if weak_spatial_match_penalty_applied:
            score *= 0.45

        final_score = min(
            1.0,
            max(0.0, score),
        )

        return CandidateDiagnostic(
            identity_id=identity.identity_id,
            age_seconds=age,
            center_distance_ratio=(
                center_distance_ratio
            ),
            center_score=center_score,
            iou_score=iou,
            area_score=area_score,
            temporal_score=temporal_score,
            transient_entity_bonus=(
                transient_entity_bonus
            ),
            weak_spatial_match_penalty_applied=(
                weak_spatial_match_penalty_applied
            ),
            final_score=final_score,
        )

    def _match_score(
        self,
        detection: Dict[str, Any],
        identity: PersonIdentity,
    ) -> float:
        """
        Preserve the original numeric-score interface.
        """
        return self._match_diagnostic(
            detection,
            identity,
        ).final_score

    def _new_identity_id(self) -> str:
        return (
            f"person-identity-"
            f"{uuid4().hex[:8]}"
        )

    def _create_identity(
        self,
        detection: Dict[str, Any],
    ) -> PersonIdentity:
        timestamp = self._now_iso()

        identity = PersonIdentity(
            identity_id=self._new_identity_id(),
            created_at=timestamp,
            last_seen_at=timestamp,
            observation_count=1,
            cx=detection.get("cx"),
            cy=detection.get("cy"),
            area=detection.get("area"),
            image_width=detection.get(
                "image_width"
            ),
            image_height=detection.get(
                "image_height"
            ),
            bbox=detection.get("bbox"),
            transient_entity_id=detection.get(
                "entity_id"
            ),
            confidence=float(
                detection.get("confidence") or 0.0
            ),
        )

        self.identities[
            identity.identity_id
        ] = identity

        entity_id = detection.get("entity_id")

        if entity_id:
            self.entity_bindings[
                entity_id
            ] = identity.identity_id

        return identity

    def _update_identity(
        self,
        identity: PersonIdentity,
        detection: Dict[str, Any],
    ) -> PersonIdentity:
        identity.last_seen_at = self._now_iso()
        identity.observation_count += 1

        for attribute in (
            "cx",
            "cy",
            "area",
            "image_width",
            "image_height",
            "bbox",
        ):
            value = detection.get(attribute)

            if value is not None:
                setattr(
                    identity,
                    attribute,
                    value,
                )

        if detection.get("entity_id"):
            entity_id = detection["entity_id"]

            self.entity_bindings[
                entity_id
            ] = identity.identity_id

            identity.transient_entity_id = (
                entity_id
            )

        identity.confidence = float(
            detection.get("confidence") or 0.0
        )

        return identity

    def assign_identity(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assign a persistent identity to one normalized person detection.

        Returns the original detection with:

            identity_id
            identity_match_score
            identity_status
            identity_ambiguous
        """
        normalized = self._normalize_detection(
            detection
        )

        result = dict(detection)

        if normalized["label"] != "person":
            result.update(
                {
                    "identity_id": None,
                    "identity_match_score": 0.0,
                    "identity_status": (
                        "NOT_A_PERSON"
                    ),
                    "identity_ambiguous": False,
                    "identity_diagnostics": {
                        "candidates": [],
                        "winner_identity_id": None,
                        "runner_up_identity_id": None,
                        "best_score": 0.0,
                        "second_score": 0.0,
                        "score_margin": 0.0,
                        "minimum_match_score": round(
                            self.minimum_match_score,
                            4,
                        ),
                        "minimum_score_margin": round(
                            self.minimum_score_margin,
                            4,
                        ),
                        "ambiguous": False,
                        "decision": "NOT_A_PERSON",
                        "reason": (
                            "Identity assignment applies only "
                            "to person detections."
                        ),
                    },
                }
            )
            self.last_match_diagnostic = (
                result["identity_diagnostics"]
            )
            return result

        current_entity_id = normalized.get(
            "entity_id"
        )

        bound_identity_id = (
            self.entity_bindings.get(
                current_entity_id
            )
            if current_entity_id
            else None
        )

        bound_identity = (
            self.identities.get(
                bound_identity_id
            )
            if bound_identity_id
            else None
        )

        previous_identity_ids = (
            [bound_identity_id]
            if bound_identity_id
            else []
        )

        scored_candidates = []

        if bound_identity is not None:
            # Exact registry-entity continuity is deterministic ownership
            # evidence. Score only the bound identity so unrelated historical
            # candidates cannot create an ambiguity-driven identity switch.
            diagnostic = self._match_diagnostic(
                normalized,
                bound_identity,
            )

            scored_candidates.append(
                (diagnostic, bound_identity)
            )

        else:
            for identity in self.identities.values():
                diagnostic = self._match_diagnostic(
                    normalized,
                    identity,
                )

                if diagnostic.final_score > 0.0:
                    scored_candidates.append(
                        (diagnostic, identity)
                    )

        scored_candidates.sort(
            key=lambda item: item[0].final_score,
            reverse=True,
        )

        best_score = (
            scored_candidates[0][0].final_score
            if scored_candidates
            else 0.0
        )

        second_score = (
            scored_candidates[1][0].final_score
            if len(scored_candidates) > 1
            else 0.0
        )

        score_margin = (
            best_score - second_score
        )

        ambiguous = (
            len(scored_candidates) > 1
            and score_margin
            < self.minimum_score_margin
        )

        winner_identity_id = (
            scored_candidates[0][1].identity_id
            if scored_candidates
            else None
        )

        runner_up_identity_id = (
            scored_candidates[1][1].identity_id
            if len(scored_candidates) > 1
            else None
        )

        best_diagnostic = (
            scored_candidates[0][0]
            if scored_candidates
            else None
        )
        best_identity = (
            scored_candidates[0][1]
            if scored_candidates
            else None
        )

        hysteresis_match = (
            best_diagnostic is not None
            and best_identity is not None
            and best_score
            >= self.hysteresis_minimum_match_score
            and best_diagnostic.age_seconds is not None
            and best_diagnostic.age_seconds
            <= self.hysteresis_max_age_seconds
            and best_identity.observation_count
            >= self.hysteresis_minimum_observations
        )

        if bound_identity is not None:
            identity = self._update_identity(
                bound_identity,
                normalized,
            )
            status = "MATCHED"
            decision_reason = (
                "The World Model entity ID is already bound "
                "to this persistent identity."
            )
        elif (
            scored_candidates
            and best_score
            >= self.minimum_match_score
            and not ambiguous
        ):
            identity = self._update_identity(
                best_identity,
                normalized,
            )
            status = "MATCHED"
            decision_reason = (
                "Best candidate met the minimum match score "
                "and exceeded the required score margin."
            )
        elif hysteresis_match:
            identity = self._update_identity(
                best_identity,
                normalized,
            )
            status = "MATCHED_HYSTERESIS"
            decision_reason = (
                "A recently established identity was retained "
                "through a brief score drop or ambiguous margin."
            )
        else:
            identity = self._create_identity(
                normalized
            )

            status = (
                "NEW_AMBIGUOUS"
                if ambiguous
                else "NEW"
            )

            if ambiguous:
                decision_reason = (
                    "The best and second-best candidates were "
                    "separated by less than the required score margin."
                )
            elif not scored_candidates:
                decision_reason = (
                    "No eligible existing identity candidate "
                    "produced a positive score."
                )
            else:
                decision_reason = (
                    "The best candidate score was below both the "
                    "normal and hysteresis match thresholds."
                )

        match_diagnostic = MatchDiagnostic(
            candidates=[
                candidate
                for candidate, _identity
                in scored_candidates
            ],
            winner_identity_id=winner_identity_id,
            runner_up_identity_id=(
                runner_up_identity_id
            ),
            best_score=best_score,
            second_score=second_score,
            score_margin=score_margin,
            minimum_match_score=(
                self.minimum_match_score
            ),
            minimum_score_margin=(
                self.minimum_score_margin
            ),
            ambiguous=ambiguous,
            decision=status,
            reason=decision_reason,
        )

        diagnostic_dict = match_diagnostic.to_dict()
        self.last_match_diagnostic = diagnostic_dict

        result.update(
            {
                "identity_id": (
                    identity.identity_id
                ),
                "identity_match_score": round(
                    best_score,
                    4,
                ),
                "identity_status": status,
                "identity_ambiguous": (
                    ambiguous
                ),
                "identity_diagnostics": (
                    diagnostic_dict
                ),
            }
        )

        debug_enabled = str(
            os.getenv("IDENTITY_DEBUG", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if debug_enabled:
            decision_telemetry = {
                **diagnostic_dict,
                "entity_id": current_entity_id,
                "previous_identity_ids": (
                    previous_identity_ids
                ),
                "assigned_identity_id": (
                    identity.identity_id
                ),
                "best_candidate_identity_id": (
                    winner_identity_id
                ),
                "match_score": round(
                    best_score,
                    4,
                ),
                "threshold": round(
                    self.minimum_match_score,
                    4,
                ),
                "hysteresis_threshold": round(
                    self.hysteresis_minimum_match_score,
                    4,
                ),
                "runner_up_score": round(
                    second_score,
                    4,
                ),
                "entity_binding_match": (
                    bound_identity is not None
                ),
                "bound_identity_id": (
                    bound_identity_id
                ),
            }

            print(
                "[IDENTITY_DIAGNOSTIC] "
                + json.dumps(
                    decision_telemetry,
                    sort_keys=True,
                )
            )

        attributes = dict(
            result.get("attributes") or {}
        )

        attributes.update(
            {
                "identity_id": (
                    identity.identity_id
                ),
                "identity_match_score": round(
                    best_score,
                    4,
                ),
                "identity_status": status,
                "identity_ambiguous": (
                    ambiguous
                ),
            }
        )

        result["attributes"] = attributes

        return result

    def assign_identities(
        self,
        detections: Iterable[
            Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        """
        Assign identities to a frame of detections.

        Each existing identity may be assigned at most once in a frame.
        This prevents two simultaneous detections from receiving the same
        identity solely because both are near the previous position.
        """
        results = []
        assigned_identity_ids = set()

        for detection in detections:
            result = self.assign_identity(
                detection
            )

            identity_id = result.get(
                "identity_id"
            )

            if (
                identity_id
                and identity_id
                in assigned_identity_ids
            ):
                normalized = (
                    self._normalize_detection(
                        detection
                    )
                )
                identity = self._create_identity(
                    normalized
                )

                result["identity_id"] = (
                    identity.identity_id
                )
                result[
                    "identity_match_score"
                ] = 0.0
                result["identity_status"] = (
                    "NEW_FRAME_CONFLICT"
                )
                result[
                    "identity_ambiguous"
                ] = True

                frame_diagnostic = dict(
                    result.get(
                        "identity_diagnostics"
                    )
                    or {}
                )
                frame_diagnostic.update(
                    {
                        "decision": (
                            "NEW_FRAME_CONFLICT"
                        ),
                        "ambiguous": True,
                        "reason": (
                            "The selected identity had already "
                            "been assigned to another detection "
                            "in the same frame."
                        ),
                    }
                )
                result["identity_diagnostics"] = (
                    frame_diagnostic
                )
                self.last_match_diagnostic = (
                    frame_diagnostic
                )

                attributes = dict(
                    result.get("attributes")
                    or {}
                )
                attributes.update(
                    {
                        "identity_id": (
                            identity.identity_id
                        ),
                        "identity_match_score": (
                            0.0
                        ),
                        "identity_status": (
                            "NEW_FRAME_CONFLICT"
                        ),
                        "identity_ambiguous": (
                            True
                        ),
                    }
                )
                result["attributes"] = (
                    attributes
                )

                identity_id = (
                    identity.identity_id
                )

            if identity_id:
                assigned_identity_ids.add(
                    identity_id
                )

            results.append(result)

        return results

    def get_identity(
        self,
        identity_id: str,
    ) -> Optional[Dict[str, Any]]:
        identity = self.identities.get(
            identity_id
        )

        if identity is None:
            return None

        return identity.to_dict()

    def get_identities(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            identity.to_dict()
            for identity in self.identities.values()
        ]

    def prune_expired(
        self,
        maximum_age_seconds: Optional[
            float
        ] = None,
    ) -> List[str]:
        maximum_age = float(
            maximum_age_seconds
            if maximum_age_seconds is not None
            else self.max_identity_age_seconds
        )

        removed = []

        for identity_id, identity in list(
            self.identities.items()
        ):
            age = self._age_seconds(
                identity.last_seen_at
            )

            if (
                age is not None
                and age > maximum_age
            ):
                removed.append(identity_id)
                del self.identities[
                    identity_id
                ]

        if removed:
            removed_set = set(removed)

            self.entity_bindings = {
                entity_id: identity_id
                for entity_id, identity_id
                in self.entity_bindings.items()
                if identity_id not in removed_set
            }

        return removed

    def reset(self):
        self.identities.clear()
        self.entity_bindings.clear()
        self.last_match_diagnostic = None
