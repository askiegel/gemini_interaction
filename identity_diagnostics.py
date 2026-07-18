#!/usr/bin/env python3

"""
Structured identity-matching diagnostics.

This module contains data structures only. It does not decide which identity
wins and does not alter PersonIdentityManager matching behavior.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateDiagnostic:
    """
    Score breakdown for one existing identity candidate.
    """

    identity_id: str

    age_seconds: Optional[float]
    center_distance_ratio: Optional[float]

    center_score: float
    iou_score: float
    area_score: float
    temporal_score: float
    transient_entity_bonus: float

    weak_spatial_match_penalty_applied: bool
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        for key in (
            "age_seconds",
            "center_distance_ratio",
            "center_score",
            "iou_score",
            "area_score",
            "temporal_score",
            "transient_entity_bonus",
            "final_score",
        ):
            value = data.get(key)

            if value is not None:
                data[key] = round(float(value), 4)

        return data


@dataclass
class MatchDiagnostic:
    """
    Explanation of one identity-assignment decision.
    """

    candidates: List[CandidateDiagnostic] = field(
        default_factory=list
    )

    winner_identity_id: Optional[str] = None
    runner_up_identity_id: Optional[str] = None

    best_score: float = 0.0
    second_score: float = 0.0
    score_margin: float = 0.0

    minimum_match_score: float = 0.0
    minimum_score_margin: float = 0.0

    ambiguous: bool = False
    decision: str = "UNKNOWN"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "winner_identity_id": self.winner_identity_id,
            "runner_up_identity_id": (
                self.runner_up_identity_id
            ),
            "best_score": round(
                float(self.best_score),
                4,
            ),
            "second_score": round(
                float(self.second_score),
                4,
            ),
            "score_margin": round(
                float(self.score_margin),
                4,
            ),
            "minimum_match_score": round(
                float(self.minimum_match_score),
                4,
            ),
            "minimum_score_margin": round(
                float(self.minimum_score_margin),
                4,
            ),
            "ambiguous": bool(self.ambiguous),
            "decision": self.decision,
            "reason": self.reason,
        }
