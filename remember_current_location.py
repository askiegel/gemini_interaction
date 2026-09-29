"""Explicit, non-motion natural-language parsing for waypoint capture."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


REMEMBER_CURRENT_LOCATION = "REMEMBER_CURRENT_LOCATION"

_NAMED_COMMAND = re.compile(
    r"^(?:remember\s+this\s+(?:place|location)|save\s+this\s+place)"
    r"\s+as\s+(.+)$",
    re.IGNORECASE,
)
_MISSING_NAME_COMMAND = re.compile(
    r"^(?:remember\s+this\s+(?:place|location)|save\s+this\s+place)"
    r"(?:\s+as)?$",
    re.IGNORECASE,
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9]+(?:[ \t]+[A-Za-z0-9]+)*$")


class RememberLocationNameError(ValueError):
    """Raised when an explicit location name cannot form an unambiguous ID."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RememberCurrentLocationIntent:
    """A parsed request to save the current validated map pose."""

    name: str
    waypoint_id: str
    intent: str = REMEMBER_CURRENT_LOCATION


def normalize_waypoint_name(name: object) -> tuple[str, str]:
    """Return display name and deterministic ID without lossy punctuation repair.

    Names must contain alphanumeric words separated only by internal
    whitespace.  The ID is lowercase with each internal whitespace run
    replaced by one underscore.
    """
    if not isinstance(name, str):
        raise RememberLocationNameError(
            "location name is invalid", reason="INVALID_NAME"
        )
    cleaned = name.strip()
    if not cleaned:
        raise RememberLocationNameError(
            "location name is required", reason="NO_NAME"
        )
    if not _SAFE_NAME.fullmatch(cleaned):
        raise RememberLocationNameError(
            "location name is invalid", reason="INVALID_NAME"
        )
    waypoint_id = re.sub(r"[ \t]+", "_", cleaned.lower())
    if not waypoint_id:
        raise RememberLocationNameError(
            "location name is invalid", reason="INVALID_NAME"
        )
    return cleaned, waypoint_id


def parse_remember_current_location(text: object) -> Optional[RememberCurrentLocationIntent]:
    """Parse only explicit remember-place commands; return ``None`` otherwise."""
    if not isinstance(text, str):
        return None
    normalized = text.strip()
    if not normalized:
        return None
    normalized = normalized.rstrip(".?!").strip()
    if _MISSING_NAME_COMMAND.fullmatch(normalized):
        raise RememberLocationNameError(
            "location name is required", reason="NO_NAME"
        )
    match = _NAMED_COMMAND.fullmatch(normalized)
    if match is None:
        return None
    name, waypoint_id = normalize_waypoint_name(match.group(1))
    return RememberCurrentLocationIntent(name=name, waypoint_id=waypoint_id)
