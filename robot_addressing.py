import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from robot_identity import RobotIdentity


BROADCAST_ALIASES = (
    "everybody",
    "everyone",
    "all robots",
    "both robots",
    "fleet",
)

GLOBAL_STOP_COMMANDS = {
    "stop",
    "halt",
    "freeze",
    "cancel",
    "cancel movement",
    "stop moving",
    "emergency stop",
}


def normalize_speech_text(value: str) -> str:
    text = str(value or "").strip().lower()

    text = re.sub(
        r"[^\w\s'-]",
        " ",
        text,
    )

    return " ".join(text.split())


def _remove_leading_alias(
    normalized_text: str,
    aliases: Iterable[str],
) -> Tuple[Optional[str], str]:
    ordered_aliases = sorted(
        {
            normalize_speech_text(alias)
            for alias in aliases
            if normalize_speech_text(alias)
        },
        key=len,
        reverse=True,
    )

    for alias in ordered_aliases:
        if normalized_text == alias:
            return alias, ""

        prefix = f"{alias} "

        if normalized_text.startswith(prefix):
            return (
                alias,
                normalized_text[len(prefix):].strip(),
            )

    return None, normalized_text


@dataclass(frozen=True)
class AddressedCommand:
    original_text: str
    command_text: str
    addressed_robot_id: Optional[str]
    addressed_robot_name: Optional[str]
    matched_alias: Optional[str]
    broadcast: bool
    explicitly_addressed: bool
    emergency_stop: bool

    def is_for(self, robot_id: str) -> bool:
        normalized_robot_id = normalize_speech_text(
            robot_id
        )

        return (
            self.broadcast
            or self.addressed_robot_id is None
            or self.addressed_robot_id == normalized_robot_id
        )

    def to_dict(self):
        return {
            "original_text": self.original_text,
            "command_text": self.command_text,
            "addressed_robot_id": self.addressed_robot_id,
            "addressed_robot_name": self.addressed_robot_name,
            "matched_alias": self.matched_alias,
            "broadcast": self.broadcast,
            "explicitly_addressed": self.explicitly_addressed,
            "emergency_stop": self.emergency_stop,
        }


class RobotAddressParser:
    def __init__(
        self,
        local_identity: RobotIdentity,
        known_identities: Iterable[RobotIdentity] = None,
        broadcast_aliases: Iterable[str] = BROADCAST_ALIASES,
    ):
        if not isinstance(local_identity, RobotIdentity):
            raise TypeError(
                "local_identity must be a RobotIdentity."
            )

        identities = [
            local_identity,
            *(known_identities or []),
        ]

        unique_identities = {}

        for identity in identities:
            if not isinstance(identity, RobotIdentity):
                raise TypeError(
                    "known_identities must contain RobotIdentity objects."
                )

            unique_identities[identity.id] = identity

        self.local_identity = local_identity
        self.known_identities = tuple(
            unique_identities.values()
        )

        self.broadcast_aliases = tuple(
            normalize_speech_text(alias)
            for alias in broadcast_aliases
            if normalize_speech_text(alias)
        )

    def parse(self, user_text: str) -> AddressedCommand:
        original_text = str(user_text or "").strip()

        if not original_text:
            raise ValueError(
                "Voice command text must not be empty."
            )

        normalized_text = normalize_speech_text(
            original_text
        )

        matched_broadcast_alias, command_text = (
            _remove_leading_alias(
                normalized_text,
                self.broadcast_aliases,
            )
        )

        if matched_broadcast_alias is not None:
            command_text = command_text or "stop"

            return AddressedCommand(
                original_text=original_text,
                command_text=command_text,
                addressed_robot_id=None,
                addressed_robot_name=None,
                matched_alias=matched_broadcast_alias,
                broadcast=True,
                explicitly_addressed=True,
                emergency_stop=(
                    command_text in GLOBAL_STOP_COMMANDS
                ),
            )

        alias_matches = []

        for identity in self.known_identities:
            for alias in identity.voice_aliases:
                normalized_alias = normalize_speech_text(
                    alias
                )

                if normalized_alias:
                    alias_matches.append(
                        (
                            normalized_alias,
                            identity,
                        )
                    )

        alias_matches.sort(
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for alias, identity in alias_matches:
            matched_alias, command_text = (
                _remove_leading_alias(
                    normalized_text,
                    [alias],
                )
            )

            if matched_alias is None:
                continue

            return AddressedCommand(
                original_text=original_text,
                command_text=command_text,
                addressed_robot_id=identity.id,
                addressed_robot_name=identity.name,
                matched_alias=matched_alias,
                broadcast=False,
                explicitly_addressed=True,
                emergency_stop=(
                    command_text in GLOBAL_STOP_COMMANDS
                ),
            )

        emergency_stop = (
            normalized_text in GLOBAL_STOP_COMMANDS
        )

        return AddressedCommand(
            original_text=original_text,
            command_text=normalized_text,
            addressed_robot_id=None,
            addressed_robot_name=None,
            matched_alias=None,
            broadcast=emergency_stop,
            explicitly_addressed=False,
            emergency_stop=emergency_stop,
        )
