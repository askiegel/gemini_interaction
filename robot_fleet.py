import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

from robot_identity import RobotIdentity, get_robot_identity


DEFAULT_FLEET_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "robot_fleet.json"
)


class RobotFleetError(ValueError):
    """Raised when fleet identity configuration is invalid."""


class RobotFleet:
    def __init__(
        self,
        local_identity: RobotIdentity,
        identities: Iterable[RobotIdentity],
    ):
        if not isinstance(local_identity, RobotIdentity):
            raise TypeError(
                "local_identity must be a RobotIdentity."
            )

        by_id: Dict[str, RobotIdentity] = {}

        for identity in identities:
            if not isinstance(identity, RobotIdentity):
                raise TypeError(
                    "Fleet entries must be RobotIdentity objects."
                )

            if identity.id in by_id:
                raise RobotFleetError(
                    f"Duplicate robot ID in fleet: {identity.id}"
                )

            by_id[identity.id] = identity

        if local_identity.id not in by_id:
            by_id[local_identity.id] = local_identity

        self.local_identity = local_identity
        self._by_id = by_id

    @property
    def identities(self) -> Tuple[RobotIdentity, ...]:
        return tuple(self._by_id.values())

    @property
    def remote_identities(self) -> Tuple[RobotIdentity, ...]:
        return tuple(
            identity
            for identity in self.identities
            if identity.id != self.local_identity.id
        )

    def get(self, robot_id: str) -> RobotIdentity:
        normalized_id = str(robot_id or "").strip().lower()

        try:
            return self._by_id[normalized_id]
        except KeyError as exc:
            raise RobotFleetError(
                f"Unknown robot ID: {normalized_id or '<empty>'}"
            ) from exc

    def to_dict(self):
        return {
            "local_robot_id": self.local_identity.id,
            "robots": [
                identity.to_dict()
                for identity in self.identities
            ],
        }


def _identity_from_fleet_entry(entry) -> RobotIdentity:
    if not isinstance(entry, dict):
        raise RobotFleetError(
            "Each fleet robot entry must be a JSON object."
        )

    return RobotIdentity.from_config(
        {
            "robot": entry,
        }
    )


def load_robot_fleet(
    path=None,
    local_identity=None,
) -> RobotFleet:
    local_identity = (
        local_identity
        or get_robot_identity()
    )

    configured_path = (
        path
        or os.getenv("ROBOT_FLEET_FILE")
        or DEFAULT_FLEET_PATH
    )

    fleet_path = Path(configured_path).expanduser()

    if not fleet_path.exists():
        return RobotFleet(
            local_identity=local_identity,
            identities=[local_identity],
        )

    try:
        payload = json.loads(
            fleet_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RobotFleetError(
            f"Fleet configuration is invalid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RobotFleetError(
            "Fleet configuration must be a JSON object."
        )

    if payload.get("fleet_version") != 1:
        raise RobotFleetError(
            "Unsupported or missing fleet_version."
        )

    entries = payload.get("robots")

    if not isinstance(entries, list):
        raise RobotFleetError(
            "Fleet configuration requires a robots array."
        )

    identities = [
        _identity_from_fleet_entry(entry)
        for entry in entries
    ]

    fleet = RobotFleet(
        local_identity=local_identity,
        identities=identities,
    )

    configured_local = fleet.get(local_identity.id)

    if configured_local.to_dict() != local_identity.to_dict():
        raise RobotFleetError(
            "The local robot identity does not match its fleet entry."
        )

    return fleet
