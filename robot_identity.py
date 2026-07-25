from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple


class RobotIdentityError(ValueError):
    """Raised when robot identity configuration is missing or invalid."""


def _normalize_required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()

    if not text:
        raise RobotIdentityError(
            f"Robot identity field '{field_name}' must be a non-empty string."
        )

    return text


def _normalize_robot_id(value: Any) -> str:
    robot_id = _normalize_required_text(value, "id").lower()

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "-_"
    )

    if any(character not in allowed for character in robot_id):
        raise RobotIdentityError(
            "Robot identity field 'id' may contain only lowercase letters, "
            "numbers, hyphens, and underscores."
        )

    return robot_id


def _normalize_aliases(
    values: Iterable[Any],
    display_name: str,
    robot_id: str,
) -> Tuple[str, ...]:
    aliases = []
    seen = set()

    candidates = list(values or [])
    candidates.extend([display_name, robot_id])

    for value in candidates:
        alias = " ".join(
            str(value or "")
            .strip()
            .lower()
            .split()
        )

        if not alias or alias in seen:
            continue

        seen.add(alias)
        aliases.append(alias)

    if not aliases:
        raise RobotIdentityError(
            "Robot identity requires at least one voice alias."
        )

    return tuple(aliases)


@dataclass(frozen=True)
class RobotIdentity:
    robot_id: str
    display_name: str
    voice_aliases: Tuple[str, ...]
    model: str
    role: str
    hostname: str
    platform_version: str

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "RobotIdentity":
        if not isinstance(config, dict):
            raise RobotIdentityError(
                "System configuration must be a dictionary."
            )

        robot = config.get("robot")

        if not isinstance(robot, dict):
            raise RobotIdentityError(
                "System configuration requires a 'robot' object."
            )

        robot_id = _normalize_robot_id(
            robot.get("id")
        )

        display_name = _normalize_required_text(
            robot.get("name"),
            "name",
        )

        return cls(
            robot_id=robot_id,
            display_name=display_name,
            voice_aliases=_normalize_aliases(
                robot.get("voice_aliases", []),
                display_name=display_name,
                robot_id=robot_id,
            ),
            model=_normalize_required_text(
                robot.get("model"),
                "model",
            ),
            role=_normalize_required_text(
                robot.get("role", "primary"),
                "role",
            ).lower(),
            hostname=_normalize_required_text(
                robot.get("hostname"),
                "hostname",
            ),
            platform_version=_normalize_required_text(
                robot.get("platform_version", "1.0.0"),
                "platform_version",
            ),
        )

    @property
    def id(self) -> str:
        return self.robot_id

    @property
    def name(self) -> str:
        return self.display_name

    def matches_alias(self, value: str) -> bool:
        normalized = " ".join(
            str(value or "")
            .strip()
            .lower()
            .split()
        )

        return normalized in self.voice_aliases

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["id"] = data.pop("robot_id")
        data["name"] = data.pop("display_name")
        data["voice_aliases"] = list(
            data["voice_aliases"]
        )

        return data


def get_robot_identity(
    config: Dict[str, Any] = None,
) -> RobotIdentity:
    if config is None:
        from config.config_manager import ConfigurationManager

        config = ConfigurationManager().get_config()

    return RobotIdentity.from_config(config)
