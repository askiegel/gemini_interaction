"""Persistent system configuration for the cognitive platform."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "system_config.json"
SUPPORTED_CONFIG_VERSION = 1

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": SUPPORTED_CONFIG_VERSION,
    "robot": {
        "name": "Tony-01",
        "model": "Mini Pupper 2",
        "hostname": "minipupper",
    },
    "network": {
        "robot_ip": "192.168.68.127",
        "robot_bridge_port": 8090,
        "brain_ip": "127.0.0.1",
        "ros_domain": 42,
    },
    "vision": {
        "server_url": "http://127.0.0.1:8000/detections/latest",
    },
    "speech": {
        "provider": "browser",
    },
    "ui": {
        "theme": "dark",
        "camera_layout": "large",
    },
}


class ConfigurationError(ValueError):
    """Raised when a configuration payload is invalid."""


def _require_object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be a JSON object.")
    return value


def _require_string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ConfigurationError(f"{field_name} must not be empty.")
    return cleaned


def _require_int(
    value: Any,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{field_name} must be between {minimum} and {maximum}."
        )
    return value


def validate_config(payload: Any) -> Dict[str, Any]:
    """Validate and normalize a complete configuration document."""
    root = _require_object(payload, "configuration")

    version = _require_int(
        root.get("config_version"),
        "config_version",
        SUPPORTED_CONFIG_VERSION,
        SUPPORTED_CONFIG_VERSION,
    )

    robot = _require_object(root.get("robot"), "robot")
    network = _require_object(root.get("network"), "network")
    vision = _require_object(root.get("vision"), "vision")
    speech = _require_object(root.get("speech"), "speech")
    ui = _require_object(root.get("ui"), "ui")

    theme = _require_string(ui.get("theme"), "ui.theme").lower()
    if theme not in {"dark", "light", "system"}:
        raise ConfigurationError(
            "ui.theme must be dark, light, or system."
        )

    camera_layout = _require_string(
        ui.get("camera_layout"),
        "ui.camera_layout",
    ).lower()
    if camera_layout not in {"large", "standard"}:
        raise ConfigurationError(
            "ui.camera_layout must be large or standard."
        )

    normalized = {
        "config_version": version,
        "robot": {
            "name": _require_string(robot.get("name"), "robot.name"),
            "model": _require_string(robot.get("model"), "robot.model"),
            "hostname": _require_string(
                robot.get("hostname"),
                "robot.hostname",
            ),
        },
        "network": {
            "robot_ip": _require_string(
                network.get("robot_ip"),
                "network.robot_ip",
            ),
            "robot_bridge_port": _require_int(
                network.get("robot_bridge_port"),
                "network.robot_bridge_port",
                1,
                65535,
            ),
            "brain_ip": _require_string(
                network.get("brain_ip"),
                "network.brain_ip",
            ),
            "ros_domain": _require_int(
                network.get("ros_domain"),
                "network.ros_domain",
                0,
                232,
            ),
        },
        "vision": {
            "server_url": _require_string(
                vision.get("server_url"),
                "vision.server_url",
            ),
        },
        "speech": {
            "provider": _require_string(
                speech.get("provider"),
                "speech.provider",
            ),
        },
        "ui": {
            "theme": theme,
            "camera_layout": camera_layout,
        },
    }
    return normalized


class ConfigurationManager:
    """Thread-safe configuration loader with atomic persistence."""

    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        configured_path = path or os.getenv("SYSTEM_CONFIG_FILE")
        self.path = Path(configured_path or DEFAULT_CONFIG_PATH).expanduser()
        self._lock = threading.RLock()
        self._config: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> Dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                self._config = validate_config(copy.deepcopy(DEFAULT_CONFIG))
                self._write_atomic(self._config)
            else:
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"Invalid JSON in {self.path}: {exc.msg}."
                    ) from exc
                self._config = validate_config(payload)
            return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._config)

    def update_config(self, payload: Any) -> Dict[str, Any]:
        normalized = validate_config(payload)
        with self._lock:
            self._write_atomic(normalized)
            self._config = normalized
            return self.get_config()

    @property
    def robot_bridge_url(self) -> str:
        override = os.getenv("ROBOT_BRIDGE_URL")
        if override:
            return override.rstrip("/")
        config = self.get_config()
        network = config["network"]
        return (
            f"http://{network['robot_ip']}:"
            f"{network['robot_bridge_port']}"
        )

    def _write_atomic(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
