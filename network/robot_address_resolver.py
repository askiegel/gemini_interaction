"""Resolve and verify the Mini Pupper network address."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


EXPECTED_SERVICE = "mini_pupper_robot_bridge"
EXPECTED_ROBOT = "mini_pupper_2"


class RobotAddressResolutionError(RuntimeError):
    """Raised when no usable Mini Pupper address can be resolved."""


@dataclass(frozen=True)
class RobotAddressResolution:
    """Result of resolving the configured Mini Pupper host."""

    configured_host: str
    address: str
    source: str


def _ipv4_literal(value: str) -> Optional[str]:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return None

    if parsed.version != 4:
        return None

    return str(parsed)


def resolve_robot_address(
    configured_host: str,
    fallback_ip: Optional[str] = None,
    *,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
) -> RobotAddressResolution:
    """
    Resolve the configured robot host to one stable IPv4 address.

    A literal IPv4 override is accepted directly. A configured fallback is
    used only when hostname resolution fails or returns no IPv4 candidates.
    """
    host = str(configured_host or "").strip()

    if not host:
        raise RobotAddressResolutionError(
            "The configured Mini Pupper hostname is empty."
        )

    literal = _ipv4_literal(host)
    if literal:
        return RobotAddressResolution(
            configured_host=host,
            address=literal,
            source="configured_ipv4",
        )

    try:
        records = getaddrinfo(
            host,
            None,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        records = []

    candidates = []

    for record in records:
        try:
            address = record[4][0]
        except (IndexError, TypeError):
            continue

        literal = _ipv4_literal(str(address))
        if literal and literal not in candidates:
            candidates.append(literal)

    if candidates:
        return RobotAddressResolution(
            configured_host=host,
            address=candidates[0],
            source="hostname_ipv4",
        )

    fallback = _ipv4_literal(str(fallback_ip or "").strip())

    if fallback:
        return RobotAddressResolution(
            configured_host=host,
            address=fallback,
            source="configured_fallback",
        )

    raise RobotAddressResolutionError(
        f"Unable to resolve Mini Pupper hostname {host!r}, "
        "and no valid IPv4 fallback is configured."
    )


def is_expected_robot_bridge(
    payload: Any,
) -> bool:
    """Return True only for the expected Mini Pupper Robot Bridge identity."""
    if not isinstance(payload, Mapping):
        return False

    return (
        payload.get("ok") is True
        and payload.get("service") == EXPECTED_SERVICE
        and payload.get("robot") == EXPECTED_ROBOT
    )
