"""Resolve, discover, and verify the Mini Pupper network address."""

from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
)


EXPECTED_SERVICE = "mini_pupper_robot_bridge"
EXPECTED_ROBOT = "mini_pupper_2"


class RobotAddressResolutionError(RuntimeError):
    """Raised when no usable Mini Pupper address can be resolved."""


@dataclass(frozen=True)
class RobotAddressResolution:
    """Result of resolving or discovering the Mini Pupper host."""

    configured_host: str
    address: str
    source: str


def _ipv4_literal(value: str) -> Optional[str]:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError:
        return None

    if (
        parsed.version != 4
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_unspecified
    ):
        return None

    return str(parsed)


def is_expected_robot_bridge(payload: Any) -> bool:
    """Return True only for the expected Mini Pupper Robot Bridge."""
    if not isinstance(payload, Mapping):
        return False

    return (
        payload.get("ok") is True
        and payload.get("service") == EXPECTED_SERVICE
        and payload.get("robot") == EXPECTED_ROBOT
    )


def discover_neighbor_ipv4_candidates(
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> list[str]:
    """
    Return usable IPv4 addresses already known to the WSL neighbor table.

    This reads existing neighbor information. It does not scan a subnet.
    """
    try:
        result = run_command(
            ["ip", "-4", "neighbor", "show"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if result.returncode != 0:
        return []

    candidates = []

    for line in result.stdout.splitlines():
        fields = line.split()

        if not fields:
            continue

        states = {field.upper() for field in fields}

        if states & {"FAILED", "INCOMPLETE", "NOARP"}:
            continue

        address = _ipv4_literal(fields[0])

        if address and address not in candidates:
            candidates.append(address)

    return candidates


def probe_expected_robot_bridge(
    address: str,
    *,
    port: int = 8090,
    timeout: float = 0.75,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Verify that an IPv4 candidate is the expected Robot Bridge."""
    candidate = _ipv4_literal(address)

    if not candidate:
        return False

    request = urllib.request.Request(
        f"http://{candidate}:{int(port)}/status",
        method="GET",
    )

    try:
        with urlopen(
            request,
            timeout=float(timeout),
        ) as response:
            if not 200 <= response.status < 300:
                return False

            body = response.read(65536)
            payload = json.loads(body.decode("utf-8"))
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ):
        return False

    return is_expected_robot_bridge(payload)


def resolve_robot_address(
    configured_host: str,
    fallback_ip: Optional[str] = None,
    *,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
    neighbor_candidates: Optional[
        Callable[[], Iterable[str]]
    ] = None,
    bridge_probe: Optional[Callable[[str], bool]] = None,
) -> RobotAddressResolution:
    """
    Resolve or discover one stable Mini Pupper IPv4 address.

    Resolution order:

    1. Configured literal IPv4
    2. Configured hostname through IPv4 name resolution
    3. Known neighbor candidates verified by Robot Bridge identity
    4. Configured compatibility fallback
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

    resolved_candidates = []

    for record in records:
        try:
            address = record[4][0]
        except (IndexError, TypeError):
            continue

        candidate = _ipv4_literal(str(address))

        if candidate and candidate not in resolved_candidates:
            resolved_candidates.append(candidate)

    if resolved_candidates:
        return RobotAddressResolution(
            configured_host=host,
            address=resolved_candidates[0],
            source="hostname_ipv4",
        )

    if neighbor_candidates is not None and bridge_probe is not None:
        try:
            known_neighbors = neighbor_candidates()
        except Exception:
            known_neighbors = []

        seen = set()

        for address in known_neighbors:
            candidate = _ipv4_literal(str(address))

            if not candidate or candidate in seen:
                continue

            seen.add(candidate)

            try:
                verified = bridge_probe(candidate)
            except Exception:
                verified = False

            if verified:
                return RobotAddressResolution(
                    configured_host=host,
                    address=candidate,
                    source="verified_neighbor",
                )

    fallback = _ipv4_literal(str(fallback_ip or "").strip())

    if fallback:
        return RobotAddressResolution(
            configured_host=host,
            address=fallback,
            source="configured_fallback",
        )

    raise RobotAddressResolutionError(
        f"Unable to resolve or discover Mini Pupper host {host!r}, "
        "and no valid IPv4 fallback is configured."
    )
