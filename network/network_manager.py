#!/usr/bin/env python3

import shutil
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class NetworkManagerError(RuntimeError):
    """Raised when local NetworkManager information cannot be collected."""


def _split_escaped(value: str, separator: str = ":") -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    escaped = False

    for character in value:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == separator:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)

    if escaped:
        current.append("\\")

    parts.append("".join(current))
    return parts


def _integer(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class NetworkManager:
    """Read-only NetworkManager diagnostics for the Brain PC.

    This class intentionally exposes no connect, disconnect, delete, or
    credential-management operations. It is safe to use from an unauthenticated
    local operator dashboard because it only reads system state.
    """

    def __init__(self, command_timeout: float = 8.0):
        self.command_timeout = float(command_timeout)

    def _run_nmcli(self, *arguments: str) -> str:
        if shutil.which("nmcli") is None:
            raise NetworkManagerError(
                "nmcli is not installed. NetworkManager visibility is unavailable."
            )

        command = ["nmcli", "--terse", "--escape", "yes", *arguments]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NetworkManagerError("nmcli timed out.") from exc
        except OSError as exc:
            raise NetworkManagerError(f"Unable to execute nmcli: {exc}") from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "nmcli failed").strip()
            raise NetworkManagerError(message)

        return result.stdout.strip()

    def _device_status(self) -> List[Dict[str, Any]]:
        output = self._run_nmcli(
            "--fields",
            "DEVICE,TYPE,STATE,CONNECTION",
            "device",
            "status",
        )
        devices: List[Dict[str, Any]] = []

        for line in output.splitlines():
            if not line.strip():
                continue
            fields = _split_escaped(line)
            fields += [""] * (4 - len(fields))
            devices.append(
                {
                    "device": fields[0],
                    "type": fields[1],
                    "state": fields[2],
                    "connection": fields[3] or None,
                }
            )

        return devices

    def _saved_connections(self) -> List[Dict[str, Any]]:
        output = self._run_nmcli(
            "--fields",
            "NAME,UUID,TYPE,DEVICE",
            "connection",
            "show",
        )
        connections: List[Dict[str, Any]] = []

        for line in output.splitlines():
            if not line.strip():
                continue
            fields = _split_escaped(line)
            fields += [""] * (4 - len(fields))
            connections.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "type": fields[2],
                    "device": fields[3] or None,
                    "active": bool(fields[3] and fields[3] != "--"),
                }
            )

        return connections

    def _wifi_networks(self, rescan: bool) -> List[Dict[str, Any]]:
        output = self._run_nmcli(
            "--fields",
            "IN-USE,SSID,SIGNAL,SECURITY,FREQ,CHAN,RATE,BARS",
            "device",
            "wifi",
            "list",
            "--rescan",
            "yes" if rescan else "no",
        )
        networks_by_key: Dict[str, Dict[str, Any]] = {}

        for line in output.splitlines():
            if not line.strip():
                continue
            fields = _split_escaped(line)
            fields += [""] * (8 - len(fields))
            ssid = fields[1] or "Hidden network"
            network = {
                "in_use": fields[0].strip() == "*",
                "ssid": ssid,
                "signal": _integer(fields[2]),
                "security": fields[3] or "Open",
                "frequency_mhz": _integer(fields[4]),
                "channel": _integer(fields[5]),
                "rate": fields[6] or None,
                "bars": fields[7] or None,
            }
            key = f"{ssid}|{network['security']}"
            previous = networks_by_key.get(key)
            if previous is None or (network["signal"] or -1) > (previous["signal"] or -1):
                networks_by_key[key] = network

        networks = list(networks_by_key.values())
        networks.sort(
            key=lambda item: (not item["in_use"], -(item["signal"] or -1), item["ssid"].lower())
        )
        return networks

    def collect(self, rescan: bool = False) -> Dict[str, Any]:
        devices = self._device_status()
        saved_connections = self._saved_connections()
        wifi_networks = self._wifi_networks(bool(rescan))

        connected_devices = [
            item for item in devices if item.get("state") in {"connected", "connecting"}
        ]
        active_wifi = next((item for item in wifi_networks if item["in_use"]), None)
        managed_devices = [
            item for item in devices if item.get("state") not in {"unmanaged", "unavailable"}
        ]
        wifi_devices = [item for item in devices if item.get("type") == "wifi"]

        return {
            "ok": True,
            "read_only": True,
            "hostname": socket.gethostname(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "connected": bool(connected_devices),
                "active_connection": (
                    connected_devices[0].get("connection") if connected_devices else None
                ),
                "active_wifi_ssid": active_wifi.get("ssid") if active_wifi else None,
                "wifi_signal": active_wifi.get("signal") if active_wifi else None,
                "device_count": len(devices),
                "saved_connection_count": len(saved_connections),
                "visible_wifi_count": len(wifi_networks),
                "managed_device_count": len(managed_devices),
                "wifi_device_count": len(wifi_devices),
                "networkmanager_managing_interfaces": bool(managed_devices),
            },
            "devices": devices,
            "saved_connections": saved_connections,
            "wifi_networks": wifi_networks,
        }
