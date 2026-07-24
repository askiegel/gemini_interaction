#!/usr/bin/env python3

import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class NetworkManagerError(RuntimeError):
    """Raised when network information cannot be collected."""


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


def _integer(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clean_windows_output(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _windows_value(line: str) -> Optional[str]:
    if ":" not in line:
        return None
    return line.split(":", 1)[1].strip()


class NetworkManager:
    """Cross-platform read-only network diagnostics.

    Native Linux uses NetworkManager through nmcli.

    WSL uses the Windows Wi-Fi subsystem through powershell.exe and netsh.
    The public response contract remains the same for the operator webpage.
    """

    def __init__(self, command_timeout: float = 12.0):
        self.command_timeout = float(command_timeout)
        self.platform_backend = self._detect_backend()

    @staticmethod
    def _is_wsl() -> bool:
        if os.environ.get("WSL_DISTRO_NAME"):
            return True

        try:
            release = platform.uname().release.lower()
        except Exception:
            release = ""

        return "microsoft" in release or "wsl" in release

    def _detect_backend(self) -> str:
        if self._is_wsl() and shutil.which("powershell.exe"):
            return "windows_wsl"

        if shutil.which("nmcli"):
            return "linux_nmcli"

        return "unsupported"

    def _run_command(self, command: List[str], timeout: Optional[float] = None) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout or self.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NetworkManagerError(
                f"Command timed out: {command[0]}"
            ) from exc
        except OSError as exc:
            raise NetworkManagerError(
                f"Unable to execute {command[0]}: {exc}"
            ) from exc

        if result.returncode != 0:
            message = (
                result.stderr
                or result.stdout
                or f"{command[0]} failed with exit code {result.returncode}"
            ).strip()
            raise NetworkManagerError(message)

        return _clean_windows_output(result.stdout)

    def _run_nmcli(self, *arguments: str) -> str:
        if shutil.which("nmcli") is None:
            raise NetworkManagerError(
                "nmcli is not installed. NetworkManager visibility is unavailable."
            )

        return self._run_command(
            ["nmcli", "--terse", "--escape", "yes", *arguments]
        )

    def _run_powershell(self, command: str) -> str:
        powershell = shutil.which("powershell.exe")

        if not powershell:
            raise NetworkManagerError(
                "powershell.exe is unavailable from WSL."
            )

        return self._run_command(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
        )

    # ------------------------------------------------------------------
    # Native Linux / NetworkManager backend
    # ------------------------------------------------------------------

    def _linux_device_status(self) -> List[Dict[str, Any]]:
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

    def _linux_saved_connections(self) -> List[Dict[str, Any]]:
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

    def _linux_wifi_networks(self, rescan: bool) -> List[Dict[str, Any]]:
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

            if previous is None or (
                network["signal"] or -1
            ) > (
                previous["signal"] or -1
            ):
                networks_by_key[key] = network

        networks = list(networks_by_key.values())

        networks.sort(
            key=lambda item: (
                not item["in_use"],
                -(item["signal"] or -1),
                item["ssid"].lower(),
            )
        )

        return networks

    def _collect_linux(self, rescan: bool) -> Dict[str, Any]:
        devices = self._linux_device_status()
        saved_connections = self._linux_saved_connections()
        wifi_networks = self._linux_wifi_networks(rescan)

        connected_devices = [
            item
            for item in devices
            if item.get("state") in {"connected", "connecting"}
        ]

        active_wifi = next(
            (item for item in wifi_networks if item["in_use"]),
            None,
        )

        managed_devices = [
            item
            for item in devices
            if item.get("state") not in {"unmanaged", "unavailable"}
        ]

        wifi_devices = [
            item
            for item in devices
            if item.get("type") == "wifi"
        ]

        return {
            "ok": True,
            "read_only": True,
            "backend": "linux_nmcli",
            "platform": "Linux",
            "hostname": socket.gethostname(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "connected": bool(connected_devices),
                "active_connection": (
                    connected_devices[0].get("connection")
                    if connected_devices
                    else None
                ),
                "active_wifi_ssid": (
                    active_wifi.get("ssid")
                    if active_wifi
                    else None
                ),
                "wifi_signal": (
                    active_wifi.get("signal")
                    if active_wifi
                    else None
                ),
                "device_count": len(devices),
                "saved_connection_count": len(saved_connections),
                "visible_wifi_count": len(wifi_networks),
                "managed_device_count": len(managed_devices),
                "wifi_device_count": len(wifi_devices),
                "networkmanager_managing_interfaces": bool(
                    managed_devices
                ),
            },
            "devices": devices,
            "saved_connections": saved_connections,
            "wifi_networks": wifi_networks,
        }

    # ------------------------------------------------------------------
    # Windows Wi-Fi backend accessed from WSL
    # ------------------------------------------------------------------

    def _windows_interface_status(self) -> Dict[str, Any]:
        output = self._run_powershell(
            "netsh wlan show interfaces"
        )

        result: Dict[str, Any] = {
            "name": None,
            "description": None,
            "state": "unknown",
            "ssid": None,
            "bssid": None,
            "band": None,
            "channel": None,
            "authentication": None,
            "cipher": None,
            "receive_rate": None,
            "transmit_rate": None,
            "signal": None,
            "profile": None,
        }

        mapping = {
            "Name": "name",
            "Description": "description",
            "State": "state",
            "SSID": "ssid",
            "AP BSSID": "bssid",
            "Band": "band",
            "Channel": "channel",
            "Authentication": "authentication",
            "Cipher": "cipher",
            "Receive rate (Mbps)": "receive_rate",
            "Transmit rate (Mbps)": "transmit_rate",
            "Signal": "signal",
            "Profile": "profile",
        }

        for raw_line in output.splitlines():
            line = raw_line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            target = mapping.get(key)

            if not target:
                continue

            if target == "signal":
                result[target] = _integer(value.rstrip("%"))
            elif target == "channel":
                result[target] = _integer(value)
            else:
                result[target] = value or None

        return result

    def _windows_saved_connections(self) -> List[Dict[str, Any]]:
        output = self._run_powershell(
            "netsh wlan show profiles"
        )

        connections: List[Dict[str, Any]] = []

        for raw_line in output.splitlines():
            line = raw_line.strip()

            if "All User Profile" not in line or ":" not in line:
                continue

            name = _windows_value(line)

            if not name:
                continue

            connections.append(
                {
                    "name": name,
                    "uuid": None,
                    "type": "wifi",
                    "device": "Wi-Fi",
                    "active": False,
                }
            )

        return connections

    def _windows_wifi_networks(
        self,
        interface: Dict[str, Any],
        rescan: bool,
    ) -> List[Dict[str, Any]]:
        if rescan:
            try:
                self._run_powershell(
                    "netsh wlan show networks mode=bssid"
                )
            except NetworkManagerError:
                pass

        output = self._run_powershell(
            "netsh wlan show networks mode=bssid"
        )

        networks: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for raw_line in output.splitlines():
            line = raw_line.strip()

            ssid_match = re.match(
                r"^SSID\s+\d+\s*:\s*(.*)$",
                line,
                re.IGNORECASE,
            )

            if ssid_match:
                if current:
                    networks.append(current)

                ssid = ssid_match.group(1).strip() or "Hidden network"

                current = {
                    "in_use": (
                        bool(interface.get("ssid"))
                        and ssid == interface.get("ssid")
                    ),
                    "ssid": ssid,
                    "signal": (
                        interface.get("signal")
                        if ssid == interface.get("ssid")
                        else None
                    ),
                    "security": "Unknown",
                    "frequency_mhz": None,
                    "channel": None,
                    "rate": None,
                    "bars": None,
                }

                continue

            if current is None:
                continue

            if line.lower().startswith("authentication"):
                value = _windows_value(line)
                current["security"] = value or "Open"

            elif line.lower().startswith("channel"):
                value = _windows_value(line)
                current["channel"] = _integer(value)

            elif line.lower().startswith("signal"):
                value = _windows_value(line)

                if value:
                    current["signal"] = _integer(value.rstrip("%"))

            elif line.lower().startswith("radio type"):
                value = _windows_value(line)

                if value:
                    current["rate"] = value

        if current:
            networks.append(current)

        networks_by_key: Dict[str, Dict[str, Any]] = {}

        for network in networks:
            key = f"{network['ssid']}|{network['security']}"
            previous = networks_by_key.get(key)

            if previous is None or (
                network.get("signal") or -1
            ) > (
                previous.get("signal") or -1
            ):
                networks_by_key[key] = network

        result = list(networks_by_key.values())

        result.sort(
            key=lambda item: (
                not item["in_use"],
                -(item.get("signal") or -1),
                item["ssid"].lower(),
            )
        )

        return result

    def _windows_network_adapters(self) -> List[Dict[str, Any]]:
        command = (
            "Get-NetAdapter | "
            "Select-Object Name,InterfaceDescription,Status,LinkSpeed | "
            "ConvertTo-Csv -NoTypeInformation"
        )

        output = self._run_powershell(command)

        devices: List[Dict[str, Any]] = []

        lines = [
            line
            for line in output.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return devices

        import csv
        import io

        reader = csv.DictReader(io.StringIO("\n".join(lines)))

        for row in reader:
            status = (row.get("Status") or "").strip()
            name = (row.get("Name") or "").strip()
            description = (
                row.get("InterfaceDescription") or ""
            ).strip()
            link_speed = (row.get("LinkSpeed") or "").strip()

            device_type = "wifi" if (
                "wi-fi" in name.lower()
                or "wireless" in description.lower()
                or "802.11" in description.lower()
            ) else "ethernet"

            devices.append(
                {
                    "device": name,
                    "type": device_type,
                    "state": status.lower() or "unknown",
                    "connection": link_speed or None,
                    "description": description or None,
                }
            )

        return devices

    def _collect_windows_wsl(self, rescan: bool) -> Dict[str, Any]:
        interface = self._windows_interface_status()
        devices = self._windows_network_adapters()
        saved_connections = self._windows_saved_connections()
        wifi_networks = self._windows_wifi_networks(
            interface,
            rescan,
        )

        connected = (
            str(interface.get("state") or "").lower()
            == "connected"
        )

        active_ssid = interface.get("ssid")
        active_profile = interface.get("profile") or active_ssid

        for connection in saved_connections:
            connection["active"] = bool(
                active_profile
                and connection.get("name") == active_profile
            )

        wifi_devices = [
            item
            for item in devices
            if item.get("type") == "wifi"
        ]

        managed_devices = [
            item
            for item in devices
            if item.get("state") not in {
                "disabled",
                "not present",
                "unknown",
            }
        ]

        return {
            "ok": True,
            "read_only": True,
            "backend": "windows_wsl",
            "platform": "Windows via WSL2",
            "hostname": socket.gethostname(),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "connected": connected,
                "active_connection": active_profile,
                "active_wifi_ssid": active_ssid,
                "wifi_signal": interface.get("signal"),
                "device_count": len(devices),
                "saved_connection_count": len(saved_connections),
                "visible_wifi_count": len(wifi_networks),
                "managed_device_count": len(managed_devices),
                "wifi_device_count": len(wifi_devices),
                "networkmanager_managing_interfaces": bool(
                    wifi_devices
                ),
            },
            "devices": devices,
            "saved_connections": saved_connections,
            "wifi_networks": wifi_networks,
            "interface_details": interface,
        }

    def collect(self, rescan: bool = False) -> Dict[str, Any]:
        if self.platform_backend == "windows_wsl":
            return self._collect_windows_wsl(bool(rescan))

        if self.platform_backend == "linux_nmcli":
            return self._collect_linux(bool(rescan))

        raise NetworkManagerError(
            "No supported network backend was detected. "
            "Expected powershell.exe under WSL or nmcli on native Linux."
        )
