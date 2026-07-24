#!/usr/bin/env python3

import base64
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape


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

    def _run_command(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        input_text: Optional[str] = None,
    ) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                input=input_text,
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

    def _run_powershell_script(self, script: str) -> str:
        """Run a PowerShell script through standard input.

        Supplying scripts through standard input prevents Wi-Fi credentials
        from appearing in the PowerShell process command line.
        """
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
                "-",
            ],
            input_text=script,
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

    # ------------------------------------------------------------------
    # Phase 7B network operations
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ssid(ssid: str) -> str:
        value = str(ssid or "").strip()

        if not value:
            raise NetworkManagerError(
                "A Wi-Fi network name is required."
            )

        if len(value.encode("utf-8")) > 32:
            raise NetworkManagerError(
                "A Wi-Fi network name cannot exceed 32 UTF-8 bytes."
            )

        if any(ord(character) < 32 for character in value):
            raise NetworkManagerError(
                "The Wi-Fi network name contains invalid control characters."
            )

        return value

    @staticmethod
    def _validate_password(
        password: Optional[str],
    ) -> Optional[str]:
        if password is None:
            return None

        value = str(password)

        if not value:
            return ""

        if len(value) < 8 or len(value) > 63:
            raise NetworkManagerError(
                "A WPA/WPA2 Wi-Fi password must contain 8 to 63 characters."
            )

        if any(ord(character) < 32 for character in value):
            raise NetworkManagerError(
                "The Wi-Fi password contains invalid control characters."
            )

        return value

    @staticmethod
    def _powershell_single_quote(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _linux_wifi_devices(self) -> List[Dict[str, Any]]:
        return [
            device
            for device in self._linux_device_status()
            if device.get("type") == "wifi"
        ]

    def _linux_connect(
        self,
        ssid: str,
        password: Optional[str],
    ) -> Dict[str, Any]:
        if password is None:
            self._run_nmcli(
                "connection",
                "up",
                "id",
                ssid,
            )
            message = (
                f"Requested connection using saved profile '{ssid}'."
            )
        else:
            arguments = [
                "device",
                "wifi",
                "connect",
                ssid,
            ]

            if password:
                arguments.extend(["password", password])

            self._run_nmcli(*arguments)

            message = (
                f"Requested connection to Wi-Fi network '{ssid}'."
            )

        return {
            "ok": True,
            "action": "connect",
            "backend": "linux_nmcli",
            "ssid": ssid,
            "message": message,
        }

    def _linux_disconnect(self) -> Dict[str, Any]:
        wifi_devices = self._linux_wifi_devices()

        connected_devices = [
            device
            for device in wifi_devices
            if str(device.get("state") or "").lower()
            in {"connected", "connecting"}
        ]

        if not connected_devices:
            return {
                "ok": True,
                "action": "disconnect",
                "backend": "linux_nmcli",
                "changed": False,
                "message": "No connected Linux Wi-Fi interface was found.",
            }

        disconnected = []

        for device in connected_devices:
            device_name = device.get("device")

            if not device_name:
                continue

            self._run_nmcli(
                "device",
                "disconnect",
                device_name,
            )
            disconnected.append(device_name)

        return {
            "ok": True,
            "action": "disconnect",
            "backend": "linux_nmcli",
            "changed": bool(disconnected),
            "devices": disconnected,
            "message": (
                "Disconnected Linux Wi-Fi interface(s): "
                + ", ".join(disconnected)
            ),
        }

    def _linux_forget(self, profile: str) -> Dict[str, Any]:
        self._run_nmcli(
            "connection",
            "delete",
            "id",
            profile,
        )

        return {
            "ok": True,
            "action": "forget",
            "backend": "linux_nmcli",
            "profile": profile,
            "message": f"Deleted saved connection '{profile}'.",
        }

    def _windows_profile_xml(
        self,
        ssid: str,
        password: str,
    ) -> str:
        escaped_ssid = xml_escape(ssid)
        ssid_hex = ssid.encode("utf-8").hex().upper()

        if password:
            escaped_password = xml_escape(password)

            security = f"""\
<security>
    <authEncryption>
        <authentication>WPA2PSK</authentication>
        <encryption>AES</encryption>
        <useOneX>false</useOneX>
    </authEncryption>
    <sharedKey>
        <keyType>passPhrase</keyType>
        <protected>false</protected>
        <keyMaterial>{escaped_password}</keyMaterial>
    </sharedKey>
</security>"""
        else:
            security = """\
<security>
    <authEncryption>
        <authentication>open</authentication>
        <encryption>none</encryption>
        <useOneX>false</useOneX>
    </authEncryption>
</security>"""

        return f"""\
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{escaped_ssid}</name>
    <SSIDConfig>
        <SSID>
            <hex>{ssid_hex}</hex>
            <name>{escaped_ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        {security}
    </MSM>
</WLANProfile>
"""

    def _windows_connect(
        self,
        ssid: str,
        password: Optional[str],
    ) -> Dict[str, Any]:
        quoted_ssid = self._powershell_single_quote(ssid)

        if password is None:
            script = f"""
$ssid = {quoted_ssid}
& netsh wlan connect name="$ssid" ssid="$ssid"
if ($LASTEXITCODE -ne 0) {{
    throw "Windows could not connect using the saved Wi-Fi profile."
}}
"""
            self._run_powershell_script(script)

            message = (
                f"Requested connection using saved Windows profile '{ssid}'."
            )
        else:
            profile_xml = self._windows_profile_xml(
                ssid,
                password,
            )

            encoded_xml = base64.b64encode(
                profile_xml.encode("utf-8")
            ).decode("ascii")

            script = f"""
$ssid = {quoted_ssid}
$profileBytes = [Convert]::FromBase64String('{encoded_xml}')
$tempPath = Join-Path $env:TEMP (
    'mini-pupper-wifi-' +
    [Guid]::NewGuid().ToString() +
    '.xml'
)

try {{
    [IO.File]::WriteAllBytes($tempPath, $profileBytes)

    & netsh wlan add profile filename="$tempPath" user=current

    if ($LASTEXITCODE -ne 0) {{
        throw "Windows could not install the Wi-Fi profile."
    }}

    & netsh wlan connect name="$ssid" ssid="$ssid"

    if ($LASTEXITCODE -ne 0) {{
        throw "Windows could not connect to the Wi-Fi network."
    }}
}}
finally {{
    Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
}}
"""
            self._run_powershell_script(script)

            message = (
                f"Installed and requested connection to Windows "
                f"Wi-Fi network '{ssid}'."
            )

        return {
            "ok": True,
            "action": "connect",
            "backend": "windows_wsl",
            "ssid": ssid,
            "message": message,
        }

    def _windows_disconnect(self) -> Dict[str, Any]:
        self._run_powershell(
            "netsh wlan disconnect"
        )

        return {
            "ok": True,
            "action": "disconnect",
            "backend": "windows_wsl",
            "changed": True,
            "message": "Requested Windows Wi-Fi disconnection.",
        }

    def _windows_forget(self, profile: str) -> Dict[str, Any]:
        quoted_profile = self._powershell_single_quote(profile)

        script = f"""
$profile = {quoted_profile}
& netsh wlan delete profile name="$profile"

if ($LASTEXITCODE -ne 0) {{
    throw "Windows could not delete the saved Wi-Fi profile."
}}
"""
        self._run_powershell_script(script)

        return {
            "ok": True,
            "action": "forget",
            "backend": "windows_wsl",
            "profile": profile,
            "message": f"Deleted saved Windows profile '{profile}'.",
        }

    def connect(
        self,
        ssid: str,
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        validated_ssid = self._validate_ssid(ssid)
        validated_password = self._validate_password(password)

        if self.platform_backend == "windows_wsl":
            return self._windows_connect(
                validated_ssid,
                validated_password,
            )

        if self.platform_backend == "linux_nmcli":
            return self._linux_connect(
                validated_ssid,
                validated_password,
            )

        raise NetworkManagerError(
            "No supported network backend was detected."
        )

    def disconnect(self) -> Dict[str, Any]:
        if self.platform_backend == "windows_wsl":
            return self._windows_disconnect()

        if self.platform_backend == "linux_nmcli":
            return self._linux_disconnect()

        raise NetworkManagerError(
            "No supported network backend was detected."
        )

    def forget(self, profile: str) -> Dict[str, Any]:
        validated_profile = self._validate_ssid(profile)

        if self.platform_backend == "windows_wsl":
            return self._windows_forget(validated_profile)

        if self.platform_backend == "linux_nmcli":
            return self._linux_forget(validated_profile)

        raise NetworkManagerError(
            "No supported network backend was detected."
        )

    def collect(self, rescan: bool = False) -> Dict[str, Any]:
        if self.platform_backend == "windows_wsl":
            return self._collect_windows_wsl(bool(rescan))

        if self.platform_backend == "linux_nmcli":
            return self._collect_linux(bool(rescan))

        raise NetworkManagerError(
            "No supported network backend was detected. "
            "Expected powershell.exe under WSL or nmcli on native Linux."
        )
