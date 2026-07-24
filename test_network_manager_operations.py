#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from network.network_manager import (
    NetworkManager,
    NetworkManagerError,
)


class NetworkOperationTests(unittest.TestCase):
    def manager(self, backend):
        with patch.object(
            NetworkManager,
            "_detect_backend",
            return_value=backend,
        ):
            return NetworkManager()

    def test_rejects_empty_ssid(self):
        manager = self.manager("linux_nmcli")

        with self.assertRaises(NetworkManagerError):
            manager.connect("   ", "password123")

    def test_rejects_short_wpa_password(self):
        manager = self.manager("linux_nmcli")

        with self.assertRaises(NetworkManagerError):
            manager.connect("Example", "short")

    def test_linux_connect_with_password(self):
        manager = self.manager("linux_nmcli")

        with patch.object(
            manager,
            "_run_nmcli",
            return_value="",
        ) as run:
            result = manager.connect(
                "Example WiFi",
                "password123",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "connect")
        self.assertEqual(result["backend"], "linux_nmcli")

        run.assert_called_once_with(
            "device",
            "wifi",
            "connect",
            "Example WiFi",
            "password",
            "password123",
        )

    def test_linux_connect_saved_profile(self):
        manager = self.manager("linux_nmcli")

        with patch.object(
            manager,
            "_run_nmcli",
            return_value="",
        ) as run:
            manager.connect("Example WiFi")

        run.assert_called_once_with(
            "connection",
            "up",
            "id",
            "Example WiFi",
        )

    def test_linux_disconnects_connected_wifi(self):
        manager = self.manager("linux_nmcli")

        devices = [
            {
                "device": "wlan0",
                "type": "wifi",
                "state": "connected",
                "connection": "Example WiFi",
            },
            {
                "device": "eth0",
                "type": "ethernet",
                "state": "connected",
                "connection": "Wired",
            },
        ]

        with (
            patch.object(
                manager,
                "_linux_device_status",
                return_value=devices,
            ),
            patch.object(
                manager,
                "_run_nmcli",
                return_value="",
            ) as run,
        ):
            result = manager.disconnect()

        self.assertTrue(result["changed"])

        run.assert_called_once_with(
            "device",
            "disconnect",
            "wlan0",
        )

    def test_linux_forget_profile(self):
        manager = self.manager("linux_nmcli")

        with patch.object(
            manager,
            "_run_nmcli",
            return_value="",
        ) as run:
            result = manager.forget("Old WiFi")

        self.assertTrue(result["ok"])

        run.assert_called_once_with(
            "connection",
            "delete",
            "id",
            "Old WiFi",
        )

    def test_windows_saved_profile_connect(self):
        manager = self.manager("windows_wsl")

        with patch.object(
            manager,
            "_run_powershell_script",
            return_value="",
        ) as run:
            result = manager.connect("NETGEAR94")

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "windows_wsl")

        script = run.call_args.args[0]

        self.assertIn("netsh wlan connect", script)
        self.assertIn("NETGEAR94", script)

    def test_windows_password_uses_standard_input_script(self):
        manager = self.manager("windows_wsl")

        with patch.object(
            manager,
            "_run_powershell_script",
            return_value="",
        ) as run:
            result = manager.connect(
                "Example WiFi",
                "password123",
            )

        self.assertTrue(result["ok"])

        script = run.call_args.args[0]

        self.assertIn("FromBase64String", script)
        self.assertIn("netsh wlan add profile", script)
        self.assertIn("netsh wlan connect", script)
        self.assertNotIn("password123", script)

    def test_windows_disconnect(self):
        manager = self.manager("windows_wsl")

        with patch.object(
            manager,
            "_run_powershell",
            return_value="",
        ) as run:
            result = manager.disconnect()

        self.assertTrue(result["ok"])
        run.assert_called_once_with("netsh wlan disconnect")

    def test_windows_forget_profile(self):
        manager = self.manager("windows_wsl")

        with patch.object(
            manager,
            "_run_powershell_script",
            return_value="",
        ) as run:
            result = manager.forget("Old WiFi")

        self.assertTrue(result["ok"])

        script = run.call_args.args[0]

        self.assertIn("netsh wlan delete profile", script)
        self.assertIn("Old WiFi", script)


if __name__ == "__main__":
    unittest.main()
