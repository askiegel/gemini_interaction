#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from network.network_manager import NetworkManager, _split_escaped


class Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class NetworkManagerTests(unittest.TestCase):
    def test_split_escaped_colons(self):
        self.assertEqual(_split_escaped(r"Cafe\:Guest:80:WPA2"), ["Cafe:Guest", "80", "WPA2"])

    @patch("network.network_manager.shutil.which", return_value="/usr/bin/nmcli")
    @patch("network.network_manager.subprocess.run")
    def test_collect_returns_read_only_status(self, run, _which):
        run.side_effect = [
            Completed("wlp2s0:wifi:connected:Home WiFi\nlo:loopback:connected:lo\n"),
            Completed("Home WiFi:uuid-1:802-11-wireless:wlp2s0\nBackup:uuid-2:802-11-wireless:\n"),
            Completed("*:Home WiFi:78:WPA2:5180:36:405 Mbit/s:▂▄▆_\n:Guest:42:--:2412:1:130 Mbit/s:▂▄__\n"),
        ]

        manager = NetworkManager()
        manager.platform_backend = "linux_nmcli"
        payload = manager.collect(rescan=True)

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["summary"]["active_wifi_ssid"], "Home WiFi")
        self.assertEqual(payload["summary"]["wifi_signal"], 78)
        self.assertEqual(payload["summary"]["visible_wifi_count"], 2)
        self.assertTrue(payload["saved_connections"][0]["active"])
        self.assertIn("yes", run.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()
