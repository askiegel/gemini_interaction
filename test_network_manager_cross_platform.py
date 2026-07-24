#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from network.network_manager import NetworkManager


WINDOWS_INTERFACE_OUTPUT = """
There is 1 interface on the system:

    Name                   : Wi-Fi
    Description            : MediaTek MT7921 Wi-Fi 6 802.11ax PCIe Adapter
    State                  : connected
    SSID                   : NETGEAR94
    AP BSSID               : 74:fe:ce:11:07:e7
    Band                   : 5 GHz
    Channel                : 48
    Authentication         : WPA2-Personal
    Cipher                 : CCMP
    Receive rate (Mbps)    : 866.7
    Transmit rate (Mbps)   : 866.7
    Signal                 : 87%
    Profile                : NETGEAR94
"""

WINDOWS_PROFILES_OUTPUT = """
Profiles on interface Wi-Fi:

User profiles
-------------
    All User Profile     : NETGEAR94
    All User Profile     : iPhone
"""

WINDOWS_NETWORKS_OUTPUT = """
Interface name : Wi-Fi
There are 2 networks currently visible.

SSID 1 : NETGEAR94
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP
    BSSID 1                 : 74:fe:ce:11:07:e7
         Signal             : 87%
         Radio type         : 802.11ac
         Channel            : 48

SSID 2 : Guest
    Network type            : Infrastructure
    Authentication          : Open
    Encryption              : None
    BSSID 1                 : aa:bb:cc:dd:ee:ff
         Signal             : 42%
         Radio type         : 802.11n
         Channel            : 6
"""

WINDOWS_ADAPTERS_OUTPUT = '''"Name","InterfaceDescription","Status","LinkSpeed"
"Wi-Fi","MediaTek MT7921 Wi-Fi 6 802.11ax PCIe Adapter","Up","866.7 Mbps"
"Ethernet","Realtek USB GbE Family Controller","Disconnected","0 bps"
'''


class CrossPlatformNetworkTests(unittest.TestCase):
    @patch.object(NetworkManager, "_detect_backend", return_value="windows_wsl")
    @patch.object(NetworkManager, "_run_powershell")
    def test_windows_wsl_collection(self, run_powershell, _detect):
        run_powershell.side_effect = [
            WINDOWS_INTERFACE_OUTPUT,
            WINDOWS_ADAPTERS_OUTPUT,
            WINDOWS_PROFILES_OUTPUT,
            WINDOWS_NETWORKS_OUTPUT,
        ]

        payload = NetworkManager().collect(rescan=False)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["backend"], "windows_wsl")
        self.assertEqual(
            payload["summary"]["active_wifi_ssid"],
            "NETGEAR94",
        )
        self.assertEqual(
            payload["summary"]["wifi_signal"],
            87,
        )
        self.assertEqual(
            payload["summary"]["saved_connection_count"],
            2,
        )
        self.assertEqual(
            payload["summary"]["visible_wifi_count"],
            2,
        )
        self.assertTrue(
            payload["saved_connections"][0]["active"]
        )
        self.assertEqual(
            payload["wifi_networks"][0]["ssid"],
            "NETGEAR94",
        )
        self.assertTrue(
            payload["wifi_networks"][0]["in_use"]
        )


if __name__ == "__main__":
    unittest.main()
