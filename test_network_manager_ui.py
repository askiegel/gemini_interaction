#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(__file__).resolve().parent

checks = {
    "runtime_api.py": ["/network-status", "NetworkManager", "read_only"],
    "voice_relay/server.py": ["/dashboard/network-status", "Network visibility service unavailable"],
    "voice_relay/index.html": ["networkPage", "Network Manager", "scanWifiButton", "Read-only mode"],
    "voice_relay/operator_console.js": ["NETWORK_URL", "wifiNetworksTable", "savedConnectionsTable"],
    "voice_relay/operator_console.css": ["network-summary-grid", "network-security-note"],
}

for filename, needles in checks.items():
    text = (ROOT / filename).read_text(encoding="utf-8")
    for needle in needles:
        assert needle in text, f"Missing {needle!r} in {filename}"

print("PASS: Read-only Network Manager integration checks passed.")
