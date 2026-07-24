#!/usr/bin/env python3
from pathlib import Path
root = Path(__file__).resolve().parent
html = (root / "voice_relay/index.html").read_text()
js = (root / "voice_relay/operator_console.js").read_text()
css = (root / "voice_relay/operator_console.css").read_text()
server = (root / "voice_relay/server.py").read_text()
api = (root / "runtime_api.py").read_text()
for marker in ["diagnosticsStatusPill", "serviceHealthGrid", "systemDiagnostics", "runtimeDiagnostics", "serviceDiagnostics"]: assert marker in html
assert '"/dashboard/diagnostics"' in server
assert '"/diagnostics"' in api
assert 'const DIAGNOSTICS_URL = "/dashboard/diagnostics"' in js
assert ".diagnostics-service-grid" in css
print("PASS: Diagnostics UI integration checks passed.")
