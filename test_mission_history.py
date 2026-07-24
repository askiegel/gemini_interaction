#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
api = (ROOT / "runtime_api.py").read_text(encoding="utf-8")
server = (ROOT / "voice_relay/server.py").read_text(encoding="utf-8")
html = (ROOT / "voice_relay/index.html").read_text(encoding="utf-8")
js = (ROOT / "voice_relay/operator_console.js").read_text(encoding="utf-8")
css = (ROOT / "voice_relay/operator_console.css").read_text(encoding="utf-8")

checks = {
    "runtime endpoint": 'path == "/mission-history"' in api,
    "history source": "mission_manager.get_history()" in api,
    "relay proxy": 'path == "/dashboard/mission-history"' in server,
    "navigation": 'data-console-page="historyPage"' in html,
    "history page": 'id="historyPage"' in html,
    "history list": 'id="missionHistoryList"' in html,
    "history javascript": 'HISTORY_URL = "/dashboard/mission-history"' in js,
    "history styles": ".mission-history-layout" in css,
    "old node compatibility": "?." not in js and "??" not in js,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("FAIL: " + ", ".join(failed))
print("PASS: Mission History integration checks passed.")
