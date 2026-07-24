#!/usr/bin/env python3
import json
import sys
import threading
import types
from pathlib import Path
from urllib.request import urlopen

config_module = types.ModuleType("config.config_manager")
class ConfigurationError(Exception):
    pass
class ConfigurationManager:
    def get_config(self): return {}
config_module.ConfigurationError = ConfigurationError
config_module.ConfigurationManager = ConfigurationManager
sys.modules["config"] = types.ModuleType("config")
sys.modules["config.config_manager"] = config_module

diagnostics_module = types.ModuleType("diagnostics")
class DiagnosticsManager:
    def __init__(self, *args, **kwargs): pass
diagnostics_module.DiagnosticsManager = DiagnosticsManager
sys.modules["diagnostics"] = diagnostics_module

runtime_module = types.ModuleType("runtime")
class CognitiveRuntime:
    pass
runtime_module.CognitiveRuntime = CognitiveRuntime
sys.modules["runtime"] = runtime_module

from runtime_api import create_server


class FakeMissionManager:
    def get_history(self):
        return []


class FakeWorldModel:
    robot_state = {"mission": "FOLLOW_PERSON", "navigation_state": "TRACKING"}
    environment = {"location": "lab"}

    def get_entities(self):
        return [{
            "entity_id": "person-001",
            "label": "person",
            "entity_type": "person",
            "first_seen": "2026-07-24T00:00:00Z",
            "last_seen": "2026-07-24T00:00:01Z",
            "confidence": 0.97,
            "attributes": {"center_x": 320},
            "history": [{"timestamp": "2026-07-24T00:00:01Z", "source": "vision", "confidence": 0.97, "location": None, "attributes": {}}],
        }]

    def get_recent_events(self):
        return [{"timestamp": "2026-07-24T00:00:01Z", "type": "entity_updated", "data": {"entity_id": "person-001"}}]


class FakeRuntime:
    running = True
    mission_manager = FakeMissionManager()
    world_model = FakeWorldModel()

    def get_status(self):
        return {"running": True}


class FakeConfigManager:
    def get_config(self):
        return {}


server = create_server(FakeRuntime(), host="127.0.0.1", port=0, config_manager=FakeConfigManager())
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with urlopen("http://127.0.0.1:%d/world-model" % server.server_address[1], timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["entities"][0]["entity_id"] == "person-001"
    assert payload["recent_events"][0]["type"] == "entity_updated"
finally:
    server.shutdown()
    server.server_close()

html = Path("voice_relay/index.html").read_text(encoding="utf-8")
js = Path("voice_relay/operator_console.js").read_text(encoding="utf-8")
server_text = Path("voice_relay/server.py").read_text(encoding="utf-8")
for required in ["worldModelPage", "worldModelList", "worldModelDetail", "worldModelEvents"]:
    assert required in html
assert "/dashboard/world-model" in js
assert "/dashboard/world-model" in server_text
assert "?." not in js and "??" not in js
print("PASS: World Model Explorer integration checks passed.")
