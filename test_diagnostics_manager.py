#!/usr/bin/env python3
import tempfile
from pathlib import Path
from diagnostics import DiagnosticsManager

class Runtime:
    loop_interval = 0.03
    class World:
        def get_entities(self): return [{"id": "one"}]
    world_model = World()
    def get_status(self):
        return {"running": True, "runtime_state": "IDLE", "uptime_seconds": 12, "active_mission": None, "queue": [], "history_count": 2, "last_error": None}

class Config:
    robot_bridge_url = "http://127.0.0.1:1"
    camera_relay_url = "http://127.0.0.1:1/camera/latest.jpg"

    def get_config(self):
        return {
            "network": {
                "robot_ip": "192.0.2.1",
                "robot_bridge_port": 8090,
            },
            "vision": {
                "server_url": "http://127.0.0.1:1/nope",
            },
        }

with tempfile.TemporaryDirectory() as directory:
    manager = DiagnosticsManager(Runtime(), Config(), project_dir=Path(directory), probe_timeout=0.01)
    payload = manager.collect()
    assert payload["ok"] is True
    assert payload["runtime"]["entity_count"] == 1
    assert payload["runtime"]["loop_hz"] == 33.3
    assert "cpu_percent" in payload["system"]
    assert set(payload["services"]) == {"runtime", "robot_bridge", "vision", "camera"}
print("PASS: DiagnosticsManager tests passed.")
