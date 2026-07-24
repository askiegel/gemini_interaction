#!/usr/bin/env python3

import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


class DiagnosticsManager:
    """Collect a fast, serializable health snapshot for the operator console."""

    def __init__(self, runtime, config_manager, project_dir=None, probe_timeout=1.0):
        self.runtime = runtime
        self.config_manager = config_manager
        self.project_dir = Path(project_dir or Path(__file__).resolve().parents[1])
        self.probe_timeout = float(probe_timeout)
        self._cpu_lock = threading.Lock()
        self._previous_cpu = None

    @staticmethod
    def _read_first_line(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return ""

    def _cpu_percent(self) -> Optional[float]:
        line = self._read_first_line("/proc/stat")
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        total = sum(values)
        with self._cpu_lock:
            current = (idle, total)
            previous = self._previous_cpu
            self._previous_cpu = current
        if previous is None:
            try:
                one_minute = os.getloadavg()[0]
                cpus = max(1, os.cpu_count() or 1)
                return round(min(100.0, one_minute * 100.0 / cpus), 1)
            except (OSError, AttributeError):
                return None
        idle_delta = idle - previous[0]
        total_delta = total - previous[1]
        if total_delta <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), 1)

    @staticmethod
    def _memory_percent() -> Optional[float]:
        values = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.strip().split()[0])
            total = values.get("MemTotal", 0.0)
            available = values.get("MemAvailable", 0.0)
            if total <= 0:
                return None
            return round(100.0 * (total - available) / total, 1)
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _uptime_seconds() -> Optional[float]:
        try:
            return round(float(Path("/proc/uptime").read_text().split()[0]), 1)
        except (OSError, ValueError, IndexError):
            return None

    def _git_value(self, *args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=self.project_dir,
                check=True, capture_output=True, text=True, timeout=1.5,
            )
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def _probe_json(self, url: str) -> Dict[str, Any]:
        started = time.monotonic()
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.probe_timeout) as response:
                raw = response.read(65536)
                payload = None
                try:
                    import json
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (UnicodeDecodeError, ValueError):
                    payload = None
                return {
                    "online": 200 <= response.status < 400,
                    "status_code": response.status,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                    "data": payload,
                    "error": None,
                }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "online": False,
                "status_code": None,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                "data": None,
                "error": str(exc),
            }

    def collect(self) -> Dict[str, Any]:
        config = self.config_manager.get_config()
        runtime_status = self.runtime.get_status()
        network = config.get("network", {})
        robot_ip = network.get("robot_ip", "")
        bridge_port = network.get("robot_bridge_port", 8090)
        vision_url = config.get("vision", {}).get("server_url", "")
        bridge_url = f"http://{robot_ip}:{bridge_port}/status"
        camera_url = f"http://{robot_ip}:8091/camera/latest.jpg"

        robot = self._probe_json(bridge_url) if robot_ip else {"online": False, "error": "Robot IP is not configured."}
        vision = self._probe_json(vision_url) if vision_url else {"online": False, "error": "Vision URL is not configured."}
        camera = self._probe_json(camera_url) if robot_ip else {"online": False, "error": "Robot IP is not configured."}

        try:
            entity_count = len(self.runtime.world_model.get_entities())
        except Exception:
            entity_count = None

        disk = shutil.disk_usage(self.project_dir)
        active = runtime_status.get("active_mission")
        robot_data = robot.get("data") if isinstance(robot.get("data"), dict) else {}
        vision_data = vision.get("data") if isinstance(vision.get("data"), dict) else {}

        return {
            "ok": True,
            "generated_at": time.time(),
            "overall": "healthy" if all((runtime_status.get("running"), robot.get("online"), vision.get("online"))) else "degraded",
            "runtime": {
                "running": bool(runtime_status.get("running")),
                "state": runtime_status.get("runtime_state"),
                "uptime_seconds": runtime_status.get("uptime_seconds"),
                "loop_hz": round(1.0 / self.runtime.loop_interval, 1) if self.runtime.loop_interval > 0 else None,
                "active_mission": active,
                "queue_length": len(runtime_status.get("queue", [])),
                "history_count": runtime_status.get("history_count", 0),
                "entity_count": entity_count,
                "last_error": runtime_status.get("last_error"),
            },
            "system": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_percent": self._cpu_percent(),
                "memory_percent": self._memory_percent(),
                "disk_percent": round(100.0 * disk.used / disk.total, 1) if disk.total else None,
                "uptime_seconds": self._uptime_seconds(),
                "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                "git_branch": self._git_value("branch", "--show-current"),
                "git_commit": self._git_value("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(self._git_value("status", "--porcelain")),
            },
            "services": {
                "runtime": {"online": bool(runtime_status.get("running")), "status": runtime_status.get("runtime_state"), "error": runtime_status.get("last_error")},
                "robot_bridge": {**robot, "status": robot_data.get("status"), "ros_ready": robot_data.get("ros_ready")},
                "vision": {**vision, "camera_running": vision_data.get("camera_running"), "last_error": vision_data.get("last_error")},
                "camera": camera,
            },
        }
