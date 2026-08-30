#!/usr/bin/env python3

"""Deterministic local ownership for Tony2 Cartographer SLAM."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


class Tony2MappingRuntime:
    """Own Cartographer, occupancy grid, and read-only telemetry on Tony2."""

    MINIMUM_SUBMAPS = 3
    MINIMUM_MATURE_SUBMAPS = 2
    MINIMUM_MATURE_VERSION = 100

    CARTOGRAPHER_MARKER = "cartographer_node"
    GRID_MARKER = "cartographer_occupancy_grid_node"
    PROBE_MARKER = "tony2_mapping_probe.py"

    def __init__(
        self,
        *,
        home=None,
        runtime_dir=None,
        base_environment=None,
    ):
        self.home = Path(
            home
            if home is not None
            else Path.home()
        )

        self.runtime_dir = Path(
            runtime_dir
            if runtime_dir is not None
            else "/tmp"
        )

        self.base_environment = dict(
            base_environment
            if base_environment is not None
            else os.environ
        )

        self.cartographer_binary = Path(
            "/opt/ros/humble/lib/"
            "cartographer_ros/cartographer_node"
        )

        self.grid_binary = Path(
            "/opt/ros/humble/lib/"
            "cartographer_ros/"
            "cartographer_occupancy_grid_node"
        )

        self.config_directory = (
            self.home
            / "ros2_ws"
            / "install"
            / "mini_pupper_slam"
            / "share"
            / "mini_pupper_slam"
            / "config"
        )

        self.probe_script = (
            self.home
            / "robot_services"
            / "cognitive"
            / "voice_relay"
            / "tony2_mapping_probe.py"
        )

        self.cartographer_pid_file = (
            self.runtime_dir
            / "tony2_cartographer.pid"
        )

        self.grid_pid_file = (
            self.runtime_dir
            / "tony2_occupancy_grid.pid"
        )

        self.probe_pid_file = (
            self.runtime_dir
            / "tony2_mapping_probe.pid"
        )

        self.snapshot_file = (
            self.runtime_dir
            / "tony2_mapping_snapshot.json"
        )

        self.cartographer_log = (
            self.runtime_dir
            / "tony2_cartographer.log"
        )

        self.grid_log = (
            self.runtime_dir
            / "tony2_occupancy_grid.log"
        )

        self.probe_log = (
            self.runtime_dir
            / "tony2_mapping_probe.log"
        )

    def child_environment(self):
        """
        Return the fixed ROS environment for Tony2 SLAM.

        FASTDDS_BUILTIN_TRANSPORTS is intentionally removed.
        Our live A/B test proved UDPv4-only transport prevents
        Cartographer's large local submap texture responses.
        """

        environment = dict(
            self.base_environment
        )

        environment["ROS_DOMAIN_ID"] = "42"
        environment["ROS_LOCALHOST_ONLY"] = "0"
        environment[
            "RMW_IMPLEMENTATION"
        ] = "rmw_fastrtps_cpp"

        environment.pop(
            "FASTDDS_BUILTIN_TRANSPORTS",
            None,
        )

        environment[
            "TONY2_MAPPING_SNAPSHOT"
        ] = str(self.snapshot_file)

        return environment

    @staticmethod
    def _read_pid(path):
        try:
            value = path.read_text(
                encoding="utf-8"
            ).strip()

            pid = int(value)

            if pid <= 0:
                return None

            return pid

        except (
            FileNotFoundError,
            ValueError,
            OSError,
        ):
            return None

    @staticmethod
    def _cmdline(pid):
        try:
            raw = Path(
                f"/proc/{pid}/cmdline"
            ).read_bytes()

            return raw.replace(
                b"\0",
                b" ",
            ).decode(
                "utf-8",
                errors="replace",
            )

        except OSError:
            return ""

    def _pid_matches(
        self,
        pid,
        marker,
    ):
        if pid is None:
            return False

        try:
            os.kill(
                pid,
                0,
            )
        except OSError:
            return False

        return marker in self._cmdline(pid)

    def _runtime_pids(self):
        cartographer_pid = self._read_pid(
            self.cartographer_pid_file
        )

        grid_pid = self._read_pid(
            self.grid_pid_file
        )

        probe_pid = self._read_pid(
            self.probe_pid_file
        )

        return {
            "cartographer": (
                cartographer_pid
                if self._pid_matches(
                    cartographer_pid,
                    self.CARTOGRAPHER_MARKER,
                )
                else None
            ),
            "occupancy_grid": (
                grid_pid
                if self._pid_matches(
                    grid_pid,
                    self.GRID_MARKER,
                )
                else None
            ),
            "telemetry_probe": (
                probe_pid
                if self._pid_matches(
                    probe_pid,
                    self.PROBE_MARKER,
                )
                else None
            ),
        }

    @staticmethod
    def _write_pid(
        path,
        pid,
    ):
        path.write_text(
            f"{int(pid)}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _safe_unlink(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _read_snapshot(self):
        try:
            payload = json.loads(
                self.snapshot_file.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(
                payload,
                dict,
            ):
                return None

            return payload

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
        ):
            return None

    def _fallback_readiness(self):
        return {
            "available": False,
            "status": "WAITING_FOR_SUBMAPS",
            "ready": False,
            "submap_count": 0,
            "mature_submap_count": 0,
            "minimum_submap_count": (
                self.MINIMUM_SUBMAPS
            ),
            "minimum_mature_submap_count": (
                self.MINIMUM_MATURE_SUBMAPS
            ),
            "minimum_mature_version": (
                self.MINIMUM_MATURE_VERSION
            ),
            "submap_progress": 0.0,
            "mature_submap_progress": 0.0,
            "submaps": [],
        }

    def status(self):
        pids = self._runtime_pids()

        cartographer_running = (
            pids["cartographer"] is not None
        )

        grid_running = (
            pids["occupancy_grid"] is not None
        )

        running = (
            cartographer_running
            and grid_running
        )

        partial = (
            cartographer_running
            != grid_running
        )

        snapshot = (
            self._read_snapshot()
            if running
            else None
        )

        readiness = self._fallback_readiness()

        if (
            isinstance(snapshot, dict)
            and isinstance(
                snapshot.get("readiness"),
                dict,
            )
        ):
            readiness = snapshot["readiness"]

        if not running:
            readiness = self._fallback_readiness()
            readiness["status"] = (
                "MAPPING_STOPPED"
                if not partial
                else "RUNTIME_ERROR"
            )

        return {
            "state": (
                "ERROR"
                if partial
                else (
                    "RUNNING"
                    if running
                    else "STOPPED"
                )
            ),
            "running": running,
            "owned": running,
            "headless": True,
            "host": "Tony2",
            "source": "tony2_local_cartographer",
            "planning_enabled": False,
            "control_enabled": False,
            "validated_map_mutable": False,
            "map_save_enabled": False,
            "candidate_minimum_submaps": (
                self.MINIMUM_SUBMAPS
            ),
            "candidate_minimum_mature_submaps": (
                self.MINIMUM_MATURE_SUBMAPS
            ),
            "candidate_minimum_mature_version": (
                self.MINIMUM_MATURE_VERSION
            ),
            "pids": pids,
            "readiness": readiness,
        }

    def live_map_status(self):
        mapping = self.status()

        if not mapping["running"]:
            return 503, {
                "ok": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": False,
                "mapping": mapping,
                "telemetry": {
                    "available": False,
                    "status": "MAPPING_STOPPED",
                    "received_at": None,
                    "age_seconds": None,
                    "error": None,
                    "map": None,
                },
                "topic": "/map",
                "source": "live_cartographer_map",
                "read_only": True,
                "authoritative": False,
            }

        snapshot = self._read_snapshot()

        if not isinstance(snapshot, dict):
            return 503, {
                "ok": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": True,
                "mapping": mapping,
                "telemetry": {
                    "available": False,
                    "status": "WAITING_FOR_MAP",
                    "received_at": None,
                    "age_seconds": None,
                    "error": None,
                    "map": None,
                },
                "topic": "/map",
                "source": "live_cartographer_map",
                "read_only": True,
                "authoritative": False,
            }

        telemetry = snapshot.get(
            "telemetry",
            {},
        )

        available = bool(
            isinstance(telemetry, dict)
            and telemetry.get("available")
            and telemetry.get("status") == "READY"
            and isinstance(
                telemetry.get("map"),
                dict,
            )
        )

        return (
            200 if available else 503,
            {
                "ok": available,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": True,
                "mapping": mapping,
                "telemetry": telemetry,
                "topic": "/map",
                "source": "live_cartographer_map",
                "read_only": True,
                "authoritative": False,
            },
        )

    def live_pose_status(self):
        """Return Tony2's current map-to-base_link pose."""

        mapping = self.status()

        if not mapping["running"]:
            return 503, {
                "ok": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": False,
                "mapping": mapping,
                "telemetry": {
                    "available": False,
                    "status": "MAPPING_STOPPED",
                    "received_at": None,
                    "age_seconds": None,
                    "error": None,
                    "pose": None,
                },
                "source": "live_cartographer_tf",
                "read_only": True,
                "authoritative": False,
            }

        snapshot = self._read_snapshot()

        if not isinstance(snapshot, dict):
            return 503, {
                "ok": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": True,
                "mapping": mapping,
                "telemetry": {
                    "available": False,
                    "status": "WAITING_FOR_POSE",
                    "received_at": None,
                    "age_seconds": None,
                    "error": None,
                    "pose": None,
                },
                "source": "live_cartographer_tf",
                "read_only": True,
                "authoritative": False,
            }

        telemetry = snapshot.get(
            "pose_telemetry",
            {},
        )

        available = bool(
            isinstance(telemetry, dict)
            and telemetry.get("available")
            and telemetry.get("status") == "READY"
            and isinstance(
                telemetry.get("pose"),
                dict,
            )
        )

        return (
            200 if available else 503,
            {
                "ok": available,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "runtime_active": True,
                "mapping": mapping,
                "telemetry": telemetry,
                "source": "live_cartographer_tf",
                "read_only": True,
                "authoritative": False,
            },
        )

    def _spawn(
        self,
        command,
        log_path,
        pid_path,
    ):
        log_handle = open(
            log_path,
            "ab",
            buffering=0,
        )

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=self.child_environment(),
                start_new_session=True,
            )
        finally:
            log_handle.close()

        self._write_pid(
            pid_path,
            process.pid,
        )

        return process.pid

    def _start_probe(self):
        current = self._runtime_pids()

        if current["telemetry_probe"] is not None:
            return current["telemetry_probe"]

        if not self.probe_script.is_file():
            raise RuntimeError(
                "Tony2 mapping telemetry probe is missing."
            )

        return self._spawn(
            [
                "/usr/bin/python3",
                "-u",
                str(self.probe_script),
                "--ros-args",
                "-r",
                "/tf:=/mayday_navigation_tf",
            ],
            self.probe_log,
            self.probe_pid_file,
        )

    def ensure_probe(self):
        """
        Adopt an already-running Tony2 SLAM session.

        This only starts the read-only local telemetry subscriber.
        It never starts Cartographer and never sends robot motion.
        """

        pids = self._runtime_pids()

        if (
            pids["cartographer"] is None
            or pids["occupancy_grid"] is None
        ):
            return None

        return self._start_probe()

    def start(self):
        """Start a fresh headless Tony2 Cartographer session."""

        current = self._runtime_pids()

        if (
            current["cartographer"] is not None
            and current["occupancy_grid"] is not None
        ):
            self._start_probe()

            return {
                "action": "ALREADY_RUNNING",
                "mapping": self.status(),
            }

        if (
            current["cartographer"] is not None
            or current["occupancy_grid"] is not None
        ):
            raise RuntimeError(
                "Partial Tony2 SLAM runtime exists; "
                "stop it before starting."
            )

        if not self.cartographer_binary.is_file():
            raise RuntimeError(
                "Cartographer binary is missing."
            )

        if not self.grid_binary.is_file():
            raise RuntimeError(
                "Occupancy-grid binary is missing."
            )

        if not (
            self.config_directory
            / "slam.lua"
        ).is_file():
            raise RuntimeError(
                "Tony2 mini_pupper_slam configuration is missing."
            )

        self._safe_unlink(
            self.snapshot_file
        )

        cartographer_pid = self._spawn(
            [
                str(self.cartographer_binary),
                "-configuration_directory",
                str(self.config_directory),
                "-configuration_basename",
                "slam.lua",
                "--ros-args",
                "-r",
                "/imu/data:=imu",
                "-r",
                "/tf:=/mayday_navigation_tf",
            ],
            self.cartographer_log,
            self.cartographer_pid_file,
        )

        try:
            grid_pid = self._spawn(
                [
                    str(self.grid_binary),
                    "-resolution",
                    "0.05",
                    "-publish_period_sec",
                    "1.0",
                ],
                self.grid_log,
                self.grid_pid_file,
            )

            probe_pid = self._start_probe()

        except Exception:
            self._terminate(
                cartographer_pid,
                self.CARTOGRAPHER_MARKER,
            )
            raise

        time.sleep(0.5)

        pids = self._runtime_pids()

        if (
            pids["cartographer"] is None
            or pids["occupancy_grid"] is None
            or pids["telemetry_probe"] is None
        ):
            raise RuntimeError(
                "Tony2 mapping runtime did not remain alive."
            )

        return {
            "action": "STARTED",
            "pids": {
                "cartographer": cartographer_pid,
                "occupancy_grid": grid_pid,
                "telemetry_probe": probe_pid,
            },
            "mapping": self.status(),
        }

    def _terminate(
        self,
        pid,
        marker,
    ):
        if not self._pid_matches(
            pid,
            marker,
        ):
            return False

        try:
            os.kill(
                pid,
                signal.SIGTERM,
            )
        except OSError:
            return False

        deadline = time.monotonic() + 2.0

        while time.monotonic() < deadline:
            if not self._pid_matches(
                pid,
                marker,
            ):
                return True

            time.sleep(0.1)

        if self._pid_matches(
            pid,
            marker,
        ):
            try:
                os.kill(
                    pid,
                    signal.SIGKILL,
                )
            except OSError:
                pass

        return True

    def stop(self):
        """Stop only Tony2-owned SLAM processes."""

        pids = self._runtime_pids()

        self._terminate(
            pids["telemetry_probe"],
            self.PROBE_MARKER,
        )

        self._terminate(
            pids["occupancy_grid"],
            self.GRID_MARKER,
        )

        self._terminate(
            pids["cartographer"],
            self.CARTOGRAPHER_MARKER,
        )

        for path in (
            self.probe_pid_file,
            self.grid_pid_file,
            self.cartographer_pid_file,
            self.snapshot_file,
        ):
            self._safe_unlink(path)

        return {
            "action": "STOPPED",
            "mapping": self.status(),
        }

    def reset(self):
        """Discard the transient live map and start a new one."""

        self.stop()

        result = self.start()
        result["action"] = "RESET"

        return result
