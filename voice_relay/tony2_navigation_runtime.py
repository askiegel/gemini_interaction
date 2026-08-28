#!/usr/bin/env python3

"""Deterministic guarded Nav2 ownership on Tony2."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


class Tony2NavigationRuntime:
    """
    Own the headless guarded Nav2 process group.

    This first feature intentionally does not expose
    any method that can submit a NavigateToPose goal.
    """

    MAXIMUM_GOAL_DISTANCE_METERS = 0.50
    EXECUTION_TIMEOUT_SECONDS = 25.0

    SUPERVISOR_MARKER = (
        "tony2_navigation_supervisor.py"
    )

    PROBE_MARKER = (
        "tony2_navigation_probe.py"
    )

    ASSET_HASHES = {
        "mayday_guarded_navigation.yaml":
            (
                "f750054c85b04ded45407053674c6e9cf"
                "7269379828360b179ec82451c144704"
            ),
        "mayday_guarded_navigate_to_pose.xml":
            (
                "2c6e6a7d2308340d27f435e80a5fb1b"
                "9142058b73d67aa352a09201c81bca069"
            ),
        (
            "mayday_disabled_"
            "navigate_through_poses.xml"
        ):
            (
                "5a810f19415615008cfa8b0511e130fab"
                "4add4c6a26afa722c6b0e099882421f"
            ),
        "latest_tf_relay.py":
            (
                "858999183a2c612076f448b79a531f835"
                "2558d90af16ef09c833540b3a2ed20e"
            ),
    }

    def __init__(
        self,
        *,
        runtime_dir=None,
        asset_dir=None,
        base_environment=None,
    ):
        module_dir = Path(
            __file__
        ).resolve().parent

        self.runtime_dir = Path(
            runtime_dir
            if runtime_dir is not None
            else "/tmp"
        )

        self.asset_dir = Path(
            asset_dir
            if asset_dir is not None
            else (
                module_dir
                / "tony2_navigation_assets"
            )
        )

        self.supervisor_script = (
            module_dir
            / "tony2_navigation_supervisor.py"
        )

        self.probe_script = (
            module_dir
            / "tony2_navigation_probe.py"
        )

        self.supervisor_pid_file = (
            self.runtime_dir
            / "tony2_navigation_supervisor.pid"
        )

        self.probe_pid_file = (
            self.runtime_dir
            / "tony2_navigation_probe.pid"
        )

        self.snapshot_file = (
            self.runtime_dir
            / "tony2_navigation_snapshot.json"
        )

        self.supervisor_log = (
            self.runtime_dir
            / "tony2_navigation_supervisor.log"
        )

        self.probe_log = (
            self.runtime_dir
            / "tony2_navigation_probe.log"
        )

        self.cartographer_pid_file = (
            self.runtime_dir
            / "tony2_cartographer.pid"
        )

        self.grid_pid_file = (
            self.runtime_dir
            / "tony2_occupancy_grid.pid"
        )

        self.base_environment = dict(
            base_environment
            if base_environment is not None
            else os.environ
        )

    def child_environment(self):
        """
        Use the established Tony2 DDS environment.

        UDPv4-only mode is deliberately removed because
        our live Cartographer A/B test demonstrated that
        forcing it breaks large local DDS responses.
        """

        environment = dict(
            self.base_environment
        )

        environment[
            "ROS_DOMAIN_ID"
        ] = "42"

        environment[
            "ROS_LOCALHOST_ONLY"
        ] = "0"

        environment[
            "RMW_IMPLEMENTATION"
        ] = "rmw_fastrtps_cpp"

        environment.pop(
            "FASTDDS_BUILTIN_TRANSPORTS",
            None,
        )

        environment[
            "TONY2_NAVIGATION_SNAPSHOT"
        ] = str(
            self.snapshot_file
        )

        return environment

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()

        with open(
            path,
            "rb",
        ) as handle:
            while True:
                chunk = handle.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    def validate_assets(self):
        for name, expected in (
            self.ASSET_HASHES.items()
        ):
            path = (
                self.asset_dir
                / name
            )

            if not path.is_file():
                raise RuntimeError(
                    "Required guarded navigation "
                    f"asset is missing: {path}"
                )

            actual = self._sha256(
                path
            )

            if actual != expected:
                raise RuntimeError(
                    "Guarded navigation asset "
                    f"hash mismatch: {name}"
                )

        if not self.supervisor_script.is_file():
            raise RuntimeError(
                "Tony2 navigation supervisor "
                "is missing."
            )

        if not self.probe_script.is_file():
            raise RuntimeError(
                "Tony2 navigation probe is missing."
            )

        return True

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
            OSError,
            ValueError,
        ):
            return None

    @staticmethod
    def _cmdline(pid):
        if pid is None:
            return ""

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

        return marker in self._cmdline(
            pid
        )

    def _runtime_pids(self):
        supervisor = self._read_pid(
            self.supervisor_pid_file
        )

        probe = self._read_pid(
            self.probe_pid_file
        )

        return {
            "supervisor": (
                supervisor
                if self._pid_matches(
                    supervisor,
                    self.SUPERVISOR_MARKER,
                )
                else None
            ),
            "probe": (
                probe
                if self._pid_matches(
                    probe,
                    self.PROBE_MARKER,
                )
                else None
            ),
        }

    def _mapping_pid_matches(
        self,
        path,
        marker,
    ):
        pid = self._read_pid(
            path
        )

        if not self._pid_matches(
            pid,
            marker,
        ):
            return None

        return pid

    def mapping_status(self):
        cartographer = (
            self._mapping_pid_matches(
                self.cartographer_pid_file,
                "cartographer_node",
            )
        )

        occupancy_grid = (
            self._mapping_pid_matches(
                self.grid_pid_file,
                (
                    "cartographer_"
                    "occupancy_grid_node"
                ),
            )
        )

        return {
            "running": (
                cartographer is not None
                and occupancy_grid is not None
            ),
            "cartographer":
                cartographer,
            "occupancy_grid":
                occupancy_grid,
        }

    @staticmethod
    def _safe_unlink(path):
        try:
            path.unlink()

        except FileNotFoundError:
            pass

    @staticmethod
    def _write_pid(
        path,
        pid,
    ):
        path.write_text(
            f"{int(pid)}\n",
            encoding="utf-8",
        )

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
            OSError,
            json.JSONDecodeError,
        ):
            return None

    def status(self):
        pids = self._runtime_pids()

        supervisor_running = (
            pids["supervisor"]
            is not None
        )

        probe_running = (
            pids["probe"]
            is not None
        )

        running = (
            supervisor_running
            and probe_running
        )

        partial = (
            supervisor_running
            != probe_running
        )

        snapshot = (
            self._read_snapshot()
            if running
            else None
        )

        planner_enabled = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "planner_enabled"
            ) is True
        )

        controller_enabled = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "controller_enabled"
            ) is True
        )

        navigator_enabled = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "navigator_enabled"
            ) is True
        )

        action_server_ready = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "action_server_ready"
            ) is True
        )

        transform_ready = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "transform_ready"
            ) is True
        )

        runtime_ready = all(
            (
                running,
                planner_enabled,
                controller_enabled,
                navigator_enabled,
                action_server_ready,
                transform_ready,
            )
        )

        mapping = (
            self.mapping_status()
        )

        return {
            "state": (
                "ERROR"
                if partial
                else (
                    "READY"
                    if runtime_ready
                    else (
                        "STARTING"
                        if running
                        else "STOPPED"
                    )
                )
            ),
            "running": running,
            "owned": running,
            "host": "Tony2",
            "source":
                "tony2_guarded_navigation",
            "headless": True,
            "mapping_required": True,
            "mapping_running":
                mapping["running"],
            "planner_enabled":
                planner_enabled,
            "controller_enabled":
                controller_enabled,
            "navigator_enabled":
                navigator_enabled,
            "action_server_ready":
                action_server_ready,
            "transform_ready":
                transform_ready,

            # Intentionally false in this feature.
            "goal_submission_enabled":
                False,

            "goal_execution_implemented":
                False,
            "maximum_goal_distance_meters":
                self.MAXIMUM_GOAL_DISTANCE_METERS,
            "execution_timeout_seconds":
                self.EXECUTION_TIMEOUT_SECONDS,
            "recoveries": False,
            "retries": False,
            "pids": pids,
            "mapping_pids": {
                "cartographer":
                    mapping["cartographer"],
                "occupancy_grid":
                    mapping["occupancy_grid"],
            },
        }

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

    def start(self):
        """
        Start guarded Nav2 with no goal submission.

        This method does not publish cmd_vel itself and
        does not create or send a NavigateToPose goal.
        """

        self.validate_assets()

        mapping = self.mapping_status()

        if not mapping["running"]:
            raise RuntimeError(
                "Tony2 Cartographer mapping must "
                "be running before guarded Nav2."
            )

        current = self._runtime_pids()

        if (
            current["supervisor"] is not None
            and current["probe"] is not None
        ):
            return {
                "action":
                    "ALREADY_RUNNING",
                "navigation":
                    self.status(),
            }

        if (
            current["supervisor"] is not None
            or current["probe"] is not None
        ):
            raise RuntimeError(
                "Partial Tony2 navigation runtime "
                "exists; stop it before starting."
            )

        self._safe_unlink(
            self.snapshot_file
        )

        supervisor_pid = (
            self._spawn(
                [
                    "/usr/bin/python3",
                    "-u",
                    str(
                        self.supervisor_script
                    ),
                    "--asset-dir",
                    str(
                        self.asset_dir
                    ),
                ],
                self.supervisor_log,
                self.supervisor_pid_file,
            )
        )

        try:
            probe_pid = self._spawn(
                [
                    "/usr/bin/python3",
                    "-u",
                    str(
                        self.probe_script
                    ),
                ],
                self.probe_log,
                self.probe_pid_file,
            )

        except Exception:
            self._terminate_group(
                supervisor_pid,
                self.SUPERVISOR_MARKER,
            )

            raise

        time.sleep(
            0.5
        )

        current = (
            self._runtime_pids()
        )

        if (
            current["supervisor"] is None
            or current["probe"] is None
        ):
            self.stop()

            raise RuntimeError(
                "Tony2 guarded navigation "
                "runtime did not remain alive."
            )

        return {
            "action": "STARTED",
            "navigation":
                self.status(),
        }

    def _terminate_group(
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
            os.killpg(
                pid,
                signal.SIGTERM,
            )

        except OSError:
            return False

        deadline = (
            time.monotonic()
            + 3.0
        )

        while (
            time.monotonic()
            < deadline
        ):
            if not self._pid_matches(
                pid,
                marker,
            ):
                return True

            time.sleep(
                0.1
            )

        if self._pid_matches(
            pid,
            marker,
        ):
            try:
                os.killpg(
                    pid,
                    signal.SIGKILL,
                )

            except OSError:
                pass

        return True

    def stop(self):
        """
        Stop the probe and the complete Nav2
        supervisor process group.
        """

        pids = self._runtime_pids()

        self._terminate_group(
            pids["probe"],
            self.PROBE_MARKER,
        )

        self._terminate_group(
            pids["supervisor"],
            self.SUPERVISOR_MARKER,
        )

        for path in (
            self.probe_pid_file,
            self.supervisor_pid_file,
            self.snapshot_file,
        ):
            self._safe_unlink(
                path
            )

        return {
            "action": "STOPPED",
            "navigation":
                self.status(),
        }

    def submit_goal(
        self,
        *_args,
        **_kwargs,
    ):
        raise RuntimeError(
            "Goal execution is intentionally "
            "disabled in this runtime feature."
        )
