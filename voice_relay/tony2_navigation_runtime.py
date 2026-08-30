#!/usr/bin/env python3

"""Deterministic guarded Nav2 ownership on Tony2."""

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path


class Tony2NavigationRuntime:
    """
    Own the headless guarded Nav2 process group.

    Own one bounded NavigateToPose goal at a time.

    Goal execution remains deterministic, map-frame
    only, distance bounded, and timeout bounded.
    """

    MAXIMUM_GOAL_DISTANCE_METERS = 0.50
    EXECUTION_TIMEOUT_SECONDS = 25.0

    MOTION_OUTPUT_CONNECTED = False

    ZENOH_SESSION_OVERRIDE = (
        'connect/endpoints=["tcp/127.0.0.1:7447"];'
        'listen/endpoints=["tcp/127.0.0.1:0"];'
        'scouting/multicast/enabled=false;'
        'transport/shared_memory/enabled=false'
    )

    SUPERVISOR_MARKER = (
        "tony2_navigation_supervisor.py"
    )

    PROBE_MARKER = (
        "tony2_navigation_probe.py"
    )

    GOAL_MARKER = (
        "tony2_navigation_goal.py"
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

        self.goal_script = (
            module_dir
            / "tony2_navigation_goal.py"
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

        self.motion_arm_file = (
            self.runtime_dir
            / "tony2_navigation_motion_arm.json"
        )

        self.motion_egress_status_file = (
            self.runtime_dir
            / (
                "tony2_navigation_"
                "motion_egress_status.json"
            )
        )

        self.supervisor_log = (
            self.runtime_dir
            / "tony2_navigation_supervisor.log"
        )

        self.probe_log = (
            self.runtime_dir
            / "tony2_navigation_probe.log"
        )

        self.goal_pid_file = (
            self.runtime_dir
            / "tony2_navigation_goal.pid"
        )

        self.goal_result_file = (
            self.runtime_dir
            / "tony2_navigation_goal_result.json"
        )

        self.goal_log = (
            self.runtime_dir
            / "tony2_navigation_goal.log"
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
        Use isolated Tony2 Nav2 middleware.

        Nav2 runs on ROS domain 43 through a localhost-only
        Zenoh router. The supervisor owns the separate
        domain-42 Fast DDS ingress process.
        """

        environment = dict(
            self.base_environment
        )

        environment[
            "ROS_DOMAIN_ID"
        ] = "43"

        environment.pop(
            "ROS_LOCALHOST_ONLY",
            None,
        )

        environment[
            "RMW_IMPLEMENTATION"
        ] = "rmw_zenoh_cpp"

        for name in (
            "FASTDDS_BUILTIN_TRANSPORTS",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "FASTDDS_DEFAULT_PROFILES_FILE",
            "ROS_DISCOVERY_SERVER",
            "ROS_SUPER_CLIENT",
            "CYCLONEDDS_URI",
        ):
            environment.pop(
                name,
                None,
            )

        environment[
            "ZENOH_CONFIG_OVERRIDE"
        ] = self.ZENOH_SESSION_OVERRIDE

        environment[
            "TONY2_NAVIGATION_SNAPSHOT"
        ] = str(
            self.snapshot_file
        )

        environment[
            "TONY2_NAVIGATION_MOTION_ARM"
        ] = str(
            self.motion_arm_file
        )

        environment[
            "TONY2_NAVIGATION_EGRESS_STATUS"
        ] = str(
            self.motion_egress_status_file
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

        goal = self._read_pid(
            self.goal_pid_file
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
            "goal": (
                goal
                if self._pid_matches(
                    goal,
                    self.GOAL_MARKER,
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

        goal_active = (
            pids.get("goal")
            is not None
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

            "goal_submission_enabled": (
                runtime_ready
                and not goal_active
                and self.MOTION_OUTPUT_CONNECTED
            ),
            "motion_output_connected":
                self.MOTION_OUTPUT_CONNECTED,
            "isolation_transport":
                "zenoh_localhost",

            "goal_execution_implemented":
                True,
            "goal_active":
                goal_active,
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

        self._safe_unlink(
            self.motion_arm_file
        )

        self._safe_unlink(
            self.motion_egress_status_file
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

        time.sleep(1.5)

        try:
            probe_pid = self._spawn(
                [
                    "/usr/bin/python3",
                    "-u",
                    str(
                        self.probe_script
                    ),
                    "--ros-args",
                    "-r",
                    "/tf:=/nav_tf",
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

        # Fail closed before stopping any navigation
        # processes. No current runtime path creates this
        # lease yet.
        self._safe_unlink(
            self.motion_arm_file
        )

        self._terminate_group(
            pids.get("goal"),
            self.GOAL_MARKER,
        )

        self._terminate_group(
            pids["probe"],
            self.PROBE_MARKER,
        )

        self._terminate_group(
            pids["supervisor"],
            self.SUPERVISOR_MARKER,
        )

        for path in (
            self.goal_pid_file,
            self.goal_result_file,
            self.probe_pid_file,
            self.supervisor_pid_file,
            self.snapshot_file,
            self.motion_arm_file,
            self.motion_egress_status_file,
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
        x,
        y,
        yaw,
    ):
        """
        Execute one guarded map-frame goal.

        The ROS helper verifies the current
        map-to-base_link pose and rejects a goal
        farther than MAXIMUM_GOAL_DISTANCE_METERS
        before sending it to NavigateToPose.
        """
        values = {
            "x": x,
            "y": y,
            "yaw": yaw,
        }

        normalized = {}

        for name, value in values.items():
            try:
                numeric = float(value)
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{name} must be numeric."
                ) from exc

            if not math.isfinite(numeric):
                raise ValueError(
                    f"{name} must be finite."
                )

            normalized[name] = numeric

        status = self.status()

        if status.get("state") != "READY":
            raise RuntimeError(
                "Tony2 guarded navigation must "
                "be READY before goal submission."
            )

        if not status.get(
            "goal_submission_enabled"
        ):
            raise RuntimeError(
                "Tony2 guarded goal submission "
                "is not currently enabled."
            )

        if status.get("goal_active"):
            raise RuntimeError(
                "A Tony2 navigation goal "
                "is already active."
            )

        if not self.goal_script.is_file():
            raise RuntimeError(
                "Tony2 guarded goal helper "
                "is missing."
            )

        self._safe_unlink(
            self.goal_result_file
        )

        command = [
            "/usr/bin/python3",
            "-u",
            str(self.goal_script),
            "--x",
            str(normalized["x"]),
            "--y",
            str(normalized["y"]),
            "--yaw",
            str(normalized["yaw"]),
            "--max-distance",
            str(
                self.MAXIMUM_GOAL_DISTANCE_METERS
            ),
            "--timeout",
            str(
                self.EXECUTION_TIMEOUT_SECONDS
            ),
            "--result-file",
            str(self.goal_result_file),
            "--ros-args",
            "-r",
            "/tf:=/nav_tf",
        ]

        log_handle = open(
            self.goal_log,
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
            self.goal_pid_file,
            process.pid,
        )

        try:
            return_code = process.wait(
                timeout=(
                    self.EXECUTION_TIMEOUT_SECONDS
                    + 15.0
                )
            )

        except subprocess.TimeoutExpired as exc:
            self._terminate_group(
                process.pid,
                self.GOAL_MARKER,
            )

            try:
                process.wait(
                    timeout=1.0
                )
            except subprocess.TimeoutExpired:
                pass

            raise RuntimeError(
                "Tony2 guarded goal helper "
                "exceeded its outer deadline."
            ) from exc

        finally:
            self._safe_unlink(
                self.goal_pid_file
            )

        try:
            payload = json.loads(
                self.goal_result_file.read_text(
                    encoding="utf-8"
                )
            )
        except (
            FileNotFoundError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "Tony2 guarded goal helper "
                "did not produce a valid result."
            ) from exc
        finally:
            self._safe_unlink(
                self.goal_result_file
            )

        if (
            return_code != 0
            or payload.get("ok") is not True
        ):
            raise RuntimeError(
                payload.get(
                    "error",
                    (
                        "Tony2 guarded navigation "
                        f"failed with code {return_code}."
                    ),
                )
            )

        return {
            "action": "SUCCEEDED",
            "goal": payload,
            "navigation": self.status(),
        }
