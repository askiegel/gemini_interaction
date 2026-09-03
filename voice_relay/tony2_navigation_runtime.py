#!/usr/bin/env python3

"""Deterministic guarded Nav2 ownership on Tony2."""

import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


try:
    from .tony2_navigation_motion_arm import (
        MotionArmLease,
        read_json_file,
    )
except ImportError:
    from tony2_navigation_motion_arm import (
        MotionArmLease,
        read_json_file,
    )


class Tony2NavigationRuntime:
    """
    Own the headless guarded Nav2 process group.

    Own one bounded NavigateToPose goal at a time.

    Goal execution remains deterministic, map-frame
    only, distance bounded, and timeout bounded.
    """

    MAXIMUM_GOAL_DISTANCE_METERS = 0.50
    EXECUTION_TIMEOUT_SECONDS = 25.0

    FIXED_MAP_MIN_X = -2.60
    FIXED_MAP_MAX_X = 4.00
    FIXED_MAP_MIN_Y = -2.55
    FIXED_MAP_MAX_Y = 2.75

    MOTION_ARM_ACK_TIMEOUT_SECONDS = 2.0
    MOTION_DISARM_TIMEOUT_SECONDS = 1.0
    MOTION_STATE_POLL_SECONDS = 0.05

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
        "mayday_supervised_route_03.yaml":
            (
                "8b9d9b9aae7875a30b0611b0d88496c"
                "caafbf54c75be1d808eea61f90aeedf1f"
            ),
        "mayday_supervised_route_03.pgm":
            (
                "e2cf598840f8fe65475205ba08d99106"
                "d4c48bcc91deff3e2df15ca138a84f51"
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
        robot_bridge_url=None,
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

        self.initial_pose_script = (
            module_dir
            / "tony2_navigation_initial_pose.py"
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

        resolved_robot_bridge_url = (
            robot_bridge_url
            if robot_bridge_url is not None
            else self.base_environment.get(
                "ROBOT_BRIDGE_URL"
            )
        )

        self.robot_bridge_url = (
            str(resolved_robot_bridge_url)
            .rstrip("/")
            if resolved_robot_bridge_url
            else None
        )

        # Voice Relay uses ThreadingHTTPServer. GO and STOP
        # can therefore execute concurrently against this
        # single runtime object.
        self._motion_lock = threading.RLock()

        # Set immediately when STOP begins. Unlike this event,
        # the generation counter is never cleared and therefore
        # records that a STOP occurred even after STOP finishes.
        self._stop_requested = threading.Event()
        self._stop_generation = 0
        self._stopping = False

        self._active_motion_lease = None
        self._active_motion_token = None
        self._goal_process = None

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

        persistent_map_yaml = (
            Path.home()
            / ".local"
            / "share"
            / "mayday"
            / "persistent_map"
            / "active"
            / "mayday_supervised_route_03.yaml"
        )

        if persistent_map_yaml.is_file():
            environment[
                "MAYDAY_FIXED_MAP_YAML"
            ] = str(
                persistent_map_yaml
            )
        else:
            environment.pop(
                "MAYDAY_FIXED_MAP_YAML",
                None,
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

    def _read_motion_egress_status(self):
        payload = read_json_file(
            self.motion_egress_status_file
        )

        if not isinstance(payload, dict):
            return {}

        return payload

    @staticmethod
    def _motion_egress_running_from(
        payload,
    ):
        return bool(
            isinstance(payload, dict)
            and payload.get("running") is True
        )

    @staticmethod
    def _motion_egress_idle_from(
        payload,
    ):
        return bool(
            isinstance(payload, dict)
            and payload.get("running") is True
            and payload.get("armed") is False
            and payload.get("token") is None
        )

    @staticmethod
    def _egress_acknowledges_token_from(
        payload,
        token,
    ):
        return bool(
            isinstance(payload, dict)
            and payload.get("running") is True
            and payload.get("armed") is True
            and payload.get("token") == token
        )

    def _egress_acknowledges_token(
        self,
        token,
    ):
        return (
            self._egress_acknowledges_token_from(
                self._read_motion_egress_status(),
                token,
            )
        )

    def _motion_lease_is_current(
        self,
        lease,
        token,
        generation,
    ):
        with self._motion_lock:
            return bool(
                not self._stopping
                and not self._stop_requested.is_set()
                and self._stop_generation
                    == generation
                and self._active_motion_lease
                    is lease
                and self._active_motion_token
                    == token
            )

    def _begin_motion_lease(self):
        with self._motion_lock:
            if (
                self._stopping
                or self._stop_requested.is_set()
            ):
                raise RuntimeError(
                    "Tony2 navigation STOP is in progress."
                )

            if self._active_motion_lease is not None:
                raise RuntimeError(
                    "Tony2 motion authorization "
                    "is already active."
                )

            status = self.status()

            if (
                status.get("state") != "READY"
                or status.get(
                    "goal_submission_enabled"
                ) is not True
            ):
                raise RuntimeError(
                    "Tony2 guarded navigation is no "
                    "longer available for GO."
                )

            if not self.robot_bridge_url:
                raise RuntimeError(
                    "Robot Bridge URL is unavailable "
                    "for guarded motion authorization."
                )

            generation = self._stop_generation

            lease = MotionArmLease(
                self.motion_arm_file,
                self.robot_bridge_url,
            )

            try:
                token = lease.start()

                # STOP can set its event without waiting for
                # this lock. If that happened during start(),
                # destroy the lease before publishing ownership.
                if self._stop_requested.is_set():
                    lease.stop()

                    self._safe_unlink(
                        self.motion_arm_file
                    )

                    raise RuntimeError(
                        "Tony2 guarded GO was cancelled "
                        "while motion authorization started."
                    )

            except Exception:
                # MotionArmLease.stop() is idempotent enough
                # for cleanup after a partially completed start.
                try:
                    lease.stop()
                except Exception:
                    pass

                self._safe_unlink(
                    self.motion_arm_file
                )

                raise

            self._active_motion_lease = lease
            self._active_motion_token = token

            return (
                lease,
                token,
                generation,
            )

    def _release_motion_lease(
        self,
        lease,
    ):
        if lease is None:
            return False

        with self._motion_lock:
            if self._active_motion_lease is not lease:
                return False

            # CRITICAL ORDER:
            # Keep ownership while the refresher is stopped.
            # STOP cannot observe "no lease" while the lease
            # thread is still capable of rewriting the file.
            try:
                lease.stop()
            finally:
                # Runtime STOP/goal completion is globally
                # fail-closed: no authorization file may remain.
                self._safe_unlink(
                    self.motion_arm_file
                )

                if self._active_motion_lease is lease:
                    self._active_motion_lease = None
                    self._active_motion_token = None

        return True

    def _wait_for_motion_arm_ack(
        self,
        lease,
        token,
        generation,
    ):
        deadline = (
            time.monotonic()
            + self.MOTION_ARM_ACK_TIMEOUT_SECONDS
        )

        while time.monotonic() < deadline:
            if not self._motion_lease_is_current(
                lease,
                token,
                generation,
            ):
                raise RuntimeError(
                    "Tony2 guarded GO was cancelled "
                    "before motion egress armed."
                )

            if self._egress_acknowledges_token(
                token
            ):
                return True

            time.sleep(
                self.MOTION_STATE_POLL_SECONDS
            )

        raise RuntimeError(
            "Tony2 motion egress did not acknowledge "
            "the transient GO lease."
        )

    def _wait_for_motion_disarm(self):
        deadline = (
            time.monotonic()
            + self.MOTION_DISARM_TIMEOUT_SECONDS
        )

        while time.monotonic() < deadline:
            payload = (
                self._read_motion_egress_status()
            )

            if self._motion_egress_idle_from(
                payload
            ):
                return True

            time.sleep(
                self.MOTION_STATE_POLL_SECONDS
            )

        return False

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

        map_server_enabled = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "map_server_enabled"
            ) is True
        )

        localization_enabled = bool(
            isinstance(snapshot, dict)
            and snapshot.get(
                "localization_enabled"
            ) is True
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
                map_server_enabled,
                localization_enabled,
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
        runtime_ready = (
            runtime_ready
            and not mapping["running"]
        )


        with self._motion_lock:
            active_token = (
                self._active_motion_token
            )

            stopping = bool(
                self._stopping
                or self._stop_requested.is_set()
            )

        egress_status = (
            self._read_motion_egress_status()
        )

        motion_egress_ready = (
            self._motion_egress_running_from(
                egress_status
            )
        )

        motion_egress_idle = (
            self._motion_egress_idle_from(
                egress_status
            )
        )

        motion_output_connected = bool(
            not stopping
            and active_token is not None
            and self._egress_acknowledges_token_from(
                egress_status,
                active_token,
            )
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
            "mapping_required": False,
            "mapping_running":
                mapping["running"],
            "mapping_conflict":
                mapping["running"],
            "fixed_map_required": True,
            "fixed_map":
                "mayday_supervised_route_03.yaml",
            "map_source":
                "local_map_server",
            "localization_mode":
                "amcl",
            "map_server_enabled":
                map_server_enabled,
            "localization_enabled":
                localization_enabled,
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

            "motion_egress_ready":
                motion_egress_ready,
            "motion_egress_idle":
                motion_egress_idle,
            "goal_submission_enabled": (
                runtime_ready
                and motion_egress_idle
                and not goal_active
                and not stopping
                and active_token is None
            ),
            "motion_output_connected":
                motion_output_connected,
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

        if mapping["running"]:
            raise RuntimeError(
                "Tony2 Cartographer mapping must "
                "be stopped before fixed-map navigation."
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

    @staticmethod
    def _process_group_exists(
        pgid,
    ):
        if pgid is None:
            return False

        try:
            pgid = int(pgid)
        except (
            TypeError,
            ValueError,
        ):
            return False

        if pgid <= 0:
            return False

        try:
            os.killpg(
                pgid,
                0,
            )

        except ProcessLookupError:
            return False

        except PermissionError:
            return True

        except OSError:
            # Fail closed for unexpected signal-zero
            # errors: do not assume the group vanished.
            return True

        return True

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
            pgid = os.getpgid(
                pid
            )

        except OSError:
            return False

        try:
            os.killpg(
                pgid,
                signal.SIGTERM,
            )

        except ProcessLookupError:
            return True

        except OSError:
            return False

        # The session leader may exit before its launch
        # children. Watch the whole process group rather
        # than only the leader PID.
        deadline = (
            time.monotonic()
            + 3.0
        )

        while (
            time.monotonic()
            < deadline
        ):
            if not self._process_group_exists(
                pgid
            ):
                return True

            time.sleep(
                0.1
            )

        if self._process_group_exists(
            pgid
        ):
            try:
                os.killpg(
                    pgid,
                    signal.SIGKILL,
                )

            except ProcessLookupError:
                return True

            except OSError:
                return False

        kill_deadline = (
            time.monotonic()
            + 1.0
        )

        while (
            time.monotonic()
            < kill_deadline
        ):
            if not self._process_group_exists(
                pgid
            ):
                return True

            time.sleep(
                0.05
            )

        return not self._process_group_exists(
            pgid
        )

    def stop(self):
        """
        Stop motion authorization before navigation processes.

        The lease refresher is stopped while ownership remains
        protected by _motion_lock. Only after that can the goal,
        probe, and Nav2 supervisor be terminated.
        """
        # Make STOP visible immediately to a GO thread that
        # currently owns _motion_lock.
        self._stop_requested.set()

        lease_was_active = False
        lease_stop_error = None

        with self._motion_lock:
            self._stop_generation += 1
            self._stopping = True

            lease = self._active_motion_lease

            if lease is not None:
                lease_was_active = True

                # CRITICAL: do not detach ownership before
                # the refresher thread has stopped.
                try:
                    lease.stop()
                except Exception as exc:
                    lease_stop_error = exc
                finally:
                    self._safe_unlink(
                        self.motion_arm_file
                    )

                    if self._active_motion_lease is lease:
                        self._active_motion_lease = None
                        self._active_motion_token = None

            else:
                self._safe_unlink(
                    self.motion_arm_file
                )

            goal_process = self._goal_process
            self._goal_process = None

        try:
            # If egress had actually been armed, removing the
            # lease makes it call Robot Bridge STOP. Give that
            # fail-safe transition a bounded opportunity before
            # killing Nav2.
            if lease_was_active:
                self._wait_for_motion_disarm()

            pids = self._runtime_pids()

            goal_pid = (
                getattr(
                    goal_process,
                    "pid",
                    None,
                )
                if goal_process is not None
                else pids.get("goal")
            )

            self._terminate_group(
                goal_pid,
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

        finally:
            with self._motion_lock:
                self._goal_process = None
                self._stopping = False

            self._stop_requested.clear()

        if lease_stop_error is not None:
            raise RuntimeError(
                "Tony2 motion lease refresher "
                "did not stop cleanly."
            ) from lease_stop_error

        return {
            "action": "STOPPED",
            "navigation":
                self.status(),
        }


    def initialize_global_localization(self):
        """
        Run stationary map-wide AMCL localization.

        The existing localization process owner is reused so
        lifecycle, fail-closed STOP behavior, egress checks,
        and trust handling remain unchanged. Its historical
        x/y/yaw compatibility arguments are not used by the
        helper to seed AMCL.
        """
        return self.initialize_operator_pose(
            0.0,
            0.0,
            0.0,
        )

    def initialize_operator_pose(
        self,
        x,
        y,
        yaw,
    ):
        """
        Set and validate one stationary operator AMCL pose.

        This path never creates a motion lease and never
        submits a navigation goal. Any failed localization
        validation stops the navigation runtime fail closed.
        """

        values = {
            "x": x,
            "y": y,
            "yaw": yaw,
        }

        normalized = {}

        for name, value in values.items():
            try:
                numeric = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{name} must be numeric."
                ) from exc

            if not math.isfinite(
                numeric
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

            normalized[name] = numeric

        if not (
            self.FIXED_MAP_MIN_X
            <= normalized["x"]
            <= self.FIXED_MAP_MAX_X
        ):
            raise ValueError(
                "Operator pose x lies outside "
                "the fixed navigation map."
            )

        if not (
            self.FIXED_MAP_MIN_Y
            <= normalized["y"]
            <= self.FIXED_MAP_MAX_Y
        ):
            raise ValueError(
                "Operator pose y lies outside "
                "the fixed navigation map."
            )

        normalized["yaw"] = math.atan2(
            math.sin(
                normalized["yaw"]
            ),
            math.cos(
                normalized["yaw"]
            ),
        )

        status = self.status()

        if status.get("running") is not True:
            raise RuntimeError(
                "Tony2 fixed navigation must be "
                "running before operator localization."
            )

        if (
            status.get(
                "map_server_enabled"
            ) is not True
            or status.get(
                "localization_enabled"
            ) is not True
        ):
            raise RuntimeError(
                "Tony2 fixed map and AMCL must be "
                "active before operator localization."
            )

        if status.get(
            "transform_ready"
        ) is True:
            raise RuntimeError(
                "Tony2 localization is already initialized; "
                "stop and restart before setting a new "
                "operator pose."
            )

        if status.get(
            "goal_submission_enabled"
        ) is True:
            raise RuntimeError(
                "GO must remain disabled while the "
                "operator pose is initialized."
            )

        if status.get(
            "goal_active"
        ) is True:
            raise RuntimeError(
                "A navigation goal is active."
            )

        if status.get(
            "motion_output_connected"
        ) is True:
            raise RuntimeError(
                "Motion output must be disconnected "
                "during operator localization."
            )

        if (
            status.get(
                "motion_egress_ready"
            ) is not True
            or status.get(
                "motion_egress_idle"
            ) is not True
        ):
            raise RuntimeError(
                "Motion egress must be running and "
                "disarmed during operator localization."
            )

        if not self.initial_pose_script.is_file():
            raise RuntimeError(
                "Tony2 operator initial-pose helper "
                "is missing."
            )

        command = [
            "/usr/bin/python3",
            "-u",
            str(
                self.initial_pose_script
            ),
            (
                "--x="
                + str(
                    normalized["x"]
                )
            ),
            (
                "--y="
                + str(
                    normalized["y"]
                )
            ),
            (
                "--yaw="
                + str(
                    normalized["yaw"]
                )
            ),
            "--ros-args",
            "-r",
            "/tf:=/nav_tf",
        ]

        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self.child_environment(),
                timeout=35.0,
                check=False,
            )

        except Exception:
            self.stop()
            raise

        payload = None

        for line in reversed(
            completed.stdout.splitlines()
        ):
            line = line.strip()

            if not line:
                continue

            try:
                candidate = json.loads(
                    line
                )

            except json.JSONDecodeError:
                continue

            if isinstance(
                candidate,
                dict,
            ):
                payload = candidate
                break

        if (
            completed.returncode != 0
            or not isinstance(
                payload,
                dict,
            )
            or payload.get("ok") is not True
        ):
            stopped = self.stop()

            error = (
                payload.get(
                    "error"
                )
                if isinstance(
                    payload,
                    dict,
                )
                else None
            )

            raise RuntimeError(
                error
                or (
                    "Tony2 operator localization "
                    "helper failed."
                )
            )

        if payload.get(
            "trusted"
        ) is not True:
            stopped = self.stop()

            return {
                "action":
                    "OPERATOR_POSE_REJECTED",
                "localization":
                    payload,
                "navigation":
                    stopped["navigation"],
            }

        deadline = (
            time.monotonic()
            + 10.0
        )

        final_status = self.status()

        while (
            final_status.get(
                "state"
            ) != "READY"
            and time.monotonic()
                < deadline
        ):
            time.sleep(
                0.2
            )

            final_status = (
                self.status()
            )

        if final_status.get(
            "state"
        ) != "READY":
            stopped = self.stop()

            return {
                "action":
                    "OPERATOR_POSE_REJECTED",
                "localization":
                    {
                        **payload,
                        "trusted": False,
                        "error": (
                            "Localization passed its "
                            "pose checks but Nav2 did not "
                            "reach READY."
                        ),
                    },
                "navigation":
                    stopped["navigation"],
            }

        return {
            "action":
                "OPERATOR_POSE_VALIDATED",
            "localization":
                payload,
            "navigation":
                final_status,
        }

    def compute_path(
        self,
        x,
        y,
        yaw,
    ):
        """
        Compute one map-frame path without execution.

        This method is planning-only. Physical controller output
        must remain disconnected for the entire operation, and
        no guarded execution ownership is created here.
        """
        normalized = {}

        for key, value in (
            ("x", x),
            ("y", y),
            ("yaw", yaw),
        ):
            if isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    f"{key} must be numeric."
                )

            try:
                value = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{key} must be numeric."
                ) from exc

            if not math.isfinite(value):
                raise ValueError(
                    f"{key} must be finite."
                )

            normalized[key] = value

        plan_script = (
            Path(__file__).resolve().parent
            / "tony2_navigation_plan.py"
        )

        plan_result_file = (
            self.runtime_dir
            / "tony2_navigation_plan_result.json"
        )

        if not plan_script.is_file():
            raise RuntimeError(
                "Tony2 read-only planning helper "
                "is missing."
            )

        # Serialize planning against guarded motion ownership.
        #
        # _motion_lock is an RLock. Holding it here prevents a
        # concurrent GO from creating a transient motion lease
        # while the read-only planner action is in flight.
        with self._motion_lock:
            status = self.status()

            if (
                status.get("state")
                != "READY"
                or status.get("running")
                is not True
                or status.get("owned")
                is not True
                or status.get(
                    "planner_enabled"
                )
                is not True
                or status.get(
                    "transform_ready"
                )
                is not True
            ):
                raise RuntimeError(
                    "Tony2 isolated Nav2 planner "
                    "must be READY before path computation."
                )

            if status.get(
                "motion_output_connected"
            ) is not False:
                raise RuntimeError(
                    "Physical motion output must remain "
                    "disconnected during read-only planning."
                )

            if status.get(
                "goal_active"
            ) is not False:
                raise RuntimeError(
                    "A navigation execution goal is active."
                )

            if (
                self._active_motion_lease
                is not None
                or self._active_motion_token
                is not None
                or self._goal_process
                is not None
            ):
                raise RuntimeError(
                    "Guarded motion ownership must be "
                    "completely idle during read-only planning."
                )

            try:
                plan_result_file.unlink()

            except FileNotFoundError:
                pass

            command = [
                "/usr/bin/python3",
                "-u",
                str(plan_script),
                (
                    "--goal-x="
                    + repr(
                        normalized["x"]
                    )
                ),
                (
                    "--goal-y="
                    + repr(
                        normalized["y"]
                    )
                ),
                (
                    "--goal-yaw="
                    + repr(
                        normalized["yaw"]
                    )
                ),
                (
                    "--result-file="
                    + str(
                        plan_result_file
                    )
                ),
            ]

            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=25.0,
                    env=self.child_environment(),
                    check=False,
                )

            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Tony2 read-only path computation "
                    "timed out."
                ) from exc

            if (
                completed.returncode != 0
                or not plan_result_file.is_file()
            ):
                output = (
                    completed.stdout
                    or ""
                ).strip()

                if len(output) > 1200:
                    output = output[-1200:]

                raise RuntimeError(
                    "Tony2 read-only path computation "
                    "failed."
                    + (
                        f" {output}"
                        if output
                        else ""
                    )
                )

            try:
                result = json.loads(
                    plan_result_file.read_text(
                        encoding="utf-8"
                    )
                )

            except (
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    "Tony2 planner returned an invalid "
                    "result file."
                ) from exc

            if (
                not isinstance(
                    result,
                    dict,
                )
                or result.get("ok")
                is not True
                or result.get("action")
                != "COMPUTE_PATH_TO_POSE_ONLY"
                or result.get("read_only")
                is not True
                or result.get("executed")
                is not False
                or result.get(
                    "navigation_goal_executed"
                )
                is not False
                or result.get(
                    "motion_enabled"
                )
                is not False
                or result.get("frame_id")
                != "map"
                or not isinstance(
                    result.get("poses"),
                    list,
                )
                or len(
                    result.get("poses")
                ) < 2
            ):
                raise RuntimeError(
                    "Tony2 planner result violated "
                    "the read-only path contract."
                )

            final_status = self.status()

            if (
                final_status.get(
                    "motion_output_connected"
                )
                is not False
                or final_status.get(
                    "goal_active"
                )
                is not False
            ):
                raise RuntimeError(
                    "Navigation execution state changed "
                    "during read-only planning."
                )

            return result


    def submit_goal(
        self,
        x,
        y,
        yaw,
    ):
        """
        Execute one explicitly authorized guarded map goal.

        GO creates one short-lived lease, requires the egress
        to acknowledge the exact token, launches one bounded
        NavigateToPose helper, then destroys authorization.
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
            (
                "--x="
                + str(normalized["x"])
            ),
            (
                "--y="
                + str(normalized["y"])
            ),
            (
                "--yaw="
                + str(normalized["yaw"])
            ),
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

        lease = None
        token = None
        generation = None
        process = None
        return_code = None
        disarm_confirmed = True

        try:
            (
                lease,
                token,
                generation,
            ) = self._begin_motion_lease()

            # No NavigateToPose process exists yet.
            self._wait_for_motion_arm_ack(
                lease,
                token,
                generation,
            )

            log_handle = open(
                self.goal_log,
                "ab",
                buffering=0,
            )

            try:
                # Atomic with STOP:
                # validate authorization, spawn, register the
                # Popen object, and persist its PID before this
                # lock can be acquired by STOP.
                with self._motion_lock:
                    if not self._motion_lease_is_current(
                        lease,
                        token,
                        generation,
                    ):
                        raise RuntimeError(
                            "Tony2 guarded GO was "
                            "cancelled before goal launch."
                        )

                    locked_status = self.status()

                    if (
                        locked_status.get("state")
                        != "READY"
                    ):
                        raise RuntimeError(
                            "Tony2 guarded navigation "
                            "stopped before goal launch."
                        )

                    if locked_status.get(
                        "goal_active"
                    ):
                        raise RuntimeError(
                            "A Tony2 navigation goal "
                            "became active concurrently."
                        )

                    if locked_status.get(
                        "motion_output_connected"
                    ) is not True:
                        raise RuntimeError(
                            "Tony2 motion egress lost "
                            "the transient GO authorization."
                        )

                    if (
                        self._stop_requested.is_set()
                        or self._stop_generation
                            != generation
                    ):
                        raise RuntimeError(
                            "Tony2 guarded GO was "
                            "cancelled by STOP."
                        )

                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=self.child_environment(),
                        start_new_session=True,
                    )

                    self._goal_process = process

                    self._write_pid(
                        self.goal_pid_file,
                        process.pid,
                    )

                    # STOP sets its event before waiting for
                    # this lock. Detect a STOP that arrived
                    # during the actual Popen call.
                    if (
                        self._stop_requested.is_set()
                        or self._stop_generation
                            != generation
                    ):
                        raise RuntimeError(
                            "Tony2 guarded GO was "
                            "cancelled during goal launch."
                        )

            finally:
                log_handle.close()

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

            with self._motion_lock:
                if (
                    self._stop_requested.is_set()
                    or self._stop_generation
                        != generation
                ):
                    raise RuntimeError(
                        "Tony2 guarded GO was "
                        "cancelled by STOP."
                    )

        except Exception:
            if process is not None:
                try:
                    running = (
                        process.poll()
                        is None
                    )
                except Exception:
                    running = True

                if running:
                    self._terminate_group(
                        process.pid,
                        self.GOAL_MARKER,
                    )

            # A helper may already have written its result
            # before another guarded execution step failed.
            # Failed GO paths must not strand that artifact.
            self._safe_unlink(
                self.goal_result_file
            )

            raise

        finally:
            with self._motion_lock:
                if self._goal_process is process:
                    self._goal_process = None

            self._safe_unlink(
                self.goal_pid_file
            )

            if lease is not None:
                try:
                    released = (
                        self._release_motion_lease(
                            lease
                        )
                    )

                    if released:
                        # Egress reports idle only after it has
                        # disarmed its controller. Controller
                        # disarm requests Robot Bridge STOP.
                        disarm_confirmed = (
                            self._wait_for_motion_disarm()
                        )

                except Exception:
                    # Release/disarm happens inside this
                    # finally block, so clean the result here
                    # before propagating its failure.
                    self._safe_unlink(
                        self.goal_result_file
                    )

                    raise

        if not disarm_confirmed:
            self._safe_unlink(
                self.goal_result_file
            )

            raise RuntimeError(
                "Tony2 motion egress did not confirm "
                "disarm after guarded goal execution."
            )

        # A STOP that completed while this GO was active
        # permanently changes the generation, even though
        # _stop_requested has since been cleared.
        with self._motion_lock:
            if (
                generation is not None
                and self._stop_generation
                    != generation
            ):
                self._safe_unlink(
                    self.goal_result_file
                )

                raise RuntimeError(
                    "Tony2 guarded GO was "
                    "cancelled by STOP."
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
