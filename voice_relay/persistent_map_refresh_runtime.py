#!/usr/bin/env python3

"""
Guarded webpage-controlled stationary persistent-map refresh.

The runtime:
- never commands robot motion;
- uses the already validated stationary_map_builder.py;
- captures a candidate without replacing the active map;
- requires explicit promotion;
- stores promoted runtime maps outside the Git checkout;
- supports candidate preview, discard, and cancellation.

Fixed-map navigation reads the promoted map through
MAYDAY_FIXED_MAP_YAML. If no dashboard-promoted map exists,
navigation falls back to the bundled validated map.
"""

import ast
import copy
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from pathlib import Path


CAPTURE_SECONDS = 30
MINIMUM_SCANS = 100

DASHBOARD_LIDAR_URL = (
    "http://127.0.0.1:8765/dashboard/lidar"
)

ROBOT_BRIDGE_URL = (
    "http://192.168.68.124:8090"
)

ROBOT_STATUS_URL = (
    ROBOT_BRIDGE_URL
    + "/status"
)

ACTIVE_MAP_BASENAME = (
    "mayday_supervised_route_03"
)

CANDIDATE_PREFIX = (
    "mayday_stationary_candidate_"
)


def _sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _simple_yaml(path):
    result = {}

    for raw in Path(path).read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw.strip()

        if (
            not line
            or line.startswith("#")
            or ":" not in line
        ):
            continue

        key, value = line.split(
            ":",
            1,
        )

        result[
            key.strip()
        ] = value.strip()

    return result


def _origin_from_yaml(value):
    parsed = ast.literal_eval(
        value
    )

    if (
        not isinstance(
            parsed,
            (list, tuple),
        )
        or len(parsed) != 3
    ):
        raise RuntimeError(
            "Map YAML origin must contain x, y, yaw."
        )

    return tuple(
        float(item)
        for item in parsed
    )


def _pgm_tokens(raw):
    index = 0
    tokens = []

    while len(tokens) < 4:
        while (
            index < len(raw)
            and chr(
                raw[index]
            ).isspace()
        ):
            index += 1

        if (
            index < len(raw)
            and raw[index:index + 1]
            == b"#"
        ):
            while (
                index < len(raw)
                and raw[index:index + 1]
                not in (b"\r", b"\n")
            ):
                index += 1

            continue

        start = index

        while (
            index < len(raw)
            and not chr(
                raw[index]
            ).isspace()
        ):
            index += 1

        if start == index:
            raise RuntimeError(
                "Invalid PGM header."
            )

        tokens.append(
            raw[start:index].decode(
                "ascii"
            )
        )

    while (
        index < len(raw)
        and chr(
            raw[index]
        ).isspace()
    ):
        index += 1

    return tokens, index


def _load_pgm(path):
    raw = Path(path).read_bytes()

    tokens, offset = _pgm_tokens(
        raw
    )

    magic = tokens[0]
    width = int(
        tokens[1]
    )
    height = int(
        tokens[2]
    )
    maximum = int(
        tokens[3]
    )

    if (
        width <= 0
        or height <= 0
        or maximum <= 0
    ):
        raise RuntimeError(
            "Invalid PGM dimensions."
        )

    count = (
        width
        * height
    )

    if magic == "P5":
        payload = raw[
            offset:
            offset + count
        ]

        if len(payload) != count:
            raise RuntimeError(
                "PGM pixel payload is truncated."
            )

        pixels = list(
            payload
        )

    elif magic == "P2":
        text = raw[
            offset:
        ].decode(
            "ascii",
            errors="strict",
        )

        cleaned = []

        for line in text.splitlines():
            line = line.split(
                "#",
                1,
            )[0]

            cleaned.extend(
                line.split()
            )

        pixels = [
            int(value)
            for value in cleaned[
                :count
            ]
        ]

        if len(pixels) != count:
            raise RuntimeError(
                "ASCII PGM pixel payload is truncated."
            )

    else:
        raise RuntimeError(
            "Unsupported PGM format: "
            + magic
        )

    if maximum != 255:
        pixels = [
            int(
                round(
                    value
                    * 255.0
                    / maximum
                )
            )
            for value in pixels
        ]

    return (
        width,
        height,
        pixels,
    )


def _map_payload(
    yaml_path,
    *,
    name,
    source_type,
):
    yaml_path = Path(
        yaml_path
    ).resolve()

    config = _simple_yaml(
        yaml_path
    )

    image_value = (
        config.get("image")
        or ""
    ).strip(
        "\"'"
    )

    if not image_value:
        raise RuntimeError(
            "Map YAML image is missing."
        )

    image_path = Path(
        image_value
    )

    if not image_path.is_absolute():
        image_path = (
            yaml_path.parent
            / image_path
        )

    image_path = (
        image_path.resolve()
    )

    resolution = float(
        config["resolution"]
    )

    origin_x, origin_y, origin_yaw = (
        _origin_from_yaml(
            config["origin"]
        )
    )

    negate = int(
        config.get(
            "negate",
            "0",
        )
    )

    occupied_thresh = float(
        config.get(
            "occupied_thresh",
            "0.65",
        )
    )

    free_thresh = float(
        config.get(
            "free_thresh",
            "0.196",
        )
    )

    (
        width,
        height,
        pixels,
    ) = _load_pgm(
        image_path
    )

    rows = [
        pixels[
            row * width:
            (row + 1) * width
        ]
        for row in range(
            height
        )
    ]

    # PGM is stored top-to-bottom. ROS OccupancyGrid row zero
    # begins at the map origin, so invert image row order.
    rows.reverse()

    data = []

    free_count = 0
    occupied_count = 0
    unknown_count = 0

    for row in rows:
        for pixel in row:
            if negate:
                occupancy = (
                    pixel
                    / 255.0
                )
            else:
                occupancy = (
                    255.0
                    - pixel
                ) / 255.0

            if (
                occupancy
                > occupied_thresh
            ):
                value = 100
                occupied_count += 1

            elif (
                occupancy
                < free_thresh
            ):
                value = 0
                free_count += 1

            else:
                value = -1
                unknown_count += 1

            data.append(
                value
            )

    return {
        "ok": True,
        "status": "READY",
        "map": {
            "name": name,
            "source_type":
                source_type,
            "frame_id": "map",
            "resolution":
                resolution,
            "width":
                width,
            "height":
                height,
            "origin": {
                "x": origin_x,
                "y": origin_y,
                "yaw": origin_yaw,
            },
            "data":
                data,
            "encoding":
                "ros_occupancy_values",
            "unknown_value":
                -1,
            "free_value":
                0,
            "occupied_value":
                100,
            "unknown_cell_count":
                unknown_count,
            "free_cell_count":
                free_count,
            "occupied_cell_count":
                occupied_count,
            "source": {
                "yaml_name":
                    yaml_path.name,
                "image_name":
                    image_path.name,
                "yaml_sha256":
                    _sha256(
                        yaml_path
                    ),
                "image_sha256":
                    _sha256(
                        image_path
                    ),
            },
        },
    }


class PersistentMapRefreshRuntime:
    """
    Own one stationary refresh at a time.

    Candidate construction and map promotion are intentionally
    separate operations.
    """

    ACTIVE_PHASES = {
        "CAPTURING",
        "BUILDING",
        "PROMOTING",
    }

    def __init__(
        self,
        *,
        module_dir=None,
        state_root=None,
        candidate_root=None,
        builder_script=None,
    ):
        self.module_dir = Path(
            module_dir
            if module_dir is not None
            else Path(
                __file__
            ).resolve().parent
        )

        self.repo_root = (
            self.module_dir.parent
        )

        self.builder_script = Path(
            builder_script
            if builder_script is not None
            else (
                self.module_dir
                / "stationary_map_builder.py"
            )
        )

        self.candidate_root = Path(
            candidate_root
            if candidate_root is not None
            else "/tmp"
        )

        self.state_root = Path(
            state_root
            if state_root is not None
            else (
                Path.home()
                / ".local"
                / "share"
                / "mayday"
                / "persistent_map"
            )
        )

        self.releases_dir = (
            self.state_root
            / "releases"
        )

        self.active_link = (
            self.state_root
            / "active"
        )

        self.bundled_asset_dir = (
            self.module_dir
            / "tony2_navigation_assets"
        )

        self._lock = (
            threading.RLock()
        )

        self._process = None
        self._worker = None

        self._state = {
            "phase": (
                "ACTIVE"
                if self.active_map_yaml().is_file()
                else "IDLE"
            ),
            "started_at":
                None,
            "finished_at":
                None,
            "progress_percent":
                0,
            "candidate":
                None,
            "last_promoted":
                None,
            "error":
                None,
            "log":
                None,
        }

    def active_map_yaml(
        self,
    ):
        return (
            self.active_link
            / (
                ACTIVE_MAP_BASENAME
                + ".yaml"
            )
        )

    def active_map_pgm(
        self,
    ):
        return (
            self.active_link
            / (
                ACTIVE_MAP_BASENAME
                + ".pgm"
            )
        )

    def busy(
        self,
    ):
        with self._lock:
            return (
                self._state[
                    "phase"
                ]
                in self.ACTIVE_PHASES
            )

    def _candidate_dirs(
        self,
    ):
        return {
            path.resolve()
            for path in (
                self.candidate_root.glob(
                    CANDIDATE_PREFIX
                    + "*"
                )
            )
            if path.is_dir()
        }

    def _builder_command(
        self,
    ):
        if not self.builder_script.is_file():
            raise RuntimeError(
                "stationary_map_builder.py is missing."
            )

        completed = subprocess.run(
            [
                sys.executable,
                str(
                    self.builder_script
                ),
                "--help",
            ],
            cwd=str(
                self.repo_root
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )

        help_text = (
            completed.stdout
            or ""
        )

        duration_options = (
            "--capture-seconds",
            "--capture-duration-seconds",
            "--duration-seconds",
            "--capture-duration",
            "--duration",
            "--seconds",
        )

        duration_option = next(
            (
                option
                for option
                in duration_options
                if option in help_text
            ),
            None,
        )

        if duration_option is None:
            raise RuntimeError(
                "Stationary map builder has no recognized "
                "capture-duration option."
            )

        candidate_name = (
            CANDIDATE_PREFIX
            + time.strftime(
                "%Y%m%dT%H%M%SZ",
                time.gmtime(),
            )
            + "_"
            + uuid.uuid4().hex[:8]
        )

        candidate_dir = (
            self.candidate_root
            / candidate_name
        )

        command = [
            sys.executable,
            "-u",
            str(
                self.builder_script
            ),
            duration_option,
            str(
                CAPTURE_SECONDS
            ),
            "--output-dir",
            str(
                candidate_dir
            ),
            "--name",
            candidate_name,
            "--minimum-scans",
            str(
                MINIMUM_SCANS
            ),
        ]

        lidar_options = (
            "--dashboard-lidar-url",
            "--lidar-url",
            "--lidar-endpoint",
            "--scan-url",
        )

        for option in lidar_options:
            if option in help_text:
                command.extend(
                    [
                        option,
                        DASHBOARD_LIDAR_URL,
                    ]
                )
                break

        robot_status_options = (
            "--robot-status-url",
            "--status-url",
        )

        robot_added = False

        for option in robot_status_options:
            if option in help_text:
                command.extend(
                    [
                        option,
                        ROBOT_STATUS_URL,
                    ]
                )
                robot_added = True
                break

        if (
            not robot_added
            and "--robot-url"
            in help_text
        ):
            command.extend(
                [
                    "--robot-url",
                    ROBOT_BRIDGE_URL,
                ]
            )

        return command

    def _validate_candidate(
        self,
        candidate_dir,
    ):
        candidate_dir = Path(
            candidate_dir
        ).resolve()

        json_files = sorted(
            candidate_dir.glob(
                "*.json"
            )
        )

        yaml_files = sorted(
            candidate_dir.glob(
                "*.yaml"
            )
        )

        pgm_files = sorted(
            candidate_dir.glob(
                "*.pgm"
            )
        )

        if (
            len(json_files) != 1
            or len(yaml_files) != 1
            or len(pgm_files) != 1
        ):
            raise RuntimeError(
                "Stationary candidate does not contain "
                "exactly one JSON/YAML/PGM set."
            )

        metadata = json.loads(
            json_files[0].read_text(
                encoding="utf-8"
            )
        )

        summary = (
            metadata.get(
                "summary"
            )
            if isinstance(
                metadata,
                dict,
            )
            else None
        )

        if not isinstance(
            summary,
            dict,
        ):
            summary = metadata

        capture = (
            summary.get(
                "capture"
            )
            or {}
        )

        accepted_scans = int(
            capture.get(
                "accepted_scans",
                summary.get(
                    "scan_count",
                    0,
                ),
            )
        )

        checks = {
            "minimum_scans":
                accepted_scans
                >= MINIMUM_SCANS,
            "motion_verified":
                capture.get(
                    "motion_verified"
                ) is True,
            "validated_not_replaced":
                summary.get(
                    "validated_map_replaced"
                ) is False,
            "no_cartographer":
                summary.get(
                    "cartographer_used"
                ) is False,
            "no_navigation":
                summary.get(
                    "navigation_used"
                ) is False,
            "no_motion":
                summary.get(
                    "motion_commanded"
                ) is False,
            "stable_geometry":
                int(
                    summary.get(
                        "stable_bin_count",
                        0,
                    )
                ) > 0,
            "occupied_geometry":
                int(
                    summary.get(
                        "occupied_cell_count",
                        0,
                    )
                ) > 0,
            "free_space":
                int(
                    summary.get(
                        "free_cell_count",
                        0,
                    )
                ) > 0,
        }

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        if failed:
            raise RuntimeError(
                "Candidate validation failed: "
                + ", ".join(
                    failed
                )
            )

        name = str(
            summary.get(
                "name"
            )
            or candidate_dir.name
        )

        return {
            "name":
                name,
            "directory":
                str(
                    candidate_dir
                ),
            "metadata":
                str(
                    json_files[0]
                ),
            "yaml":
                str(
                    yaml_files[0]
                ),
            "pgm":
                str(
                    pgm_files[0]
                ),
            "accepted_scans":
                accepted_scans,
            "scan_count":
                int(
                    summary.get(
                        "scan_count",
                        accepted_scans,
                    )
                ),
            "stable_bin_count":
                int(
                    summary.get(
                        "stable_bin_count",
                        0,
                    )
                ),
            "free_cell_count":
                int(
                    summary.get(
                        "free_cell_count",
                        0,
                    )
                ),
            "occupied_cell_count":
                int(
                    summary.get(
                        "occupied_cell_count",
                        0,
                    )
                ),
            "unknown_cell_count":
                int(
                    summary.get(
                        "unknown_cell_count",
                        0,
                    )
                ),
            "resolution":
                float(
                    summary.get(
                        "resolution",
                        0.0,
                    )
                ),
            "width":
                int(
                    summary.get(
                        "width",
                        0,
                    )
                ),
            "height":
                int(
                    summary.get(
                        "height",
                        0,
                    )
                ),
            "motion_verified":
                True,
            "validated_map_replaced":
                False,
            "cartographer_used":
                False,
            "navigation_used":
                False,
            "motion_commanded":
                False,
        }

    def _capture_worker(
        self,
        previous_candidates,
        started,
        log_path,
    ):
        process = None

        try:
            command = (
                self._builder_command()
            )

            with open(
                log_path,
                "wb",
                buffering=0,
            ) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(
                        self.repo_root
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                with self._lock:
                    self._process = (
                        process
                    )

                while True:
                    return_code = (
                        process.poll()
                    )

                    if (
                        return_code
                        is not None
                    ):
                        break

                    elapsed = max(
                        0.0,
                        time.monotonic()
                        - started,
                    )

                    with self._lock:
                        if (
                            self._state[
                                "phase"
                            ]
                            == "CANCELLED"
                        ):
                            return

                        if (
                            elapsed
                            < CAPTURE_SECONDS
                        ):
                            self._state[
                                "phase"
                            ] = "CAPTURING"

                            self._state[
                                "progress_percent"
                            ] = min(
                                90,
                                int(
                                    90
                                    * elapsed
                                    / CAPTURE_SECONDS
                                ),
                            )

                        else:
                            self._state[
                                "phase"
                            ] = "BUILDING"

                            self._state[
                                "progress_percent"
                            ] = 95

                    time.sleep(
                        0.2
                    )

            if return_code != 0:
                tail = ""

                try:
                    tail = (
                        Path(
                            log_path
                        ).read_text(
                            encoding="utf-8",
                            errors="replace",
                        )[-3000:]
                    )

                except Exception:
                    pass

                raise RuntimeError(
                    "Stationary map builder failed "
                    f"with exit code {return_code}. "
                    + tail
                )

            current = (
                self._candidate_dirs()
            )

            candidates = [
                path
                for path
                in (
                    current
                    - previous_candidates
                )
                if (
                    path.stat().st_mtime
                    >= (
                        time.time()
                        - 120.0
                    )
                )
            ]

            if not candidates:
                candidates = [
                    path
                    for path
                    in current
                    if (
                        path.stat().st_mtime
                        >= (
                            time.time()
                            - 120.0
                        )
                    )
                ]

            if not candidates:
                raise RuntimeError(
                    "Builder completed but created "
                    "no stationary candidate directory."
                )

            candidate_dir = max(
                candidates,
                key=lambda path:
                    path.stat().st_mtime,
            )

            candidate = (
                self._validate_candidate(
                    candidate_dir
                )
            )

            with self._lock:
                self._state.update(
                    {
                        "phase":
                            "CANDIDATE_READY",
                        "finished_at":
                            time.time(),
                        "progress_percent":
                            100,
                        "candidate":
                            candidate,
                        "error":
                            None,
                    }
                )

        except Exception as exc:
            with self._lock:
                if (
                    self._state[
                        "phase"
                    ]
                    != "CANCELLED"
                ):
                    self._state.update(
                        {
                            "phase":
                                "ERROR",
                            "finished_at":
                                time.time(),
                            "progress_percent":
                                0,
                            "error":
                                str(exc),
                        }
                    )

        finally:
            with self._lock:
                self._process = None

    def start_refresh(
        self,
    ):
        with self._lock:
            if (
                self._state[
                    "phase"
                ]
                in self.ACTIVE_PHASES
            ):
                raise RuntimeError(
                    "Persistent map refresh is already running."
                )

            if (
                self._state.get(
                    "candidate"
                )
                is not None
            ):
                raise RuntimeError(
                    "A stationary map candidate is already "
                    "waiting for Use New Map or Discard."
                )

            previous = (
                self._candidate_dirs()
            )

            started = (
                time.monotonic()
            )

            log_path = Path(
                "/tmp"
            ) / (
                "mayday_persistent_map_refresh_"
                + time.strftime(
                    "%Y%m%dT%H%M%SZ",
                    time.gmtime(),
                )
                + ".log"
            )

            self._state.update(
                {
                    "phase":
                        "CAPTURING",
                    "started_at":
                        time.time(),
                    "finished_at":
                        None,
                    "progress_percent":
                        0,
                    "candidate":
                        None,
                    "error":
                        None,
                    "log":
                        str(
                            log_path
                        ),
                }
            )

            worker = threading.Thread(
                target=(
                    self._capture_worker
                ),
                args=(
                    previous,
                    started,
                    str(
                        log_path
                    ),
                ),
                daemon=True,
                name=(
                    "mayday-persistent-"
                    "map-refresh"
                ),
            )

            self._worker = (
                worker
            )

            worker.start()

            return self.status()

    def cancel(
        self,
    ):
        with self._lock:
            process = (
                self._process
            )

            if (
                self._state[
                    "phase"
                ]
                not in {
                    "CAPTURING",
                    "BUILDING",
                }
            ):
                return self.status()

            self._state.update(
                {
                    "phase":
                        "CANCELLED",
                    "finished_at":
                        time.time(),
                    "progress_percent":
                        0,
                    "error":
                        None,
                }
            )

        if (
            process is not None
            and process.poll()
            is None
        ):
            try:
                os.killpg(
                    process.pid,
                    signal.SIGTERM,
                )

            except ProcessLookupError:
                pass

            try:
                process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(
                        process.pid,
                        signal.SIGKILL,
                    )

                except ProcessLookupError:
                    pass

        return self.status()

    def discard(
        self,
    ):
        with self._lock:
            candidate = (
                copy.deepcopy(
                    self._state.get(
                        "candidate"
                    )
                )
            )

            if candidate is None:
                self._state.update(
                    {
                        "phase": (
                            "ACTIVE"
                            if self.active_map_yaml().is_file()
                            else "IDLE"
                        ),
                        "error":
                            None,
                        "progress_percent":
                            0,
                    }
                )

                return self.status()

            candidate_dir = Path(
                candidate[
                    "directory"
                ]
            ).resolve()

            allowed_root = (
                self.candidate_root.resolve()
            )

            if (
                candidate_dir.parent
                != allowed_root
                or not candidate_dir.name.startswith(
                    CANDIDATE_PREFIX
                )
            ):
                raise RuntimeError(
                    "Refusing to discard an unexpected path."
                )

            self._state.update(
                {
                    "candidate":
                        None,
                    "phase": (
                        "ACTIVE"
                        if self.active_map_yaml().is_file()
                        else "IDLE"
                    ),
                    "progress_percent":
                        0,
                    "error":
                        None,
                }
            )

        shutil.rmtree(
            candidate_dir,
            ignore_errors=True,
        )

        return self.status()

    def candidate_map_payload(
        self,
    ):
        with self._lock:
            candidate = (
                copy.deepcopy(
                    self._state.get(
                        "candidate"
                    )
                )
            )

        if candidate is None:
            return None

        payload = _map_payload(
            candidate["yaml"],
            name=candidate["name"],
            source_type=(
                "stationary_lidar_candidate"
            ),
        )

        payload[
            "candidate"
        ] = candidate

        return payload

    def active_map_payload(
        self,
    ):
        """
        Present a dashboard-promoted map through the exact
        read-only telemetry contract already consumed by the
        persistent-map renderer and planning overlay.

        Candidate review continues to use its own candidate
        payload. This adapter exists only for the active map.
        """
        yaml_path = (
            self.active_map_yaml()
        )

        if not yaml_path.is_file():
            return None

        parsed = _map_payload(
            yaml_path,
            name=ACTIVE_MAP_BASENAME,
            source_type=(
                "dashboard_promoted_"
                "persistent_map"
            ),
        )

        raw_map = (
            parsed.get("map")
        )

        if not isinstance(
            raw_map,
            dict,
        ):
            raise RuntimeError(
                "Promoted persistent map payload "
                "contains no occupancy grid."
            )

        data = (
            raw_map.get("data")
        )

        width = int(
            raw_map.get(
                "width",
                0,
            )
        )

        height = int(
            raw_map.get(
                "height",
                0,
            )
        )

        if (
            not isinstance(
                data,
                list,
            )
            or width <= 0
            or height <= 0
            or len(data)
                != width * height
        ):
            raise RuntimeError(
                "Promoted persistent map occupancy "
                "grid is invalid."
            )

        dashboard_map = (
            copy.deepcopy(
                raw_map
            )
        )

        # Existing dashboard map and planning code use
        # occupancyMap.cells. Keep data as well so the
        # promoted-map representation remains compatible with
        # candidate/review readers.
        dashboard_map[
            "cells"
        ] = list(
            data
        )

        return {
            "ok":
                True,
            "status":
                "READY",
            "source":
                "dashboard_promoted_persistent_map",
            "read_only":
                True,
            "authoritative":
                True,
            "persistent_map_source":
                "dashboard_promoted",
            "telemetry": {
                "available":
                    True,
                "status":
                    "MAP_READY",
                "map":
                    dashboard_map,
            },

            # Preserve the generic map object for review code
            # that recursively accepts either data or cells.
            "map":
                raw_map,
        }


    def promote(
        self,
    ):
        with self._lock:
            if (
                self._state[
                    "phase"
                ]
                != "CANDIDATE_READY"
            ):
                raise RuntimeError(
                    "No validated stationary candidate "
                    "is ready for promotion."
                )

            candidate = (
                copy.deepcopy(
                    self._state[
                        "candidate"
                    ]
                )
            )

            self._state[
                "phase"
            ] = "PROMOTING"

            self._state[
                "progress_percent"
            ] = 100

        candidate_yaml = Path(
            candidate[
                "yaml"
            ]
        )

        candidate_pgm = Path(
            candidate[
                "pgm"
            ]
        )

        release_id = (
            time.strftime(
                "%Y%m%dT%H%M%SZ",
                time.gmtime(),
            )
            + "_"
            + uuid.uuid4().hex[:8]
        )

        release_dir = (
            self.releases_dir
            / release_id
        )

        release_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        active_yaml_name = (
            ACTIVE_MAP_BASENAME
            + ".yaml"
        )

        active_pgm_name = (
            ACTIVE_MAP_BASENAME
            + ".pgm"
        )

        release_yaml = (
            release_dir
            / active_yaml_name
        )

        release_pgm = (
            release_dir
            / active_pgm_name
        )

        try:
            shutil.copy2(
                candidate_pgm,
                release_pgm,
            )

            yaml_text = (
                candidate_yaml.read_text(
                    encoding="utf-8"
                )
            )

            replaced, count = (
                re.subn(
                    r"(?m)^(\s*image\s*:\s*).+$",
                    (
                        r"\1"
                        + active_pgm_name
                    ),
                    yaml_text,
                    count=1,
                )
            )

            if count != 1:
                raise RuntimeError(
                    "Candidate YAML image entry "
                    "could not be rewritten."
                )

            release_yaml.write_text(
                replaced,
                encoding="utf-8",
            )

            promotion_metadata = {
                "promoted_at_unix_seconds":
                    time.time(),
                "release_id":
                    release_id,
                "source_candidate":
                    candidate,
                "active_yaml":
                    str(
                        release_yaml
                    ),
                "active_pgm":
                    str(
                        release_pgm
                    ),
                "yaml_sha256":
                    _sha256(
                        release_yaml
                    ),
                "pgm_sha256":
                    _sha256(
                        release_pgm
                    ),
                "navigation_frame":
                    "candidate_local",
                "robot_pose_at_candidate_origin":
                    {
                        "x": 0.0,
                        "y": 0.0,
                        "yaw": 0.0,
                    },
            }

            (
                release_dir
                / "PROMOTION_METADATA.json"
            ).write_text(
                json.dumps(
                    promotion_metadata,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            # Verify the release can actually be parsed as a ROS map
            # before switching the active symlink.
            _map_payload(
                release_yaml,
                name=ACTIVE_MAP_BASENAME,
                source_type=(
                    "dashboard_promoted_"
                    "persistent_map"
                ),
            )

            self.state_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_link = (
                self.state_root
                / (
                    ".active-"
                    + uuid.uuid4().hex
                )
            )

            os.symlink(
                str(
                    release_dir.resolve()
                ),
                str(
                    temporary_link
                ),
            )

            if (
                self.active_link.exists()
                and not self.active_link.is_symlink()
            ):
                raise RuntimeError(
                    "Persistent-map active path exists "
                    "but is not a managed symlink."
                )

            os.replace(
                temporary_link,
                self.active_link,
            )

            active_payload = (
                self.active_map_payload()
            )

            if active_payload is None:
                raise RuntimeError(
                    "Promoted active map could not be reloaded."
                )

        except Exception:
            with self._lock:
                self._state[
                    "phase"
                ] = "ERROR"

                self._state[
                    "error"
                ] = (
                    "Persistent map promotion failed."
                )

            raise

        candidate_dir = Path(
            candidate[
                "directory"
            ]
        ).resolve()

        shutil.rmtree(
            candidate_dir,
            ignore_errors=True,
        )

        with self._lock:
            self._state.update(
                {
                    "phase":
                        "ACTIVE",
                    "finished_at":
                        time.time(),
                    "progress_percent":
                        100,
                    "candidate":
                        None,
                    "last_promoted":
                        promotion_metadata,
                    "error":
                        None,
                }
            )

        return self.status()

    def status(
        self,
    ):
        with self._lock:
            result = copy.deepcopy(
                self._state
            )

        active_yaml = (
            self.active_map_yaml()
        )

        result.update(
            {
                "capture_seconds":
                    CAPTURE_SECONDS,
                "minimum_scans":
                    MINIMUM_SCANS,
                "busy":
                    result[
                        "phase"
                    ]
                    in self.ACTIVE_PHASES,
                "candidate_ready":
                    result.get(
                        "candidate"
                    )
                    is not None,
                "active_map_override":
                    active_yaml.is_file(),
                "active_map_yaml": (
                    str(
                        active_yaml
                    )
                    if active_yaml.is_file()
                    else None
                ),
                "fallback_map_yaml":
                    str(
                        self.bundled_asset_dir
                        / (
                            ACTIVE_MAP_BASENAME
                            + ".yaml"
                        )
                    ),
            }
        )

        return result


_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()


def get_persistent_map_refresh_runtime():
    global _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = (
                PersistentMapRefreshRuntime()
            )

        return _RUNTIME
