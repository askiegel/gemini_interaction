#!/usr/bin/env python3

"""
Fail-closed transient motion-arm lease for Tony2 Nav2.

This module is pure stdlib.

Nothing here sends robot motion.  A valid lease is only a
short-lived authorization signal consumed by the separate
guarded motion-egress process.
"""

import json
import math
import os
import secrets
import threading
import time
from pathlib import Path


ARM_LEASE_SECONDS = 0.50
ARM_REFRESH_SECONDS = 0.10

MINIMUM_LEASE_SECONDS = 0.20
MAXIMUM_LEASE_SECONDS = 1.00


def read_json_file(path):
    path = Path(path)

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def atomic_write_json(
    path,
    payload,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        (
            path.name
            + ".tmp."
            + str(os.getpid())
            + "."
            + secrets.token_hex(4)
        )
    )

    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def validate_arm_payload(
    payload,
    *,
    now=None,
):
    if not isinstance(payload, dict):
        return None

    token = payload.get("token")
    robot_bridge_url = payload.get(
        "robot_bridge_url"
    )

    expires_at = payload.get(
        "expires_at"
    )

    if (
        not isinstance(token, str)
        or len(token) < 16
    ):
        return None

    if (
        not isinstance(robot_bridge_url, str)
        or not robot_bridge_url.startswith(
            ("http://", "https://")
        )
    ):
        return None

    try:
        expires_at = float(
            expires_at
        )
    except (TypeError, ValueError):
        return None

    if not math.isfinite(
        expires_at
    ):
        return None

    now = (
        time.time()
        if now is None
        else float(now)
    )

    if expires_at <= now:
        return None

    return {
        "token": token,
        "robot_bridge_url":
            robot_bridge_url.rstrip("/"),
        "expires_at": expires_at,
    }


class MotionArmLease:
    """
    Continuously refresh one short-lived arm lease.

    Stopping the refresher removes the lease. If this
    process dies, the last written lease expires shortly
    afterward instead of remaining armed indefinitely.
    """

    def __init__(
        self,
        path,
        robot_bridge_url,
        *,
        lease_seconds=ARM_LEASE_SECONDS,
        refresh_seconds=ARM_REFRESH_SECONDS,
    ):
        self.path = Path(path)

        self.robot_bridge_url = (
            str(robot_bridge_url)
            .rstrip("/")
        )

        self.lease_seconds = float(
            lease_seconds
        )

        self.refresh_seconds = float(
            refresh_seconds
        )

        if not (
            MINIMUM_LEASE_SECONDS
            <= self.lease_seconds
            <= MAXIMUM_LEASE_SECONDS
        ):
            raise ValueError(
                "lease_seconds outside guarded range."
            )

        if not (
            0.0
            < self.refresh_seconds
            < self.lease_seconds
        ):
            raise ValueError(
                "refresh_seconds must be positive "
                "and shorter than lease_seconds."
            )

        if not self.robot_bridge_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "robot_bridge_url must be HTTP(S)."
            )

        self.token = secrets.token_hex(
            24
        )

        self._stop = threading.Event()
        self._thread = None

    def _payload(self):
        return {
            "token": self.token,
            "robot_bridge_url":
                self.robot_bridge_url,
            "expires_at": (
                time.time()
                + self.lease_seconds
            ),
        }

    def refresh(self):
        payload = self._payload()

        atomic_write_json(
            self.path,
            payload,
        )

        return payload

    def _run(self):
        while not self._stop.is_set():
            self.refresh()

            self._stop.wait(
                self.refresh_seconds
            )

    def start(self):
        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            raise RuntimeError(
                "Motion arm lease already running."
            )

        self._stop.clear()

        # Write synchronously before returning.
        self.refresh()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="tony2-navigation-motion-arm",
        )

        self._thread.start()

        return self.token

    def stop(self):
        self._stop.set()

        thread = self._thread

        if thread is not None:
            thread.join(
                timeout=1.0
            )

        self._thread = None

        current = read_json_file(
            self.path
        )

        if (
            isinstance(current, dict)
            and current.get("token")
            != self.token
        ):
            return False

        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

        return True
