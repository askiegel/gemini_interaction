#!/usr/bin/env python3

import json
import tempfile
import time
from pathlib import Path

from voice_relay.tony2_navigation_motion_arm import (
    ARM_LEASE_SECONDS,
    ARM_REFRESH_SECONDS,
    MotionArmLease,
    atomic_write_json,
    read_json_file,
    validate_arm_payload,
)


def test_missing_lease_is_invalid():
    assert (
        validate_arm_payload(None)
        is None
    )


def test_expired_lease_is_invalid():
    payload = {
        "token": "a" * 32,
        "robot_bridge_url":
            "http://robot.invalid:8090",
        "expires_at": 10.0,
    }

    assert (
        validate_arm_payload(
            payload,
            now=10.0,
        )
        is None
    )


def test_live_lease_is_valid():
    payload = {
        "token": "b" * 32,
        "robot_bridge_url":
            "http://robot.invalid:8090/",
        "expires_at": 20.0,
    }

    result = validate_arm_payload(
        payload,
        now=10.0,
    )

    assert result == {
        "token": "b" * 32,
        "robot_bridge_url":
            "http://robot.invalid:8090",
        "expires_at": 20.0,
    }


def test_atomic_json_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "lease.json"
        )

        atomic_write_json(
            path,
            {"value": 7},
        )

        assert read_json_file(
            path
        ) == {"value": 7}


def test_motion_arm_lease_refreshes_and_removes():
    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "arm.json"
        )

        lease = MotionArmLease(
            path,
            "http://robot.invalid:8090",
        )

        token = lease.start()

        first = read_json_file(
            path
        )

        assert first["token"] == token

        validated = validate_arm_payload(
            first
        )

        assert validated is not None

        first_expiry = float(
            first["expires_at"]
        )

        time.sleep(
            ARM_REFRESH_SECONDS * 2.5
        )

        second = read_json_file(
            path
        )

        assert second["token"] == token

        assert (
            float(second["expires_at"])
            > first_expiry
        )

        assert lease.stop() is True
        assert path.exists() is False


def test_lease_duration_is_short_lived():
    assert ARM_LEASE_SECONDS == 0.50

    assert (
        ARM_REFRESH_SECONDS
        < ARM_LEASE_SECONDS
    )


def test_stop_does_not_remove_replaced_token():
    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "arm.json"
        )

        lease = MotionArmLease(
            path,
            "http://robot.invalid:8090",
        )

        lease.refresh()

        replacement = {
            "token": "z" * 32,
            "robot_bridge_url":
                "http://robot.invalid:8090",
            "expires_at":
                time.time() + 0.5,
        }

        atomic_write_json(
            path,
            replacement,
        )

        assert lease.stop() is False

        assert (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )["token"]
            == replacement["token"]
        )
