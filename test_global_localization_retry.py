#!/usr/bin/env python3

"""Offline contracts for retrying an unresolved AMCL global-localization session."""

import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from voice_relay.tony2_navigation_runtime import Tony2NavigationRuntime


def _status(*, transform_ready, localization_validated=False, state="READY"):
    return {
        "state": state,
        "running": True,
        "map_server_enabled": True,
        "localization_enabled": True,
        "transform_ready": transform_ready,
        "localization_validated": localization_validated,
        "goal_submission_enabled": False,
        "goal_active": False,
        "motion_output_connected": False,
        "motion_egress_ready": True,
        "motion_egress_idle": True,
    }


def _global_payload(*, trusted):
    return {
        "ok": True,
        "trusted": trusted,
        "global_localization_requested": True,
        "initial_pose_supplied": False,
        "seed_pose_used": False,
        "stationary_required": True,
        "navigation_goal_executed": False,
        "motion_enabled": False,
    }


def _runtime():
    directory = tempfile.TemporaryDirectory()
    runtime = Tony2NavigationRuntime(runtime_dir=directory.name)
    return runtime, directory


def _completed(payload):
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload))


def test_recoverable_global_localization_retry_is_unseeded_and_can_validate():
    runtime, directory = _runtime()
    try:
        first = _global_payload(trusted=False)
        second = _global_payload(trusted=True)
        with patch.object(runtime, "status", side_effect=[
            _status(transform_ready=False, state="STARTING"),
            _status(transform_ready=True, state="READY"),
            _status(transform_ready=True, state="READY"),
            _status(transform_ready=True, state="READY"),
        ]), patch(
            "voice_relay.tony2_navigation_runtime.subprocess.run",
            side_effect=[_completed(first), _completed(second)],
        ) as run, patch.object(runtime, "stop") as stop:
            first_result = runtime.initialize_global_localization()
            assert first_result["action"] == "ACTIVE_LOCALIZATION_REQUIRED"
            assert runtime._localization_state == "ACTIVE_LOCALIZATION_REQUIRED"
            assert runtime._localization_validated is False

            second_result = runtime.initialize_global_localization()

        assert second_result["action"] == "OPERATOR_POSE_VALIDATED"
        assert runtime._localization_state == "LOCALIZED"
        assert runtime._localization_validated is True
        assert stop.call_count == 0
        assert run.call_count == 2
        for call in run.call_args_list:
            command = call.args[0]
            assert "--seed-pose" not in command
            assert "/reinitialize_global_localization" not in command
    finally:
        directory.cleanup()


def test_recoverable_global_localization_retry_can_remain_recoverable():
    runtime, directory = _runtime()
    try:
        payload = _global_payload(trusted=False)
        with patch.object(runtime, "status", side_effect=[
            _status(transform_ready=False, state="STARTING"),
            _status(transform_ready=True, state="READY"),
            _status(transform_ready=True, state="READY"),
            _status(transform_ready=True, state="READY"),
        ]), patch(
            "voice_relay.tony2_navigation_runtime.subprocess.run",
            side_effect=[_completed(payload), _completed(payload)],
        ), patch.object(runtime, "stop") as stop:
            assert runtime.initialize_global_localization()["action"] == "ACTIVE_LOCALIZATION_REQUIRED"
            retry = runtime.initialize_global_localization()

        assert retry["action"] == "ACTIVE_LOCALIZATION_REQUIRED"
        assert runtime._localization_state == "ACTIVE_LOCALIZATION_REQUIRED"
        assert runtime._localization_validated is False
        stop.assert_not_called()
    finally:
        directory.cleanup()


def test_operator_and_home_initialization_cannot_use_global_retry_exception():
    runtime, directory = _runtime()
    try:
        runtime._localization_state = "ACTIVE_LOCALIZATION_REQUIRED"
        with patch.object(runtime, "status", return_value=_status(transform_ready=True)), patch(
            "voice_relay.tony2_navigation_runtime.subprocess.run"
        ) as run:
            for initializer in (
                lambda: runtime.initialize_operator_pose(0.1, 0.2, 0.3),
                runtime.initialize_home_localization,
            ):
                try:
                    initializer()
                except RuntimeError as exc:
                    assert "already initialized" in str(exc)
                else:
                    raise AssertionError("seeded/operator initialization unexpectedly accepted")
            run.assert_not_called()
    finally:
        directory.cleanup()


def test_validated_localization_cannot_use_global_retry_exception():
    runtime, directory = _runtime()
    try:
        runtime._localization_state = "LOCALIZED"
        runtime._localization_validated = True
        with patch.object(runtime, "status", return_value=_status(
            transform_ready=True, localization_validated=True
        )), patch("voice_relay.tony2_navigation_runtime.subprocess.run") as run:
            try:
                runtime.initialize_global_localization()
            except RuntimeError as exc:
                assert "already initialized" in str(exc)
            else:
                raise AssertionError("validated localization unexpectedly accepted")
            run.assert_not_called()
    finally:
        directory.cleanup()
