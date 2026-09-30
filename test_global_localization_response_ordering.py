"""Offline response-ordering contracts for global AMCL localization."""

from unittest.mock import Mock, patch

from voice_relay.server import VoiceRelayHandler


def _navigation(*, validated, state):
    return {
        "state": "READY",
        "running": True,
        "map_server_enabled": True,
        "localization_enabled": True,
        "transform_ready": True,
        "localization_validated": validated,
        "localization_state": state,
        "goal_submission_enabled": validated,
        "goal_active": False,
        "motion_output_connected": False,
        "motion_egress_ready": True,
        "motion_egress_idle": True,
    }


def _bridge_and_camera():
    return [
        {
            "ok": True,
            "data": {
                "ros_ready": True,
                "motion": {"linear_x": 0, "angular_z": 0, "streaming": False},
            },
        },
        {"ok": True, "data": {"camera_running": True, "last_error": None}},
    ]


def _trusted_global_result(stale_navigation):
    return {
        "action": "OPERATOR_POSE_VALIDATED",
        "localization": {
            "trusted": True,
            "global_localization_requested": True,
            "initial_pose_supplied": False,
            "seed_pose_used": False,
            "stationary_required": True,
            "navigation_goal_executed": False,
            "motion_enabled": False,
        },
        "navigation": stale_navigation,
    }


def test_global_success_uses_post_validation_canonical_snapshot_not_stale_result():
    stale = _navigation(validated=False, state="GLOBAL_LOCALIZING")
    canonical = _navigation(validated=True, state="LOCALIZED")
    runtime = Mock()
    runtime.status.side_effect = [stale, canonical]
    runtime.initialize_global_localization.return_value = _trusted_global_result(stale)
    handler = object.__new__(VoiceRelayHandler)

    with patch("voice_relay.server.get_tony2_navigation_runtime", return_value=runtime), patch(
        "voice_relay.server.request_json", side_effect=_bridge_and_camera()
    ):
        status_code, payload = handler.navigation_initialize_global_localization()

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["action"] == "navigation_initialize_global_localization"
    assert payload["navigation"] == canonical
    assert payload["navigation"]["localization_validated"] is True
    assert payload["navigation"]["transform_ready"] is True
    assert payload.get("reason") != "GLOBAL_LOCALIZATION_FAILED"


def test_recoverable_global_result_remains_recoverable_after_canonical_reread():
    recoverable = _navigation(validated=False, state="ACTIVE_LOCALIZATION_REQUIRED")
    runtime = Mock()
    runtime.status.side_effect = [recoverable, recoverable]
    runtime.initialize_global_localization.return_value = {
        "action": "ACTIVE_LOCALIZATION_REQUIRED",
        "localization": {"trusted": False, "global_localization_requested": True},
        "navigation": _navigation(validated=False, state="GLOBAL_LOCALIZING"),
    }
    handler = object.__new__(VoiceRelayHandler)

    with patch("voice_relay.server.get_tony2_navigation_runtime", return_value=runtime), patch(
        "voice_relay.server.request_json", side_effect=_bridge_and_camera()
    ):
        status_code, payload = handler.navigation_initialize_global_localization()

    assert status_code == 503
    assert payload["reason"] == "ACTIVE_LOCALIZATION_REQUIRED"
    assert payload["navigation"] == recoverable
    assert payload["navigation"]["localization_validated"] is False


def test_hard_global_failure_remains_fail_closed():
    status = _navigation(validated=False, state="GLOBAL_LOCALIZING")
    runtime = Mock()
    runtime.status.side_effect = [status, status]
    runtime.initialize_global_localization.side_effect = RuntimeError("AMCL failed")
    handler = object.__new__(VoiceRelayHandler)

    with patch("voice_relay.server.get_tony2_navigation_runtime", return_value=runtime), patch(
        "voice_relay.server.request_json", side_effect=_bridge_and_camera()
    ):
        status_code, payload = handler.navigation_initialize_global_localization()

    assert status_code == 503
    assert payload["reason"] == "GLOBAL_LOCALIZATION_FAILED"
    assert payload["error"] == "AMCL failed"
