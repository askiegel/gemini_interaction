from pathlib import Path
from unittest.mock import patch

from voice_relay.server import VoiceRelayHandler


HTML = Path("voice_relay/index.html").read_text(encoding="utf-8")
SERVER = Path("voice_relay/server.py").read_text(encoding="utf-8")


def _handler():
    return VoiceRelayHandler.__new__(VoiceRelayHandler)


def test_find_marvin_defaults_to_dry_run_without_touching_runtime():
    handler = _handler()

    def forbidden_status():
        raise AssertionError(
            "dry-run Find Marvin must not query mission readiness"
        )

    handler.dashboard_status = forbidden_status

    status_code, payload = handler.submit_find_marvin()

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["accepted"] is False
    assert payload["executed"] is False
    assert payload["dry_run"] is True
    assert payload["target"] == "marvin"
    assert payload["mission"] is None


def test_find_marvin_explicit_false_is_dry_run():
    handler = _handler()

    def forbidden_status():
        raise AssertionError(
            "execute=False must not reach live mission preflight"
        )

    handler.dashboard_status = forbidden_status

    status_code, payload = handler.submit_find_marvin(
        execute=False
    )

    assert status_code == 200
    assert payload["accepted"] is False
    assert payload["executed"] is False
    assert payload["dry_run"] is True


def test_find_marvin_browser_sends_live_mode_authorization():
    assert (
        "execute: isLiveModeEnabled()"
        in HTML
    )


def test_find_marvin_http_defaults_missing_execute_to_false():
    assert (
        'execute = payload.get("execute", False)'
        in SERVER
    )
    assert (
        "self.submit_find_marvin(\n"
        "                execute=execute\n"
        "            )"
        in SERVER
    )


def test_find_marvin_http_rejects_non_boolean_execute():
    assert (
        "if not isinstance(execute, bool):"
        in SERVER
    )


def test_find_marvin_live_submission_carries_marvin_semantic_identity():
    assert '"target": "marvin"' in SERVER


def test_find_marvin_live_submission_posts_marvin_target_after_preflight():
    handler = _handler()
    admission_snapshot = {
        "ok": True,
        "evaluation_timestamp": "2026-10-02T12:00:00+00:00",
        "admission_ready": True,
        "reasons": [],
        "runtime": {}, "lidar": {}, "forward_interlock": {}, "bridge": {},
    }
    with patch("voice_relay.server.request_json", side_effect=[
        {"ok": True, "status_code": 200, "data": admission_snapshot,
         "error": None},
        {"status_code": 200, "data": {"ok": True, "accepted": True},
         "error": None},
    ]) as request:
        status_code, _payload = handler.submit_find_marvin(execute=True)
    assert status_code == 200
    assert request.call_count == 2
    assert request.call_args_list[0].args == (
        "GET", "http://127.0.0.1:8770/find-marvin/admission-snapshot",
    )
    assert request.call_args_list[1].args == (
        "POST", "http://127.0.0.1:8770/missions",
    )
    assert request.call_args.kwargs["payload"]["intent"]["target"] == "marvin"
