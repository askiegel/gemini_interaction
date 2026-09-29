"""Offline lifecycle contracts for recoverable unseeded localization evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = (ROOT / "voice_relay" / "tony2_navigation_runtime.py").read_text()
SERVER = (ROOT / "voice_relay" / "server.py").read_text()

def test_unseeded_insufficient_evidence_is_recoverable_without_stop():
    start = RUNTIME.index('if payload.get(\n            "trusted"\n        ) is not True:')
    end = RUNTIME.index('deadline = (', start)
    section = RUNTIME[start:end]
    assert 'not seed_pose' in section
    assert '"ACTIVE_LOCALIZATION_REQUIRED"' in section
    assert '"localization_state"' in RUNTIME
    assert 'stopped = self.stop()' in section

def test_recoverable_state_cannot_authorize_navigation_or_motion():
    assert '"localization_validated": localization_validated' in RUNTIME
    assert '"goal_submission_enabled": (' in RUNTIME
    assert 'and localization_validated' in RUNTIME
    assert '"localization_state": self._localization_state' in RUNTIME

def test_home_and_hard_failures_keep_stop_cleanup():
    assert 'if not seed_pose and payload.get("global_localization_requested") is True:' in RUNTIME
    assert 'stopped = self.stop()' in RUNTIME
    assert 'self._localization_state = "UNLOCALIZED"' in RUNTIME

def test_server_exposes_structured_recoverable_and_hard_reasons():
    assert 'result.get("action") == "ACTIVE_LOCALIZATION_REQUIRED"' in SERVER
    assert '"reason": "ACTIVE_LOCALIZATION_REQUIRED"' in SERVER
    assert '"reason": "GLOBAL_LOCALIZATION_FAILED"' in SERVER

def test_no_motion_or_home_fallback_added():
    start = SERVER.index('def navigation_initialize_global_localization')
    section = SERVER[start:SERVER.index('def mapping_pose_status', start)]
    assert 'initialize_home_localization' not in section
    assert 'cmd_vel' not in section
    assert 'NavigateToPose' not in section
