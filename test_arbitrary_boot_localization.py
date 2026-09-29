"""Offline contracts for stationary arbitrary-boot AMCL localization."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = (ROOT / "voice_relay" / "tony2_navigation_runtime.py").read_text()
HELPER = (ROOT / "voice_relay" / "tony2_navigation_initial_pose.py").read_text()
SERVER = (ROOT / "voice_relay" / "server.py").read_text()

def section():
    start = SERVER.index("    def navigation_initialize_global_localization")
    return SERVER[start:SERVER.index("    def mapping_pose_status", start)]

def test_global_path_is_existing_unseeded_amcl_without_home_coordinates():
    assert "def initialize_global_localization(self):" in RUNTIME
    assert "runtime.initialize_global_localization()" in section()
    assert "initialize_home_localization" not in section()
    assert "-0.449999944" not in section()

def test_global_amcl_mechanism_is_stationary_and_has_no_goal():
    assert '"/reinitialize_global_localization"' in HELPER
    assert "distribute particles across the saved map" in HELPER
    assert "NO_MOTION_UPDATES = 40" in HELPER
    assert '"motion_enabled":\n                False' in HELPER
    assert '"navigation_goal_executed":\n                False' in HELPER

def test_global_route_fails_closed_and_has_no_motion_or_goal_call():
    body = section()
    for item in ("BRIDGE_NOT_STOPPED", "CAMERA_NOT_READY", "NAVIGATION_GOAL_ACTIVE", "LOCALIZATION_STACK_NOT_READY", "GLOBAL_LOCALIZATION_FAILED", "ACTIVE_LOCALIZATION_REQUIRED"):
        assert item in body
    assert '"GET", f"{ROBOT_BRIDGE_URL}/status"' in body
    assert "NavigateToPose" not in body
    assert "cmd_vel" not in body

def test_success_uses_generic_canonical_authority_and_preserves_home_and_marvin():
    body = section()
    for item in ('"global_localization_requested"', '"initial_pose_supplied"', '"seed_pose_used"', '"localization_validated"', '"transform_ready"'):
        assert item in body
    assert "def initialize_home_localization(self):" in RUNTIME
    assert "FIND_MARVIN" in (ROOT / "runtime.py").read_text()
