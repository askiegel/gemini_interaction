#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parent

HELPER = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_initial_pose.py"
).read_text(encoding="utf-8")

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(encoding="utf-8")

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(encoding="utf-8")

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(encoding="utf-8")


def localization_server_section():
    start = SERVER.index(
        "    def navigation_initialize_localization"
    )

    end = SERVER.index(
        "    def navigation_goal",
        start,
    )

    return SERVER[start:end]



def test_helper_supports_global_and_seeded_localization():
    assert '"/reinitialize_global_localization"' in HELPER
    assert '"/set_initial_pose"' in HELPER
    assert '"--seed-pose"' in HELPER
    assert "SetInitialPose" in HELPER




def test_localization_modes_preserve_stationary_updates():
    assert "NO_MOTION_UPDATES = 40" in HELPER
    assert "not args.seed_pose" in HELPER
    assert "bool(args.seed_pose)" in HELPER



def test_trust_no_longer_depends_on_fixed_seed():
    assert (
        "        trusted = (\n"
        "            covariance_tight\n"
        "            and alignment_good\n"
        "        )"
        in HELPER
    )

    assert "and seed_consistent" not in HELPER


def test_multi_scan_validation_is_preserved():
    assert "SCAN_CONFIRMATION_SAMPLES = 5" in HELPER

    assert (
        "SCAN_CONFIRMATION_REQUIRED_PASSES = 3"
        in HELPER
    )

    assert "score_scan_alignment" in HELPER
    assert "mean_endpoint_error_m" in HELPER
    assert "within_0_10m_ratio" in HELPER
    assert "known_ratio" in HELPER
    assert "inside_ratio" in HELPER
    assert "covariance_tight" in HELPER


def test_runtime_exposes_global_entry_point():
    assert (
        "def initialize_global_localization(self):"
        in RUNTIME
    )

    assert (
        "return self.initialize_operator_pose("
        in RUNTIME
    )



def test_server_uses_known_home_entry_point():
    section = localization_server_section()

    assert "runtime.initialize_home_localization()" in section
    assert "runtime.initialize_global_localization()" not in section




def test_server_requires_seeded_home_result():
    section = localization_server_section()
    compact = "".join(section.split())

    assert 'localization.get("seed_pose_used")isTrue' in compact
    assert 'localization.get("initial_pose_supplied")isTrue' in compact
    assert (
        'localization.get("global_localization_requested")'
        'isFalse'
        in compact
    )




def test_stationary_guard_precedes_home_request():
    section = localization_server_section()

    stationary = section.index(
        "self.ensure_mayday_stationary()"
    )

    request = section.index(
        "runtime.initialize_home_localization()"
    )

    assert stationary < request



def test_localization_route_has_no_motion_path():
    section = localization_server_section()

    assert "NavigateToPose" not in section
    assert "cmd_vel" not in section
    assert "motion_arm" not in section



def test_navigation_ui_accepts_home_seeded_result():
    compact = "".join(HTML.split())

    assert ".global_localization_requested===false" in compact
    assert ".initial_pose_supplied===true" in compact
    assert ".seed_pose_used===true" in compact



def test_covariance_gate_is_defined_before_trust():
    definition = HELPER.index(
        "covariance_tight = ("
    )

    trust = HELPER.index(
        "trusted = ("
    )

    assert definition < trust

    assert (
        "sigma_x <= MAX_POSITION_SIGMA_METERS"
        in HELPER
    )

    assert (
        "sigma_y <= MAX_POSITION_SIGMA_METERS"
        in HELPER
    )

    assert (
        "sigma_yaw <= MAX_YAW_SIGMA_RADIANS"
        in HELPER
    )
