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


def test_helper_uses_global_localization():
    assert '"/reinitialize_global_localization"' in HELPER
    assert '"/request_nomotion_update"' in HELPER

    assert "SetInitialPose" not in HELPER
    assert '"/set_initial_pose"' not in HELPER


def test_global_search_uses_twenty_stationary_updates():
    assert "NO_MOTION_UPDATES = 20" in HELPER

    assert (
        '"global_localization_requested":\n'
        "                True"
        in HELPER
    )

    assert (
        '"initial_pose_supplied":\n'
        "                False"
        in HELPER
    )


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


def test_server_does_not_supply_home_pose():
    section = localization_server_section()

    assert (
        "runtime.initialize_global_localization()"
        in section
    )

    assert (
        "runtime.initialize_operator_pose("
        not in section
    )

    assert "fixed Home" not in section
    assert "pose (0, 0, 0)" not in section


def test_server_requires_global_result():
    section = localization_server_section()

    assert (
        'localization.get(\n'
        '                "initial_pose_supplied"\n'
        "            ) is False"
        in section
    )

    assert (
        'localization.get(\n'
        '                "global_localization_requested"\n'
        "            ) is True"
        in section
    )

    assert (
        'localization.get("trusted") is True'
        in section
    )


def test_stationary_guard_precedes_global_request():
    section = localization_server_section()

    stationary = section.index(
        "self.ensure_mayday_stationary()"
    )

    request = section.index(
        "runtime.initialize_global_localization()"
    )

    assert stationary < request


def test_localization_route_has_no_motion_path():
    section = localization_server_section()

    assert "NavigateToPose" not in section
    assert "cmd_vel" not in section
    assert "motion_arm" not in section


def test_navigation_ui_accepts_global_result():
    assert (
        ".global_localization_requested === true"
        in HTML
    )

    assert (
        ".initial_pose_supplied === false"
        in HTML
    )


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
