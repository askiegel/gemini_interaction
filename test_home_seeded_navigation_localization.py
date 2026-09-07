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


def test_seed_and_global_amcl_modes_exist():
    assert "SetInitialPose" in HELPER
    assert '"/set_initial_pose"' in HELPER
    assert '"/reinitialize_global_localization"' in HELPER
    assert '"--seed-pose"' in HELPER


def test_home_pose_is_fixed():
    assert "def initialize_home_localization(self):" in RUNTIME
    assert "-0.05," in RUNTIME
    assert "-1.7906250685463345," in RUNTIME
    assert "seed_pose=True" in RUNTIME


def test_global_recovery_is_preserved():
    assert "def initialize_global_localization(self):" in RUNTIME
    assert "seed_pose=False" in RUNTIME



def test_normal_server_uses_global_localization():
    start = SERVER.index(
        "    def navigation_initialize_localization"
    )
    end = SERVER.index(
        "    def navigation_goal",
        start,
    )

    section = SERVER[start:end]

    assert (
        "runtime.initialize_global_localization()"
        in section
    )

    assert (
        "runtime.initialize_home_localization()"
        not in section
    )


def test_no_contradictory_dashboard_contract():
    start = HTML.index(
        "function navigationInitializationSucceeded(result)"
    )
    end = HTML.index(
        "async function",
        start,
    )

    compact = "".join(
        HTML[start:end].split()
    )

    required = (
        'initialization.localization_method==="amcl_global"',
        'initialization.search_scope==="full_saved_map"',
        "initialization.seed_pose_used===false",
        "initialization.global_localization_requested===true",
        "initialization.initial_pose_supplied===false",
        "Number(initialization.nomotion_updates_requested)>=1",
    )

    for marker in required:
        assert marker in compact

    assert (
        'initialization.localization_method==="amcl_seeded"'
        not in compact
    )

    assert (
        'initialization.search_scope==="known_home_pose"'
        not in compact
    )

def test_trust_thresholds_unchanged():
    required = (
        "SCAN_CONFIRMATION_SAMPLES = 5",
        "SCAN_CONFIRMATION_REQUIRED_PASSES = 3",
        "MAX_MEAN_ENDPOINT_ERROR_METERS = 0.18",
        "MIN_WITHIN_10CM_RATIO = 0.35",
        "MIN_KNOWN_RATIO = 0.55",
        "MIN_INSIDE_RATIO = 0.75",
        "covariance_tight",
        "alignment_good",
    )

    for marker in required:
        assert marker in HELPER

def test_seed_flag_precedes_ros_argument_tail():
    start = RUNTIME.index(
        "    def initialize_operator_pose"
    )

    section = RUNTIME[start:]

    seed = section.index(
        '"--seed-pose"'
    )

    ros_args = section.index(
        '"--ros-args"'
    )

    assert seed < ros_args
