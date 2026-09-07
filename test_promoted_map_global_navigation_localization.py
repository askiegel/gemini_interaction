from pathlib import Path


ROOT = Path(__file__).resolve().parent

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(encoding="utf-8")

INDEX = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(encoding="utf-8")


def server_navigation_initializer():
    marker = "    def navigation_initialize_localization("
    start = SERVER.index(marker)

    end = SERVER.find(
        "\n    def ",
        start + len(marker),
    )

    if end < 0:
        end = len(SERVER)

    return SERVER[start:end]


def planning_controller():
    marker = (
        "/* Isolated guarded read-only "
        "planning path controller */"
    )

    start = INDEX.index(marker)
    end = INDEX.find("</script>", start)

    assert end > start

    return INDEX[start:end]


def test_dashboard_initializer_uses_mapwide_amcl():
    source = server_navigation_initializer()

    assert (
        "runtime.initialize_global_localization()"
        in source
    )

    assert (
        "runtime.initialize_home_localization()"
        not in source
    )


def test_browser_accepts_global_localization_contract():
    source = planning_controller()

    assert source.count(
        '=== "amcl_global"'
    ) == 2

    assert (
        '=== "amcl_seeded"'
        not in source
    )

    assert source.count(
        '=== "full_saved_map"'
    ) == 2

    assert (
        '=== "known_home_pose"'
        not in source
    )


def test_browser_global_localization_is_unseeded():
    source = planning_controller()

    assert source.count(
        "initialization.seed_pose_used === false"
    ) == 2

    assert source.count(
        "initialization.global_localization_requested"
    ) >= 2

    assert source.count(
        "initialization.initial_pose_supplied"
    ) >= 2


def test_server_trust_gate_requires_unseeded_global_amcl():
    source = server_navigation_initializer()

    compact = " ".join(
        source.split()
    )

    required = (
        '"initial_pose_supplied" ) is False',
        '"global_localization_requested" ) is True',
        '"seed_pose_used" ) is False',
        '"localization_method" ) == "amcl_global"',
        '"search_scope" ) == "full_saved_map"',
        '"stationary_required" ) is True',
        '"navigation_goal_executed" ) is False',
        '"motion_enabled" ) is False',
    )

    for marker in required:
        assert marker in compact

    forbidden = (
        '"initial_pose_supplied" ) is True',
        '"global_localization_requested" ) is False',
        '"seed_pose_used" ) is True',
    )

    for marker in forbidden:
        assert marker not in compact
