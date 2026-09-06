from pathlib import Path


ROOT = Path(__file__).resolve().parent
VOICE = ROOT / "voice_relay"

SUPERVISOR = (
    VOICE
    / "tony2_navigation_supervisor.py"
)

SOURCE = (
    VOICE
    / "tony2_navigation_isolation_source.py"
)

SINK = (
    VOICE
    / "tony2_navigation_isolation_sink.py"
)

PROBE = (
    VOICE
    / "tony2_navigation_probe.py"
)

RUNTIME = (
    VOICE
    / "tony2_navigation_runtime.py"
)


def read(path):
    return path.read_text(
        encoding="utf-8"
    )


def test_fixed_map_server_is_local():
    source = read(SUPERVISOR)

    for expected in (
        'package="nav2_map_server"',
        'executable="map_server"',
        'name="map_server"',
        '"mayday_supervised_route_03.yaml"',
    ):
        assert expected in source


def test_amcl_is_local_global_localizer():
    source = read(SUPERVISOR)

    for expected in (
        'package="nav2_amcl"',
        'executable="amcl"',
        'name="amcl"',
        '"base_footprint"',
        '"map"',
        '"odom"',
        '"tf_broadcast":',
    ):
        assert expected in source


def test_localization_lifecycle_is_owned():
    source = read(SUPERVISOR)

    start = source.index(
        '"node_names": ['
    )

    block = source[
        start:
        start + 600
    ]

    for expected in (
        '"map_server",',
        '"amcl",',
        '"planner_server",',
        '"controller_server",',
        '"bt_navigator",',
    ):
        assert expected in block


def test_live_map_is_not_bridged():
    source = read(SOURCE)
    sink = read(SINK)

    for forbidden in (
        "CHANNEL_MAP",
        'MAP_INPUT = "/map"',
        'MAP_OUTPUT = "/map"',
        "OccupancyGrid",
    ):
        assert forbidden not in source
        assert forbidden not in sink


def test_readiness_requires_localization():
    source = read(PROBE)

    for expected in (
        '"map_server": None',
        '"amcl": None',
        '"map_server_enabled":',
        '"localization_enabled":',
        "map_server_active,",
        "localization_active,",
        "transform_ready,",
    ):
        assert expected in source


def test_runtime_is_fixed_map_mode():
    source = read(RUNTIME)

    for expected in (
        '"mapping_required": False',
        '"mapping_conflict":',
        '"fixed_map_required": True',
        '"mayday_supervised_route_03.yaml"',
        '"local_map_server"',
        '"localization_mode":',
        '"amcl"',
        'not mapping["running"]',
        'if mapping["running"]:',
        (
            "be stopped before "
            "fixed-map navigation."
        ),
    ):
        assert expected in source


def test_map_assets_are_hash_validated():
    source = read(RUNTIME)

    assert (
        '"mayday_supervised_route_03.yaml":'
        in source
    )

    assert (
        '"mayday_supervised_route_03.pgm":'
        in source
    )


def test_motion_contract_remains_bounded():
    source = read(RUNTIME)

    assert (
        "MAXIMUM_GOAL_DISTANCE_METERS = 20.0"
        in source
    )

    assert (
        "EXECUTION_TIMEOUT_SECONDS = 120.0"
        in source
    )

    assert (
        "MOTION_ARM_ACK_TIMEOUT_SECONDS"
        in source
    )
