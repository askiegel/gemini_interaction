#!/usr/bin/env python3

from pathlib import Path

from voice_relay.persistent_map_refresh_runtime import (
    CAPTURE_SECONDS,
    MINIMUM_SCANS,
    PersistentMapRefreshRuntime,
)


ROOT = Path(
    __file__
).resolve().parent

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(
    encoding="utf-8"
)

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)

REFRESH_RUNTIME = (
    ROOT
    / "voice_relay"
    / "persistent_map_refresh_runtime.py"
).read_text(
    encoding="utf-8"
)

NAV_RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(
    encoding="utf-8"
)

NAV_SUPERVISOR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_supervisor.py"
).read_text(
    encoding="utf-8"
)


def test_refresh_capture_contract():
    assert CAPTURE_SECONDS == 30
    assert MINIMUM_SCANS == 100

    runtime = (
        PersistentMapRefreshRuntime()
    )

    command = (
        runtime._builder_command()
    )

    for option in (
        "--duration-seconds",
        "--lidar-url",
        "--robot-status-url",
        "--output-dir",
        "--name",
        "--minimum-scans",
    ):
        assert option in command

    assert (
        command[
            command.index(
                "--duration-seconds"
            )
            + 1
        ]
        == "30"
    )

    assert (
        command[
            command.index(
                "--minimum-scans"
            )
            + 1
        ]
        == "100"
    )


def test_existing_stationary_builder_is_reused():
    assert (
        "stationary_map_builder.py"
        in REFRESH_RUNTIME
    )


def test_capture_does_not_auto_promote():
    start = REFRESH_RUNTIME.index(
        "    def start_refresh("
    )

    end = REFRESH_RUNTIME.index(
        "    def cancel(",
        start,
    )

    section = (
        REFRESH_RUNTIME[
            start:end
        ]
    )

    assert ".promote(" not in section
    assert "CANDIDATE_READY" in REFRESH_RUNTIME


def test_candidate_must_preserve_stationary_safety():
    for marker in (
        '"motion_verified"',
        '"validated_map_replaced"',
        '"cartographer_used"',
        '"navigation_used"',
        '"motion_commanded"',
        '"stable_bin_count"',
        '"occupied_cell_count"',
        '"free_cell_count"',
    ):
        assert marker in REFRESH_RUNTIME


def test_refresh_runtime_has_no_motion_interface():
    for marker in (
        "NavigateToPose",
        "MotionArmLease",
        "geometry_msgs",
        '"/cmd_vel"',
        "'/cmd_vel'",
    ):
        assert marker not in REFRESH_RUNTIME


def test_server_exposes_refresh_routes():
    for route in (
        "/dashboard/persistent-map/refresh",
        "/dashboard/persistent-map/refresh-status",
        "/dashboard/persistent-map/refresh-candidate",
        "/dashboard/persistent-map/promote",
        "/dashboard/persistent-map/discard",
        "/dashboard/persistent-map/cancel",
    ):
        assert route in SERVER


def test_server_stops_navigation_and_mapping():
    start = SERVER.index(
        "    def prepare_persistent_map_refresh("
    )

    end = SERVER.index(
        "    def persistent_map_refresh_start(",
        start,
    )

    section = (
        SERVER[
            start:end
        ]
    )

    first_stationary = section.index(
        "self.ensure_mayday_stationary()"
    )

    navigation_stop = section.index(
        "navigation_runtime.stop()"
    )

    mapping_stop = section.index(
        "mapping_runtime.stop()"
    )

    assert (
        first_stationary
        < navigation_stop
    )

    assert (
        first_stationary
        < mapping_stop
    )

    assert (
        '"motion_output_connected"'
        in section
    )

    assert (
        '"goal_active"'
        in section
    )


def test_refresh_blocks_conflicting_actions():
    assert (
        "refresh_conflict_paths"
        in SERVER
    )

    for route in (
        "/dashboard/navigation-start",
        "/dashboard/navigation-goal",
        "/dashboard/navigation-initialize-localization",
        "/dashboard/mapping-navigation-start",
        "/dashboard/mapping-navigation-goal",
        "/dashboard/mapping-start",
    ):
        assert route in SERVER


def test_active_map_is_stored_outside_repo():
    assert '".local"' in REFRESH_RUNTIME
    assert '"share"' in REFRESH_RUNTIME
    assert '"mayday"' in REFRESH_RUNTIME
    assert '"persistent_map"' in REFRESH_RUNTIME

    assert (
        "os.symlink("
        in REFRESH_RUNTIME
    )

    assert (
        "os.replace("
        in REFRESH_RUNTIME
    )


def test_navigation_uses_promoted_map_on_next_start():
    assert (
        "MAYDAY_FIXED_MAP_YAML"
        in NAV_RUNTIME
    )

    assert (
        "MAYDAY_FIXED_MAP_YAML"
        in NAV_SUPERVISOR
    )

    assert (
        "mayday_supervised_route_03.yaml"
        in NAV_SUPERVISOR
    )


def test_dashboard_map_route_prefers_promoted_map():
    assert (
        "runtime.active_map_payload()"
        in SERVER
    )

    assert (
        'if path == "/dashboard/map":'
        in SERVER
    )


def test_webpage_has_refresh_and_review_controls():
    assert (
        "MAYDAY_PERSISTENT_MAP_REFRESH_UI"
        in HTML
    )

    for marker in (
        "Refresh Persistent Map",
        "persistentMapRefreshButton",
        "persistentMapCancelButton",
        "persistentMapCandidateCanvas",
        "persistentMapPromoteButton",
        "persistentMapDiscardButton",
        "Use New Map",
        "Discard",
    ):
        assert marker in HTML


def test_webpage_requires_confirmation():
    assert (
        "window.confirm("
        in HTML
    )

    assert (
        "Use this candidate as the new"
        in HTML
    )



def test_promoted_map_matches_existing_dashboard_schema(
    tmp_path,
):
    state_root = (
        tmp_path
        / "persistent_map"
    )

    active = (
        state_root
        / "active"
    )

    active.mkdir(
        parents=True,
    )

    pgm = (
        active
        / "mayday_supervised_route_03.pgm"
    )

    pgm.write_text(
        (
            "P2\n"
            "2 2\n"
            "255\n"
            "254 0\n"
            "205 254\n"
        ),
        encoding="ascii",
    )

    yaml = (
        active
        / "mayday_supervised_route_03.yaml"
    )

    yaml.write_text(
        (
            "image: mayday_supervised_route_03.pgm\n"
            "mode: trinary\n"
            "resolution: 0.050000\n"
            "origin: [0.000000, 0.000000, 0.000000]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        ),
        encoding="utf-8",
    )

    runtime = (
        PersistentMapRefreshRuntime(
            state_root=state_root,
        )
    )

    payload = (
        runtime.active_map_payload()
    )

    assert payload["ok"] is True

    assert (
        payload[
            "persistent_map_source"
        ]
        == "dashboard_promoted"
    )

    telemetry = (
        payload["telemetry"]
    )

    assert (
        telemetry["available"]
        is True
    )

    map_payload = (
        telemetry["map"]
    )

    assert map_payload["width"] == 2
    assert map_payload["height"] == 2

    assert (
        len(
            map_payload["cells"]
        )
        == 4
    )

    assert (
        map_payload["cells"]
        == map_payload["data"]
    )

    assert (
        payload["map"]["data"]
        == map_payload["cells"]
    )
