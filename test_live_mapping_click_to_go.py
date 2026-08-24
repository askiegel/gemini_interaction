#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parent

HTML = (
    ROOT / "voice_relay" / "index.html"
).read_text(encoding="utf-8")

CSS = (
    ROOT / "voice_relay" / "operator_console.css"
).read_text(encoding="utf-8")

JS = (
    ROOT / "voice_relay" / "operator_console.js"
).read_text(encoding="utf-8")

SERVER = (
    ROOT / "voice_relay" / "server.py"
).read_text(encoding="utf-8")

START = JS.index(
    "/* Guarded live-mapping click-to-go */"
)

CLICK_TO_GO = JS[START:]


def test_live_map_has_guarded_click_to_go_controls():
    for identifier in (
        "liveMappingGoalCanvas",
        "liveMappingGoalReadout",
        "liveMappingGoalDistance",
        "liveMappingGoalMessage",
        "liveMappingGoButton",
        "liveMappingClearGoalButton",
    ):
        assert f'id="{identifier}"' in HTML

    assert "GO — max 0.50 m" in HTML
    assert "Selection alone never moves Mayday." in HTML


def test_overlay_is_separate_from_read_only_map_canvas():
    assert "#liveMappingGoalCanvas" in CSS
    assert "position: absolute" in CSS
    assert "cursor: crosshair" in CSS

    read_only_start = JS.index(
        "/* Read-only live Cartographer mapping map */"
    )

    candidate_start = JS.index(
        "/* Read-only candidate map review */"
    )

    read_only_map = JS[
        read_only_start:candidate_start
    ]

    assert 'method: "POST"' not in read_only_map
    assert "/mapping-navigation/" not in read_only_map


def test_click_only_stages_a_target():
    select_start = CLICK_TO_GO.index(
        "async function selectLiveMappingGoal"
    )

    send_start = CLICK_TO_GO.index(
        "async function sendLiveMappingGoal"
    )

    selection = CLICK_TO_GO[
        select_start:send_start
    ]

    assert "selectedLiveGoal = candidate;" in selection
    assert 'method: "POST"' not in selection
    assert "GOAL_ENDPOINT" not in selection


def test_selection_requires_exactly_free_bounded_space():
    assert (
        "MAX_LIVE_MAPPING_GOAL_DISTANCE_METERS = 0.50"
        in CLICK_TO_GO
    )

    assert (
        "MIN_LIVE_MAPPING_GOAL_DISTANCE_METERS = 0.03"
        in CLICK_TO_GO
    )

    assert "if (occupancy !== 0)" in CLICK_TO_GO
    assert "if (targetOccupancy !== 0)" in CLICK_TO_GO

    assert (
        "distance\n"
        "            > MAX_LIVE_MAPPING_GOAL_DISTANCE_METERS"
        in CLICK_TO_GO
    )

    assert (
        "Straight-line staging check found"
        in CLICK_TO_GO
    )


def test_map_origin_yaw_is_used_for_click_conversion():
    assert (
        "const localX = cosine * dx + sine * dy;"
        in CLICK_TO_GO
    )

    assert (
        "const localY = -sine * dx + cosine * dy;"
        in CLICK_TO_GO
    )

    assert (
        "Number(origin.x)\n"
        "                + cosine * localX\n"
        "                - sine * localY"
        in CLICK_TO_GO
    )

    assert (
        "Number(origin.y)\n"
        "                + sine * localX\n"
        "                + cosine * localY"
        in CLICK_TO_GO
    )


def test_go_uses_mapping_navigation_only():
    for endpoint in (
        '"/dashboard/mapping-pose"',
        '"/dashboard/mapping-navigation-start"',
        '"/dashboard/mapping-navigation-status"',
        '"/dashboard/mapping-navigation-goal"',
        '"/dashboard/mapping-navigation-stop"',
    ):
        assert endpoint in CLICK_TO_GO

    assert "NAVIGATION_START_SETTLE_MS" not in CLICK_TO_GO

    assert (
        "await waitForMappingNavigationReady();"
        in CLICK_TO_GO
    )

    assert "window.confirm(" in CLICK_TO_GO
    assert "finally {" in CLICK_TO_GO

    assert (
        '"/dashboard/navigation-goal"'
        not in CLICK_TO_GO
    )

    assert "cmd_vel" not in CLICK_TO_GO


def test_go_waits_for_authoritative_navigation_readiness():
    assert (
        "NAVIGATION_READY_TIMEOUT_MS = 20000"
        in CLICK_TO_GO
    )

    assert (
        "NAVIGATION_READY_POLL_MS = 250"
        in CLICK_TO_GO
    )

    assert (
        '"/dashboard/mapping-navigation-status"'
        in CLICK_TO_GO
    )

    for readiness_field in (
        "navigation.running === true",
        "navigation.owned === true",
        "navigation.goal_submission_enabled === true",
        "navigation.controller_enabled === true",
        "navigation.navigator_enabled === true",
    ):
        assert readiness_field in CLICK_TO_GO

    assert (
        "await waitForMappingNavigationReady();"
        in CLICK_TO_GO
    )

    assert (
        "Guarded navigation did not become ready "
        in CLICK_TO_GO
    )


def test_go_has_no_retry_loop():
    send_start = CLICK_TO_GO.index(
        "async function sendLiveMappingGoal"
    )

    initialize_start = CLICK_TO_GO.index(
        "function initialize()",
        send_start,
    )

    send = CLICK_TO_GO[
        send_start:initialize_start
    ]

    assert "for (" not in send
    assert "while (" not in send

    # One explicit guarded goal submission only.
    assert send.count(
        "fetchJson(\n"
        "                GOAL_ENDPOINT,"
    ) == 1


def test_voice_relay_exposes_mapping_pose_proxy():
    assert "def mapping_pose_status(self):" in SERVER

    assert (
        'f"{ROBOT_BRIDGE_URL}/telemetry/mapping-pose"'
        in SERVER
    )

    assert (
        'if path == "/dashboard/mapping-pose":'
        in SERVER
    )


def test_voice_relay_exposes_mapping_navigation_proxies():
    assert (
        "def mapping_navigation_status(self):"
        in SERVER
    )

    assert (
        "def mapping_navigation_control_action("
        "self, action):"
        in SERVER
    )

    assert (
        "def mapping_navigation_goal(self, payload):"
        in SERVER
    )

    assert (
        'f"/mapping-navigation/{action}"'
        in SERVER
    )

    assert (
        'f"{ROBOT_BRIDGE_URL}/mapping-navigation/goal"'
        in SERVER
    )

    assert (
        '"/mapping-navigation/status"'
        in SERVER
    )

    for route in (
        "/dashboard/mapping-navigation-status",
        "/dashboard/mapping-navigation-start",
        "/dashboard/mapping-navigation-stop",
        "/dashboard/mapping-navigation-goal",
    ):
        assert route in SERVER


def test_browser_cannot_override_backend_safety_parameters():
    forbidden = (
        "maximum_goal_distance_meters",
        "maximum_execution_seconds",
        "behavior_tree",
        "retries",
        "recoveries",
        "cmd_vel",
        "linear_x",
        "angular_z",
    )

    for marker in forbidden:
        assert marker not in CLICK_TO_GO

    assert (
        "Robot Bridge remains authoritative"
        in SERVER
    )
