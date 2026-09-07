from pathlib import Path


ROOT = Path(__file__).resolve().parent

HTML = (
    ROOT
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)


def function_source(name, next_name):
    start = HTML.index(
        f"    function {name}"
    )

    end = HTML.index(
        f"\n    function {next_name}",
        start,
    )

    return HTML[start:end]


def test_marker_uses_active_planning_overlay_pose():
    source = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    assert "latestPose.position" in source
    assert "mapToCanvas(" in source


def test_marker_has_clear_exact_position_reference():
    source = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    assert "crosshairInner" in source
    assert "crosshairOuter" in source
    assert "centerRadius" in source

    assert (
        "context.arc(\n"
        "            0,\n"
        "            0,\n"
        "            centerRadius,"
        in source
    )


def test_marker_has_no_heading_arrow():
    source = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    assert "headingStart" not in source
    assert "headingEnd" not in source
    assert "Heading arrow." not in source
    assert "context.rotate(-yaw)" not in source


def test_marker_is_distinct_from_selected_goal():
    marker = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    path = function_source(
        "drawPath()",
        "updateButtons()",
    )

    assert "#f59e0b" in marker
    assert "#22c55e" not in marker

    # Existing selected-goal contract remains green.
    assert "#22c55e" in path


def test_draw_path_always_renders_mayday_last():
    source = function_source(
        "drawPath()",
        "updateButtons()",
    )

    assert (
        "drawMaydayPositionMarker(\n"
        "            context,\n"
        "            drawing\n"
        "        );"
        in source
    )


def test_fresh_pose_repaints_planning_overlay():
    start = HTML.index(
        "    function renderLocalization("
    )

    end = HTML.index(
        "\n    async function fetchJson(",
        start,
    )

    source = HTML[start:end]

    assert "drawPath();" in source


def test_marker_is_read_only():
    source = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    forbidden = (
        "fetch(",
        "POST",
        "cmd_vel",
        "navigation-goal",
        "navigation_goal",
    )

    for token in forbidden:
        assert token not in source


PROBE = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_probe.py"
).read_text(
    encoding="utf-8"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(
    encoding="utf-8"
)

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(
    encoding="utf-8"
)


def test_navigation_probe_observes_amcl_pose_read_only():
    assert (
        "PoseWithCovarianceStamped"
        in PROBE
    )
    assert '"/amcl_pose"' in PROBE
    assert (
        "pose_observed_at_monotonic"
        in PROBE
    )
    assert (
        "x_standard_deviation"
        in PROBE
    )
    assert (
        "yaw_standard_deviation_radians"
        in PROBE
    )


def test_navigation_runtime_exposes_fresh_pose():
    assert (
        "def live_pose_status(self):"
        in RUNTIME
    )
    assert (
        'age_seconds < 3.0'
        in RUNTIME
    )
    assert (
        '"tony2_navigation_amcl"'
        in RUNTIME
    )


def test_dashboard_has_read_only_navigation_pose_route():
    assert (
        "def navigation_pose_status(self):"
        in SERVER
    )
    assert (
        'path == "/dashboard/navigation-pose"'
        in SERVER
    )


def test_planning_overlay_reads_navigation_pose():
    contract = (
        '    const MAP_ENDPOINT = "/dashboard/map";\n'
        '    const LOCALIZATION_ENDPOINT =\n'
        '        "/dashboard/navigation-pose";\n'
        '    const REFRESH_MS = 750;'
    )

    assert contract in HTML


def test_mayday_label_is_small_and_counter_rotated():
    source = function_source(
        "drawMaydayPositionMarker(",
        "drawPath()",
    )

    assert 'context.rotate(Math.PI);' in source
    assert 'context.font = "700 10px system-ui";' in source
    assert 'context.strokeText(' in source
    assert '"Mayday"' in source



def test_navigation_marker_pose_refreshes_from_live_tf():
    assert (
        "def _current_map_pose(self):"
        in PROBE
    )

    assert (
        'self._tf_buffer.lookup_transform('
        in PROBE
    )

    assert '"map"' in PROBE
    assert '"base_link"' in PROBE

    assert (
        '"pose": current_pose'
        in PROBE
    )

    assert (
        "current_pose_observed_at_monotonic"
        in PROBE
    )
