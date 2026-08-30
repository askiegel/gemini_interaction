from pathlib import Path


ROOT = Path(__file__).resolve().parent

SUPERVISOR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_supervisor.py"
)

SOURCE = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_isolation_source.py"
)

SINK = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_isolation_sink.py"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
)


def test_isolation_source_consumes_filtered_mapping_tree():
    source = SOURCE.read_text(
        encoding="utf-8"
    )

    assert (
        'TF_INPUT = "/mayday_navigation_tf"'
        in source
    )

    assert (
        'TF_INPUT = "/tf"'
        not in source
    )


def test_isolation_sink_publishes_local_nav_tf():
    sink = SINK.read_text(
        encoding="utf-8"
    )

    assert (
        'TF_OUTPUT = "/nav_tf"'
        in sink
    )


def test_nav2_nodes_continue_using_local_nav_tf():
    source = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    assert (
        '"/tf",\n'
        '            "/nav_tf",'
        in source
    )


def test_navigation_probe_uses_nav_tf():
    source = RUNTIME.read_text(
        encoding="utf-8"
    )

    start = source.index(
        "probe_pid = self._spawn("
    )

    end = source.index(
        "self.probe_log",
        start,
    )

    command = source[start:end]

    assert '"/tf:=/nav_tf"' in command


def test_goal_helper_uses_nav_tf():
    source = RUNTIME.read_text(
        encoding="utf-8"
    )

    start = source.index(
        "command = [",
        source.index(
            "def submit_goal("
        ),
    )

    end = source.index(
        "log_handle = open(",
        start,
    )

    command = source[start:end]

    assert '"/tf:=/nav_tf"' in command


def test_filtered_tf_change_does_not_add_motion():
    for path in (
        SOURCE,
        SINK,
    ):
        source = path.read_text(
            encoding="utf-8"
        )

        assert "geometry_msgs.msg import Twist" not in source
        assert "create_publisher(Twist" not in source
        assert "NavigateToPose" not in source
        assert "ActionClient" not in source
        assert "cmd_vel" not in source
