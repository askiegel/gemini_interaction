from pathlib import Path


ROOT = Path(__file__).resolve().parent

SUPERVISOR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_supervisor.py"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
)


def test_navigation_relay_consumes_filtered_mapping_tree():
    source = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    assert (
        '"input_topic:=/mayday_navigation_tf"'
        in source
    )

    assert (
        '"output_topic:=/nav_tf"'
        in source
    )

    assert (
        '"input_topic:=/tf"'
        not in source
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
        source.index("def submit_goal("),
    )

    end = source.index(
        "log_handle = open(",
        start,
    )

    command = source[start:end]

    assert '"/tf:=/nav_tf"' in command


def test_navigation_tf_change_does_not_add_motion():
    supervisor = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    # Existing controller cmd_vel routing is allowed.
    # This feature must not add direct Twist publication.
    assert "geometry_msgs.msg import Twist" not in supervisor
    assert "create_publisher(Twist" not in supervisor
