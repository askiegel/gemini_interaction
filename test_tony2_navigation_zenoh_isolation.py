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


def read(path):
    return path.read_text(
        encoding="utf-8"
    )


def test_domain42_source_has_exact_five_inputs():
    source = read(SOURCE)

    for expected in (
        'SCAN_INPUT = "/scan"',
        'ODOM_INPUT = "/odom"',
        'TF_INPUT = "/mayday_navigation_tf"',
        'TF_STATIC_INPUT = "/tf_static"',
        'MAP_INPUT = "/map"',
    ):
        assert expected in source


def test_domain43_sink_has_nav2_outputs():
    sink = read(SINK)

    for expected in (
        'SCAN_OUTPUT = "/tony2_nav_scan"',
        'ODOM_OUTPUT = "/tony2_nav_odom"',
        'TF_OUTPUT = "/nav_tf"',
        'TF_STATIC_OUTPUT = "/tf_static"',
        'MAP_OUTPUT = "/map"',
    ):
        assert expected in sink


def test_transport_is_loopback_tcp_only():
    source = read(SOURCE)
    sink = read(SINK)

    assert 'HOST = "127.0.0.1"' in source
    assert 'HOST = "127.0.0.1"' in sink
    assert '"43143"' in source
    assert '"43143"' in sink


def test_supervisor_starts_local_zenoh_router():
    supervisor = read(SUPERVISOR)

    assert (
        "/opt/ros/humble/lib/"
        in supervisor
    )

    assert (
        "rmw_zenoh_cpp/rmw_zenohd"
        in supervisor
    )

    assert (
        'tcp/127.0.0.1:7447'
        in supervisor
    )

    assert (
        "scouting/multicast/enabled=false"
        in supervisor
    )

    assert (
        "transport/shared_memory/enabled=false"
        in supervisor
    )


def test_source_process_is_only_domain42_fastdds_ingress():
    supervisor = read(SUPERVISOR)

    assert '"ROS_DOMAIN_ID=42"' in supervisor
    assert (
        '"RMW_IMPLEMENTATION="\n'
        '                        "rmw_fastrtps_cpp"'
        in supervisor
    )

    assert (
        '"FASTDDS_BUILTIN_TRANSPORTS"'
        in supervisor
    )


def test_runtime_uses_isolated_zenoh_domain43():
    runtime = read(RUNTIME)

    assert (
        '"ROS_DOMAIN_ID"\n'
        '        ] = "43"'
        in runtime
    )

    assert (
        '"RMW_IMPLEMENTATION"\n'
        '        ] = "rmw_zenoh_cpp"'
        in runtime
    )

    assert (
        "ZENOH_SESSION_OVERRIDE"
        in runtime
    )

    assert (
        "tcp/127.0.0.1:7447"
        in runtime
    )


def test_nav2_controller_output_is_disconnected():
    supervisor = read(SUPERVISOR)

    assert (
        "/tony2_nav_"
        in supervisor
    )

    assert (
        "cmd_vel_blocked"
        in supervisor
    )

    assert (
        '"/cmd_vel",'
        not in supervisor
    )


def test_goal_submission_is_forced_closed():
    runtime = read(RUNTIME)

    assert (
        "MOTION_OUTPUT_CONNECTED = False"
        in runtime
    )

    assert (
        "and self.MOTION_OUTPUT_CONNECTED"
        in runtime
    )

    assert (
        '"motion_output_connected":'
        in runtime
    )


def test_isolation_bridge_has_no_motion_capability():
    for path in (
        SOURCE,
        SINK,
    ):
        source = read(path)

        for forbidden in (
            "Twist",
            "cmd_vel",
            "NavigateToPose",
            "ActionClient",
        ):
            assert forbidden not in source


def test_old_same_domain_sensor_relay_is_not_owned():
    supervisor = read(SUPERVISOR)

    assert (
        "tony2_navigation_sensor_relay.py"
        not in supervisor
    )


def test_sink_does_not_shadow_rclpy_node_publishers_property():
    sink = read(SINK)

    assert "self.publishers" not in sink

    assert (
        "self._channel_publishers"
        in sink
    )
