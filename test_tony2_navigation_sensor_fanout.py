from pathlib import Path


ROOT = Path(__file__).resolve().parent

SUPERVISOR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_supervisor.py"
)

RELAY = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_sensor_relay.py"
)


def test_sensor_relay_has_one_remote_scan_ingress():
    source = RELAY.read_text(
        encoding="utf-8"
    )

    assert 'SCAN_INPUT = "/scan"' in source

    assert (
        'SCAN_OUTPUT = "/tony2_nav_scan"'
        in source
    )


def test_sensor_relay_has_one_remote_odom_ingress():
    source = RELAY.read_text(
        encoding="utf-8"
    )

    assert 'ODOM_INPUT = "/odom"' in source

    assert (
        'ODOM_OUTPUT = "/tony2_nav_odom"'
        in source
    )


def test_nav2_consumes_local_sensor_topics():
    source = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    assert (
        '"/scan",\n'
        '            "/tony2_nav_scan",'
        in source
    )

    assert (
        '"/odom",\n'
        '            "/tony2_nav_odom",'
        in source
    )


def test_sensor_ingress_is_owned_by_supervisor():
    source = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    assert (
        '"tony2_navigation_sensor_relay.py"'
        in source
    )

    assert "sensor_ingress = ExecuteProcess(" in source

    assert (
        "sensor_ingress,\n"
        "            delayed_navigation,"
        in source
    )


def test_filtered_tf_route_is_preserved():
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


def test_sensor_relay_has_no_motion_capability():
    source = RELAY.read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "Twist",
        "cmd_vel",
        "NavigateToPose",
        "ActionClient",
    ):
        assert forbidden not in source
