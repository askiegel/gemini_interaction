#!/usr/bin/env python3

from pathlib import Path

from voice_relay.tony2_navigation_motion_egress import (
    INPUT_STALE_SECONDS,
    INPUT_TOPIC,
    MAX_ANGULAR_Z,
    MAX_LINEAR_X,
    MotionEgressController,
    ROBOT_WATCHDOG_SECONDS,
)


ROOT = Path(
    __file__
).resolve().parent

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

EGRESS = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_motion_egress.py"
)


class FakeRobot:

    def __init__(
        self,
        stream_result=None,
        stop_result=None,
        stream_exception=None,
        stop_exception=None,
    ):
        self.stream_calls = []
        self.stop_calls = 0

        self.stream_result = (
            {
                "ok": True,
                "mode": "streaming",
            }
            if stream_result is None
            else stream_result
        )

        self.stop_result = (
            {
                "ok": True,
                "action": "stop",
            }
            if stop_result is None
            else stop_result
        )

        self.stream_exception = (
            stream_exception
        )

        self.stop_exception = (
            stop_exception
        )

    def streaming_motion(
        self,
        linear_x,
        angular_z,
        watchdog_timeout,
    ):
        self.stream_calls.append(
            {
                "linear_x": linear_x,
                "angular_z": angular_z,
                "watchdog_timeout":
                    watchdog_timeout,
            }
        )

        if self.stream_exception:
            raise self.stream_exception

        return self.stream_result

    def stop(self):
        self.stop_calls += 1

        if self.stop_exception:
            raise self.stop_exception

        return self.stop_result


def test_egress_defaults_are_guarded():
    assert (
        INPUT_TOPIC
        == "/tony2_nav_cmd_vel_egress"
    )

    assert MAX_LINEAR_X == 0.20
    assert MAX_ANGULAR_Z == 1.00

    assert (
        INPUT_STALE_SECONDS
        < ROBOT_WATCHDOG_SECONDS
    )


def test_disabled_egress_never_contacts_robot():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=False,
    )

    result = controller.accept(
        0.10,
        0.20,
        now=1.0,
    )

    assert result["forwarded"] is False

    assert (
        result["reason"]
        == "MOTION_OUTPUT_DISABLED"
    )

    assert robot.stream_calls == []
    assert robot.stop_calls == 0


def test_enabled_egress_uses_streaming_motion():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.10,
        -0.30,
        now=1.0,
    )

    assert result["ok"] is True
    assert result["forwarded"] is True

    assert robot.stream_calls == [
        {
            "linear_x": 0.10,
            "angular_z": -0.30,
            "watchdog_timeout": 0.50,
        }
    ]

    assert robot.stop_calls == 0


def test_linear_limit_violation_fail_stops():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.2001,
        0.0,
        now=1.0,
    )

    assert result["forwarded"] is False

    assert (
        result["reason"]
        == "LINEAR_LIMIT_EXCEEDED"
    )

    assert robot.stream_calls == []
    assert robot.stop_calls == 1


def test_angular_limit_violation_fail_stops():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.0,
        1.0001,
        now=1.0,
    )

    assert result["forwarded"] is False

    assert (
        result["reason"]
        == "ANGULAR_LIMIT_EXCEEDED"
    )

    assert robot.stream_calls == []
    assert robot.stop_calls == 1


def test_nonfinite_command_fail_stops():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        float("nan"),
        0.0,
        now=1.0,
    )

    assert (
        result["reason"]
        == "NONFINITE_COMMAND"
    )

    assert robot.stream_calls == []
    assert robot.stop_calls == 1


def test_unsupported_twist_axis_fail_stops():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.05,
        0.0,
        unsupported_axis=True,
        now=1.0,
    )

    assert (
        result["reason"]
        == "UNSUPPORTED_TWIST_AXIS"
    )

    assert robot.stream_calls == []
    assert robot.stop_calls == 1


def test_failed_stream_request_fail_stops():
    robot = FakeRobot(
        stream_result={
            "ok": False,
            "error": "offline",
        }
    )

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.05,
        0.0,
        now=1.0,
    )

    assert (
        result["reason"]
        == "STREAM_REQUEST_FAILED"
    )

    assert len(robot.stream_calls) == 1
    assert robot.stop_calls == 1


def test_stream_exception_fail_stops():
    robot = FakeRobot(
        stream_exception=TimeoutError(
            "timeout"
        )
    )

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    result = controller.accept(
        0.05,
        0.0,
        now=1.0,
    )

    assert (
        result["reason"]
        == "STREAM_REQUEST_EXCEPTION"
    )

    assert len(robot.stream_calls) == 1
    assert robot.stop_calls == 1


def test_stale_nav2_input_requests_stop_once():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=True,
    )

    assert controller.accept(
        0.08,
        0.10,
        now=10.0,
    )["ok"] is True

    fresh = controller.tick(
        now=(
            10.0
            + INPUT_STALE_SECONDS
            - 0.01
        )
    )

    assert fresh["action"] == "FRESH"
    assert robot.stop_calls == 0

    stale = controller.tick(
        now=(
            10.0
            + INPUT_STALE_SECONDS
            + 0.01
        )
    )

    assert (
        stale["reason"]
        == "NAV2_COMMAND_STALE"
    )

    assert robot.stop_calls == 1

    idle = controller.tick(
        now=20.0
    )

    assert idle["action"] == "IDLE"
    assert robot.stop_calls == 1


def test_disabled_shutdown_does_not_contact_robot():
    robot = FakeRobot()

    controller = MotionEgressController(
        robot,
        enabled=False,
    )

    result = controller.shutdown()

    assert result["action"] == "DISABLED"
    assert robot.stop_calls == 0


def test_adapter_has_no_ros_motion_publisher():
    source = EGRESS.read_text(
        encoding="utf-8"
    )

    assert "create_publisher" not in source
    assert '"/cmd_vel"' not in source

    assert "create_subscription" in source

    assert (
        "RobotBridgeClient"
        in source
    )

    assert (
        "streaming_motion"
        in source
    )


def test_adapter_is_integrated_but_disabled():
    supervisor = SUPERVISOR.read_text(
        encoding="utf-8"
    )

    runtime = RUNTIME.read_text(
        encoding="utf-8"
    )

    egress = EGRESS.read_text(
        encoding="utf-8"
    )

    assert (
        "tony2_navigation_motion_egress.py"
        in supervisor
    )

    assert (
        "cmd_vel_egress"
        in supervisor
    )

    assert (
        "cmd_vel_blocked"
        not in supervisor
    )

    assert (
        "--enable-motion"
        not in supervisor
    )

    assert (
        "MOTION_OUTPUT_CONNECTED = False"
        in runtime
    )

    before_main = egress.split(
        "def main():",
        1,
    )[0]

    assert (
        "from robot_bridge.client "
        "import RobotBridgeClient"
        not in before_main
    )

    assert (
        "if args.enable_motion:"
        in egress
    )
