#!/usr/bin/env python3

from behavior_manager import BehaviorManager


EPSILON = 1e-9


class FakeStreamingRobot:
    def __init__(self):
        self.commands = []

    def streaming_motion(
        self,
        linear_x,
        angular_z,
        watchdog_timeout,
    ):
        command = {
            "ok": True,
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "watchdog_timeout": float(
                watchdog_timeout
            ),
        }

        self.commands.append(command)
        return command


def assert_close(
    actual,
    expected,
):
    assert abs(actual - expected) < EPSILON, (
        f"Expected {expected}, received {actual}"
    )


def make_manager():
    manager = BehaviorManager.__new__(
        BehaviorManager
    )
    manager.robot = FakeStreamingRobot()
    manager._reset_follow_servo_state()
    return manager


def test_streaming_rate_limit():
    manager = make_manager()

    first = manager._execute_follow_streaming_motion(
        linear_x=0.0,
        angular_z=0.80,
    )

    second = manager._execute_follow_streaming_motion(
        linear_x=0.0,
        angular_z=0.80,
    )

    third = manager._execute_follow_streaming_motion(
        linear_x=0.0,
        angular_z=-0.80,
    )

    assert_close(
        first["angular_z"],
        0.09,
    )
    assert_close(
        second["angular_z"],
        0.18,
    )
    assert_close(
        third["angular_z"],
        0.09,
    )

    print("PASS: streaming turns ramp gradually")
    print("PASS: direction reversal moves toward zero")


def test_linear_speed_unchanged():
    manager = make_manager()

    result = manager._execute_follow_streaming_motion(
        linear_x=0.14,
        angular_z=0.20,
    )

    assert_close(
        result["linear_x"],
        0.14,
    )
    assert_close(
        result["angular_z"],
        0.09,
    )

    print("PASS: linear speed remains unchanged")


def test_zero_stops_residual_turn():
    manager = make_manager()

    manager._follow_previous_angular_command = 0.18

    result = manager._execute_follow_streaming_motion(
        linear_x=0.14,
        angular_z=0.0,
    )

    assert_close(
        result["angular_z"],
        0.0,
    )

    print("PASS: centered command removes residual rotation")


def test_filtered_error_and_hysteresis():
    manager = make_manager()

    manager._prepare_follow_servo_mission(
        "follow-one"
    )

    first = manager._follow_effective_horizontal_error(
        -180.0
    )

    assert (
        manager._follow_steering_latch
        == "LEFT"
    )
    assert (
        first
        < -manager.CENTER_TOLERANCE_PIXELS
    )

    second = manager._follow_effective_horizontal_error(
        -40.0
    )

    assert (
        manager._follow_steering_latch
        == "CENTER"
    )
    assert second == -40.0

    final = manager._follow_effective_horizontal_error(
        0.0
    )

    assert (
        manager._follow_steering_latch
        == "CENTER"
    )
    assert final == 0.0

    print("PASS: noisy error remains directionally stable")
    print("PASS: steering releases near image center")


def test_new_mission_resets_history():
    manager = make_manager()

    manager._prepare_follow_servo_mission(
        "follow-one"
    )
    manager._follow_effective_horizontal_error(
        -200.0
    )
    manager._limit_follow_angular_command(
        0.80
    )

    changed = manager._prepare_follow_servo_mission(
        "follow-two"
    )

    assert changed is True
    assert (
        manager._follow_filtered_horizontal_error
        is None
    )
    assert (
        manager._follow_steering_latch
        == "CENTER"
    )
    assert_close(
        manager._follow_previous_angular_command,
        0.0,
    )

    print("PASS: new follow mission resets servo history")


def main():
    print("==========================================")
    print("SMOOTH FOLLOW_PERSON CONTROLLER TEST")
    print("==========================================")

    test_streaming_rate_limit()
    test_linear_speed_unchanged()
    test_zero_stops_residual_turn()
    test_filtered_error_and_hysteresis()
    test_new_mission_resets_history()

    print()
    print(
        "All smooth FOLLOW_PERSON controller "
        "tests passed."
    )


if __name__ == "__main__":
    main()
