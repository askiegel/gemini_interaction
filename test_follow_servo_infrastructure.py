#!/usr/bin/env python3

from behavior_manager import BehaviorManager


EPSILON = 1e-9


def make_manager():
    manager = BehaviorManager.__new__(
        BehaviorManager
    )
    manager._reset_follow_servo_state()
    return manager


def assert_close(
    actual,
    expected,
):
    assert abs(actual - expected) < EPSILON, (
        f"Expected {expected}, received {actual}"
    )


def test_state_reset():
    manager = make_manager()

    assert (
        manager._follow_filtered_horizontal_error
        is None
    )
    assert manager._follow_steering_latch == "CENTER"
    assert_close(
        manager._follow_previous_angular_command,
        0.0,
    )
    assert manager._follow_servo_mission_id is None

    print("PASS: servo state resets cleanly")


def test_mission_lifecycle():
    manager = make_manager()

    changed = manager._prepare_follow_servo_mission(
        "mission-one"
    )

    assert changed is True
    assert (
        manager._follow_servo_mission_id
        == "mission-one"
    )

    manager._follow_filtered_horizontal_error = (
        -120.0
    )
    manager._follow_steering_latch = "LEFT"
    manager._follow_previous_angular_command = 0.27

    changed = manager._prepare_follow_servo_mission(
        "mission-one"
    )

    assert changed is False
    assert_close(
        manager._follow_filtered_horizontal_error,
        -120.0,
    )
    assert manager._follow_steering_latch == "LEFT"
    assert_close(
        manager._follow_previous_angular_command,
        0.27,
    )

    changed = manager._prepare_follow_servo_mission(
        "mission-two"
    )

    assert changed is True
    assert (
        manager._follow_servo_mission_id
        == "mission-two"
    )
    assert (
        manager._follow_filtered_horizontal_error
        is None
    )
    assert manager._follow_steering_latch == "CENTER"
    assert_close(
        manager._follow_previous_angular_command,
        0.0,
    )

    print("PASS: mission changes reset servo history")


def test_low_pass_filter():
    manager = make_manager()

    first = manager._filter_follow_horizontal_error(
        -160.0
    )
    second = manager._filter_follow_horizontal_error(
        -80.0
    )

    assert_close(first, -160.0)
    assert_close(second, -140.0)

    print("PASS: horizontal error uses low-pass filtering")


def test_left_hysteresis():
    manager = make_manager()

    result = (
        manager._follow_effective_horizontal_error(
            -140.0
        )
    )

    assert manager._follow_steering_latch == "LEFT"
    assert (
        result
        < -manager.CENTER_TOLERANCE_PIXELS
    )

    for _ in range(10):
        result = (
            manager._follow_effective_horizontal_error(
                -60.0
            )
        )

    assert manager._follow_steering_latch == "LEFT"
    assert (
        result
        < -manager.CENTER_TOLERANCE_PIXELS
    )

    for _ in range(10):
        result = (
            manager._follow_effective_horizontal_error(
                0.0
            )
        )

    assert manager._follow_steering_latch == "CENTER"
    assert (
        abs(result)
        <= manager.CENTER_TOLERANCE_PIXELS
    )

    print("PASS: left steering uses hysteresis")


def test_right_hysteresis():
    manager = make_manager()

    result = (
        manager._follow_effective_horizontal_error(
            140.0
        )
    )

    assert manager._follow_steering_latch == "RIGHT"
    assert (
        result
        > manager.CENTER_TOLERANCE_PIXELS
    )

    for _ in range(10):
        result = (
            manager._follow_effective_horizontal_error(
                60.0
            )
        )

    assert manager._follow_steering_latch == "RIGHT"
    assert (
        result
        > manager.CENTER_TOLERANCE_PIXELS
    )

    for _ in range(10):
        result = (
            manager._follow_effective_horizontal_error(
                0.0
            )
        )

    assert manager._follow_steering_latch == "CENTER"
    assert (
        abs(result)
        <= manager.CENTER_TOLERANCE_PIXELS
    )

    print("PASS: right steering uses hysteresis")


def test_rate_limiter():
    manager = make_manager()

    expected = [
        0.09,
        0.18,
        0.27,
        0.36,
    ]

    actual = [
        manager._limit_follow_angular_command(
            0.90
        )
        for _ in expected
    ]

    for received, wanted in zip(
        actual,
        expected,
    ):
        assert_close(received, wanted)

    reversal = (
        manager._limit_follow_angular_command(
            -0.90
        )
    )

    assert_close(reversal, 0.27)

    print("PASS: angular velocity ramps gradually")
    print("PASS: reversal moves toward zero first")


def test_zero_command_settles():
    manager = make_manager()

    manager._follow_previous_angular_command = 0.18

    result = manager._limit_follow_angular_command(
        0.0
    )

    # The rate limiter would normally step from 0.18 to 0.09,
    # but the controller deliberately removes the final residual
    # command so the robot does not continue creeping in a turn.
    assert_close(result, 0.0)
    assert_close(
        manager._follow_previous_angular_command,
        0.0,
    )

    print("PASS: zero command removes residual turn")


def test_large_sign_change_resets_filter():
    manager = make_manager()

    left = manager._filter_follow_horizontal_error(
        -180.0
    )

    right = manager._filter_follow_horizontal_error(
        105.0
    )

    assert_close(left, -180.0)
    assert_close(right, 105.0)

    assert (
        manager._follow_filtered_horizontal_error
        == 105.0
    )

    print(
        "PASS: large center crossing resets "
        "the horizontal filter"
    )



def main():
    print("==========================================")
    print("FOLLOW_PERSON SERVO INFRASTRUCTURE TEST")
    print("==========================================")

    test_state_reset()
    test_mission_lifecycle()
    test_low_pass_filter()
    test_left_hysteresis()
    test_right_hysteresis()
    test_rate_limiter()
    test_zero_command_settles()
    test_large_sign_change_resets_filter()

    print()
    print(
        "All FOLLOW_PERSON servo infrastructure "
        "tests passed."
    )


if __name__ == "__main__":
    main()
