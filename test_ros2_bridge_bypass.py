"""Offline policy tests for the legacy ROS2 bridge motion boundary."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from ros2_bridge.bridge import ROS2Bridge
from ros2_bridge import motion_test


def bridge_with_fake_ros():
    bridge = ROS2Bridge.__new__(ROS2Bridge)
    bridge.motion_topic = "/cmd_vel"
    bridge.last_command = None
    bridge.ros2_available = True
    bridge.bridge_status = "READY"
    bridge._publish_ros2_motion = Mock(return_value={"ok": True, "executed": True})
    return bridge


def test_positive_forward_is_denied_without_publish():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=0.05)

    assert result["ok"] is False
    assert "Positive forward motion is denied" in result["reason"]
    bridge._publish_ros2_motion.assert_not_called()


def test_positive_forward_with_rotation_is_denied_without_publish():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=0.05, angular_z=0.4)

    assert result["ok"] is False
    bridge._publish_ros2_motion.assert_not_called()


def test_zero_linear_motion_preserves_ros2_behavior():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=0.0, angular_z=0.4)

    assert result == {"ok": True, "executed": True}
    bridge._publish_ros2_motion.assert_called_once()


def test_reverse_motion_preserves_ros2_behavior():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=-0.05)

    assert result == {"ok": True, "executed": True}
    bridge._publish_ros2_motion.assert_called_once()


def test_pure_rotation_preserves_ros2_behavior():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=0.0, angular_z=-0.4)

    assert result == {"ok": True, "executed": True}
    bridge._publish_ros2_motion.assert_called_once()


def test_explicit_zero_stop_remains_available():
    bridge = bridge_with_fake_ros()

    result = bridge.motion_request(linear_x=0.0, angular_z=0.0, duration_sec=0.0, source="stop")

    assert result == {"ok": True, "executed": True}
    bridge._publish_ros2_motion.assert_called_once()


def test_legacy_motion_test_positive_request_is_denied():
    bridge = bridge_with_fake_ros()

    with patch.object(bridge, "_publish_ros2_motion") as publish:
        result = bridge.motion_request(
            linear_x=0.05,
            angular_z=0.0,
            duration_sec=0.5,
            source="ros2_bridge_motion_test",
        )

    assert result["ok"] is False
    publish.assert_not_called()


def test_motion_test_utility_cannot_publish_forward():
    bridge = bridge_with_fake_ros()

    with patch.object(motion_test, "ROS2Bridge", return_value=bridge), \
         redirect_stdout(StringIO()):
        motion_test.main()

    bridge._publish_ros2_motion.assert_not_called()


def test_malformed_inputs_do_not_reach_ros_publish():
    for kwargs in (
        {"linear_x": "forward"},
        {"linear_x": None},
        {"linear_x": float("nan")},
        {"linear_x": float("inf")},
        {"duration_sec": float("nan")},
    ):
        bridge = bridge_with_fake_ros()
        result = bridge.motion_request(**kwargs)
        assert result["ok"] is False
        bridge._publish_ros2_motion.assert_not_called()


def test_policy_has_no_robot_bridge_or_world_model_dependency():
    bridge = bridge_with_fake_ros()

    with patch("robot_bridge.client.RobotBridgeClient", side_effect=AssertionError("unexpected RobotBridgeClient")), \
         patch.object(bridge, "_publish_ros2_motion") as publish:
        result = bridge.motion_request(linear_x=0.1)

    assert result["ok"] is False
    publish.assert_not_called()
