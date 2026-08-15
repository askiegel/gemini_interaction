from pathlib import Path
import subprocess


SCRIPT_PATH = Path("scripts/prepare_mayday_session.sh")
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
NORMALIZED_SCRIPT = SCRIPT.replace("[", "").replace("]", "")


def test_session_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_default_mode_is_non_mutating_check():
    assert 'MODE="${1:---check}"' in SCRIPT
    assert "--check|--prepare" in SCRIPT
    assert 'if [ "$MODE" = "--prepare" ]; then' in SCRIPT


def test_prepare_mode_stops_optional_runtimes():
    for endpoint in (
        "/navigation/stop",
        "/planning/stop",
        "/mapping/stop",
        "/localization/stop",
        "/stop",
    ):
        assert endpoint in SCRIPT


def test_ros_environment_contract_is_fixed():
    for contract in (
        'test "$ROS_DISTRO" = "humble"',
        'test "$ROS_DOMAIN_ID" = "42"',
        'test "$ROS_LOCALHOST_ONLY" = "0"',
    ):
        assert contract in SCRIPT


def test_standard_bringup_is_required():
    assert "mini_pupper_bringup bringup.launch.py" in SCRIPT
    assert "Expected exactly one standard bringup owner" in SCRIPT

    for process in (
        "robot_state_publisher",
        "quadruped_controller_node",
        "state_estimation_node",
        "base_to_footprint_ekf",
        "footprint_to_odom_ekf",
        "ldlidar_stl_ros2_node",
        "servo_interface",
        "imu_interface",
    ):
        assert process in NORMALIZED_SCRIPT


def test_navigation_and_slam_must_be_absent():
    for process in (
        "guarded_navigation.launch.py",
        "planning.launch.py",
        "localization.launch.py",
        "nav2_amcl",
        "nav2_planner",
        "nav2_controller",
        "nav2_bt_navigator",
        "slam_toolbox",
        "cartographer",
    ):
        assert process in NORMALIZED_SCRIPT


def test_preserved_platform_services_are_required():
    assert '"$ROBOT/status"' in SCRIPT
    assert '"$CAMERA/camera/latest.jpg"' in SCRIPT
    assert '"$DASHBOARD/"' in SCRIPT
    assert '"$ROBOT/telemetry/lidar"' in SCRIPT
    assert 'payload.get("speech")' in SCRIPT


def test_repository_changes_are_reported_not_cleaned():
    assert "status --short --branch" in SCRIPT

    for destructive_command in (
        "git reset",
        "git checkout",
        "git clean",
        "git restore",
        "rm -rf",
    ):
        assert destructive_command not in SCRIPT


def test_final_velocity_must_be_zero():
    assert 'result.get("linear_x") != 0.0' in SCRIPT
    assert 'result.get("angular_z") != 0.0' in SCRIPT


def test_success_result_is_unambiguous():
    assert "===== SESSION READY =====" in SCRIPT
