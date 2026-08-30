#!/usr/bin/env python3

"""Regression tests for guarded goal ROS argument parsing."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent

GOAL = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_goal.py"
)


def run_helper(
    tmp_path,
    extra_args,
):
    result_file = (
        tmp_path
        / "goal_result.json"
    )

    command = [
        "/usr/bin/python3",
        str(GOAL),
        "--x",
        "nan",
        "--y",
        "0",
        "--yaw",
        "0",
        "--max-distance",
        "0.5",
        "--timeout",
        "25",
        "--result-file",
        str(result_file),
        *extra_args,
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5.0,
    )

    return (
        result,
        result_file,
    )


def test_ros_remap_arguments_reach_application_parser_safely(
    tmp_path,
):
    result, result_file = run_helper(
        tmp_path,
        [
            "--ros-args",
            "-r",
            "/tf:=/nav_tf",
        ],
    )

    # nan is deliberately rejected after argument parsing,
    # before rclpy.init() can create any ROS node.
    assert result.returncode != 0

    assert (
        "All numeric arguments must be finite."
        in result.stdout
    )

    assert (
        "unrecognized arguments: --ros-args"
        not in result.stdout
    )

    assert (
        "/tf:=/nav_tf"
        not in result.stdout
    )

    assert not result_file.exists()


def test_unknown_application_arguments_remain_strict(
    tmp_path,
):
    result, result_file = run_helper(
        tmp_path,
        [
            "--unexpected-option",
            "bad",
            "--ros-args",
            "-r",
            "/tf:=/nav_tf",
        ],
    )

    assert result.returncode == 2

    assert (
        "unrecognized arguments:"
        in result.stdout
    )

    assert (
        "--unexpected-option"
        in result.stdout
    )

    assert not result_file.exists()
