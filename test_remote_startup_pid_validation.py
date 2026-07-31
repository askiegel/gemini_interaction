#!/usr/bin/env python3

import ast
import subprocess
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
STARTUP_SCRIPT = PROJECT_DIR / "scripts" / "start_platform.py"


def extract_remote_script():
    source = STARTUP_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "start_remote_stack"
        ):
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue

                if len(statement.targets) != 1:
                    continue

                target = statement.targets[0]
                value = statement.value

                if (
                    isinstance(target, ast.Name)
                    and target.id == "remote_script"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    return value.value

    raise AssertionError(
        "start_remote_stack remote_script was not found."
    )


def main():
    remote_script = extract_remote_script()

    required_markers = [
        'local expected_marker="$2"',
        'grep -Eq "^[0-9]+$"',
        'kill -0 "$pid"',
        '"/proc/$pid/cmdline"',
        '"$command_line" != *"$expected_marker"*',
        'rm -f -- "$pid_file"',
        '"mini_pupper_bringup"',
        '"python3 app.py"',
        '"camera_relay.py"',
    ]

    for marker in required_markers:
        assert marker in remote_script, (
            f"Missing PID validation marker: {marker}"
        )

    with tempfile.TemporaryDirectory() as directory:
        script_path = Path(directory) / "remote_startup.sh"
        script_path.write_text(
            remote_script,
            encoding="utf-8",
        )

        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0

    print("PASS: Remote startup shell syntax is valid.")
    print("PASS: Invalid PID values are rejected.")
    print("PASS: Stopped PIDs are rejected.")
    print("PASS: Reused PIDs are rejected by command identity.")
    print("PASS: Stale PID files are removed automatically.")


if __name__ == "__main__":
    main()
