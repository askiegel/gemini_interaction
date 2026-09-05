from pathlib import Path


ROOT = Path(__file__).resolve().parent

RUNNER = (
    ROOT
    / "scripts"
    / "run_tony2_service.py"
)

SYSTEMD = (
    ROOT
    / "deploy"
    / "systemd"
)

UNITS = (
    SYSTEMD
    / "mayday-vision-server.service",
    SYSTEMD
    / "mayday-vision-service.service",
    SYSTEMD
    / "mayday-cognitive-runtime.service",
    SYSTEMD
    / "mayday-voice-relay.service",
)


def test_boot_files_exist():
    assert RUNNER.is_file()

    for path in UNITS:
        assert path.is_file()


def test_exact_existing_entrypoints():
    text = RUNNER.read_text(
        encoding="utf-8"
    )

    required = (
        "start_platform.py",
        "server:app",
        '"8000"',
        "vision_service.py",
        '"0.35"',
        "runtime_api.py",
        "voice_relay/server.py",
    )

    for item in required:
        assert item in text


def test_runner_preserves_platform_urls():
    text = RUNNER.read_text(
        encoding="utf-8"
    )

    required = (
        "ROBOT_BRIDGE_URL",
        "CAMERA_RELAY_URL",
        "VISION_SERVER_URL",
        "COGNITIVE_RUNTIME_URL",
        "VOICE_RELAY_URL",
        "VISION_CAMERA_URL",
    )

    for item in required:
        assert item in text


def test_runner_preserves_legacy_pid_files():
    text = RUNNER.read_text(
        encoding="utf-8"
    )

    required = (
        "vision_server.pid",
        "vision_service.pid",
        "runtime_api.pid",
        "voice_relay.pid",
    )

    for item in required:
        assert item in text


def test_units_boot_and_restart():
    for path in UNITS:
        text = path.read_text(
            encoding="utf-8"
        )

        assert (
            "WantedBy=multi-user.target"
            in text
        )

        assert (
            "Restart=on-failure"
            in text
        )

        assert "User=tkieg" in text
        assert "KillMode=control-group" in text


def test_dependency_order():
    vision_service = (
        SYSTEMD
        / "mayday-vision-service.service"
    ).read_text(
        encoding="utf-8"
    )

    runtime = (
        SYSTEMD
        / "mayday-cognitive-runtime.service"
    ).read_text(
        encoding="utf-8"
    )

    voice = (
        SYSTEMD
        / "mayday-voice-relay.service"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "mayday-vision-server.service"
        in vision_service
    )

    assert (
        "mayday-vision-service.service"
        in runtime
    )

    assert (
        "mayday-cognitive-runtime.service"
        in voice
    )


def test_cognitive_services_reference_env_file():
    for name in (
        "mayday-vision-service.service",
        "mayday-cognitive-runtime.service",
        "mayday-voice-relay.service",
    ):
        text = (
            SYSTEMD
            / name
        ).read_text(
            encoding="utf-8"
        )

        assert (
            "EnvironmentFile="
            in text
        )

        assert (
            "/robot_services/cognitive/.env"
            in text
        )


def test_no_secret_is_embedded():
    forbidden = (
        "GEMINI_API_KEY=",
        "GOOGLE_API_KEY=",
        "OPENAI_API_KEY=",
        "sk-",
        "AIza",
    )

    for path in UNITS:
        text = path.read_text(
            encoding="utf-8"
        )

        for item in forbidden:
            assert item not in text


def test_no_robot_motion_surface():
    text = RUNNER.read_text(
        encoding="utf-8"
    )

    forbidden = (
        "cmd_vel",
        "Twist(",
        "NavigateToPose",
        "FollowPath",
        "navigation/goal",
    )

    for item in forbidden:
        assert item not in text
