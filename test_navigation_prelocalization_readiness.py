from pathlib import Path


HTML = (
    Path(__file__).resolve().parent
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)


def start_controller():
    marker = (
        "/* Isolated guarded read-only "
        "planning path controller */"
    )

    start = HTML.index(marker)
    end = HTML.index(
        "</script>",
        start,
    )

    return HTML[start:end]


def test_start_waits_for_stable_prelocalization_readiness():
    source = start_controller()

    required = (
        "PRELOCALIZATION_POLL_MS = 250",
        "PRELOCALIZATION_MAX_POLLS = 80",
        "PRELOCALIZATION_STABLE_SAMPLES = 3",
        "waitForPreLocalizationReady()",
        "navigation.running === true",
        "navigation.map_server_enabled === true",
        "navigation.localization_enabled === true",
        "navigation.action_server_ready === true",
        "navigation.motion_egress_ready === true",
        "navigation.motion_egress_idle === true",
        "navigation.goal_active === false",
        "navigation.motion_output_connected === false",
    )

    for marker in required:
        assert marker in source


def test_start_does_not_use_fixed_delay_or_blind_retries():
    source = start_controller()

    forbidden = (
        "INITIALIZE_DISCOVERY_DELAY_MS",
        "INITIALIZE_RETRY_DELAY_MS",
        "INITIALIZE_MAX_ATTEMPTS",
        "Retrying stationary global localization",
    )

    for marker in forbidden:
        assert marker not in source


def test_start_performs_one_initialization_after_readiness():
    source = start_controller()

    start = source.index(
        "async function startPlanning()"
    )

    body = source[start:]

    readiness = body.index(
        "await waitForPreLocalizationReady();"
    )

    initialization = body.index(
        "const initializeResult = await fetchJson("
    )

    assert readiness < initialization

    initialization_region = body[
        readiness:
        body.index(
            "const initialization = (",
            initialization,
        )
    ]

    assert (
        initialization_region.count(
            "INITIALIZE_ENDPOINT"
        )
        == 1
    )
