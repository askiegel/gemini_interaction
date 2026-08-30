from pathlib import Path


RUNTIME = (
    Path(__file__).resolve().parent
    / "voice_relay"
    / "tony2_mapping_runtime.py"
)


def runtime_source():
    return RUNTIME.read_text(
        encoding="utf-8"
    )


def test_cartographer_uses_filtered_mayday_tf():
    source = runtime_source()

    cartographer_start = source.index(
        "cartographer_pid = self._spawn("
    )

    cartographer_end = source.index(
        "self.cartographer_log",
        cartographer_start,
    )

    command = source[
        cartographer_start:cartographer_end
    ]

    assert (
        '"/tf:=/mayday_navigation_tf"'
        in command
    )


def test_mapping_probe_uses_filtered_mayday_tf():
    source = runtime_source()

    probe_start = source.index(
        "return self._spawn(",
        source.index("def _start_probe("),
    )

    probe_end = source.index(
        "self.probe_log",
        probe_start,
    )

    command = source[
        probe_start:probe_end
    ]

    assert (
        '"/tf:=/mayday_navigation_tf"'
        in command
    )


def test_mapping_runtime_does_not_reintroduce_raw_tf_remap():
    source = runtime_source()

    assert (
        '"/tf:=/mayday_navigation_tf"'
        in source
    )

    assert (
        '"/mayday_navigation_tf:=/tf"'
        not in source
    )


def test_feature_has_no_motion_capability():
    source = runtime_source()

    start = source.index(
        "def _start_probe("
    )

    end = source.index(
        "def ensure_probe(",
        start,
    )

    probe_source = source[start:end]

    for forbidden in (
        "Twist",
        "cmd_vel",
        "NavigateToPose",
        "ActionClient",
    ):
        assert forbidden not in probe_source
