from pathlib import Path


HTML = (
    Path(__file__).resolve().parent
    / "voice_relay"
    / "index.html"
).read_text(
    encoding="utf-8"
)


def selection_source():
    start = HTML.index(
        "function selectGoal(event)"
    )

    end = HTML.index(
        "\n    function ",
        start + 10,
    )

    return HTML[start:end]


def test_click_goal_uses_current_map_pose():
    source = selection_source()

    required = (
        "latestPose",
        "latestPose.position",
        "currentPosition.x",
        "currentPosition.y",
        "deltaX",
        "deltaY",
    )

    for marker in required:
        assert marker in source


def test_click_goal_heading_follows_destination():
    source = selection_source()

    assert (
        "destinationYaw = Math.atan2("
        in source
    )

    assert (
        "deltaY,"
        in source
    )

    assert (
        "deltaX"
        in source
    )


def test_selected_goal_uses_computed_heading():
    source = selection_source()

    assert '''selectedGoal = {
            x: goal.x,
            y: goal.y,
            yaw: destinationYaw,
        };''' in source

    assert '''selectedGoal = {
            x: goal.x,
            y: goal.y,
            yaw: goal.yaw,
        };''' not in source


def test_near_zero_click_preserves_current_heading():
    source = selection_source()

    assert (
        "destinationYaw = currentYaw;"
        in source
    )

    assert (
        "Math.hypot("
        in source
    )

    assert "> 0.01" in source
