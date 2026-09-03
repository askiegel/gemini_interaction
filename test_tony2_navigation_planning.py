from pathlib import Path


ROOT = Path(__file__).resolve().parent

SERVER = (
    ROOT
    / "voice_relay"
    / "server.py"
).read_text(
    encoding="utf-8"
)

RUNTIME = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_runtime.py"
).read_text(
    encoding="utf-8"
)

HELPER = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_plan.py"
).read_text(
    encoding="utf-8"
)

MARKER = (
    "/* Isolated guarded read-only "
    "planning path controller */"
)


def planning_ui_source():
    candidates = (
        ROOT
        / "voice_relay"
        / "operator_console.js",
        ROOT
        / "voice_relay"
        / "index.html",
    )

    matches = []

    for path in candidates:
        text = path.read_text(
            encoding="utf-8"
        )

        if MARKER in text:
            matches.append(text)

    assert len(matches) == 1

    source = matches[0]

    start = source.index(
        MARKER
    )

    end = source.index(
        "\n})();",
        start,
    )

    return source[
        start:end
    ]


CONTROL = planning_ui_source()


def method_source(
    source,
    name,
    next_name,
):
    start = source.index(
        f"    def {name}("
    )

    end = source.index(
        f"    def {next_name}(",
        start,
    )

    return source[
        start:end
    ]


def function_source(
    name,
    next_name,
):
    candidates = (
        f"    function {name}(",
        f"    async function {name}(",
    )

    starts = [
        CONTROL.find(item)
        for item in candidates
        if CONTROL.find(item) >= 0
    ]

    assert starts

    start = min(starts)

    next_candidates = (
        f"    function {next_name}(",
        f"    async function {next_name}(",
    )

    ends = [
        CONTROL.find(
            item,
            start + 1,
        )
        for item in next_candidates
        if CONTROL.find(
            item,
            start + 1,
        ) >= 0
    ]

    assert ends

    return CONTROL[
        start:min(ends)
    ]


def test_helper_is_compute_path_only():
    required = (
        "ComputePathToPose",
        'ACTION_NAME = "/compute_path_to_pose"',
        "request.use_start = False",
        'request.planner_id = ""',
        '"COMPUTE_PATH_TO_POSE_ONLY"',
        '"read_only": True',
        '"executed": False',
        '"navigation_goal_executed":',
        '"motion_enabled": False',
        '"frame_id": "map"',
    )

    for marker in required:
        assert marker in HELPER

    forbidden = (
        "NavigateToPose",
        "FollowPath",
        "geometry_msgs.msg.Twist",
        '"/cmd_vel"',
        "MotionArmLease",
    )

    # The docstring is allowed to explain that execution
    # capabilities are absent.
    executable_source = HELPER[
        HELPER.index(
            "import argparse"
        ):
    ]

    for marker in forbidden:
        assert marker not in executable_source


def test_runtime_has_no_planning_motion_lease():
    source = method_source(
        RUNTIME,
        "compute_path",
        "submit_goal",
    )

    assert "self.child_environment()" in source
    assert "tony2_navigation_plan.py" in source
    assert '"planner_enabled"' in source
    assert '"transform_ready"' in source
    assert '"motion_output_connected"' in source
    assert "self._motion_lock" in source

    forbidden = (
        "_begin_motion_lease(",
        "_wait_for_motion_arm_ack(",
        "submit_goal(",
        "NavigateToPose",
    )

    for marker in forbidden:
        assert marker not in source


def test_dashboard_server_exposes_read_only_tony2_path():
    source = method_source(
        SERVER,
        "navigation_compute_path",
        "navigation_goal",
    )

    assert "get_tony2_navigation_runtime()" in source
    assert "self.ensure_mayday_stationary()" in source
    assert "runtime.compute_path(" in source
    assert '"goal_x"' in source
    assert '"goal_y"' in source
    assert '"goal_yaw"' in source
    assert "math.isfinite" in source

    forbidden = (
        '"/planning/compute-path"',
        "runtime.submit_goal(",
        "NavigateToPose",
        "cmd_vel",
    )

    for marker in forbidden:
        assert marker not in source

    assert (
        '"/dashboard/navigation-compute-path"'
        in SERVER
    )


def test_planning_overlay_uses_tony2_navigation_stack():
    required = (
        '"/dashboard/navigation-control"',
        '"/dashboard/navigation-start"',
        '"/dashboard/navigation-stop"',
        (
            '"/dashboard/'
            'navigation-initialize-localization"'
        ),
        (
            '"/dashboard/'
            'navigation-compute-path"'
        ),
    )

    for marker in required:
        assert marker in CONTROL


def test_planning_start_uses_global_tony2_localization():
    source = function_source(
        "startPlanning",
        "performAction",
    )

    assert (
        source.index("START_ENDPOINT")
        < source.index("INITIALIZE_ENDPOINT")
    )

    required = (
        "navigationWasStarted(",
        "validInitialization(",
        "initialization.final_pose",
        "initialization.uncertainty",
        "latestPose = {",
        'frame_id: "map"',
        "planningReady = true",
    )

    for marker in required:
        assert marker in source

    assert "POSE_REFRESH_ENDPOINT" not in source
    assert "LOCALIZATION_ENDPOINT" not in source


def test_planning_initialization_requires_trusted_start_anywhere():
    source = function_source(
        "validInitialization",
        "validPoseRefresh",
    )

    required = (
        "initialization.trusted === true",
        (
            'initialization.localization_method\n'
            '                === "amcl_global"'
        ),
        (
            'initialization.search_scope\n'
            '                === "full_saved_map"'
        ),
        (
            "initialization.seed_pose_used\n"
            "                === false"
        ),
        (
            "initialization.global_localization_requested\n"
            "                === true"
        ),
        (
            "initialization.initial_pose_supplied\n"
            "                === false"
        ),
        (
            "initialization.nomotion_updates_requested\n"
            "                === 20"
        ),
        (
            "initialization.navigation_goal_executed\n"
            "                === false"
        ),
        (
            "initialization.motion_enabled\n"
            "                === false"
        ),
        (
            "navigation.motion_output_connected\n"
            "                === false"
        ),
    )

    for marker in required:
        assert marker in source


def test_compute_button_uses_only_tony2_planner_action():
    source = function_source(
        "computePath",
        "navigationWasStarted",
    )

    assert "COMPUTE_ENDPOINT" in source

    assert (
        '"COMPUTE_PATH_TO_POSE_ONLY"'
        in source
    )

    for marker in (
        "path.read_only !== true",
        "path.executed !== false",
        (
            "path.navigation_goal_executed\n"
            "                    !== false"
        ),
        "path.motion_enabled !== false",
        'path.frame_id !== "map"',
    ):
        assert marker in source

    forbidden = (
        "POSE_REFRESH_ENDPOINT",
        "LOCALIZATION_ENDPOINT",
        "NAVIGATION_GOAL_ENDPOINT",
        "NavigateToPose",
        "cmd_vel",
    )

    for marker in forbidden:
        assert marker not in source


def test_planning_refresh_uses_navigation_status_not_robot_amcl():
    start = CONTROL.index(
        "    async function refresh()"
    )

    candidates = []

    for marker in (
        "\n    function ",
        "\n    async function ",
    ):
        found = CONTROL.find(
            marker,
            start + 1,
        )

        if found >= 0:
            candidates.append(found)

    end = (
        min(candidates)
        if candidates
        else len(CONTROL)
    )

    source = CONTROL[
        start:end
    ]

    assert "STATUS_ENDPOINT" in source
    assert "MAP_ENDPOINT" in source
    assert "renderPlanning(" in source

    assert "LOCALIZATION_ENDPOINT" not in source
    assert "POSE_REFRESH_ENDPOINT" not in source

def test_tony2_planning_controls_are_visible():
    html = (
        ROOT
        / "voice_relay"
        / "index.html"
    ).read_text(
        encoding="utf-8"
    )

    expected = {
        "startPlanningButton": False,
        "stopPlanningButton": True,
        "computePlanningPathButton": True,
    }

    for identifier, should_be_disabled in expected.items():
        marker = (
            f'id="{identifier}"'
        )

        start = html.index(
            marker
        )

        button_start = html.rfind(
            "<button",
            0,
            start,
        )

        button_end = html.index(
            ">",
            start,
        )

        opening = html[
            button_start:button_end
        ]

        assert " hidden" not in opening

        if should_be_disabled:
            assert " disabled" in opening
        else:
            assert " disabled" not in opening

    assert (
        "For a read-only path preview, start planning,"
        in html
    )

    assert (
        "then compute the path."
        in html
    )


def test_planning_controls_are_next_to_persistent_map():
    html = (
        ROOT
        / "voice_relay"
        / "index.html"
    ).read_text(
        encoding="utf-8"
    )

    map_index = html.index(
        'id="mapCanvas"'
    )

    planning_index = html.index(
        '<section class="planning-path-controls">'
    )

    telemetry_index = html.index(
        "Map Telemetry"
    )

    assert (
        map_index
        < planning_index
        < telemetry_index
    )

    planning_end = html.index(
        "</section>",
        planning_index,
    )

    planning = html[
        planning_index:planning_end
    ]

    assert "Start Planning" in planning
    assert "Stop Planning" in planning
    assert "Compute Selected Path" in planning

    assert (
        "without execution"
        in planning
    )

def test_confirmed_stopped_state_clears_stale_planning_display():
    required = (
        "let stoppedPlanningCleanupPending = false;",
        "function clearStoppedPlanningDisplay()",
        "latestPose = null;",
        "latestPoseReceivedAt = null;",
        "selectedGoal = null;",
        "computedPath = null;",
        '"planningSelectedGoal",',
        '"planningPathSummary",',
        '"planningPoseUncertainty",',
        (
            '"Start Planning to localize Mayday "'
        ),
        "const wasPlanningRunning =",
        "wasPlanningRunning",
        "|| stoppedPlanningCleanupPending",
        "clearStoppedPlanningDisplay();",
        (
            "stoppedPlanningCleanupPending =\n"
            "                        false;"
        ),
    )

    for marker in required:
        assert marker in CONTROL

    # A normal STOPPED refresh must not repeatedly clear a
    # newly selected goal used by the separate guarded GO flow.
    stopped = CONTROL.index(
        "const shouldClearStoppedDisplay"
    )

    guarded_clear = CONTROL.index(
        "if (shouldClearStoppedDisplay)",
        stopped,
    )

    fallback = CONTROL.index(
        "} else {",
        guarded_clear,
    )

    fallback_end = CONTROL.index(
        "setState(",
        fallback,
    )

    fallback_source = CONTROL[
        fallback:fallback_end
    ]

    assert "selectedGoal = null;" not in fallback_source
    assert "computedPath = null;" not in fallback_source


def test_stop_mayday_defers_cleanup_until_backend_reports_stopped():
    start = CONTROL.index(
        "async function stopMayday()"
    )

    boundaries = []

    for marker in (
        "\n    function ",
        "\n    async function ",
    ):
        found = CONTROL.find(
            marker,
            start + 1,
        )

        if found >= 0:
            boundaries.append(found)

    assert boundaries

    source = CONTROL[
        start:min(boundaries)
    ]

    assert (
        "stoppedPlanningCleanupPending ="
        in source
    )

    # The STOP handler itself must not fabricate a clean
    # stopped display before navigation-control confirms it.
    assert (
        "clearStoppedPlanningDisplay();"
        not in source
    )
