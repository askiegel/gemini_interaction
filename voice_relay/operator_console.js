"use strict";

/*
 * Mini Pupper 2 Operator Console v2
 *
 * This module performs presentation only. All steering, distance, and
 * tracking decisions come directly from CognitiveRuntime.tracking.
 */

(() => {
    const STATUS_URL = "/dashboard/status";
    const REFRESH_INTERVAL_MS = 500;

    let previousVisionTimestamp = null;
    let previousVisionTimestampReceivedAt = null;
    let estimatedFps = null;

    function findCardByHeading(title) {
        return Array.from(
            document.querySelectorAll(".card")
        ).find((card) => {
            const heading = card.querySelector("h2");

            return (
                heading &&
                heading.textContent.trim() === title
            );
        });
    }

    function createMetric(name, id, initialValue = "—") {
        const row = document.createElement("div");
        row.className = "operator-value";

        const label = document.createElement("span");
        label.className = "operator-value-name";
        label.textContent = name;

        const value = document.createElement("span");
        value.className = "operator-value-data";
        value.id = id;
        value.textContent = initialValue;

        row.append(label, value);
        return row;
    }

    function createSectionTitle(text) {
        const title = document.createElement("div");
        title.className = "operator-section-title";
        title.textContent = text;
        return title;
    }

    function buildTelemetryCard() {
        const card = document.createElement("div");
        card.className = "card operator-telemetry-card";

        const heading = document.createElement("h2");
        heading.textContent = "Operator Telemetry";

        const state = document.createElement("div");
        state.id = "operatorState";
        state.className = "operator-state-large stopped";
        state.textContent = "STOPPED";

        const trackingGrid =
            document.createElement("div");

        trackingGrid.className = "operator-value-grid";

        [
            ["Target", "operatorTarget"],
            ["Confidence", "operatorConfidence"],
            ["Direction", "operatorDirection"],
            ["Horizontal error", "operatorError"],
            ["Distance", "operatorDistance"],
            ["Target area", "operatorArea"],
            ["Detection age", "operatorAge"],
            ["Vision FPS", "operatorFps"],
        ].forEach(([name, id]) => {
            trackingGrid.appendChild(
                createMetric(name, id)
            );
        });

        const runtimeGrid =
            document.createElement("div");

        runtimeGrid.className = "operator-value-grid";

        [
            ["Runtime", "operatorRuntime"],
            ["Mission", "operatorMission"],
            ["Queue", "operatorQueue"],
            ["Robot", "operatorRobot"],
            ["Camera", "operatorCamera"],
        ].forEach(([name, id]) => {
            runtimeGrid.appendChild(
                createMetric(name, id)
            );
        });

        card.append(
            heading,
            state,
            createSectionTitle("Tracking"),
            trackingGrid,
            createSectionTitle("Platform"),
            runtimeGrid
        );

        return card;
    }

    function addCameraIndicators(cameraStage) {
        const banner = document.createElement("div");
        banner.id = "operatorCameraBanner";
        banner.className =
            "operator-camera-banner stopped";
        banner.textContent = "STOPPED";

        const leftArrow = document.createElement("div");
        leftArrow.id = "operatorLeftArrow";
        leftArrow.className =
            "operator-steering-arrow left";
        leftArrow.textContent = "←";
        leftArrow.hidden = true;

        const rightArrow = document.createElement("div");
        rightArrow.id = "operatorRightArrow";
        rightArrow.className =
            "operator-steering-arrow right";
        rightArrow.textContent = "→";
        rightArrow.hidden = true;

        const centered = document.createElement("div");
        centered.id = "operatorCentered";
        centered.className =
            "operator-centered-indicator";
        centered.textContent = "● CENTERED";

        cameraStage.append(
            banner,
            leftArrow,
            rightArrow,
            centered
        );
    }

    function addCameraActions(cameraCard) {
        const actions = document.createElement("div");
        actions.className = "operator-camera-actions";

        const snapshotButton =
            document.createElement("button");

        snapshotButton.type = "button";
        snapshotButton.textContent = "Open Camera Snapshot";

        snapshotButton.addEventListener(
            "click",
            () => {
                const image =
                    document.getElementById(
                        "cameraImage"
                    );

                if (!image || !image.src) {
                    window.alert(
                        "The live camera is not available."
                    );
                    return;
                }

                window.open(
                    image.src,
                    "_blank",
                    "noopener,noreferrer"
                );
            }
        );

        actions.appendChild(snapshotButton);

        const cameraCaption =
            cameraCard.querySelector(
                ".camera-caption"
            );

        if (cameraCaption) {
            cameraCaption.insertAdjacentElement(
                "afterend",
                actions
            );
        }
        else {
            cameraCard.appendChild(actions);
        }
    }

    function buildLayout() {
        const existingGrid =
            document.querySelector(".dashboard-grid");

        const cameraCard =
            findCardByHeading("Perception");

        if (!existingGrid || !cameraCard) {
            console.error(
                "Operator Console could not locate dashboard cards."
            );
            return false;
        }

        const operatorConsole =
            document.createElement("section");

        operatorConsole.className =
            "operator-console";

        cameraCard.classList.add(
            "operator-camera-card"
        );

        const telemetryCard =
            buildTelemetryCard();

        operatorConsole.append(
            cameraCard,
            telemetryCard
        );

        existingGrid.insertAdjacentElement(
            "beforebegin",
            operatorConsole
        );

        existingGrid.classList.add(
            "operator-controls-grid"
        );

        const cameraStage =
            cameraCard.querySelector(
                ".camera-stage"
            );

        if (cameraStage) {
            addCameraIndicators(cameraStage);
        }

        addCameraActions(cameraCard);

        return true;
    }

    function formatNumber(value) {
        const numeric = Number(value);

        if (!Number.isFinite(numeric)) {
            return "—";
        }

        return Math.round(numeric).toLocaleString();
    }

    function setText(id, value) {
        const element = document.getElementById(id);

        if (element) {
            element.textContent =
                value === null ||
                value === undefined ||
                value === ""
                    ? "—"
                    : String(value);
        }
    }

    function updateFps(visionTimestamp) {
        if (
            !visionTimestamp ||
            visionTimestamp === previousVisionTimestamp
        ) {
            return;
        }

        const now = performance.now();

        if (previousVisionTimestampReceivedAt !== null) {
            const elapsedSeconds =
                (
                    now -
                    previousVisionTimestampReceivedAt
                ) / 1000;

            if (elapsedSeconds > 0) {
                const currentFps =
                    1 / elapsedSeconds;

                estimatedFps =
                    estimatedFps === null
                        ? currentFps
                        : (
                            estimatedFps * 0.75 +
                            currentFps * 0.25
                        );
            }
        }

        previousVisionTimestamp =
            visionTimestamp;

        previousVisionTimestampReceivedAt = now;
    }

    function stateClass(state, direction) {
        const normalizedState =
            String(state || "").toUpperCase();

        const normalizedDirection =
            String(direction || "").toUpperCase();

        if (
            normalizedState === "STOPPED" ||
            normalizedState === "IDLE"
        ) {
            return "stopped";
        }

        if (
            normalizedDirection === "CENTER" ||
            normalizedState === "CENTERED" ||
            normalizedState === "APPROACHING" ||
            normalizedState === "ARRIVED" ||
            normalizedState ===
                "MAINTAINING_DISTANCE"
        ) {
            return "centered";
        }

        return "active";
    }

    function updateIndicators(tracking) {
        const state =
            tracking.state || "IDLE";

        const direction =
            String(
                tracking.steering_direction || ""
            ).toUpperCase();

        const classification =
            stateClass(state, direction);

        const banner =
            document.getElementById(
                "operatorCameraBanner"
            );

        const statePanel =
            document.getElementById(
                "operatorState"
            );

        const leftArrow =
            document.getElementById(
                "operatorLeftArrow"
            );

        const rightArrow =
            document.getElementById(
                "operatorRightArrow"
            );

        const centered =
            document.getElementById(
                "operatorCentered"
            );

        if (banner) {
            banner.textContent =
                String(state).replaceAll("_", " ");

            banner.className =
                "operator-camera-banner " +
                classification;
        }

        if (statePanel) {
            statePanel.textContent =
                String(state).replaceAll("_", " ");

            statePanel.className =
                "operator-state-large " +
                classification;
        }

        if (leftArrow) {
            leftArrow.hidden = direction !== "LEFT";
        }

        if (rightArrow) {
            rightArrow.hidden = direction !== "RIGHT";
        }

        if (centered) {
            centered.classList.toggle(
                "visible",
                classification === "centered"
            );
        }
    }

    function updateConsole(status) {
        const runtime = status.runtime || {};
        const missions = status.missions || {};
        const robot = status.robot || {};
        const vision = status.vision || {};
        const tracking = runtime.tracking || {};
        const activeMission = missions.active || {};

        updateFps(vision.timestamp);

        updateIndicators(tracking);

        setText(
            "operatorTarget",
            tracking.target_label || "None"
        );

        setText(
            "operatorConfidence",
            tracking.target_confidence === null ||
            tracking.target_confidence === undefined
                ? "—"
                : (
                    Math.round(
                        Number(
                            tracking.target_confidence
                        ) * 100
                    ) + "%"
                )
        );

        setText(
            "operatorDirection",
            tracking.steering_direction || "—"
        );

        setText(
            "operatorError",
            tracking.horizontal_error === null ||
            tracking.horizontal_error === undefined
                ? "—"
                : (
                    Math.round(
                        Number(
                            tracking.horizontal_error
                        )
                    ) + " px"
                )
        );

        setText(
            "operatorDistance",
            tracking.distance_state || "—"
        );

        setText(
            "operatorArea",
            formatNumber(
                tracking.target_area
            )
        );

        setText(
            "operatorAge",
            tracking.detection_age_ms === null ||
            tracking.detection_age_ms === undefined
                ? "—"
                : (
                    formatNumber(
                        tracking.detection_age_ms
                    ) + " ms"
                )
        );

        setText(
            "operatorFps",
            estimatedFps === null
                ? "Calculating…"
                : estimatedFps.toFixed(1)
        );

        setText(
            "operatorRuntime",
            runtime.state || "OFFLINE"
        );

        setText(
            "operatorMission",
            activeMission.mission_type || "None"
        );

        setText(
            "operatorQueue",
            missions.queue_count || 0
        );

        setText(
            "operatorRobot",
            robot.ready
                ? "READY"
                : (
                    robot.connected
                        ? "NOT READY"
                        : "OFFLINE"
                )
        );

        setText(
            "operatorCamera",
            vision.camera_running
                ? "RUNNING"
                : "UNAVAILABLE"
        );
    }

    async function refreshOperatorConsole() {
        try {
            const response = await fetch(
                STATUS_URL,
                {
                    cache: "no-store"
                }
            );

            if (!response.ok) {
                throw new Error(
                    `Dashboard status returned ${response.status}`
                );
            }

            const status = await response.json();
            updateConsole(status);
        }
        catch (error) {
            setText(
                "operatorRuntime",
                "OFFLINE"
            );

            console.error(
                "Operator Console status error:",
                error
            );
        }
    }

    function initialize() {
        if (!buildLayout()) {
            return;
        }

        refreshOperatorConsole();

        window.setInterval(
            refreshOperatorConsole,
            REFRESH_INTERVAL_MS
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    }
    else {
        initialize();
    }
})();
