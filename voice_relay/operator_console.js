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
            ["Tracking mode", "operatorTrackingMode"],
            ["Locked identity", "operatorLockedIdentity"],
            ["Locked entity", "operatorLockedEntity"],
            ["Waiting time", "operatorWaitingTime"],
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

        const trackingMode = String(
            tracking.tracking_mode || "UNLOCKED"
        ).trim().toUpperCase();

        setText(
            "operatorTrackingMode",
            trackingMode.replaceAll("_", " ")
        );

        setText(
            "operatorLockedIdentity",
            tracking.locked_identity_id ||
                tracking.identity_id ||
                "None"
        );

        setText(
            "operatorLockedEntity",
            tracking.locked_entity_id ||
                tracking.entity_id ||
                "None"
        );

        const waitingAgeSeconds =
            tracking.waiting_age_seconds;

        setText(
            "operatorWaitingTime",
            waitingAgeSeconds === null ||
            waitingAgeSeconds === undefined
                ? (
                    trackingMode ===
                    "WAITING_FOR_IDENTITY"
                        ? "Starting…"
                        : "—"
                )
                : (
                    Number(
                        waitingAgeSeconds
                    ).toFixed(1) + " s"
                )
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

/* =========================================================
 * Operator Console v3 page navigation
 * ========================================================= */

(() => {
    function activateConsolePage(pageId, remember = true) {
        const pages = document.querySelectorAll(".console-page");
        const buttons = document.querySelectorAll("[data-console-page]");
        const target = document.getElementById(pageId);

        if (!target) {
            return;
        }

        pages.forEach((page) => {
            const selected = page === target;
            page.hidden = !selected;
            page.classList.toggle("active", selected);
        });

        buttons.forEach((button) => {
            const selected = button.dataset.consolePage === pageId;
            button.classList.toggle("active", selected);
            button.setAttribute("aria-selected", selected ? "true" : "false");
        });

        if (remember) {
            try {
                window.sessionStorage.setItem("miniPupperConsolePage", pageId);
            }
            catch (error) {
                // Navigation remains functional without session storage.
            }
        }

        window.scrollTo({top: 0, behavior: "smooth"});
    }

    function initializeConsoleNavigation() {
        const buttons = document.querySelectorAll("[data-console-page]");

        buttons.forEach((button) => {
            button.addEventListener("click", () => {
                activateConsolePage(button.dataset.consolePage);
            });
        });

        let initialPage = "missionPage";

        try {
            const savedPage = window.sessionStorage.getItem("miniPupperConsolePage");
            if (savedPage && document.getElementById(savedPage)) {
                initialPage = savedPage;
            }
        }
        catch (error) {
            initialPage = "missionPage";
        }

        activateConsolePage(initialPage, false);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeConsoleNavigation);
    }
    else {
        initializeConsoleNavigation();
    }
})();

/* =========================================================
 * Operator Console v3 configuration administration
 * ========================================================= */

(() => {
    const CONFIGURATION_URL = "/dashboard/config";

    const fieldMap = {
        configRobotName: ["robot", "name"],
        configRobotModel: ["robot", "model"],
        configRobotHostname: ["robot", "hostname"],
        configRobotIp: ["network", "robot_ip"],
        configRobotBridgePort: ["network", "robot_bridge_port"],
        configBrainIp: ["network", "brain_ip"],
        configRosDomain: ["network", "ros_domain"],
        configVisionServerUrl: ["vision", "server_url"],
        configSpeechProvider: ["speech", "provider"],
        configUiTheme: ["ui", "theme"],
        configCameraLayout: ["ui", "camera_layout"],
    };

    let loadedConfiguration = null;

    function element(id) {
        return document.getElementById(id);
    }

    function nestedValue(source, path) {
        return path.reduce((value, key) => {
            if (value && typeof value === "object") {
                return value[key];
            }
            return undefined;
        }, source);
    }

    function setStatus(text, state) {
        const pill = element("configurationStatusPill");
        if (!pill) {
            return;
        }
        pill.textContent = text;
        pill.classList.remove("ready", "planned", "error");
        pill.classList.add(state);
    }

    function showMessage(text, kind = "info") {
        const message = element("configurationMessage");
        if (!message) {
            return;
        }
        message.textContent = text;
        message.className = `configuration-message ${kind}`;
        message.hidden = false;
    }

    function clearMessage() {
        const message = element("configurationMessage");
        if (message) {
            message.hidden = true;
            message.textContent = "";
        }
    }

    function setBusy(busy) {
        const saveButton = element("saveConfigurationButton");
        const reloadButton = element("reloadConfigurationButton");
        if (saveButton) {
            saveButton.disabled = busy;
            saveButton.textContent = busy ? "Saving…" : "Save Configuration";
        }
        if (reloadButton) {
            reloadButton.disabled = busy;
        }
    }

    function populateForm(configuration) {
        Object.entries(fieldMap).forEach(([id, path]) => {
            const input = element(id);
            const value = nestedValue(configuration, path);
            if (input && value !== undefined && value !== null) {
                input.value = String(value);
            }
        });
    }

    function buildConfigurationFromForm() {
        return {
            config_version: Number(
                (loadedConfiguration && loadedConfiguration.config_version != null ? loadedConfiguration.config_version : 1)
            ),
            robot: {
                name: element("configRobotName").value.trim(),
                model: element("configRobotModel").value.trim(),
                hostname: element("configRobotHostname").value.trim(),
            },
            network: {
                robot_ip: element("configRobotIp").value.trim(),
                robot_bridge_port: Number(
                    element("configRobotBridgePort").value
                ),
                brain_ip: element("configBrainIp").value.trim(),
                ros_domain: Number(element("configRosDomain").value),
            },
            vision: {
                server_url: element("configVisionServerUrl").value.trim(),
            },
            speech: {
                provider: element("configSpeechProvider").value,
            },
            ui: {
                theme: element("configUiTheme").value,
                camera_layout: element("configCameraLayout").value,
            },
        };
    }

    async function parseResponse(response) {
        const raw = await response.text();
        let payload = {};
        if (raw) {
            try {
                payload = JSON.parse(raw);
            }
            catch (error) {
                throw new Error("The configuration service returned invalid JSON.");
            }
        }
        if (!response.ok || payload.ok === false) {
            throw new Error(
                payload.error || `Configuration request returned ${response.status}.`
            );
        }
        return payload;
    }

    async function loadConfiguration() {
        clearMessage();
        setStatus("Loading configuration", "planned");
        setBusy(true);

        try {
            const response = await fetch(CONFIGURATION_URL, {
                method: "GET",
                cache: "no-store",
            });
            const payload = await parseResponse(response);
            loadedConfiguration = payload.config;
            populateForm(loadedConfiguration);
            setStatus("Configuration ready", "ready");
        }
        catch (error) {
            setStatus("Configuration offline", "error");
            showMessage(error.message, "error");
            console.error("Configuration load error:", error);
        }
        finally {
            setBusy(false);
        }
    }

    async function saveConfiguration(event) {
        event.preventDefault();
        clearMessage();

        const form = element("configurationForm");
        if (!form.reportValidity()) {
            return;
        }

        setBusy(true);
        setStatus("Saving configuration", "planned");

        try {
            const configuration = buildConfigurationFromForm();
            const response = await fetch(CONFIGURATION_URL, {
                method: "PUT",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(configuration),
            });
            const payload = await parseResponse(response);
            loadedConfiguration = payload.config;
            populateForm(loadedConfiguration);
            setStatus("Configuration saved", "ready");
            showMessage(
                "Configuration saved. Restart affected services before relying on changed connection settings.",
                "success"
            );
        }
        catch (error) {
            setStatus("Save failed", "error");
            showMessage(error.message, "error");
            console.error("Configuration save error:", error);
        }
        finally {
            setBusy(false);
        }
    }

    function initializeConfigurationAdministration() {
        const form = element("configurationForm");
        const reloadButton = element("reloadConfigurationButton");
        if (!form || !reloadButton) {
            return;
        }
        form.addEventListener("submit", saveConfiguration);
        reloadButton.addEventListener("click", loadConfiguration);
        loadConfiguration();
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initializeConfigurationAdministration,
            {once: true}
        );
    }
    else {
        initializeConfigurationAdministration();
    }
})();


/* Operator Console v3 live diagnostics */
(() => {
    const DIAGNOSTICS_URL = "/dashboard/diagnostics";
    let timer = null;

    const el = id => document.getElementById(id);
    const escapeHtml = value => String(value != null ? value : "—").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[char]));
    const formatPercent = value => Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}%` : "—";
    const formatSeconds = value => {
        if (!Number.isFinite(Number(value))) return "—";
        const seconds = Math.max(0, Number(value));
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return days ? `${days}d ${hours}h` : hours ? `${hours}h ${minutes}m` : `${minutes}m`;
    };
    const value = (label, content) => `<div class="diagnostics-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(content)}</strong></div>`;

    function serviceState(name, service) {
        if (!service || !service.online) return "offline";
        if (name === "robot_bridge" && service.ros_ready === false) return "warning";
        if (name === "vision" && service.camera_running === false) return "warning";
        return "healthy";
    }

    function render(payload) {
        const pill = el("diagnosticsStatusPill");
        pill.textContent = payload.overall === "healthy" ? "Platform healthy" : "Platform degraded";
        pill.className = `console-status-pill ${payload.overall === "healthy" ? "ready" : "error"}`;
        el("diagnosticsMessage").hidden = true;

        const labels = {runtime:"Runtime", robot_bridge:"Robot Bridge", vision:"Vision", camera:"Camera"};
        el("serviceHealthGrid").innerHTML = Object.entries(payload.services || {}).map(([name, service]) => {
            const state = serviceState(name, service);
            const status = service.online ? (service.status || (state === "warning" ? "Warning" : "Online")) : "Offline";
            return `<article class="card diagnostics-service-card ${state}"><h3><span class="diagnostics-indicator"></span>${escapeHtml(labels[name] || name)}</h3><p>${escapeHtml(status)}</p></article>`;
        }).join("");

        const sys = payload.system || {};
        el("systemDiagnostics").innerHTML = [
            value("CPU", formatPercent(sys.cpu_percent)), value("Memory", formatPercent(sys.memory_percent)),
            value("Disk", formatPercent(sys.disk_percent)), value("Host uptime", formatSeconds(sys.uptime_seconds)),
            value("Hostname", sys.hostname), value("Python", sys.python_version),
            value("Git branch", sys.git_branch), value("Git commit", `${sys.git_commit || "—"}${sys.git_dirty ? " (modified)" : ""}`),
        ].join("");

        const run = payload.runtime || {};
        el("runtimeDiagnostics").innerHTML = [
            value("State", run.state), value("Loop rate", run.loop_hz ? `${run.loop_hz} Hz` : "—"),
            value("Runtime uptime", formatSeconds(run.uptime_seconds)), value("Active mission", (run.active_mission && run.active_mission.mission_type) || "None"),
            value("Mission queue", run.queue_length), value("Mission history", run.history_count),
            value("World entities", run.entity_count), value("Last error", run.last_error || "None"),
        ].join("");

        el("serviceDiagnostics").innerHTML = `<table class="diagnostics-table"><thead><tr><th>Service</th><th>Status</th><th>Latency</th><th>Details</th></tr></thead><tbody>${Object.entries(payload.services || {}).map(([name, service]) => `<tr><td>${escapeHtml(labels[name] || name)}</td><td>${escapeHtml(service.online ? "Online" : "Offline")}</td><td>${escapeHtml(Number.isFinite(Number(service.latency_ms)) ? `${service.latency_ms} ms` : "—")}</td><td>${escapeHtml(service.error || service.last_error || service.status || "Ready")}</td></tr>`).join("")}</tbody></table>`;
    }

    async function refresh() {
        try {
            const response = await fetch(DIAGNOSTICS_URL, {cache:"no-store"});
            const payload = await response.json();
            if (!response.ok || payload.ok === false) throw new Error(payload.error || `Diagnostics returned ${response.status}.`);
            render(payload);
        } catch (error) {
            const pill = el("diagnosticsStatusPill");
            if (pill) { pill.textContent = "Diagnostics offline"; pill.className = "console-status-pill error"; }
            const message = el("diagnosticsMessage");
            if (message) { message.textContent = error.message; message.hidden = false; }
        }
    }

    function initialize() {
        if (!el("diagnosticsPage")) return;
        var refreshDiagnosticsButton = el("refreshDiagnosticsButton");
        if (refreshDiagnosticsButton) refreshDiagnosticsButton.addEventListener("click", refresh);
        refresh();
        timer = window.setInterval(refresh, 3000);
        window.addEventListener("beforeunload", () => window.clearInterval(timer), {once:true});
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once:true}); else initialize();
})();


/* Operator Console v4 mission history */
(function () {
    var HISTORY_URL = "/dashboard/mission-history";
    var missions = [];
    var selectedMissionId = null;
    var timer = null;

    function byId(id) { return document.getElementById(id); }
    function text(value) { return value == null || value === "" ? "—" : String(value); }
    function escapeHtml(value) { return text(value).replace(/[&<>"']/g, function (character) { return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[character]; }); }
    function formatTimestamp(value) { if (!value) return "—"; var date = new Date(value); return isNaN(date.getTime()) ? value : date.toLocaleString(); }
    function duration(mission) { if (!mission.started_at || !mission.completed_at) return "—"; var start = new Date(mission.started_at); var end = new Date(mission.completed_at); var seconds = (end - start) / 1000; return isFinite(seconds) && seconds >= 0 ? seconds.toFixed(1) + " s" : "—"; }
    function missionLabel(mission) { return text(mission.mission_type) + (mission.target ? " — " + mission.target : ""); }

    function filteredMissions() {
        var search = (byId("missionHistorySearch").value || "").toLowerCase().trim();
        var status = byId("missionHistoryFilter").value;
        return missions.filter(function (mission) {
            if (status !== "ALL" && mission.status !== status) return false;
            if (!search) return true;
            return [mission.mission_type, mission.target, mission.speech, mission.status, mission.source, mission.mission_id].join(" ").toLowerCase().indexOf(search) !== -1;
        });
    }

    function renderDetail(mission) {
        var detail = byId("missionHistoryDetail");
        if (!mission) { detail.className = "mission-history-empty"; detail.textContent = "Select a mission to inspect its details."; return; }
        detail.className = "mission-history-detail";
        detail.innerHTML = '<div class="mission-history-speech">' + escapeHtml(mission.speech || "No speech recorded.") + '</div>' +
            '<div class="mission-history-detail-grid">' +
            field("Mission", missionLabel(mission)) + field("Status", mission.status) +
            field("Mission ID", mission.mission_id) + field("Source", mission.source) +
            field("Created", formatTimestamp(mission.created_at)) + field("Started", formatTimestamp(mission.started_at)) +
            field("Completed", formatTimestamp(mission.completed_at)) + field("Duration", duration(mission)) +
            field("Priority", mission.priority) + field("Target", mission.target || "None") +
            '</div>';
    }

    function field(label, value) { return '<div class="mission-history-detail-field"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>'; }

    function render() {
        var list = byId("missionHistoryList");
        var visible = filteredMissions();
        byId("missionHistorySummary").textContent = visible.length + " shown of " + missions.length + " recorded missions.";
        if (!visible.length) { list.innerHTML = '<div class="mission-history-empty">No missions match the current filters.</div>'; renderDetail(null); return; }
        if (!selectedMissionId || !visible.some(function (mission) { return mission.mission_id === selectedMissionId; })) selectedMissionId = visible[0].mission_id;
        list.innerHTML = visible.map(function (mission) {
            var selected = mission.mission_id === selectedMissionId ? " selected" : "";
            return '<button type="button" class="mission-history-item status-' + escapeHtml(String(mission.status || "unknown").toLowerCase()) + selected + '" data-mission-id="' + escapeHtml(mission.mission_id) + '">' +
                '<div class="mission-history-item-top"><span class="mission-history-item-title">' + escapeHtml(missionLabel(mission)) + '</span><span class="mission-history-badge">' + escapeHtml(mission.status) + '</span></div>' +
                '<div class="mission-history-item-meta">' + escapeHtml(formatTimestamp(mission.created_at)) + ' · ' + escapeHtml(duration(mission)) + '</div></button>';
        }).join("");
        Array.prototype.forEach.call(list.querySelectorAll("[data-mission-id]"), function (button) {
            button.addEventListener("click", function () { selectedMissionId = button.getAttribute("data-mission-id"); render(); });
        });
        renderDetail(visible.filter(function (mission) { return mission.mission_id === selectedMissionId; })[0]);
    }

    async function refresh() {
        var pill = byId("missionHistoryStatusPill");
        var message = byId("missionHistoryMessage");
        try {
            var response = await fetch(HISTORY_URL, {cache: "no-store"});
            var payload = await response.json();
            if (!response.ok || payload.ok === false) throw new Error(payload.error || "Mission history request failed.");
            missions = Array.isArray(payload.missions) ? payload.missions : [];
            pill.textContent = missions.length ? missions.length + " missions recorded" : "No missions recorded";
            pill.className = "console-status-pill ready";
            message.hidden = true;
            render();
        } catch (error) {
            pill.textContent = "History offline";
            pill.className = "console-status-pill error";
            message.textContent = error.message;
            message.hidden = false;
        }
    }

    function initialize() {
        if (!byId("historyPage")) return;
        byId("refreshMissionHistoryButton").addEventListener("click", refresh);
        byId("missionHistorySearch").addEventListener("input", render);
        byId("missionHistoryFilter").addEventListener("change", render);
        refresh();
        timer = window.setInterval(refresh, 5000);
        window.addEventListener("beforeunload", function () { window.clearInterval(timer); }, {once: true});
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true}); else initialize();
})();


/* Operator Console v5 World Model Explorer */
(function () {
    var WORLD_MODEL_URL = "/dashboard/world-model";
    var entities = [];
    var recentEvents = [];
    var robotState = {};
    var selectedEntityId = null;
    var timer = null;

    function byId(id) { return document.getElementById(id); }
    function text(value) { return value == null || value === "" ? "—" : String(value); }
    function escapeHtml(value) { return text(value).replace(/[&<>"']/g, function (character) { return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[character]; }); }
    function formatTimestamp(value) { if (!value) return "—"; var date = new Date(value); return isNaN(date.getTime()) ? value : date.toLocaleString(); }
    function ageSeconds(value) { if (!value) return Infinity; var date = new Date(value); return isNaN(date.getTime()) ? Infinity : Math.max(0, (Date.now() - date.getTime()) / 1000); }
    function formatAge(value) { var seconds = ageSeconds(value); if (!isFinite(seconds)) return "unknown"; if (seconds < 1) return "now"; if (seconds < 60) return Math.floor(seconds) + "s ago"; if (seconds < 3600) return Math.floor(seconds / 60) + "m ago"; if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago"; return Math.floor(seconds / 86400) + "d ago"; }
    function confidence(value) { var number = Number(value); return isFinite(number) ? (number * 100).toFixed(1) + "%" : "—"; }
    function jsonBlock(value) { return '<pre class="world-model-json">' + escapeHtml(JSON.stringify(value || {}, null, 2)) + '</pre>'; }
    function field(label, value) { return '<div class="world-model-detail-field"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>'; }

    function updateTypeFilter() {
        var select = byId("worldModelTypeFilter");
        var current = select.value;
        var types = {};
        entities.forEach(function (entity) { types[entity.entity_type || "unknown"] = true; });
        select.innerHTML = '<option value="ALL">All types</option>' + Object.keys(types).sort().map(function (type) { return '<option value="' + escapeHtml(type) + '">' + escapeHtml(type) + '</option>'; }).join("");
        select.value = types[current] ? current : "ALL";
    }

    function filteredEntities() {
        var search = (byId("worldModelSearch").value || "").toLowerCase().trim();
        var type = byId("worldModelTypeFilter").value;
        var age = byId("worldModelAgeFilter").value;
        return entities.filter(function (entity) {
            if (type !== "ALL" && (entity.entity_type || "unknown") !== type) return false;
            if (age !== "ALL" && ageSeconds(entity.last_seen) > Number(age)) return false;
            if (!search) return true;
            var observations = Array.isArray(entity.history) ? entity.history : [];
            var sources = observations.map(function (observation) { return observation.source; }).join(" ");
            return [entity.entity_id, entity.label, entity.entity_type, sources, JSON.stringify(entity.attributes || {})].join(" ").toLowerCase().indexOf(search) !== -1;
        });
    }

    function renderSummary() {
        var types = {};
        var fresh = 0;
        entities.forEach(function (entity) { types[entity.entity_type || "unknown"] = (types[entity.entity_type || "unknown"] || 0) + 1; if (ageSeconds(entity.last_seen) <= 5) fresh += 1; });
        var mission = robotState.mission || "None";
        byId("worldModelSummary").innerHTML = [
            ["Entities", entities.length], ["Fresh (5s)", fresh], ["Entity types", Object.keys(types).length], ["Robot mission", mission]
        ].map(function (item) { return '<article class="card world-model-summary-card"><span>' + escapeHtml(item[0]) + '</span><strong>' + escapeHtml(item[1]) + '</strong></article>'; }).join("");
    }

    function renderDetail(entity) {
        var detail = byId("worldModelDetail");
        if (!entity) { detail.className = "world-model-empty"; detail.textContent = "Select an entity to inspect its memory and observation history."; return; }
        var history = Array.isArray(entity.history) ? entity.history.slice().reverse() : [];
        detail.className = "world-model-detail";
        detail.innerHTML = '<div class="world-model-detail-title"><div><h3>' + escapeHtml(entity.label) + '</h3><p>' + escapeHtml(entity.entity_id) + '</p></div><span class="world-model-confidence">' + escapeHtml(confidence(entity.confidence)) + '</span></div>' +
            '<div class="world-model-detail-grid">' + field("Type", entity.entity_type) + field("Last seen", formatTimestamp(entity.last_seen)) + field("Age", formatAge(entity.last_seen)) + field("First seen", formatTimestamp(entity.first_seen)) + field("Observations", history.length) + field("Confidence", confidence(entity.confidence)) + '</div>' +
            '<section class="world-model-section"><h3>Attributes</h3>' + jsonBlock(entity.attributes) + '</section>' +
            '<section class="world-model-section"><h3>Observation History</h3><div class="world-model-observations">' + (history.length ? history.map(function (observation) { return '<article><div><strong>' + escapeHtml(observation.source || "unknown") + '</strong><span>' + escapeHtml(formatTimestamp(observation.timestamp)) + '</span></div><div class="world-model-observation-meta">Confidence ' + escapeHtml(confidence(observation.confidence)) + '</div>' + jsonBlock({location: observation.location, attributes: observation.attributes}) + '</article>'; }).join("") : '<div class="world-model-empty">No observation history.</div>') + '</div></section>';
    }

    function renderEvents() {
        var container = byId("worldModelEvents");
        var events = recentEvents.slice().reverse().slice(0, 20);
        container.innerHTML = events.length ? events.map(function (event) { return '<article class="world-model-event"><div><strong>' + escapeHtml(event.type) + '</strong><span>' + escapeHtml(formatTimestamp(event.timestamp)) + '</span></div>' + jsonBlock(event.data) + '</article>'; }).join("") : '<div class="world-model-empty">No recent events recorded.</div>';
    }

    function render() {
        var visible = filteredEntities();
        byId("worldModelCount").textContent = visible.length + " shown of " + entities.length + " persistent entities.";
        if (!selectedEntityId || !visible.some(function (entity) { return entity.entity_id === selectedEntityId; })) selectedEntityId = visible.length ? visible[0].entity_id : null;
        var list = byId("worldModelList");
        list.innerHTML = visible.length ? visible.map(function (entity) {
            var selected = entity.entity_id === selectedEntityId ? " selected" : "";
            var fresh = ageSeconds(entity.last_seen) <= 5 ? " fresh" : "";
            return '<button type="button" class="world-model-item' + selected + fresh + '" data-entity-id="' + escapeHtml(entity.entity_id) + '"><div class="world-model-item-top"><span><strong>' + escapeHtml(entity.label) + '</strong><small>' + escapeHtml(entity.entity_type || "unknown") + '</small></span><b>' + escapeHtml(confidence(entity.confidence)) + '</b></div><div class="world-model-item-meta"><span>' + escapeHtml(entity.entity_id) + '</span><span>' + escapeHtml(formatAge(entity.last_seen)) + '</span></div></button>';
        }).join("") : '<div class="world-model-empty">No entities match the current filters.</div>';
        Array.prototype.forEach.call(list.querySelectorAll("[data-entity-id]"), function (button) { button.addEventListener("click", function () { selectedEntityId = button.getAttribute("data-entity-id"); render(); }); });
        renderDetail(visible.filter(function (entity) { return entity.entity_id === selectedEntityId; })[0]);
        renderSummary();
        renderEvents();
    }

    async function refresh() {
        var pill = byId("worldModelStatusPill");
        var message = byId("worldModelMessage");
        try {
            var response = await fetch(WORLD_MODEL_URL, {cache: "no-store"});
            var payload = await response.json();
            if (!response.ok || payload.ok === false) throw new Error(payload.error || "World Model request failed.");
            entities = Array.isArray(payload.entities) ? payload.entities : [];
            recentEvents = Array.isArray(payload.recent_events) ? payload.recent_events : [];
            robotState = payload.robot_state || {};
            updateTypeFilter();
            pill.textContent = entities.length + " entities remembered";
            pill.className = "console-status-pill ready";
            message.hidden = true;
            render();
        } catch (error) {
            pill.textContent = "World Model offline";
            pill.className = "console-status-pill error";
            message.textContent = error.message;
            message.hidden = false;
        }
    }

    function initialize() {
        if (!byId("worldModelPage")) return;
        byId("refreshWorldModelButton").addEventListener("click", refresh);
        byId("worldModelSearch").addEventListener("input", render);
        byId("worldModelTypeFilter").addEventListener("change", render);
        byId("worldModelAgeFilter").addEventListener("change", render);
        refresh();
        timer = window.setInterval(refresh, 3000);
        window.addEventListener("beforeunload", function () { window.clearInterval(timer); }, {once: true});
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, {once: true}); else initialize();
})();

/* Operator Console v6 Network Manager */
(function () {
    var NETWORK_URL = "/dashboard/network-status";
    var CONNECT_URL = "/network/connect";
    var DISCONNECT_URL = "/network/disconnect";
    var FORGET_URL = "/network/forget";

    var refreshTimer = null;
    var operationInProgress = false;
    var lastPayload = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function text(value, fallback) {
        if (
            value === null ||
            typeof value === "undefined" ||
            value === ""
        ) {
            return fallback || "—";
        }

        return String(value);
    }

    function escapeNetworkHtml(value) {
        return text(value, "").replace(
            /[&<>"']/g,
            function (character) {
                return {
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    "\"": "&quot;",
                    "'": "&#39;"
                }[character];
            }
        );
    }

    function setMessage(message, kind) {
        var element = byId("networkMessage");

        if (!element) {
            return;
        }

        element.hidden = !message;
        element.textContent = message || "";

        element.className =
            "configuration-message " +
            (kind === "success" ? "success" : "error");
    }

    function setNetworkPill(label, state) {
        var pill = byId("networkStatusPill");

        if (!pill) {
            return;
        }

        pill.textContent = label;
        pill.className =
            "console-status-pill " + state;
    }

    function setBusy(busy, label) {
        operationInProgress = Boolean(busy);

        var scanButton = byId("scanWifiButton");
        var disconnectButton =
            byId("disconnectNetworkButton");

        if (scanButton) {
            scanButton.disabled = operationInProgress;
        }

        if (disconnectButton) {
            disconnectButton.disabled =
                operationInProgress ||
                !(
                    lastPayload &&
                    lastPayload.summary &&
                    lastPayload.summary.connected
                );
        }

        var actionButtons = document.querySelectorAll(
            "#networkPage [data-network-action]"
        );

        actionButtons.forEach(function (button) {
            button.disabled = operationInProgress;
        });

        if (operationInProgress) {
            setNetworkPill(
                label || "Network operation...",
                "planned"
            );
        }
    }

    function signalMarkup(signal) {
        var number = Number(signal);

        if (!Number.isFinite(number)) {
            return "—";
        }

        var bounded = Math.max(
            0,
            Math.min(100, number)
        );

        return (
            '<span class="network-signal">' +
                '<span class="network-signal-meter">' +
                    '<i style="width:' +
                        bounded +
                        '%"></i>' +
                '</span>' +
                '<strong>' +
                    bounded +
                    '%</strong>' +
            '</span>'
        );
    }

    function networkIsOpen(network) {
        var security = text(
            network.security,
            ""
        ).toLowerCase();

        return (
            !security ||
            security === "open" ||
            security === "none" ||
            security === "--"
        );
    }

    function renderWifi(networks) {
        var target = byId("wifiNetworksTable");

        if (!target) {
            return;
        }

        if (!networks.length) {
            target.innerHTML =
                '<div class="mission-history-empty">' +
                'No Wi-Fi networks were reported.' +
                '</div>';
            return;
        }

        target.innerHTML =
            '<table class="network-table">' +
                '<thead>' +
                    '<tr>' +
                        '<th>SSID</th>' +
                        '<th>Status</th>' +
                        '<th>Signal</th>' +
                        '<th>Security</th>' +
                        '<th>Channel</th>' +
                        '<th>Rate</th>' +
                        '<th>Action</th>' +
                    '</tr>' +
                '</thead>' +
                '<tbody>' +
                    networks.map(function (network) {
                        var ssid = text(
                            network.ssid,
                            ""
                        );

                        var action = network.in_use
                            ? (
                                '<button ' +
                                    'type="button" ' +
                                    'class="secondary-action" ' +
                                    'data-network-action="disconnect">' +
                                    'Disconnect' +
                                '</button>'
                            )
                            : (
                                '<button ' +
                                    'type="button" ' +
                                    'class="secondary-action" ' +
                                    'data-network-action="connect" ' +
                                    'data-network-ssid="' +
                                    escapeNetworkHtml(ssid) +
                                    '" ' +
                                    'data-network-open="' +
                                    (
                                        networkIsOpen(network)
                                            ? "true"
                                            : "false"
                                    ) +
                                    '">' +
                                    'Connect' +
                                '</button>'
                            );

                        return (
                            '<tr class="' +
                                (
                                    network.in_use
                                        ? "network-active-row"
                                        : ""
                                ) +
                                '">' +
                                '<td><strong>' +
                                    escapeNetworkHtml(ssid) +
                                '</strong></td>' +
                                '<td>' +
                                    (
                                        network.in_use
                                            ? "Connected"
                                            : "Available"
                                    ) +
                                '</td>' +
                                '<td>' +
                                    signalMarkup(network.signal) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        network.security || "Open"
                                    ) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        text(network.channel)
                                    ) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        text(network.rate)
                                    ) +
                                '</td>' +
                                '<td>' +
                                    action +
                                '</td>' +
                            '</tr>'
                        );
                    }).join("") +
                '</tbody>' +
            '</table>';
    }

    function renderDevices(devices) {
        var target = byId("networkDevicesTable");

        if (!target) {
            return;
        }

        if (!devices.length) {
            target.innerHTML =
                '<div class="mission-history-empty">' +
                'No network devices were reported.' +
                '</div>';
            return;
        }

        target.innerHTML =
            '<table class="network-table">' +
                '<thead>' +
                    '<tr>' +
                        '<th>Device</th>' +
                        '<th>Type</th>' +
                        '<th>State</th>' +
                        '<th>Connection</th>' +
                    '</tr>' +
                '</thead>' +
                '<tbody>' +
                    devices.map(function (device) {
                        return (
                            '<tr>' +
                                '<td><strong>' +
                                    escapeNetworkHtml(device.device) +
                                '</strong></td>' +
                                '<td>' +
                                    escapeNetworkHtml(device.type) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(device.state) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        text(device.connection)
                                    ) +
                                '</td>' +
                            '</tr>'
                        );
                    }).join("") +
                '</tbody>' +
            '</table>';
    }

    function renderSaved(connections) {
        var target = byId("savedConnectionsTable");

        if (!target) {
            return;
        }

        if (!connections.length) {
            target.innerHTML =
                '<div class="mission-history-empty">' +
                'No saved connections were reported.' +
                '</div>';
            return;
        }

        target.innerHTML =
            '<table class="network-table">' +
                '<thead>' +
                    '<tr>' +
                        '<th>Name</th>' +
                        '<th>Type</th>' +
                        '<th>Device</th>' +
                        '<th>Status</th>' +
                        '<th>Actions</th>' +
                    '</tr>' +
                '</thead>' +
                '<tbody>' +
                    connections.map(function (connection) {
                        var name = text(
                            connection.name,
                            ""
                        );

                        var connectAction =
                            connection.active
                                ? (
                                    '<button ' +
                                        'type="button" ' +
                                        'class="secondary-action" ' +
                                        'data-network-action=' +
                                        '"disconnect">' +
                                        'Disconnect' +
                                    '</button>'
                                )
                                : (
                                    '<button ' +
                                        'type="button" ' +
                                        'class="secondary-action" ' +
                                        'data-network-action="connect-saved" ' +
                                        'data-network-ssid="' +
                                        escapeNetworkHtml(name) +
                                        '">' +
                                        'Connect' +
                                    '</button>'
                                );

                        var forgetAction =
                            '<button ' +
                                'type="button" ' +
                                'class="secondary-action" ' +
                                'data-network-action="forget" ' +
                                'data-network-profile="' +
                                escapeNetworkHtml(name) +
                                '">' +
                                'Forget' +
                            '</button>';

                        return (
                            '<tr class="' +
                                (
                                    connection.active
                                        ? "network-active-row"
                                        : ""
                                ) +
                                '">' +
                                '<td><strong>' +
                                    escapeNetworkHtml(name) +
                                '</strong></td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        connection.type
                                    ) +
                                '</td>' +
                                '<td>' +
                                    escapeNetworkHtml(
                                        text(connection.device)
                                    ) +
                                '</td>' +
                                '<td>' +
                                    (
                                        connection.active
                                            ? "Active"
                                            : "Saved"
                                    ) +
                                '</td>' +
                                '<td>' +
                                    connectAction +
                                    " " +
                                    forgetAction +
                                '</td>' +
                            '</tr>'
                        );
                    }).join("") +
                '</tbody>' +
            '</table>';
    }

    function render(payload) {
        lastPayload = payload;

        var summary = payload.summary || {};

        byId(
            "networkConnectionSummary"
        ).textContent = summary.connected
            ? text(
                summary.active_connection,
                "Connected"
            )
            : "Disconnected";

        byId(
            "networkWifiSummary"
        ).textContent = text(
            summary.active_wifi_ssid,
            "Not connected"
        );

        byId(
            "networkSignalSummary"
        ).textContent =
            Number.isFinite(
                Number(summary.wifi_signal)
            )
                ? summary.wifi_signal + "%"
                : "—";

        byId(
            "networkHostnameSummary"
        ).textContent = text(payload.hostname);

        byId(
            "networkManagedSummary"
        ).textContent =
            text(
                summary.managed_device_count,
                "0"
            ) +
            " of " +
            text(
                summary.device_count,
                "0"
            );

        var collectedAt = payload.collected_at
            ? new Date(payload.collected_at)
            : null;

        byId(
            "networkUpdatedSummary"
        ).textContent =
            collectedAt &&
            !isNaN(collectedAt.getTime())
                ? collectedAt.toLocaleTimeString()
                : "—";

        renderWifi(payload.wifi_networks || []);
        renderDevices(payload.devices || []);
        renderSaved(payload.saved_connections || []);

        setNetworkPill(
            summary.connected
                ? "Network online"
                : "Network disconnected",
            summary.connected
                ? "success"
                : "error"
        );

        var disconnectButton =
            byId("disconnectNetworkButton");

        if (disconnectButton) {
            disconnectButton.disabled =
                operationInProgress ||
                !summary.connected;
        }

        if (payload.backend === "windows_wsl") {
            setMessage(
                "Windows Wi-Fi is being managed through " +
                "the WSL cross-platform backend.",
                "success"
            );
        }
        else if (
            !summary.networkmanager_managing_interfaces
        ) {
            setMessage(
                "NetworkManager is installed, but it is " +
                "not managing any reported interface. " +
                "Wi-Fi scans and saved profiles may be " +
                "unavailable until the host network " +
                "renderer uses NetworkManager.",
                "error"
            );
        }
        else if (!summary.wifi_device_count) {
            setMessage(
                "No Wi-Fi adapter was reported by the " +
                "active network backend. Ethernet " +
                "information remains available.",
                "error"
            );
        }
        else {
            setMessage("");
        }
    }

    function parseJsonResponse(response) {
        return response.json()
            .catch(function () {
                return {
                    ok: false,
                    error:
                        "The server returned an invalid " +
                        "JSON response."
                };
            })
            .then(function (payload) {
                return {
                    response: response,
                    payload: payload
                };
            });
    }

    function postJson(url, payload) {
        return fetch(
            url,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify(payload)
            }
        )
            .then(parseJsonResponse)
            .then(function (result) {
                if (
                    !result.response.ok ||
                    result.payload.ok === false
                ) {
                    throw new Error(
                        result.payload.error ||
                        "Network operation failed."
                    );
                }

                return result.payload;
            });
    }

    function refresh(rescan) {
        var scanButton = byId("scanWifiButton");

        if (scanButton && rescan) {
            scanButton.disabled = true;
            scanButton.textContent = "Scanning...";
        }

        return fetch(
            NETWORK_URL +
                (rescan ? "?rescan=true" : ""),
            {
                cache: "no-store"
            }
        )
            .then(parseJsonResponse)
            .then(function (result) {
                if (
                    !result.response.ok ||
                    result.payload.ok === false
                ) {
                    throw new Error(
                        result.payload.error ||
                        "Network request failed."
                    );
                }

                render(result.payload);

                return result.payload;
            })
            .catch(function (error) {
                setMessage(
                    error.message || String(error),
                    "error"
                );

                setNetworkPill(
                    "Network unavailable",
                    "error"
                );

                throw error;
            })
            .finally(function () {
                if (scanButton) {
                    scanButton.disabled =
                        operationInProgress;

                    scanButton.textContent =
                        "Scan Again";
                }
            });
    }

    function connectNetwork(ssid, requestPassword) {
        if (!ssid) {
            setMessage(
                "The selected Wi-Fi network has no SSID.",
                "error"
            );
            return;
        }

        var password;

        if (requestPassword) {
            password = window.prompt(
                "Enter the Wi-Fi password for:\n\n" +
                ssid +
                "\n\nThe password will not be displayed " +
                "or stored by the dashboard."
            );

            if (password === null) {
                return;
            }

            if (!password) {
                setMessage(
                    "A password is required for " +
                    ssid +
                    ".",
                    "error"
                );
                return;
            }
        }

        var payload = {
            ssid: ssid
        };

        if (typeof password === "string") {
            payload.password = password;
        }

        setBusy(
            true,
            "Connecting..."
        );

        setMessage(
            "Connecting to " + ssid + "...",
            "success"
        );

        postJson(
            CONNECT_URL,
            payload
        )
            .then(function () {
                return refresh(true);
            })
            .then(function () {
                setMessage(
                    "Connected to " + ssid + ".",
                    "success"
                );
            })
            .catch(function (error) {
                setMessage(
                    error.message || String(error),
                    "error"
                );
            })
            .finally(function () {
                setBusy(false);
            });
    }

    function disconnectNetwork() {
        var confirmed = window.confirm(
            "Disconnect the Brain PC from its " +
            "current network connection?\n\n" +
            "The Operator Console may become " +
            "temporarily unavailable."
        );

        if (!confirmed) {
            return;
        }

        setBusy(
            true,
            "Disconnecting..."
        );

        setMessage(
            "Disconnecting the active network...",
            "success"
        );

        postJson(
            DISCONNECT_URL,
            {}
        )
            .then(function () {
                return refresh(false);
            })
            .then(function () {
                setMessage(
                    "The active network was disconnected.",
                    "success"
                );
            })
            .catch(function (error) {
                setMessage(
                    error.message || String(error),
                    "error"
                );
            })
            .finally(function () {
                setBusy(false);
            });
    }

    function forgetNetwork(profile) {
        if (!profile) {
            setMessage(
                "The selected saved connection has no " +
                "profile name.",
                "error"
            );
            return;
        }

        var confirmed = window.confirm(
            "Forget the saved network profile:\n\n" +
            profile +
            "\n\nThe stored Wi-Fi credentials will be " +
            "removed from the Brain PC."
        );

        if (!confirmed) {
            return;
        }

        setBusy(
            true,
            "Forgetting profile..."
        );

        setMessage(
            "Removing saved profile " +
            profile +
            "...",
            "success"
        );

        postJson(
            FORGET_URL,
            {
                profile: profile
            }
        )
            .then(function () {
                return refresh(true);
            })
            .then(function () {
                setMessage(
                    "Forgot saved profile " +
                    profile +
                    ".",
                    "success"
                );
            })
            .catch(function (error) {
                setMessage(
                    error.message || String(error),
                    "error"
                );
            })
            .finally(function () {
                setBusy(false);
            });
    }

    function handleNetworkAction(event) {
        var button = event.target.closest(
            "[data-network-action]"
        );

        if (
            !button ||
            operationInProgress
        ) {
            return;
        }

        var action = button.getAttribute(
            "data-network-action"
        );

        if (action === "connect") {
            connectNetwork(
                button.getAttribute(
                    "data-network-ssid"
                ),
                button.getAttribute(
                    "data-network-open"
                ) !== "true"
            );
        }
        else if (action === "connect-saved") {
            connectNetwork(
                button.getAttribute(
                    "data-network-ssid"
                ),
                false
            );
        }
        else if (action === "disconnect") {
            disconnectNetwork();
        }
        else if (action === "forget") {
            forgetNetwork(
                button.getAttribute(
                    "data-network-profile"
                )
            );
        }
    }

    function initialize() {
        var page = byId("networkPage");

        if (!page) {
            return;
        }

        var scanButton = byId("scanWifiButton");
        var disconnectButton =
            byId("disconnectNetworkButton");

        if (scanButton) {
            scanButton.addEventListener(
                "click",
                function () {
                    if (!operationInProgress) {
                        refresh(true).catch(
                            function () {}
                        );
                    }
                }
            );
        }

        if (disconnectButton) {
            disconnectButton.addEventListener(
                "click",
                function () {
                    if (!operationInProgress) {
                        disconnectNetwork();
                    }
                }
            );
        }

        page.addEventListener(
            "click",
            handleNetworkAction
        );

        refresh(false).catch(
            function () {}
        );

        if (refreshTimer) {
            clearInterval(refreshTimer);
        }

        refreshTimer = setInterval(
            function () {
                if (!operationInProgress) {
                    refresh(false).catch(
                        function () {}
                    );
                }
            },
            10000
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize
        );
    }
    else {
        initialize();
    }
})();

/* Live read-only LD06 visualization */
(function () {
    "use strict";

    const ENDPOINT = "/dashboard/lidar";
    const REFRESH_MS = 250;
    const DISPLAY_RANGE_METERS = 4.0;

    /*
     * The physical lidar_link axes differ from the operator view.
     * Rotate scan points by 90 degrees so physical forward is up,
     * then mirror the horizontal display axis so Mayday's physical
     * right side appears on the canvas right. This is presentation-only
     * and does not modify ROS sensor data.
     */
    const SCAN_DISPLAY_ROTATION_RADIANS = Math.PI / 2;

    let timer = null;
    let requestInFlight = false;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");
        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setText(id, value) {
        const element = byId(id);
        if (element) element.textContent = value;
    }

    function setOffline(message) {
        const pill = byId("lidarStatusPill");
        const canvasMessage = byId("lidarCanvasMessage");
        const error = byId("lidarError");

        if (pill) {
            pill.textContent = "LiDAR Offline";
            pill.className = "console-status-pill error";
        }

        setText("lidarConnection", "Offline");
        setText("lidarFreshness", "No live scan");

        if (canvasMessage) {
            canvasMessage.textContent = message;
            canvasMessage.hidden = false;
        }

        if (error) {
            error.textContent = message;
            error.hidden = false;
        }
    }

    function resizeCanvas(canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(320, canvas.clientWidth);
        const height = Math.max(420, canvas.clientHeight);
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        return {
            context,
            width,
            height,
        };
    }

    function drawGrid(context, width, height, scale) {
        const centerX = width / 2;
        const centerY = height / 2;

        context.fillStyle = "#020617";
        context.fillRect(0, 0, width, height);

        context.strokeStyle = "rgba(148, 163, 184, 0.24)";
        context.lineWidth = 1;

        for (
            let meters = 1;
            meters <= DISPLAY_RANGE_METERS;
            meters += 1
        ) {
            context.beginPath();
            context.arc(
                centerX,
                centerY,
                meters * scale,
                0,
                Math.PI * 2
            );
            context.stroke();

            context.fillStyle = "#94a3b8";
            context.font = "12px system-ui";
            context.fillText(
                `${meters} m`,
                centerX + 5,
                centerY - meters * scale + 15
            );
        }

        context.beginPath();
        context.moveTo(centerX, 0);
        context.lineTo(centerX, height);
        context.moveTo(0, centerY);
        context.lineTo(width, centerY);
        context.stroke();

        context.fillStyle = "#94a3b8";
        context.font = "700 13px system-ui";
        context.textAlign = "center";
        context.fillText("FORWARD", centerX, 20);
        context.textAlign = "start";
    }

    function drawRobot(context, centerX, centerY) {
        context.save();
        context.translate(centerX, centerY);

        context.fillStyle = "#f59e0b";
        context.strokeStyle = "#fbbf24";
        context.lineWidth = 2;

        context.beginPath();
        context.moveTo(0, -20);
        context.lineTo(14, 13);
        context.lineTo(0, 8);
        context.lineTo(-14, 13);
        context.closePath();
        context.fill();
        context.stroke();

        context.restore();
    }

    function drawScan(scan) {
        const canvas = byId("lidarCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);
        const context = drawing.context;
        const width = drawing.width;
        const height = drawing.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.max(
            80,
            Math.min(width, height) / 2 - 34
        );
        const scale = radius / DISPLAY_RANGE_METERS;

        drawGrid(context, width, height, scale);

        context.fillStyle = "#38bdf8";
        context.shadowColor = "rgba(56, 189, 248, 0.65)";
        context.shadowBlur = 3;

        let angle = (
            Number(scan.angle_min)
            + SCAN_DISPLAY_ROTATION_RADIANS
        );
        const increment = Number(scan.angle_increment);
        const minimum = Number(scan.range_min);
        const maximum = Math.min(
            Number(scan.range_max),
            DISPLAY_RANGE_METERS
        );

        scan.ranges.forEach(function (rawRange) {
            if (rawRange !== null) {
                const range = Number(rawRange);

                if (
                    Number.isFinite(range)
                    && range >= minimum
                    && range <= maximum
                ) {
                    const x = (
                        centerX
                        - Math.sin(angle) * range * scale
                    );
                    const y = (
                        centerY
                        - Math.cos(angle) * range * scale
                    );

                    context.beginPath();
                    context.arc(x, y, 2.2, 0, Math.PI * 2);
                    context.fill();
                }
            }

            angle += increment;
        });

        context.shadowBlur = 0;
        drawRobot(context, centerX, centerY);
    }

    function render(payload) {
        const telemetry = payload.telemetry;
        const scan = telemetry.scan;
        const pill = byId("lidarStatusPill");
        const message = byId("lidarCanvasMessage");
        const error = byId("lidarError");
        const age = Number(telemetry.age_seconds);

        drawScan(scan);

        if (pill) {
            pill.textContent = "LiDAR Live";
            pill.className = "console-status-pill ready";
        }

        if (message) message.hidden = true;
        if (error) error.hidden = true;

        setText("lidarConnection", "Connected");
        setText("lidarFrame", scan.frame_id || "—");
        setText(
            "lidarSamples",
            Number(scan.sample_count).toLocaleString()
        );
        setText(
            "lidarValidSamples",
            Number(scan.valid_sample_count).toLocaleString()
        );
        setText("lidarAge", `${age.toFixed(3)} s`);
        setText(
            "lidarFreshness",
            `Updated ${age.toFixed(2)} seconds ago`
        );
        setText(
            "lidarScanTime",
            `${Number(scan.scan_time).toFixed(3)} s`
        );
        setText(
            "lidarRange",
            `${Number(scan.range_min).toFixed(2)}–`
            + `${Number(scan.range_max).toFixed(1)} m`
        );
    }

    async function refresh() {
        if (!perceptionIsVisible() || requestInFlight) return;

        requestInFlight = true;

        try {
            const response = await fetch(ENDPOINT, {
                cache: "no-store",
            });
            const payload = await response.json();

            if (
                !response.ok
                || !payload.ok
                || !payload.telemetry
                || !payload.telemetry.available
                || !payload.telemetry.scan
            ) {
                throw new Error(
                    payload.error
                    || "Mayday LiDAR telemetry is unavailable."
                );
            }

            render(payload);
        } catch (error) {
            setOffline(error.message);
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("lidarCanvas")) return;

        const canvas = byId("lidarCanvas");
        const context = canvas.getContext("2d");

        context.fillStyle = "#020617";
        context.fillRect(0, 0, canvas.width, canvas.height);

        refresh();
        timer = window.setInterval(refresh, REFRESH_MS);

        window.addEventListener(
            "resize",
            function () {
                if (perceptionIsVisible()) refresh();
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Minimal guarded localization buttons */
(function () {
    "use strict";

    const STATUS =
        "/dashboard/localization-control";
    const START =
        "/dashboard/localization-start";
    const STOP =
        "/dashboard/localization-stop";

    let busy = false;

    function byId(id) {
        return document.getElementById(id);
    }

    function setButtons(startDisabled, stopDisabled) {
        byId("startLocalizationButton").disabled =
            startDisabled;
        byId("stopLocalizationButton").disabled =
            stopDisabled;
    }

    function setDisplay(label, state, message, error) {
        const status = byId(
            "localizationControlState"
        );
        const text = byId(
            "localizationControlMessage"
        );

        status.textContent = label;
        status.className =
            "localization-control-state " + state;

        text.textContent = message;
        text.className = (
            "localization-control-message"
            + (error ? " error" : "")
        );
    }

    function validate(payload) {
        if (
            !payload
            || payload.ok !== true
            || !payload.localization
        ) {
            throw new Error(
                payload && (
                    payload.error || payload.message
                )
                || "Localization control failed."
            );
        }

        const control = payload.localization;

        if (
            control.planning_enabled !== false
            || control.control_enabled !== false
        ) {
            throw new Error(
                "Unsafe localization state rejected."
            );
        }

        return control;
    }

    function render(payload) {
        const control = validate(payload);

        if (control.running && !control.owned) {
            setDisplay(
                "External",
                "error",
                "Localization is externally managed.",
                true
            );
            setButtons(true, true);
            return;
        }

        if (control.running) {
            setDisplay(
                "Running",
                "running",
                "Localization is running. "
                + "Waiting for a fresh pose.",
                false
            );
            setButtons(true, false);
            return;
        }

        setDisplay(
            "Stopped",
            "stopped",
            "Localization is stopped.",
            false
        );
        setButtons(false, true);
    }

    async function read(response) {
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(
                payload.error
                || payload.message
                || `HTTP ${response.status}`
            );
        }

        return payload;
    }

    async function refresh() {
        if (busy) return;

        try {
            const response = await fetch(
                STATUS,
                {cache: "no-store"}
            );
            render(await read(response));
        } catch (error) {
            setDisplay(
                "Unavailable",
                "error",
                error.message,
                true
            );
            setButtons(true, true);
        }
    }

    async function act(endpoint, label) {
        if (busy) return;

        busy = true;
        setButtons(true, true);
        setDisplay(
            label,
            "busy",
            `${label} guarded localization...`,
            false
        );

        try {
            const response = await fetch(
                endpoint,
                {
                    method: "POST",
                    cache: "no-store",
                }
            );
            render(await read(response));
        } catch (error) {
            setDisplay(
                "Error",
                "error",
                error.message,
                true
            );
            setButtons(true, true);
        } finally {
            busy = false;
            window.setTimeout(refresh, 250);
        }
    }

    function initialize() {
        const start = byId(
            "startLocalizationButton"
        );
        const stop = byId(
            "stopLocalizationButton"
        );

        if (!start || !stop) return;

        start.addEventListener(
            "click",
            function () {
                act(START, "Starting");
            }
        );

        stop.addEventListener(
            "click",
            function () {
                act(STOP, "Stopping");
            }
        );

        refresh();
        window.setInterval(refresh, 1000);
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Guarded supervised mapping controls */
(function () {
    "use strict";

    const STATUS_ENDPOINT =
        "/dashboard/mapping-control";
    const START_ENDPOINT =
        "/dashboard/mapping-start";
    const STOP_ENDPOINT =
        "/dashboard/mapping-stop";
    const SAVE_ENDPOINT =
        "/dashboard/mapping-save-candidate";
    const REFRESH_MS = 1000;

    let timer = null;
    let requestInFlight = false;
    let actionInFlight = false;
    let latestReadinessReady = false;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setState(label, state) {
        const indicator = byId("mappingControlState");

        if (!indicator) return;

        indicator.textContent = label;
        indicator.className =
            "mapping-control-state " + state;
    }

    function setMessage(message, isError) {
        const element = byId("mappingControlMessage");

        if (!element) return;

        element.textContent = message;
        element.className = (
            "mapping-control-message"
            + (isError ? " error" : "")
        );
    }

    function updateButtons(running) {
        const start = byId("startMappingButton");
        const stop = byId("stopMappingButton");
        const save = byId("saveCandidateButton");

        if (start) {
            start.disabled = actionInFlight || running;
        }

        if (stop) {
            stop.disabled = actionInFlight || !running;
        }

        if (save) {
            save.disabled = actionInFlight || !running;

            if (
                !save.disabled
                && !latestReadinessReady
            ) {
                save.disabled = true;
            }

            save.title = (
                latestReadinessReady
                ? "Mapping is mature and ready to save."
                : "Wait for the live readiness requirements."
            );
        }
    }

    function clampProgress(value) {
        const number = Number(value);

        if (!Number.isFinite(number)) return 0;

        return Math.max(0, Math.min(1, number));
    }

    function setProgress(trackId, barId, progress) {
        const track = byId(trackId);
        const bar = byId(barId);
        const percentage = Math.round(
            clampProgress(progress) * 100
        );

        if (track) {
            track.setAttribute(
                "aria-valuenow",
                String(percentage)
            );
        }

        if (bar) {
            bar.style.width = `${percentage}%`;
        }
    }

    function fallbackReadiness(mapping, running) {
        return {
            available: false,
            status: (
                running
                ? "WAITING_FOR_SUBMAPS"
                : "MAPPING_STOPPED"
            ),
            ready: false,
            submap_count: 0,
            mature_submap_count: 0,
            minimum_submap_count:
                mapping.candidate_minimum_submaps,
            minimum_mature_submap_count:
                mapping.candidate_minimum_mature_submaps,
            minimum_mature_version:
                mapping.candidate_minimum_mature_version,
            submap_progress: 0,
            mature_submap_progress: 0,
            submaps: [],
        };
    }

    function renderReadiness(readiness, running) {
        const workspace = byId(
            "mappingReadinessLive"
        );
        const status = byId(
            "mappingReadinessStatus"
        );
        const submapText = byId(
            "mappingSubmapProgressText"
        );
        const matureText = byId(
            "mappingMatureProgressText"
        );
        const versions = byId(
            "mappingSubmapVersions"
        );

        const submapCount = Number(
            readiness.submap_count || 0
        );
        const matureCount = Number(
            readiness.mature_submap_count || 0
        );
        const requiredSubmaps = Number(
            readiness.minimum_submap_count || 0
        );
        const requiredMature = Number(
            readiness.minimum_mature_submap_count || 0
        );
        const requiredVersion = Number(
            readiness.minimum_mature_version || 0
        );

        let label = "Waiting";
        let state = "waiting";

        if (!running) {
            label = "Mapping stopped";
            state = "stopped";
        } else if (
            readiness.status === "READY_TO_SAVE"
            && readiness.ready === true
        ) {
            label = "Ready to save";
            state = "ready";
        } else if (
            readiness.status === "BUILDING_SUBMAPS"
        ) {
            label = "Building map";
            state = "building";
        } else if (
            readiness.status === "INVALID_SUBMAP_LIST"
        ) {
            label = "Readiness error";
            state = "error";
        } else {
            label = "Waiting for submaps";
            state = "waiting";
        }

        if (workspace) {
            workspace.className = (
                "mapping-readiness-live " + state
            );
        }

        if (status) {
            status.textContent = label;
            status.className = (
                "mapping-readiness-live-status "
                + state
            );
        }

        if (submapText) {
            submapText.textContent = (
                `${submapCount} / ${requiredSubmaps}`
            );
        }

        if (matureText) {
            matureText.textContent = (
                `${matureCount} / ${requiredMature}`
                + ` at v${requiredVersion}`
            );
        }

        setProgress(
            "mappingSubmapProgress",
            "mappingSubmapProgressBar",
            readiness.submap_progress
        );
        setProgress(
            "mappingMatureProgress",
            "mappingMatureProgressBar",
            readiness.mature_submap_progress
        );

        const submaps = (
            Array.isArray(readiness.submaps)
            ? readiness.submaps
            : []
        );

        if (versions) {
            versions.textContent = (
                submaps.length
                ? submaps.map(
                    (submap) => (
                        `#${submap.index} v${submap.version}`
                    )
                ).join(" · ")
                : "No live submaps"
            );
        }
    }

    function renderStatus(payload) {
        const mapping = payload.mapping;

        if (!mapping || mapping.planning_enabled !== false) {
            throw new Error(
                "Mapping safety state is unavailable."
            );
        }

        if (mapping.validated_map_mutable !== false) {
            throw new Error(
                "Validated-map protection is unavailable."
            );
        }

        const running = (
            mapping.running === true
            && mapping.owned === true
            && mapping.state === "RUNNING"
        );
        const readiness = (
            mapping.readiness
            || fallbackReadiness(mapping, running)
        );

        latestReadinessReady = (
            running
            && readiness.ready === true
            && readiness.status === "READY_TO_SAVE"
        );

        updateButtons(running);
        renderReadiness(readiness, running);
        setTextThreshold(mapping);

        if (running && latestReadinessReady) {
            setState("Ready", "running");
            setMessage(
                "Mapping maturity requirements are met. "
                + "Stop Mayday, then save a review candidate.",
                false
            );
        } else if (running) {
            setState("Running", "running");
            setMessage(
                "Headless mapping is active. Continue the "
                + "supervised route until readiness is complete.",
                false
            );
        } else {
            setState("Stopped", "stopped");
            setMessage(
                "Mapping is stopped. The validated map is unchanged.",
                false
            );
        }
    }

    function setTextThreshold(mapping) {
        const element = byId(
            "mappingReadinessThreshold"
        );

        if (!element) return;

        element.textContent = (
            `${mapping.candidate_minimum_submaps} submaps, `
            + `${mapping.candidate_minimum_mature_submaps} mature, `
            + `version ${mapping.candidate_minimum_mature_version}`
        );
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch (error) {
            return {
                ok: false,
                error: "Mapping service returned invalid data.",
            };
        }
    }

    async function refresh() {
        if (
            !perceptionIsVisible()
            || requestInFlight
            || actionInFlight
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const response = await fetch(
                STATUS_ENDPOINT,
                {cache: "no-store"}
            );
            const payload = await readJson(response);

            if (!response.ok || !payload.ok) {
                throw new Error(
                    payload.error
                    || "Mapping status is unavailable."
                );
            }

            renderStatus(payload);
        } catch (error) {
            latestReadinessReady = false;
            setState("Unavailable", "error");
            setMessage(error.message, true);
            updateButtons(false);
        } finally {
            requestInFlight = false;
        }
    }

    async function performAction(
        endpoint,
        busyLabel,
        busyMessage
    ) {
        if (actionInFlight) return;

        actionInFlight = true;
        setState(busyLabel, "busy");
        setMessage(busyMessage, false);
        updateButtons(false);

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                cache: "no-store",
                headers: {
                    "Content-Type": "application/json",
                },
            });
            const payload = await readJson(response);

            if (!response.ok || !payload.ok) {
                throw new Error(
                    payload.error
                    || payload.message
                    || "Mapping action failed."
                );
            }

            if (payload.mapping) {
                renderStatus(payload);
            }

            if (
                endpoint === SAVE_ENDPOINT
                && payload.candidate
            ) {
                setState("Candidate Saved", "stopped");
                setMessage(
                    "Review candidate saved. Mapping stopped; "
                    + "the validated map remains unchanged.",
                    false
                );

            }
        } catch (error) {
            setState("Action Failed", "error");
            setMessage(error.message, true);
        } finally {
            actionInFlight = false;
            await refresh();
        }
    }

    function startMapping() {
        performAction(
            START_ENDPOINT,
            "Starting...",
            "Starting guarded headless mapping..."
        );
    }

    function stopMapping() {
        if (
            !window.confirm(
                "Stop mapping without saving a candidate?"
            )
        ) {
            return;
        }

        performAction(
            STOP_ENDPOINT,
            "Stopping...",
            "Stopping mapping without saving..."
        );
    }

    function saveCandidate() {
        if (!latestReadinessReady) {
            setMessage(
                "Mapping is still building. Wait until live "
                + "readiness reports Ready to save.",
                true
            );
            return;
        }

        if (
            !window.confirm(
                "Stop mapping and save a review-only candidate? "
                + "This will not replace the validated map."
            )
        ) {
            return;
        }

        performAction(
            SAVE_ENDPOINT,
            "Saving...",
            "Checking submap maturity and saving candidate..."
        );
    }

    function initialize() {
        const start = byId("startMappingButton");
        const stop = byId("stopMappingButton");
        const save = byId("saveCandidateButton");

        if (!start || !stop || !save) return;

        start.addEventListener("click", startMapping);
        stop.addEventListener("click", stopMapping);
        save.addEventListener("click", saveCandidate);

        refresh();
        timer = window.setInterval(
            refresh,
            REFRESH_MS
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Read-only live Cartographer mapping map */
(function () {
    "use strict";

    const ENDPOINT = "/dashboard/mapping-map";
    const REFRESH_MS = 1000;

    let timer = null;
    let requestInFlight = false;
    let liveMap = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setText(id, value) {
        const element = byId(id);
        if (element) element.textContent = value;
    }

    function setStatus(label, state) {
        const status = byId("liveMappingStatus");

        if (!status) return;

        status.textContent = label;
        status.className =
            "live-mapping-status " + state;
    }

    function resizeCanvas(canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(320, canvas.clientWidth);
        const height = Math.max(420, canvas.clientHeight);
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        return {
            context,
            width,
            height,
        };
    }

    function clearCanvas() {
        const canvas = byId("liveMappingCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);

        drawing.context.fillStyle = "#020617";
        drawing.context.fillRect(
            0,
            0,
            drawing.width,
            drawing.height
        );
    }

    function createMapImage(occupancyMap) {
        const width = Number(occupancyMap.width);
        const height = Number(occupancyMap.height);
        const imageCanvas =
            document.createElement("canvas");

        imageCanvas.width = width;
        imageCanvas.height = height;

        const context = imageCanvas.getContext("2d");
        const image = context.createImageData(
            width,
            height
        );

        occupancyMap.cells.forEach(
            function (rawValue, index) {
                const value = Number(rawValue);
                const mapX = index % width;
                const mapY = Math.floor(index / width);
                const canvasY = height - 1 - mapY;
                const pixelIndex = (
                    (canvasY * width + mapX) * 4
                );

                let red = 15;
                let green = 23;
                let blue = 42;

                if (value >= 0 && value <= 100) {
                    const shade = Math.round(
                        248 - value * 2.2
                    );

                    red = shade;
                    green = shade;
                    blue = shade;
                }

                image.data[pixelIndex] = red;
                image.data[pixelIndex + 1] = green;
                image.data[pixelIndex + 2] = blue;
                image.data[pixelIndex + 3] = 255;
            }
        );

        context.putImageData(image, 0, 0);
        return imageCanvas;
    }

    function drawMap(occupancyMap) {
        const canvas = byId("liveMappingCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);
        const context = drawing.context;
        const padding = 32;
        const sourceWidth = Number(occupancyMap.width);
        const sourceHeight = Number(occupancyMap.height);
        const scale = Math.min(
            (
                drawing.width - padding * 2
            ) / sourceWidth,
            (
                drawing.height - padding * 2
            ) / sourceHeight
        );
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const drawX = (
            drawing.width - drawWidth
        ) / 2;
        const drawY = (
            drawing.height - drawHeight
        ) / 2;
        const image = createMapImage(occupancyMap);

        context.fillStyle = "#020617";
        context.fillRect(
            0,
            0,
            drawing.width,
            drawing.height
        );

        context.imageSmoothingEnabled = false;
        context.drawImage(
            image,
            drawX,
            drawY,
            drawWidth,
            drawHeight
        );

        context.strokeStyle = "#38bdf8";
        context.lineWidth = 1;
        context.strokeRect(
            drawX - 0.5,
            drawY - 0.5,
            drawWidth + 1,
            drawHeight + 1
        );

        context.fillStyle = "#7dd3fc";
        context.font = "700 13px system-ui";
        context.textAlign = "center";
        context.fillText(
            "LIVE MAP +Y",
            drawing.width / 2,
            Math.max(18, drawY - 10)
        );
        context.textAlign = "start";
    }

    function resetMetrics() {
        setText("liveMappingFrame", "—");
        setText("liveMappingDimensions", "—");
        setText("liveMappingResolution", "—");
        setText("liveMappingOrigin", "—");
        setText("liveMappingUnknown", "—");
        setText("liveMappingFree", "—");
        setText("liveMappingProbability", "—");
        setText("liveMappingOccupied", "—");
        setText("liveMappingTotal", "—");
        setText("liveMappingAge", "—");
    }

    function setMessage(message) {
        const element = byId("liveMappingMessage");

        if (!element) return;

        element.textContent = message;
        element.hidden = false;
    }

    function setStopped() {
        liveMap = null;
        clearCanvas();
        resetMetrics();
        setStatus("Mapping stopped", "stopped");
        setText("liveMappingRuntime", "Stopped");
        setText("liveMappingFreshness", "Stopped");
        setMessage(
            "Start supervised mapping to build a live map."
        );

        const error = byId("liveMappingError");
        if (error) error.hidden = true;
    }

    function setWaiting() {
        liveMap = null;
        clearCanvas();
        resetMetrics();
        setStatus("Waiting for grid", "waiting");
        setText("liveMappingRuntime", "Running");
        setText("liveMappingFreshness", "Waiting");
        setMessage(
            "Cartographer is running. Waiting for the first map grid..."
        );

        const error = byId("liveMappingError");
        if (error) error.hidden = true;
    }

    function setUnavailable(message) {
        liveMap = null;
        clearCanvas();
        resetMetrics();
        setStatus("Map unavailable", "error");
        setText("liveMappingRuntime", "Unavailable");
        setText("liveMappingFreshness", "Unavailable");
        setMessage(message);

        const error = byId("liveMappingError");

        if (error) {
            error.textContent = message;
            error.hidden = false;
        }
    }

    function render(payload) {
        const telemetry = payload.telemetry;
        const occupancyMap = telemetry.map;
        const origin = occupancyMap.origin;
        const age = Number(telemetry.age_seconds);
        const message = byId("liveMappingMessage");
        const error = byId("liveMappingError");

        if (
            payload.read_only !== true
            || payload.authoritative !== false
            || payload.runtime_active !== true
            || telemetry.available !== true
            || telemetry.status !== "READY"
            || occupancyMap.frame_id !== "map"
            || occupancyMap.encoding
                !== "ros_occupancy_probabilities"
        ) {
            throw new Error(
                "Live mapping safety state is invalid."
            );
        }

        liveMap = occupancyMap;
        drawMap(liveMap);

        if (message) message.hidden = true;
        if (error) error.hidden = true;

        setStatus("Live mapping", "ready");
        setText("liveMappingRuntime", "Running");
        setText(
            "liveMappingFreshness",
            Number.isFinite(age)
                ? `Updated ${age.toFixed(1)} seconds ago`
                : "Live"
        );
        setText("liveMappingFrame", occupancyMap.frame_id);
        setText(
            "liveMappingDimensions",
            `${Number(occupancyMap.width).toLocaleString()} × `
            + `${Number(occupancyMap.height).toLocaleString()} cells`
        );
        setText(
            "liveMappingResolution",
            `${Number(occupancyMap.resolution).toFixed(3)} m/cell`
        );
        setText(
            "liveMappingOrigin",
            `${Number(origin.x).toFixed(2)}, `
            + `${Number(origin.y).toFixed(2)}, `
            + `${Number(origin.yaw).toFixed(2)} rad`
        );
        setText(
            "liveMappingUnknown",
            Number(
                occupancyMap.unknown_cell_count
            ).toLocaleString()
        );
        setText(
            "liveMappingFree",
            Number(
                occupancyMap.free_cell_count
            ).toLocaleString()
        );
        setText(
            "liveMappingProbability",
            Number(
                occupancyMap.probability_cell_count
            ).toLocaleString()
        );
        setText(
            "liveMappingOccupied",
            Number(
                occupancyMap.occupied_cell_count
            ).toLocaleString()
        );
        setText(
            "liveMappingTotal",
            Number(
                occupancyMap.cell_count
            ).toLocaleString()
        );
        setText(
            "liveMappingAge",
            Number.isFinite(age)
                ? `${age.toFixed(2)} s`
                : "—"
        );
    }

    async function refresh() {
        if (
            !perceptionIsVisible()
            || requestInFlight
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const response = await fetch(ENDPOINT, {
                cache: "no-store",
            });
            const payload = await response.json();

            if (
                response.status === 503
                && payload.runtime_active === false
                && payload.telemetry
                && payload.telemetry.status
                    === "MAPPING_STOPPED"
            ) {
                setStopped();
                return;
            }

            if (
                response.status === 503
                && payload.runtime_active === true
                && payload.telemetry
                && payload.telemetry.map === null
            ) {
                setWaiting();
                return;
            }

            if (!response.ok || !payload.ok) {
                throw new Error(
                    payload.error
                    || (
                        payload.telemetry
                        && payload.telemetry.error
                    )
                    || "Live mapping telemetry is unavailable."
                );
            }

            render(payload);
        } catch (error) {
            setUnavailable(error.message);
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("liveMappingCanvas")) return;

        setStopped();
        refresh();

        timer = window.setInterval(
            refresh,
            REFRESH_MS
        );

        window.addEventListener(
            "resize",
            function () {
                if (
                    perceptionIsVisible()
                    && liveMap
                ) {
                    drawMap(liveMap);
                }
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Read-only candidate map review */
(function () {
    "use strict";

    const ENDPOINT = "/dashboard/map-candidates";
    const REFRESH_MS = 5000;

    let timer = null;
    let requestInFlight = false;
    let reviewMap = null;
    let reviewCandidates = [];
    let selectedCandidateName = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setText(id, value) {
        const element = byId(id);
        if (element) element.textContent = value;
    }

    function setStatus(label, state) {
        const status = byId("candidateReviewStatus");

        if (!status) return;

        status.textContent = label;
        status.className =
            "candidate-review-status " + state;
    }

    function resizeCanvas(canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(320, canvas.clientWidth);
        const height = Math.max(440, canvas.clientHeight);
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        return {
            context,
            width,
            height,
        };
    }

    function clearCanvas() {
        const canvas = byId("candidateReviewCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);

        drawing.context.fillStyle = "#020617";
        drawing.context.fillRect(
            0,
            0,
            drawing.width,
            drawing.height
        );
    }

    function createMapImage(occupancyMap) {
        const width = Number(occupancyMap.width);
        const height = Number(occupancyMap.height);
        const imageCanvas =
            document.createElement("canvas");

        imageCanvas.width = width;
        imageCanvas.height = height;

        const context = imageCanvas.getContext("2d");
        const image = context.createImageData(
            width,
            height
        );

        occupancyMap.cells.forEach(
            function (value, index) {
                const mapX = index % width;
                const mapY = Math.floor(index / width);
                const canvasY = height - 1 - mapY;
                const pixelIndex = (
                    (canvasY * width + mapX) * 4
                );

                let red = 15;
                let green = 23;
                let blue = 42;

                if (value === 0) {
                    red = 248;
                    green = 250;
                    blue = 252;
                } else if (value === 100) {
                    red = 245;
                    green = 158;
                    blue = 11;
                }

                image.data[pixelIndex] = red;
                image.data[pixelIndex + 1] = green;
                image.data[pixelIndex + 2] = blue;
                image.data[pixelIndex + 3] = 255;
            }
        );

        context.putImageData(image, 0, 0);
        return imageCanvas;
    }

    function drawMap(occupancyMap) {
        const canvas = byId("candidateReviewCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);
        const context = drawing.context;
        const padding = 32;
        const sourceWidth = Number(occupancyMap.width);
        const sourceHeight = Number(occupancyMap.height);
        const scale = Math.min(
            (
                drawing.width - padding * 2
            ) / sourceWidth,
            (
                drawing.height - padding * 2
            ) / sourceHeight
        );
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const drawX = (
            drawing.width - drawWidth
        ) / 2;
        const drawY = (
            drawing.height - drawHeight
        ) / 2;
        const image = createMapImage(occupancyMap);

        context.fillStyle = "#020617";
        context.fillRect(
            0,
            0,
            drawing.width,
            drawing.height
        );

        context.imageSmoothingEnabled = false;
        context.drawImage(
            image,
            drawX,
            drawY,
            drawWidth,
            drawHeight
        );

        context.strokeStyle = "#f59e0b";
        context.lineWidth = 1;
        context.strokeRect(
            drawX - 0.5,
            drawY - 0.5,
            drawWidth + 1,
            drawHeight + 1
        );

        context.fillStyle = "#fbbf24";
        context.font = "700 13px system-ui";
        context.textAlign = "center";
        context.fillText(
            "CANDIDATE — MAP +Y",
            drawing.width / 2,
            Math.max(18, drawY - 10)
        );
        context.textAlign = "start";
    }

    function candidateIsRenderable(candidate) {
        return Boolean(
            candidate
            && candidate.review_ready === true
            && candidate.classification
                === "REVIEW_READY"
            && candidate.map
        );
    }

    function selectCandidate(candidate) {
        if (!candidateIsRenderable(candidate)) return;

        selectedCandidateName = candidate.name;
        renderInventory(reviewCandidates);
        renderReadyCandidate(candidate);
        finishReadyCandidate(candidate);
    }

    function renderInventory(candidates) {
        const list = byId("candidateInventoryList");
        if (!list) return;

        list.replaceChildren();

        candidates.forEach(function (candidate) {
            const selectable =
                candidateIsRenderable(candidate);
            const item = document.createElement(
                selectable ? "button" : "div"
            );
            const name = document.createElement("strong");
            const classification =
                document.createElement("span");

            if (selectable) {
                item.type = "button";
            }

            item.className = (
                "candidate-inventory-item "
                + (selectable ? "ready" : "invalid")
                + (
                    candidate.name
                        === selectedCandidateName
                        ? " selected"
                        : ""
                )
            );

            name.textContent = candidate.name;
            classification.textContent =
                candidate.classification;

            item.appendChild(name);
            item.appendChild(classification);

            if (selectable) {
                item.setAttribute(
                    "aria-pressed",
                    candidate.name
                        === selectedCandidateName
                        ? "true"
                        : "false"
                );
                item.addEventListener(
                    "click",
                    function () {
                        selectCandidate(candidate);
                    }
                );
            }

            list.appendChild(item);
        });
    }

    function renderReadyCandidate(candidate) {
        const summary = candidate.map_summary;
        const comparison = candidate.comparison;
        const dimension =
            comparison.dimension_delta_cells;
        const origin = comparison.origin_delta_meters;
        const message = byId("candidateReviewMessage");

        reviewMap = candidate.map;
        drawMap(reviewMap);

        if (message) message.hidden = true;

        setText("candidateReviewName", candidate.name);
        setText(
            "candidateReviewClassification",
            candidate.classification
        );
        setText(
            "candidateReviewDimensions",
            `${Number(summary.width).toLocaleString()} × `
            + `${Number(summary.height).toLocaleString()} cells`
        );
        setText(
            "candidateReviewResolution",
            `${Number(summary.resolution).toFixed(3)} m/cell`
        );
    }

    function signedNumber(value, digits) {
        const number = Number(value);

        if (!Number.isFinite(number)) return "—";

        const fixed = number.toFixed(digits);
        return number > 0 ? `+${fixed}` : fixed;
    }

    function finishReadyCandidate(candidate) {
        const comparison = candidate.comparison;
        const dimension =
            comparison.dimension_delta_cells;
        const origin = comparison.origin_delta_meters;

        setText(
            "candidateDimensionDelta",
            `${signedNumber(dimension.width, 0)} × `
            + `${signedNumber(dimension.height, 0)} cells`
        );
        setText(
            "candidateOriginDelta",
            `${signedNumber(origin.x, 2)}, `
            + `${signedNumber(origin.y, 2)} m`
        );
        setText(
            "candidateCellDelta",
            Number(
                comparison.cell_count_delta
            ).toLocaleString()
        );
        setText(
            "candidateFrameMatch",
            comparison.same_frame === true
                ? "Match"
                : "Different"
        );
        setText(
            "candidateResolutionMatch",
            comparison.same_resolution === true
                ? "Match"
                : "Different"
        );
    }

    function setUnavailable(message) {
        reviewMap = null;
        clearCanvas();

        const canvasMessage =
            byId("candidateReviewMessage");
        const error = byId("candidateReviewError");

        setStatus("Review unavailable", "error");

        if (canvasMessage) {
            canvasMessage.textContent = message;
            canvasMessage.hidden = false;
        }

        if (error) {
            error.textContent = message;
            error.hidden = false;
        }
    }

    function render(payload) {
        const telemetry = payload.telemetry || payload;
        const candidates = telemetry.candidates || [];
        const readyCandidates = candidates.filter(
            candidateIsRenderable
        );
        let ready = readyCandidates.find(
            function (candidate) {
                return (
                    candidate.name
                    === selectedCandidateName
                );
            }
        );
        const error = byId("candidateReviewError");

        reviewCandidates = candidates;

        if (!ready && readyCandidates.length > 0) {
            ready = readyCandidates[
                readyCandidates.length - 1
            ];
            selectedCandidateName = ready.name;
        }

        if (
            telemetry.read_only !== true
            || telemetry.promotion_enabled !== false
        ) {
            throw new Error(
                "Candidate review safety state is invalid."
            );
        }

        renderInventory(candidates);

        setText(
            "candidateReviewCounts",
            `${Number(
                telemetry.review_ready_count
            ).toLocaleString()} ready / `
            + `${Number(
                telemetry.invalid_count
            ).toLocaleString()} blocked`
        );

        if (error) error.hidden = true;

        if (!ready) {
            throw new Error(
                "No review-ready candidate is available."
            );
        }

        renderReadyCandidate(ready);
        finishReadyCandidate(ready);

        setStatus(
            `${telemetry.review_ready_count} review-ready`,
            "ready"
        );
    }

    async function refresh() {
        if (
            !perceptionIsVisible()
            || requestInFlight
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const response = await fetch(ENDPOINT, {
                cache: "no-store",
            });
            const payload = await response.json();

            if (!response.ok || !payload.ok) {
                throw new Error(
                    payload.error
                    || "Candidate review is unavailable."
                );
            }

            render(payload);
        } catch (error) {
            setUnavailable(error.message);
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("candidateReviewCanvas")) return;

        clearCanvas();
        refresh();

        timer = window.setInterval(
            refresh,
            REFRESH_MS
        );

        window.addEventListener(
            "resize",
            function () {
                if (
                    perceptionIsVisible()
                    && reviewMap
                ) {
                    drawMap(reviewMap);
                }
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Read-only localized LiDAR map overlay */
(function () {
    "use strict";

    const MAP_ENDPOINT = "/dashboard/map";
    const LIDAR_ENDPOINT = "/dashboard/lidar";
    const LOCALIZATION_ENDPOINT =
        "/dashboard/localization";

    const REFRESH_MS = 500;
    const MAP_PADDING = 32;
    const DISPLAY_RANGE_METERS = 6.0;

    /*
     * Hardware-validated lidar_link correction:
     * raw zero is Mayday's right side. Adding +90 degrees makes
     * zero correspond to Mayday's physical forward direction.
     */
    const SCAN_TO_BASE_ROTATION_RADIANS =
        Math.PI / 2;

    let timer = null;
    let requestInFlight = false;
    let occupancyMap = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setStatus(label, state) {
        const status = byId(
            "localizedLidarOverlayStatus"
        );

        if (!status) return;

        status.textContent = label;
        status.className =
            "localized-lidar-overlay-status " + state;
    }

    function resizeCanvas(canvas) {
        const mapCanvas = byId("mapCanvas");
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(
            320,
            mapCanvas
                ? mapCanvas.clientWidth
                : canvas.clientWidth
        );
        const height = Math.max(
            480,
            mapCanvas
                ? mapCanvas.clientHeight
                : canvas.clientHeight
        );
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");

        context.setTransform(
            ratio,
            0,
            0,
            ratio,
            0,
            0
        );

        return {
            context,
            width,
            height,
        };
    }

    function clearScan() {
        const canvas = byId("localizedLidarCanvas");

        if (!canvas) return;

        const drawing = resizeCanvas(canvas);

        drawing.context.clearRect(
            0,
            0,
            drawing.width,
            drawing.height
        );
    }

    function mapGeometry(drawing) {
        const sourceWidth = Number(occupancyMap.width);
        const sourceHeight = Number(occupancyMap.height);

        const scale = Math.min(
            (
                drawing.width - MAP_PADDING * 2
            ) / sourceWidth,
            (
                drawing.height - MAP_PADDING * 2
            ) / sourceHeight
        );

        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;

        return {
            sourceWidth,
            sourceHeight,
            resolution: Number(
                occupancyMap.resolution
            ),
            origin: occupancyMap.origin,
            scale,
            drawX: (
                drawing.width - drawWidth
            ) / 2,
            drawY: (
                drawing.height - drawHeight
            ) / 2,
        };
    }

    function mapPointToCanvas(
        mapX,
        mapY,
        geometry
    ) {
        const cellX = (
            mapX - Number(geometry.origin.x)
        ) / geometry.resolution;

        const cellY = (
            mapY - Number(geometry.origin.y)
        ) / geometry.resolution;

        return {
            x: (
                geometry.drawX
                + cellX * geometry.scale
            ),
            y: (
                geometry.drawY
                + (
                    geometry.sourceHeight
                    - cellY
                ) * geometry.scale
            ),
            inside: (
                cellX >= 0
                && cellX <= geometry.sourceWidth
                && cellY >= 0
                && cellY <= geometry.sourceHeight
            ),
        };
    }

    function drawScan(scan, pose) {
        const canvas = byId("localizedLidarCanvas");

        if (
            !canvas
            || !occupancyMap
            || !scan
            || !pose
        ) {
            clearScan();
            return 0;
        }

        const drawing = resizeCanvas(canvas);
        const context = drawing.context;
        const geometry = mapGeometry(drawing);

        const robotX = Number(pose.position.x);
        const robotY = Number(pose.position.y);
        const robotYaw = Number(pose.yaw_radians);

        const angleMinimum = Number(scan.angle_min);
        const angleIncrement = Number(
            scan.angle_increment
        );
        const rangeMinimum = Number(scan.range_min);
        const rangeMaximum = Math.min(
            Number(scan.range_max),
            DISPLAY_RANGE_METERS
        );

        context.clearRect(
            0,
            0,
            drawing.width,
            drawing.height
        );

        context.fillStyle = "#a3e635";
        context.shadowColor =
            "rgba(163, 230, 53, 0.80)";
        context.shadowBlur = 4;

        let drawn = 0;

        scan.ranges.forEach(
            function (rawRange, index) {
                if (
                    typeof rawRange !== "number"
                    || !Number.isFinite(rawRange)
                    || rawRange < rangeMinimum
                    || rawRange > rangeMaximum
                ) {
                    return;
                }

                const rawAngle = (
                    angleMinimum
                    + index * angleIncrement
                );

                const mapBearing = (
                    robotYaw
                    + rawAngle
                    + SCAN_TO_BASE_ROTATION_RADIANS
                );

                const mapX = (
                    robotX
                    + rawRange * Math.cos(mapBearing)
                );

                const mapY = (
                    robotY
                    + rawRange * Math.sin(mapBearing)
                );

                const point = mapPointToCanvas(
                    mapX,
                    mapY,
                    geometry
                );

                if (!point.inside) return;

                context.beginPath();
                context.arc(
                    point.x,
                    point.y,
                    2.1,
                    0,
                    Math.PI * 2
                );
                context.fill();

                drawn += 1;
            }
        );

        context.shadowBlur = 0;

        return drawn;
    }

    async function loadMap() {
        if (occupancyMap) return true;

        const response = await fetch(
            MAP_ENDPOINT,
            {cache: "no-store"}
        );
        const payload = await response.json();

        if (
            !response.ok
            || !payload.ok
            || !payload.telemetry
            || !payload.telemetry.map
        ) {
            return false;
        }

        occupancyMap = payload.telemetry.map;
        return true;
    }

    async function refresh() {
        if (
            requestInFlight
            || !perceptionIsVisible()
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const mapReady = await loadMap();

            if (!mapReady) {
                clearScan();
                setStatus("Map unavailable", "error");
                return;
            }

            const responses = await Promise.all([
                fetch(
                    LOCALIZATION_ENDPOINT,
                    {cache: "no-store"}
                ),
                fetch(
                    LIDAR_ENDPOINT,
                    {cache: "no-store"}
                ),
            ]);

            const localizationResponse = responses[0];
            const lidarResponse = responses[1];

            const localization =
                await localizationResponse.json();
            const lidar = await lidarResponse.json();

            if (
                localizationResponse.status === 503
                || localization.runtime_active !== true
                || !localization.telemetry
                || localization.telemetry.available
                    !== true
                || !localization.telemetry.pose
            ) {
                clearScan();
                setStatus(
                    "Localized scan stopped",
                    "stopped"
                );
                return;
            }

            if (
                !lidarResponse.ok
                || !lidar.ok
                || !lidar.telemetry
                || lidar.telemetry.available !== true
                || !lidar.telemetry.scan
            ) {
                clearScan();
                setStatus(
                    "Waiting for live scan",
                    "waiting"
                );
                return;
            }

            if (
                Number(lidar.telemetry.age_seconds)
                > 1.0
            ) {
                clearScan();
                setStatus(
                    "Live scan is stale",
                    "waiting"
                );
                return;
            }

            const pose = localization.telemetry.pose;
            const scan = lidar.telemetry.scan;

            if (
                pose.frame_id !== "map"
                || scan.frame_id !== "lidar_link"
            ) {
                clearScan();
                setStatus(
                    "Frame mismatch",
                    "error"
                );
                return;
            }

            const drawn = drawScan(scan, pose);

            setStatus(
                `Live map scan: ${drawn} points`,
                drawn > 0 ? "ready" : "waiting"
            );
        } catch (error) {
            clearScan();
            setStatus(
                "Localized scan unavailable",
                "error"
            );
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("localizedLidarCanvas")) {
            return;
        }

        clearScan();
        setStatus(
            "Localized scan stopped",
            "stopped"
        );

        refresh();

        timer = window.setInterval(
            refresh,
            REFRESH_MS
        );

        window.addEventListener(
            "resize",
            function () {
                clearScan();
                refresh();
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();

/* Read-only saved occupancy-map visualization */
(function () {
    "use strict";

    const MAP_ENDPOINT = "/dashboard/map";
    const WAITING_REFRESH_MS = 500;

    let timer = null;
    let requestInFlight = false;
    let loadedMap = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setText(id, value) {
        const element = byId(id);
        if (element) element.textContent = value;
    }

    function resizeCanvas(canvas) {
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(320, canvas.clientWidth);
        const height = Math.max(480, canvas.clientHeight);
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        return {
            context,
            width,
            height,
        };
    }

    function createMapImage(occupancyMap) {
        const width = Number(occupancyMap.width);
        const height = Number(occupancyMap.height);
        const cells = occupancyMap.cells;
        const imageCanvas = document.createElement("canvas");

        imageCanvas.width = width;
        imageCanvas.height = height;

        const context = imageCanvas.getContext("2d");
        const image = context.createImageData(width, height);

        cells.forEach(function (value, index) {
            const mapX = index % width;
            const mapY = Math.floor(index / width);
            const canvasY = height - 1 - mapY;
            const pixelIndex = (
                (canvasY * width + mapX) * 4
            );

            let red = 15;
            let green = 23;
            let blue = 42;

            if (value === 0) {
                red = 248;
                green = 250;
                blue = 252;
            } else if (value === 100) {
                red = 56;
                green = 189;
                blue = 248;
            }

            image.data[pixelIndex] = red;
            image.data[pixelIndex + 1] = green;
            image.data[pixelIndex + 2] = blue;
            image.data[pixelIndex + 3] = 255;
        });

        context.putImageData(image, 0, 0);
        return imageCanvas;
    }

    function drawMap(occupancyMap) {
        const canvas = byId("mapCanvas");
        if (!canvas) return;

        const drawing = resizeCanvas(canvas);
        const context = drawing.context;
        const width = drawing.width;
        const height = drawing.height;
        const padding = 32;
        const sourceWidth = Number(occupancyMap.width);
        const sourceHeight = Number(occupancyMap.height);
        const scale = Math.min(
            (width - padding * 2) / sourceWidth,
            (height - padding * 2) / sourceHeight
        );
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const drawX = (width - drawWidth) / 2;
        const drawY = (height - drawHeight) / 2;
        const imageCanvas = createMapImage(occupancyMap);

        context.fillStyle = "#020617";
        context.fillRect(0, 0, width, height);

        context.imageSmoothingEnabled = false;
        context.drawImage(
            imageCanvas,
            drawX,
            drawY,
            drawWidth,
            drawHeight
        );

        context.strokeStyle = "#64748b";
        context.lineWidth = 1;
        context.strokeRect(
            drawX - 0.5,
            drawY - 0.5,
            drawWidth + 1,
            drawHeight + 1
        );

        context.fillStyle = "#94a3b8";
        context.font = "700 13px system-ui";
        context.textAlign = "center";
        context.fillText(
            "MAP +Y",
            width / 2,
            Math.max(18, drawY - 10)
        );
        context.textAlign = "start";
    }

    function setOffline(message) {
        const pill = byId("mapStatusPill");
        const canvasMessage = byId("mapCanvasMessage");
        const error = byId("mapError");

        if (pill) {
            pill.textContent = "Map Offline";
            pill.className = "console-status-pill error";
        }

        setText("mapConnection", "Offline");
        setText("mapFreshness", "Map unavailable");

        if (canvasMessage) {
            canvasMessage.textContent = message;
            canvasMessage.hidden = false;
        }

        if (error) {
            error.textContent = message;
            error.hidden = false;
        }
    }

    function render(payload) {
        const telemetry = payload.telemetry;
        const occupancyMap = telemetry.map;
        const origin = occupancyMap.origin;
        const pill = byId("mapStatusPill");
        const message = byId("mapCanvasMessage");
        const error = byId("mapError");

        loadedMap = occupancyMap;
        drawMap(occupancyMap);

        if (pill) {
            pill.textContent = "Map Ready";
            pill.className = "console-status-pill ready";
        }

        if (message) message.hidden = true;
        if (error) error.hidden = true;

        setText("mapConnection", "Connected");
        setText("mapName", occupancyMap.name || "—");
        setText(
            "mapDimensions",
            `${Number(occupancyMap.width).toLocaleString()} × `
            + `${Number(occupancyMap.height).toLocaleString()} cells`
        );
        setText(
            "mapResolution",
            `${Number(occupancyMap.resolution).toFixed(3)} m/cell`
        );
        setText(
            "mapOrigin",
            `${Number(origin.x).toFixed(2)}, `
            + `${Number(origin.y).toFixed(2)}, `
            + `${Number(origin.yaw).toFixed(2)} rad`
        );
        setText(
            "mapUnknownCells",
            Number(
                occupancyMap.unknown_cell_count
            ).toLocaleString()
        );
        setText(
            "mapFreeCells",
            Number(
                occupancyMap.free_cell_count
            ).toLocaleString()
        );
        setText(
            "mapOccupiedCells",
            Number(
                occupancyMap.occupied_cell_count
            ).toLocaleString()
        );
        setText(
            "mapTotalCells",
            Number(occupancyMap.cell_count).toLocaleString()
        );
        setText(
            "mapFreshness",
            "Validated saved map loaded"
        );
    }

    async function refresh() {
        if (
            loadedMap
            || !perceptionIsVisible()
            || requestInFlight
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const response = await fetch(MAP_ENDPOINT, {
                cache: "no-store",
            });
            const payload = await response.json();

            if (
                !response.ok
                || !payload.ok
                || !payload.telemetry
                || !payload.telemetry.available
                || !payload.telemetry.map
            ) {
                throw new Error(
                    payload.error
                    || "Mayday saved map is unavailable."
                );
            }

            render(payload);

            if (timer !== null) {
                window.clearInterval(timer);
                timer = null;
            }
        } catch (error) {
            setOffline(error.message);
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("mapCanvas")) return;

        const canvas = byId("mapCanvas");
        const context = canvas.getContext("2d");

        context.fillStyle = "#020617";
        context.fillRect(0, 0, canvas.width, canvas.height);

        refresh();
        timer = window.setInterval(
            refresh,
            WAITING_REFRESH_MS
        );

        window.addEventListener(
            "resize",
            function () {
                if (
                    perceptionIsVisible()
                    && loadedMap
                ) {
                    drawMap(loadedMap);
                }
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();


/* Read-only localization pose overlay */
(function () {
    "use strict";

    const LOCALIZATION_ENDPOINT =
        "/dashboard/localization";
    const MAP_ENDPOINT =
        "/dashboard/map";
    const REFRESH_MS = 500;
    const MAP_PADDING = 32;

    let timer = null;
    let requestInFlight = false;
    let occupancyMap = null;
    let lastPose = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function perceptionIsVisible() {
        const page = byId("perceptionPage");

        return Boolean(
            page
            && !page.hidden
            && page.classList.contains("active")
        );
    }

    function setText(id, value) {
        const element = byId(id);
        if (element) element.textContent = value;
    }

    function setOverlayStatus(label, state) {
        const status = byId("localizationOverlayStatus");

        if (!status) return;

        status.textContent = label;
        status.className =
            "localization-overlay-status " + state;
    }

    function resizeOverlayCanvas(canvas) {
        const mapCanvas = byId("mapCanvas");
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(
            320,
            mapCanvas
                ? mapCanvas.clientWidth
                : canvas.clientWidth
        );
        const height = Math.max(
            480,
            mapCanvas
                ? mapCanvas.clientHeight
                : canvas.clientHeight
        );
        const pixelWidth = Math.round(width * ratio);
        const pixelHeight = Math.round(height * ratio);

        if (
            canvas.width !== pixelWidth
            || canvas.height !== pixelHeight
        ) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }

        const context = canvas.getContext("2d");
        context.setTransform(
            ratio,
            0,
            0,
            ratio,
            0,
            0
        );

        return {
            context,
            width,
            height,
        };
    }

    function clearPose() {
        const canvas = byId("localizationPoseCanvas");
        if (!canvas) return;

        const drawing = resizeOverlayCanvas(canvas);
        drawing.context.clearRect(
            0,
            0,
            drawing.width,
            drawing.height
        );

        lastPose = null;
    }

    function mapToCanvas(position, drawing) {
        const sourceWidth = Number(occupancyMap.width);
        const sourceHeight = Number(occupancyMap.height);
        const resolution = Number(occupancyMap.resolution);
        const origin = occupancyMap.origin;

        const scale = Math.min(
            (
                drawing.width
                - MAP_PADDING * 2
            ) / sourceWidth,
            (
                drawing.height
                - MAP_PADDING * 2
            ) / sourceHeight
        );

        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const drawX = (
            drawing.width - drawWidth
        ) / 2;
        const drawY = (
            drawing.height - drawHeight
        ) / 2;

        const mapCellX = (
            Number(position.x) - Number(origin.x)
        ) / resolution;
        const mapCellY = (
            Number(position.y) - Number(origin.y)
        ) / resolution;

        return {
            x: drawX + mapCellX * scale,
            y: (
                drawY
                + (sourceHeight - mapCellY) * scale
            ),
            inside: (
                mapCellX >= 0
                && mapCellX <= sourceWidth
                && mapCellY >= 0
                && mapCellY <= sourceHeight
            ),
        };
    }

    function drawRobot(pose) {
        if (!occupancyMap || !pose) return;

        const canvas = byId("localizationPoseCanvas");
        if (!canvas) return;

        const drawing = resizeOverlayCanvas(canvas);
        const context = drawing.context;
        const point = mapToCanvas(
            pose.position,
            drawing
        );
        const yaw = Number(pose.yaw_radians);

        context.clearRect(
            0,
            0,
            drawing.width,
            drawing.height
        );

        if (
            !point.inside
            || !Number.isFinite(yaw)
        ) {
            setOverlayStatus(
                "Pose outside saved map",
                "error"
            );
            return;
        }

        context.save();
        context.translate(point.x, point.y);

        /*
         * ROS map yaw zero points along +X. Canvas +Y points down,
         * so the rendered heading uses negative sine vertically.
         */
        context.rotate(-yaw);

        context.shadowColor =
            "rgba(245, 158, 11, 0.85)";
        context.shadowBlur = 10;

        context.fillStyle = "#f59e0b";
        context.strokeStyle = "#fff7ed";
        context.lineWidth = 2;

        context.beginPath();
        context.moveTo(19, 0);
        context.lineTo(-12, -11);
        context.lineTo(-7, 0);
        context.lineTo(-12, 11);
        context.closePath();
        context.fill();
        context.stroke();

        context.restore();

        context.save();
        context.fillStyle = "#f8fafc";
        context.strokeStyle = "rgba(15, 23, 42, 0.9)";
        context.lineWidth = 4;
        context.font = "800 13px system-ui";
        context.textAlign = "center";
        context.strokeText(
            "Mayday",
            point.x,
            point.y - 22
        );
        context.fillText(
            "Mayday",
            point.x,
            point.y - 22
        );
        context.restore();

        lastPose = pose;
    }

    function renderStopped(payload) {
        clearPose();

        const telemetry = payload.telemetry || {};
        const status = telemetry.status
            || "LOCALIZATION_STOPPED";

        setOverlayStatus(
            status === "LOCALIZATION_STOPPED"
                ? "Localization stopped"
                : "Waiting for localization",
            "stopped"
        );

        setText(
            "localizationConnection",
            status === "LOCALIZATION_STOPPED"
                ? "Stopped"
                : "Waiting"
        );
        setText("localizationPose", "—");
        setText("localizationHeading", "—");
        setText("localizationPoseAge", "—");
    }

    function renderPose(payload) {
        const telemetry = payload.telemetry;
        const pose = telemetry.pose;

        if (
            payload.runtime_active !== true
            || telemetry.available !== true
            || !pose
            || pose.frame_id !== "map"
        ) {
            renderStopped(payload);
            return;
        }

        drawRobot(pose);

        setOverlayStatus(
            "Localization active",
            "ready"
        );
        setText("localizationConnection", "Active");
        setText(
            "localizationPose",
            `${Number(pose.position.x).toFixed(2)}, `
            + `${Number(pose.position.y).toFixed(2)} m`
        );
        setText(
            "localizationHeading",
            `${Number(pose.yaw_degrees).toFixed(1)}°`
        );
        setText(
            "localizationPoseAge",
            Number.isFinite(
                Number(telemetry.age_seconds)
            )
                ? `${Number(
                    telemetry.age_seconds
                ).toFixed(1)} s`
                : "—"
        );
    }

    async function loadMap() {
        if (occupancyMap) return true;

        const response = await fetch(MAP_ENDPOINT, {
            cache: "no-store",
        });
        const payload = await response.json();

        if (
            !response.ok
            || !payload.ok
            || !payload.telemetry
            || !payload.telemetry.map
        ) {
            return false;
        }

        occupancyMap = payload.telemetry.map;
        return true;
    }

    async function refresh() {
        if (
            !perceptionIsVisible()
            || requestInFlight
        ) {
            return;
        }

        requestInFlight = true;

        try {
            const mapReady = await loadMap();

            if (!mapReady) {
                clearPose();
                setOverlayStatus(
                    "Map unavailable",
                    "error"
                );
                setText(
                    "localizationConnection",
                    "Map unavailable"
                );
                return;
            }

            const response = await fetch(
                LOCALIZATION_ENDPOINT,
                {cache: "no-store"}
            );
            const payload = await response.json();

            /*
             * A 503 from the guarded Robot Bridge endpoint is an
             * expected stopped/waiting state, not a pose to cache.
             */
            if (
                response.status === 503
                && payload.runtime_active === false
            ) {
                renderStopped(payload);
                return;
            }

            if (
                !response.ok
                || !payload.ok
            ) {
                renderStopped(payload);
                return;
            }

            renderPose(payload);
        } catch (error) {
            clearPose();
            setOverlayStatus(
                "Localization unavailable",
                "error"
            );
            setText(
                "localizationConnection",
                "Offline"
            );
            setText("localizationPose", "—");
            setText("localizationHeading", "—");
            setText("localizationPoseAge", "—");
        } finally {
            requestInFlight = false;
        }
    }

    function initialize() {
        if (!byId("localizationPoseCanvas")) {
            return;
        }

        clearPose();
        renderStopped({
            telemetry: {
                status: "LOCALIZATION_STOPPED",
            },
        });

        refresh();
        timer = window.setInterval(
            refresh,
            REFRESH_MS
        );

        window.addEventListener(
            "resize",
            function () {
                const pose = lastPose;
                clearPose();

                if (
                    perceptionIsVisible()
                    && pose
                    && occupancyMap
                ) {
                    drawRobot(pose);
                }
            }
        );

        window.addEventListener(
            "beforeunload",
            function () {
                if (timer !== null) {
                    window.clearInterval(timer);
                }
            },
            {once: true}
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();
