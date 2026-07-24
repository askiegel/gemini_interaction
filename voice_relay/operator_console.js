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
