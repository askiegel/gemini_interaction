"use strict";
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function deferred() {
    let resolve, reject;
    const promise = new Promise((a, b) => { resolve = a; reject = b; });
    return {promise, resolve, reject};
}
function fixture(fetch) {
    const listeners = {};
    const document = {hidden: false, addEventListener: (name, fn) => { listeners[name] = fn; }};
    const window = {fetch, setTimeout, clearTimeout};
    let now = 0;
    const context = vm.createContext({window, document, Response, AbortController,
        DOMException, performance: {now: () => now}});
    vm.runInContext(fs.readFileSync(path.join(__dirname, "voice_relay/operator_console.js"), "utf8").split("/* Operator console modules */")[0], context);
    return {api: window.MaydayTelemetry, document, listeners, advance: (ms) => { now += ms; }};
}

test("same-endpoint requests coalesce and independent consumers read complete JSON", async () => {
    const request = deferred(); let calls = 0;
    const f = fixture(() => { calls++; return calls === 1 ? request.promise : Promise.resolve(Response.json({state: "IDLE"})); });
    const first = f.api.fetch("/dashboard/status");
    const second = f.api.fetch("/dashboard/status");
    assert.equal(calls, 1);
    request.resolve(Response.json({state: "IDLE"}));
    assert.deepEqual(await (await first).json(), {state: "IDLE"});
    assert.deepEqual(await (await second).json(), {state: "IDLE"});
    await f.api.fetch("/dashboard/status");
    assert.equal(calls, 1);
    f.advance(500);
    await f.api.fetch("/dashboard/status");
    assert.equal(calls, 2);
});

test("old response finishing after a new generation cannot update the UI", async () => {
    const old = deferred(); let calls = 0, oldSignal;
    const f = fixture((_url, options) => {
        calls++;
        if (calls === 1) { oldSignal = options.signal; return old.promise; }
        return Promise.resolve(Response.json({sequence: 2}));
    });
    const obsolete = f.api.fetch("/dashboard/lidar");
    const rejected = assert.rejects(obsolete, {name: "AbortError"});
    f.document.hidden = true; f.listeners.visibilitychange();
    assert.equal(oldSignal.aborted, true);
    f.document.hidden = false;
    const latest = await (await f.api.fetch("/dashboard/lidar")).json();
    old.resolve(Response.json({sequence: 1}));
    await rejected;
    assert.equal(latest.sequence, 2);
});

test("a superseded JSON body cannot overwrite newer state", async () => {
    const f = fixture(() => Promise.resolve(Response.json({sequence: 1})));
    const old = await f.api.fetch("/dashboard/status");
    f.api.invalidate();
    await assert.rejects(old.json(), {name: "AbortError"});
});

test("text-based poll consumers also reject superseded bodies", async () => {
    const f = fixture(() => Promise.resolve(Response.json({sequence: 1})));
    const old = await f.api.fetch("/dashboard/network-status");
    f.api.invalidate();
    await assert.rejects(old.text(), {name: "AbortError"});
});

test("hidden tabs make no GET request; STOP is forwarded exactly once", async () => {
    const calls = [];
    const f = fixture((url, options) => { calls.push({url, options}); return Promise.resolve(Response.json({ok: true})); });
    f.document.hidden = true;
    await assert.rejects(f.api.fetch("/dashboard/status"), {name: "AbortError"});
    assert.equal(calls.length, 0);
    const options = {method: "POST", body: "{}"};
    assert.equal((await f.api.fetch("/stop", options)).status, 200);
    assert.deepEqual(calls, [{url: "/stop", options}]);
});

test("live GETs use no-store and different endpoints do not block one another", async () => {
    const calls = [];
    const pending = deferred();
    const f = fixture((url, options) => {
        calls.push({url, options});
        return url === "/dashboard/status" ? pending.promise : Promise.resolve(Response.json({scan: 1}));
    });
    const status = f.api.fetch("/dashboard/status");
    const lidar = await f.api.fetch("/dashboard/lidar");
    assert.equal((await lidar.json()).scan, 1);
    assert.equal(calls.length, 2);
    assert.ok(calls.every(call => call.options.cache === "no-store"));
    pending.resolve(Response.json({state: "IDLE"})); await status;
});

test("raw scan visibility depends on Bridge, not Cognitive safety freshness", () => {
    const f = fixture(() => {});
    const raw = {ok: true, telemetry: {available: true, age_seconds: .05, scan: {stamp_seconds: 1}},
        cognitive: {valid: false, reason: "stale"}};
    assert.equal(f.api.bridgeScanVisible(raw), true);
    raw.telemetry.age_seconds = .5;
    assert.equal(f.api.bridgeScanVisible(raw), false);
});

test("mission canvas redraws changing scans while status is delayed or stale", async () => {
    const status = deferred(); const intervals = new Map(); const arcs = [];
    const context2d = new Proxy({}, {get: (_target, key) => key === "arc" ? (...args) => arcs.push(args) : () => {}});
    const nodes = new Map();
    function node(id) {
        if (!nodes.has(id)) nodes.set(id, {hidden: false, textContent: "", width: 640, height: 480,
            classList: {contains: () => true}, getContext: () => context2d});
        return nodes.get(id);
    }
    let stamp = 1, range = 1;
    const raw = () => ({ok: true, telemetry: {available: true, age_seconds: .04,
        scan: {stamp_seconds: stamp, angle_min: 0, angle_increment: .1,
            range_min: .01, range_max: 8, ranges: [range]}}});
    const helper = fixture(() => {}).api;
    const document = {readyState: "complete", getElementById: node};
    const window = {setInterval: (fn, ms) => intervals.set(ms, fn), MaydayTelemetry: {
        bridgeScanVisible: helper.bridgeScanVisible,
        fetch: url => url === "/dashboard/status" ? status.promise : Promise.resolve(Response.json(raw())),
    }};
    const source = fs.readFileSync(path.join(__dirname, "voice_relay/operator_console.js"), "utf8")
        .split("/* Mission camera companion: read-only robot-relative LiDAR */")[1]
        .split("/* =========================================================")[0];
    vm.runInNewContext(source, {window, document});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(node("missionLidarMessage").hidden, true);
    const initialArcs = arcs.length;
    status.resolve(Response.json({runtime: {lidar: {running: true, available: false, valid: false,
        reason: "stale", effective_age_seconds: .45}}}));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(node("missionCognitiveLidarState").textContent, "stale");
    stamp = 2; range = 2;
    await intervals.get(125)();
    assert.ok(arcs.length > initialArcs);
    assert.equal(node("missionLidarMessage").hidden, true);
    assert.equal(node("missionLidarState").textContent, "Bridge scan fresh");
    assert.ok(intervals.has(500));
});

test("head-loaded console registers initialization before the DOM exists without fetching", () => {
    const registrations = []; let calls = 0;
    const document = {readyState: "loading", hidden: false, getElementById: () => null,
        querySelector: () => null, querySelectorAll: () => [],
        addEventListener: (event, fn) => registrations.push({event, fn})};
    const window = {fetch: () => { calls++; throw new Error("premature fetch"); },
        addEventListener: () => {}, setTimeout, clearTimeout};
    const source = fs.readFileSync(path.join(__dirname, "voice_relay/operator_console.js"), "utf8");
    vm.runInNewContext(source, {window, document, Response, AbortController,
        DOMException, performance, console});
    assert.equal(typeof window.MaydayTelemetry.fetch, "function");
    assert.ok(registrations.filter(item => item.event === "DOMContentLoaded").length >= 20);
    assert.equal(calls, 0);
});
