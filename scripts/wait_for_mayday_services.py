#!/usr/bin/env python3
"""Wait for local Mayday application HTTP readiness without changing state."""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


SERVICES = (
    (
        "mayday-cognitive-runtime.service",
        "http://127.0.0.1:8770/health",
        "cognitive_runtime",
    ),
    (
        "mayday-voice-relay.service",
        "http://127.0.0.1:8765/",
        "voice_relay",
    ),
)


def service_status(unit):
    """Return only unprivileged systemd process state for one unit."""
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {
        "unit": unit,
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "pid": values.get("MainPID"),
        "systemctl_returncode": result.returncode,
    }


def http_ready(url, name):
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            body = response.read()
            ready = response.status == 200
            if name == "cognitive_runtime":
                try:
                    ready = ready and json.loads(body.decode("utf-8")).get("ok") is True
                except (UnicodeDecodeError, ValueError, AttributeError):
                    ready = False
            return {"url": url, "http_status": response.status, "ready": ready}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"url": url, "ready": False, "error": str(exc)}


def readiness_snapshot():
    snapshot = {}
    for unit, url, name in SERVICES:
        snapshot[name] = {
            "service": service_status(unit),
            "http": http_ready(url, name),
        }
    return snapshot


def ready(snapshot):
    return all(
        item["service"]["active_state"] == "active"
        and item["http"].get("ready") is True
        for item in snapshot.values()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Wait for local Mayday application readiness without restart."
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        parser.error("--timeout and --interval must be positive.")

    deadline = time.monotonic() + args.timeout
    while True:
        snapshot = readiness_snapshot()
        if ready(snapshot):
            print(json.dumps(snapshot, sort_keys=True))
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(snapshot, sort_keys=True), file=sys.stderr)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
