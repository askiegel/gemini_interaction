#!/usr/bin/env python3

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from voice_relay.server import VoiceRelayHandler


def request_json(
    method,
    url,
    payload=None,
):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(
            payload
        ).encode("utf-8")

        headers["Content-Type"] = (
            "application/json"
        )

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=20.0,
        ) as response:
            return (
                response.status,
                json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                ),
            )

    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            json.loads(
                exc.read().decode("utf-8")
            ),
        )


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        VoiceRelayHandler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    server_thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        print("===== RELAY STATUS =====")

        status_code, status = request_json(
            "GET",
            f"{base_url}/status",
        )

        print(status_code, status)

        assert status_code == 200
        assert status["ok"] is True

        assert (
            status["submission_mode"]
            == "persistent_runtime"
        )

        print()
        print("===== DRY-RUN COMMAND =====")

        status_code, dry_run = request_json(
            "POST",
            f"{base_url}/command",
            {
                "command": "Turn left",
                "execute": False,
            },
        )

        print(status_code, dry_run)

        assert status_code == 200
        assert dry_run["ok"] is True
        assert dry_run["mode"] == "dry-run"

        assert (
            dry_run["submitted_to_runtime"]
            is False
        )

        assert (
            "No robot command was sent."
            in dry_run["stdout"]
        )

        print()
        print("PASS: relay status identifies runtime mode")
        print("PASS: dry-run mode remains available")
        print("PASS: dry run does not submit a mission")
        print()
        print(
            "Browser voice relay runtime test passed."
        )

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
