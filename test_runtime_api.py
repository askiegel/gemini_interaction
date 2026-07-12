#!/usr/bin/env python3

import json
import threading
import urllib.error
import urllib.request

from runtime_api import create_server


class FakeRuntime:
    def __init__(self):
        self.running = True
        self.submissions = []
        self.active_mission = None
        self.queue = []
        self.history_count = 0
        self.last_result = None
        self.last_error = None

    def submit_text(self, command):
        mission_number = len(self.submissions) + 1

        mission = {
            "mission_id": f"mission-test-{mission_number}",
            "mission_type": (
                "FIND_OBJECT"
                if "backpack" in command.lower()
                else "TURN_LEFT"
            ),
            "status": (
                "ACTIVE"
                if self.active_mission is None
                else "QUEUED"
            ),
            "target": (
                "backpack"
                if "backpack" in command.lower()
                else None
            ),
            "speech": "Test mission accepted.",
            "created_at": "2026-07-12T12:00:00",
            "started_at": None,
            "completed_at": None,
            "priority": 6,
            "source": "cognitive",
        }

        intent = {
            "intent": mission["mission_type"],
            "speech": mission["speech"],
            "target": mission["target"],
        }

        self.submissions.append(command)
        self.history_count += 1

        if self.active_mission is None:
            self.active_mission = mission
        else:
            self.queue.append(mission)

        return {
            "command": command,
            "intent": intent,
            "mission": dict(mission),
        }

    def get_status(self):
        return {
            "ok": True,
            "service": "mini_pupper_cognitive_runtime",
            "running": self.running,
            "runtime_state": "TESTING",
            "uptime_seconds": 1.0,
            "active_mission": self.active_mission,
            "queue": list(self.queue),
            "history_count": self.history_count,
            "last_result": self.last_result,
            "last_error": self.last_error,
        }


def request_json(
    method,
    url,
    payload=None,
):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=5.0,
        ) as response:
            return (
                response.status,
                json.loads(
                    response.read().decode("utf-8")
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
    runtime = FakeRuntime()

    server = create_server(
        runtime=runtime,
        host="127.0.0.1",
        port=0,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="runtime-api-test",
        daemon=True,
    )

    server_thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        print("===== HEALTH ENDPOINT =====")
        status_code, health = request_json(
            "GET",
            f"{base_url}/health",
        )
        print(status_code, health)

        assert status_code == 200
        assert health["ok"] is True
        assert health["runtime_running"] is True

        print()
        print("===== INITIAL STATUS ENDPOINT =====")
        status_code, status = request_json(
            "GET",
            f"{base_url}/status",
        )
        print(status_code, status)

        assert status_code == 200
        assert status["active_mission"] is None
        assert status["queue"] == []

        print()
        print("===== SUBMIT FIRST MISSION =====")
        status_code, first = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "command": "Find my backpack",
            },
        )
        print(status_code, first)

        assert status_code == 202
        assert first["ok"] is True
        assert first["accepted"] is True
        assert (
            first["mission"]["mission_type"]
            == "FIND_OBJECT"
        )
        assert first["mission"]["status"] == "ACTIVE"

        print()
        print("===== SUBMIT SECOND MISSION =====")
        status_code, second = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "command": "Turn left",
            },
        )
        print(status_code, second)

        assert status_code == 202
        assert second["ok"] is True
        assert second["mission"]["status"] == "QUEUED"

        print()
        print("===== MISSIONS ENDPOINT =====")
        status_code, missions = request_json(
            "GET",
            f"{base_url}/missions",
        )
        print(status_code, missions)

        assert status_code == 200
        assert (
            missions["active_mission"]["mission_type"]
            == "FIND_OBJECT"
        )
        assert len(missions["queue"]) == 1
        assert (
            missions["queue"][0]["mission_type"]
            == "TURN_LEFT"
        )
        assert missions["history_count"] == 2

        print()
        print("===== EMPTY COMMAND VALIDATION =====")
        status_code, invalid = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "command": "   ",
            },
        )
        print(status_code, invalid)

        assert status_code == 400
        assert invalid["ok"] is False
        assert invalid["accepted"] is False

        print()
        print("===== UNKNOWN ENDPOINT =====")
        status_code, missing = request_json(
            "GET",
            f"{base_url}/missing",
        )
        print(status_code, missing)

        assert status_code == 404
        assert missing["ok"] is False

        print()
        print("PASS: runtime health endpoint")
        print("PASS: runtime status endpoint")
        print("PASS: first mission accepted as active")
        print("PASS: second mission accepted as queued")
        print("PASS: mission queue visible through HTTP")
        print("PASS: empty commands rejected safely")
        print("PASS: unknown endpoints return 404")
        print()
        print("Runtime API offline integration test passed.")

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
