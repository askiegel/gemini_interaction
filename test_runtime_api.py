#!/usr/bin/env python3

import json
import threading
import urllib.error
import urllib.request

from runtime_api import create_server


class FakeMission:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return dict(self.data)


class FakeRuntime:
    def __init__(self):
        self.running = True
        self.submissions = []
        self.active_mission = None
        self.queue = []
        self.history_count = 0
        self.last_result = None
        self.last_error = None

    def _accept_intent(
        self,
        intent,
        command=None,
    ):
        mission_number = len(self.submissions) + 1
        mission_type = intent.get("intent", "UNKNOWN")

        mission = {
            "mission_id": f"mission-test-{mission_number}",
            "mission_type": mission_type,
            "status": (
                "ACTIVE"
                if self.active_mission is None
                else "QUEUED"
            ),
            "target": intent.get("target"),
            "speech": intent.get(
                "speech",
                "Test mission accepted.",
            ),
            "created_at": "2026-07-12T12:00:00",
            "started_at": None,
            "completed_at": None,
            "priority": 6,
            "source": "cognitive",
        }

        self.submissions.append(
            {
                "command": command,
                "intent": dict(intent),
            }
        )

        self.history_count += 1

        if self.active_mission is None:
            self.active_mission = mission
        else:
            self.queue.append(mission)

        return FakeMission(mission)

    def submit_text(self, command):
        if "backpack" in command.lower():
            intent = {
                "intent": "FIND_OBJECT",
                "speech": "Test mission accepted.",
                "target": "backpack",
            }
        else:
            intent = {
                "intent": "TURN_LEFT",
                "speech": "Test mission accepted.",
                "target": None,
            }

        mission = self._accept_intent(
            intent,
            command=command,
        )

        return {
            "command": command,
            "intent": intent,
            "mission": mission.to_dict(),
        }

    def submit_intent(self, intent):
        return self._accept_intent(intent)

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
        print("===== COMMAND SUBMISSION =====")
        status_code, first = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "command": "Find my backpack",
            },
        )
        print(status_code, first)

        assert status_code == 202
        assert first["accepted"] is True
        assert first["submission_mode"] == "command"
        assert (
            first["mission"]["mission_type"]
            == "FIND_OBJECT"
        )
        assert first["mission"]["status"] == "ACTIVE"

        print()
        print("===== PARSED INTENT SUBMISSION =====")
        parsed_intent = {
            "intent": "TURN_LEFT",
            "speech": "Turning left.",
            "target": None,
        }

        status_code, second = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "source_text": "Turn left",
                "intent": parsed_intent,
            },
        )
        print(status_code, second)

        assert status_code == 202
        assert second["accepted"] is True
        assert (
            second["submission_mode"]
            == "parsed_intent"
        )
        assert second["intent"] == parsed_intent
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
        print("===== BOTH INPUT TYPES REJECTED =====")
        status_code, invalid_both = request_json(
            "POST",
            f"{base_url}/missions",
            {
                "command": "Turn left",
                "intent": parsed_intent,
            },
        )
        print(status_code, invalid_both)

        assert status_code == 400
        assert invalid_both["ok"] is False

        print()
        print("===== EMPTY REQUEST REJECTED =====")
        status_code, invalid_empty = request_json(
            "POST",
            f"{base_url}/missions",
            {},
        )
        print(status_code, invalid_empty)

        assert status_code == 400
        assert invalid_empty["ok"] is False

        print()
        print("PASS: command submission remains supported")
        print("PASS: parsed intents are accepted")
        print("PASS: parsed intent is queued without provider call")
        print("PASS: mission queue remains visible")
        print("PASS: ambiguous submissions are rejected")
        print("PASS: empty submissions are rejected")
        print()
        print("Runtime API parsed-intent test passed.")

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
