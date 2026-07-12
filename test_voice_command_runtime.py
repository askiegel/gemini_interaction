#!/usr/bin/env python3

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from voice_command import submit_intent_to_runtime


class TestHandler(BaseHTTPRequestHandler):
    received_payload = None

    def do_POST(self):
        content_length = int(
            self.headers.get("Content-Length", "0")
        )

        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))

        TestHandler.received_payload = payload

        response_payload = {
            "ok": True,
            "accepted": True,
            "submission_mode": "parsed_intent",
            "command": payload.get("source_text"),
            "intent": payload.get("intent"),
            "mission": {
                "mission_id": "mission-test-runtime",
                "mission_type": "FIND_OBJECT",
                "status": "ACTIVE",
                "target": "backpack",
                "speech": "Looking for your backpack.",
                "created_at": "2026-07-12T12:00:00",
                "started_at": "2026-07-12T12:00:00",
                "completed_at": None,
                "priority": 6,
                "source": "cognitive",
            },
        }

        body = json.dumps(
            response_payload
        ).encode("utf-8")

        self.send_response(202)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        del format_string
        del args


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        TestHandler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    server_thread.start()

    host, port = server.server_address

    intent = {
        "intent": "FIND_OBJECT",
        "speech": "Looking for your backpack.",
        "target": "backpack",
    }

    try:
        result = submit_intent_to_runtime(
            user_text="Find my backpack",
            intent=intent,
            runtime_url=f"http://{host}:{port}",
        )

        print("===== RECEIVED PAYLOAD =====")
        print(TestHandler.received_payload)

        print()
        print("===== RUNTIME RESPONSE =====")
        print(result)

        assert TestHandler.received_payload == {
            "source_text": "Find my backpack",
            "intent": intent,
        }

        assert result["ok"] is True
        assert result["accepted"] is True
        assert (
            result["submission_mode"]
            == "parsed_intent"
        )

        assert (
            result["mission"]["mission_type"]
            == "FIND_OBJECT"
        )

        print()
        print("PASS: voice command submits parsed intent")
        print("PASS: source speech is preserved")
        print("PASS: runtime mission response is returned")
        print()
        print("Voice command runtime submission test passed.")

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
