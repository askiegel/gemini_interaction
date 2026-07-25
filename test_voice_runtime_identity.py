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

        TestHandler.received_payload = json.loads(
            raw_body.decode("utf-8")
        )

        payload = TestHandler.received_payload

        response = {
            "ok": True,
            "accepted": True,
            "submission_mode": "parsed_intent",
            "robot_id": payload.get("robot_id"),
            "command": payload.get("source_text"),
            "intent": payload.get("intent"),
            "addressing": payload.get("addressing"),
            "mission": {
                "mission_id": "mission-identity-test",
                "mission_type": "FOLLOW_PERSON",
                "status": "ACTIVE",
            },
        }

        body = json.dumps(response).encode("utf-8")

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

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    host, port = server.server_address

    intent = {
        "intent": "FOLLOW_PERSON",
        "speech": "Okay, I will follow you.",
        "target": "person",
    }

    addressing = {
        "original_text": "Mayday, follow me.",
        "command_text": "follow me",
        "addressed_robot_id": "mayday",
        "broadcast": False,
    }

    try:
        result = submit_intent_to_runtime(
            user_text="Mayday, follow me.",
            intent=intent,
            runtime_url=f"http://{host}:{port}",
            robot_id="mayday",
            addressing=addressing,
        )

        expected = {
            "source_text": "Mayday, follow me.",
            "intent": intent,
            "robot_id": "mayday",
            "addressing": addressing,
        }

        if TestHandler.received_payload != expected:
            raise AssertionError(
                "Runtime payload mismatch.\n"
                f"Expected: {expected!r}\n"
                f"Actual:   {TestHandler.received_payload!r}"
            )

        if result.get("robot_id") != "mayday":
            raise AssertionError(
                "Runtime response did not preserve robot_id."
            )

        print("PASS: runtime payload includes robot_id")
        print("PASS: original source speech is preserved")
        print("PASS: addressing metadata is preserved")
        print("PASS: parsed intent is preserved")
        print()
        print("Voice Runtime Identity test passed.")

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
