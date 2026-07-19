#!/usr/bin/env python3

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

from voice_relay.server import (
    VoiceRelayHandler,
    set_conversation_service_for_testing,
)


@dataclass(frozen=True)
class FakeConversationResult:
    reply: str
    decision_type: str
    mission_type: str = None
    target: str = None
    requires_confirmation: bool = False
    mission_submitted: bool = False
    mission_submission: dict = None

    def to_dict(self):
        return {
            "reply": self.reply,
            "decision_type": self.decision_type,
            "mission_type": self.mission_type,
            "target": self.target,
            "requires_confirmation": self.requires_confirmation,
            "mission_submitted": self.mission_submitted,
            "mission_submission": self.mission_submission,
        }


class FakeConversationService:
    def __init__(self):
        self.received_text = []
        self.history = []

    def process_text(
        self,
        user_text,
        submit_missions=True,
    ):
        self.received_text.append(user_text)
        self.last_submit_missions = submit_missions

        self.history.append(
            {
                "role": "user",
                "text": user_text,
            }
        )

        reply = (
            f"I remember {len(self.received_text)} "
            "browser message(s)."
        )

        self.history.append(
            {
                "role": "assistant",
                "text": reply,
            }
        )

        return FakeConversationResult(
            reply=reply,
            decision_type="CONVERSATION",
        )

    def get_history(self):
        return list(self.history)

    def clear_history(self):
        self.history.clear()


def request_json(method, url, payload=None):
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
            timeout=10.0,
        ) as response:
            raw_body = response.read().decode("utf-8")

            return (
                response.status,
                json.loads(raw_body),
            )

    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8")

        return (
            exc.code,
            json.loads(raw_body),
        )


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

    print(f"PASS: {message}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)

    print(f"PASS: {message}")


def main():
    print("==========================================")
    print("BROWSER CONVERSATION ENDPOINT TEST")
    print("==========================================")

    fake_service = FakeConversationService()
    set_conversation_service_for_testing(fake_service)

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
        print()
        print("===== FIRST CONVERSATION TURN =====")

        status, first = request_json(
            "POST",
            f"{base_url}/conversation",
            {
                "command": "Hello Mini Pupper",
                "execute": False,
            },
        )

        assert_equal(
            status,
            200,
            "first browser conversation request succeeds",
        )
        assert_true(
            first["ok"],
            "first browser response reports success",
        )
        assert_equal(
            first["reply"],
            "I remember 1 browser message(s).",
            "first conversational reply is returned",
        )
        assert_equal(
            first["decision_type"],
            "CONVERSATION",
            "decision type is returned",
        )
        assert_true(
            not first["mission_submitted"],
            "ordinary conversation submits no mission",
        )
        assert_equal(
            first["mode"],
            "dry-run",
            "execute false produces dry-run mode",
        )
        assert_true(
            not first["executed"],
            "dry-run response reports no execution",
        )
        assert_true(
            not fake_service.last_submit_missions,
            "dry-run suppresses runtime mission submission",
        )

        print()
        print("===== LIVE CONVERSATION FLAG =====")

        status, live = request_json(
            "POST",
            f"{base_url}/conversation",
            {
                "text": "Follow me",
                "execute": True,
            },
        )

        assert_equal(
            status,
            200,
            "live conversation request succeeds",
        )
        assert_equal(
            live["mode"],
            "live",
            "execute true produces live mode",
        )
        assert_true(
            live["executed"],
            "live response reports execution enabled",
        )
        assert_true(
            fake_service.last_submit_missions,
            "live mode permits validated mission submission",
        )

        print()
        print("===== SECOND CONVERSATION TURN =====")

        status, second = request_json(
            "POST",
            f"{base_url}/conversation",
            {
                "text": "What did I just say?",
            },
        )

        assert_equal(
            status,
            200,
            "second browser conversation request succeeds",
        )
        assert_equal(
            second["reply"],
            "I remember 3 browser message(s).",
            "same persistent service handles the second turn",
        )
        assert_equal(
            fake_service.received_text,
            [
                "Hello Mini Pupper",
                "Follow me",
                "What did I just say?",
            ],
            "browser text reaches the persistent service in order",
        )

        print()
        print("===== HISTORY ENDPOINT =====")

        status, history = request_json(
            "POST",
            f"{base_url}/conversation/history",
            {},
        )

        assert_equal(
            status,
            200,
            "history endpoint succeeds",
        )
        assert_equal(
            len(history["history"]),
            6,
            "history contains three user and three assistant turns",
        )

        print()
        print("===== CLEAR HISTORY =====")

        status, cleared = request_json(
            "POST",
            f"{base_url}/conversation/clear",
            {},
        )

        assert_equal(
            status,
            200,
            "clear-history endpoint succeeds",
        )
        assert_true(
            cleared["history_cleared"],
            "clear-history endpoint reports completion",
        )
        assert_equal(
            fake_service.get_history(),
            [],
            "persistent conversation history is cleared",
        )

        print()
        print("===== INPUT VALIDATION =====")

        status, missing = request_json(
            "POST",
            f"{base_url}/conversation",
            {},
        )

        assert_equal(
            status,
            400,
            "missing conversation text is rejected",
        )
        assert_true(
            not missing["ok"],
            "invalid request reports failure",
        )

        status, empty = request_json(
            "POST",
            f"{base_url}/conversation",
            {
                "text": "   ",
            },
        )

        assert_equal(
            status,
            400,
            "empty conversation text is rejected",
        )

        print()
        print("Browser Conversation Endpoint test passed.")
        print(
            "No Gemini request, Runtime API request, ROS command, "
            "or robot command was sent."
        )

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)

        set_conversation_service_for_testing(None)


if __name__ == "__main__":
    main()
