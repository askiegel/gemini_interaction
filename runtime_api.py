#!/usr/bin/env python3

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from runtime import CognitiveRuntime


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770


class RuntimeAPIHandler(BaseHTTPRequestHandler):
    """
    HTTP interface for submitting missions to the persistent cognitive runtime.

    Endpoints:

        GET  /health
        GET  /status
        GET  /missions
        POST /missions
    """

    server_version = "MiniPupperRuntimeAPI/1.0"

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

    def send_json(self, status_code: int, payload: Dict[str, Any]):
        body = json.dumps(
            payload,
            indent=2,
            default=str,
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_cors_headers()
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        content_length = int(
            self.headers.get("Content-Length", "0")
        )

        if content_length <= 0:
            return {}

        raw_body = self.rfile.read(content_length)

        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The request body must contain valid JSON."
            ) from exc

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "mini_pupper_cognitive_runtime_api",
                    "runtime_running": self.server.runtime.running,
                },
            )
            return

        if path == "/status":
            self.send_json(
                200,
                self.server.runtime.get_status(),
            )
            return

        if path == "/missions":
            runtime_status = self.server.runtime.get_status()

            self.send_json(
                200,
                {
                    "ok": True,
                    "active_mission": runtime_status.get(
                        "active_mission"
                    ),
                    "queue": runtime_status.get("queue", []),
                    "history_count": runtime_status.get(
                        "history_count",
                        0,
                    ),
                    "last_result": runtime_status.get(
                        "last_result"
                    ),
                    "last_error": runtime_status.get(
                        "last_error"
                    ),
                },
            )
            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "Not found.",
            },
        )

    def do_POST(self):
        path = urlparse(self.path).path

        if path != "/missions":
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found.",
                },
            )
            return

        try:
            request_data = self.read_json_body()

            if not isinstance(request_data, dict):
                raise ValueError(
                    "The request body must be a JSON object."
                )

            command = str(
                request_data.get("command", "")
            ).strip()

            if not command:
                raise ValueError(
                    "A non-empty command is required."
                )

            submission = self.server.runtime.submit_text(
                command
            )

            self.send_json(
                202,
                {
                    "ok": True,
                    "accepted": True,
                    "command": command,
                    "intent": submission["intent"],
                    "mission": submission["mission"],
                    "runtime": self.server.runtime.get_status(),
                },
            )

        except ValueError as exc:
            self.send_json(
                400,
                {
                    "ok": False,
                    "accepted": False,
                    "error": str(exc),
                },
            )

        except Exception as exc:
            self.send_json(
                500,
                {
                    "ok": False,
                    "accepted": False,
                    "error": str(exc),
                },
            )

    def log_message(self, format_string, *args):
        print(
            f"[runtime-api] {self.address_string()} "
            f"{format_string % args}"
        )


class CognitiveRuntimeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        runtime,
    ):
        super().__init__(
            server_address,
            RuntimeAPIHandler,
        )
        self.runtime = runtime


def create_server(
    runtime,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
):
    """
    Create the runtime API server.

    Passing port=0 allows the operating system to select an available port,
    which is useful for offline tests.
    """
    return CognitiveRuntimeHTTPServer(
        (host, int(port)),
        runtime,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the persistent Mini Pupper 2 cognitive runtime "
            "with its HTTP mission-submission API."
        )
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"HTTP bind address. Default: {DEFAULT_HOST}",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"HTTP port. Default: {DEFAULT_PORT}",
    )

    args = parser.parse_args()

    runtime = CognitiveRuntime()

    runtime_thread = threading.Thread(
        target=runtime.run_forever,
        name="cognitive-runtime",
        daemon=True,
    )

    runtime_thread.start()

    server = create_server(
        runtime=runtime,
        host=args.host,
        port=args.port,
    )

    actual_host, actual_port = server.server_address

    print("============================================")
    print(" Mini Pupper 2 Cognitive Runtime API")
    print("============================================")
    print(f"URL:      http://{actual_host}:{actual_port}")
    print("Health:   GET  /health")
    print("Status:   GET  /status")
    print("Missions: GET  /missions")
    print("Submit:   POST /missions")
    print()
    print("Example:")
    print(
        "curl -X POST "
        f"http://{actual_host}:{actual_port}/missions "
        "-H 'Content-Type: application/json' "
        "-d '{\"command\":\"Find my backpack\"}'"
    )
    print()
    print("Leave this terminal running.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print("Stopping Cognitive Runtime API.")

    finally:
        server.server_close()
        runtime.stop()
        runtime_thread.join(timeout=5.0)

        print("Cognitive Runtime API stopped.")


if __name__ == "__main__":
    main()
