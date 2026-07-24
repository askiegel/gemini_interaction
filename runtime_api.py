#!/usr/bin/env python3

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from diagnostics import DiagnosticsManager
from config.config_manager import (
    ConfigurationError,
    ConfigurationManager,
)
from runtime import CognitiveRuntime


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770


class RuntimeAPIHandler(BaseHTTPRequestHandler):
    """
    HTTP interface for the persistent cognitive runtime.

    Endpoints:

        GET  /health
        GET  /status
        GET  /missions
        GET  /config
        GET  /diagnostics
        PUT  /config
        POST /missions

    POST /missions accepts either:

        {"command": "Find my backpack"}

    or a provider-produced intent:

        {
          "intent": {
            "intent": "FIND_OBJECT",
            "speech": "Okay, I'll look for your backpack.",
            "target": "backpack"
          }
        }

    The parsed-intent form prevents voice_command.py and the runtime from
    invoking the AI provider twice for the same spoken command.
    """

    server_version = "MiniPupperRuntimeAPI/1.2"

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, OPTIONS",
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

        if path == "/config":
            self.send_json(
                200,
                {
                    "ok": True,
                    "config": self.server.config_manager.get_config(),
                },
            )
            return

        if path == "/diagnostics":
            try:
                self.send_json(200, self.server.diagnostics_manager.collect())
            except Exception as exc:
                self.send_json(500, {"ok": False, "error": str(exc)})
            return

        if path == "/mission-history":
            try:
                history = self.server.runtime.mission_manager.get_history()
                history = list(reversed(history))
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "count": len(history),
                        "missions": history,
                    },
                )
            except Exception as exc:
                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
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

    def do_PUT(self):
        path = urlparse(self.path).path

        if path != "/config":
            self.send_json(
                404,
                {"ok": False, "error": "Not found."},
            )
            return

        try:
            request_data = self.read_json_body()
            if not isinstance(request_data, dict):
                raise ConfigurationError(
                    "The request body must be a JSON object."
                )

            config_payload = request_data.get(
                "config",
                request_data,
            )
            updated = self.server.config_manager.update_config(
                config_payload
            )
            self.send_json(
                200,
                {
                    "ok": True,
                    "saved": True,
                    "config": updated,
                },
            )
        except (ConfigurationError, TypeError, ValueError) as exc:
            self.send_json(
                400,
                {
                    "ok": False,
                    "saved": False,
                    "error": str(exc),
                },
            )
        except Exception as exc:
            self.send_json(
                500,
                {
                    "ok": False,
                    "saved": False,
                    "error": str(exc),
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

            command_value = request_data.get("command")
            intent_value = request_data.get("intent")

            has_command = (
                isinstance(command_value, str)
                and bool(command_value.strip())
            )

            has_intent = isinstance(intent_value, dict)

            if has_command and has_intent:
                raise ValueError(
                    "Provide either command or intent, not both."
                )

            if not has_command and not has_intent:
                raise ValueError(
                    "A non-empty command or parsed intent is required."
                )

            if has_intent:
                parsed_intent = dict(intent_value)

                intent_name = str(
                    parsed_intent.get("intent", "")
                ).strip()

                if not intent_name:
                    raise ValueError(
                        "The parsed intent must include an intent name."
                    )

                mission = self.server.runtime.submit_intent(
                    parsed_intent
                )

                submission_mode = "parsed_intent"
                command = str(
                    request_data.get("source_text", "")
                ).strip()

                submission = {
                    "command": command or None,
                    "intent": parsed_intent,
                    "mission": mission.to_dict(),
                }

            else:
                command = command_value.strip()

                submission = self.server.runtime.submit_text(
                    command
                )

                submission_mode = "command"

            self.send_json(
                202,
                {
                    "ok": True,
                    "accepted": True,
                    "submission_mode": submission_mode,
                    "command": submission.get("command"),
                    "intent": submission["intent"],
                    "mission": submission["mission"],
                    "runtime": self.server.runtime.get_status(),
                },
            )

        except (TypeError, ValueError) as exc:
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
        config_manager=None,
    ):
        super().__init__(
            server_address,
            RuntimeAPIHandler,
        )

        self.runtime = runtime
        self.config_manager = (
            config_manager or ConfigurationManager()
        )
        self.diagnostics_manager = DiagnosticsManager(
            runtime=self.runtime,
            config_manager=self.config_manager,
        )


def create_server(
    runtime,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    config_manager=None,
):
    """
    Create the runtime API server.

    Passing port=0 allows the operating system to select an available port,
    which is useful for offline tests.
    """
    return CognitiveRuntimeHTTPServer(
        (host, int(port)),
        runtime,
        config_manager=config_manager,
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
    print("Config:   GET  /config")
    print("Diag:     GET  /diagnostics")
    print("Update:   PUT  /config")
    print("Submit:   POST /missions")
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
