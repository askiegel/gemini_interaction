#!/usr/bin/env python3

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "0.0.0.0"
PORT = 8765
PROJECT_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = Path(__file__).resolve().parent / "index.html"


class VoiceRelayHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, status_code, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")

        self.send_response(status_code)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error": "index.html was not found",
                    },
                )
                return

            body = HTML_FILE.read_bytes()

            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/status":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "mini_pupper_browser_voice_relay",
                    "default_mode": "dry-run",
                    "live_execution_available": True,
                    "project_directory": str(PROJECT_DIR),
                },
            )
            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "Not found",
            },
        )

    def do_POST(self):
        if self.path != "/command":
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found",
                },
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            request_data = json.loads(raw_body.decode("utf-8"))

            command = str(request_data.get("command", "")).strip()
            execute = request_data.get("execute", False)

            if not isinstance(execute, bool):
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": "The execute field must be true or false.",
                    },
                )
                return

            if not command:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": "A non-empty command is required.",
                    },
                )
                return

            command_args = [
                sys.executable,
                "voice_command.py",
                "--text",
                command,
            ]

            if execute:
                command_args.append("--execute")

            process = subprocess.run(
                command_args,
                cwd=PROJECT_DIR,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

            mode = "live" if execute else "dry-run"

            self.send_json(
                200 if process.returncode == 0 else 500,
                {
                    "ok": process.returncode == 0,
                    "mode": mode,
                    "executed": execute,
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
            )

        except subprocess.TimeoutExpired:
            self.send_json(
                504,
                {
                    "ok": False,
                    "error": "The cognitive command timed out.",
                },
            )

        except json.JSONDecodeError:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": "The request body must contain valid JSON.",
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

    def log_message(self, format_string, *args):
        print(
            f"[voice-relay] {self.address_string()} "
            f"{format_string % args}"
        )


def main():
    server = ThreadingHTTPServer((HOST, PORT), VoiceRelayHandler)

    print("============================================")
    print(" Mini Pupper 2 Browser Voice Relay")
    print("============================================")
    print(f"Project: {PROJECT_DIR}")
    print(f"URL:     http://localhost:{PORT}")
    print("Default: DRY RUN")
    print()
    print("Live execution must be enabled explicitly in the browser.")
    print("Leave this terminal running.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping voice relay.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
