#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HOST = "0.0.0.0"
PORT = 8765

PROJECT_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = Path(__file__).resolve().parent / "index.html"

COGNITIVE_RUNTIME_URL = os.getenv(
    "COGNITIVE_RUNTIME_URL",
    "http://127.0.0.1:8770",
).rstrip("/")

VISION_SERVER_URL = os.getenv(
    "VISION_SERVER_URL",
    "http://127.0.0.1:8000/detections/latest",
)

ROBOT_BRIDGE_URL = os.getenv(
    "ROBOT_BRIDGE_URL",
    "http://192.168.68.127:8090",
).rstrip("/")


def request_json(
    method,
    url,
    payload=None,
    timeout=5.0,
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
            timeout=timeout,
        ) as response:
            raw_body = response.read().decode("utf-8")

            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "data": (
                    json.loads(raw_body)
                    if raw_body
                    else {}
                ),
                "error": None,
            }

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")

        try:
            response_data = json.loads(body)
        except json.JSONDecodeError:
            response_data = {
                "error": body or f"HTTP {exc.code}",
            }

        return {
            "ok": False,
            "status_code": exc.code,
            "data": response_data,
            "error": response_data.get(
                "error",
                f"HTTP {exc.code}",
            ),
        }

    except (
        urllib.error.URLError,
        TimeoutError,
    ) as exc:
        return {
            "ok": False,
            "status_code": None,
            "data": None,
            "error": str(exc),
        }

    except json.JSONDecodeError:
        return {
            "ok": False,
            "status_code": None,
            "data": None,
            "error": "Service returned invalid JSON.",
        }


class VoiceRelayHandler(BaseHTTPRequestHandler):
    server_version = "MiniPupperOperatorDashboard/1.0"

    def send_cors_headers(self):
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

    def send_json(self, status_code, payload):
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

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        if content_length <= 0:
            return {}

        raw_body = self.rfile.read(content_length)

        try:
            return json.loads(
                raw_body.decode("utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The request body must contain valid JSON."
            ) from exc

    def run_voice_command(
        self,
        command,
        execute,
    ):
        command_args = [
            sys.executable,
            "voice_command.py",
            "--text",
            command,
        ]

        if execute:
            command_args.extend(
                [
                    "--runtime",
                    "--runtime-url",
                    COGNITIVE_RUNTIME_URL,
                ]
            )

        process = subprocess.run(
            command_args,
            cwd=PROJECT_DIR,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

        return {
            "ok": process.returncode == 0,
            "mode": (
                "persistent-runtime"
                if execute
                else "dry-run"
            ),
            "executed": execute,
            "submitted_to_runtime": execute,
            "command": command,
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }

    def dashboard_status(self):
        runtime_response = request_json(
            "GET",
            f"{COGNITIVE_RUNTIME_URL}/status",
            timeout=3.0,
        )

        mission_response = request_json(
            "GET",
            f"{COGNITIVE_RUNTIME_URL}/missions",
            timeout=3.0,
        )

        vision_response = request_json(
            "GET",
            VISION_SERVER_URL,
            timeout=3.0,
        )

        robot_response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/status",
            timeout=3.0,
        )

        runtime = (
            runtime_response["data"]
            if runtime_response["ok"]
            else None
        )

        missions = (
            mission_response["data"]
            if mission_response["ok"]
            else None
        )

        vision = (
            vision_response["data"]
            if vision_response["ok"]
            else None
        )

        robot = (
            robot_response["data"]
            if robot_response["ok"]
            else None
        )

        active_mission = None
        queue = []
        last_result = None
        history_count = 0

        if missions:
            active_mission = missions.get(
                "active_mission"
            )

            queue = missions.get(
                "queue",
                [],
            )

            last_result = missions.get(
                "last_result"
            )

            history_count = missions.get(
                "history_count",
                0,
            )

        elif runtime:
            active_mission = runtime.get(
                "active_mission"
            )

            queue = runtime.get(
                "queue",
                [],
            )

            last_result = runtime.get(
                "last_result"
            )

            history_count = runtime.get(
                "history_count",
                0,
            )

        detections = (
            vision.get("detections", [])
            if vision
            else []
        )

        return {
            "ok": True,
            "service": (
                "mini_pupper_operator_dashboard"
            ),
            "relay": {
                "connected": True,
                "submission_mode": (
                    "persistent_runtime"
                ),
            },
            "runtime": {
                "connected": runtime_response["ok"],
                "running": (
                    runtime.get("running")
                    if runtime
                    else False
                ),
                "state": (
                    runtime.get("runtime_state")
                    if runtime
                    else "OFFLINE"
                ),
                "uptime_seconds": (
                    runtime.get("uptime_seconds")
                    if runtime
                    else None
                ),
                "last_error": (
                    runtime.get("last_error")
                    if runtime
                    else runtime_response["error"]
                ),
            },
            "missions": {
                "active": active_mission,
                "queue": queue,
                "queue_count": len(queue),
                "history_count": history_count,
                "last_result": last_result,
            },
            "vision": {
                "connected": vision_response["ok"],
                "camera_running": (
                    vision.get("camera_running")
                    if vision
                    else False
                ),
                "description": (
                    vision.get("description")
                    if vision
                    else "Vision server offline."
                ),
                "detection_count": len(detections),
                "detections": detections,
                "camera_url": (
                    vision.get("camera_url")
                    if vision
                    else None
                ),
                "last_error": (
                    vision.get("last_error")
                    if vision
                    else vision_response["error"]
                ),
            },
            "robot": {
                "connected": robot_response["ok"],
                "ready": bool(
                    robot
                    and robot.get("ok")
                    and robot.get("ros_ready")
                ),
                "status": (
                    robot.get("status")
                    if robot
                    else "OFFLINE"
                ),
                "ros_ready": (
                    robot.get("ros_ready")
                    if robot
                    else False
                ),
                "last_error": (
                    robot.get("ros_error")
                    if robot
                    else robot_response["error"]
                ),
            },
        }

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error": (
                            "index.html was not found"
                        ),
                    },
                )
                return

            body = HTML_FILE.read_bytes()

            self.send_response(200)
            self.send_cors_headers()

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )

            self.send_header(
                "Content-Length",
                str(len(body)),
            )

            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/status":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": (
                        "mini_pupper_browser_voice_relay"
                    ),
                    "submission_mode": (
                        "persistent_runtime"
                    ),
                    "runtime_url": (
                        COGNITIVE_RUNTIME_URL
                    ),
                    "vision_url": VISION_SERVER_URL,
                    "robot_bridge_url": (
                        ROBOT_BRIDGE_URL
                    ),
                    "project_directory": (
                        str(PROJECT_DIR)
                    ),
                },
            )
            return

        if path == "/dashboard/status":
            self.send_json(
                200,
                self.dashboard_status(),
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
        path = urlparse(self.path).path

        try:
            if path == "/command":
                request_data = self.read_json_body()

                command = str(
                    request_data.get(
                        "command",
                        "",
                    )
                ).strip()

                execute = request_data.get(
                    "execute",
                    False,
                )

                if not isinstance(execute, bool):
                    raise ValueError(
                        "The execute field must be "
                        "true or false."
                    )

                if not command:
                    raise ValueError(
                        "A non-empty command is required."
                    )

                result = self.run_voice_command(
                    command=command,
                    execute=execute,
                )

                self.send_json(
                    200 if result["ok"] else 500,
                    result,
                )
                return

            if path == "/stop":
                bridge_stop = request_json(
                    "POST",
                    f"{ROBOT_BRIDGE_URL}/stop",
                    payload={},
                    timeout=5.0,
                )

                runtime_stop = self.run_voice_command(
                    command="Stop",
                    execute=True,
                )

                success = (
                    bridge_stop["ok"]
                    and runtime_stop["ok"]
                )

                self.send_json(
                    200 if success else 500,
                    {
                        "ok": success,
                        "action": "stop",
                        "robot_bridge": bridge_stop,
                        "runtime": runtime_stop,
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

        except subprocess.TimeoutExpired:
            self.send_json(
                504,
                {
                    "ok": False,
                    "error": (
                        "The cognitive command timed out."
                    ),
                },
            )

        except (TypeError, ValueError) as exc:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": str(exc),
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

    def log_message(
        self,
        format_string,
        *args,
    ):
        print(
            f"[voice-relay] "
            f"{self.address_string()} "
            f"{format_string % args}"
        )


def main():
    server = ThreadingHTTPServer(
        (HOST, PORT),
        VoiceRelayHandler,
    )

    print("============================================")
    print(" Mini Pupper 2 Operator Dashboard")
    print("============================================")
    print(f"Project:  {PROJECT_DIR}")
    print(f"URL:      http://localhost:{PORT}")
    print(f"Runtime:  {COGNITIVE_RUNTIME_URL}")
    print(f"Vision:   {VISION_SERVER_URL}")
    print(f"Robot:    {ROBOT_BRIDGE_URL}")
    print()
    print("Leave this terminal running.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print("Stopping operator dashboard.")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()
