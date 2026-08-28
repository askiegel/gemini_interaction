#!/usr/bin/env python3

import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from config.config_manager import ConfigurationManager
from conversation_manager import ConversationError
from conversation_service import create_conversation_service


HOST = "0.0.0.0"
PORT = 8765

HTML_FILE = Path(__file__).resolve().parent / "index.html"
OPERATOR_CSS_FILE = (
    Path(__file__).resolve().parent
    / "operator_console.css"
)
OPERATOR_JS_FILE = (
    Path(__file__).resolve().parent
    / "operator_console.js"
)

COGNITIVE_RUNTIME_URL = os.getenv(
    "COGNITIVE_RUNTIME_URL",
    "http://127.0.0.1:8770",
).rstrip("/")

VISION_SERVER_URL = os.getenv(
    "VISION_SERVER_URL",
    "http://127.0.0.1:8000/detections/latest",
)

ROBOT_BRIDGE_URL = (
    ConfigurationManager()
    .robot_bridge_url
    .rstrip("/")
)


_CONVERSATION_SERVICE = None
_CONVERSATION_SERVICE_LOCK = Lock()
_ROBOT_SPEECH_SUBMITTER = None


def get_conversation_service():
    """
    Return the persistent browser ConversationService.

    The service is created only when the first conversational request arrives.
    Keeping one instance alive preserves ConversationManager history across
    browser requests while the Voice Relay process is running.
    """
    global _CONVERSATION_SERVICE

    if _CONVERSATION_SERVICE is not None:
        return _CONVERSATION_SERVICE

    with _CONVERSATION_SERVICE_LOCK:
        if _CONVERSATION_SERVICE is None:
            _CONVERSATION_SERVICE = create_conversation_service(
                runtime_url=COGNITIVE_RUNTIME_URL,
            )

    return _CONVERSATION_SERVICE


def set_conversation_service_for_testing(service):
    """
    Replace the persistent service for an offline test.

    Production code does not call this function.
    """
    global _CONVERSATION_SERVICE

    with _CONVERSATION_SERVICE_LOCK:
        _CONVERSATION_SERVICE = service


def set_robot_speech_submitter_for_testing(
    submitter,
):
    """
    Replace Pupper speech delivery for an offline test.

    Production code does not call this function.
    """
    global _ROBOT_SPEECH_SUBMITTER
    _ROBOT_SPEECH_SUBMITTER = submitter


def submit_robot_speech(reply):
    normalized_reply = (
        str(reply or "").strip()
    )

    if not normalized_reply:
        return {
            "ok": False,
            "destination": "mini_pupper",
            "fallback_required": False,
            "skipped": True,
            "error": "The conversational reply was empty.",
        }

    try:
        if _ROBOT_SPEECH_SUBMITTER is not None:
            raw_result = _ROBOT_SPEECH_SUBMITTER(
                normalized_reply
            )
        else:
            raw_result = request_json(
                "POST",
                f"{ROBOT_BRIDGE_URL}/speak",
                payload={
                    "text": normalized_reply,
                },
                timeout=35.0,
            )
    except Exception as exc:
        return {
            "ok": False,
            "destination": "mini_pupper",
            "fallback_required": True,
            "skipped": False,
            "error": str(exc),
        }

    if not isinstance(raw_result, dict):
        return {
            "ok": False,
            "destination": "mini_pupper",
            "fallback_required": True,
            "skipped": False,
            "error": (
                "Robot speech submitter returned "
                "an invalid response."
            ),
        }

    if "status_code" in raw_result:
        bridge_response = raw_result.get("data")
        request_ok = bool(raw_result.get("ok"))
        request_error = raw_result.get("error")
    else:
        bridge_response = raw_result
        request_ok = bool(raw_result.get("ok"))
        request_error = raw_result.get("error")

    bridge_ok = bool(
        isinstance(bridge_response, dict)
        and bridge_response.get("ok")
    )
    delivered = request_ok and bridge_ok

    return {
        "ok": delivered,
        "destination": "mini_pupper",
        "fallback_required": not delivered,
        "skipped": False,
        "error": (
            None
            if delivered
            else (
                request_error
                or (
                    bridge_response.get("error")
                    if isinstance(
                        bridge_response,
                        dict,
                    )
                    else None
                )
                or "Mini Pupper speech delivery failed."
            )
        ),
        "bridge_response": bridge_response,
    }


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


_TONY2_MAPPING_RUNTIME = None


def get_tony2_mapping_runtime():
    """Return the single local Tony2 mapping owner."""

    global _TONY2_MAPPING_RUNTIME

    if _TONY2_MAPPING_RUNTIME is None:
        from tony2_mapping_runtime import (
            Tony2MappingRuntime,
        )

        _TONY2_MAPPING_RUNTIME = (
            Tony2MappingRuntime()
        )

    return _TONY2_MAPPING_RUNTIME


class VoiceRelayHandler(BaseHTTPRequestHandler):
    server_version = "MiniPupperOperatorDashboard/1.0"

    def send_cors_headers(self):
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, OPTIONS",
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

    def lidar_status(self):
        """
        Proxy transient LiDAR telemetry without making it authoritative.

        The browser remains isolated from ROS 2 and from direct robot network
        discovery. The World Model remains the cognitive source of truth.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/lidar",
            timeout=3.0,
        )

        if response["ok"] and isinstance(
            response["data"],
            dict,
        ):
            return 200, response["data"]

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "service": "mini_pupper_operator_dashboard",
                "error": (
                    response["error"]
                    or "Mayday LiDAR telemetry is unavailable."
                ),
            },
        )

    def localization_control_status(self):
        """Proxy Robot Bridge localization ownership state."""
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/localization/status",
            timeout=5.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Localization control is unavailable."
                ),
            },
        )

    def localization_control_action(self, action):
        """Forward only fixed guarded start or stop actions."""
        if action not in ("start", "stop"):
            return 400, {
                "ok": False,
                "error": "Unsupported localization action.",
            }

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/localization/{action}",
            timeout=20.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or f"Localization {action} failed."
                ),
            },
        )

    def mapping_control_status(self):
        """Proxy guarded mapping ownership and safety state."""
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/mapping/status",
            timeout=5.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Mapping control is unavailable."
                ),
            },
        )

    def mapping_control_action(self, action):
        """
        Forward one fixed guarded mapping action.

        No browser-provided command, map path, launch parameter, motion
        value, or candidate name is accepted.
        """
        routes = {
            "start": ("start", 20.0),
            "stop": ("stop", 20.0),
            "save-candidate": (
                "save-candidate",
                180.0,
            ),
        }

        if action not in routes:
            return 400, {
                "ok": False,
                "error": "Unsupported mapping action.",
            }

        route, timeout = routes[action]
        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/mapping/{route}",
            timeout=timeout,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or f"Mapping {action} failed."
                ),
            },
        )

    def mayday_mapping_navigation_active(self):
        """
        Fail closed if legacy mapping-navigation state
        cannot be verified before changing live mapping.
        """

        response = request_json(
            "GET",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/mapping-navigation/status"
            ),
            timeout=5.0,
        )

        data = response.get("data")

        if (
            not response.get("ok")
            or not isinstance(data, dict)
        ):
            return (
                True,
                "Mayday mapping-navigation state "
                "is unavailable.",
            )

        navigation = data.get(
            "mapping_navigation",
            data,
        )

        if not isinstance(navigation, dict):
            return (
                True,
                "Mayday mapping-navigation state "
                "is invalid.",
            )

        active = any(
            navigation.get(field) is True
            for field in (
                "running",
                "execution_enabled",
                "navigation_enabled",
                "controller_enabled",
                "navigator_enabled",
                "goal_submission_enabled",
            )
        )

        return active, None

    def ensure_mayday_stationary(self):
        """
        Publish safety zero and verify Mayday reports
        zero commanded motion before a mapping action.
        """

        stop_response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/stop",
            timeout=10.0,
        )

        stop_data = stop_response.get("data")

        if (
            not stop_response.get("ok")
            or not isinstance(stop_data, dict)
            or stop_data.get("ok") is False
        ):
            return (
                "Mayday safety STOP could not "
                "be confirmed."
            )

        status_response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/status",
            timeout=5.0,
        )

        status = status_response.get("data")

        if (
            not status_response.get("ok")
            or not isinstance(status, dict)
        ):
            return (
                "Mayday stationary state "
                "could not be verified."
            )

        motion = status.get("motion")

        if not isinstance(motion, dict):
            return (
                "Mayday motion telemetry "
                "is unavailable."
            )

        try:
            linear_x = float(
                motion.get("linear_x")
            )
            angular_z = float(
                motion.get("angular_z")
            )
        except (
            TypeError,
            ValueError,
        ):
            return (
                "Mayday motion telemetry "
                "is invalid."
            )

        if (
            linear_x != 0.0
            or angular_z != 0.0
            or motion.get("streaming") is not False
        ):
            return (
                "Mayday is not in a verified "
                "stopped state."
            )

        return None

    def tony2_mapping_control_status(self):
        """Return Tony2 local Cartographer ownership."""

        runtime = get_tony2_mapping_runtime()

        try:
            runtime.ensure_probe()
            mapping = runtime.status()
        except Exception as exc:
            return 503, {
                "ok": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "error": str(exc),
            }

        navigation_active, navigation_error = (
            self.mayday_mapping_navigation_active()
        )

        mapping["navigation_active"] = (
            navigation_active
        )

        mapping[
            "navigation_status_available"
        ] = navigation_error is None

        if navigation_error is not None:
            mapping[
                "navigation_status_error"
            ] = navigation_error

        healthy = (
            mapping.get("state") != "ERROR"
        )

        return (
            200 if healthy else 503,
            {
                "ok": healthy,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "mapping": mapping,
            },
        )

    def tony2_mapping_control_action(self, action):
        """
        Start, stop, or reset Tony2 mapping only.

        Browser requests cannot supply a ROS command,
        launch path, map path, velocity, or parameter.
        """

        if action not in {
            "start",
            "stop",
            "reset",
        }:
            return 400, {
                "ok": False,
                "error": (
                    "Unsupported Tony2 "
                    "mapping action."
                ),
            }

        navigation_active, navigation_error = (
            self.mayday_mapping_navigation_active()
        )

        if navigation_error is not None:
            return 503, {
                "ok": False,
                "error": navigation_error,
            }

        if navigation_active:
            return 409, {
                "ok": False,
                "error": (
                    "Stop guarded navigation before "
                    "changing the live mapping session."
                ),
            }

        stationary_error = (
            self.ensure_mayday_stationary()
        )

        if stationary_error is not None:
            return 503, {
                "ok": False,
                "error": stationary_error,
            }

        runtime = get_tony2_mapping_runtime()

        try:
            if action == "start":
                result = runtime.start()
            elif action == "stop":
                result = runtime.stop()
            else:
                result = runtime.reset()

        except Exception as exc:
            return 503, {
                "ok": False,
                "error": str(exc),
                "mapping": runtime.status(),
            }

        mapping = result.get(
            "mapping",
            runtime.status(),
        )

        mapping["navigation_active"] = False
        mapping[
            "navigation_status_available"
        ] = True

        return 200, {
            "ok": True,
            "action": result.get(
                "action",
                action.upper(),
            ),
            "mapping": mapping,
        }

    def tony2_live_mapping_map_status(self):
        """Return Tony2's current live /map snapshot."""

        runtime = get_tony2_mapping_runtime()

        try:
            runtime.ensure_probe()
            return runtime.live_map_status()

        except Exception as exc:
            return 503, {
                "ok": False,
                "runtime_active": False,
                "read_only": True,
                "authoritative": False,
                "service": (
                    "mini_pupper_operator_dashboard"
                ),
                "telemetry": {
                    "available": False,
                    "status": (
                        "MAPPING_MAP_UNAVAILABLE"
                    ),
                    "map": None,
                },
                "error": str(exc),
            }

    def planning_control_status(self):
        """Proxy guarded planning ownership and safety state."""
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/planning/status",
            timeout=5.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Planning control is unavailable."
                ),
            },
        )

    def planning_control_action(self, action):
        """Forward only fixed planning start or stop actions."""
        if action not in ("start", "stop"):
            return 400, {
                "ok": False,
                "error": "Unsupported planning action.",
            }

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/planning/{action}",
            timeout=20.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or f"Planning {action} failed."
                ),
            },
        )

    def planning_initialize_localization(self):
        """
        Request only Robot Bridge's fixed stationary AMCL sequence.

        No browser payload, pose, ROS service, topic, frame, planner,
        controller, command, or motion value is accepted.
        """
        response = request_json(
            "POST",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/planning/initialize-localization"
            ),
            payload={},
            timeout=75.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or (
                        "Planning localization "
                        "initialization failed."
                    )
                ),
            },
        )

    def planning_refresh_localization(self):
        """
        Request only Robot Bridge's fixed stationary AMCL refresh.

        No browser payload, pose, ROS service, topic, frame, planner,
        controller, command, or motion value is accepted.
        """
        response = request_json(
            "POST",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/planning/refresh-localization"
            ),
            payload={},
            timeout=20.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or (
                        "Planning localization "
                        "refresh failed."
                    )
                ),
            },
        )

    def planning_compute_path(self, payload):
        """
        Forward one finite map-frame goal for read-only planning.

        The browser cannot choose a ROS action, frame, planner plugin,
        start pose, launch parameter, command, or map path.
        """
        if not isinstance(payload, dict):
            return 400, {
                "ok": False,
                "error": "A JSON request body is required.",
            }

        if set(payload) != {
            "goal_x",
            "goal_y",
            "goal_yaw",
        }:
            return 400, {
                "ok": False,
                "error": (
                    "Exactly goal_x, goal_y, and goal_yaw "
                    "must be supplied."
                ),
            }

        normalized = {}

        for key in (
            "goal_x",
            "goal_y",
            "goal_yaw",
        ):
            value = payload.get(key)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be a finite number.",
                }

            value = float(value)

            if not math.isfinite(value):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be finite.",
                }

            normalized[key] = value

        response = request_json(
            "POST",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/planning/compute-path"
            ),
            payload=normalized,
            timeout=35.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Path computation failed."
                ),
            },
        )

    def navigation_control_status(self):
        """Proxy guarded navigation ownership and limits."""
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/navigation/status",
            timeout=5.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Navigation control is unavailable."
                ),
            },
        )

    def navigation_control_action(self, action):
        """Forward only fixed navigation start or stop actions."""
        if action not in ("start", "stop"):
            return 400, {
                "ok": False,
                "error": "Unsupported navigation action.",
            }

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/navigation/{action}",
            timeout=30.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or f"Navigation {action} failed."
                ),
            },
        )

    def navigation_initialize_localization(self):
        """
        Request Robot Bridge's fixed navigation AMCL sequence.

        No browser-selected service, topic, pose, frame, launch
        parameter, controller, velocity, or motion value is accepted.
        """
        response = request_json(
            "POST",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/navigation/initialize-localization"
            ),
            payload={},
            timeout=75.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or (
                        "Navigation localization "
                        "initialization failed."
                    )
                ),
            },
        )

    def navigation_goal(self, payload):
        """
        Forward one exact finite map-frame goal.

        Robot Bridge remains authoritative for the live pose,
        fixed distance bound, execution timeout, cancellation, and STOP.
        """
        if not isinstance(payload, dict):
            return 400, {
                "ok": False,
                "error": "A JSON request body is required.",
            }

        required = {
            "goal_x",
            "goal_y",
            "goal_yaw",
        }

        if set(payload) != required:
            return 400, {
                "ok": False,
                "error": (
                    "Exactly goal_x, goal_y, and goal_yaw "
                    "must be supplied."
                ),
            }

        normalized = {}

        for key in sorted(required):
            value = payload.get(key)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be a finite number.",
                }

            value = float(value)

            if not math.isfinite(value):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be finite.",
                }

            normalized[key] = value

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/navigation/goal",
            payload=normalized,
            timeout=25.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Guarded navigation goal failed."
                ),
            },
        )


    def mapping_pose_status(self):
        """
        Proxy Mayday's current Cartographer map-frame base_link pose.

        This endpoint is read-only and cannot initialize localization,
        publish transforms, start navigation, or command motion.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/mapping-pose",
            timeout=5.0,
        )

        if isinstance(response["data"], dict):
            return (
                response["status_code"] or 503,
                response["data"],
            )

        return (
            response["status_code"] or 503,
            {
                "ok": False,
                "runtime_active": False,
                "service": "mini_pupper_operator_dashboard",
                "telemetry": {
                    "available": False,
                    "status": "MAPPING_POSE_UNAVAILABLE",
                    "pose": None,
                },
                "error": (
                    response["error"]
                    or "Mayday live mapping pose is unavailable."
                ),
            },
        )

    def mapping_navigation_status(self):
        """
        Return Robot Bridge's guarded mapping-navigation status.

        Robot Bridge remains authoritative for navigation ownership,
        controller state, navigator state, and goal-submission readiness.
        """
        response = request_json(
            "GET",
            (
                f"{ROBOT_BRIDGE_URL}"
                "/mapping-navigation/status"
            ),
            timeout=5.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or (
                        "Guarded live-mapping navigation "
                        "status is unavailable."
                    )
                ),
            },
        )

    def mapping_navigation_control_action(self, action):
        """
        Start or stop only Robot Bridge's guarded mapping-navigation mode.

        The browser cannot select a launch file, ROS node, controller,
        velocity, topic, frame, behavior tree, timeout, or distance limit.
        """
        if action not in {"start", "stop"}:
            return 400, {
                "ok": False,
                "error": "Unsupported mapping-navigation action.",
            }

        response = request_json(
            "POST",
            (
                f"{ROBOT_BRIDGE_URL}"
                f"/mapping-navigation/{action}"
            ),
            timeout=35.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or (
                        "Guarded live-mapping navigation "
                        f"{action} failed."
                    )
                ),
            },
        )

    def mapping_navigation_goal(self, payload):
        """
        Forward one finite map-frame goal to mapping-navigation only.

        Robot Bridge remains authoritative for current pose validation,
        the 0.50-meter maximum, 25-second execution limit, cancellation,
        behavior tree selection, controller ownership, and safety STOP.
        """
        if not isinstance(payload, dict):
            return 400, {
                "ok": False,
                "error": "A JSON request body is required.",
            }

        required = {
            "goal_x",
            "goal_y",
            "goal_yaw",
        }

        if set(payload) != required:
            return 400, {
                "ok": False,
                "error": (
                    "Exactly goal_x, goal_y, and goal_yaw "
                    "must be supplied."
                ),
            }

        normalized = {}

        for key in sorted(required):
            value = payload.get(key)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be a finite number.",
                }

            value = float(value)

            if not math.isfinite(value):
                return 400, {
                    "ok": False,
                    "error": f"{key} must be finite.",
                }

            normalized[key] = value

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/mapping-navigation/goal",
            payload=normalized,
            timeout=35.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Guarded live-mapping goal failed."
                ),
            },
        )

    def localization_status(self):
        """
        Proxy guarded localization pose as read-only presentation data.

        Robot Bridge remains responsible for determining whether AMCL is
        active. A stopped localization runtime cannot expose a cached pose.
        This proxy cannot start localization, publish an initial pose,
        execute a navigation goal, or command robot motion.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/localization",
            timeout=5.0,
        )

        if isinstance(response["data"], dict):
            return (
                response["status_code"] or 503,
                response["data"],
            )

        return (
            response["status_code"] or 503,
            {
                "ok": False,
                "runtime_active": False,
                "service": "mini_pupper_operator_dashboard",
                "telemetry": {
                    "available": False,
                    "pose": None,
                    "status": "LOCALIZATION_UNAVAILABLE",
                },
                "error": (
                    response["error"]
                    or "Mayday localization telemetry is unavailable."
                ),
            },
        )

    def map_status(self):
        """
        Proxy the validated saved map as read-only presentation data.

        The browser remains isolated from ROS 2 and direct robot discovery.
        This proxy cannot start localization, planning, or robot motion.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/map",
            timeout=5.0,
        )

        if response["ok"] and isinstance(
            response["data"],
            dict,
        ):
            return 200, response["data"]

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "service": "mini_pupper_operator_dashboard",
                "error": (
                    response["error"]
                    or "Mayday saved-map telemetry is unavailable."
                ),
            },
        )

    def live_mapping_map_status(self):
        """
        Proxy the current Cartographer grid as read-only data.

        Robot Bridge owns mapping runtime state and stale-grid clearing.
        This proxy cannot start, stop, save, promote, navigate, or move.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/mapping-map",
            timeout=10.0,
        )

        if isinstance(response["data"], dict):
            return (
                response["status_code"] or 503,
                response["data"],
            )

        return (
            response["status_code"] or 503,
            {
                "ok": False,
                "runtime_active": False,
                "read_only": True,
                "authoritative": False,
                "service": "mini_pupper_operator_dashboard",
                "telemetry": {
                    "available": False,
                    "status": "MAPPING_MAP_UNAVAILABLE",
                    "map": None,
                },
                "error": (
                    response["error"]
                    or "Mayday live mapping telemetry is unavailable."
                ),
            },
        )

    def candidate_map_status(self):
        """
        Proxy candidate maps as read-only review data.

        Candidate classification and occupancy parsing remain owned by the
        Robot Bridge. This proxy cannot promote, delete, save, or modify a
        candidate or the validated map.
        """
        response = request_json(
            "GET",
            f"{ROBOT_BRIDGE_URL}/telemetry/map-candidates",
            timeout=10.0,
        )

        if response["ok"] and isinstance(
            response["data"],
            dict,
        ):
            return 200, response["data"]

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "service": "mini_pupper_operator_dashboard",
                "error": (
                    response["error"]
                    or "Mayday candidate-map review is unavailable."
                ),
            },
        )

    def candidate_map_promote(self, payload):
        """
        Promote one explicitly selected review-ready candidate.

        The Robot Bridge remains responsible for candidate validation,
        runtime exclusion, backup creation, replacement, verification,
        telemetry reload, and rollback.
        """
        confirmation = "PROMOTE REVIEWED CANDIDATE"

        if not isinstance(payload, dict):
            return 400, {
                "ok": False,
                "error": "A JSON request body is required.",
            }

        if set(payload) != {
            "candidate_name",
            "confirmation",
        }:
            return 400, {
                "ok": False,
                "error": (
                    "Exactly candidate_name and confirmation "
                    "must be supplied."
                ),
            }

        candidate_name = payload.get("candidate_name")
        supplied_confirmation = payload.get(
            "confirmation"
        )

        if (
            not isinstance(candidate_name, str)
            or not candidate_name.startswith(
                "mayday_map_candidate_"
            )
        ):
            return 400, {
                "ok": False,
                "error": "Candidate name is invalid.",
            }

        if supplied_confirmation != confirmation:
            return 400, {
                "ok": False,
                "error": (
                    "Explicit promotion confirmation is "
                    "required."
                ),
            }

        response = request_json(
            "POST",
            f"{ROBOT_BRIDGE_URL}/map/promote-candidate",
            payload={
                "candidate_name": candidate_name,
                "confirmation": supplied_confirmation,
            },
            timeout=180.0,
        )

        return (
            response["status_code"] or 503,
            response["data"] or {
                "ok": False,
                "error": (
                    response["error"]
                    or "Candidate promotion failed."
                ),
            },
        )


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
                "tracking": (
                    runtime.get("tracking", {})
                    if runtime
                    else {}
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
                "timestamp": (
                    vision.get("timestamp")
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

    def relay_runtime_json(
        self,
        method,
        runtime_path,
        payload=None,
        timeout=15.0,
    ):
        """
        Forward a JSON request to the Cognitive Runtime API.

        The browser communicates only with the Voice Relay. The relay
        preserves the Runtime API status code and response body so the
        Operator Console receives the authoritative backend result.
        """
        response = request_json(
            method,
            f"{COGNITIVE_RUNTIME_URL}{runtime_path}",
            payload=payload,
            timeout=timeout,
        )

        status_code = response["status_code"] or 503

        response_payload = response["data"]

        if not isinstance(response_payload, dict):
            response_payload = {
                "ok": False,
                "error": (
                    response["error"]
                    or "Cognitive Runtime API is unavailable."
                ),
            }

        self.send_json(
            status_code,
            response_payload,
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        static_files = {
            "/operator_console.css": (
                OPERATOR_CSS_FILE,
                "text/css; charset=utf-8",
            ),
            "/operator_console.js": (
                OPERATOR_JS_FILE,
                "application/javascript; charset=utf-8",
            ),
        }

        if path in static_files:
            static_path, content_type = (
                static_files[path]
            )

            if not static_path.exists():
                self.send_json(
                    404,
                    {
                        "ok": False,
                        "error": (
                            f"{static_path.name} "
                            "was not found."
                        ),
                    },
                )
                return

            body = static_path.read_bytes()

            self.send_response(200)
            self.send_cors_headers()
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

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

        if path == "/dashboard/lidar":
            status_code, payload = self.lidar_status()
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/navigation-control":
            status_code, payload = (
                self.navigation_control_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/planning-control":
            status_code, payload = (
                self.planning_control_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/localization-control":
            status_code, payload = (
                self.localization_control_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/localization":
            status_code, payload = self.localization_status()
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/map":
            status_code, payload = self.map_status()
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-pose":
            status_code, payload = self.mapping_pose_status()
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-map":
            status_code, payload = (
                self.tony2_live_mapping_map_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-control":
            status_code, payload = (
                self.tony2_mapping_control_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-navigation-status":
            status_code, payload = (
                self.mapping_navigation_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/map-candidates":
            status_code, payload = (
                self.candidate_map_status()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/status":
            self.send_json(
                200,
                self.dashboard_status(),
            )
            return

        if path == "/dashboard/config":
            response = request_json(
                "GET",
                f"{COGNITIVE_RUNTIME_URL}/config",
                timeout=5.0,
            )

            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "error": response["error"] or "Runtime configuration service unavailable.",
                },
            )
            return

        if path == "/dashboard/world-model":
            response = request_json(
                "GET",
                f"{COGNITIVE_RUNTIME_URL}/world-model",
                timeout=8.0,
            )
            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "error": response["error"] or "World Model service unavailable.",
                },
            )
            return

        if path in (
            "/dashboard/network-status",
            "/dashboard/network",
        ):
            query = urlparse(self.path).query
            target_url = f"{COGNITIVE_RUNTIME_URL}/network-status"
            if query:
                target_url = f"{target_url}?{query}"
            response = request_json(
                "GET",
                target_url,
                timeout=12.0,
            )
            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "read_only": True,
                    "error": response["error"] or "Network visibility service unavailable.",
                },
            )
            return

        if path == "/dashboard/mission-history":
            response = request_json(
                "GET",
                f"{COGNITIVE_RUNTIME_URL}/mission-history",
                timeout=5.0,
            )
            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "error": response["error"] or "Mission history service unavailable.",
                },
            )
            return

        if path == "/dashboard/diagnostics":
            response = request_json(
                "GET",
                f"{COGNITIVE_RUNTIME_URL}/diagnostics",
                timeout=8.0,
            )
            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "error": response["error"] or "Runtime diagnostics service unavailable.",
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

    def do_PUT(self):
        path = urlparse(self.path).path

        if path != "/dashboard/config":
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found",
                },
            )
            return

        try:
            payload = self.read_json_body()

            if not isinstance(payload, dict):
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": "The request body must be a JSON object.",
                    },
                )
                return

            response = request_json(
                "PUT",
                f"{COGNITIVE_RUNTIME_URL}/config",
                payload=payload,
                timeout=5.0,
            )

            self.send_json(
                response["status_code"] or 503,
                response["data"] or {
                    "ok": False,
                    "error": response["error"] or "Runtime configuration service unavailable.",
                },
            )

        except ValueError as exc:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": str(exc),
                },
            )

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/dashboard/map-promote-candidate":
            try:
                payload = self.read_json_body()
            except ValueError as exc:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return

            status_code, response_payload = (
                self.candidate_map_promote(payload)
            )
            self.send_json(
                status_code,
                response_payload,
            )
            return

\
        if path in (
            "/dashboard/mapping-navigation-start",
            "/dashboard/mapping-navigation-stop",
        ):
            action = (
                "start"
                if path.endswith("-start")
                else "stop"
            )
            status_code, payload = (
                self.mapping_navigation_control_action(action)
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-navigation-goal":
            try:
                request_payload = self.read_json_body()
            except ValueError as exc:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return

            status_code, payload = (
                self.mapping_navigation_goal(
                    request_payload
                )
            )
            self.send_json(status_code, payload)
            return

        if path in (
            "/dashboard/navigation-start",
            "/dashboard/navigation-stop",
        ):
            action = (
                "start"
                if path.endswith("-start")
                else "stop"
            )
            status_code, payload = (
                self.navigation_control_action(action)
            )
            self.send_json(status_code, payload)
            return

        if (
            path
            == "/dashboard/navigation-initialize-localization"
        ):
            status_code, payload = (
                self.navigation_initialize_localization()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/navigation-goal":
            try:
                request_payload = self.read_json_body()
            except ValueError as exc:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return

            status_code, payload = self.navigation_goal(
                request_payload
            )
            self.send_json(status_code, payload)
            return

        if path in (
            "/dashboard/planning-start",
            "/dashboard/planning-stop",
        ):
            action = (
                "start"
                if path.endswith("-start")
                else "stop"
            )
            status_code, payload = (
                self.planning_control_action(action)
            )
            self.send_json(status_code, payload)
            return

        if (
            path
            == "/dashboard/planning-initialize-localization"
        ):
            status_code, payload = (
                self.planning_initialize_localization()
            )
            self.send_json(status_code, payload)
            return

        if (
            path
            == "/dashboard/planning-refresh-localization"
        ):
            status_code, payload = (
                self.planning_refresh_localization()
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/planning-compute-path":
            try:
                request_payload = self.read_json_body()
            except ValueError as exc:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return

            status_code, payload = (
                self.planning_compute_path(
                    request_payload
                )
            )
            self.send_json(status_code, payload)
            return

        tony2_mapping_actions = {
            "/dashboard/mapping-start": "start",
            "/dashboard/mapping-stop": "stop",
            "/dashboard/mapping-reset": "reset",
        }

        if path in tony2_mapping_actions:
            status_code, payload = (
                self.tony2_mapping_control_action(
                    tony2_mapping_actions[path]
                )
            )
            self.send_json(status_code, payload)
            return

        if path == "/dashboard/mapping-save-candidate":
            status_code, payload = (
                self.mapping_control_action(
                    "save-candidate"
                )
            )
            self.send_json(status_code, payload)
            return

        if path in (
            "/dashboard/localization-start",
            "/dashboard/localization-stop",
        ):
            action = (
                "start"
                if path.endswith("-start")
                else "stop"
            )
            status_code, payload = (
                self.localization_control_action(action)
            )
            self.send_json(status_code, payload)
            return


        if path in (
            "/network/connect",
            "/network/disconnect",
            "/network/forget",
        ):
            try:
                content_type = self.headers.get(
                    "Content-Type",
                    "",
                )

                media_type = content_type.split(
                    ";",
                    1,
                )[0].strip().lower()

                if media_type != "application/json":
                    raise ValueError(
                        "Content-Type must be application/json."
                    )

                request_data = self.read_json_body()

                if not isinstance(request_data, dict):
                    raise ValueError(
                        "The request body must be a JSON object."
                    )

                self.relay_runtime_json(
                    "POST",
                    path,
                    payload=request_data,
                    timeout=20.0,
                )

            except ValueError as exc:
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

            return


        try:
            if path == "/conversation":
                payload = self.read_json_body()

                if not isinstance(payload, dict):
                    self.send_json(
                        400,
                        {
                            "ok": False,
                            "error": (
                                "The request body must be a JSON object."
                            ),
                        },
                    )
                    return

                user_text = payload.get(
                    "text",
                    payload.get("command"),
                )

                if not isinstance(user_text, str):
                    self.send_json(
                        400,
                        {
                            "ok": False,
                            "error": (
                                "Conversation requests require a "
                                "string field named 'text' or 'command'."
                            ),
                        },
                    )
                    return

                user_text = user_text.strip()

                if not user_text:
                    self.send_json(
                        400,
                        {
                            "ok": False,
                            "error": (
                                "Conversation text cannot be empty."
                            ),
                        },
                    )
                    return

                execute = payload.get("execute", False)

                if not isinstance(execute, bool):
                    self.send_json(
                        400,
                        {
                            "ok": False,
                            "error": (
                                "The execute field must be a boolean."
                            ),
                        },
                    )
                    return

                try:
                    service = get_conversation_service()
                    result = service.process_text(
                        user_text,
                        submit_missions=execute,
                    )

                except ConversationError as exc:
                    self.send_json(
                        400,
                        {
                            "ok": False,
                            "error_type": "conversation_error",
                            "error": str(exc),
                        },
                    )
                    return

                except RuntimeError as exc:
                    self.send_json(
                        503,
                        {
                            "ok": False,
                            "error_type": "runtime_error",
                            "error": str(exc),
                        },
                    )
                    return

                except Exception as exc:
                    self.send_json(
                        500,
                        {
                            "ok": False,
                            "error_type": "internal_error",
                            "error": str(exc),
                        },
                    )
                    return

                result_data = result.to_dict()

                if (
                    result_data.get("accepted", True)
                    and not result_data.get(
                        "ignored",
                        False,
                    )
                ):
                    speech_output = submit_robot_speech(
                        result.reply
                    )
                else:
                    speech_output = {
                        "ok": False,
                        "destination": "mini_pupper",
                        "fallback_required": False,
                        "skipped": True,
                        "error": (
                            "Reply was not addressed "
                            "to this robot."
                        ),
                    }

                response = {
                    "ok": True,
                    "mode": (
                        "live"
                        if execute
                        else "dry-run"
                    ),
                    "executed": execute,
                    **result_data,
                    "speech_output": speech_output,
                }

                self.send_json(200, response)
                return

            if path == "/conversation/history":
                service = get_conversation_service()

                self.send_json(
                    200,
                    {
                        "ok": True,
                        "history": service.get_history(),
                    },
                )
                return

            if path == "/conversation/clear":
                service = get_conversation_service()
                service.clear_history()

                self.send_json(
                    200,
                    {
                        "ok": True,
                        "history_cleared": True,
                    },
                )
                return

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
