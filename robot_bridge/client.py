import json
import math
import time
import urllib.request
import urllib.error
import http.client

from robot_bridge.http_response import (
    MAX_ERROR_BODY_BYTES, decode_error_body, server_error, transport_error, completed_response,
)

LOCAL_FORWARD_TIMEOUT_SECONDS = 120.0





class RobotBridgeClient:
    @staticmethod
    def _pre_transport_guard_rejection(guard):
        """Negative dispatch certificate, issued only before the HTTP call.

        A boolean callback stays the admission boundary. Its bounded decision
        explains a refusal; it never changes that admission or grants authority.
        """
        fields = {
            "reason", "guard_stage", "evaluation_monotonic_seconds",
            "control_generation_current", "motion_owner_current",
            "mission_context_current", "watchdog_not_exhausted", "action_fits_deadline",
            "source_stamp", "source_receipt_monotonic_seconds", "source_age_seconds", "source_fresh",
            "producer_session", "acquisition_sequence", "local_selection_accepted",
            "local_evaluation_monotonic_seconds",
            "local_selection_reason", "local_safety_permitted", "local_safety_reason",
            "interlock_permitted", "interlock_reason",
            "requested_action_type", "requested_direction", "final_selected_action_type",
            "final_selected_direction", "planning_lidar_sequence", "final_jit_lidar_sequence",
            "selection_rejection_class",
        }
        decision = getattr(guard, "decision", None)
        certificate = {}
        if isinstance(decision, dict):
            for key in fields:
                value = decision.get(key)
                if value is None or type(value) in (bool, int):
                    certificate[key] = value
                elif type(value) is float and math.isfinite(value):
                    certificate[key] = value
                elif type(value) is str:
                    certificate[key] = value[:160]
        certificate.setdefault("reason", "UNKNOWN_FINAL_GUARD_FAILURE")
        if not certificate.get("reason"):
            certificate["reason"] = "UNKNOWN_FINAL_GUARD_FAILURE"
        certificate.update(version=1, transport_boundary="before_POST_motion",
            transport_boundary_monotonic_seconds=time.monotonic(),
            transport_attempted=False, delivery_uncertain=False)
        return certificate

    def __init__(
        self,
        base_url=None,
        timeout=3.0,
        config_manager=None,
        forward_interlock=None,
    ):
        if base_url is not None:
            self.config_manager = config_manager
            resolved_url = base_url
        else:
            if config_manager is None:
                from config.config_manager import (
                    ConfigurationManager,
                )

                config_manager = (
                    ConfigurationManager()
                )

            self.config_manager = config_manager
            resolved_url = (
                self.config_manager.robot_bridge_url
            )
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout
        self.forward_interlock = forward_interlock

    def configure_forward_interlock(self, interlock):
        self.forward_interlock = interlock

    def _request(self, method, path, payload=None, timeout=None):
        url = f"{self.base_url}{path}"

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
            with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                try:
                    if method == "POST" and path == "/motion":
                        result = decode_error_body(response.read(MAX_ERROR_BODY_BYTES + 1))
                        if isinstance(result, dict) and any(k in result for k in ("state", "completed_progress_credit", "controller_session", "generation")):
                            return completed_response(response.status, result, payload)
                        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                            return server_error(response.status)
                        return result  # Existing legacy bounded success contract.
                    return json.loads(response.read().decode("utf-8"))
                except (ValueError, UnicodeError, RecursionError):
                    result = server_error(response.status)
                    result.update(error="bridge_response_invalid", response_origin="server_protocol_error")
                    return result
                except (OSError, http.client.HTTPException) as exc:
                    result = transport_error(exc)
                    result.update(http_status=response.status, http_response_received=True,
                                  transport_attempted=None, delivery_uncertain=True,
                                  response_origin="transport_uncertain")
                    return result

        except urllib.error.HTTPError as exc:
            try:
                body = decode_error_body(exc.read(MAX_ERROR_BODY_BYTES + 1))
            except (ValueError, UnicodeError, OSError, http.client.HTTPException, RecursionError):
                body = None
            finally:
                try:
                    exc.close()
                except OSError:
                    pass  # Closing an unreadable error stream grants no authority.
            return server_error(exc.code, body, endpoint=path)

        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError, UnicodeError, RecursionError) as exc:
            return transport_error(exc)

    def local_forward(self):
        return self._request("POST", "/local-motion/forward", {}, timeout=LOCAL_FORWARD_TIMEOUT_SECONDS)

    def status(self):
        return self._request("GET", "/status")

    def stop(self):
        result = self._request("POST", "/stop")
        if self.forward_interlock is not None:
            self.forward_interlock.stop_active()
        return result

    def motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        duration=0.25,
        streaming=False,
        watchdog_timeout=0.50,
        linear_y=0.0,
        dispatch_guard=None,
    ):
        if isinstance(linear_y, bool):
            return {"ok": False, "forwarded": False, "error": "invalid_lateral_parameters"}
        try:
            linear_y = float(linear_y)
        except (TypeError, ValueError, OverflowError):
            return {"ok": False, "forwarded": False, "error": "invalid_lateral_parameters"}
        if not math.isfinite(linear_y):
            return {"ok": False, "forwarded": False, "error": "invalid_lateral_parameters"}
        try:
            linear_x = float(linear_x)
            angular_z = float(angular_z)
            duration = float(duration)
        except (TypeError, ValueError, OverflowError):
            return {"ok": False, "forwarded": False, "error": "invalid_motion_parameters"}
        payload = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": float(duration),
        }

        if linear_y:
            if (not all(math.isfinite(v) for v in (linear_x, angular_z, float(duration)))
                    or linear_x != 0 or angular_z != 0 or abs(linear_y) > 0.08
                    or not 0 < float(duration) <= 1.00):
                return {"ok": False, "forwarded": False, "error": "invalid_lateral_parameters"}
            readiness = self.status()
            readiness = readiness if isinstance(readiness, dict) else {}
            capabilities = readiness.get("motion_capabilities")
            capabilities = capabilities if isinstance(capabilities, dict) else {}
            if (readiness.get("ok") is not True or readiness.get("ros_ready") is not True
                    or readiness.get("status") != "READY" or capabilities.get("linear_y") is not True
                    or type(capabilities.get("max_linear_y")) not in (int, float)
                    or not math.isfinite(capabilities["max_linear_y"])
                    or capabilities["max_linear_y"] < abs(linear_y)):
                return {"ok": False, "forwarded": False, "error": "bridge_lateral_support_unavailable"}
            payload["linear_y"] = linear_y

        if streaming:
            payload.update(
                {
                    "streaming": True,
                    "watchdog_timeout": float(
                        watchdog_timeout
                    ),
                }
            )

        if linear_x > 0 or linear_y != 0:
            if not streaming:
                if not math.isfinite(angular_z) or angular_z != 0.0:
                    return {
                        "ok": False,
                        "forwarded": False,
                        "error": "bounded_forward_requires_zero_angular",
                    }
                if not math.isfinite(float(duration)) or float(duration) <= 0.0:
                    return {
                        "ok": False,
                        "forwarded": False,
                        "error": "invalid_bounded_forward_duration",
                    }
            if self.forward_interlock is None:
                return {"ok": False, "forwarded": False, "error": "forward_interlock_not_configured"}
            interlock = self.forward_interlock
            try:
                generation = interlock.begin_positive_dispatch(streaming=streaming)
            except PermissionError as exc:
                return {"ok": False, "forwarded": False, "error": str(exc)}
            result = None
            transport_attempted = False
            try:
                if dispatch_guard is not None and dispatch_guard() is not True:
                    result = {"ok": False, "forwarded": False, "error": "motion_dispatch_preempted",
                        "transport_attempted": False, "delivery_uncertain": False,
                        "dispatch_guard_rejection": self._pre_transport_guard_rejection(dispatch_guard)}
                    return result
                transport_attempted = True
                result = self._request("POST", "/motion", payload)
                return result
            finally:
                dispatch_valid = interlock.finalize_positive_dispatch(
                    generation,
                    result,
                )
                if not streaming:
                    interlock.stop_active()
                    outcome_reader = getattr(
                        interlock,
                        "dispatch_outcome",
                        None,
                    )
                    outcome = (
                        outcome_reader(generation)
                        if dispatch_valid is False
                        and callable(outcome_reader)
                        else None
                    )
                    if dispatch_valid is False and transport_attempted and isinstance(result, dict):
                        transport_result = dict(result)
                        result.clear()
                        result.update(transport_result)
                        result.update({
                            "ok": False,
                            "forwarded": bool(
                                transport_result.get(
                                    "forwarded",
                                    True,
                                )
                            ),
                            "confirmed_forwarded": False,
                            "transport_attempted": True,
                            "bounded_forward_invalidated": linear_y == 0,
                            "bounded_lateral_invalidated": linear_y != 0,
                            "interlock_stop_succeeded": (
                                isinstance(outcome, dict) and outcome.get("stop_succeeded") is True),
                            "interlock_dispatch_outcome": outcome,
                            "transport_result": transport_result,
                            "error": ("bounded_lateral_invalidated" if linear_y else "bounded_forward_invalidated"),
                            "reason": (
                                outcome.get("reason")
                                if isinstance(outcome, dict)
                                else "forward_interlock_invalidated"
                            ),
                        })

        result = self._request("POST", "/motion", payload)
        if self.forward_interlock is not None:
            self.forward_interlock.stop_active()
        return result

    def streaming_motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        watchdog_timeout=0.50,
        linear_y=0.0,
    ):
        """
        Refresh a continuous velocity command.

        The Robot Bridge republishes the command until another streaming
        update arrives, STOP is requested, or the deadman watchdog expires.
        """
        return self.motion(
            linear_x=linear_x,
            angular_z=angular_z,
            duration=0.25,
            streaming=True,
            watchdog_timeout=watchdog_timeout,
            linear_y=linear_y,
        )

    def move_forward(self, speed=0.10, seconds=1.0, *, dispatch_guard=None):
        return self.motion(
            linear_x=speed,
            angular_z=0.0,
            duration=seconds,
            **({"dispatch_guard": dispatch_guard} if dispatch_guard is not None else {}),
        )

    def move_backward(self, speed=0.10, seconds=1.0):
        return self.motion(
            linear_x=-abs(speed),
            angular_z=0.0,
            duration=seconds,
        )

    def turn_left(self, speed=0.5, seconds=1.0):
        return self.motion(
            linear_x=0.0,
            angular_z=abs(speed),
            duration=seconds,
        )

    def turn_right(self, speed=0.5, seconds=1.0):
        return self.motion(
            linear_x=0.0,
            angular_z=-abs(speed),
            duration=seconds,
        )

    def move_lateral(self, *, speed, seconds, dispatch_guard=None):
        """Same bounded Bridge transport and freshness watchdog as forward."""
        return self.motion(linear_x=0.0, linear_y=speed, angular_z=0.0, duration=seconds, dispatch_guard=dispatch_guard)
