import json
import math
import urllib.request
import urllib.error

LOCAL_FORWARD_TIMEOUT_SECONDS = 120.0





class RobotBridgeClient:
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
                body = response.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as exc:
            return {
                "ok": False,
                "error": f"HTTP {exc.code}",
                "url": url,
            }

        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "url": url,
            }

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
                    result = {"ok": False, "forwarded": False, "error": "motion_dispatch_preempted"}
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
