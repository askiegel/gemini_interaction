import json
import math
import urllib.request
import urllib.error

from robot_bridge.forward_interlock import FORWARD_POLICY_STRICT




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

    def _request(self, method, path, payload=None):
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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
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
        forward_policy=FORWARD_POLICY_STRICT,
    ):
        linear_x = float(linear_x)
        angular_z = float(angular_z)
        payload = {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": float(duration),
        }

        if streaming:
            payload.update(
                {
                    "streaming": True,
                    "watchdog_timeout": float(
                        watchdog_timeout
                    ),
                }
            )

        if linear_x > 0:
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
                dispatch_options = {"streaming": streaming}
                if forward_policy != FORWARD_POLICY_STRICT:
                    dispatch_options["policy"] = forward_policy
                generation = interlock.begin_positive_dispatch(
                    **dispatch_options
                )
            except PermissionError as exc:
                return {"ok": False, "forwarded": False, "error": str(exc)}
            result = None
            try:
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
                    if dispatch_valid is False and isinstance(result, dict):
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
                            "bounded_forward_invalidated": True,
                            "transport_result": transport_result,
                            "error": "bounded_forward_invalidated",
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
        )

    def move_forward(
        self,
        speed=0.10,
        seconds=1.0,
        *,
        forward_policy=FORWARD_POLICY_STRICT,
    ):
        return self.motion(
            linear_x=speed,
            angular_z=0.0,
            duration=seconds,
            forward_policy=forward_policy,
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
