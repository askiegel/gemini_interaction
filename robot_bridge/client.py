import json
import urllib.request
import urllib.error





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
            if self.forward_interlock is None:
                return {"ok": False, "forwarded": False, "error": "forward_interlock_not_configured"}
            interlock = self.forward_interlock
            try:
                generation = interlock.begin_positive_dispatch(streaming=streaming)
            except PermissionError as exc:
                return {"ok": False, "forwarded": False, "error": str(exc)}
            result = None
            try:
                result = self._request("POST", "/motion", payload)
                return result
            finally:
                interlock.finalize_positive_dispatch(generation, result)

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

    def move_forward(self, speed=0.10, seconds=1.0):
        return self.motion(
            linear_x=speed,
            angular_z=0.0,
            duration=seconds,
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
