import json
import urllib.request
import urllib.error

from config.config_manager import ConfigurationManager




class RobotBridgeClient:
    def __init__(
        self,
        base_url=None,
        timeout=3.0,
        config_manager=None,
    ):
        self.config_manager = (
            config_manager or ConfigurationManager()
        )
        resolved_url = (
            base_url or self.config_manager.robot_bridge_url
        )
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout

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
        return self._request("POST", "/stop")

    def motion(
        self,
        linear_x=0.0,
        angular_z=0.0,
        duration=0.25,
        streaming=False,
        watchdog_timeout=0.50,
    ):
        payload = {
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
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

        return self._request(
            "POST",
            "/motion",
            payload,
        )

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
