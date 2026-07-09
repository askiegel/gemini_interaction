import json
import urllib.request
import urllib.error

from config.robot_bridge import ROBOT_BRIDGE_URL


DEFAULT_BASE_URL = ROBOT_BRIDGE_URL


class RobotBridgeClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=3.0):
        self.base_url = base_url.rstrip("/")
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

    def motion(self, linear_x=0.0, angular_z=0.0, duration=0.25):
        return self._request(
            "POST",
            "/motion",
            {
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
                "duration": float(duration),
            },
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
