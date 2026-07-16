#!/usr/bin/env python3

from unittest.mock import patch

from robot_bridge.client import RobotBridgeClient


def main():
    client = RobotBridgeClient(
        base_url="http://robot.invalid",
    )

    captured = {}

    def fake_request(
        method,
        path,
        payload=None,
    ):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload

        return {
            "ok": True,
            "mode": "streaming",
        }

    with patch.object(
        client,
        "_request",
        side_effect=fake_request,
    ):
        result = client.streaming_motion(
            linear_x=0.12,
            angular_z=-0.30,
            watchdog_timeout=0.50,
        )

    assert result["ok"] is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/motion"

    payload = captured["payload"]

    assert payload["linear_x"] == 0.12
    assert payload["angular_z"] == -0.30
    assert payload["streaming"] is True
    assert payload["watchdog_timeout"] == 0.50

    print("PASS: streaming client uses /motion")
    print("PASS: streaming flag is included")
    print("PASS: watchdog timeout is included")
    print()
    print("Robot Bridge streaming client test passed.")


if __name__ == "__main__":
    main()
