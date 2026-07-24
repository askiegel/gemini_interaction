import http.client
import json
import threading
import unittest
from types import SimpleNamespace

from network.network_manager import NetworkManagerError
from runtime_api import CognitiveRuntimeHTTPServer


class FakeConfigurationManager:
    def get_config(self):
        return {}


class FakeWorldModel:
    robot_state = {}
    environment = {}

    def get_entities(self):
        return []

    def get_recent_events(self):
        return []


class FakeMissionManager:
    def get_history(self):
        return []


class FakeRuntime:
    def __init__(self):
        self.world_model = FakeWorldModel()
        self.mission_manager = FakeMissionManager()

    def get_status(self):
        return {
            "active_mission": None,
            "queue": [],
            "history_count": 0,
            "last_result": None,
        }

    def submit_text(self, command):
        return {
            "command": command,
            "intent": {
                "intent": "STOP",
            },
            "mission": {
                "mission_id": "mission-test",
                "mission_type": "STOP",
                "status": "ACTIVE",
            },
        }

    def submit_intent(self, parsed_intent):
        return SimpleNamespace(
            to_dict=lambda: {
                "mission_id": "mission-test",
                "mission_type": parsed_intent["intent"],
                "status": "ACTIVE",
            }
        )


class FakeNetworkManager:
    def __init__(self):
        self.calls = []
        self.error = None

    def connect(self, ssid, password=None):
        if self.error:
            raise self.error

        self.calls.append(
            (
                "connect",
                ssid,
                password,
            )
        )

        return {
            "ok": True,
            "action": "connect",
            "backend": "test",
            "ssid": ssid,
            "message": f"Connected to {ssid}.",
        }

    def disconnect(self):
        if self.error:
            raise self.error

        self.calls.append(
            (
                "disconnect",
            )
        )

        return {
            "ok": True,
            "action": "disconnect",
            "backend": "test",
            "message": "Disconnected.",
        }

    def forget(self, profile):
        if self.error:
            raise self.error

        self.calls.append(
            (
                "forget",
                profile,
            )
        )

        return {
            "ok": True,
            "action": "forget",
            "backend": "test",
            "profile": profile,
            "message": f"Forgot {profile}.",
        }


class NetworkAPITestCase(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()

        self.server = CognitiveRuntimeHTTPServer(
            ("127.0.0.1", 0),
            runtime=self.runtime,
            config_manager=FakeConfigurationManager(),
        )

        self.network_manager = FakeNetworkManager()
        self.server.network_manager = self.network_manager

        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

        self.host = "127.0.0.1"
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def post(
        self,
        path,
        payload=None,
        content_type="application/json",
        raw_body=None,
    ):
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=3,
        )

        if raw_body is not None:
            body = raw_body
        elif payload is None:
            body = ""
        else:
            body = json.dumps(payload)

        headers = {}

        if content_type is not None:
            headers["Content-Type"] = content_type

        connection.request(
            "POST",
            path,
            body=body,
            headers=headers,
        )

        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        connection.close()

        return (
            response.status,
            json.loads(response_body),
        )

    def test_connect_with_password(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
                "password": "password123",
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        self.assertEqual(
            self.network_manager.calls,
            [
                (
                    "connect",
                    "NETGEAR94",
                    "password123",
                )
            ],
        )

    def test_connect_using_saved_profile(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        self.assertEqual(
            self.network_manager.calls,
            [
                (
                    "connect",
                    "NETGEAR94",
                    None,
                )
            ],
        )

    def test_disconnect(self):
        status, payload = self.post(
            "/network/disconnect",
            {},
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        self.assertEqual(
            self.network_manager.calls,
            [
                (
                    "disconnect",
                )
            ],
        )

    def test_forget_profile(self):
        status, payload = self.post(
            "/network/forget",
            {
                "profile": "NETGEAR94",
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

        self.assertEqual(
            self.network_manager.calls,
            [
                (
                    "forget",
                    "NETGEAR94",
                )
            ],
        )

    def test_missing_content_type_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
            },
            content_type=None,
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "Content-Type",
            payload["error"],
        )

    def test_wrong_content_type_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
            },
            content_type="text/plain",
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "application/json",
            payload["error"],
        )

    def test_invalid_json_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            content_type="application/json",
            raw_body="{not-valid-json",
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "valid JSON",
            payload["error"],
        )

    def test_non_object_json_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            [
                "NETGEAR94",
            ],
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "JSON object",
            payload["error"],
        )

    def test_empty_ssid_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "   ",
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "ssid",
            payload["error"],
        )

    def test_non_string_password_is_rejected(self):
        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
                "password": 12345,
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "password",
            payload["error"],
        )

    def test_nonempty_disconnect_body_is_rejected(self):
        status, payload = self.post(
            "/network/disconnect",
            {
                "unexpected": True,
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])

    def test_empty_forget_profile_is_rejected(self):
        status, payload = self.post(
            "/network/forget",
            {
                "profile": "   ",
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "profile",
            payload["error"],
        )

    def test_network_manager_error_returns_503(self):
        self.network_manager.error = NetworkManagerError(
            "Network backend unavailable."
        )

        status, payload = self.post(
            "/network/connect",
            {
                "ssid": "NETGEAR94",
            },
        )

        self.assertEqual(status, 503)
        self.assertFalse(payload["ok"])
        self.assertIn(
            "unavailable",
            payload["error"],
        )

    def test_unknown_endpoint_returns_404(self):
        status, payload = self.post(
            "/network/unknown",
            {},
        )

        self.assertEqual(status, 404)
        self.assertFalse(payload["ok"])

    def test_existing_missions_endpoint_still_works(self):
        status, payload = self.post(
            "/missions",
            {
                "command": "Stop",
            },
        )

        self.assertEqual(status, 202)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["accepted"])

        self.assertEqual(
            payload["submission_mode"],
            "command",
        )


if __name__ == "__main__":
    unittest.main()
