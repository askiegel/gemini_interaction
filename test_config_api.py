#!/usr/bin/env python3

import json
import sys
import tempfile
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

# runtime_api imports CognitiveRuntime for production startup. The API server
# test supplies its own fake runtime, so avoid importing optional Gemini SDKs.
runtime_stub = types.ModuleType("runtime")
runtime_stub.CognitiveRuntime = object
sys.modules.setdefault("runtime", runtime_stub)

from config.config_manager import ConfigurationManager
from runtime_api import create_server


class FakeRuntime:
    running = True

    def get_status(self):
        return {"ok": True, "running": True}


def request_json(method, url, payload=None):
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
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    with tempfile.TemporaryDirectory() as directory:
        manager = ConfigurationManager(
            Path(directory) / "system_config.json"
        )
        server = create_server(
            runtime=FakeRuntime(),
            host="127.0.0.1",
            port=0,
            config_manager=manager,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        try:
            print("===== GET CONFIG =====")
            status, response = request_json("GET", f"{base_url}/config")
            assert status == 200
            assert response["config"]["robot"]["name"] == "Mayday"
            assert response["config"]["robot"]["hostname"] == (
                "minipupperv2.local"
            )

            print("===== PUT CONFIG =====")
            updated = response["config"]
            updated["robot"]["name"] = "Tony-02"
            updated["network"]["robot_ip"] = "192.168.68.155"
            status, response = request_json(
                "PUT",
                f"{base_url}/config",
                {"config": updated},
            )
            assert status == 200
            assert response["saved"] is True
            assert response["config"]["robot"]["name"] == "Tony-02"

            print("===== REJECT INVALID CONFIG =====")
            invalid = response["config"]
            invalid["network"]["robot_bridge_port"] = 70000
            status, response = request_json(
                "PUT",
                f"{base_url}/config",
                invalid,
            )
            assert status == 400
            assert response["saved"] is False

        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)

    print("PASS: Runtime configuration API tests passed.")


if __name__ == "__main__":
    main()
