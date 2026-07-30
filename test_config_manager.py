#!/usr/bin/env python3

import json
import tempfile
from pathlib import Path

from config.config_manager import (
    ConfigurationError,
    ConfigurationManager,
    DEFAULT_CONFIG,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "system_config.json"
        manager = ConfigurationManager(path)

        print("===== DEFAULT CREATION =====")
        assert path.exists()
        assert manager.get_config()["robot"]["name"] == "Mayday"

        updated = manager.get_config()
        updated["robot"]["name"] = "Tony-02"
        updated["network"]["robot_ip"] = "192.168.68.155"
        saved = manager.update_config(updated)

        print("===== ATOMIC UPDATE =====")
        assert saved["robot"]["name"] == "Tony-02"
        assert manager.robot_bridge_url == "http://192.168.68.155:8090"
        assert json.loads(path.read_text())["robot"]["name"] == "Tony-02"

        print("===== DEFENSIVE COPY =====")
        copy_value = manager.get_config()
        copy_value["robot"]["name"] = "mutated"
        assert manager.get_config()["robot"]["name"] == "Tony-02"

        print("===== INVALID CONFIGURATION =====")
        invalid = dict(DEFAULT_CONFIG)
        invalid["network"] = dict(DEFAULT_CONFIG["network"])
        invalid["network"]["ros_domain"] = 999
        try:
            manager.update_config(invalid)
        except ConfigurationError:
            pass
        else:
            raise AssertionError("Invalid ROS domain was accepted.")

    print("PASS: ConfigurationManager tests passed.")


if __name__ == "__main__":
    main()
