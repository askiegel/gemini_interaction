#!/usr/bin/env python3

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
HTML_FILE = PROJECT_DIR / "voice_relay" / "index.html"
JS_FILE = PROJECT_DIR / "voice_relay" / "operator_console.js"
CSS_FILE = PROJECT_DIR / "voice_relay" / "operator_console.css"
SERVER_FILE = PROJECT_DIR / "voice_relay" / "server.py"


def require(text, fragment, label):
    assert fragment in text, f"Missing {label}: {fragment}"


def main():
    html = HTML_FILE.read_text(encoding="utf-8")
    js = JS_FILE.read_text(encoding="utf-8")
    css = CSS_FILE.read_text(encoding="utf-8")
    server = SERVER_FILE.read_text(encoding="utf-8")

    for field_id in (
        "configurationForm",
        "configRobotName",
        "configRobotModel",
        "configRobotHostname",
        "configRobotIp",
        "configRobotBridgePort",
        "configBrainIp",
        "configRosDomain",
        "configVisionServerUrl",
        "configSpeechProvider",
        "configUiTheme",
        "configCameraLayout",
        "saveConfigurationButton",
        "reloadConfigurationButton",
    ):
        require(html, f'id="{field_id}"', field_id)

    require(js, 'const CONFIGURATION_URL = "/dashboard/config";', "configuration URL")
    require(js, 'method: "PUT"', "PUT request")
    require(js, "loadConfiguration();", "initial configuration load")
    require(css, ".configuration-field", "configuration field styling")
    require(server, 'if path == "/dashboard/config":', "GET configuration proxy")
    require(server, "def do_PUT(self):", "PUT handler")
    require(server, 'f"{COGNITIVE_RUNTIME_URL}/config"', "runtime configuration forwarding")

    print("PASS: Administration configuration UI checks passed.")


if __name__ == "__main__":
    main()
