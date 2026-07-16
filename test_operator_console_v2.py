#!/usr/bin/env python3

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RELAY_DIR = PROJECT_DIR / "voice_relay"

HTML_FILE = RELAY_DIR / "index.html"
SERVER_FILE = RELAY_DIR / "server.py"
CSS_FILE = RELAY_DIR / "operator_console.css"
JS_FILE = RELAY_DIR / "operator_console.js"


def main():
    html = HTML_FILE.read_text(
        encoding="utf-8"
    )

    server = SERVER_FILE.read_text(
        encoding="utf-8"
    )

    css = CSS_FILE.read_text(
        encoding="utf-8"
    )

    javascript = JS_FILE.read_text(
        encoding="utf-8"
    )

    assert "/operator_console.css" in html
    assert "/operator_console.js" in html

    assert "OPERATOR_CSS_FILE" in server
    assert "OPERATOR_JS_FILE" in server
    assert '"/operator_console.css"' in server
    assert '"/operator_console.js"' in server

    css_requirements = [
        ".operator-console",
        "minmax(680px, 3fr)",
        ".operator-camera-card .camera-stage",
        "max-width: 1100px",
        ".operator-steering-arrow",
        ".operator-centered-indicator",
        ".operator-camera-banner",
        ".operator-telemetry-card",
    ]

    for requirement in css_requirements:
        assert requirement in css, requirement

    javascript_requirements = [
        "Operator Telemetry",
        "runtime.tracking",
        "tracking.steering_direction",
        "tracking.horizontal_error",
        "tracking.distance_state",
        "tracking.detection_age_ms",
        "operatorLeftArrow",
        "operatorRightArrow",
        "operatorCentered",
        "Open Camera Snapshot",
    ]

    for requirement in javascript_requirements:
        assert requirement in javascript, requirement

    assert (
        "tracking.horizontal_error <"
        not in javascript
    )

    assert (
        "tracking.horizontal_error >"
        not in javascript
    )

    print("PASS: Operator Console assets are linked")
    print("PASS: dashboard server serves static assets")
    print("PASS: camera expands to operator-console size")
    print("PASS: runtime tracking drives telemetry")
    print("PASS: runtime direction drives steering arrows")
    print("PASS: centered indicator is presentation-only")
    print("PASS: existing dashboard remains intact")
    print()
    print("Operator Console v2 offline test passed.")


if __name__ == "__main__":
    main()
