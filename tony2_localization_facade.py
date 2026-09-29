"""Small Cognitive-side client for Tony2 localization authority.

This facade deliberately exposes only read status and the existing unseeded
global-localization retry.  It has no navigation-goal, Home-seeding, or motion
methods; physical scan turns stay owned by :class:`CognitiveRuntime`.
"""

import json
import urllib.error
import urllib.request


class Tony2LocalizationFacade:
    """Access canonical Tony2 localization state through Voice Relay."""

    def __init__(self, base_url="http://127.0.0.1:8765", timeout_seconds=15.0):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def get_localization_status(self):
        """Return the read-only canonical map-pose status response."""
        return self._request("GET", "/dashboard/navigation-pose")

    def retry_global_localization(self):
        """Request exactly one existing unseeded localization attempt."""
        return self._request(
            "POST", "/dashboard/navigation-initialize-global-localization",
        )

    def _request(self, method, path):
        request = urllib.request.Request(
            self.base_url + path,
            data=(b"{}" if method == "POST" else None),
            headers=(
                {"Content-Type": "application/json"}
                if method == "POST" else {}
            ),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {
                    "ok": False, "reason": "LOCALIZATION_RESPONSE_MALFORMED",
                }
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return payload
            return {"ok": False, "reason": "LOCALIZATION_REQUEST_FAILED", "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "reason": "LOCALIZATION_REQUEST_FAILED", "error": str(exc)}
