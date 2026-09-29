import json

from tony2_localization_facade import Tony2LocalizationFacade


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_facade_uses_only_pose_status_and_unseeded_retry(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url, request.data, timeout))
        return _Response({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    facade = Tony2LocalizationFacade("http://voice", timeout_seconds=7.0)
    assert facade.get_localization_status() == {"ok": True}
    assert facade.retry_global_localization() == {"ok": True}
    assert requests == [
        ("GET", "http://voice/dashboard/navigation-pose", None, 7.0),
        (
            "POST",
            "http://voice/dashboard/navigation-initialize-global-localization",
            b"{}",
            7.0,
        ),
    ]
