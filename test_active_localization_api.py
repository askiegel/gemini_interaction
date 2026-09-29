from types import SimpleNamespace
from unittest.mock import Mock

from runtime_api import RuntimeAPIHandler


def _call(body, result):
    runtime = SimpleNamespace(run_bounded_active_localization=Mock(return_value=result))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/active-localization/scan"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value=body)
    handler.send_json = Mock()
    handler.do_POST()
    return handler, runtime


def test_active_localization_endpoint_accepts_only_empty_object():
    handler, runtime = _call({}, {"ok": True})
    assert handler.send_json.call_args.args == (200, {"ok": True})
    runtime.run_bounded_active_localization.assert_called_once_with()


def test_active_localization_endpoint_rejects_caller_motion_fields():
    handler, runtime = _call({"linear_x": 0.0}, {"ok": True})
    status, response = handler.send_json.call_args.args
    assert status == 400
    assert response["ok"] is False
    runtime.run_bounded_active_localization.assert_not_called()


def test_active_localization_terminal_failure_is_not_http_success():
    result = {"ok": False, "terminal_reason": "ACTIVE_LOCALIZATION_EXHAUSTED"}
    handler, runtime = _call({}, result)
    assert handler.send_json.call_args.args == (409, result)
    runtime.run_bounded_active_localization.assert_called_once_with()
