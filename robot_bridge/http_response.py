"""Bounded Bridge error certificates; HTTP failures never grant motion authority."""
import errno
import json
import math
import re
import socket

MAX_ERROR_BODY_BYTES = 16_384
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_BOOL_FIELDS = {
    "transport_attempted", "delivery_uncertain", "controller_gait_zero",
    "physical_stop_verified", "completed_progress_credit", "automatic_stop",
    "forwarded", "confirmed_forwarded",
}
_ID_FIELDS = {"controller_session", "bridge_epoch", "command_id", "host_boot_id"}
_INT_FIELDS = {"generation": 2**64 - 1, "sequence": 2**64 - 1,
               "start_ns": 2**63 - 1, "deadline_ns": 2**63 - 1}


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate response key")
        out[key] = value
    return out


def decode_error_body(raw):
    if len(raw) > MAX_ERROR_BODY_BYTES:
        raise ValueError("response too large")
    def invalid_constant(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=invalid_constant)


def _bounded_fields(body):
    """Only known scalar certificate fields; raw payloads and URLs are excluded."""
    if not isinstance(body, dict):
        raise ValueError("object required")
    out = {}
    for key in _BOOL_FIELDS:
        if key in body:
            if type(body[key]) is not bool:
                raise ValueError("invalid boolean")
            out[key] = body[key]
    for key in _ID_FIELDS:
        if key in body:
            if type(body[key]) is not str or not _ID.fullmatch(body[key]):
                raise ValueError("invalid identity")
            out[key] = body[key]
    for key, maximum in _INT_FIELDS.items():
        if key in body:
            if type(body[key]) is not int or not 0 <= body[key] <= maximum:
                raise ValueError("invalid integer")
            out[key] = body[key]
    for key, maximum in (("linear_x", .5), ("linear_y", .1), ("angular_z", 1.), ("duration", 2.)):
        if key in body:
            value = body[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or abs(value) > maximum or key == "duration" and value <= 0):
                raise ValueError("invalid command scalar")
            out[key] = value
    if all(k in out for k in ("duration", "start_ns", "deadline_ns")):
        if int(out["duration"] * 1e9) != out["deadline_ns"] - out["start_ns"]:
            raise ValueError("contradictory immutable duration")
    for key in ("reason", "stop_reason"):
        if key in body:
            if type(body[key]) is not str or not _TOKEN.fullmatch(body[key]):
                raise ValueError("invalid reason")
            out[key] = body[key]
    if "stop_result" in body:
        stop = body["stop_result"]
        if not isinstance(stop, dict) or type(stop.get("ok")) is not bool:
            raise ValueError("invalid STOP evidence")
        allowed = {"reason", "controller_gait_zero", "physical_stop_verified"}
        out["stop_result"] = _bounded_fields({k: v for k, v in stop.items() if k in allowed})
        out["stop_result"]["ok"] = stop["ok"]
        if not stop["ok"] and stop.get("controller_gait_zero") is True:
            raise ValueError("contradictory STOP evidence")
    return out


def server_error(status, body=None, endpoint=None):
    result = dict(ok=False, error=f"HTTP {status}", http_status=status,
                  response_origin="server_error", http_response_received=True,
                  transport_attempted=None, delivery_uncertain=True,
                  completed_progress_credit=False, retry_permitted=False,
                  evidence_valid=False)
    try:
        if status < 400:
            raise ValueError("unexpected success status for error outcome")
        fields = _bounded_fields(body)
        if body.get("ok") is not False:
            raise ValueError("error response cannot claim success")
        state = body.get("state")
        if endpoint == "/stop" and state is None:
            stop = fields.get("stop_result")
            if not stop or stop["ok"] or fields.get("physical_stop_verified") is True:
                raise ValueError("invalid failed STOP certificate")
            result.update(fields, state="uncertain", evidence_valid=True)
            return result  # STOP failure never certifies zero or grants motion.
        if state not in ("refused", "interrupted", "uncertain"):
            raise ValueError("missing or invalid command outcome")
        if fields.get("completed_progress_credit") is not False:
            raise ValueError("missing or contradictory progress result")
        attempted = fields.get("transport_attempted")
        if type(attempted) is not bool:
            raise ValueError("missing transport evidence")
        if state == "refused" and (attempted or fields.get("delivery_uncertain") is True
                                   or fields.get("forwarded") is True):
            raise ValueError("contradictory zero-transport refusal")
        if state == "interrupted" and (not attempted
                or fields.get("controller_gait_zero") is not True
                or fields.get("automatic_stop") is not True
                or "stop_reason" not in fields):
            raise ValueError("interruption requires transport and resolved software STOP")
        if fields.get("confirmed_forwarded") is True or fields.get("physical_stop_verified") is True:
            raise ValueError("failed response cannot prove completion or physical stop")
        if state in ("interrupted", "uncertain") and attempted:
            if not all(key in fields for key in ("generation", "start_ns", "deadline_ns",
                                                "controller_session", "bridge_epoch")):
                raise ValueError("missing command identity")
            if fields["generation"] == 0 or not 0 < fields["deadline_ns"] - fields["start_ns"] <= 2_000_000_000:
                raise ValueError("invalid immutable command deadline")
        if "reason" not in fields:
            raise ValueError("missing reason")
        if (fields.get("controller_gait_zero") is True and fields.get("automatic_stop") is False
                or fields.get("stop_result", {}).get("ok") is False and fields.get("controller_gait_zero") is True):
            raise ValueError("contradictory zero evidence")
        if state == "uncertain" and fields.get("delivery_uncertain") is False:
            raise ValueError("uncertain outcome cannot claim certain delivery")
        result.update(fields, state=state, evidence_valid=True,
                      delivery_uncertain=(state == "uncertain" or fields.get("delivery_uncertain", False)))
    except (ValueError, TypeError, RecursionError):
        result.update(reason="bridge_error_evidence_invalid", state="uncertain")
    return result


def transport_error(exc):
    reason = getattr(exc, "reason", exc)
    disconnected = (isinstance(reason, socket.gaierror)
                    or isinstance(reason, OSError) and reason.errno in
                    (errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH))
    timeout = isinstance(reason, TimeoutError)
    return dict(ok=False, error=("bridge_connection_failed" if disconnected else
                                "bridge_response_timeout" if timeout else "bridge_delivery_uncertain"),
                response_origin="connection_failure" if disconnected else "transport_uncertain",
                http_status=None, http_response_received=False,
                transport_attempted=False if disconnected else None,
                delivery_uncertain=not disconnected, completed_progress_credit=False,
                retry_permitted=False, evidence_valid=False, state="uncertain")


def completed_response(status, body, expected_payload):
    """A guarded completion is positive only with a coherent bounded certificate."""
    try:
        fields = _bounded_fields(body)
        if (status != 200 or body.get("state") != "completed" or body.get("ok") is not True
                or fields.get("transport_attempted") is not True
                or fields.get("controller_gait_zero") is not True
                or fields.get("completed_progress_credit") is not True
                or fields.get("delivery_uncertain") is True
                or fields.get("physical_stop_verified") is True
                or fields.get("automatic_stop") is False
                or fields.get("forwarded") is False
                or fields.get("confirmed_forwarded") is False
                or fields.get("stop_result", {}).get("ok") is False):
            raise ValueError("contradictory completion")
        if not all(k in fields for k in ("generation", "start_ns", "deadline_ns", "controller_session", "bridge_epoch", "reason", "stop_reason")):
            raise ValueError("missing completion identity")
        if fields["generation"] == 0 or not 0 < fields["deadline_ns"] - fields["start_ns"] <= 2_000_000_000:
            raise ValueError("invalid completion deadline")
        for key in ("linear_x", "linear_y", "angular_z", "duration"):
            if key not in fields or fields[key] != expected_payload.get(key, 0.):
                raise ValueError("completion does not match requested command")
        if (body.get("action") != "motion" or body.get("mode") != "bounded"
                or body.get("returned_immediately") is not False
                or fields.get("automatic_stop") is not True):
            raise ValueError("incomplete bounded completion")
        for key in ("action", "mode"):
            if key in body:
                if type(body[key]) is not str or not _TOKEN.fullmatch(body[key]):
                    raise ValueError("invalid completion label")
                fields[key] = body[key]
        if "returned_immediately" in body:
            if body["returned_immediately"] is not False:
                raise ValueError("completion cannot be immediate")
            fields["returned_immediately"] = False
        return dict(fields, ok=True, state="completed", http_status=status,
                    response_origin="server_success", http_response_received=True,
                    evidence_valid=True, retry_permitted=False)
    except (ValueError, TypeError, RecursionError):
        result = server_error(status)
        result["error"] = "bridge_completion_evidence_invalid"
        return result
