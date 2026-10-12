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


RECOVERABLE_ACTIVE_LIDAR_REASONS = frozenset({
    "active_lidar_intrusion", "active_lidar_stale", "active_lidar_unavailable",
    "active_lidar_cache_busy", "active_lidar_coverage_missing",
})


def _scan_identity(value):
    if not isinstance(value, dict):
        raise ValueError("missing scan identity")
    session = value.get("producer_session")
    seq, stamp = value.get("acquisition_sequence"), value.get("source_stamp_ns")
    if (type(session) is not str or not _ID.fullmatch(session)
            or type(seq) is not int or not 0 < seq < 2**64
            or type(stamp) is not int or not 0 <= stamp < 2**64):
        raise ValueError("invalid native scan identity")
    return dict(producer_session=session, acquisition_sequence=seq, source_stamp_ns=stamp)


def _controller_fields(value, *, stopped=False):
    if not isinstance(value, dict):
        raise ValueError("invalid controller certificate")
    out = {}
    for k in ("controller_session", "bridge_epoch"):
        if type(value.get(k)) is not str or not _ID.fullmatch(value[k]):
            raise ValueError("invalid controller identity")
        out[k] = value[k]
    for k in ("evaluated_ns", "last_operation", "last_request_generation",
              "last_request_sequence", "last_request_issued_ns") + (("generation", "sequence") if stopped else ()):
        if type(value.get(k)) is not int or not 0 <= value[k] < 2**64:
            raise ValueError("invalid controller integer")
        out[k] = value[k]
    if value.get("last_request_accepted") is not True:
        raise ValueError("request not acknowledged")
    out["last_request_accepted"] = True
    if stopped:
        if value.get("active") is not False or value.get("gait_zero") is not True:
            raise ValueError("controller not inactive and zero")
        out.update(active=False, gait_zero=True)
        for k in ("linear_x", "linear_y", "angular_z"):
            if type(value.get(k)) not in (int, float) or value[k] != 0:
                raise ValueError("nonzero command")
            out[k] = 0.0
    return out


def certified_active_lidar_interruption(body):
    """Resolved transported interruption, never permission to replay a command."""
    if not isinstance(body, dict) or body.get("evidence_valid") is not True:
        return None
    try:
        if (body.get("state") != "interrupted" or body.get("transport_attempted") is not True
                or body.get("delivery_uncertain") is not False or body.get("ok") is not False
                or body.get("completed_progress_credit") is not False
                or body.get("controller_gait_zero") is not True or body.get("automatic_stop") is not True
                or body.get("physical_stop_verified") is not False
                or body.get("reason") not in RECOVERABLE_ACTIVE_LIDAR_REASONS):
            return None
        stop = _controller_fields(body.get("stop_certificate"), stopped=True)
        accepted = _controller_fields(body.get("controller_acceptance"))
        for cert in (stop, accepted):
            if any(cert[k] != body.get(k) for k in ("controller_session", "bridge_epoch")):
                return None
            if cert["last_request_generation"] != body.get("generation"):
                return None
        if (stop["last_operation"] != 3 or accepted["last_operation"] not in (1, 2)
                or stop["generation"] != body.get("generation")
                or stop["sequence"] != stop["last_request_sequence"]
                or stop["sequence"] <= accepted["last_request_sequence"]
                or not body["start_ns"] <= accepted["last_request_issued_ns"] < body["deadline_ns"]
                or not accepted["last_request_issued_ns"] <= stop["last_request_issued_ns"] <= stop["evaluated_ns"]):
            return None
        gate = body.get("lidar_safety")
        if not isinstance(gate, dict) or gate.get("permitted") is not False or gate.get("reason") != body["reason"]:
            return None
        floor = _scan_identity(body.get("last_verified_lidar"))
        if "producer_session" in gate:
            current = _scan_identity(gate)
            if (current["producer_session"] != floor["producer_session"]
                    or current["acquisition_sequence"] < floor["acquisition_sequence"]
                    or current["source_stamp_ns"] < floor["source_stamp_ns"]
                    or (current["acquisition_sequence"] == floor["acquisition_sequence"]) !=
                       (current["source_stamp_ns"] == floor["source_stamp_ns"])):
                return None
            floor = current
        elif body["reason"] not in {"active_lidar_unavailable", "active_lidar_cache_busy"}:
            return None
        if gate.get("source_fault") is not None:
            return None
        return dict(scan_floor=floor,
                    sequence_floor_source="interrupting_scan" if "producer_session" in gate else "last_verified_lidar",
                    stop_certificate=stop, controller_acceptance=accepted,
                    reason=body["reason"], generation=body["generation"],
                    controller_session=body["controller_session"], bridge_epoch=body["bridge_epoch"],
                    axes=tuple(body[k] for k in ("linear_x", "linear_y", "angular_z")),
                    duration=body["duration"])
    except (KeyError, ValueError, TypeError):
        return None


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
    if body.get("lidar_safety") is not None:
        gate = body["lidar_safety"]
        if (not isinstance(gate, dict) or type(gate.get("permitted")) is not bool
                or type(gate.get("reason")) is not str or not _TOKEN.fullmatch(gate["reason"])):
            raise ValueError("invalid LiDAR certificate")
        bounded = {"permitted": gate["permitted"], "reason": gate["reason"]}
        if "producer_session" in gate:
            bounded.update(_scan_identity(gate))
        for k in ("effective_age_seconds", "received_monotonic_seconds", "protected_radius_m"):
            if k in gate:
                if type(gate[k]) not in (int, float) or not math.isfinite(gate[k]) or gate[k] < 0:
                    raise ValueError("invalid LiDAR scalar")
                if k == "protected_radius_m" and gate[k] != .45:
                    raise ValueError("incompatible protected radius")
                bounded[k] = gate[k]
        if "source_fault" in gate:
            if type(gate["source_fault"]) is not str or not _TOKEN.fullmatch(gate["source_fault"]):
                raise ValueError("invalid source fault")
            bounded["source_fault"] = gate["source_fault"]
        out["lidar_safety"] = bounded
    for k, parser in (("last_verified_lidar", _scan_identity),
                      ("controller_acceptance", _controller_fields)):
        if body.get(k) is not None:
            out[k] = parser(body[k])
    if body.get("stop_certificate") is not None:
        stop = _controller_fields(body["stop_certificate"], stopped=True)
        if (any(stop[k] != body.get(k) for k in ("controller_session", "bridge_epoch", "generation"))
                or stop["last_operation"] != 3
                or stop["generation"] != stop["last_request_generation"]
                or stop["sequence"] != stop["last_request_sequence"]
                or stop["last_request_issued_ns"] > stop["evaluated_ns"]):
            raise ValueError("contradictory scoped STOP")
        out["stop_certificate"] = stop
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
