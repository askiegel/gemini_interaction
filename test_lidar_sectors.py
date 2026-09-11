"""Offline geometry and handler tests; no sockets, ROS, or services."""

import json
import math
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from voice_relay.lidar_sectors import calculate_sectors, lidar_sector_payload
from voice_relay.server import ROBOT_BRIDGE_URL, VoiceRelayHandler


def scan(ranges, start=-90, step=45):
    return {
        "frame_id": "lidar_link",
        "angle_min": math.radians(start),
        "angle_increment": math.radians(step),
        "range_min": 0.1,
        "range_max": 10.0,
        "ranges": ranges,
        "stamp_seconds": 123.5,
    }


def envelope(value):
    return {"ok": True, "telemetry": {
        "available": True, "age_seconds": 0.05,
        "received_at": "2026-09-11T12:00:00Z", "scan": value,
    }}


def test_angle_metadata_and_robot_orientation():
    result = calculate_sectors(scan([1, 2, 3, 4, 5]))
    for name, distance in zip(
        ("right", "front_right", "front", "front_left", "left"),
        (1, 2, 3, 4, 5),
    ):
        assert result[name] == {
            "valid_sample_count": 1, "minimum_clearance_m": distance,
            "robust_clearance_m": distance, "available": True,
        }


@pytest.mark.parametrize("start,step", [(270, 45), (-450, 45), (90, -45)])
def test_angle_normalization_and_negative_increment(start, step):
    ranges = [1, 2, 3, 4, 5]
    if step < 0:
        ranges.reverse()
    assert calculate_sectors(scan(ranges, start, step)) == calculate_sectors(
        scan([1, 2, 3, 4, 5])
    )


def test_invalid_ranges_filtered_and_limits_included():
    result = calculate_sectors(scan(
        [None, math.nan, math.inf, -math.inf, 0.09, 10.1, True, "2", 0.1, 10],
        -5, 1,
    ))
    assert result["front"]["valid_sample_count"] == 2
    assert result["front"]["minimum_clearance_m"] == 0.1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("ranges", [[], [None], [math.inf]])
def test_empty_sectors(ranges):
    for sector in calculate_sectors(scan(ranges)).values():
        assert sector == {
            "valid_sample_count": 0, "minimum_clearance_m": None,
            "robust_clearance_m": None, "available": False,
        }


def test_robust_percentile_is_deterministic_and_resists_isolated_return():
    result = calculate_sectors(scan([0.1] + [3] * 10, -5, 1))["front"]
    assert result["minimum_clearance_m"] == 0.1
    assert result["robust_clearance_m"] == 3
    result = calculate_sectors(scan([4, 1, 3, 2], -5, 1))["front"]
    assert result["robust_clearance_m"] == pytest.approx(1.3)


@pytest.mark.parametrize("angle,sector", [
    (-120, "right"), (-60, "front_right"), (-20, "front"),
    (20, "front_left"), (60, "left"), (120, "left"),
])
def test_boundaries_assigned_once(angle, sector):
    result = calculate_sectors(scan([2], angle))
    assert result[sector]["valid_sample_count"] == 1
    assert sum(item["valid_sample_count"] for item in result.values()) == 1


@pytest.mark.parametrize("angle", [-180, 180, 540, -121, 121])
def test_rear_returns_are_excluded(angle):
    assert not any(x["available"] for x in calculate_sectors(scan([2], angle)).values())


@pytest.mark.parametrize("field,value", [
    ("angle_min", None), ("angle_increment", 0), ("angle_increment", math.inf),
    ("range_min", -1), ("range_max", 0.01), ("ranges", None),
])
def test_invalid_geometry(field, value):
    value_scan = scan([2])
    value_scan[field] = value
    with pytest.raises(ValueError):
        calculate_sectors(value_scan)
    result = lidar_sector_payload(envelope(value_scan))
    assert result["ok"] is False
    assert not any(x["available"] for x in result["sectors"].values())


def test_bridge_frame_rotation_and_source_metadata():
    result = lidar_sector_payload(envelope(scan([1, 2, 3], -180, 90)))
    for name, distance in (("right", 1), ("front", 2), ("left", 3)):
        assert result["sectors"][name]["minimum_clearance_m"] == distance
    assert result["source"] == {
        "frame_id": "lidar_link", "stamp_seconds": 123.5,
        "received_at": "2026-09-11T12:00:00Z", "age_seconds": 0.05,
        "rotation_to_robot_radians": math.pi / 2,
    }


@pytest.mark.parametrize("source", [None, {}, {"ok": False},
    {"ok": True, "telemetry": {"available": False}},
    envelope({**scan([1]), "frame_id": "unknown"}),
])
def test_unavailable_or_unknown_frame(source):
    result = lidar_sector_payload(source)
    assert result["ok"] is False
    assert not any(x["available"] for x in result["sectors"].values())


@pytest.mark.parametrize("method", ["do_GET", "do_POST"])
def test_endpoint_only_calls_lidar_get_and_never_other_handler_actions(method):
    handler = object.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/lidar-sectors"
    handler.send_json = Mock()
    handler.headers = {}
    # Any other handler action (including runtime creation, motion, navigation,
    # localization and body parsing) fails before it can have side effects.
    allowed = {"do_GET", "do_POST", "lidar_status", "lidar_sectors_status", "send_json"}
    with ExitStack() as stack:
        for name, value in vars(VoiceRelayHandler).items():
            if callable(value) and name not in allowed:
                stack.enter_context(patch.object(handler, name, side_effect=AssertionError(name)))
        request = stack.enter_context(patch("voice_relay.server.request_json", return_value={
            "ok": True, "data": envelope(scan([2])), "status_code": 200, "error": None,
        }))
        stack.enter_context(patch("socket.socket", side_effect=AssertionError("network")))
        stack.enter_context(patch("subprocess.Popen", side_effect=AssertionError("process")))
        getattr(handler, method)()
    status, payload = handler.send_json.call_args.args
    if method == "do_GET":
        assert status == 200
        assert payload["read_only"] is True
        assert payload["perception"] == "local_obstacle_sectors"
        request.assert_called_once_with("GET", f"{ROBOT_BRIDGE_URL}/telemetry/lidar", timeout=3.0)
    else:
        assert status == 405
        request.assert_not_called()


def test_endpoint_preserves_upstream_failure():
    handler = object.__new__(VoiceRelayHandler)
    with patch.object(handler, "lidar_status", return_value=(503, {"ok": False, "error": "offline"})):
        status, payload = handler.lidar_sectors_status()
    assert status == 503
    assert payload["error"] == "offline"
    assert not any(x["available"] for x in payload["sectors"].values())
