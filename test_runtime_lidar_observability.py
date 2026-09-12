"""Offline tests for the read-only transient LiDAR runtime endpoint."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

from runtime_api import RuntimeAPIHandler
from world_model import WorldModel


SESSION = "production-session"


def sector(state="CLEAR", robust=1.25, minimum=1.10):
    return {
        "state": state,
        "available": True,
        "robust_clearance_m": robust,
        "minimum_clearance_m": minimum,
    }


def snapshot(*, session=SESSION, received=None, age=0.01):
    received = time.monotonic() if received is None else received
    return {
        "available": True,
        "valid": True,
        "reason": "fresh",
        "producer_session": session,
        "acquisition_sequence": 17,
        "received_monotonic_seconds": received,
        "age_at_receipt_seconds": age,
        "sectors": {
            "front": sector("CAUTION", 0.80, 0.70),
            "front_left": sector("CLEAR", 1.40, 1.20),
            "left": sector("CLEAR", 1.60, 1.35),
            "front_right": sector("BLOCKED", 0.30, 0.20),
            "right": sector("CLEAR", 0.55, 0.45),
        },
        "source": {
            "stamp_seconds": 1234.5,
            "age_seconds": 0.01,
        },
    }


class FakeWorker:
    def __init__(self, *, running=True, session=SESSION, sequence=17):
        self.running = running
        self.session = session
        self.sequence = sequence
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def call_endpoint(world, worker):
    runtime = SimpleNamespace(world_model=world, lidar_worker=worker)
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/lidar/transient"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.send_json = Mock()
    handler.do_GET()
    status, payload = handler.send_json.call_args.args
    return status, payload, handler


def test_fresh_snapshot_is_returned_intact_with_exact_session(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    published = snapshot()
    world.publish_lidar_obstacles(published)
    worker = FakeWorker()

    status, result, _ = call_endpoint(world, worker)

    assert status == 200
    assert result["read_only"] is True
    assert result["worker_running"] is True
    assert result["producer_session"] == SESSION
    assert result["acquisition_sequence"] == 17
    returned = result["snapshot"]
    assert returned["producer_session"] == SESSION
    assert returned["reason"] == "fresh"
    assert returned["sectors"] == published["sectors"]
    assert returned["sectors"]["left"]["robust_clearance_m"] == 1.60
    assert returned["sectors"]["left"]["minimum_clearance_m"] == 1.35
    assert returned["sectors"]["front"]["state"] == "CAUTION"


def test_world_model_reader_uses_current_worker_session(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    world.publish_lidar_obstacles(snapshot())
    worker = FakeWorker(session="exact-current-session")
    world.publish_lidar_obstacles(snapshot(session=worker.session))
    original = world.get_lidar_obstacles
    calls = []

    def reader(*, expected_session, now=None):
        calls.append(expected_session)
        return original(expected_session=expected_session, now=now)

    world.get_lidar_obstacles = reader
    _, result, _ = call_endpoint(world, worker)

    assert calls == ["exact-current-session"]
    assert result["producer_session"] == "exact-current-session"
    assert result["snapshot"]["producer_session"] == "exact-current-session"


def test_stale_snapshot_remains_stale_and_is_not_rejuvenated(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    world.publish_lidar_obstacles(
        snapshot(received=time.monotonic() - 1.0, age=0.01)
    )

    _, result, _ = call_endpoint(world, FakeWorker())

    assert result["snapshot"]["valid"] is False
    assert result["snapshot"]["available"] is False
    assert result["snapshot"]["reason"] == "stale"


def test_unavailable_without_snapshot_is_structured(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))

    status, result, _ = call_endpoint(world, FakeWorker(running=False))

    assert status == 200
    assert result["worker_running"] is False
    assert result["producer_session"] == SESSION
    assert isinstance(result["snapshot"], dict)
    assert result["snapshot"]["available"] is False
    assert result["snapshot"]["valid"] is False


def test_session_mismatch_is_exposed_as_untrusted(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    world.publish_lidar_obstacles(snapshot(session="old-session"))

    _, result, _ = call_endpoint(world, FakeWorker(session=SESSION))

    assert result["snapshot"]["available"] is False
    assert result["snapshot"]["valid"] is False
    assert result["snapshot"]["reason"] == "producer_session_mismatch"


def test_partial_snapshot_is_surfaced_without_inventing_sectors(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    partial = {
        "available": True,
        "valid": True,
        "reason": "fresh",
        "producer_session": SESSION,
        "received_monotonic_seconds": time.monotonic(),
        "age_at_receipt_seconds": 0.01,
        "sectors": {"front": sector()},
    }
    world.publish_lidar_obstacles(partial)

    _, result, _ = call_endpoint(world, FakeWorker())

    assert result["snapshot"]["sectors"] == {"front": sector()}
    assert "left" not in result["snapshot"]["sectors"]


def test_transient_reader_failure_is_structured_without_side_effects():
    world = SimpleNamespace(
        get_lidar_obstacles=Mock(side_effect=RuntimeError("cache unavailable")),
    )
    worker = FakeWorker()

    status, result, _ = call_endpoint(world, worker)

    assert status == 200
    assert result["snapshot"] is None
    assert result["error"] == "cache unavailable"
    world.get_lidar_obstacles.assert_called_once_with(
        expected_session=SESSION,
    )


def test_endpoint_is_json_serializable_and_has_no_motion_side_effects(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    world.publish_lidar_obstacles(snapshot())
    worker = FakeWorker()
    runtime = SimpleNamespace(
        world_model=world,
        lidar_worker=worker,
        behavior_manager=SimpleNamespace(execute_guarded_turn=Mock()),
        robot_client=SimpleNamespace(motion=Mock(), stop=Mock()),
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/lidar/transient"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.send_json = Mock()

    handler.do_GET()
    _, result = handler.send_json.call_args.args

    json.dumps(result)
    runtime.behavior_manager.execute_guarded_turn.assert_not_called()
    runtime.robot_client.motion.assert_not_called()
    runtime.robot_client.stop.assert_not_called()
    assert worker.start_calls == 0
    assert worker.stop_calls == 0
