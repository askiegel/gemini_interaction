"""Offline worker tests with fake telemetry, clocks, and temporary World Models."""

import copy
import json
import math
import threading
from unittest.mock import Mock, patch

import pytest

from lidar_perception import LidarPerceptionWorker, read_lidar_state, unavailable_state
from voice_relay.lidar_sectors import lidar_sector_payload
from world_model import WorldModel


def payload(stamp=100, age=0.05):
    return {"ok": True, "telemetry": {
        "available": True, "age_seconds": age,
        "scan": {"frame_id": "lidar_link", "stamp_seconds": stamp,
                 "angle_min": -math.pi, "angle_increment": math.pi / 4,
                 "range_min": 0.1, "range_max": 10,
                 "ranges": [0.3, 0.5, 1.3, None, 0.8]},
    }}


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup(tmp_path):
    clock = Clock()
    source = payload()
    world = WorldModel(str(tmp_path / "world.json"))

    def fetch():
        clock.now += 0.02
        return copy.deepcopy(source)

    worker = LidarPerceptionWorker(world, fetch=fetch, monotonic=clock, wall_clock=clock)
    return world, worker, clock, source


def test_faithful_sectors_metadata_and_age(setup):
    world, worker, clock, source = setup
    state = worker.run_once()
    assert state["valid"] and state["available"]
    assert state["sectors"] == lidar_sector_payload(source)["sectors"]
    assert {s["state"] for s in state["sectors"].values()} == {"CLEAR", "CAUTION", "BLOCKED", "UNKNOWN"}
    assert state["source"]["frame_id"] == "lidar_link"
    assert state["source"]["stamp_seconds"] == 100
    assert state["source"]["age_seconds"] == 0.05
    assert state["request_latency_seconds"] == pytest.approx(0.02)
    assert state["effective_age_seconds"] == pytest.approx(0.07)
    assert state["acquisition_started_at"] == 10
    assert state["completed_at"] == clock.now
    assert state["producer_session"] == worker.session
    assert state["acquisition_sequence"] == 1
    clock.now += 0.1
    current = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)
    assert current["effective_age_seconds"] == pytest.approx(0.17)


@pytest.mark.parametrize("age,valid", [(0.30, True), (0.3000001, False)])
def test_exact_boundary(setup, age, valid):
    _, worker, _, _ = setup
    state = worker.run_once()
    state.update(received_monotonic_seconds=0, age_at_receipt_seconds=age)
    assert read_lidar_state(state, expected_session=worker.session, now=0)["valid"] is valid


@pytest.mark.parametrize("age", [None, -0.1, math.nan, math.inf, -math.inf, "0.1", True])
def test_invalid_age(setup, age):
    world, worker, clock, source = setup
    source["telemetry"]["age_seconds"] = age
    assert not worker.run_once()["valid"]
    assert not world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["available"]
    del source["telemetry"]["age_seconds"]
    assert not worker.run_once()["valid"]


@pytest.mark.parametrize("raw", [None, [], {}, {"ok": True},
    {"ok": True, "telemetry": {"available": True, "scan": []}}])
def test_malformed_payload(setup, raw):
    _, worker, _, _ = setup
    worker._fetch = lambda: raw
    assert not worker.run_once()["valid"]


@pytest.mark.parametrize("field,value", [("frame_id", "map"), ("stamp_seconds", None),
    ("stamp_seconds", math.nan), ("stamp_seconds", -1), ("angle_increment", 0)])
def test_invalid_source(setup, field, value):
    _, worker, _, source = setup
    source["telemetry"]["scan"][field] = value
    assert not worker.run_once()["available"]


def test_acquisition_failure_overwrites_clear(setup):
    world, worker, clock, _ = setup
    worker.run_once()

    def fail():
        raise OSError("offline")

    worker._fetch = fail
    assert worker.run_once()["reason"] == "offline"
    assert not world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["valid"]


def test_duplicate_frozen_and_new_scan(setup):
    _, worker, clock, source = setup
    first = worker.run_once()
    clock.now += 0.08
    second = worker.run_once()
    assert second["valid"]
    assert second["age_at_receipt_seconds"] > first["age_at_receipt_seconds"]
    source["telemetry"]["age_seconds"] = 0  # Broken upstream age cannot rejuvenate.
    clock.now += 0.25
    frozen = worker.run_once()
    assert not frozen["valid"] and frozen["reason"] == "stale"
    source["telemetry"]["scan"]["stamp_seconds"] += 0.1
    assert worker.run_once()["valid"]
    source["telemetry"]["scan"]["stamp_seconds"] -= 1
    assert worker.run_once()["reason"] == "scan_stamp_regressed"


def test_session_restart_and_unrelated_updates(setup):
    world, worker, clock, _ = setup
    state = worker.run_once()
    assert not read_lidar_state(state, expected_session=None, now=clock.now)["valid"]
    assert not read_lidar_state(state, expected_session="new", now=clock.now)["valid"]
    other = WorldModel(world.storage_path)
    other.update_robot_state(battery=99)
    clock.now += 1
    assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["reason"] == "stale"
    replacement = LidarPerceptionWorker(other, fetch=lambda: payload(), monotonic=clock)
    assert replacement.session != worker.session
    assert not world.get_lidar_obstacles(expected_session=replacement.session, now=clock.now)["valid"]


def test_stop_during_background_request_cannot_republish(setup):
    world, worker, clock, _ = setup
    entered, release = threading.Event(), threading.Event()

    def fetch():
        entered.set()
        assert release.wait(2)
        return payload()

    worker._fetch = fetch
    worker.start()
    assert entered.wait(1)
    worker.stop()
    release.set()
    worker._thread.join(1)
    assert not worker._thread.is_alive()
    assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["reason"] == "stopped"
    with pytest.raises(RuntimeError):
        worker.start()


def test_transport_is_get_only(setup):
    _, worker, _, _ = setup
    worker._fetch = None
    worker._base_url = "http://robot.invalid"
    with patch("robot_bridge.client.RobotBridgeClient._request", return_value=payload()) as request:
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("process forbidden")):
            worker.run_once()
            worker.stop()
    request.assert_called_once_with("GET", "/telemetry/lidar")


def test_concurrent_world_updates_preserve_snapshot(setup):
    world, worker, clock, _ = setup
    other = WorldModel(world.storage_path)

    def write_other():
        for index in range(10):
            other.update_robot_state(counter=index)

    thread = threading.Thread(target=write_other)
    thread.start()
    for _ in range(10):
        worker.run_once()
    thread.join()
    world.reload()
    assert world.robot_state["counter"] == 9
    assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["producer_session"] == worker.session
    assert "lidar_obstacles" not in world.robot_state


def test_read_failure_and_clock_regression_fail_closed(setup):
    world, worker, clock, _ = setup
    state = worker.run_once()
    with patch.object(world, "reload", side_effect=OSError("unreadable")) as reload:
        assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["valid"]
        reload.assert_not_called()
    assert read_lidar_state(state, expected_session=worker.session, now=clock.now - 1)["reason"] == "invalid_freshness"


def test_failed_publication_cannot_extend_old_snapshot(setup):
    world, worker, clock, _ = setup
    worker.run_once()
    with patch.object(world, "publish_lidar_obstacles", side_effect=OSError("disk failure")):
        with pytest.raises(OSError):
            worker.run_once()
        worker.stop()
        assert worker.last_error == "disk failure"
    clock.now += 1
    assert not world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["valid"]


def test_publication_and_read_do_not_alias_caller_state(setup):
    world, worker, clock, _ = setup
    result = worker.run_once()
    result["sectors"]["front"]["state"] = "BROKEN"
    current = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)
    assert current["sectors"]["front"]["state"] == "CLEAR"
    current["sectors"]["front"]["state"] = "BROKEN"
    assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["sectors"]["front"]["state"] == "CLEAR"


def test_lidar_publish_and_read_do_not_persist_or_lock_disk(setup):
    world, worker, clock, _ = setup
    state = worker.run_once()
    with patch.object(world, "update_robot_state", side_effect=AssertionError("LiDAR used persisted state")), \
         patch.object(world, "reload", side_effect=AssertionError("LiDAR reloaded persisted state")), \
         patch.object(world, "save", side_effect=AssertionError("LiDAR saved persisted state")), \
         patch.object(world, "_open_lock_file", side_effect=AssertionError("LiDAR acquired file lock")):
        world.publish_lidar_obstacles(state)
        current = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)
    assert current["producer_session"] == worker.session


def test_lidar_publish_stores_independent_copy_and_read_returns_copy(setup):
    world, worker, clock, _ = setup
    state = worker.run_once()
    state["sectors"]["front"]["state"] = "BROKEN"
    current = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)
    assert current["sectors"]["front"]["state"] == "CLEAR"
    current["sectors"]["front"]["state"] = "BROKEN"
    assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["sectors"]["front"]["state"] == "CLEAR"


def test_persisted_lidar_snapshot_is_ignored_after_reload(tmp_path):
    path = tmp_path / "world.json"
    world = WorldModel(str(path))
    stale = unavailable_state("fresh", "old-session")
    world.update_robot_state(battery=88)
    persisted = json.loads(path.read_text())
    persisted["robot_state"]["lidar_obstacles"] = stale
    path.write_text(json.dumps(persisted))
    reloaded = WorldModel(str(path))
    assert "lidar_obstacles" not in reloaded.robot_state
    result = reloaded.get_lidar_obstacles(expected_session="old-session")
    assert not result["valid"]
    assert result["reason"] == "producer_session_mismatch"
    reloaded.update_robot_state(battery=87)
    assert "lidar_obstacles" not in json.loads(path.read_text())["robot_state"]


def test_new_world_model_has_no_authoritative_lidar_until_publish(tmp_path):
    world = WorldModel(str(tmp_path / "world.json"))
    result = world.get_lidar_obstacles(expected_session="new-session")
    assert not result["valid"]
    assert result["reason"] == "producer_session_mismatch"


def test_lidar_cache_is_not_coupled_to_slow_persistence(setup):
    world, worker, clock, _ = setup
    state = worker.run_once()
    slow = Mock(side_effect=AssertionError("LiDAR touched persistence"))
    with patch.object(world, "save", slow), patch.object(world, "reload", slow), patch.object(world, "_open_lock_file", slow):
        for _ in range(100):
            world.publish_lidar_obstacles(state)
            assert world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)["producer_session"] == worker.session
    slow.assert_not_called()


def test_reload_preserves_live_transient_lidar_over_legacy_persisted_state(setup):
    world, worker, clock, _ = setup
    published = worker.run_once()
    before = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now)
    before_signature = {
        "producer_session": before["producer_session"],
        "acquisition_sequence": before["acquisition_sequence"],
        "stamp_seconds": before["source"]["stamp_seconds"],
        "front_state": before["sectors"]["front"]["state"],
        "age": before["effective_age_seconds"],
    }

    world.update_robot_state(battery=88)
    world.robot_state["battery"] = 0
    with open(world.storage_path, encoding="utf-8") as state_file:
        persisted = json.load(state_file)
    persisted["robot_state"]["lidar_obstacles"] = {
        **published,
        "producer_session": "legacy-session",
        "acquisition_sequence": 9999,
        "source": {**published["source"], "stamp_seconds": 9999},
        "sectors": {**published["sectors"], "front": {"state": "BLOCKED", "available": True}},
    }
    with open(world.storage_path, "w", encoding="utf-8") as state_file:
        json.dump(persisted, state_file)

    world.reload()
    after = world.get_lidar_obstacles(expected_session=worker.session, now=clock.now + 0.10)
    assert world.robot_state["battery"] == 88
    assert after["producer_session"] == before_signature["producer_session"]
    assert after["acquisition_sequence"] == before_signature["acquisition_sequence"]
    assert after["source"]["stamp_seconds"] == before_signature["stamp_seconds"]
    assert after["sectors"]["front"]["state"] == before_signature["front_state"]
    assert after["effective_age_seconds"] == pytest.approx(before_signature["age"] + 0.10)
    assert after["effective_age_seconds"] != before_signature["age"]
    assert "lidar_obstacles" not in world.robot_state
