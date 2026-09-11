"""Runtime lifecycle integration with fake acquisition and no sockets/ROS."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from lidar_perception import LidarPerceptionWorker
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler
from test_lidar_perception import payload
from world_model import WorldModel


@pytest.fixture(autouse=True)
def forbid_network():
    with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
         patch("robot_bridge.client.RobotBridgeClient._request", side_effect=AssertionError("bridge forbidden")):
        yield


def make_runtime(tmp_path, factory=None):
    return CognitiveRuntime(
        provider=object(), world_model=WorldModel(str(tmp_path / "world.json")),
        vision_adapter=object(),
        robot_client=SimpleNamespace(base_url="http://robot.invalid", stop=Mock(return_value={"ok": True})),
        behavior_manager=Mock(), loop_interval=0,
        lidar_worker_factory=factory,
    )


def test_construction_one_worker_no_io_and_run_once_does_not_start(tmp_path):
    factory = Mock(side_effect=LidarPerceptionWorker)
    runtime = make_runtime(tmp_path, factory)
    worker = runtime.lidar_worker
    factory.assert_called_once_with(runtime.world_model, base_url="http://robot.invalid")
    for _ in range(3):
        runtime.run_once()
    assert runtime.lidar_worker is worker
    assert not worker.running
    assert worker.sequence == 0
    status = runtime.get_status()["lidar_perception"]
    assert status["producer_session"] == worker.session
    assert not status["available"] and not status["valid"]
    runtime.stop()


@pytest.mark.parametrize("raises", [False, True])
def test_lifecycle_starts_once_and_always_stops(tmp_path, raises):
    acquired = threading.Event()

    def fetch():
        acquired.set()
        return payload()

    runtime = make_runtime(tmp_path, lambda world, **kw: LidarPerceptionWorker(world, fetch=fetch, **kw))
    worker = runtime.lidar_worker
    original_stop = worker.stop

    def cycle():
        assert acquired.wait(1)
        runtime._start_lidar()  # Repeated startup cannot spawn a second producer.
        if raises:
            raise RuntimeError("cycle failed")
        runtime.stop()

    runtime.run_once = cycle
    with patch.object(worker, "start", wraps=worker.start) as start, \
         patch.object(worker, "stop", wraps=original_stop) as stop:
        if raises:
            with pytest.raises(RuntimeError, match="cycle failed"):
                runtime.run_forever()
        else:
            runtime.run_forever()
        runtime.stop()
        runtime._stop_lidar()
        start.assert_called_once()
        assert stop.called
    assert not worker.running
    assert runtime.world_model.get_lidar_obstacles(expected_session=worker.session)["reason"] == "stopped"
    runtime.robot_client.stop.assert_called_once()  # Existing runtime finalizer only.


def test_replacement_session_invalidates_old_state(tmp_path):
    factory = lambda world, **kw: LidarPerceptionWorker(world, fetch=payload, **kw)
    first = make_runtime(tmp_path, factory)
    first.lidar_worker.run_once()
    first.stop()
    second = make_runtime(tmp_path, factory)
    assert first.lidar_worker.session != second.lidar_worker.session
    assert first.world_model.get_lidar_obstacles(expected_session=first.lidar_worker.session)["reason"] == "stopped"
    assert not second.get_status()["lidar_perception"]["valid"]
    second.stop()


@pytest.mark.parametrize("failure", [False, True])
def test_background_acquisition_and_status(tmp_path, failure):
    published = threading.Event()

    def fetch():
        if failure:
            raise OSError("offline")
        return payload()

    runtime = make_runtime(tmp_path, lambda world, **kw: LidarPerceptionWorker(world, fetch=fetch, **kw))
    publish = runtime.world_model.publish_lidar_obstacles

    def notify(state):
        publish(state)
        published.set()

    runtime.world_model.publish_lidar_obstacles = notify

    def cycle():
        assert published.wait(1)
        status = runtime.get_status()["lidar_perception"]
        assert status["running"]
        assert status["acquisition_sequence"] >= 1
        assert status["valid"] is not failure
        assert status["available"] is not failure
        assert status["reason"] == ("offline" if failure else "fresh")
        assert status["front_state"] == ("UNKNOWN" if failure else "CLEAR")
        if not failure:
            assert status["effective_age_seconds"] >= 0.05
        runtime.stop()

    runtime.run_once = cycle
    runtime.run_forever()
    assert runtime.get_status()["lidar_perception"]["reason"] == "stopped"


def test_worker_start_failure_does_not_crash_runtime(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.lidar_worker.start = Mock(side_effect=RuntimeError("thread failed"))
    runtime.run_once = runtime.stop
    runtime.run_forever()
    assert runtime.get_status()["lidar_perception"]["last_error"] == "thread failed"
    assert not runtime.world_model.get_lidar_obstacles(expected_session=runtime.lidar_worker.session)["valid"]


def test_factory_failure_is_observable_and_runtime_continues(tmp_path):
    runtime = make_runtime(tmp_path, Mock(side_effect=RuntimeError("creation failed")))
    runtime.run_once = runtime.stop
    runtime.run_forever()
    status = runtime.get_status()["lidar_perception"]
    assert not status["valid"] and status["producer_session"] is None
    assert status["last_error"] == "creation failed"


def test_existing_status_route_exposes_summary_without_network(tmp_path):
    runtime = make_runtime(tmp_path)
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/status"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.send_json = Mock()
    handler.do_GET()
    status, result = handler.send_json.call_args.args
    assert status == 200
    assert result["lidar_perception"]["producer_session"] == runtime.lidar_worker.session
    assert "sectors" not in result["lidar_perception"]
    runtime.stop()


def test_startup_exception_and_existing_robot_stop_failure_still_cleanup(tmp_path):
    runtime = make_runtime(tmp_path)
    update = runtime.world_model.update_robot_state

    def fail_startup(**values):
        if values.get("runtime_state") == "STARTING":
            raise RuntimeError("startup failed")
        return update(**values)

    runtime.world_model.update_robot_state = fail_startup
    runtime.robot_client.stop.side_effect = OSError("fake stop failure")
    with pytest.raises(RuntimeError, match="startup failed"):
        runtime.run_forever()
    assert runtime.get_status()["lidar_perception"]["reason"] == "stopped"
    assert not runtime.lidar_worker.running


def test_existing_runtime_api_suite_with_in_memory_transport():
    """Reuse existing API assertions without creating its HTTP test server."""
    import io
    import json
    from urllib.parse import urlparse
    import test_runtime_api as suite

    server = SimpleNamespace(
        server_address=("localhost", 0), serve_forever=lambda: None,
        shutdown=lambda: None, server_close=lambda: None,
        config_manager=SimpleNamespace(get_config=lambda: {"robot": {"id": "mayday"}}),
    )

    def create_server(runtime, **kwargs):
        server.runtime = runtime
        return server

    def request(method, url, payload=None):
        handler = object.__new__(RuntimeAPIHandler)
        handler.path = urlparse(url).path
        handler.server = server
        raw = json.dumps(payload).encode() if payload is not None else b""
        handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler.send_json = Mock()
        getattr(handler, "do_" + method)()
        return handler.send_json.call_args.args

    with patch.object(suite, "create_server", side_effect=create_server), \
         patch.object(suite, "request_json", side_effect=request):
        suite.main()
