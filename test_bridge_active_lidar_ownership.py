"""Offline cross-layer safety: real Bridge manager, ROS serialization and C++ gate.

Requires the retained Bridge test fixtures on PYTHONPATH and CHAMP_GUARD_CLI.
No ROS initialization, sockets, services or actuator interfaces are used.
"""
import copy
import io
import json
import threading
import time
import urllib.error
from unittest.mock import Mock

import pytest

# Cross-repository fixtures are provided only by the isolated offline launcher.
pytest.importorskip('champ_heartbeat', reason='requires pinned offline Bridge fixtures')
pytest.importorskip('test_champ_heartbeat', reason='requires offline native watchdog fixture')

from behavior_manager import BehaviorManager, _GuardedTurnMonitor
from guarded_turn_policy import LEGACY_BROAD_SIDE, ROTATIONAL_SWEPT_FOOTPRINT
from robot_bridge.client import RobotBridgeClient
from robot_bridge.forward_interlock import ForwardMotionInterlock
from test_forward_interlock import state
from test_guarded_turn_execution import FakeWorldModel, snapshot
from test_champ_heartbeat import clear_lidar, gate, rig
from champ_heartbeat import START, HEARTBEAT, STOP, EMERGENCY


AXES = [(0.1, 0., 0.), (-0.1, 0., 0.), (0., 0.08, 0.),
        (0., -0.08, 0.), (0., 0., 0.25), (0., 0., -0.25)]


class Reply(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connected(rig, monkeypatch, *, bridge_owner=True):
    """Use the actual client decoder; only urllib transport is simulated."""
    g, node, adapter, manager = rig
    current = state()
    requests = []
    client = RobotBridgeClient(base_url='http://offline.invalid')
    interlock = ForwardMotionInterlock(
        lambda **_: copy.deepcopy(current), expected_session='s',
        stop_callback=client.stop, bridge_guarded_bounded=bridge_owner,
        monotonic=lambda: g.clock()/1e9)
    client.configure_forward_interlock(interlock)
    interlock.refresh()

    def respond(request, **_):
        path = request.full_url.removeprefix(client.base_url)
        requests.append((request.method, path))
        if path == '/status':
            body = dict(ok=True, status='READY', ros_ready=True,
                        motion_capabilities=dict(linear_y=True, max_linear_y=.1))
            return Reply(json.dumps(body).encode())
        assert request.method == 'POST' and path in ('/motion', '/stop')
        if path == '/stop':
            return Reply(json.dumps(manager.emergency_stop()).encode())
        payload = json.loads(request.data)
        axes = tuple(payload.get(k, 0.) for k in ('linear_x', 'linear_y', 'angular_z'))
        body = manager.execute(*axes, payload['duration'])
        body.update(action='motion', mode='bounded', returned_immediately=False,
                    automatic_stop=body.get('controller_gait_zero', False),
                    duration=payload['duration'], linear_x=axes[0],
                    linear_y=axes[1], angular_z=axes[2])
        raw = json.dumps(body).encode()
        if body['ok']:
            return Reply(raw)
        code = 409 if body['state'] == 'refused' else 503
        raise urllib.error.HTTPError(request.full_url, code, 'simulated Bridge', {}, io.BytesIO(raw))

    monkeypatch.setattr('urllib.request.urlopen', respond)
    return client, interlock, current, requests


def move(client, axes, duration=.5):
    return client.motion(linear_x=axes[0], linear_y=axes[1], angular_z=axes[2], duration=duration)


@pytest.mark.parametrize('axes', AXES)
@pytest.mark.parametrize('observation', ['stale', 'missing', 'invalid', 'exception', 'session'])
def test_http_observation_cannot_stop_native_fresh_motion(rig, monkeypatch, axes, observation):
    g, node, _, manager = rig
    client, interlock, current, requests = connected(rig, monkeypatch)
    start = g.clock()
    observed = []
    def scan():
        if g.clock() - start > 90_000_000:
            if observation == 'stale': current['effective_age_seconds'] = .53692
            elif observation == 'missing': current['available'] = False
            elif observation == 'invalid': current['valid'] = False
            elif observation == 'session': current['producer_session'] = 'different'
            else:
                def failed(**_): raise TimeoutError('delayed HTTP acquisition')
                interlock.reader = failed
            observed.append(interlock.refresh())
        return clear_lidar(g.clock)
    manager.lidar_reader = scan
    result = move(client, axes)
    assert observed and all(not permitted for permitted, _ in observed)
    assert result['ok'] and result['state'] == 'completed'
    assert result['completed_progress_credit'] and result['evidence_valid']
    assert all(path != '/stop' for _, path in requests)
    assert node.messages[-1]['operation'] == STOP
    assert node.messages[-1]['issued_ns'] >= result['deadline_ns']
    assert not g.last['active'] and g.last['gait_zero']
    # The old HTTP observation cannot authorize a successor either.
    before = list(requests)
    assert not move(client, axes)['ok']
    assert [r for r in requests if r[0] == 'POST'] == [r for r in before if r[0] == 'POST']
    assert not interlock.status()['pending_forward']
    assert not interlock.status()['active_forward']


@pytest.mark.parametrize('axes', AXES)
@pytest.mark.parametrize('fault', ['intrusion', 'stale', 'frozen', 'missing', 'invalid', 'coverage', 'reader_lost'])
def test_native_fault_stops_every_direction_and_never_completes(rig, monkeypatch, axes, fault):
    g, node, _, manager = rig
    client, _, _, requests = connected(rig, monkeypatch)
    start = g.clock()
    frozen = clear_lidar(g.clock)
    def scan():
        sample = clear_lidar(g.clock)
        if fault == 'frozen':
            sample = copy.deepcopy(frozen)
            sample['age_seconds'] = g.clock()/1e9 - sample['received_monotonic_seconds']
        elif g.clock()-start > 90_000_000:
            if fault == 'intrusion': sample['scan']['ranges'][20] = .4
            elif fault == 'stale': sample['age_seconds'] = .301
            elif fault == 'missing': sample['available'] = False
            elif fault == 'invalid': sample['scan']['frame_id'] = 'invalid'
            elif fault == 'coverage': sample['scan']['ranges'][:20] = [None]*20
            elif fault == 'reader_lost': manager.lidar_reader = None
        return sample
    manager.lidar_reader = scan
    result = move(client, axes, .8)
    assert result['evidence_valid'] and result['state'] == 'interrupted'
    assert result['reason'].startswith('active_lidar_')
    assert not result['ok'] and not result['completed_progress_credit']
    assert result['stop_certificate']['gait_zero'] and result['controller_gait_zero']
    assert node.messages[-1]['operation'] == STOP
    assert node.messages[-1]['issued_ns'] < result['deadline_ns']
    assert not g.last['active'] and g.last['gait_zero']
    assert sum(path == '/motion' for _, path in requests) == 1


@pytest.mark.parametrize('axes', AXES)
def test_absent_native_guard_reader_refuses_without_start(rig, monkeypatch, axes):
    _, node, _, manager = rig
    client, _, _, _ = connected(rig, monkeypatch)
    manager.lidar_reader = None
    result = move(client, axes)
    assert not result['ok'] and not result['transport_attempted']
    assert not result['completed_progress_credit']
    assert result['reason'] == 'active_lidar_unavailable'
    assert not any(p['operation'] == START for p in node.messages)


@pytest.mark.parametrize('axes', AXES)
def test_controller_expiration_is_not_promoted_to_completion(rig, monkeypatch, axes):
    g, node, _, _ = rig
    client, _, _, _ = connected(rig, monkeypatch)
    def stall(packet):
        if packet['operation'] == HEARTBEAT:
            g.clock.sleep(.26)
            g.tick()
    node.hook = stall
    result = move(client, axes, .8)
    assert not result['ok'] and not result['completed_progress_credit']
    assert not g.last['active'] and g.last['gait_zero']


@pytest.mark.parametrize('axes', AXES)
def test_scoped_stop_uncertainty_quarantines_successors(rig, monkeypatch, axes):
    g, node, adapter, manager = rig
    client, interlock, _, _ = connected(rig, monkeypatch)
    original = adapter.snapshot
    def wrong_stop():
        sample, received = original()
        if sample.get('last_operation') == STOP:
            sample = dict(sample, last_request_issued_ns=sample['last_request_issued_ns']-1)
        return sample, received
    adapter.snapshot = wrong_stop
    # Failed Cognitive emergency acknowledgement must also remain latched.
    node.fail = lambda p: p['operation'] == EMERGENCY
    result = move(client, axes, .1)
    assert not result['ok'] and not result['completed_progress_credit']
    assert manager.latched and interlock.status()['last_stop_error']
    interlock.refresh()
    assert move(client, axes)['error'] == 'forward_interlock_stop_unverified'
    assert not g.last['active'] and g.last['gait_zero']


@pytest.mark.parametrize('axes', AXES)
@pytest.mark.parametrize('cause', ['operator', 'mission_cancel', 'shutdown'])
def test_explicit_stop_and_shutdown_still_cancel_without_credit(rig, monkeypatch, axes, cause):
    g, node, _, _ = rig
    client, interlock, _, _ = connected(rig, monkeypatch)
    def stop(packet):
        if packet['operation'] == START:
            if cause == 'shutdown': interlock.stop()
            else: assert client.stop()['ok']
    node.hook = stop
    result = move(client, axes)
    assert not result['ok'] and not result['completed_progress_credit']
    assert not g.last['active'] and g.last['gait_zero']
    assert any(p['operation'] == EMERGENCY for p in node.messages)


def monitor(robot, *, mode=ROTATIONAL_SWEPT_FOOTPRINT):
    world = FakeWorldModel(snapshot())
    turn = _GuardedTurnMonitor(
        world_model=world, robot=robot, direction='LEFT', angular_speed=.25,
        duration=.5, expected_session='session-1', generation=1,
        initial_validation={'permitted': True, 'reason': 'initial'}, now=10.,
        safety_mode=mode)
    return turn, world


def tick(turn):
    # Deterministic one-iteration monitor; no thread or timing sleeps needed.
    turn._stop.wait = lambda _: turn._stop.set()
    turn._run()


@pytest.mark.parametrize('fault', ['stale', 'missing', 'geometry', 'session'])
def test_native_turn_monitor_keeps_lifecycle_without_http_veto(fault):
    client = RobotBridgeClient(base_url='http://offline.invalid')
    client.stop = Mock(return_value={'ok': True})
    turn, world = monitor(client)
    assert turn.begin_transport(time.monotonic())
    world.state.update(valid=False, available=False, reason=fault)
    tick(turn)
    assert not turn.is_invalidated()
    assert turn.status()['active_lidar_safety_owner'] == 'robot_bridge'
    assert not world.calls and not client.stop.called


def test_native_turn_still_validates_before_transport():
    client = RobotBridgeClient(base_url='http://offline.invalid')
    client.stop = Mock(return_value={'ok': True})
    turn, world = monitor(client)
    world.state.update(valid=False, available=False, reason='invalid')
    tick(turn)
    assert turn.is_invalidated() and not turn.begin_transport(0.)
    assert world.calls and client.stop.called


def test_non_equivalent_legacy_turn_policy_is_retained():
    client = RobotBridgeClient(base_url='http://offline.invalid')
    client.stop = Mock(return_value={'ok': True})
    turn, world = monitor(client, mode=LEGACY_BROAD_SIDE)
    turn.begin_transport(time.monotonic())
    world.state.update(valid=False, available=False, reason='invalid')
    tick(turn)
    assert turn.is_invalidated() and client.stop.called
    assert turn.status()['active_lidar_safety_owner'] == 'cognitive'


def test_native_turn_retains_deadline_and_operator_stop_fences():
    client = RobotBridgeClient(base_url='http://offline.invalid')
    client.stop = Mock(return_value={'ok': True})
    turn, _ = monitor(client)
    turn.begin_transport(time.monotonic()-.55)
    tick(turn)
    assert turn.status()['deadline_stop_attempted'] and client.stop.called
    assert turn.cancel('operator_stop')
    invalidated, _, stop = turn.finalize()
    assert invalidated and stop[0]['ok']
    assert client.stop.call_count == 2


def test_legacy_success_cannot_claim_native_guarded_completion(monkeypatch):
    client = RobotBridgeClient(base_url='http://offline.invalid')
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: Reply(b'{"ok":true}'))
    result = client.motion(angular_z=.25, duration=.5)
    assert not result['ok'] and not result['completed_progress_credit']
    assert result['delivery_uncertain'] and not result['evidence_valid']


def test_native_health_delay_preserves_pending_shutdown_and_failed_stop_fence():
    current = state()
    stop = Mock(return_value={'ok': False})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session='s',
        stop_callback=stop, bridge_guarded_bounded=True)
    interlock.refresh()
    dispatch = interlock.begin_positive_dispatch(streaming=False)
    current['effective_age_seconds'] = 1.
    interlock.refresh()
    stop.assert_not_called()
    interlock.stop()
    assert stop.call_count == 1
    assert not interlock.finalize_positive_dispatch(dispatch, {
        'ok': True, 'evidence_valid': True, 'delivery_uncertain': False})
    assert stop.call_count == 2 and interlock.status()['last_stop_error']
    assert not interlock.dispatch_outcome(dispatch)['stop_succeeded']


def test_native_mode_never_relaxes_streaming_health_guard():
    current = state()
    stop = Mock(return_value={'ok': True})
    interlock = ForwardMotionInterlock(lambda **_: current, expected_session='s',
        stop_callback=stop, bridge_guarded_bounded=True)
    interlock.refresh()
    dispatch = interlock.begin_positive_dispatch(streaming=True)
    current['effective_age_seconds'] = .301
    interlock.refresh()
    assert stop.call_count == 1
    assert not interlock.finalize_positive_dispatch(dispatch, {'ok': True})
    assert stop.call_count == 2


@pytest.mark.parametrize('direction', ['LEFT', 'RIGHT'])
def test_guarded_turn_execution_uses_native_safety_and_keeps_prechecks(rig, monkeypatch, direction):
    g, _, _, native = rig
    client, interlock, current, requests = connected(rig, monkeypatch)
    world = FakeWorldModel(snapshot())
    behavior = BehaviorManager(robot_client=client, world_model=world)
    start = g.clock()
    def scan():
        if g.clock()-start > 90_000_000:
            current.update(valid=False, reason='HTTP delayed')
            world.state.update(valid=False, available=False, reason='stale')
            interlock.refresh()
        return clear_lidar(g.clock)
    native.lidar_reader = scan
    result = behavior.execute_guarded_turn(direction, .25, .15,
        expected_lidar_session='session-1', now=10.,
        safety_mode=ROTATIONAL_SWEPT_FOOTPRINT)
    assert result['ok'] and result['transport_result']['completed_progress_credit']
    assert result['active_lidar_safety_owner'] == 'robot_bridge'
    assert not result['generation_invalidated']
    assert not any(path == '/stop' for _, path in requests)
    before = [r for r in requests if r[0] == 'POST']
    denied = behavior.execute_guarded_turn(direction, .25, .15,
        expected_lidar_session='session-1', now=10.,
        safety_mode=ROTATIONAL_SWEPT_FOOTPRINT)
    assert not denied['ok'] and not denied['transport_attempted']
    assert [r for r in requests if r[0] == 'POST'] == before


@pytest.mark.parametrize('kind', ['native', 'custom', 'subclass'])
def test_runtime_selects_native_ownership_only_for_guarded_client(tmp_path, kind):
    from types import SimpleNamespace
    from runtime import CognitiveRuntime
    from world_model import WorldModel
    class CustomClient(RobotBridgeClient):
        pass
    native = kind == 'native'
    if kind == 'custom':
        client = SimpleNamespace(base_url='http://offline.invalid', stop=Mock(return_value={'ok': True}))
    else:
        client = (RobotBridgeClient if native else CustomClient)(base_url='http://offline.invalid')
    runtime = CognitiveRuntime(provider=object(),
        world_model=WorldModel(str(tmp_path/'world.json')), vision_adapter=object(),
        robot_client=client, behavior_manager=Mock())
    assert runtime.forward_interlock.bridge_guarded_bounded is native
    if native:
        assert client.forward_interlock is runtime.forward_interlock
    assert not runtime.lidar_worker.running


def test_delayed_acquisition_keeps_native_command_running(rig, monkeypatch, tmp_path):
    from lidar_perception import LidarPerceptionWorker
    from world_model import WorldModel
    g, _, _, native = rig
    client, interlock, _, requests = connected(rig, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    delayed = False
    def fetch():
        # Simulated HTTP acquisition, with the production age/World Model path.
        telemetry = clear_lidar(g.clock)
        telemetry['scan']['stamp_seconds'] = g.clock()/1e9
        if delayed:
            entered.set()
            assert release.wait(2.)
        return dict(ok=True, telemetry=telemetry)
    world = WorldModel(str(tmp_path/'world.json'))
    worker = LidarPerceptionWorker(world, fetch=fetch, monotonic=lambda: g.clock()/1e9)
    assert worker.run_once()['valid']
    interlock.expected_session = worker.session
    interlock.reader = lambda **kw: world.get_lidar_obstacles(**kw, now=g.clock()/1e9)
    assert interlock.refresh()[0]
    delayed = True
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    samples = []
    def scan():
        samples.append(interlock.refresh())
        return clear_lidar(g.clock)
    native.lidar_reader = scan
    try:
        assert entered.wait(1.)
        result = move(client, AXES[0], .5)
        assert result['ok'] and result['completed_progress_credit']
        assert any(not valid for valid, _ in samples)
        assert not any(path == '/stop' for _, path in requests)
        assert not g.last['active'] and g.last['gait_zero']
    finally:
        release.set()
        thread.join(2.)
    assert not thread.is_alive()
    assert not world.get_lidar_obstacles(expected_session=worker.session, now=g.clock()/1e9)['valid']


def test_invalidated_client_result_cannot_keep_completion_credit():
    client = RobotBridgeClient(base_url='http://offline.invalid')
    interlock = ForwardMotionInterlock(lambda **_: state(), expected_session='s',
        stop_callback=lambda: {'ok': True}, bridge_guarded_bounded=True)
    client.configure_forward_interlock(interlock)
    interlock.refresh()
    def transport(*_):
        interlock.stop()
        return dict(ok=True, completed_progress_credit=True, evidence_valid=True,
                    delivery_uncertain=False)
    client._request = transport
    result = move(client, AXES[0])
    assert not result['ok'] and not result['completed_progress_credit']
    assert result['transport_result']['completed_progress_credit']


def test_runtime_operator_stop_cancels_active_and_queued_missions():
    from test_runtime_stop_preemption import main
    main()  # Existing fake clients, temporary state, and in-flight STOP race.


@pytest.mark.parametrize('bridge_owner', [False, True])
def test_retained_304ms_alignment_observation_owner_comparison(rig, monkeypatch, bridge_owner, record_property):
    g, _, _, native = rig
    client, interlock, current, requests = connected(rig, monkeypatch, bridge_owner=bridge_owner)
    start = g.clock()
    def scan():
        if g.clock()-start > 90_000_000:
            current['effective_age_seconds'] = .304518967008
            interlock.refresh()
        sample = clear_lidar(g.clock)
        sample.update(age_seconds=.073700528,
                      received_monotonic_seconds=g.clock()/1e9-.073700528)
        return sample
    native.lidar_reader = scan
    result = move(client, AXES[-1], .5)
    assert bool(result['ok']) is bridge_owner
    assert bool(result['completed_progress_credit']) is bridge_owner
    assert any(path == '/stop' for _, path in requests) is (not bridge_owner)
    assert sum(path == '/motion' for _, path in requests) == 1
    assert not g.last['active'] and g.last['gait_zero']
    record_property('retained_timing_counterfactual', json.dumps({
        'bridge_owner': bridge_owner, 'http_age_seconds': .304518967008,
        'native_age_seconds': .073700528, 'ok': result['ok'],
        'completed_progress_credit': result['completed_progress_credit'],
        'cognitive_stop_requests': sum(path == '/stop' for _, path in requests)}))


@pytest.mark.parametrize('fails', [False, True])
def test_native_pending_stop_reservation_still_fences_completion_and_successor(fails):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2.)
        return {'ok': not fails}
    interlock = ForwardMotionInterlock(lambda **_: state(), expected_session='s',
        stop_callback=stop, bridge_guarded_bounded=True)
    interlock.refresh()
    dispatch = interlock.begin_positive_dispatch(streaming=False)
    thread = threading.Thread(target=interlock._dispatch_stop)
    thread.start()
    try:
        assert entered.wait(1.)
        interlock.refresh()
        with pytest.raises(PermissionError, match='stop_pending'):
            interlock.begin_positive_dispatch(streaming=False)
        assert not interlock.finalize_positive_dispatch(dispatch,
            dict(ok=True, evidence_valid=True, delivery_uncertain=False))
        outcome = interlock.dispatch_outcome(dispatch)
        assert not outcome['valid']
    finally:
        release.set()
        thread.join(2.)
    assert not thread.is_alive()
    interlock.refresh()
    if fails:
        with pytest.raises(PermissionError, match='stop_unverified'):
            interlock.begin_positive_dispatch(streaming=False)


@pytest.mark.parametrize('axes', AXES)
@pytest.mark.parametrize('intrusion', [False, True])
def test_current_client_through_unchanged_flask_routes(rig, monkeypatch, axes, intrusion):
    from types import SimpleNamespace
    import app as bridge  # Import only; launcher denies service subprocesses/ROS init.
    g, _, _, native = rig
    client, interlock, current, _ = connected(rig, monkeypatch)
    monkeypatch.setattr(bridge, 'ros_ready', True)
    monkeypatch.setattr(bridge, 'publisher_node', SimpleNamespace(champ_commands=native))
    local = bridge.app.test_client()
    requests = []
    def respond(request, **_):
        path = request.full_url.removeprefix(client.base_url)
        requests.append((request.method, path))
        response = local.open(path, method=request.method,
            json=json.loads(request.data) if request.data else None)
        if response.status_code != 200:
            raise urllib.error.HTTPError(request.full_url, response.status_code,
                'in-process Flask', {}, io.BytesIO(response.data))
        return Reply(response.data)
    monkeypatch.setattr('urllib.request.urlopen', respond)
    start = g.clock()
    def scan():
        sample = clear_lidar(g.clock)
        if g.clock()-start > 90_000_000:
            current['effective_age_seconds'] = .304518967008
            interlock.refresh()
            if intrusion: sample['scan']['ranges'][20] = .4
        return sample
    native.lidar_reader = scan
    result = move(client, axes)
    assert result['evidence_valid'] and bool(result['ok']) is (not intrusion)
    assert result['completed_progress_credit'] is (not intrusion)
    assert result['state'] == ('interrupted' if intrusion else 'completed')
    assert result['controller_gait_zero'] and not g.last['active']
    assert not any(path == '/stop' for _, path in requests)
    assert sum(path == '/motion' for _, path in requests) == 1
