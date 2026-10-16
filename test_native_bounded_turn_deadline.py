"""Offline native lifecycle, with real Bridge manager/serialization/C++ expiry.

The isolated evidence launcher supplies pinned cross-repository fixtures. No
network, ROS initialization, services or actuators are used. Simulated clocks
make admission, execution and response delays independently reproducible.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import pytest

pytest.importorskip('test_champ_heartbeat', reason='requires pinned offline native fixtures')
import behavior_manager as behavior_module
from behavior_manager import BehaviorManager
from guarded_turn_policy import LEGACY_BROAD_SIDE, ROTATIONAL_SWEPT_FOOTPRINT
from test_bridge_active_lidar_ownership import AXES, Reply, connected, move
from test_guarded_turn_execution import FakeWorldModel, snapshot
from test_champ_heartbeat import clear_lidar, gate, rig
from champ_heartbeat import START, HEARTBEAT, STOP, EMERGENCY


def harness(rig, monkeypatch, *, admission_delay=0., response_delay=0., baseline=False,
            flask_route=False):
    g, node, _, native = rig
    client, interlock, current, requests = connected(rig, monkeypatch)
    client.timeout = 15.  # Existing Runtime's configured I/O timeout, unchanged.
    world = FakeWorldModel(snapshot())
    behavior = BehaviorManager(robot_client=client, world_model=world)
    monitors, packets, transport_times = [], [], []
    origin = g.clock()
    clock = lambda: 42677.647030759 + (g.clock()-origin)/1e9
    monkeypatch.setattr(behavior_module.time, 'monotonic', clock)
    monitor_type = behavior_module._GuardedTurnMonitor
    if baseline:
        path = os.environ.get('MAYDAY_DEADLINE_BASELINE_FILE')
        if not path:
            pytest.skip('retained baseline artifact required')
        spec = importlib.util.spec_from_file_location('retained_deadline_behavior', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monitor_type = module._GuardedTurnMonitor
        monkeypatch.setattr(behavior_module, '_GuardedTurnMonitor', monitor_type)
    monkeypatch.setattr(monitor_type, 'start', lambda self: monitors.append(self))
    original_sleep = g.clock.sleep
    checking = False
    def advance(seconds):
        nonlocal checking
        original_sleep(seconds)
        if not monitors or checking:
            return
        checking = True
        turn = monitors[0]
        try:
            turn._stop.clear()
            turn._stop.wait = lambda _: turn._stop.set()
            turn._run()  # One deterministic sampling tick, never a ROS/thread loop.
            turn._stop.clear()
        finally:
            checking = False
    native.sleep = advance
    original_transport = urllib.request.urlopen
    if flask_route:
        from types import SimpleNamespace
        import app as bridge
        monkeypatch.setattr(bridge, 'ros_ready', True)
        monkeypatch.setattr(bridge, 'publisher_node', SimpleNamespace(champ_commands=native))
        local = bridge.app.test_client()
        def original_transport(req, **kw):
            path = req.full_url.removeprefix(client.base_url)
            requests.append((req.method, path))
            response = local.open(path, method=req.method,
                json=json.loads(req.data) if req.data else None)
            if response.status_code != 200:
                raise urllib.error.HTTPError(req.full_url, response.status_code,
                    'in-process native Flask', {}, io.BytesIO(response.data))
            return Reply(response.data)
    def request(req, **kw):
        assert kw['timeout'] == 15.
        if req.full_url.endswith('/motion'):
            transport_times.append(dict(event='HTTP dispatch', local=clock(), native=g.clock()))
            advance(admission_delay)
            transport_times.append(dict(event='before Bridge execute', local=clock(), native=g.clock()))
            response = original_transport(req, **kw)
            advance(response_delay)
            transport_times.append(dict(event='HTTP reply', local=clock(), native=g.clock()))
            return response
        return original_transport(req, **kw)
    monkeypatch.setattr('urllib.request.urlopen', request)
    node.hook = lambda p: packets.append(dict(p, local_observed=clock()))
    def turn(direction='RIGHT', mode=ROTATIONAL_SWEPT_FOOTPRINT, duration=.5):
        return behavior.execute_guarded_turn(direction, .25, duration,
            expected_lidar_session='session-1', now=10., safety_mode=mode)
    return dict(g=g, node=node, native=native, client=client, interlock=interlock,
                behavior=behavior, world=world, monitors=monitors, packets=packets,
                requests=requests, times=transport_times, clock=clock,
                advance=advance, turn=turn)


@pytest.mark.parametrize('baseline', [True, False])
def test_retained_duration_admission_race_counterfactual(rig, monkeypatch, baseline, record_property):
    h = harness(rig, monkeypatch, admission_delay=.08, baseline=baseline)
    result = h['turn']()
    assert bool(result['ok']) is (not baseline)
    ack = result['transport_result']
    assert ack['completed_progress_credit'] is (not baseline)
    assert ack['state'] == ('interrupted' if baseline else 'completed')
    assert result['deadline_stop_attempted'] is baseline
    assert any(p == '/stop' for _, p in h['requests']) is baseline
    assert not h['g'].last['active'] and h['g'].last['gait_zero']
    assert sum(p == '/motion' for _, p in h['requests']) == 1
    record_property('deadline_counterfactual', json.dumps(dict(
        baseline=baseline, admission_delay_seconds=.08, result=result,
        packets=h['packets'], transport_events=h['times'])))


@pytest.mark.parametrize('direction', ['LEFT', 'RIGHT'])
@pytest.mark.parametrize('mode', [ROTATIONAL_SWEPT_FOOTPRINT, LEGACY_BROAD_SIDE])
@pytest.mark.parametrize('admission_delay', [0., .1])
@pytest.mark.parametrize('response_delay', [0., .4])
@pytest.mark.parametrize('duration', [.1, .5, 1.])
def test_native_window_and_delayed_reply_are_separate(rig, monkeypatch, direction, mode,
                                                     admission_delay, response_delay, duration):
    h = harness(rig, monkeypatch, admission_delay=admission_delay, response_delay=response_delay)
    result = h['turn'](direction, mode, duration)
    ack = result['transport_result']
    assert result['ok'] and ack['completed_progress_credit']
    assert result['normal_completion'] and result['window_complete']
    assert result['bounded_duration_owner'] == 'robot_bridge'
    assert result['active_lidar_safety_owner'] == ('robot_bridge' if mode == ROTATIONAL_SWEPT_FOOTPRINT else 'cognitive')
    assert not result['deadline_stop_attempted'] and not result['transport_completion_timed_out']
    assert h['monitors'][0]._deadline is None
    assert h['monitors'][0]._completion_deadline is None
    assert ack['stop_certificate']['last_request_issued_ns'] >= ack['deadline_ns']
    assert ack['stop_certificate']['gait_zero']
    assert ack['deadline_ns']-ack['start_ns'] == int(duration*1e9)
    active = [p for p in h['packets'] if p['operation'] in (START, HEARTBEAT)]
    assert active and len({p['deadline_ns'] for p in active}) == 1
    assert not any(p == '/stop' for _, p in h['requests'])
    assert not h['g'].last['active'] and h['g'].last['gait_zero']
    assert h['behavior']._guarded_turn_owner_generation is None


@pytest.mark.parametrize('axes', AXES)
def test_shared_native_path_retains_deadline_and_scoped_zero_for_all_axes(rig, monkeypatch, axes):
    h = harness(rig, monkeypatch, admission_delay=.1, response_delay=.4)
    ack = move(h['client'], axes, .5)
    assert ack['ok'] and ack['completed_progress_credit']
    assert ack['stop_certificate']['last_operation'] == STOP
    assert ack['stop_certificate']['last_request_issued_ns'] >= ack['deadline_ns']
    assert not any(p == '/stop' for _, p in h['requests'])
    assert not h['g'].last['active'] and h['g'].last['gait_zero']


@pytest.mark.parametrize('fault', ['intrusion', 'guard_missing', 'controller_expiry', 'stop_uncertain'])
@pytest.mark.parametrize('direction', ['LEFT', 'RIGHT'])
def test_native_faults_still_fail_closed(rig, monkeypatch, fault, direction):
    h = harness(rig, monkeypatch)
    g, node, native = h['g'], h['node'], h['native']
    start = g.clock()
    if fault == 'guard_missing':
        native.lidar_reader = None
    elif fault == 'intrusion':
        def scan():
            value = clear_lidar(g.clock)
            if g.clock()-start > 90_000_000:
                value['scan']['ranges'][20] = .4
            return value
        native.lidar_reader = scan
    elif fault == 'controller_expiry':
        def stall(packet):
            h['packets'].append(dict(packet))
            if packet['operation'] == HEARTBEAT:
                g.clock.sleep(.26)
                g.tick()
        node.hook = stall
    else:
        adapter = rig[2]
        original = adapter.snapshot
        def wrong_stop():
            sample, received = original()
            if sample.get('last_operation') == STOP:
                sample = dict(sample, last_request_issued_ns=sample['last_request_issued_ns']-1)
            return sample, received
        adapter.snapshot = wrong_stop
        node.fail = lambda p: p['operation'] == EMERGENCY
    result = h['turn'](direction)
    ack = result['transport_result']
    assert not result['ok'] and not ack['completed_progress_credit']
    assert not result['normal_completion']
    assert not result['deadline_stop_attempted']
    assert not g.last['active'] and g.last['gait_zero']
    if fault == 'guard_missing':
        assert not any(p['operation'] == START for p in node.messages)
    if fault == 'stop_uncertain':
        assert native.latched and h['interlock'].status()['last_stop_error']
        h['interlock'].refresh()
        assert not h['interlock'].status()['forward_permitted']


@pytest.mark.parametrize('error', ['timeout', 'invalid_reply', 'unscoped_reply'])
def test_uncertain_http_delivery_still_stops_without_completion(rig, monkeypatch, error):
    h = harness(rig, monkeypatch)
    original = urllib.request.urlopen
    def request(req, **kw):
        if req.full_url.endswith('/motion'):
            response = original(req, **kw)  # Native deadline/STOP complete first.
            if error == 'timeout':
                response.close()
                raise TimeoutError('lost bounded response')
            body = json.loads(response.read())
            if error == 'invalid_reply': body['controller_gait_zero'] = False
            else: body['stop_certificate']['generation'] += 1
            return Reply(json.dumps(body).encode())
        return original(req, **kw)
    monkeypatch.setattr('urllib.request.urlopen', request)
    result = h['turn']()
    assert not result['ok'] and result['delivery_uncertain']
    assert not result['transport_result']['completed_progress_credit']
    assert any(p == '/stop' for _, p in h['requests'])
    assert not h['g'].last['active']
    assert sum(p == '/motion' for _, p in h['requests']) == 1


@pytest.mark.parametrize('when', ['active', 'reply_pending'])
def test_operator_stop_remains_immediate_and_old_result_cannot_complete(rig, monkeypatch, when):
    h = harness(rig, monkeypatch)
    if when == 'active':
        def cancel(packet):
            h['packets'].append(dict(packet))
            if packet['operation'] == HEARTBEAT:
                h['behavior']._execute_stop()
        h['node'].hook = cancel
    else:
        original = urllib.request.urlopen
        def request(req, **kw):
            response = original(req, **kw)
            if req.full_url.endswith('/motion'):
                h['behavior']._execute_stop()
            return response
        monkeypatch.setattr('urllib.request.urlopen', request)
    result = h['turn']()
    assert not result['ok'] and result['generation_invalidated']
    assert not result['normal_completion']
    assert any(p['operation'] == EMERGENCY for p in h['node'].messages)
    assert not h['g'].last['active'] and h['g'].last['gait_zero']
    # Cancellation cannot promote the turn/mission. A command that had already
    # completed before cancellation retains that separate historical fact.
    assert result['transport_result']['completed_progress_credit'] is (when == 'reply_pending')


def test_retained_cancelled_mission_is_never_reclassified_as_completed(rig, monkeypatch):
    path = os.environ.get('MAYDAY_RETAINED_TURN_FILE')
    if not path:
        pytest.skip('retained mission artifact required')
    retained = json.loads(Path(path).read_text())['last_result']['history'][0]['result']['turn_result']['transport_result']
    h = harness(rig, monkeypatch)
    original = urllib.request.urlopen
    def request(req, **kw):
        if req.full_url.endswith('/motion'):
            raise urllib.error.HTTPError(req.full_url, 503, 'retained cancelled command', {},
                io.BytesIO(json.dumps(retained).encode()))
        return original(req, **kw)
    monkeypatch.setattr('urllib.request.urlopen', request)
    result = h['turn']()
    assert not result['ok'] and result['delivery_uncertain']
    assert result['transport_result']['reason'] == 'command_cancelled'
    assert not result['transport_result']['completed_progress_credit']
    assert not result['deadline_stop_attempted']


def test_native_pending_transport_cannot_admit_a_second_guarded_turn(rig, monkeypatch):
    h = harness(rig, monkeypatch)
    denied = []
    original = urllib.request.urlopen
    def request(req, **kw):
        if req.full_url.endswith('/motion'):
            denied.append(h['turn']('LEFT'))
        return original(req, **kw)
    monkeypatch.setattr('urllib.request.urlopen', request)
    result = h['turn']()
    assert result['ok'] and len(denied) == 1
    assert denied[0]['reason'] == 'turn_already_active'
    assert not denied[0]['transport_attempted']
    assert sum(p == '/motion' for _, p in h['requests']) == 1


def test_pre_admission_delivery_uncertainty_still_stops_without_start(rig, monkeypatch):
    h = harness(rig, monkeypatch)
    original = urllib.request.urlopen
    attempts = []
    def request(req, **kw):
        if req.full_url.endswith('/motion'):
            attempts.append(1)
            raise TimeoutError('no admission acknowledgement')
        return original(req, **kw)
    monkeypatch.setattr('urllib.request.urlopen', request)
    result = h['turn']()
    assert not result['ok'] and result['delivery_uncertain']
    assert not result['transport_result']['completed_progress_credit']
    assert len(attempts) == 1 and any(p == '/stop' for _, p in h['requests'])
    assert not any(p['operation'] == START for p in h['node'].messages)
    assert not h['g'].last['active']


def test_unverified_client_subclass_keeps_legacy_timeouts(monkeypatch):
    from robot_bridge.client import RobotBridgeClient
    from test_bridge_active_lidar_ownership import monitor, tick
    class CustomClient(RobotBridgeClient):
        pass
    client = CustomClient(base_url='http://offline.invalid')
    client.stop = lambda: {'ok': True}
    turn, _ = monitor(client)
    assert turn.begin_transport(behavior_module.time.monotonic()-1.)
    tick(turn)
    assert turn.status()['bounded_duration_owner'] == 'cognitive'
    assert turn.status()['deadline_stop_attempted']
    assert turn.status()['transport_completion_timed_out']
    assert turn.is_invalidated()


@pytest.mark.parametrize('direction', ['LEFT', 'RIGHT'])
@pytest.mark.parametrize('intrusion', [False, True])
def test_guarded_turn_through_exact_unchanged_flask_route(rig, monkeypatch, direction, intrusion):
    h = harness(rig, monkeypatch, admission_delay=.1, response_delay=.4, flask_route=True)
    start = h['g'].clock()
    if intrusion:
        def scan():
            value = clear_lidar(h['g'].clock)
            if h['g'].clock()-start > 200_000_000:
                value['scan']['ranges'][20] = .4
            return value
        h['native'].lidar_reader = scan
    result = h['turn'](direction)
    ack = result['transport_result']
    assert result['ok'] is (not intrusion)
    assert ack['completed_progress_credit'] is (not intrusion)
    assert not result['deadline_stop_attempted']
    if intrusion:
        assert ack['reason'] == 'active_lidar_intrusion'
    else:
        assert ack['stop_certificate']['last_request_issued_ns'] >= ack['deadline_ns']
        assert not any(p == '/stop' for _, p in h['requests'])
    assert not h['g'].last['active'] and h['g'].last['gait_zero']
    assert sum(p == '/motion' for _, p in h['requests']) == 1
