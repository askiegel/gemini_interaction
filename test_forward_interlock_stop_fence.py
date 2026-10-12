"""Deterministic offline STOP/completion/next-owner races; no HTTP/ROS."""
import threading

import pytest
from robot_bridge.forward_interlock import ForwardMotionInterlock
from robot_bridge.client import RobotBridgeClient


def fresh():
    return dict(available=True, valid=True, reason='fresh', producer_session='scan-1',
                effective_age_seconds=.01, sectors={'front': dict(available=True, state='CLEAR')})


@pytest.mark.parametrize('fails', [False, True])
def test_successor_refused_until_old_stop_callback_resolves(fails):
    state = fresh()
    entered, release = threading.Event(), threading.Event()
    calls, errors = [], []
    def stop():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2.)
            if fails:
                raise RuntimeError('mock STOP failure')
        return {'ok': True}
    i = ForwardMotionInterlock(lambda **kw: dict(state), expected_session='scan-1', stop_callback=stop)
    i.refresh()
    old = i.begin_positive_dispatch(streaming=False)
    state.update(available=False, valid=False, reason='stale')
    def monitor():
        try:
            i.refresh()
        except Exception as exc:
            errors.append(exc)
    t = threading.Thread(target=monitor)
    t.start()
    try:
        assert entered.wait(1.)
        state.update(available=True, valid=True, reason='fresh')
        assert i.finalize_positive_dispatch(old, {'ok': True}) is False
        i.refresh()
        assert i.status()['stop_pending']
        assert not i.status()['forward_permitted']
        with pytest.raises(PermissionError, match='stop_pending'):
            i.begin_positive_dispatch(streaming=False)
    finally:
        release.set()
        t.join(2.)
    assert not t.is_alive() and not errors
    assert not i.status()['stop_pending']
    i.refresh()
    # A resolved failed STOP is reported, not represented as physical zero.
    assert (i.status()['last_stop_error'] is not None) is fails
    if fails:
        with pytest.raises(PermissionError, match='stop_unverified'):
            i.begin_positive_dispatch(streaming=False)
    else:
        new = i.begin_positive_dispatch(streaming=False)
        assert new > old
        i.finalize_positive_dispatch(new, {'ok': True})
    assert len(calls) == 2


def test_explicit_stop_reservation_blocks_reentrant_dispatch():
    attempts = []
    i = None
    def stop():
        with pytest.raises(PermissionError, match='stop_pending'):
            i.begin_positive_dispatch(streaming=False)
        attempts.append(True)
        return {'ok': True}
    i = ForwardMotionInterlock(lambda **kw: fresh(), expected_session='scan-1', stop_callback=stop)
    i.refresh()
    i._dispatch_stop()
    assert attempts == [True] and not i.status()['stop_pending']


def test_failed_callback_releases_reservation_but_quarantines_new_commands():
    i = ForwardMotionInterlock(lambda **kw: fresh(), expected_session='scan-1',
        stop_callback=lambda: {'ok': False})
    i.refresh()
    assert i._dispatch_stop()['error']
    assert i._stops_inflight == 0 and i.status()['last_stop_error']
    assert not i.status()['forward_permitted']
    i.refresh()
    assert i.status()['reason'] == 'forward_interlock_stop_unverified'
    with pytest.raises(PermissionError, match='stop_unverified'):
        i.begin_positive_dispatch(streaming=False)


@pytest.mark.parametrize('error', [TimeoutError('STOP acknowledgement lost'), OSError('STOP connection failed')])
def test_uncertain_stop_cannot_be_cleared_by_healthy_scans(error):
    def stop(): raise error
    i = ForwardMotionInterlock(lambda **kw: fresh(), expected_session='scan-1', stop_callback=stop)
    i.refresh()
    event = i._dispatch_stop()
    assert event['error'] and event['result'] is None
    i.refresh()
    assert not i.status()['forward_permitted']
    with pytest.raises(PermissionError, match='stop_unverified'):
        i.begin_positive_dispatch(streaming=False)


@pytest.mark.parametrize('axes', [(.1,0,0),(-.1,0,0),(0,.08,0),(0,-.08,0),(0,0,.25),(0,0,-.25)])
def test_failed_stop_refuses_every_bounded_client_direction(axes):
    i = ForwardMotionInterlock(lambda **kw: fresh(), expected_session='scan-1',
        stop_callback=lambda: {'ok': False})
    i.refresh()
    i._dispatch_stop()
    i.refresh()
    client = RobotBridgeClient(base_url='http://offline.invalid', forward_interlock=i)
    calls = []
    def request(method, path, payload=None):
        calls.append(path)
        if path == '/status':
            return dict(ok=True, status='READY', ros_ready=True,
                        motion_capabilities={'linear_y': True, 'max_linear_y': .1})
        if path == '/stop': return {'ok': True}
        raise AssertionError('Quarantined command attempted transport')
    client._request = request
    result = client.motion(linear_x=axes[0], linear_y=axes[1], angular_z=axes[2], duration=.2)
    assert result['error'] == 'forward_interlock_stop_unverified'
    assert '/motion' not in calls
    assert client.stop()['ok']  # Unconditional emergency STOP remains available.


@pytest.mark.parametrize('axes', [(-.1,0,0),(0,0,.25),(0,0,-.25)])
def test_healthy_reverse_and_rotation_use_existing_fence_without_changing_command(axes):
    i = ForwardMotionInterlock(lambda **kw: fresh(), expected_session='scan-1', stop_callback=lambda: {'ok': True})
    i.refresh()
    client = RobotBridgeClient(base_url='http://offline.invalid', forward_interlock=i)
    commands = []
    def request(method, path, payload=None):
        assert path == '/motion' and i.status()['pending_forward']
        commands.append(payload)
        return {'ok': True}
    client._request = request
    assert client.motion(linear_x=axes[0], angular_z=axes[2], duration=.2)['ok']
    assert commands == [dict(linear_x=axes[0], angular_z=axes[2], duration=.2)]
    assert not i.status()['pending_forward'] and not i.status()['active_forward']
