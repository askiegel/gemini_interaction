"""Offline contract and fail-closed transport admission for lateral commands."""
import pytest
from behavior_manager import _bridge_status_zero
from robot_bridge.client import RobotBridgeClient


class Interlock:
    def __init__(self): self.pending = False
    def begin_positive_dispatch(self, *, streaming):
        self.pending = True
        return 1
    def finalize_positive_dispatch(self, generation, result): return True
    def stop_active(self): self.pending = False


def client_with_transport(monkeypatch, status=None):
    interlock = Interlock()
    client = RobotBridgeClient(base_url='http://offline.invalid', forward_interlock=interlock)
    requests = []
    if status is None:
        status = {'ok':True,'ros_ready':True,'status':'READY',
            'motion_capabilities':{'linear_y':True,'max_linear_y':.1}}
    def request(method, path, payload=None):
        requests.append((method,path,payload))
        if method == 'GET': return status
        return {'ok':True, 'action':'motion','mode':'bounded','automatic_stop':True,
            'returned_immediately':False, **(payload or {})}
    monkeypatch.setattr(client, '_request', request)
    return client, requests, interlock


@pytest.mark.parametrize('y', [.08,-.08])
def test_lateral_uses_same_bounded_endpoint_and_interlock(monkeypatch,y):
    client,requests,interlock=client_with_transport(monkeypatch)
    result=client.move_lateral(speed=y,seconds=.5)
    assert result['ok'] and not interlock.pending
    assert requests == [('GET','/status',None),('POST','/motion',
        {'linear_x':0.,'linear_y':y,'angular_z':0.,'duration':.5})]


@pytest.mark.parametrize('parameters',[
    {'linear_y':True},{'linear_y':None},{'linear_y':'bad'},
    {'linear_y':float('nan')},{'linear_y':float('inf')},
    {'linear_y':.080001},{'linear_y':-.080001},
    {'linear_y':.08,'duration':1.00001},{'linear_y':.08,'duration':0},
    {'linear_y':.08,'duration':float('nan')},{'linear_y':.08,'duration':'bad'},
    {'linear_y':.08,'linear_x':.1},{'linear_y':.08,'angular_z':.25},
])
def test_invalid_lateral_fails_before_any_transport(monkeypatch,parameters):
    client,requests,_=client_with_transport(monkeypatch)
    result=client.motion(**parameters)
    assert not result['ok'] and result['forwarded'] is False and requests==[]


@pytest.mark.parametrize('status',[
    {},{'ok':True,'ros_ready':True,'status':'READY'},
    {'ok':True,'ros_ready':True,'status':'READY','motion_capabilities':'bad'},
    {'ok':True,'ros_ready':True,'status':'READY','motion_capabilities':{'linear_y':True,'max_linear_y':float('nan')}},
    {'ok':True,'ros_ready':True,'status':'READY','motion_capabilities':{'linear_y':True,'max_linear_y':.07}},
])
def test_old_or_malformed_bridge_cannot_receive_strafe(monkeypatch,status):
    client,requests,_=client_with_transport(monkeypatch,status)
    result=client.move_lateral(speed=.08,seconds=.5)
    assert not result['ok'] and result['error']=='bridge_lateral_support_unavailable'
    assert all(method=='GET' for method,_,_ in requests)


def test_lateral_requires_configured_interlock(monkeypatch):
    client,requests,_=client_with_transport(monkeypatch)
    client.forward_interlock=None
    result=client.move_lateral(speed=.08,seconds=.5)
    assert not result['ok'] and result['error']=='forward_interlock_not_configured'
    assert all(method=='GET' for method,_,_ in requests)


def test_final_dispatch_guard_runs_after_capability_request(monkeypatch):
    client,requests,interlock=client_with_transport(monkeypatch)
    def preempted():
        assert requests==[('GET','/status',None)]
        return False
    result=client.move_lateral(speed=.08,seconds=.5,dispatch_guard=preempted)
    assert result['error']=='motion_dispatch_preempted' and not result['forwarded']
    assert not interlock.pending and len(requests)==1


def test_old_forward_and_turn_payloads_omit_lateral(monkeypatch):
    client,requests,_=client_with_transport(monkeypatch)
    client.motion(.10,0,.50)
    client.motion(0,-.25,.50)
    assert requests==[
        ('POST','/motion',{'linear_x':.10,'angular_z':0.,'duration':.50}),
        ('POST','/motion',{'linear_x':0.,'angular_z':-.25,'duration':.50})]


def test_stopped_checks_reject_lateral_even_with_zero_x_yaw():
    status={'ok':True,'ros_ready':True,'motion':{'linear_x':0.,'angular_z':0.,'streaming':False}}
    assert _bridge_status_zero(status)  # Legacy omission remains compatible.
    status['motion']['linear_y']=.08
    assert not _bridge_status_zero(status)
    status['motion']['linear_y']=0.
    assert _bridge_status_zero(status)


@pytest.mark.parametrize('method',['submit_marvin_one_step_test','submit_marvin_centering_step','submit_marvin_guarded_approach'])
def test_dashboard_preflight_rejects_active_lateral_motion(monkeypatch,method):
    from test_marvin_one_step import safe_status
    from voice_relay.server import VoiceRelayHandler
    handler=object.__new__(VoiceRelayHandler)
    status=safe_status();status['robot']['motion']['linear_y']=.08
    handler.dashboard_status=lambda:status
    def forbidden(*args,**kwargs): pytest.fail('Strafing robot must not receive a new mission')
    monkeypatch.setattr('voice_relay.server.request_json',forbidden)
    code,payload=getattr(handler,method)(execute=True)
    assert code==409 and 'Robot Bridge lateral motion is not zero' in payload['reasons']
