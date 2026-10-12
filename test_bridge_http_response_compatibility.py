"""Actual loopback HTTP, Flask and Cognitive client; no robot/ROS/actuator I/O."""
import json
import socket
import threading
import time

import pytest
flask = pytest.importorskip("flask")
from werkzeug.serving import make_server
from robot_bridge.client import RobotBridgeClient
from robot_bridge.http_response import MAX_ERROR_BODY_BYTES


def certificate(state="interrupted"):
    return dict(ok=False, state=state, reason="command_receipt_expired",
                transport_attempted=True, completed_progress_credit=False,
                controller_gait_zero=True, physical_stop_verified=False,
                automatic_stop=True, stop_reason="scoped_gait_zero",
                generation=7, start_ns=1_000_000_000, deadline_ns=1_200_000_000,
                linear_x=0.,linear_y=0.,angular_z=.2,duration=.2,
                action="motion",mode="bounded",returned_immediately=False,
                controller_session="controller-session", bridge_epoch="bridge-epoch")


@pytest.fixture
def http_server():
    app = flask.Flask(__name__)
    response = {"body": certificate(), "status": 503, "delay": 0, "calls": 0}
    @app.post("/motion")
    @app.post("/stop")
    def motion():
        response["calls"] += 1
        time.sleep(response["delay"])
        body = response["body"]
        if isinstance(body, (str, bytes)):
            return flask.Response(body, status=response["status"])
        return flask.jsonify(body), response["status"]
    server = make_server("127.0.0.1", 0, app, threaded=True)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    client = RobotBridgeClient(base_url=f"http://127.0.0.1:{server.server_port}", timeout=.15)
    yield client, response
    server.shutdown(); t.join(2); server.server_close()


def perform(http_server, body, status=503):
    client, response = http_server
    response.update(body=body, status=status)
    # Pure yaw avoids needing an unrelated forward interlock in this HTTP test.
    result = client.motion(angular_z=.2, duration=.2)
    assert response["calls"] == 1  # No automatic retransmission.
    assert result.get("retry_permitted") is False
    return result


def test_actual_flask_transported_interruption_keeps_identity_and_stop(http_server):
    body = certificate(); body.update(secret="do-not-retain", scan=[1]*100, image="bytes", url="credential")
    body["stop_result"] = dict(ok=True, reason="scoped_gait_zero", controller_gait_zero=True, token="secret")
    r = perform(http_server, body)
    assert r["http_status"] == 503 and r["response_origin"] == "server_error"
    assert r["state"] == "interrupted" and r["reason"] == "command_receipt_expired"
    assert r["transport_attempted"] is True and not r["delivery_uncertain"]
    assert r["generation"] == 7 and r["deadline_ns"] == 1_200_000_000
    assert r["controller_session"] == "controller-session" and r["bridge_epoch"] == "bridge-epoch"
    assert r["stop_result"]["ok"] and r["controller_gait_zero"]
    assert not r["ok"] and not r["completed_progress_credit"] and r["evidence_valid"]
    assert not any(k in r for k in ("secret", "scan", "image", "url"))
    assert "token" not in r["stop_result"]
    assert len(json.dumps(r)) < 2048


def test_actual_flask_zero_transport_refusal(http_server):
    r = perform(http_server, dict(ok=False,state="refused", reason="owner_busy_or_unresolved",
                                transport_attempted=False, completed_progress_credit=False,
                                controller_gait_zero=False, physical_stop_verified=False), 409)
    assert r["evidence_valid"] and r["state"] == "refused"
    assert r["transport_attempted"] is False and not r["delivery_uncertain"]
    assert not r["ok"] and r["http_status"] == 409


def test_actual_flask_uncertain_publication_keeps_uncertainty(http_server):
    r = perform(http_server, dict(certificate("uncertain"), reason="start_publication_uncertain"))
    assert r["state"] == "uncertain" and r["transport_attempted"] is True
    assert r["delivery_uncertain"] and not r["completed_progress_credit"]


@pytest.mark.parametrize("body",["not-json", "<html>failure</html>", "[]", "null", '{"ok":false,"ok":true}',
                                '{"ok":false,"age":NaN}', b'\xff', "x"*(MAX_ERROR_BODY_BYTES+1)])
def test_invalid_error_body_is_uncertain_bounded_not_retried(http_server,body):
    r=perform(http_server,body)
    assert r["state"] == "uncertain" and r["delivery_uncertain"]
    assert r["transport_attempted"] is None and not r["evidence_valid"]
    assert r["http_status"] == 503 and not r["completed_progress_credit"]


@pytest.mark.parametrize("change",[
    {"ok":True},{"completed_progress_credit":True},{"transport_attempted":False},
    {"state":"completed"},{"state":"nonsense"},{"transport_attempted":"false"},
    {"controller_gait_zero":"true"},{"controller_gait_zero":False},{"generation":True},{"generation":2**64},
    {"generation":0},{"deadline_ns":1},{"deadline_ns":4_000_000_001},
    {"bridge_epoch":"not an identity"},{"reason":"secret https://credentials"},
    {"reason":"x"*161},{"physical_stop_verified":True},{"confirmed_forwarded":True},
    {"automatic_stop":False},{"stop_result":{"ok":False,"controller_gait_zero":True}},
])
def test_contradictory_or_malformed_error_evidence_fails_closed(http_server,change):
    r=perform(http_server,dict(certificate(),**change))
    assert not r["ok"] and not r["evidence_valid"] and r["delivery_uncertain"]
    assert r["state"] == "uncertain" and not r["completed_progress_credit"]
    assert r["transport_attempted"] is None


@pytest.mark.parametrize("field",["reason","controller_session","bridge_epoch","generation","start_ns","deadline_ns","completed_progress_credit","transport_attempted","controller_gait_zero","automatic_stop","stop_reason"])
def test_missing_certificate_field_cannot_authorize_or_certify_no_transport(http_server,field):
    body=certificate();del body[field];r=perform(http_server,body)
    assert not r["evidence_valid"] and r["delivery_uncertain"] and not r["ok"]


def test_timeout_after_request_is_uncertain_without_retry(http_server):
    client,response=http_server;response["delay"]=.3;client.timeout=.03
    r=client.motion(angular_z=.2,duration=.2)
    assert r["error"] == "bridge_response_timeout" and r["delivery_uncertain"]
    assert r["transport_attempted"] is None and not r["http_response_received"]
    time.sleep(.32);assert response["calls"] == 1


def test_actual_connection_refusal_distinguished_from_server_refusal():
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    r=RobotBridgeClient(base_url=f'http://127.0.0.1:{port}').motion(angular_z=.2,duration=.2)
    assert r["response_origin"] == "connection_failure" and r["http_status"] is None
    assert r["transport_attempted"] is False and not r["delivery_uncertain"]
    assert not r["ok"] and not r["retry_permitted"]


def test_lost_ack_after_receiving_request_is_uncertain():
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(1);received=[]
    def discard():
        connection,_=sock.accept()
        with connection:
            received.append(connection.recv(4096))
        sock.close()  # Real HTTP peer drops acknowledgement after receiving POST.
    t=threading.Thread(target=discard);t.start()
    r=RobotBridgeClient(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}').motion(angular_z=.2,duration=.2)
    t.join(2)
    assert len(received)==1 and received[0].startswith(b'POST /motion')
    assert r["delivery_uncertain"] and r["transport_attempted"] is None
    assert not r["completed_progress_credit"] and not r["retry_permitted"]


def test_successful_guarded_completion_compatible_and_bounded(http_server):
    body=dict(certificate("completed"),ok=True,completed_progress_credit=True,
              reason="bounded_deadline_completed", secret="discard",scan=[1]*100)
    r=perform(http_server,body,200)
    assert r["ok"] and r["completed_progress_credit"] and r["state"] == "completed"
    assert r["evidence_valid"] and r["http_status"] == 200
    assert r["generation"] == 7 and "secret" not in r and "scan" not in r


@pytest.mark.parametrize("change",[{"state":"interrupted"},{"completed_progress_credit":False},
                                     {"controller_gait_zero":False},{"delivery_uncertain":True},
                                     {"deadline_ns":0},{"returned_immediately":True},
                                     {"forwarded":False},{"confirmed_forwarded":False},
                                     {"stop_result":{"ok":False,"controller_gait_zero":False}},
                                     {"linear_x":.1},{"linear_y":.05},{"angular_z":.1},
                                     {"duration":.3},{"mode":"streaming"},{"action":"other"}])
def test_success_status_cannot_hide_interruption_or_bad_completion(http_server,change):
    body=dict(certificate("completed"),ok=True,completed_progress_credit=True);body.update(change)
    r=perform(http_server,body,200)
    assert not r["ok"] and not r["completed_progress_credit"] and r["delivery_uncertain"]
    assert r["http_status"] == 200 and not r["evidence_valid"]


def test_invalid_success_json_is_uncertain(http_server):
    r=perform(http_server,'not-json',200)
    assert not r["ok"] and not r["completed_progress_credit"] and r["delivery_uncertain"]
    assert r["http_status"] == 200 and r["http_response_received"]


def test_failed_stop_certificate_is_preserved_without_claiming_zero(http_server):
    client, reply=http_server
    reply['body']=dict(ok=False,action='stop',stop_result=dict(ok=False,reason='unverified_emergency_zero',controller_gait_zero=False),cancel_result={'raw':'discard'})
    r=client.stop()
    assert reply['calls']==1 and not r['ok'] and r['evidence_valid']
    assert r['stop_result']['reason']=='unverified_emergency_zero'
    assert not r['stop_result']['controller_gait_zero'] and r['delivery_uncertain']
    assert 'cancel_result' not in r and not r['completed_progress_credit']


def test_missing_guarded_success_state_does_not_masquerade_as_legacy(http_server):
    body=dict(certificate('completed'),ok=True,completed_progress_credit=True);del body['state']
    r=perform(http_server,body,200)
    assert not r['ok'] and not r['completed_progress_credit'] and r['delivery_uncertain']
