from datetime import datetime, timedelta, timezone
from camera_motion_gate import evaluate_camera_gate
from behavior_manager import BehaviorManager
from robot_bridge.client import RobotBridgeClient

NOW=datetime(2026,1,1,tzinfo=timezone.utc)
def payload(detections=[]): return {"camera_running":True,"timestamp":NOW.isoformat(),"detections":detections,"image_width":100}
def test_camera_gate_health_and_person_vetoes():
    assert not evaluate_camera_gate({},NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({**payload(),"timestamp":"bad"},NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({**payload(),"timestamp":(NOW-timedelta(seconds=2)).isoformat()},NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate(payload(),NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(payload([{"label":"person","confidence":.25,"center_x":25}]),NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate(payload([{"label":"person","confidence":.24,"center_x":50}]),NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate(payload([{"label":"human","confidence":.9,"center_x":10}]),NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(payload([{"label":"person","confidence":.9}]),NOW)["camera_semantic_clear"]
class Vision:
    def __init__(self,p): self.p=p
    def fetch_vision_payload(self): return self.p
class Robot:
    def __init__(self): self.calls=[]
    def stop(self): self.calls.append("stop"); return {"ok":True}
    def local_forward(self): self.calls.append("local"); return {"ok":True,"executed":True,"stop_reason":"duration_complete","ownership":"x"}
    def motion(self,*a,**k): raise AssertionError("no fallback")
def test_behavior_uses_stop_or_single_local_forward(monkeypatch):
    import behavior_manager
    monkeypatch.setattr(behavior_manager,"evaluate_camera_gate",lambda p:{"camera_semantic_clear":False,"reason":"camera_stale"})
    robot=Robot(); result=BehaviorManager(robot_client=robot,vision_adapter=Vision({}))._execute_move_forward(None)
    assert not result["executed"] and robot.calls==["stop"]
    monkeypatch.setattr(behavior_manager,"evaluate_camera_gate",lambda p:{"camera_semantic_clear":True,"reason":"camera_semantic_clear"})
    robot=Robot(); result=BehaviorManager(robot_client=robot,vision_adapter=Vision({}))._execute_move_forward(None)
    assert result["executed"] and robot.calls==["local"] and result["robot_result"]["ownership"]=="x"
def test_client_posts_empty_object(monkeypatch):
    client=RobotBridgeClient(base_url="http://x"); calls=[]
    monkeypatch.setattr(client,"_request",lambda *args: calls.append(args) or {"ok":True})
    client.local_forward(); assert calls==[("POST","/local-motion/forward",{})]


def test_camera_gate_timestamp_and_detection_contract():
    assert not evaluate_camera_gate({**payload(), "camera_running": False}, NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({**payload(), "camera_running": None}, NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({"camera_running": True, "detections": []}, NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate({**payload(), "timestamp": "2026-01-01T00:00:00Z"}, NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate({**payload(), "timestamp": (NOW-timedelta(seconds=1)).isoformat()}, NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({**payload(), "detections": None}, NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({**payload(), "detections": {}}, NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate(payload([{"label":"cat", "confidence":1}]), NOW)["camera_semantic_clear"]

def test_camera_gate_geometry_widths_and_lane_boundaries():
    def person(data): return payload([{"label":"person", "confidence":.9, **data}])
    assert evaluate_camera_gate(person({"center_x":24}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"center_x":25}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"center_x":50}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"center_x":75}), NOW)["camera_semantic_clear"]
    assert evaluate_camera_gate(person({"center_x":76}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"x1":40,"x2":60}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"bbox":{"x1":40,"x2":60}}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate(person({"bbox":{"left":40,"right":60}}), NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({"camera_running":True,"timestamp":NOW.isoformat(),"frame_width":100,"detections":[{"label":"person","confidence":.9,"center_x":50}]}, NOW)["camera_semantic_clear"]
    assert not evaluate_camera_gate({"camera_running":True,"timestamp":NOW.isoformat(),"width":100,"detections":[{"label":"person","confidence":.9,"center_x":50}]}, NOW)["camera_semantic_clear"]

def test_bridge_result_and_exception_are_not_retried(monkeypatch):
    import behavior_manager
    monkeypatch.setattr(behavior_manager,"evaluate_camera_gate",lambda p:{"camera_semantic_clear":True,"reason":"camera_semantic_clear"})
    class Failing(Robot):
        def local_forward(self): self.calls.append("local"); raise RuntimeError("offline")
    robot=Failing(); result=BehaviorManager(robot_client=robot,vision_adapter=Vision({}))._execute_move_forward(None)
    assert not result["ok"] and not result["executed"] and robot.calls==["local"]
    class Rejected(Robot):
        def local_forward(self): self.calls.append("local"); return {"ok":False,"executed":False,"stop_reason":"forward_obstacle","clearance":.3}
    robot=Rejected(); result=BehaviorManager(robot_client=robot,vision_adapter=Vision({}))._execute_move_forward(None)
    assert not result["executed"] and result["reason"]=="forward_obstacle" and result["robot_result"]["clearance"]==.3 and robot.calls==["local"]

def test_stop_regression_uses_stop_not_local_forward():
    robot=Robot(); result=BehaviorManager(robot_client=robot,vision_adapter=Vision({}))._execute_stop()
    assert result["behavior"]=="STOP" and robot.calls==["stop"]
