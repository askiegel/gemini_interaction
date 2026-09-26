"""Offline runtime plumbing contracts for the dry-run Find-Marvin adapter."""

from copy import deepcopy
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import behavior_manager as behavior_manager_module
from behavior_manager import BehaviorManager
from marvin_arrival_policy import evaluate_marvin_arrival
from marvin_pursuit_state import READY_TO_APPROACH, evaluate_marvin_pursuit_state
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


STAMP = "2026-09-26T16:00:00+00:00"
IDENTITY = "marvin-identity"


def preview():
    box = {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0}
    return {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "target_found": True,
        "identity_confirmed": True, "identity_id": IDENTITY,
        "source_timestamp": STAMP, "bbox": dict(box),
        "tracking": {"target_label": "marvin", "bbox": dict(box),
                     "identity_id": IDENTITY},
    }


def locked(**updates):
    value = {
        "found": True, "stale": False, "tracking_mode": "LOCKED",
        "identity_id": IDENTITY, "locked_identity_id": IDENTITY,
        "entity_id": "entity-1", "last_seen": STAMP,
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0},
        "image_width": 640.0, "image_height": 480.0,
        "cx": 194.0,
    }
    value.update(updates)
    return value


def snapshot(mode="LOCKED", identity=IDENTITY):
    return {"tracking_mode": mode, "locked_identity_id": identity}


class FakeTargetLock:
    def __init__(self, result=None, state=None):
        self.result = result or locked()
        self.state = state or snapshot()
        self.mission_id = None
        self.target_label = "marvin"
        self.calls = []

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        return deepcopy(self.result)

    def snapshot(self):
        return deepcopy(self.state)


def manager(monkeypatch, target_lock=None, preview_value=None):
    value = preview() if preview_value is None else preview_value
    instance = BehaviorManager(robot_client=object())
    instance.target_lock = target_lock or FakeTargetLock()
    monkeypatch.setattr(instance, "preview_find_object", lambda _target: deepcopy(value))
    return instance


def test_state_provider_is_fresh_and_preserves_authoritative_bundle(monkeypatch):
    lock = FakeTargetLock()
    instance = manager(monkeypatch, lock)
    first = instance.build_find_marvin_controller_state()
    second = instance.build_find_marvin_controller_state()
    assert len(lock.calls) == 2 and first == second
    assert first["selected_identity_id"] == IDENTITY
    assert first["target_lock_snapshot"]["tracking_mode"] == "LOCKED"
    assert first["preview_result"]["authoritative"] is False
    assert first["target_lock_result"]["bbox"]["y2"] == 364.0
    assert first["identity_evidence"] == first["target_lock_result"]


def test_provider_bundle_feeds_pursuit_and_arrival_policies(monkeypatch):
    bundle = manager(monkeypatch).build_find_marvin_controller_state(now=STAMP)
    pursuit = evaluate_marvin_pursuit_state(
        bundle["preview_result"], bundle["target_lock_result"],
        bundle["target_lock_snapshot"], selected_identity_id=IDENTITY,
        identity_evidence=bundle["identity_evidence"],
        bridge_result=bundle["bridge_result"], now=STAMP,
    )
    arrival = evaluate_marvin_arrival(
        bundle["target_lock_result"], bundle["target_lock_snapshot"],
        selected_identity_id=IDENTITY, now=STAMP,
    )
    assert pursuit["state"] == READY_TO_APPROACH
    assert arrival["arrived_at_marvin"] is True


def test_waiting_identity_and_preview_failure_remain_fail_closed(monkeypatch):
    waiting = FakeTargetLock(
        locked(found=False, tracking_mode="WAITING_FOR_IDENTITY"),
        snapshot("WAITING_FOR_IDENTITY"),
    )
    bundle = manager(monkeypatch, waiting).build_find_marvin_controller_state()
    assert bundle["selected_identity_id"] == IDENTITY
    assert bundle["target_lock_snapshot"]["tracking_mode"] == "WAITING_FOR_IDENTITY"
    broken = manager(monkeypatch)
    monkeypatch.setattr(broken, "preview_find_object", lambda _target: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        broken.build_find_marvin_controller_state()


def test_malformed_target_lock_result_fails_closed(monkeypatch):
    instance = manager(monkeypatch, FakeTargetLock(result="invalid"))
    with pytest.raises(RuntimeError, match="target_lock_result_malformed"):
        instance.build_find_marvin_controller_state()


def test_runtime_dry_run_never_authorizes_execution_or_motion():
    runtime = object.__new__(CognitiveRuntime)
    calls = []

    class Behavior:
        def execute_find_marvin_controller(self, provider, **kwargs):
            calls.append((provider(), kwargs))
            return {"ok": True, "completed": False, "actions_executed": 0,
                    "next_route": "search", "history": [{"pursuit_state": "SEARCHING"}]}

    runtime.behavior_manager = Behavior()
    runtime.build_find_marvin_controller_state = lambda: {"fresh": True}
    value = runtime.dry_run_find_marvin_controller(execute=False)
    assert value["controller_ready"] is True and value["execution_authorized"] is False
    assert value["motion_executed"] is False and value["next_route"] == "search"
    assert calls[0][1] == {"max_actions": 1, "dry_run": True}
    rejected = runtime.dry_run_find_marvin_controller(execute=True)
    assert rejected["ok"] is False and rejected["motion_executed"] is False
    assert len(calls) == 1


def test_runtime_endpoint_accepts_only_explicit_false_execute():
    runtime = SimpleNamespace(
        dry_run_find_marvin_controller=Mock(return_value={"ok": True, "motion_executed": False})
    )
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/controller"
    handler.server = SimpleNamespace(runtime=runtime)
    handler.require_json_request = Mock(return_value={"execute": False})
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_POST()
    assert responses == [(200, {"ok": True, "motion_executed": False})]
    runtime.dry_run_find_marvin_controller.assert_called_once_with(execute=False)
    handler.require_json_request = Mock(return_value={"execute": True})
    handler.do_POST()
    assert runtime.dry_run_find_marvin_controller.call_count == 1
    assert responses[-1][1]["ok"] is False


def test_runtime_path_has_no_direct_primitive_or_navigation_calls():
    runtime_source = inspect.getsource(CognitiveRuntime.dry_run_find_marvin_controller).lower()
    builder_source = inspect.getsource(BehaviorManager.build_find_marvin_controller_state).lower()
    for source in (runtime_source, builder_source):
        for forbidden in ("robot.local_forward", "execute_guarded_turn", "execute_marvin_search_step", "execute_marvin_pursuit_step", "nav2", "robot_bridge"):
            assert forbidden not in source
