"""Offline runtime plumbing contracts for the dry-run Find-Marvin adapter."""

from copy import deepcopy
from datetime import datetime, timezone
import inspect
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import behavior_manager as behavior_manager_module
import runtime as runtime_module
from behavior_manager import BehaviorManager
from marvin_arrival_policy import evaluate_marvin_arrival
from marvin_pursuit_state import READY_TO_APPROACH, evaluate_marvin_pursuit_state
from runtime import CognitiveRuntime
from runtime_api import RuntimeAPIHandler


STAMP = datetime.now(timezone.utc).isoformat()
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


def continuity_preview():
    value = preview()
    value.update({
        "source": "marvin_local_tracker",
        "entity_id": "entity-1",
    })
    value["target_observation"] = {
        "found": True,
        "stale": False,
        "source_timestamp": STAMP,
        "entity_id": "entity-1",
        "identity_id": IDENTITY,
        "bbox": dict(value["bbox"]),
        "image_width": 640.0,
        "image_height": 480.0,
        "identity_ambiguous": False,
    }
    return value


def confirmed_world_entity(identity=IDENTITY):
    return {
        "entity_id": "entity-1",
        "label": "marvin",
        "entity_type": "person",
        "attributes": {
            "identity_id": identity,
            "operator_confirmed": True,
            "identity_confirmation_source": "marvin_local_tracker_preview",
            "identity_confirmation_timestamp": STAMP,
        },
    }


class FakeWorldModel:
    def __init__(self, entities):
        self.entities = deepcopy(entities)
        self.update_calls = []

    def get_entities(self):
        return deepcopy(self.entities)

    def get_entity(self, entity_id):
        return next(
            (item for item in self.entities if item["entity_id"] == entity_id),
            None,
        )

    def update_entity(self, **kwargs):
        self.update_calls.append(deepcopy(kwargs))
        entity = next(
            (item for item in self.entities if item["entity_id"] == kwargs["entity_id"]),
            None,
        )
        if entity is not None:
            entity["label"] = kwargs["label"]
            entity["entity_type"] = kwargs["entity_type"]
            entity["confidence"] = kwargs["confidence"]
            entity["last_seen"] = STAMP
            entity.setdefault("attributes", {}).update(deepcopy(kwargs["attributes"]))
            entity.setdefault("history", []).append({
                "timestamp": STAMP,
                "source": kwargs["source"],
                "confidence": kwargs["confidence"],
                "location": deepcopy(kwargs["location"]),
                "attributes": deepcopy(kwargs["attributes"]),
            })


class FakeTargetLock:
    def __init__(self, result=None, state=None, on_resolve=None):
        self.result = result or locked()
        self.state = state or snapshot()
        self.mission_id = None
        self.target_label = "marvin"
        self.calls = []
        self.on_resolve = on_resolve

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_resolve is not None:
            self.on_resolve()
        return deepcopy(self.result)

    def snapshot(self):
        return deepcopy(self.state)


def manager(monkeypatch, target_lock=None, preview_value=None, world_model=None):
    value = preview() if preview_value is None else preview_value
    instance = BehaviorManager(robot_client=object())
    instance.target_lock = target_lock or FakeTargetLock()
    instance.world_model = world_model
    monkeypatch.setattr(instance, "preview_find_object", lambda _target: deepcopy(value))
    return instance


def test_state_provider_is_fresh_and_preserves_authoritative_bundle(monkeypatch):
    lock = FakeTargetLock()
    instance = manager(monkeypatch, lock)
    first = instance.build_find_marvin_controller_state(now=STAMP)
    second = instance.build_find_marvin_controller_state(now=STAMP)
    assert len(lock.calls) == 2 and first == second
    assert first["selected_identity_id"] == IDENTITY
    assert first["target_lock_snapshot"]["tracking_mode"] == "LOCKED"
    assert first["preview_result"]["authoritative"] is False
    assert first["target_lock_result"]["bbox"]["y2"] == 364.0
    assert first["identity_evidence"] == first["target_lock_result"]


def test_valid_continuity_refreshes_existing_entity_before_targetlock_resolve(monkeypatch):
    entity = confirmed_world_entity()
    world = FakeWorldModel([entity])
    lock = FakeTargetLock(on_resolve=lambda: world.update_calls or pytest.fail(
        "TargetLock.resolve must run after the authorized World Model refresh"
    ))
    instance = manager(
        monkeypatch, lock, continuity_preview(), world_model=world,
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)

    assert bundle["identity_continuity"] == {
        "ok": True,
        "allow_refresh": True,
        "reason": "same_confirmed_identity_observed",
        "entity_id": "entity-1",
        "identity_id": IDENTITY,
    }
    assert bundle["identity_refresh"]["allow_refresh"] is True
    assert bundle["identity_refresh"]["observation_update"] == {
        "bbox": {"x1": 100.0, "y1": 100.0, "x2": 288.0, "y2": 364.0},
        "image_width": 640.0,
        "image_height": 480.0,
        "source_timestamp": STAMP,
        "source": "marvin_local_tracker",
    }
    assert len(world.update_calls) == 1
    write = world.update_calls[0]
    assert write["entity_id"] == "entity-1"
    assert write["label"] == "marvin" and write["entity_type"] == "person"
    assert "identity_id" not in write["attributes"]
    assert "entity_id" not in write["attributes"]
    assert world.entities[0]["entity_id"] == "entity-1"
    assert world.entities[0]["attributes"]["identity_id"] == IDENTITY
    assert world.entities[0]["attributes"]["operator_confirmed"] is True
    assert lock.snapshot() == snapshot()
    assert len(lock.calls) == 1


def test_valid_refresh_dry_run_invokes_no_motion_executor(monkeypatch):
    world = FakeWorldModel([confirmed_world_entity()])
    instance = manager(
        monkeypatch,
        FakeTargetLock(),
        continuity_preview(),
        world_model=world,
    )
    executor_calls = []
    monkeypatch.setattr(
        instance,
        "execute_marvin_search_step",
        lambda *args, **kwargs: executor_calls.append("search"),
    )
    monkeypatch.setattr(
        instance,
        "execute_marvin_pursuit_step",
        lambda *args, **kwargs: executor_calls.append("pursuit"),
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "state": "SEARCHING",
            "pursuit_authorized": False,
            "selected_identity_id": IDENTITY,
            "reason": "no_current_lock",
        },
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_arrival",
        lambda *args, **kwargs: {
            "ok": True,
            "arrived_at_marvin": False,
            "selected_identity_id": IDENTITY,
            "reason": "not_at_standoff",
        },
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)
    result = instance.execute_find_marvin_controller(
        lambda: bundle, max_actions=1, dry_run=True,
    )

    assert bundle["identity_refresh"]["allow_refresh"] is True
    assert result["dry_run"] is True
    assert result["actions_executed"] == 0
    assert result["next_route"] == "search"
    assert executor_calls == []


def test_failed_continuity_is_read_only_and_dry_run_calls_no_executor(monkeypatch):
    waiting = FakeTargetLock(
        locked(found=False, tracking_mode="WAITING_FOR_IDENTITY"),
        snapshot("WAITING_FOR_IDENTITY"),
    )
    original_lock = waiting.snapshot()
    world = FakeWorldModel([confirmed_world_entity()])
    original_entities = deepcopy(world.entities)
    instance = manager(monkeypatch, waiting, preview(), world_model=world)
    executor_calls = []
    monkeypatch.setattr(
        instance, "execute_marvin_search_step",
        lambda *args, **kwargs: executor_calls.append("search"),
    )
    monkeypatch.setattr(
        instance, "execute_marvin_pursuit_step",
        lambda *args, **kwargs: executor_calls.append("pursuit"),
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_pursuit_state",
        lambda *args, **kwargs: {
            "state": "REACQUIRE_REQUIRED",
            "pursuit_authorized": False,
            "selected_identity_id": IDENTITY,
            "reason": "waiting_for_identity",
        },
    )
    monkeypatch.setattr(
        behavior_manager_module,
        "evaluate_marvin_arrival",
        lambda *args, **kwargs: {
            "ok": True,
            "arrived_at_marvin": False,
            "selected_identity_id": IDENTITY,
            "reason": "target_lock_not_locked",
        },
    )

    bundle = instance.build_find_marvin_controller_state(now=STAMP)
    assert bundle["identity_continuity"]["allow_refresh"] is False
    assert waiting.snapshot() == original_lock
    assert world.entities == original_entities
    assert world.update_calls == []

    result = instance.execute_find_marvin_controller(
        lambda: bundle, max_actions=1, dry_run=True,
    )
    assert result["next_route"] == "search"
    assert result["actions_executed"] == 0
    assert executor_calls == []


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


def test_unlocked_target_lock_uses_read_only_visual_session_builder(monkeypatch):
    waiting = FakeTargetLock(
        locked(found=False, tracking_mode="WAITING_FOR_IDENTITY"),
        snapshot("WAITING_FOR_IDENTITY"),
    )
    world = FakeWorldModel([confirmed_world_entity()])
    bundle = manager(monkeypatch, waiting, world_model=world).build_find_marvin_controller_state()
    assert bundle["selected_identity_id"] is None
    assert bundle["target_lock_snapshot"]["tracking_mode"] == "WAITING_FOR_IDENTITY"
    assert waiting.calls == [] and world.update_calls == []
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


def _v2_preview(error=203.0, *, quality=0.97, matched=True, stamp=101,
                identity_source="gemini_marvin_candidate_selection",
                identity_confirmed=True):
    # Each fake observation gets its own receipt timestamp; a large suite
    # must not expire it while earlier test modules are running.
    frame_timestamp = datetime.now(timezone.utc).isoformat()
    width, height = 640.0, 480.0
    cx = width / 2.0 + error
    bbox = {"x1": cx - 40.0, "y1": 100.0, "x2": cx + 40.0, "y2": 300.0}
    return {
        "ok": True, "preview": True, "authoritative": False,
        "target": "marvin", "target_found": True,
        "source": "marvin_local_tracker", "identity_confirmed": identity_confirmed,
        "identity_source": identity_source, "identity_source_frame_stamp_ns": stamp,
        "motion_authorized_marvin_candidate": True,
        "source_timestamp": frame_timestamp, "vision_timestamp": frame_timestamp,
        "proposal_label": "chair", "proposal_confidence": 0.91,
        "bbox": dict(bbox), "image_width": width, "image_height": height,
        "opencv_tracker": {"active": True, "matched": matched,
                           "quality": quality, "threshold": 0.80,
                           "bbox": dict(bbox), "image_width": width,
                           "image_height": height, "center_x": cx,
                           "center_y": 200.0, "horizontal_error": error,
                           "source_frame_stamp_ns": stamp + 1,
                           "received_monotonic_seconds": time.monotonic(),
                           "reason": "matched"},
    }


class _ReadOnlyV2Behavior:
    def __init__(self, preview):
        self.preview = preview
        self.motion_calls = 0

    @staticmethod
    def _marvin_v2_preview_is_verified(preview):
        return BehaviorManager._marvin_v2_preview_is_verified(preview)

    def observe_find_marvin_v2(self):
        return {"preview_result": self.preview, "target_lock_result": {},
                "target_lock_snapshot": {}, "selected_identity_id": None,
                "identity_evidence": None, "bridge_result": None}

    def execute_find_marvin_controller(self, *args, **kwargs):
        self.motion_calls += 1
        raise AssertionError("controller execution is forbidden")


def _v2_runtime(preview):
    runtime = object.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime._marvin_alignment_observation = None
    runtime._marvin_alignment_consensus = []
    runtime._marvin_alignment_geometry_history = []
    runtime._marvin_alignment_consumed_source_frame_stamps = set()
    behavior = _ReadOnlyV2Behavior(preview)
    runtime.behavior_manager = behavior
    runtime.marvin_camera_model = {
        "fx_pixels": 320.0, "cx_pixels": 320.0, "image_width": 640,
        "x_m": 0.0, "y_m": 0.0, "yaw_degrees": 0.0, "range_uncertainty_m": 0.0,
    }
    runtime.lidar_worker = SimpleNamespace(session="v2-lidar", running=True)
    runtime.world_model = SimpleNamespace(get_lidar_obstacles=Mock(return_value={
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": "v2-lidar", "effective_age_seconds": 0.0,
        "acquisition_sequence": 1,
        "local_motion_geometry": {"valid": True, "frame_id": "lidar_link",
                                  "points": [{"x_m": 1.0, "y_m": y / 1000.0}
                                             for y in range(-200, 201)]},
    }))
    from test_marvin_lidar_standoff import lidar_at
    runtime.world_model.get_lidar_obstacles.return_value = lidar_at(1.0)
    original_points = runtime.world_model.get_lidar_obstacles.return_value["local_motion_geometry"]["points"]
    target_points = original_points[-7:]
    def read(**kwargs):
        import math
        scan = runtime.world_model.get_lidar_obstacles.return_value
        if scan["local_motion_geometry"]["points"] is original_points:
            center = behavior.preview["opencv_tracker"]["center_x"]
            bearing = math.atan((320.0 - center) / 320.0)
            for n, point in enumerate(target_points):
                point.update(x_m=math.cos(bearing), y_m=math.sin(bearing) + (n-3)/1000.0)
        return scan
    runtime.world_model.get_lidar_obstacles.side_effect = read
    return runtime, behavior


def _observe_v2_series(runtime, behavior, values):
    results = []
    for stamp, error, updates in values:
        preview = _v2_preview(error, stamp=stamp, **updates)
        behavior.preview = preview
        results.append(runtime.observe_find_marvin_v2())
    return results


def _v2_semantic_tracker_association_failure(stamp):
    """Build the runtime preview from a real BehaviorManager rejection."""
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    manager._marvin_v2_tracker_episode = None
    tracker_stamps = iter((900 + stamp, 901 + stamp))

    def acquire(candidate, _diagnostics, *, episode, existing_tracker=None, **_kwargs):
        if existing_tracker is None:
            episode["marvin_tracker"] = object()
        bbox = manager._expand_marvin_tracker_seed_bbox(
            candidate["bbox"], candidate["image_width"], candidate["image_height"],
        )
        return {"opencv_tracker": {
            "source_frame_stamp_ns": next(tracker_stamps),
            "bbox": bbox,
        }}

    manager._acquire_marvin_tracker_observation_from_candidate = acquire
    frame = SimpleNamespace(received_at=STAMP)
    candidate = lambda bbox: {"bbox": bbox, "image_width": 640, "image_height": 480}
    manager._acquire_strict_v2_tracker_observation_from_candidate(
        candidate({"x1": 212, "y1": 0, "x2": 640, "y2": 384}), {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=1, execution_guard=None, frame=frame,
    )
    manager._acquire_strict_v2_tracker_observation_from_candidate(
        candidate({"x1": 221, "y1": 0, "x2": 619, "y2": 376}), {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=2, execution_guard=None, frame=frame,
    )
    rejected = manager._acquire_strict_v2_tracker_observation_from_candidate(
        candidate({"x1": 355, "y1": 16, "x2": 640, "y2": 403}), {},
        identity_source="gemini_marvin_candidate_selection",
        identity_source_frame_stamp_ns=3, execution_guard=None, frame=frame,
    )
    assert rejected["reason"] == "marvin_v2_semantic_tracker_association_failed"
    assert manager._marvin_v2_tracker_episode is None

    # preview_find_object deliberately builds a negative strict preview for
    # this result.  Preserve the actual diagnostic it propagates to runtime.
    preview = _v2_preview(175.0, stamp=stamp)
    preview.update(
        target_found=False,
        identity_confirmed=False,
        identity_source=None,
        motion_authorized_marvin_candidate=False,
        reason=rejected["reason"],
        strict_tracker_episode=rejected["strict_tracker_episode"],
    )
    return preview


@pytest.mark.parametrize(("error", "decision"), [(-203.0, "TURN_LEFT"), (203.0, "TURN_RIGHT"), (0.0, "FORWARD")])
def test_v2_observe_uses_fresh_gemini_and_exposes_nonexecuting_decision(error, decision):
    runtime, behavior = _v2_runtime(_v2_preview(error))
    result = runtime.observe_find_marvin_v2()
    assert result["ok"] is True
    assert result["read_only"] is True and result["authoritative"] is False
    assert result["executed"] is False
    assert result["identity_source"] == "gemini_marvin_candidate_selection"
    assert result["proposal_label"] == "chair"
    assert result["controller"]["decision"] == decision
    assert behavior.motion_calls == 0


@pytest.mark.parametrize("preview", [
    _v2_preview(203.0, identity_source="marvin_session_continuity"),
    _v2_preview(203.0, identity_confirmed=False),
    _v2_preview(203.0, quality=0.79),
    _v2_preview(203.0, matched=False),
])
def test_v2_observe_fails_closed_without_strict_identity_and_tracker(preview):
    runtime, behavior = _v2_runtime(preview)
    result = runtime.observe_find_marvin_v2()
    assert result["ok"] is False and result["executed"] is False
    assert result["controller"]["decision"] == "REVERIFY_REQUIRED"
    assert behavior.motion_calls == 0


def test_newer_non_authorizing_v2_observation_clears_alignment_authorization():
    runtime = object.__new__(CognitiveRuntime)
    turn = Mock(side_effect=AssertionError("stale observation must not turn"))
    robot = SimpleNamespace(
        stop=Mock(side_effect=AssertionError("stale observation must not stop")),
        motion=Mock(side_effect=AssertionError("stale observation must not move")),
    )
    runtime._state_lock = threading.RLock()
    runtime._marvin_alignment_observation = None
    runtime._marvin_alignment_consensus = []
    runtime._marvin_alignment_geometry_history = []
    runtime._marvin_alignment_consumed_source_frame_stamps = set()
    behavior = _ReadOnlyV2Behavior(_v2_preview(203.0, stamp=101))
    behavior._execute_target_directed_turn = turn
    behavior.robot = robot
    runtime.behavior_manager = behavior
    runtime.running = True
    runtime.lidar_worker = SimpleNamespace(session="test-lidar", running=True)
    runtime.world_model = SimpleNamespace(
        get_lidar_obstacles=Mock(return_value={
            "available": True, "valid": True, "reason": "fresh",
            "producer_session": "test-lidar",
            "local_motion_geometry": {"valid": True},
        }),
    )

    _observe_v2_series(runtime, behavior, [
        (101, 203.0, {}), (102, 204.0, {}), (103, 202.0, {}),
    ])
    stamp = behavior.preview["opencv_tracker"]["source_frame_stamp_ns"]
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == stamp

    # A newer unmatched tracker frame is a non-authorizing V2 observation.
    # It must clear, rather than leave, the prior turn authorization cached.
    newer = _v2_preview(203.0, matched=False, stamp=104)
    behavior.preview = newer
    second = runtime.observe_find_marvin_v2()

    assert second["ok"] is False
    assert runtime._marvin_alignment_observation is None
    stale_attempt = runtime.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=stamp,
    )
    assert stale_attempt["reason"] == "marvin_alignment_observation_not_current"
    turn.assert_not_called()
    robot.motion.assert_not_called()


@pytest.mark.parametrize(("low_error", "high_error", "direction"), [
    (105.0, 175.0, "TURN_RIGHT"),
    (-105.0, -175.0, "TURN_LEFT"),
])
def test_v2_association_failure_clears_current_authorization_before_reacquisition(
    low_error, high_error, direction,
):
    """A BehaviorManager association failure is a full downstream boundary."""
    runtime, behavior = _v2_runtime(_v2_preview(low_error, stamp=100))

    # Each current strict observation can authorize one bounded action.
    _observe_v2_series(runtime, behavior, [
        (100, low_error, {}), (101, low_error + (2 if low_error > 0 else -2), {}),
        (102, low_error + (1 if low_error > 0 else -1), {}),
    ])
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 103
    assert runtime._marvin_alignment_observation["controller_decision"] == direction

    # This is the negative preview shape returned by the actual strict-V2
    # BehaviorManager association boundary.  It must invalidate every older
    # runtime admission layer before a competing mode can re-acquire.
    behavior.preview = _v2_semantic_tracker_association_failure(103)
    failed = runtime.observe_find_marvin_v2()
    assert failed["strict_tracker_episode"]["semantic_association"] == "failed"
    assert runtime._marvin_alignment_geometry_history == []
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation is None

    # The first new strict observation is a new episode's current action
    # evidence; it cannot reuse any LOW stamp or observation.
    _observe_v2_series(runtime, behavior, [
        (104, high_error, {}), (105, high_error + (2 if high_error > 0 else -2), {}),
    ])
    assert len(runtime._marvin_alignment_geometry_history) == 2
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 106
    assert runtime._marvin_alignment_observation["controller_decision"] == direction


@pytest.mark.parametrize(("error", "decision"), [
    (60.0, "TURN_RIGHT"), (-60.0, "TURN_LEFT"),
])
def test_v2_current_strict_observation_authorizes_one_bounded_turn(
    error, decision,
):
    runtime, behavior = _v2_runtime(_v2_preview(error, stamp=100))
    offset = 2 if error > 0 else -2
    results = []
    for stamp, current_error in ((100, error), (101, error + offset)):
        behavior.preview = _v2_preview(current_error, stamp=stamp)
        results.append(runtime.observe_find_marvin_v2())
        assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == stamp + 1
    behavior.preview = _v2_preview(error + 2 * offset, stamp=102)
    results.append(runtime.observe_find_marvin_v2())
    assert all(result["controller"]["decision"] == decision for result in results)
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 103
    assert runtime._marvin_alignment_observation["controller_decision"] == decision
    assert results[0]["opencv_tracker"]["source_frame_stamp_ns"] == 101
    assert results[1]["opencv_tracker"]["source_frame_stamp_ns"] == 102


def test_v2_post_action_locked_tracker_evidence_can_authorize_new_frame_without_gemini():
    continuity = _v2_preview(80.0, stamp=501,
                             identity_source="marvin_locked_tracker_continuity")
    continuity["post_action_tracker_continuity"] = True
    continuity["post_action_source_frame_stamp_ns"] = 500
    continuity["marvin_tracking_episode"] = {
        "episode_id": "marvin-v2-450",
        "post_action_source_frame_stamp_ns": 500,
        "state": "POST_ACTION_TRACKED",
    }
    runtime, behavior = _v2_runtime(continuity)

    result = runtime.observe_find_marvin_v2()

    assert result["ok"] is True
    assert result["identity_confirmed"] is True
    assert result["fresh_gemini_required"] is False
    assert result["post_action_tracker_continuity"] is True
    assert result["controller"]["decision"] == "TURN_RIGHT"
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 502
    assert runtime._marvin_alignment_observation["identity_source"] == "marvin_locked_tracker_continuity"
    assert runtime._marvin_alignment_observation["post_action_tracker_continuity"] is True
    assert behavior.motion_calls == 0


def test_post_action_identity_source_without_continuity_proof_cannot_authorize():
    continuity = _v2_preview(80.0, stamp=601,
                             identity_source="marvin_locked_tracker_continuity")
    continuity["post_action_tracker_continuity"] = True
    continuity["post_action_source_frame_stamp_ns"] = 600
    continuity["marvin_tracking_episode"] = {"state": "TRACKING"}
    runtime, _behavior = _v2_runtime(continuity)

    result = runtime.observe_find_marvin_v2()

    assert result["controller"]["decision"] == "REVERIFY_REQUIRED"
    assert runtime._marvin_alignment_observation is None


def test_behavior_manager_locked_post_action_tracker_flows_to_runtime_authorization():
    bbox = {"x1": 380, "y1": 150, "x2": 500, "y2": 250}
    stamp = 9001
    manager = object.__new__(BehaviorManager)
    manager._continue_strict_v2_tracker_after_action = lambda: {
        "found": True, "stale": False, "target": "marvin", "label": "marvin",
        "source": "marvin_local_tracker",
        "source_timestamp": datetime.now(timezone.utc).isoformat(),
        "bbox": bbox, "cx": 440.0, "cy": 200.0, "area": 12000,
        "image_width": 640, "image_height": 480,
        "identity_confirmed": True,
        "identity_source": "marvin_locked_tracker_continuity",
        "identity_source_frame_stamp_ns": 8000,
        "motion_authorized_marvin_candidate": True,
        "post_action_tracker_continuity": True,
        "post_action_source_frame_stamp_ns": 9000,
        "strict_tracker_episode": {"active": True, "continued_existing_tracker": True},
        "marvin_tracking_episode": {
            "episode_id": "marvin-v2-8000",
            "post_action_source_frame_stamp_ns": 9000,
            "state": "POST_ACTION_TRACKED",
        },
        "opencv_tracker": {
            "active": True, "matched": True, "quality": 0.96, "threshold": 0.80,
            "bbox": bbox, "image_width": 640, "image_height": 480,
            "center_x": 440.0, "center_y": 200.0,
            "horizontal_error": 120.0, "source_frame_stamp_ns": stamp,
        },
    }
    runtime = object.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime._marvin_alignment_observation = None
    runtime._marvin_alignment_consensus = []
    runtime._marvin_alignment_geometry_history = []
    runtime._marvin_alignment_consumed_source_frame_stamps = set()
    runtime.behavior_manager = manager

    result = runtime.observe_find_marvin_v2()

    assert result["ok"] is True
    assert result["identity_source"] == "marvin_locked_tracker_continuity"
    assert result["fresh_gemini_required"] is False
    assert result["opencv_tracker"]["source_frame_stamp_ns"] == stamp
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == stamp
    assert runtime._marvin_alignment_observation["post_action_tracker_continuity"] is True


def test_two_turns_use_new_post_action_tracker_frames_without_runtime_restart(monkeypatch):
    monkeypatch.setattr(runtime_module.time, "time_ns", lambda: 1010)
    manager = object.__new__(BehaviorManager)
    manager._marvin_v2_tracker_episode_lock = threading.RLock()
    tracker_object = object()
    manager._marvin_v2_tracker_episode = {
        "marvin_tracker": tracker_object,
        "tracker_bbox": {"x1": 300, "y1": 100, "x2": 500, "y2": 300},
        "last_tracker_source_frame_stamp_ns": 1000,
        "last_tracker_received_at": STAMP,
        "identity_source": "gemini_marvin_candidate_selection",
        "identity_source_frame_stamp_ns": 900,
        "episode_id": "marvin-v2-900",
    }
    manager.semantic_vision = SimpleNamespace(fetch_frame=Mock())
    continuation_calls = []

    def confirm(current_tracker, **kwargs):
        continuation_calls.append((current_tracker, kwargs))
        stamp = 1001 + len(continuation_calls) - 1
        bbox = {"x1": 390, "y1": 140, "x2": 510, "y2": 240}
        opencv = {
            "active": True, "matched": True, "quality": 0.95, "threshold": 0.80,
            "bbox": bbox, "image_width": 640, "image_height": 480,
            "center_x": 450.0, "center_y": 190.0,
            "horizontal_error": 130.0, "source_frame_stamp_ns": stamp,
            "received_monotonic_seconds": runtime_module.time.monotonic(),
        }
        return {
            "found": True, "stale": False, "target": "marvin", "label": "marvin",
            "source": "marvin_local_tracker",
            "source_timestamp": datetime.now(timezone.utc).isoformat(),
            "bbox": bbox, "cx": 450.0, "cy": 190.0, "area": 12000,
            "image_width": 640, "image_height": 480,
            "opencv_tracker": opencv,
        }

    monkeypatch.setattr(manager, "_confirm_marvin_local_tracker_frames", confirm)
    robot = SimpleNamespace(stop=Mock(return_value={"ok": True}))
    turns = []
    manager.robot = robot
    manager._execute_target_directed_turn = lambda direction, speed, duration, **kwargs: (
        turns.append((direction, speed, duration, kwargs)) or
        {"ok": True, "permitted": True, "confirmed_forwarded": True}
    )
    runtime = object.__new__(CognitiveRuntime)
    runtime.running = True
    runtime._state_lock = threading.RLock()
    receipt = time.monotonic()
    runtime._marvin_alignment_observation = {
        "source_frame_stamp_ns": 1000,
        "received_monotonic_seconds": receipt,
        "identity_confirmed": True,
        "identity_source": "gemini_marvin_candidate_selection",
        "opencv_tracker": {
            "active": True, "matched": True, "quality": 0.95,
            "threshold": 0.80, "bbox": {"x1": 300, "y1": 100, "x2": 500, "y2": 300},
            "source_frame_stamp_ns": 1000,
            "received_monotonic_seconds": receipt,
        },
        "controller_state": "VISUAL_READY_TO_ALIGN",
        "controller_decision": "TURN_RIGHT",
    }
    runtime._marvin_alignment_consensus = []
    runtime._marvin_alignment_geometry_history = []
    runtime._marvin_alignment_consumed_source_frame_stamps = set()
    runtime.behavior_manager = manager
    runtime.world_model = SimpleNamespace(get_lidar_obstacles=lambda **_kwargs: {
        "available": True, "valid": True, "reason": "fresh",
        "producer_session": "test-session", "local_motion_geometry": {"valid": True},
    })
    runtime.lidar_worker = SimpleNamespace(session="test-session", running=True)

    first = runtime.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=1000,
    )
    assert first["motion_executed"] is True
    assert runtime.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=1000,
    )["reason"] == "marvin_alignment_observation_already_consumed"

    next_observation = runtime.observe_find_marvin_v2()
    assert next_observation["identity_source"] == "marvin_locked_tracker_continuity"
    next_stamp = next_observation["opencv_tracker"]["source_frame_stamp_ns"]
    second = runtime.execute_single_marvin_alignment(
        direction="RIGHT", angular_speed=0.25, duration=0.50,
        source_frame_stamp_ns=next_stamp,
    )

    assert second["motion_executed"] is True
    assert [call[0] for call in turns] == ["RIGHT", "RIGHT"]
    assert len(continuation_calls) == 1
    assert continuation_calls[0][0] is tracker_object
    assert continuation_calls[0][1]["minimum_source_frame_stamp_ns"] == 1000
    assert manager._marvin_v2_tracker_episode["last_tracker_source_frame_stamp_ns"] == next_stamp
    assert robot.stop.call_count == 2


def test_v2_centered_observations_authorize_only_current_guarded_forward():
    runtime, behavior = _v2_runtime(_v2_preview(0.0, stamp=200))
    _observe_v2_series(runtime, behavior, [
        (200, 0.0, {}), (201, 20.0, {}), (202, -20.0, {}),
    ])
    assert runtime._marvin_alignment_observation["controller_state"] == "VISUAL_READY_TO_APPROACH"
    assert runtime._marvin_alignment_observation["controller_decision"] == "FORWARD"
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 203
    assert runtime._marvin_alignment_consensus == []


def test_v2_arrival_never_authorizes_another_pursuit_action(monkeypatch):
    runtime, behavior = _v2_runtime(_v2_preview(0.0, stamp=210))
    from marvin_target_range_association import MarvinTargetRangeAssociation
    runtime._marvin_target_range_association = MarvinTargetRangeAssociation()
    runtime._marvin_target_range_association.anchor = {
        "measured_distance_m": .5, "target_distance_m": .5, "translation_bound_m": 0.,
        "producer_session": "v2-lidar", "acquisition_sequence": 0, "source_frame_stamp_ns": 0}
    runtime.world_model.get_lidar_obstacles.return_value["local_motion_geometry"]["points"] = [
        {"x_m": 0.5, "y_m": y / 1000.0} for y in range(-3, 4)
    ]
    monkeypatch.setattr(
        runtime_module, "evaluate_marvin_visual_arrival",
        lambda *_args, **_kwargs: {"ok": True, "arrived_at_marvin": True},
    )
    result = runtime.observe_find_marvin_v2()
    assert result["controller"]["decision"] == "ARRIVED"
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []


def test_v2_forward_resets_alignment_admission_without_discarding_episode_evidence():
    """CENTERED is an admission reset, not a semantic-tracker replacement."""
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=220))
    _observe_v2_series(runtime, behavior, [
        (220, 60.0, {}), (221, 62.0, {}),
    ])
    assert runtime._marvin_alignment_consensus == []

    # An associated strict tracker remains visible in the preview, but once
    # its controller result is FORWARD the runtime must discard all pending
    # physical-alignment evidence.
    centered = _v2_preview(20.0, stamp=222)
    centered["strict_tracker_episode"] = {
        "active": True,
        "continued_existing_tracker": True,
        "initialized_this_observation": False,
        "semantic_association": "accepted",
        "association_metric": "iou",
        "association_iou": 0.91,
        "association_threshold": 0.70,
        "reset_reason": None,
    }
    behavior.preview = centered
    result = runtime.observe_find_marvin_v2()
    assert result["controller"]["decision"] == "FORWARD"
    assert result["strict_tracker_episode"]["active"] is True
    assert runtime._marvin_alignment_geometry_history == []
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation["controller_decision"] == "FORWARD"

    # A later associated turn uses only its own current source stamp.
    behavior.preview = _v2_preview(61.0, stamp=223)
    runtime.observe_find_marvin_v2()
    assert len(runtime._marvin_alignment_geometry_history) == 1
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 224


def test_v2_geometry_outlier_resets_current_authorization_until_new_valid_observation():
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=300))
    _observe_v2_series(runtime, behavior, [
        (300, 60.0, {}), (301, 62.0, {}), (302, 175.0, {}),
    ])
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []
    _observe_v2_series(runtime, behavior, [(303, 60.0, {})])
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 304


def test_v2_seed_geometry_discontinuity_blocks_high_quality_outlier_before_consensus():
    runtime, behavior = _v2_runtime(_v2_preview(99.0, stamp=700))
    results = _observe_v2_series(runtime, behavior, [
        # center_x 419.0, then 423.5: stable newly seeded geometry.
        (700, 99.0, {}), (701, 103.5, {}),
        # center_x 509.5: observed gross jump despite a high tracker score.
        (702, 189.5, {"quality": 0.99}),
    ])

    assert results[0]["geometry_continuity"] == {
        "accepted": True,
        "reason": "geometry_baseline_established",
        "history_length": 1,
        "center_delta_px": None,
    }
    assert results[1]["geometry_continuity"]["accepted"] is True
    assert results[2]["geometry_continuity"] == {
        "accepted": False,
        "reason": "geometry_center_discontinuity",
        "history_length": 0,
        "center_delta_px": 88.25,
    }
    assert runtime._marvin_alignment_geometry_history == []
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation is None


def test_v2_seed_geometry_recovery_uses_only_new_current_observation():
    runtime, behavior = _v2_runtime(_v2_preview(99.0, stamp=710))
    _observe_v2_series(runtime, behavior, [
        (710, 99.0, {}), (711, 103.5, {}), (712, 189.5, {}),
    ])
    assert runtime._marvin_alignment_observation is None

    recovered = _observe_v2_series(runtime, behavior, [(713, 99.0, {})])
    assert recovered[0]["geometry_continuity"]["history_length"] == 1
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 714


def test_v2_seed_geometry_continuity_accepts_exact_30_pixels_only():
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=717))
    exact, over = _observe_v2_series(runtime, behavior, [
        # center_x 380.0 establishes the baseline; 410.0 is exactly 30 px.
        (717, 60.0, {}), (718, 90.0, {}),
    ])
    assert exact["geometry_continuity"]["accepted"] is True
    assert over["geometry_continuity"] == {
        "accepted": True,
        "reason": "geometry_continuous",
        "history_length": 2,
        "center_delta_px": 30.0,
    }

    over_runtime, over_behavior = _v2_runtime(_v2_preview(60.0, stamp=719))
    _observe_v2_series(over_runtime, over_behavior, [(719, 60.0, {})])
    rejected = _observe_v2_series(
        over_runtime, over_behavior, [(720, 90.0001, {})],
    )[0]
    assert rejected["geometry_continuity"]["accepted"] is False
    assert rejected["geometry_continuity"]["reason"] == "geometry_center_discontinuity"
    assert rejected["geometry_continuity"]["center_delta_px"] > 30.0
    assert over_runtime._marvin_alignment_geometry_history == []
    assert over_runtime._marvin_alignment_consensus == []
    assert over_runtime._marvin_alignment_observation is None


def test_v2_centered_observation_resets_turn_history_and_becomes_forward_current_observation():
    runtime, behavior = _v2_runtime(_v2_preview(99.0, stamp=720))
    _observe_v2_series(runtime, behavior, [
        (720, 99.0, {}), (721, 95.0, {}), (722, 44.0, {}),
    ])
    assert runtime._marvin_alignment_geometry_history == []
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_observation["controller_decision"] == "FORWARD"


def test_v2_current_authorization_rolls_to_newest_stamp():
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=350))
    _observe_v2_series(runtime, behavior, [
        # center_x values 380, 395, and 410: exactly 30 px of spread.
        (350, 60.0, {}), (351, 75.0, {}), (352, 90.0, {}),
    ])
    third_stamp = runtime._marvin_alignment_observation["source_frame_stamp_ns"]
    assert third_stamp == 353
    _observe_v2_series(runtime, behavior, [(353, 80.0, {})])
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] == 354
    assert runtime._marvin_alignment_observation["source_frame_stamp_ns"] != third_stamp


@pytest.mark.parametrize("updates", [
    {"matched": False},
    {"quality": 0.79},
    {"identity_confirmed": False},
])
def test_v2_non_strict_observation_resets_alignment_consensus(updates):
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=400))
    _observe_v2_series(runtime, behavior, [
        (400, 60.0, {}), (401, 62.0, {}), (402, 60.0, updates),
    ])
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_geometry_history == []


def test_v2_direction_change_and_nonmonotonic_stamp_reset_consensus():
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=500))
    _observe_v2_series(runtime, behavior, [
        (500, 60.0, {}), (501, 62.0, {}), (502, -60.0, {}),
    ])
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_geometry_history == []
    _observe_v2_series(runtime, behavior, [
        (503, 60.0, {}), (503, 62.0, {}),
    ])
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []


def test_v2_malformed_bbox_resets_alignment_consensus():
    runtime, behavior = _v2_runtime(_v2_preview(60.0, stamp=600))
    malformed = _v2_preview(60.0, stamp=602)
    malformed["opencv_tracker"]["bbox"] = {"x1": 1, "x2": 2}
    _observe_v2_series(runtime, behavior, [
        (600, 60.0, {}), (601, 62.0, {}),
    ])
    behavior.preview = malformed
    runtime.observe_find_marvin_v2()
    assert runtime._marvin_alignment_observation is None
    assert runtime._marvin_alignment_consensus == []
    assert runtime._marvin_alignment_geometry_history == []


def test_v2_observe_get_route_calls_only_observer():
    runtime = SimpleNamespace(observe_find_marvin_v2=Mock(return_value={
        "ok": True, "read_only": True, "authoritative": False, "executed": False,
    }))
    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-marvin/v2/observe"
    handler.server = SimpleNamespace(runtime=runtime)
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_GET()
    assert responses[0][0] == 200 and responses[0][1]["executed"] is False
    runtime.observe_find_marvin_v2.assert_called_once_with()
