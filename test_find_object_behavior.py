#!/usr/bin/env python3

import json
import math
import threading

import pytest

import behavior_manager as behavior_module
from behavior_manager import BehaviorManager
from mission_types import create_mission
from runtime import CognitiveRuntime
from tracking_state import build_tracking_state, empty_tracking_state


class FakeRobotBridgeClient:
    def __init__(self):
        self.calls = []

    def move_forward(
        self,
        speed=0.10,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "move_forward",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def turn_left(
        self,
        speed=0.5,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "turn_left",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def turn_right(
        self,
        speed=0.5,
        seconds=1.0,
    ):
        self.calls.append(
            (
                "turn_right",
                speed,
                seconds,
            )
        )

        return {
            "ok": True,
            "automatic_stop": True,
        }

    def stop(self):
        self.calls.append(
            ("stop",)
        )

        return {
            "ok": True,
            "action": "stop",
        }


class SequencedVisionAdapter:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.last_result = None
        self.candidate_calls = 0
        self._allow_last_result = False

    def find_target(self, target):
        if not self.results:
            if self._allow_last_result and self.last_result is not None:
                return dict(self.last_result)
            raise AssertionError("Vision sequence exhausted.")

        self.calls += 1

        result = dict(
            self.results.pop(0)
        )

        result.setdefault(
            "target",
            target,
        )
        self.last_result = dict(result)

        return result

    def fetch_target_candidates(self, target):
        self.candidate_calls += 1
        observation = self.last_result or found_target(target)
        if observation.get("found") is not True or observation.get("stale") is True:
            return {
                "timestamp": f"post-{self.calls}-{self.candidate_calls}",
                "camera_running": True,
                "detections": [],
            }
        cx = float(observation.get("cx") or 320.0)
        cy = float(observation.get("cy") or 240.0)
        bbox = observation.get("bbox") or {
            "x1": cx - 60.0,
            "y1": cy - 60.0,
            "x2": cx + 60.0,
            "y2": cy + 60.0,
        }
        return {
            "timestamp": f"post-{self.calls}-{self.candidate_calls}",
            "camera_running": True,
            "detections": [{
                "label": target,
                "confidence": observation.get("confidence", 0.9),
                "x1": bbox["x1"],
                "y1": bbox["y1"],
                "x2": bbox["x2"],
                "y2": bbox["y2"],
                "center_x": cx,
                "center_y": cy,
                "area": observation.get("area", 12000.0),
                "image_width": observation.get("image_width", 640.0),
                "image_height": observation.get("image_height", 480.0),
            }],
        }

    @staticmethod
    def normalize_detection(detection):
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        return {
            "label": detection["label"],
            "confidence": float(detection["confidence"]),
            "cx": float(detection["center_x"]),
            "cy": float(detection["center_y"]),
            "area": float(detection["area"]),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_width": float(detection["image_width"]),
            "image_height": float(detection["image_height"]),
        }

    def process_detection_frame(self, detections):
        if detections:
            self._allow_last_result = True
            detection = detections[0]
            normalized = self.normalize_detection(detection)
            normalized.update(
                found=True,
                stale=False,
                target=detection.get("label"),
                last_seen=f"post-{self.calls}-{self.candidate_calls}",
            )
            self.last_result = normalized
        return detections


class CandidateVisionAdapter(SequencedVisionAdapter):
    def __init__(self, observations, candidate_payloads):
        super().__init__(observations)
        self.candidate_payloads = list(candidate_payloads)
        self.candidate_calls = 0
        self.promotions = []

    def fetch_target_candidates(self, target):
        self.candidate_calls += 1
        if not self.candidate_payloads:
            raise AssertionError("Candidate sequence exhausted.")
        return dict(self.candidate_payloads.pop(0))

    @staticmethod
    def normalize_detection(detection):
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        return {
            "label": detection["label"],
            "confidence": float(detection["confidence"]),
            "cx": float(detection["center_x"]),
            "cy": float(detection["center_y"]),
            "area": float(detection["area"]),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_width": float(detection["image_width"]),
            "image_height": float(detection["image_height"]),
        }

    def process_detection_frame(self, detections):
        self.promotions.extend(detections)
        return super().process_detection_frame(detections)


def candidate_detection(
    timestamp,
    *,
    bbox=(200, 160, 500, 460),
    confidence=0.15,
    label="backpack",
):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return {
        "timestamp": timestamp,
        "camera_running": True,
        "label": label,
        "found": True,
        "detections": [{
            "label": label,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "area": width * height,
            "image_width": 640,
            "image_height": 480,
            }],
        }

class AlwaysPermittedInterlock:
    def __init__(self):
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1
        return True, "fresh_clear"


class GuardedSearchRobot:
    def __init__(self, move_result=None, move_exception=None, interlock=None):
        self.calls = []
        self.move_result = move_result or {"ok": True, "automatic_stop": True}
        self.move_exception = move_exception
        self.forward_interlock = (
            interlock if interlock is not None else AlwaysPermittedInterlock()
        )

    def move_forward(self, speed, seconds):
        self.calls.append(("move_forward", speed, seconds))
        if self.move_exception is not None:
            raise self.move_exception
        return dict(self.move_result)

    def stop(self):
        self.calls.append(("stop",))
        return {"ok": True}


def not_found(target="backpack", stale=False):
    return {
        "found": False,
        "stale": stale,
        "target": target,
        "reason": "Target not visible.",
    }


def found_target(target="backpack"):
    return {
        "found": True,
        "stale": False,
        "target": target,
        "entity_id": "backpack-001",
        "last_seen": "0",
        "confidence": 0.9,
        "cx": 320.0,
        "cy": 240.0,
        "area": 12000.0,
        "image_width": 640.0,
        "image_height": 480.0,
    }


def semantic_only_target(target="backpack"):
    result = found_target(target)
    result.update(
        cx=None,
        cy=None,
        area=None,
        bbox=None,
        image_width=None,
        image_height=None,
    )
    return result


def guarded_search_manager(observations, turn_results=None):
    robot = GuardedSearchRobot()
    vision = SequencedVisionAdapter(observations)
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    # These search/centering tests exercise the historical one-step result
    # contract.  Multi-step behavior is covered explicitly below.
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.lidar_session = "session-1"
    calls = []
    results = list(turn_results or [])

    def execute_guarded_turn(direction, speed, duration, *, expected_lidar_session, now=None):
        calls.append((direction, speed, duration, expected_lidar_session))
        if results:
            return dict(results.pop(0))
        return {"ok": True, "permitted": True, "reason": "completed"}

    manager.execute_guarded_turn = execute_guarded_turn
    return manager, robot, vision, calls


def _mission():
    return create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Find my backpack",
        status="ACTIVE",
    )


def test_guarded_search_visible_before_turn_uses_zero_chunks():
    manager, robot, _vision, calls = guarded_search_manager(
        [found_target()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is True
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 0
    assert calls == []
    assert robot.calls == [("move_forward", 0.08, 0.50)]
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["executed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert result["confirmation_diagnostics"]["confirmation_status"] == (
        "target_confirmed"
    )


def test_guarded_search_rechecks_camera_after_first_chunk():
    manager, _robot, vision, calls = guarded_search_manager(
        [not_found(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1
    assert calls[0] == ("LEFT", 0.30, 1.0, "session-1")
    assert vision.calls == 2


def test_fresh_semantic_only_target_is_not_acquired_immediately():
    manager, _robot, _vision, calls = guarded_search_manager(
        [semantic_only_target()] * 4
    )
    result = manager.execute(_mission())
    assert result["target_found"] is False
    assert result["state"] == "SEARCH_EXHAUSTED"
    assert result["turn_chunks_attempted"] == 3
    assert len(calls) == 3


def test_semantic_only_target_then_actionable_geometry_is_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [semantic_only_target(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is True
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("cx", math.nan),
        ("cy", math.inf),
        ("area", 0.0),
        ("area", -1.0),
        ("image_width", 0.0),
        ("image_height", -1.0),
    ],
)
def test_invalid_target_geometry_is_not_acquired(field, value):
    invalid = found_target()
    invalid[field] = value
    manager, _robot, _vision, calls = guarded_search_manager(
        [invalid, found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_search_stops_after_second_chunk_when_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found(), not_found(), found_target()]
    )
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 2
    assert len(calls) == 2
    assert all(call[2] == 1.0 for call in calls)


def test_guarded_search_exhausts_at_three_independent_chunks():
    manager, _robot, vision, calls = guarded_search_manager(
        [not_found(), not_found(), not_found(), not_found()]
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["completed"] is False
    assert result["search_exhausted"] is True
    assert result["turn_chunks_attempted"] == 3
    assert result["turn_chunks_completed"] == 3
    assert len(calls) == 3
    assert all(call[1:3] == (0.30, 1.0) for call in calls)
    assert vision.calls == 4


def test_guarded_search_denial_stops_without_replay():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found()],
        [{"ok": False, "permitted": False, "reason": "stale_lidar"}],
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["completed"] is False
    assert result["state"] == "SEARCH_BLOCKED"
    assert result["reason"] == "stale_lidar"
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 0
    assert result["last_guarded_turn_result"]["reason"] == "stale_lidar"
    assert len(calls) == 1


def test_guarded_search_failure_after_successful_chunk_stops_immediately():
    manager, _robot, _vision, calls = guarded_search_manager(
        [not_found(), not_found(), not_found()],
        [
            {"ok": True, "permitted": True},
            {"ok": False, "permitted": True, "reason": "transport_exception"},
        ],
    )
    result = manager.execute(_mission())
    assert result["ok"] is False
    assert result["turn_chunks_attempted"] == 2
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 2


def test_stale_camera_detection_is_not_acquired():
    manager, _robot, _vision, calls = guarded_search_manager(
        [found_target(), found_target()]
    )
    # Mark the first observation stale after construction without changing
    # the existing target schema.
    manager.vision.results[0]["stale"] = True
    result = manager.execute(_mission())
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_search_results_are_json_serializable():
    cases = (
        ([found_target()], None),
        ([not_found()] * 4, None),
        ([not_found()], [{"ok": False, "reason": "turn_side_not_clear"}]),
    )
    for observations, turn_results in cases:
        manager, _robot, _vision, _calls = guarded_search_manager(
            observations, turn_results,
        )
        json.dumps(manager.execute(_mission()))


def test_same_candidate_timestamp_does_not_confirm(monkeypatch):
    payload = candidate_detection("frame-1")
    vision = CandidateVisionAdapter([], [payload, payload, payload])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    clock = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: next(clock, 1.0))
    monkeypatch.setattr(behavior_module.time, "sleep", lambda _seconds: None)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"


def test_confirmation_diagnostics_mark_cutoff_and_duplicate_frames(monkeypatch):
    cutoff = candidate_detection("frame-0")
    fresh = candidate_detection("frame-1")
    payloads = [cutoff, fresh, fresh, fresh]
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=vision,
    )
    clock = [0.0]
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        behavior_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    confirmed, status, diagnostics = (
        manager._confirm_target_candidates_with_status(
            "backpack",
            minimum_timestamp="frame-0",
            return_diagnostics=True,
        )
    )
    assert confirmed is None
    assert status == "target_reconfirmation_failed"
    assert any(attempt["before_cutoff"] for attempt in diagnostics["attempts"])
    assert any(attempt["duplicate_timestamp"] for attempt in diagnostics["attempts"])
    assert diagnostics["distinct_fresh_timestamps"] == 1
    assert diagnostics["actionable_frames"] == 1
    assert diagnostics["evidence_frames_evaluated"] == 1


def test_confirmation_diagnostics_show_two_distinct_supporting_frames():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", confidence=0.12),
            candidate_detection("frame-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed, status, diagnostics = (
        manager._confirm_target_candidates_with_status(
            "backpack", return_diagnostics=True
        )
    )
    assert confirmed is not None
    assert status == "target_confirmed"
    assert diagnostics["distinct_fresh_timestamps"] == 3
    assert diagnostics["actionable_frames"] == 3
    assert diagnostics["evidence_frames_evaluated"] == 3
    assert diagnostics["confirmation_window_seconds"] == 0.90
    assert diagnostics["qualified_support_reached"] is True
    assert diagnostics["qualified_fallback_used"] is False
    assert any(
        attempt["cluster_reached_support"]
        for attempt in diagnostics["attempts"]
    )
    assert diagnostics["confirmation_status"] == "target_confirmed"
    assert all("raw_detection" not in attempt for attempt in diagnostics["attempts"])
    json.dumps(diagnostics)


def test_confirmation_diagnostics_identify_geometry_rejection():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1", bbox=(0, 0, 100, 100)),
            candidate_detection("frame-2", bbox=(300, 300, 400, 400)),
            candidate_detection("frame-3", bbox=(500, 0, 600, 100)),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed, status, diagnostics = (
        manager._confirm_target_candidates_with_status(
            "backpack", return_diagnostics=True
        )
    )
    assert confirmed is None
    assert status == "target_reconfirmation_failed"
    outcomes = [
        outcome
        for attempt in diagnostics["attempts"]
        for outcome in attempt["association_outcomes"]
    ]
    assert outcomes
    assert any(
        outcome["matched"] is False
        and outcome["rejection_reason"] in {
            "center_distance",
            "geometric_thresholds",
        }
        for outcome in outcomes
    )


def test_confirmation_diagnostics_record_fetch_failure_after_actionable_frame():
    class FailingVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            if self.candidate_calls:
                raise TimeoutError("candidate endpoint unavailable")
            return super().fetch_target_candidates(target)

    vision = FailingVision([], [candidate_detection("frame-1")])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed, status, diagnostics = (
        manager._confirm_target_candidates_with_status(
            "backpack", return_diagnostics=True
        )
    )
    assert confirmed is None
    assert status == "target_reconfirmation_failed"
    assert diagnostics["attempts"][-1]["fetch_error"]["type"] == "TimeoutError"
    assert diagnostics["actionable_frames"] == 1
    assert diagnostics["qualified_support_reached"] is False
    assert diagnostics["qualified_fallback_used"] is False


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("candidate endpoint unavailable"),
        None,
        {"camera_running": True, "detections": []},
    ],
    ids=["timeout", "non_dict_payload", "missing_timestamp"],
)
def test_qualified_support_falls_back_after_optional_fetch_failure(failure):
    class FailingAfterSupportVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            if self.candidate_calls >= 2:
                self.candidate_calls += 1
                if isinstance(failure, BaseException):
                    raise failure
                return failure
            return super().fetch_target_candidates(target)

    vision = FailingAfterSupportVision(
        [], [candidate_detection("frame-1"), candidate_detection("frame-2")]
    )
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    confirmed, status, diagnostics = (
        manager._confirm_target_candidates_with_status(
            "backpack", return_diagnostics=True
        )
    )
    assert confirmed is not None
    assert status == "target_confirmed"
    assert diagnostics["qualified_support_reached"] is True
    assert diagnostics["qualified_fallback_used"] is True
    assert diagnostics["terminal_reason"] == (
        "qualified_support_preserved_after_fetch_error"
    )
    assert diagnostics["attempts"][-1]["fetch_error"]["type"] in {
        "TimeoutError", "invalid_payload", "invalid_timestamp"
    }


def test_camera_health_failure_does_not_use_qualified_support_fallback():
    camera_off = candidate_detection("frame-3")
    camera_off["camera_running"] = False
    vision = CandidateVisionAdapter(
        [], [candidate_detection("frame-1"), candidate_detection("frame-2"), camera_off]
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    assert confirmed is None
    assert status == "target_reconfirmation_failed"
    assert diagnostics["qualified_support_reached"] is True
    assert diagnostics["qualified_fallback_used"] is False
    assert diagnostics["terminal_reason"] == "camera_not_running"


def test_qualified_support_with_duplicates_until_window_expiry_confirms(monkeypatch):
    class RepeatingVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            if self.candidate_calls >= 2:
                self.candidate_calls += 1
                return dict(candidate_detection("frame-2"))
            return super().fetch_target_candidates(target)

    vision = RepeatingVision(
        [], [candidate_detection("frame-1"), candidate_detection("frame-2")]
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    clock = [0.0]
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        behavior_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    assert confirmed is not None
    assert status == "target_confirmed"
    assert diagnostics["qualified_fallback_used"] is False
    assert diagnostics["terminal_reason"] == "support_reached"


def _run_timed_cutoff_confirmation(monkeypatch, payloads, times):
    class TimedVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            index = self.candidate_calls
            self.candidate_calls += 1
            clock[0] = times[index]
            return dict(self.candidate_payloads.pop(0))

    clock = [0.0]
    vision = TimedVision([], payloads)
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        behavior_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = manager._confirm_target_candidates_with_status(
        "backpack",
        minimum_timestamp="2026-09-13T21:42:19.795554+00:00",
        return_diagnostics=True,
    )
    return result, vision


def test_pre_cutoff_frame_does_not_consume_fresh_frame_quota(monkeypatch):
    stale = candidate_detection("2026-09-13T21:42:19.745385+00:00")
    fresh_b = candidate_detection("2026-09-13T21:42:20.034769+00:00")
    fresh_c = candidate_detection("2026-09-13T21:42:20.309719+00:00")
    fresh_d = candidate_detection("2026-09-13T21:42:20.559719+00:00")
    (confirmed, status, diagnostics), vision = _run_timed_cutoff_confirmation(
        monkeypatch,
        [stale, fresh_b, fresh_c, fresh_d],
        [0.10, 0.30, 0.55, 0.78],
    )
    assert vision.candidate_calls == 4
    assert diagnostics["attempts"][0]["before_cutoff"] is True
    assert diagnostics["distinct_fresh_timestamps"] == 3
    assert confirmed is not None
    assert status == "target_confirmed"


def test_pre_cutoff_duplicates_do_not_consume_fresh_frame_quota(monkeypatch):
    stale = candidate_detection("2026-09-13T21:42:19.745385+00:00")
    fresh_b = candidate_detection("2026-09-13T21:42:20.034769+00:00")
    fresh_c = candidate_detection("2026-09-13T21:42:20.309719+00:00")
    fresh_d = candidate_detection("2026-09-13T21:42:20.559719+00:00")
    (confirmed, status, diagnostics), vision = _run_timed_cutoff_confirmation(
        monkeypatch,
        [stale, stale, fresh_b, fresh_c, fresh_d],
        [0.10, 0.15, 0.30, 0.50, 0.70],
    )
    assert vision.candidate_calls == 5
    assert any(attempt["duplicate_timestamp"] for attempt in diagnostics["attempts"])
    assert diagnostics["distinct_fresh_timestamps"] == 3
    assert confirmed is not None
    assert status == "target_confirmed"


def test_three_fresh_frame_quota_still_limits_distinct_frames(monkeypatch):
    payloads = [
        candidate_detection("2026-09-13T21:42:19.745385+00:00"),
        candidate_detection("2026-09-13T21:42:20.034769+00:00"),
        candidate_detection("2026-09-13T21:42:20.309719+00:00"),
        candidate_detection("2026-09-13T21:42:20.559719+00:00"),
        candidate_detection("2026-09-13T21:42:20.809719+00:00"),
    ]
    (_confirmed, _status, diagnostics), vision = _run_timed_cutoff_confirmation(
        monkeypatch,
        payloads,
        [0.10, 0.30, 0.50, 0.70, 0.85],
    )
    assert vision.candidate_calls == 4
    assert diagnostics["distinct_fresh_timestamps"] == 3


def test_pre_cutoff_duplicates_still_expire_confirmation_window(monkeypatch):
    stale = candidate_detection("2026-09-13T21:42:19.745385+00:00")

    class RepeatingStaleVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            return dict(stale)

    vision = RepeatingStaleVision([], [stale])
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    clock = [0.0]
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        behavior_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack",
        minimum_timestamp="2026-09-13T21:42:19.795554+00:00",
        return_diagnostics=True,
    )
    assert confirmed is None
    assert status == "target_lost"
    assert diagnostics["distinct_fresh_timestamps"] == 0
    assert diagnostics["elapsed_seconds"] >= 0.90


def test_empty_fresh_frames_do_not_consume_evidence_quota(monkeypatch):
    def empty(timestamp):
        payload = candidate_detection(timestamp)
        payload["detections"] = []
        return payload

    payloads = [
        candidate_detection("2026-09-13T21:42:19.745385+00:00"),
        empty("2026-09-13T21:42:19.895385+00:00"),
        empty("2026-09-13T21:42:20.045385+00:00"),
        empty("2026-09-13T21:42:20.195385+00:00"),
        candidate_detection("2026-09-13T21:42:20.345385+00:00"),
        candidate_detection("2026-09-13T21:42:20.495385+00:00"),
        candidate_detection("2026-09-13T21:42:20.645385+00:00"),
    ]
    (confirmed, status, diagnostics), vision = _run_timed_cutoff_confirmation(
        monkeypatch,
        payloads,
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    )
    assert vision.candidate_calls == 7
    assert confirmed is not None
    assert status == "target_confirmed"
    assert diagnostics["distinct_fresh_timestamps"] == 6
    assert diagnostics["actionable_frames"] == 3
    assert diagnostics["evidence_frames_evaluated"] == 3


def test_empty_fresh_stream_remains_bounded(monkeypatch):
    class EmptyVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            timestamp = (
                f"2026-09-13T21:42:20.{self.candidate_calls:06d}+00:00"
            )
            clock[0] += 0.10
            return {
                "timestamp": timestamp,
                "camera_running": True,
                "detections": [],
            }

    clock = [0.0]
    vision = EmptyVision([], [])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack",
        minimum_timestamp="2026-09-13T21:42:19.795554+00:00",
        return_diagnostics=True,
    )
    assert confirmed is None
    assert status == "target_lost"
    assert diagnostics["evidence_frames_evaluated"] == 0
    assert diagnostics["elapsed_seconds"] >= 0.90
    assert diagnostics["fetch_attempts"] < 20


def test_candidate_bearing_incompatible_frames_consume_evidence_quota(monkeypatch):
    def empty(timestamp):
        payload = candidate_detection(timestamp)
        payload["detections"] = []
        return payload

    payloads = [
        empty("2026-09-13T21:42:19.895385+00:00"),
        empty("2026-09-13T21:42:20.045385+00:00"),
        candidate_detection("2026-09-13T21:42:20.195385+00:00", bbox=(0, 0, 100, 100)),
        candidate_detection("2026-09-13T21:42:20.345385+00:00", bbox=(300, 0, 400, 100)),
        candidate_detection("2026-09-13T21:42:20.495385+00:00", bbox=(500, 0, 600, 100)),
    ]
    (confirmed, status, diagnostics), vision = _run_timed_cutoff_confirmation(
        monkeypatch,
        payloads,
        [0.10, 0.20, 0.30, 0.40, 0.50],
    )
    assert confirmed is None
    assert status == "target_reconfirmation_failed"
    assert vision.candidate_calls == 5
    assert diagnostics["actionable_frames"] == 3
    assert diagnostics["evidence_frames_evaluated"] == 3


def test_post_centering_empty_frames_allow_later_fresh_confirmation(monkeypatch):
    def empty(timestamp):
        payload = candidate_detection(timestamp)
        payload["detections"] = []
        return payload

    payloads = [
        candidate_detection("-1"),
        empty("1"),
        empty("2"),
        empty("3"),
        candidate_detection("4"),
        candidate_detection("5"),
        candidate_detection("6"),
    ]
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: {
        "ok": True, "permitted": True, "reason": "completed"
    }
    manager._promote_confirmed_target = lambda _candidate: located_target(320)
    manager._execute_find_object_approach = lambda *args, **kwargs: {
        "state": "CENTERED", "ok": True
    }
    clock = [0.0]
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    result = manager._center_acquired_target(
        "backpack",
        located_target(145),
        {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
         "turn_chunks_completed": 0},
    )
    assert result["state"] == "CENTERED"
    assert vision.candidate_calls == 7


def test_post_centering_empty_recovery_uses_extended_bounded_window(monkeypatch):
    class EmptyVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            clock[0] += 0.10
            return {
                "timestamp": f"empty-{self.candidate_calls}",
                "camera_running": True,
                "detections": [],
            }

    clock = [0.0]
    vision = EmptyVision([], [])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: {
        "ok": True, "permitted": True, "reason": "completed"
    }
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    result = manager._center_acquired_target(
        "backpack",
        located_target(145),
        {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
         "turn_chunks_completed": 0},
    )
    assert result["state"] == "TARGET_LOST_DURING_CENTERING"
    diagnostics = result["confirmation_diagnostics"]
    assert diagnostics["confirmation_window_seconds"] == 1.50
    assert diagnostics["elapsed_seconds"] >= 1.50
    assert diagnostics["evidence_frames_evaluated"] == 0


def test_post_forward_confirmation_uses_extended_window_and_returns_early():
    class ClockedVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            clock[0] += 0.20
            return dict(self.candidate_payloads.pop(0))

    clock = [0.0]
    payloads = []
    for index in range(3):
        empty = candidate_detection(f"empty-{index}")
        empty["detections"] = []
        payloads.append(empty)
    payloads.extend([
        candidate_detection("target-1"),
        candidate_detection("target-2"),
        candidate_detection("target-3"),
    ])
    vision = ClockedVision([located_target(320)], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager._promote_confirmed_target = lambda _candidate: located_target(320)
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    diagnostics = result["confirmation_diagnostics"]
    assert diagnostics["confirmation_window_seconds"] == 1.50
    assert diagnostics["elapsed_seconds"] < 1.50
    assert diagnostics["evidence_frames_evaluated"] == 3


def test_initial_preview_recovers_after_long_empty_detector_gap(monkeypatch):
    class ClockedVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            clock[0] += 0.10
            return super().fetch_target_candidates(target)

    clock = [0.0]
    payloads = []
    for index in range(10):
        empty = candidate_detection(f"initial-empty-{index}")
        empty["detections"] = []
        payloads.append(empty)
    payloads.extend(
        candidate_detection(f"initial-target-{index}")
        for index in range(3)
    )
    vision = ClockedVision([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])

    result = manager.preview_find_object("backpack")

    assert result["ok"] is True
    assert result["source"] == "vision_candidate"
    diagnostics = result["confirmation_diagnostics"]
    assert diagnostics["confirmation_window_seconds"] == 1.50
    assert diagnostics["fetch_attempts"] == 13
    assert diagnostics["elapsed_seconds"] < 1.50
    assert diagnostics["qualified_support_reached"] is True


def test_initial_preview_empty_stream_remains_bounded_at_extended_window(monkeypatch):
    class EmptyVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            clock[0] += 0.10
            return {
                "timestamp": f"initial-empty-{self.candidate_calls}",
                "camera_running": True,
                "detections": [],
            }

    clock = [0.0]
    vision = EmptyVision([], [])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])

    result = manager.preview_find_object("backpack")

    assert result["ok"] is False
    assert result["confirmation_diagnostics"]["confirmation_window_seconds"] == 1.50
    assert result["confirmation_diagnostics"]["elapsed_seconds"] >= 1.50
    assert manager.robot.calls == []


def test_initial_preview_fast_confirmation_returns_before_extended_bound(monkeypatch):
    class ClockedVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            clock[0] += 0.05
            return super().fetch_target_candidates(target)

    clock = [0.0]
    vision = ClockedVision([], centered_candidate_payloads("initial-fast"))
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])

    result = manager.preview_find_object("backpack")

    assert result["ok"] is True
    assert result["confirmation_diagnostics"]["confirmation_window_seconds"] == 1.50
    assert result["confirmation_diagnostics"]["elapsed_seconds"] < 1.50


def _run_confirmation_mode(monkeypatch, payloads, *, diagnostics, minimum=None,
                           manager_factory=None):
    vision = CandidateVisionAdapter([], payloads)
    manager = (
        manager_factory(vision)
        if manager_factory is not None
        else BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    )
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(behavior_module.time, "sleep", fake_sleep)
    result = manager._confirm_target_candidates_with_status(
        "backpack",
        minimum_timestamp=minimum,
        return_diagnostics=diagnostics,
    )
    return result, vision.candidate_calls, sleeps


def test_confirmation_diagnostics_on_off_are_behaviorally_equivalent(monkeypatch):
    scenarios = [
        (
            "compatible",
            [
                candidate_detection("frame-1"),
                candidate_detection("frame-2"),
                candidate_detection("frame-3"),
            ],
            None,
        ),
        (
            "cutoff_duplicate_one_support",
            [
                candidate_detection("frame-0"),
                candidate_detection("frame-1"),
                candidate_detection("frame-1"),
                candidate_detection("frame-1"),
            ],
            "frame-0",
        ),
        (
            "incompatible",
            [
                candidate_detection("frame-1", bbox=(0, 0, 100, 100)),
                candidate_detection("frame-2", bbox=(300, 300, 400, 400)),
                candidate_detection("frame-3", bbox=(500, 0, 600, 100)),
            ],
            None,
        ),
    ]
    for _name, payloads, minimum in scenarios:
        off, off_fetches, off_sleeps = _run_confirmation_mode(
            monkeypatch,
            payloads,
            diagnostics=False,
            minimum=minimum,
        )
        on, on_fetches, on_sleeps = _run_confirmation_mode(
            monkeypatch,
            payloads,
            diagnostics=True,
            minimum=minimum,
        )
        assert (off[0] is not None) == (on[0] is not None)
        assert off[1] == on[1]
        assert off_fetches == on_fetches
        assert off_sleeps == on_sleeps


def test_confirmation_diagnostics_on_off_equivalence_for_fetch_failure(monkeypatch):
    class FailingVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            if self.candidate_calls:
                raise TimeoutError("candidate endpoint unavailable")
            return super().fetch_target_candidates(target)

    def factory(vision):
        return FailingVision([], [candidate_detection("frame-1")])

    # Use equivalent failing adapters for each run while retaining the same
    # deterministic clock and observable call/sleep comparison.
    def run(diagnostics):
        vision = factory(None)
        manager = BehaviorManager(
            robot_client=GuardedSearchRobot(), vision_adapter=vision
        )
        clock = [0.0]
        sleeps = []
        monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(
            behavior_module.time,
            "sleep",
            lambda seconds: (sleeps.append(seconds), clock.__setitem__(0, clock[0] + seconds)),
        )
        return (
            manager._confirm_target_candidates_with_status(
                "backpack", return_diagnostics=diagnostics
            ),
            vision.candidate_calls,
            sleeps,
        )

    off, off_fetches, off_sleeps = run(False)
    on, on_fetches, on_sleeps = run(True)
    assert off[0] is None and on[0] is None
    assert off[1] == on[1] == "target_reconfirmation_failed"
    assert off_fetches == on_fetches == 1
    assert off_sleeps == on_sleeps == []


def test_confirmation_diagnostics_are_independent_of_track_id_and_entity_id():
    class MetadataVision(CandidateVisionAdapter):
        @staticmethod
        def normalize_detection(detection):
            normalized = CandidateVisionAdapter.normalize_detection(detection)
            normalized["track_id"] = detection.get("track_id")
            normalized["entity_id"] = detection.get("entity_id")
            return normalized

    payloads = [
        candidate_detection("frame-1"),
        candidate_detection("frame-2"),
        candidate_detection("frame-3"),
    ]
    payloads[0]["detections"][0].update(track_id=101, entity_id="backpack-001")
    payloads[1]["detections"][0].update(track_id=202, entity_id="backpack-002")
    vision = MetadataVision([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    assert confirmed is not None
    assert status == "target_confirmed"
    assert max(
        attempt["cluster_support_counts"]
        and max(attempt["cluster_support_counts"])
        for attempt in diagnostics["attempts"]
    ) >= 2


def test_confirmation_diagnostics_are_recursively_bounded_and_json_safe():
    vision = CandidateVisionAdapter(
        [],
        [candidate_detection("frame-1"), candidate_detection("frame-2")],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    _confirmed, _status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    json.dumps(diagnostics)
    forbidden = {"raw_detection", "image", "image_bytes", "frame_bytes"}
    summary_fields = {
        "label", "confidence", "center_x", "center_y", "area", "bbox"
    }

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            if "label" in value and "bbox" in value:
                assert set(value).issubset(summary_fields)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(diagnostics)
    assert len(diagnostics["attempts"]) == diagnostics["fetch_attempts"]
    assert diagnostics["fetch_attempts"] <= 3


@pytest.mark.parametrize(
    "first,second",
    [
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 205, "y1": 205, "x2": 305, "y2": 305}},
        ),
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
            {"label": "backpack", "cx": 290, "cy": 250, "area": 15000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 240, "y1": 200, "x2": 340, "y2": 350}},
        ),
        (
            {"label": "backpack", "cx": 220, "cy": 300, "area": 50000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 80, "y1": 220, "x2": 350, "y2": 450}},
            {"label": "backpack", "cx": 230, "cy": 275, "area": 17000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 150, "y1": 220, "x2": 315, "y2": 330}},
        ),
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
            {"label": "backpack", "cx": 350, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 300, "y1": 200, "x2": 400, "y2": 300}},
        ),
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
            {"label": "backpack", "cx": 250, "cy": 250, "area": 50000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 400, "y1": 400, "x2": 500, "y2": 500}},
        ),
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
            {"label": "suitcase", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
        ),
        (
            {"label": "backpack", "cx": 250, "cy": 250, "area": float("nan"),
             "image_width": 640, "image_height": 480, "bbox": None},
            {"label": "backpack", "cx": 250, "cy": 250, "area": 10000,
             "image_width": 640, "image_height": 480,
             "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}},
        ),
    ],
)
def test_match_details_preserves_association_decision(first, second):
    assert BehaviorManager._target_observations_match(first, second) == (
        BehaviorManager._target_observation_match_details(first, second)["matched"]
    )


def test_confirmation_diagnostics_terminal_reasons(monkeypatch):
    compatible = CandidateVisionAdapter(
        [], [
            candidate_detection("frame-1"),
            candidate_detection("frame-2"),
            candidate_detection("frame-3"),
        ]
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=compatible)
    _confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    assert status == "target_confirmed"
    assert diagnostics["terminal_reason"] == "support_reached"

    camera_off = candidate_detection("frame-1")
    camera_off["camera_running"] = False
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=CandidateVisionAdapter([], [camera_off]),
    )
    _confirmed, status, diagnostics = manager._confirm_target_candidates_with_status(
        "backpack", return_diagnostics=True
    )
    assert status == "target_lost"
    assert diagnostics["terminal_reason"] == "camera_not_running"


def test_malformed_same_label_candidates_are_target_lost():
    payloads = []
    for timestamp in ("bad-1", "bad-2", "bad-3"):
        payload = candidate_detection(timestamp)
        payload["detections"][0]["area"] = float("nan")
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_lost"


def test_two_consistent_distinct_frames_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", confidence=0.12),
            candidate_detection("frame-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["source_timestamp"] in {"frame-1", "frame-2"}


def test_bbox_shape_variation_matches_by_center_and_area():
    first = {
        "label": "backpack",
        "cx": 213.5,
        "cy": 368.0,
        "area": 40836.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 89.0, "y1": 286.0, "x2": 338.0, "y2": 450.0},
    }
    second = {
        "label": "backpack",
        "cx": 229.5,
        "cy": 293.5,
        "area": 23095.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 152.0, "y1": 219.0, "x2": 307.0, "y2": 368.0},
    }
    assert behavior_module.BehaviorManager._target_bbox_iou(first, second) < 0.50
    assert (
        behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
            first, second
        )
        >= 0.50
    )
    assert behavior_module.BehaviorManager._target_observations_match(first, second)


def test_nested_live_backpack_boxes_match_by_intersection_over_smaller():
    first = {
        "label": "backpack",
        "cx": 213.5,
        "cy": 337.5,
        "area": 59474.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 82.0, "y1": 224.0, "x2": 345.0, "y2": 451.0},
    }
    second = {
        "label": "backpack",
        "cx": 232.5,
        "cy": 277.5,
        "area": 17331.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 153.0, "y1": 223.0, "x2": 312.0, "y2": 332.0},
    }
    assert behavior_module.BehaviorManager._target_bbox_iou(first, second) < 0.50
    assert max(first["area"], second["area"]) / min(first["area"], second["area"]) > 2.0
    containment = behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
        first, second
    )
    assert containment >= 0.50
    assert behavior_module.BehaviorManager._target_observations_match(first, second)


def test_nested_association_rejects_nonoverlap_and_different_labels():
    base = {
        "label": "backpack",
        "cx": 200.0,
        "cy": 250.0,
        "area": 10000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 150.0, "y1": 200.0, "x2": 250.0, "y2": 300.0},
    }
    separated = dict(
        base,
        cx=205.0,
        cy=450.0,
        area=25000.0,
        bbox={"x1": 155.0, "y1": 400.0, "x2": 305.0, "y2": 566.67},
    )
    other_label = dict(separated, label="suitcase")
    assert behavior_module.BehaviorManager._target_bbox_intersection_over_smaller(
        base, separated
    ) < 0.50
    assert not behavior_module.BehaviorManager._target_observations_match(base, separated)
    assert not behavior_module.BehaviorManager._target_observations_match(base, other_label)


def test_target_association_rejects_area_or_center_outliers():
    base = {
        "label": "backpack",
        "cx": 200.0,
        "cy": 250.0,
        "area": 10000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 150.0, "y1": 200.0, "x2": 250.0, "y2": 300.0},
    }
    area_outlier = dict(
        base,
        area=21000.0,
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    center_outlier = dict(
        base,
        cx=261.0,
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    other_label = dict(
        base,
        label="suitcase",
        bbox={"x1": 400.0, "y1": 20.0, "x2": 500.0, "y2": 120.0},
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, area_outlier
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, center_outlier
    )
    assert not behavior_module.BehaviorManager._target_observations_match(
        base, other_label
    )


def test_target_reconfirmation_failure_is_distinct_from_target_loss():
    payload = candidate_detection("same-frame")
    vision = CandidateVisionAdapter([], [payload, payload, payload])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"

    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [payload, payload, payload],
    )
    result = manager.execute(_mission())
    assert result["state"] == "TARGET_RECONFIRMATION_FAILED"
    assert result["completed"] is False
    assert len(calls) == 1


def test_only_one_consistent_frame_does_not_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", bbox=(20, 20, 120, 120)),
            candidate_detection("frame-3", bbox=(350, 20, 450, 120)),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None
    assert manager._last_target_confirmation_status == "target_reconfirmation_failed"


def test_large_cluster_wins_using_all_frame_candidates():
    frame_two = candidate_detection(
        "frame-2",
        bbox=(235, 235, 310, 310),
        confidence=0.30,
    )
    frame_two["detections"].append(
        candidate_detection(
            "frame-2",
            bbox=(205, 165, 505, 465),
            confidence=0.10,
        )["detections"][0]
    )
    vision = CandidateVisionAdapter(
        [],
        [candidate_detection("frame-1"), frame_two, candidate_detection("frame-3")],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["area"] > 50000


def test_umbrella_never_substitutes_for_backpack():
    payloads = []
    for timestamp in ("frame-1", "frame-2", "frame-3"):
        payload = candidate_detection(timestamp, label="umbrella", confidence=0.99)
        payload["detections"].append(
            candidate_detection(timestamp, confidence=0.10)["detections"][0]
        )
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["label"] == "backpack"


def test_malformed_candidate_geometry_does_not_confirm():
    payloads = []
    for timestamp in ("frame-1", "frame-2", "frame-3"):
        payload = candidate_detection(timestamp)
        payload["detections"][0]["area"] = float("nan")
        payloads.append(payload)
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    assert manager._confirm_target_candidates("backpack") is None


def test_candidate_endpoint_failure_does_not_promote_or_acquire():
    vision = CandidateVisionAdapter([not_found()], [])
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: calls.append(args) or {
        "ok": True,
        "permitted": True,
    }
    result = manager.execute(_mission())
    assert result["target_found"] is False
    assert result["turn_chunks_attempted"] == 1
    assert vision.promotions == []
    assert len(calls) == 1


def test_two_consistent_small_candidates_can_confirm():
    vision = CandidateVisionAdapter(
        [],
        [
            candidate_detection("frame-1", bbox=(220, 220, 295, 295)),
            candidate_detection("frame-2", bbox=(221, 221, 296, 296)),
            candidate_detection("frame-3", bbox=(222, 222, 297, 297)),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert confirmed["area"] == 5625.0


def test_confirmed_candidate_is_promoted_without_turn():
    vision = CandidateVisionAdapter(
        [not_found(), found_target()],
        [
            candidate_detection("frame-1"),
            candidate_detection("frame-2", confidence=0.12),
            candidate_detection("frame-3", confidence=0.10),
            candidate_detection("post-1"),
            candidate_detection("post-2", confidence=0.12),
            candidate_detection("post-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: calls.append(args) or {
        "ok": True,
        "permitted": True,
    }
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 0
    assert len(vision.promotions) == 2
    assert calls == []


def test_confirmed_candidate_after_one_turn_stops_search():
    first_attempt = [
        candidate_detection("a-1"),
        candidate_detection("a-2", bbox=(20, 20, 120, 120)),
        candidate_detection("a-3", bbox=(350, 20, 450, 120)),
    ]
    second_attempt = [
        candidate_detection("b-1"),
        candidate_detection("b-2", confidence=0.11),
        candidate_detection("b-3", confidence=0.10),
    ]
    vision = CandidateVisionAdapter(
        [not_found(), not_found(), found_target()],
        first_attempt + second_attempt + [
            candidate_detection("post-1"),
            candidate_detection("post-2", confidence=0.11),
            candidate_detection("post-3", confidence=0.10),
        ],
    )
    manager = BehaviorManager(robot_client=GuardedSearchRobot(), vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    calls = []

    def turn(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ok": True, "permitted": True}

    manager.execute_guarded_turn = turn
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["target_found"] is True
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert len(calls) == 1


def located_target(cx, target="backpack"):
    result = found_target(target)
    result["cx"] = float(cx)
    result["bbox"] = {
        "x1": float(cx) - 60.0,
        "y1": 160.0,
        "x2": float(cx) + 60.0,
        "y2": 460.0,
    }
    return result


def centered_candidate_payloads(prefix, cx=320):
    half_width = 120
    bbox = (cx - half_width, 160, cx + half_width, 460)
    return [
        candidate_detection(f"{prefix}-1", bbox=bbox, confidence=0.12),
        candidate_detection(f"{prefix}-2", bbox=bbox, confidence=0.11),
        candidate_detection(f"{prefix}-3", bbox=bbox, confidence=0.10),
    ]


def make_centering_manager(observations, candidate_payloads, turn_results=None):
    vision = CandidateVisionAdapter(observations, candidate_payloads)
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=vision,
    )
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.lidar_session = "session-1"
    calls = []
    results = list(turn_results or [])

    def turn(direction, speed, duration, *, expected_lidar_session, now=None):
        calls.append((direction, speed, duration, expected_lidar_session))
        if results:
            return dict(results.pop(0))
        return {"ok": True, "permitted": True, "reason": "completed"}

    manager.execute_guarded_turn = turn
    return manager, vision, calls


def _multi_step_manager(candidate_payloads, *, initial_cx=320.0, interlock=None):
    vision = CandidateVisionAdapter(
        [located_target(initial_cx)],
        candidate_payloads,
    )
    robot = GuardedSearchRobot(interlock=interlock)
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 4
    return manager, robot, vision


def _move_calls(robot):
    return [call for call in robot.calls if call[0] == "move_forward"]


def test_four_step_approach_sequence_is_bounded_and_uses_fixed_forward_pulses():
    payloads = []
    for step in range(4):
        payloads.extend(centered_candidate_payloads(f"step-{step}"))
    manager, robot, vision = _multi_step_manager(payloads)

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 4
    assert result["approach_chunks_completed"] == 4
    assert result["maximum_approach_chunks"] == 4
    assert len(_move_calls(robot)) == 4
    assert all(call == ("move_forward", 0.08, 0.50) for call in _move_calls(robot))
    assert len(result["approach_steps"]) == 4
    assert all(
        step["post_motion_confirmation_diagnostics"][
            "confirmation_window_seconds"
        ] == 1.50
        for step in result["approach_steps"]
    )
    assert vision.candidate_calls == 12


def test_approach_recenters_between_forward_steps():
    payloads = []
    # Four post-forward confirmations plus one post-centering confirmation.
    for step in range(5):
        payloads.extend(centered_candidate_payloads(f"recenter-{step}"))
    manager, robot, vision = _multi_step_manager(payloads)
    promotions = [0]

    def promote(_candidate):
        promotions[0] += 1
        return located_target(145.0 if promotions[0] == 1 else 320.0)

    manager._promote_confirmed_target = promote
    turns = []
    manager.execute_guarded_turn = (
        lambda direction, speed, duration, **kwargs: turns.append(
            (direction, speed, duration)
        ) or {"ok": True, "permitted": True, "reason": "completed"}
    )

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert len(_move_calls(robot)) == 4
    assert turns == [("LEFT", 0.20, 0.50)]
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["centering_turn_chunks_completed"] == 1
    assert vision.candidate_calls == 15


class _SequenceInterlock(AlwaysPermittedInterlock):
    def __init__(self, permissions):
        super().__init__()
        self.permissions = list(permissions)

    def refresh(self):
        self.refresh_calls += 1
        permitted = self.permissions.pop(0) if self.permissions else False
        return permitted, "fresh_clear" if permitted else "front_not_clear"


def test_lidar_blocks_second_approach_step_without_retry():
    payloads = centered_candidate_payloads("first")
    payloads.extend(centered_candidate_payloads("unused"))
    interlock = _SequenceInterlock([True, False])
    manager, robot, _vision = _multi_step_manager(payloads, interlock=interlock)

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_BLOCKED"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert len(_move_calls(robot)) == 1
    assert interlock.refresh_calls == 2


def test_target_loss_after_second_step_stops_without_third_forward():
    payloads = centered_candidate_payloads("first")
    manager, robot, _vision = _multi_step_manager(payloads)

    result = manager.execute(_mission())

    assert result["state"] == "TARGET_LOST_AFTER_APPROACH"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 2
    assert result["approach_chunks_completed"] == 2
    assert len(_move_calls(robot)) == 2


def test_camera_off_after_a_forward_step_stops_without_retry():
    camera_off = candidate_detection("camera-off")
    camera_off["camera_running"] = False
    payloads = [camera_off]
    manager, robot, _vision = _multi_step_manager(payloads)

    result = manager.execute(_mission())

    assert result["state"] == "TARGET_LOST_AFTER_APPROACH"
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert len(_move_calls(robot)) == 1


def test_centering_failure_between_steps_prevents_next_forward():
    manager, robot, _vision = _multi_step_manager(
        centered_candidate_payloads("first")
    )
    manager._promote_confirmed_target = lambda _candidate: located_target(145)
    turns = []

    def deny_turn(direction, speed, duration, **kwargs):
        turns.append((direction, speed, duration))
        return {"ok": False, "permitted": False, "reason": "front_not_clear"}

    manager.execute_guarded_turn = deny_turn
    result = manager.execute(_mission())

    assert result["state"] == "CENTERING_BLOCKED"
    assert result["completed"] is False
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert len(_move_calls(robot)) == 1
    assert turns == [("LEFT", 0.20, 0.50)]


class _TwoResultRobot(GuardedSearchRobot):
    def __init__(self):
        super().__init__()
        self.forward_results = [
            {"ok": True, "automatic_stop": True},
            {"ok": False, "error": "transport_failed"},
        ]

    def move_forward(self, speed, seconds):
        self.calls.append(("move_forward", speed, seconds))
        return dict(self.forward_results.pop(0))


def test_forward_transport_failure_between_steps_is_terminal():
    payloads = centered_candidate_payloads("first")
    manager, _robot, _vision = _multi_step_manager(payloads)
    robot = _TwoResultRobot()
    manager.robot = robot

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_FAILED"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 2
    assert result["approach_chunks_completed"] == 1
    assert len(_move_calls(robot)) == 2


def test_detector_metadata_changes_between_steps_do_not_break_continuity():
    payloads = []
    for step in range(4):
        payloads.extend(centered_candidate_payloads(f"metadata-{step}"))
    manager, robot, _vision = _multi_step_manager(payloads)
    promotions = [0]

    def promote(_candidate):
        promotions[0] += 1
        target = located_target(320)
        target["track_id"] = 100 + promotions[0]
        target["entity_id"] = f"backpack-{promotions[0]:03d}"
        return target

    manager._promote_confirmed_target = promote
    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert result["ok"] is True
    assert len(_move_calls(robot)) == 4


def test_preemption_between_approach_steps_blocks_all_followup_actions():
    manager, robot, _vision = _multi_step_manager(
        centered_candidate_payloads("preempt-first")
    )
    authorized = [True]
    manager.execution_authorization_provider = lambda: authorized[0]

    def promote(_candidate):
        authorized[0] = False
        return located_target(145.0)

    manager._promote_confirmed_target = promote
    turns = []
    manager.execute_guarded_turn = (
        lambda *args, **kwargs: turns.append(args)
        or {"ok": True, "permitted": True}
    )

    result = manager.execute(_mission())

    assert result["state"] == "PREEMPTED"
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert len(_move_calls(robot)) == 1
    assert turns == []


def test_uncertain_forward_delivery_stops_multi_step_sequence_without_retry():
    robot = GuardedSearchRobot(
        move_result={
            "ok": True,
            "delivery_uncertain": True,
            "confirmed_forwarded": False,
            "forwarded": True,
        }
    )
    vision = CandidateVisionAdapter([located_target(320)], [])
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 4

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_FAILED"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 0
    assert len(_move_calls(robot)) == 1


def test_post_motion_cutoff_is_new_for_each_forward_step():
    payloads = []
    for step in range(4):
        payloads.extend(centered_candidate_payloads(f"z-step-{step}"))
    manager, robot, _vision = _multi_step_manager(payloads)
    cutoffs = []
    original = manager._confirm_target_candidates_with_status

    def record_cutoff(*args, **kwargs):
        cutoffs.append(kwargs.get("minimum_timestamp"))
        return original(*args, **kwargs)

    manager._confirm_target_candidates_with_status = record_cutoff
    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert len(_move_calls(robot)) == 4
    assert len(cutoffs) == 4
    assert len(set(cutoffs)) == 4
    assert all(cutoff is not None for cutoff in cutoffs)


def test_empty_frames_recover_between_multiple_approach_steps():
    payloads = []
    for step in range(4):
        empty = candidate_detection(f"empty-{step}")
        empty["detections"] = []
        payloads.append(empty)
        payloads.extend(centered_candidate_payloads(f"recover-{step}"))
    manager, robot, _vision = _multi_step_manager(payloads)

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_SEQUENCE_COMPLETE"
    assert result["approach_chunks_completed"] == 4
    assert len(_move_calls(robot)) == 4


def test_already_centered_target_completes_without_turn():
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: pytest.fail(
        "centered target must not turn"
    )
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["completed"] is True
    assert result["turn_chunks_attempted"] == 0
    assert result["centering_turn_chunks_attempted"] == 0
    assert result["steering_direction"] == "CENTERED"


class ApproachRefreshInterlock:
    def __init__(self, permitted=True, reason="fresh_clear"):
        self.permitted = permitted
        self.reason = reason
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1
        return self.permitted, self.reason


def test_approach_refreshes_existing_interlock_once_before_dispatch():
    interlock = ApproachRefreshInterlock()
    robot = GuardedSearchRobot(interlock=interlock)
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert interlock.refresh_calls == 1
    assert robot.calls == [("move_forward", 0.08, 0.50)]


def test_approach_interlock_denial_is_terminal_without_transport():
    interlock = ApproachRefreshInterlock(False, "front_not_clear")
    robot = GuardedSearchRobot(interlock=interlock)
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_BLOCKED"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 0
    assert robot.calls == []


@pytest.mark.parametrize(
    "move_result,state",
    [
        (
            {"ok": False, "bounded_forward_invalidated": True,
             "forwarded": True, "reason": "front_not_clear"},
            "APPROACH_BLOCKED",
        ),
        (
            {"ok": False, "error": "transport_failed"},
            "APPROACH_FAILED",
        ),
    ],
)
def test_approach_failure_is_terminal_and_never_replayed(move_result, state):
    robot = GuardedSearchRobot(move_result=move_result)
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    result = manager.execute(_mission())
    assert result["state"] == state
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 0
    assert len(robot.calls) == 1


def test_approach_transport_exception_is_terminal_without_replay():
    robot = GuardedSearchRobot(move_exception=TimeoutError("uncertain"))
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=SequencedVisionAdapter([located_target(320)]),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_FAILED"
    assert result["ok"] is False
    assert result["completed"] is True
    assert result["error_type"] == "TimeoutError"
    assert result["approach_result"]["transport_attempted"] is True
    assert len(robot.calls) == 1


def test_approach_requires_post_motion_candidate_timestamp():
    target = located_target(320)
    target["last_seen"] = "z-cutoff"
    payloads = [
        candidate_detection("a-before"),
        candidate_detection("b-before"),
        candidate_detection("c-before"),
    ]
    robot = GuardedSearchRobot()
    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=CandidateVisionAdapter([target], payloads),
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    result = manager.execute(_mission())
    assert result["state"] == "TARGET_LOST_AFTER_APPROACH"
    assert result["completed"] is True
    assert len(robot.calls) == 1


class IdentityChangingCandidateVision(CandidateVisionAdapter):
    def __init__(self, observations, candidate_payloads, post_entity_id=None):
        super().__init__(observations, candidate_payloads)
        self.post_entity_id = post_entity_id

    @staticmethod
    def normalize_detection(detection):
        normalized = CandidateVisionAdapter.normalize_detection(detection)
        if "track_id" in detection:
            normalized["track_id"] = detection["track_id"]
        return normalized

    def process_detection_frame(self, detections):
        result = super().process_detection_frame(detections)
        if self.post_entity_id is not None and self.last_result is not None:
            self.last_result["entity_id"] = self.post_entity_id
        return result


def _identity_change_payloads(track_ids):
    payloads = []
    for index, track_id in enumerate(track_ids, start=1):
        payload = candidate_detection(f"identity-frame-{index}")
        payload["detections"][0]["track_id"] = track_id
        payloads.append(payload)
    return payloads


def test_approach_allows_post_motion_track_id_change():
    pre_motion = located_target(320)
    pre_motion["track_id"] = 101
    vision = IdentityChangingCandidateVision(
        [pre_motion],
        _identity_change_payloads([202, 202, 303]),
    )
    robot = GuardedSearchRobot()
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    turn_calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: turn_calls.append(
        (args, kwargs)
    )

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert result["target_observation"]["track_id"] != 101
    assert robot.calls == [("move_forward", 0.08, 0.50)]
    assert turn_calls == []


def test_approach_allows_post_motion_entity_id_change():
    pre_motion = located_target(320)
    pre_motion["entity_id"] = "backpack-001"
    vision = IdentityChangingCandidateVision(
        [pre_motion],
        _identity_change_payloads([401, 402, 403]),
        post_entity_id="backpack-002",
    )
    robot = GuardedSearchRobot()
    manager = BehaviorManager(robot_client=robot, vision_adapter=vision)
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    turn_calls = []
    manager.execute_guarded_turn = lambda *args, **kwargs: turn_calls.append(
        (args, kwargs)
    )

    result = manager.execute(_mission())

    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["ok"] is True
    assert result["completed"] is True
    assert result["approach_chunks_attempted"] == 1
    assert result["approach_chunks_completed"] == 1
    assert result["target_observation"]["entity_id"] == "backpack-002"
    assert robot.calls == [("move_forward", 0.08, 0.50)]
    assert turn_calls == []


def test_centering_publishes_live_tracking_state_updates():
    manager, _vision, _calls = make_centering_manager(
        [located_target(145), located_target(320)],
        centered_candidate_payloads("telemetry")
        + centered_candidate_payloads("telemetry-post"),
    )
    tracking = empty_tracking_state()
    updates = []

    def publish(result):
        nonlocal tracking
        tracking = build_tracking_state(result, previous=tracking)
        updates.append(dict(tracking))

    manager.tracking_state_callback = publish
    result = manager.execute(_mission())

    centering = next(
        item for item in updates
        if item["state"] == "CENTERING"
    )
    assert centering["behavior"] == "FIND_OBJECT"
    assert centering["target_label"] == "backpack"
    assert centering["horizontal_error"] < -50
    assert centering["steering_direction"] == "LEFT"
    assert centering["target_area"] > 0
    assert centering["bbox"]["x1"] == 85.0
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert tracking["state"] == "APPROACH_STEP_COMPLETE"
    assert tracking["bbox"]["x1"] == 200.0
    assert tracking["steering_direction"] == "CENTER"
    assert abs(tracking["horizontal_error"]) <= 50
    assert tracking["locked_identity_id"] is None
    assert tracking["locked_entity_id"] is None


def test_runtime_tracking_callback_updates_status_state_without_identity():
    runtime = object.__new__(CognitiveRuntime)
    runtime._state_lock = threading.RLock()
    runtime._behavior_execution_generation = 7
    runtime._control_generation = 7
    runtime.tracking_state = empty_tracking_state()

    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_label": "backpack",
        "target_confidence": 0.15,
        "target_center_x": 145.0,
        "target_center_y": 240.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "image_center_x": 320.0,
        "horizontal_error": -175.0,
        "target_area": 42000.0,
        "steering_direction": "LEFT",
    })
    assert runtime.tracking_state["behavior"] == "FIND_OBJECT"
    assert runtime.tracking_state["state"] == "CENTERING"
    assert runtime.tracking_state["target_label"] == "backpack"
    assert runtime.tracking_state["horizontal_error"] == -175.0
    assert runtime.tracking_state["target_area"] > 0
    assert runtime.tracking_state["steering_direction"] == "LEFT"
    assert runtime.tracking_state["locked_identity_id"] is None

    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERED",
        "target_label": "backpack",
        "target_center_x": 320.0,
        "image_width": 640.0,
        "horizontal_error": 0.0,
        "steering_direction": "CENTERED",
    })
    assert runtime.tracking_state["state"] == "CENTERED"
    assert runtime.tracking_state["steering_direction"] == "CENTER"

    runtime._control_generation = 8
    runtime._publish_behavior_tracking({
        "behavior": "FIND_OBJECT",
        "state": "CENTERING",
        "target_label": "backpack",
        "horizontal_error": -200.0,
        "steering_direction": "LEFT",
    })
    assert runtime.tracking_state["state"] == "CENTERED"


def test_left_target_uses_guarded_centering_constants():
    manager, _vision, calls = make_centering_manager(
        [located_target(145), located_target(320)],
        centered_candidate_payloads("left")
        + centered_candidate_payloads("left-post"),
    )
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["centering_turn_chunks_completed"] == 1
    assert calls == [("LEFT", 0.20, 0.50, "session-1")]


def test_right_target_uses_right_guarded_centering_direction():
    manager, _vision, calls = make_centering_manager(
        [located_target(500), located_target(320)],
        centered_candidate_payloads("right")
        + centered_candidate_payloads("right-post"),
    )
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert calls[0] == ("RIGHT", 0.20, 0.50, "session-1")


def test_centering_requires_new_confirmation_after_each_turn():
    manager, vision, calls = make_centering_manager(
        [located_target(145), located_target(500), located_target(320)],
        centered_candidate_payloads("first", 500)
        + centered_candidate_payloads("second", 320)
        + centered_candidate_payloads("post", 320),
    )
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["centering_turn_chunks_attempted"] == 2
    assert result["centering_turn_chunks_completed"] == 2
    assert len(calls) == 2
    assert vision.candidate_calls == 9


def test_centering_stale_pre_turn_frame_cannot_confirm(monkeypatch):
    stale = candidate_detection("-1")

    class RepeatingStaleVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            return dict(stale)

    vision = RepeatingStaleVision([], [stale])
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: {
        "ok": True, "permitted": True, "reason": "completed"
    }
    clock = [0.0]
    monkeypatch.setattr(behavior_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        behavior_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = manager._center_acquired_target(
        "backpack",
        located_target(145),
        {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
         "turn_chunks_completed": 0},
    )
    assert result["state"] == "TARGET_LOST_DURING_CENTERING"
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["confirmation_status"] == "target_lost"
    diagnostics = result["confirmation_diagnostics"]
    assert diagnostics["distinct_fresh_timestamps"] == 0
    assert any(attempt["before_cutoff"] for attempt in diagnostics["attempts"])


def test_centering_uses_fresh_post_turn_candidates_not_stale(monkeypatch):
    payloads = [
        candidate_detection("-1"),
        candidate_detection("1"),
        candidate_detection("2"),
        candidate_detection("3"),
    ]
    vision = CandidateVisionAdapter([], payloads)
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: {
        "ok": True, "permitted": True, "reason": "completed"
    }
    selected = []
    manager._promote_confirmed_target = lambda candidate: (
        selected.append(candidate["source_timestamp"]) or located_target(320)
    )
    manager._execute_find_object_approach = lambda *args, **kwargs: {
        "state": "CENTERED", "ok": True
    }
    result = manager._center_acquired_target(
        "backpack",
        located_target(145),
        {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
         "turn_chunks_completed": 0},
    )
    assert result["state"] == "CENTERED"
    assert selected and selected[0] in {"1", "2", "3"}
    assert selected[0] != "-1"
    assert vision.candidate_calls == 4


def test_centering_post_turn_confirmation_propagates_diagnostics():
    class FailingVision(CandidateVisionAdapter):
        def fetch_target_candidates(self, target):
            self.candidate_calls += 1
            return dict(candidate_detection("-1"))

    vision = FailingVision([], [candidate_detection("-1")])
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=vision
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1
    manager.execute_guarded_turn = lambda *args, **kwargs: {
        "ok": True, "permitted": True, "reason": "completed"
    }
    clock = [0.0]
    original_monotonic = behavior_module.time.monotonic
    original_sleep = behavior_module.time.sleep
    behavior_module.time.monotonic = lambda: clock[0]
    behavior_module.time.sleep = lambda seconds: clock.__setitem__(
        0, clock[0] + seconds
    )
    try:
        result = manager._center_acquired_target(
            "backpack",
            located_target(145),
            {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
             "turn_chunks_completed": 0},
        )
    finally:
        behavior_module.time.monotonic = original_monotonic
        behavior_module.time.sleep = original_sleep
    assert result["state"] == "TARGET_LOST_DURING_CENTERING"
    assert result["confirmation_status"] == "target_lost"
    assert isinstance(result["confirmation_diagnostics"], dict)
    assert result["confirmation_diagnostics"]["attempts"]
    assert result["confirmation_diagnostics"]["minimum_timestamp"] == "0"


def test_centering_cutoff_is_created_after_turn_returns():
    target = located_target(145)
    target["last_seen"] = "2026-09-13T20:00:00+00:00"
    events = []
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(), vision_adapter=SequencedVisionAdapter([])
    )
    manager.lidar_session = "session-1"
    manager.FIND_APPROACH_MAX_CHUNKS = 1

    def turn(*args, **kwargs):
        events.append("turn_returned")
        return {"ok": True, "permitted": True, "reason": "completed"}

    def confirm(
        target_name,
        *,
        minimum_timestamp,
        return_diagnostics,
        confirmation_window_seconds,
    ):
        events.append("confirmation_called")
        assert events == ["turn_returned", "confirmation_called"]
        assert confirmation_window_seconds == 1.50
        assert behavior_module.BehaviorManager._vision_timestamp_is_iso(
            minimum_timestamp
        )
        return target, "target_confirmed", {
            "confirmation_status": "target_confirmed",
            "minimum_timestamp": minimum_timestamp,
            "attempts": [],
        }

    manager.execute_guarded_turn = turn
    manager._confirm_target_candidates_with_status = confirm
    manager._promote_confirmed_target = lambda _candidate: located_target(320)
    manager._execute_find_object_approach = lambda *args, **kwargs: {
        "state": "CENTERED", "ok": True
    }
    result = manager._center_acquired_target(
        "backpack", target,
        {"behavior": "FIND_OBJECT", "turn_chunks_attempted": 0,
         "turn_chunks_completed": 0},
    )
    assert result["state"] == "CENTERED"
    assert events == ["turn_returned", "confirmation_called"]


def test_target_loss_after_centering_turn_stops_without_second_turn():
    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [],
    )
    result = manager.execute(_mission())
    assert result["state"] == "TARGET_LOST_DURING_CENTERING"
    assert result["completed"] is False
    assert result["target_found"] is False
    assert result["centering_turn_chunks_attempted"] == 1
    assert len(calls) == 1


def test_guarded_centering_denial_stops_without_retry():
    manager, _vision, calls = make_centering_manager(
        [located_target(145)],
        [],
        [{"ok": False, "permitted": False, "reason": "turn_side_not_clear"}],
    )
    result = manager.execute(_mission())
    assert result["state"] == "CENTERING_BLOCKED"
    assert result["completed"] is False
    assert result["last_guarded_turn_result"]["reason"] == "turn_side_not_clear"
    assert len(calls) == 1


def test_centering_exhausts_at_eight_chunks():
    observations = [located_target(145)] + [located_target(145)] * 8
    payloads = []
    for index in range(8):
        payloads.extend(centered_candidate_payloads(f"exhaust-{index}", 145))
    manager, _vision, calls = make_centering_manager(observations, payloads)
    result = manager.execute(_mission())
    assert result["state"] == "CENTERING_EXHAUSTED"
    assert result["completed"] is False
    assert result["target_found"] is True
    assert result["centering_turn_chunks_attempted"] == 8
    assert result["centering_turn_chunks_completed"] == 8
    assert len(calls) == 8


def test_search_and_centering_counters_remain_separate():
    failed_search = [
        candidate_detection("search-1", bbox=(10, 10, 100, 100)),
        candidate_detection("search-2", bbox=(300, 10, 390, 100)),
        candidate_detection("search-3", bbox=(500, 10, 590, 100)),
    ]
    acquired = centered_candidate_payloads("acquired", 145)
    centered = centered_candidate_payloads("centered", 320)
    post_approach = centered_candidate_payloads("post-approach", 320)
    manager, _vision, calls = make_centering_manager(
        [not_found(), located_target(145), located_target(320)],
        failed_search + acquired + centered + post_approach,
    )
    result = manager.execute(_mission())
    assert result["state"] == "APPROACH_STEP_COMPLETE"
    assert result["turn_chunks_attempted"] == 1
    assert result["turn_chunks_completed"] == 1
    assert result["centering_turn_chunks_attempted"] == 1
    assert result["centering_turn_chunks_completed"] == 1
    assert len(calls) == 2


def test_invalid_center_geometry_does_not_start_centering():
    invalid = located_target(float("nan"))
    manager = BehaviorManager(
        robot_client=GuardedSearchRobot(),
        vision_adapter=SequencedVisionAdapter([invalid]),
    )
    result = manager.execute(_mission())
    assert result["centering_turn_chunks_attempted"] == 0


def detection(cx, area):
    return {
        "found": True,
        "target": "backpack",
        "label": "backpack",
        "confidence": 0.90,
        "cx": float(cx),
        "cy": 240.0,
        "area": float(area),
        "image_width": 640.0,
        "image_height": 480.0,
    }


def main():
    mission = create_mission(
        mission_type="FIND_OBJECT",
        target="backpack",
        speech="Find my backpack",
        status="ACTIVE",
    )

    robot = FakeRobotBridgeClient()

    vision = SequencedVisionAdapter(
        [
            detection(
                cx=180,
                area=35000,
            ),
            detection(
                cx=460,
                area=40000,
            ),
            detection(
                cx=320,
                area=50000,
            ),
            detection(
                cx=320,
                area=100000,
            ),
        ]
    )

    manager = BehaviorManager(
        robot_client=robot,
        vision_adapter=vision,
    )

    print(
        "===== CYCLE 1: IMAGE-LEFT TARGET ====="
    )

    left_result = manager.execute(
        mission
    )

    print(left_result)

    assert left_result["ok"] is True
    assert left_result["executed"] is True
    assert left_result["completed"] is False

    assert (
        left_result["state"]
        == "CENTERING_LEFT"
    )

    assert robot.calls[-1][0] == (
        "turn_left"
    )

    print()
    print(
        "===== CYCLE 2: IMAGE-RIGHT TARGET ====="
    )

    right_result = manager.execute(
        mission
    )

    print(right_result)

    assert right_result["ok"] is True
    assert right_result["executed"] is True
    assert right_result["completed"] is False

    assert (
        right_result["state"]
        == "CENTERING_RIGHT"
    )

    assert robot.calls[-1][0] == (
        "turn_right"
    )

    print()
    print(
        "===== CYCLE 3: CENTERED TARGET ====="
    )

    approach_result = manager.execute(
        mission
    )

    print(approach_result)

    assert approach_result["ok"] is True
    assert approach_result["executed"] is True

    assert (
        approach_result["completed"]
        is False
    )

    assert (
        approach_result["state"]
        == "APPROACHING"
    )

    assert robot.calls[-1][0] == (
        "move_forward"
    )

    print()
    print(
        "===== CYCLE 4: ARRIVAL TARGET ====="
    )

    arrived_result = manager.execute(
        mission
    )

    print(arrived_result)

    assert arrived_result["ok"] is True
    assert arrived_result["executed"] is True

    assert (
        arrived_result["completed"]
        is True
    )

    assert (
        arrived_result["state"]
        == "ARRIVED"
    )

    assert robot.calls[-1][0] == (
        "stop"
    )

    assert [
        call[0]
        for call in robot.calls
    ] == [
        "turn_left",
        "turn_right",
        "move_forward",
        "stop",
    ]

    assert vision.calls == 4
    assert vision.results == []

    print()
    print(
        "PASS: image-left target causes "
        "one left correction"
    )

    print(
        "PASS: image-right target causes "
        "one right correction"
    )

    print(
        "PASS: centered target causes "
        "one approach step"
    )

    print(
        "PASS: close target causes ARRIVED"
    )

    print(
        "PASS: intermediate steps keep "
        "the mission active"
    )

    print(
        "PASS: only ARRIVED completes "
        "the mission"
    )

    print()
    print(
        "All single-cycle FIND_OBJECT "
        "steering tests passed."
    )

    print(
        "No commands were sent to "
        "the physical robot."
    )


if __name__ == "__main__":
    main()
