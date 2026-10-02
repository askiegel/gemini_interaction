#!/usr/bin/env python3

import json
import inspect
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from behavior_manager import BehaviorManager
from marvin_arrival_policy import evaluate_marvin_visual_arrival
from marvin_pursuit_state import evaluate_marvin_pursuit_state
from mission_types import create_mission
from runtime_api import RuntimeAPIHandler
from tracking_state import build_tracking_state, empty_tracking_state
from vision_adapter import VisionAdapter


CONTINUITY_GENERATION = "vision-generation-a"
from voice_relay.server import FIND_OBJECT_PREVIEW_TIMEOUT_SECONDS, VoiceRelayHandler


def detection(timestamp="frame-1", cx=145.0):
    return {
        "timestamp": timestamp,
        "camera_running": True,
        "detections": [{
            "label": "backpack",
            "confidence": 0.12,
            "x1": cx - 60,
            "y1": 160,
            "x2": cx + 60,
            "y2": 460,
            "center_x": cx,
            "center_y": 310,
            "area": 36000,
            "image_width": 640,
            "image_height": 480,
        }],
    }


class ReadOnlyRobot:
    def __getattr__(self, name):
        if name in {"motion", "stop", "move_forward", "turn_left", "turn_right"}:
            raise AssertionError(f"preview called robot method {name}")
        raise AttributeError(name)


class UntouchedTargetLock:
    def __getattr__(self, name):
        raise AssertionError(f"preview accessed TargetLock.{name}")


class WorldModelObservation:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []
        self.writes = 0

    def find_latest_entity_by_label(self, label, **kwargs):
        self.calls.append((label, kwargs))
        return dict(self.observation)

    def update_robot_state(self, **kwargs):
        self.writes += 1
        raise AssertionError("preview wrote World Model")


class CandidateVision:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.process_calls = 0
        self.queries = []
        self.proposal_calls = 0

    def fetch_target_candidates(self, target):
        self.queries.append(target)
        return dict(self.payloads.pop(0))

    def fetch_detection_proposals(self):
        self.proposal_calls += 1
        return dict(self.payloads.pop(0))

    @staticmethod
    def normalize_detection(item):
        normalized = {
            "label": item["label"],
            "confidence": item["confidence"],
            "cx": item["center_x"],
            "cy": item["center_y"],
            "area": item["area"],
            "bbox": {
                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"],
            },
            "image_width": item["image_width"],
            "image_height": item["image_height"],
        }
        for key in (
            "track_id", "tracker_source", "marvin_continuity",
            "identity_ambiguous",
        ):
            if key in item:
                normalized[key] = item[key]
        return normalized

    def process_detection_frame(self, _detections):
        self.process_calls += 1
        raise AssertionError("preview promoted a candidate")


class MarvinSemanticVision:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.frame_index = 0
        self.identity_bboxes = []
        self.selection_candidates = []

    def fetch_frame(self):
        self.calls.append("frame")
        if self.error:
            raise self.error
        self.frame_index += 1
        return SimpleNamespace(
            data=b"frame", width=640, height=480,
            received_at=f"2026-09-19T12:00:0{self.frame_index}+00:00",
        )

    def describe_marvin(self, frame):
        raise AssertionError("preview must use identity-only confirmation")

    def confirm_marvin_identity(self, frame, bbox):
        assert frame is not None
        self.calls.append("confirm_marvin_identity")
        self.identity_bboxes.append(dict(bbox))
        if self.error:
            raise self.error
        return {
            "target": "marvin",
            "confirmed": self.result.get("found", True),
            "source": "gemini_marvin_identity",
        }

    def select_marvin_candidate(self, frame, candidates):
        self.calls.append("select_marvin_candidate")
        self.selection_candidates.append([dict(item) for item in candidates])
        if self.error:
            raise self.error
        return {
            "target": "marvin",
            "confirmed": self.result.get("found", True),
            "candidate_index": self.result.get("candidate_index", 0),
            "source": "gemini_marvin_candidate_selection",
        }


def no_marvin_candidates():
    return CandidateVision([{
        "timestamp": "frame-1", "camera_running": True, "detections": [],
    }])


def marvin_yolo_candidates():
    return CandidateVision([
        {
            **detection("frame-1", cx=450),
            "detections": [{
                "label": "chair", "confidence": 0.10,
                "x1": 400, "y1": 100, "x2": 500, "y2": 300,
                "center_x": 450, "center_y": 200, "area": 20000,
                "image_width": 640, "image_height": 480,
            }],
        },
        {
            "timestamp": "frame-2", "camera_running": True,
            "detections": [{
                "label": "toilet", "confidence": 0.11,
                "x1": 402, "y1": 102, "x2": 502, "y2": 302,
                "center_x": 452, "center_y": 202, "area": 20000,
                "image_width": 640, "image_height": 480,
            }],
        },
        {
            "timestamp": "frame-3", "camera_running": True,
            "detections": [{
                "label": "teddy bear", "confidence": 0.10,
                "x1": 403, "y1": 103, "x2": 503, "y2": 303,
                "center_x": 453, "center_y": 203, "area": 20000,
                "image_width": 640, "image_height": 480,
            }],
        },
    ])


def marvin_yolo_candidates_with_continuity(*continuities):
    payloads = []
    for index, continuity in enumerate(continuities, start=1):
        if isinstance(continuity, int):
            continuity = (CONTINUITY_GENERATION, continuity)
        generation, tracker_id = continuity
        for frame in range(3):
            payload = {
                "timestamp": f"continuity-{index}-{frame}",
                "camera_running": True,
                "detections": [{
                    "label": "teddy bear", "confidence": 0.11,
                    "x1": 402, "y1": 102, "x2": 502, "y2": 302,
                    "center_x": 452, "center_y": 202, "area": 20000,
                    "image_width": 640, "image_height": 480,
                    "marvin_continuity": {
                        "tracker_id": tracker_id,
                        "tracker_source": "marvin_continuity_botsort",
                        "tracker_generation": generation,
                    },
                }],
            }
            payloads.append(payload)
    return CandidateVision(payloads)


def proposal_frame(timestamp, *detections):
    return {
        "timestamp": timestamp,
        "camera_running": True,
        "detections": [
            {
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "area": (x2 - x1) * (y2 - y1),
                "image_width": 640,
                "image_height": 480,
            }
            for label, confidence, x1, y1, x2, y2 in detections
        ],
    }


class PreviewTracker:
    def __init__(self, _frame, seed_bbox, boxes=None):
        self.seed_bbox = dict(seed_bbox)
        self.boxes = list(boxes or [
            {"x1": 270, "y1": 110, "x2": 370, "y2": 330},
            {"x1": 275, "y1": 112, "x2": 375, "y2": 332},
        ])

    def update(self, _frame):
        return dict(self.boxes.pop(0)) if self.boxes else None


def use_preview_tracker(manager, boxes=None):
    manager.marvin_local_tracker_factory = lambda frame, bbox: PreviewTracker(
        frame, bbox, boxes,
    )


def marvin_result(**updates):
    return dict({
        "target": "marvin",
        "found": True,
        "coarse_direction": "RIGHT",
        "bbox": {"x1": 400, "y1": 100, "x2": 500, "y2": 300},
        "image_width": 640,
        "image_height": 480,
        "source": "gemini_marvin",
    }, **updates)


def test_world_model_preview_is_read_only_and_actionable():
    observation = {
        "found": True,
        "stale": False,
        "label": "backpack",
        "confidence": 0.8,
        "cx": 145.0,
        "cy": 310.0,
        "area": 36000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 85, "y1": 160, "x2": 205, "y2": 460},
    }
    world = WorldModelObservation(observation)
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), world_model=world)
    result = manager.preview_find_object("Backpack")
    assert result["ok"] is True
    assert result["source"] == "world_model"
    assert result["authoritative"] is True
    assert result["state"] == "PREVIEW"
    assert result["bbox"]["x1"] == 85
    assert result["horizontal_error"] == -175.0
    assert world.calls[0][1]["refresh"] is False
    assert world.writes == 0


def test_candidate_preview_confirms_without_promotion_or_motion():
    vision = CandidateVision([
        detection("one"),
        detection("two", cx=146),
        detection("three", cx=147),
    ])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    manager._promote_confirmed_target = lambda *_args: (_ for _ in ()).throw(
        AssertionError("preview promoted a candidate")
    )
    result = manager.preview_find_object("backpack")
    assert result["ok"] is True
    assert result["source"] == "vision_candidate"
    assert result["authoritative"] is False
    assert result["target_found"] is True
    assert result["bbox"] is not None
    assert vision.process_calls == 0
    assert manager._last_target_confirmation_status == "sentinel"


def test_preview_duplicate_frames_fail_closed_without_promotion():
    payload = detection("same")
    vision = CandidateVision([payload, payload, payload])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    result = manager.preview_find_object("backpack")
    assert result["ok"] is False
    assert result["preview"] is True
    assert "confirmed" in result["reason"]
    assert vision.process_calls == 0
    assert manager._last_target_confirmation_status == "sentinel"


def test_marvin_preview_uses_one_semantic_acquisition_after_yolo_fails():
    semantic = MarvinSemanticVision(marvin_result())
    world = WorldModelObservation({})
    vision = marvin_yolo_candidates()
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision, world_model=world,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)
    manager.execute = lambda *_args: (_ for _ in ()).throw(
        AssertionError("preview submitted a mission")
    )

    result = manager.preview_find_object("marvin")

    assert result["ok"] is True
    assert result["target"] == "marvin"
    assert result["source_timestamp"] == result["target_observation"]["source_timestamp"]
    assert result["vision_timestamp"] == result["source_timestamp"]
    assert result["target_found"] is True
    assert result["source"] == "marvin_local_tracker"
    assert result["authoritative"] is False
    assert result["bbox"] == {"x1": 275, "y1": 112, "x2": 375, "y2": 332}
    assert result["image_width"] == 640
    assert result["image_height"] == 480
    assert result["target_center_x"] == 325.0
    assert result["target_center_y"] == 222.0
    assert result["target"] == "marvin"
    assert result["detector_confidence"] == 0.11
    assert result["confirmation_diagnostics"]["confirmation_window_seconds"] == 2.0
    tracking = build_tracking_state(result)
    assert tracking["source"] == "marvin_local_tracker"
    assert tracking["bbox"] == result["bbox"]
    assert tracking["vision_timestamp"] == result["source_timestamp"]
    assert isinstance(tracking["detection_age_ms"], int)
    assert semantic.calls == ["frame", "select_marvin_candidate", "frame", "frame"]
    assert semantic.selection_candidates[0][0]["proposal_label"] == "toilet"
    assert result["proposal_label"] == "toilet"
    assert result["geometry_source"] == "yolo_proposal"
    assert result["identity_source"] == "gemini_marvin_candidate_selection"
    assert result["yolo_seed_bbox"] == {"x1": 402, "y1": 102, "x2": 502, "y2": 302}
    assert result["tracker_seed_bbox"] == {"x1": 382, "y1": 92, "x2": 522, "y2": 312}
    assert result["tracker_seed_source"] == "bounded_yolo_proposal_expansion"
    assert vision.proposal_calls == 3
    assert vision.queries == []
    assert vision.process_calls == 0
    assert world.writes == 0


def test_marvin_person_only_proposal_fails_before_semantics_or_tracker():
    vision = CandidateVision([
        proposal_frame(
            f"person-{index}",
            ("person", 0.127, 378, 30, 617, 401),
        )
        for index in range(1, 4)
    ])
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    manager.marvin_local_tracker_factory = lambda *_args: (_ for _ in ()).throw(
        AssertionError("person proposal initialized Marvin tracker")
    )

    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert result["target_found"] is False
    assert "marvin_person_proposal_rejected" in result["reason"]
    assert result.get("identity_confirmed") is not True
    assert result.get("confirmation_status") != "target_confirmed"
    assert semantic.selection_candidates == []
    assert semantic.calls == []
    assert vision.proposal_calls == 3
    pursuit = evaluate_marvin_pursuit_state(
        result, None, None, now="2026-09-19T12:00:03+00:00",
    )
    assert pursuit["pursuit_authorized"] is False


def test_three_frame_person_support_cannot_confirm_marvin():
    vision = CandidateVision([
        proposal_frame(
            f"person-support-{index}",
            ("person", 0.9, 200 + index, 80, 350 + index, 400),
        )
        for index in range(1, 4)
    ])
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0

    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert "marvin_person_proposal_rejected" in result["reason"]
    assert semantic.selection_candidates == []
    assert vision.proposal_calls == 3


def test_marvin_filters_person_before_selecting_non_person_candidate():
    frames = []
    for index in range(1, 4):
        frames.append(proposal_frame(
            f"mixed-{index}",
            ("person", 0.99, 80, 50, 240, 420),
            ("teddy bear", 0.2, 390 + index, 100, 490 + index, 300),
        ))
    vision = CandidateVision(frames)
    semantic = MarvinSemanticVision(marvin_result(candidate_index=0))
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    result = manager.preview_find_object("marvin")

    assert result["ok"] is True
    assert [item["proposal_label"] for item in semantic.selection_candidates[0]] == [
        "teddy bear",
    ]
    assert result["proposal_label"] == "teddy bear"
    assert result["identity_confirmed"] is True


def test_person_cannot_add_support_to_overlapping_non_person_cluster():
    vision = CandidateVision([
        proposal_frame(
            f"overlap-{index}",
            ("person", 0.9, 200, 100, 400, 350),
            *(
                [("teddy bear", 0.2, 205, 105, 395, 345)]
                if index == 1 else []
            ),
        )
        for index in range(1, 4)
    ])
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0

    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert result.get("identity_confirmed") is not True
    assert semantic.selection_candidates == []
    assert vision.proposal_calls == 3


def test_generic_find_object_person_world_model_path_is_unchanged():
    observation = {
        "found": True,
        "stale": False,
        "label": "person",
        "confidence": 0.8,
        "cx": 320.0,
        "cy": 240.0,
        "area": 20000.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "bbox": {"x1": 220, "y1": 140, "x2": 420, "y2": 340},
    }
    world = WorldModelObservation(observation)
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), world_model=world)

    result = manager.preview_find_object("person")

    assert result["ok"] is True
    assert result["target_label"] == "person"
    assert result["source"] == "world_model"


def test_marvin_preview_preserves_existing_proposal_tracker_metadata():
    vision = marvin_yolo_candidates()
    for payload in vision.payloads:
        payload["detections"][0].update(
            track_id=77,
            tracker_source="botsort_reid",
        )
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=MarvinSemanticVision(marvin_result()),
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    result = manager.preview_find_object("marvin")

    assert result["track_id"] == 77
    assert result["tracker_source"] == "botsort_reid"
    assert result["target_observation"]["track_id"] == 77
    assert result["target_observation"]["tracker_source"] == "botsort_reid"
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["track_id"] == 77
    assert payload["tracker_source"] == "botsort_reid"


def test_vision_adapter_normalizes_only_complete_marvin_continuity_metadata():
    adapter = VisionAdapter.__new__(VisionAdapter)
    adapter.last_payload = {}
    detection = {
        "label": "person", "confidence": 0.1,
        "x1": 10, "y1": 20, "x2": 40, "y2": 100,
        "marvin_continuity": {
            "tracker_id": 16,
            "tracker_source": "marvin_continuity_botsort",
            "tracker_generation": CONTINUITY_GENERATION,
        },
    }

    normalized = adapter.normalize_detection(detection)

    assert normalized["marvin_continuity"] == detection["marvin_continuity"]
    assert normalized["identity_id"] is None
    detection["marvin_continuity"] = {"tracker_id": 16}
    assert "marvin_continuity" not in adapter.normalize_detection(detection)


def test_marvin_preview_preserves_provider_continuity_metadata_without_identity_promotion():
    vision = marvin_yolo_candidates()
    for payload in vision.payloads:
        payload["detections"][0]["marvin_continuity"] = {
            "tracker_id": 16,
            "tracker_source": "marvin_continuity_botsort",
            "tracker_generation": CONTINUITY_GENERATION,
        }
    world = WorldModelObservation({})
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=MarvinSemanticVision(marvin_result()), world_model=world,
    )
    manager.target_lock = UntouchedTargetLock()
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    result = manager.preview_find_object("marvin")
    expected = {
        "tracker_id": 16,
        "tracker_source": "marvin_continuity_botsort",
        "tracker_generation": CONTINUITY_GENERATION,
    }
    assert result["marvin_continuity"] == expected
    assert result["target_observation"]["marvin_continuity"] == expected
    assert "identity_id" not in result
    assert world.writes == 0
    assert world.calls == []
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["marvin_continuity"] == expected


def test_marvin_preview_reuses_semantic_authority_for_stable_continuity():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(16, 16),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    first = manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert first["identity_source"] == "gemini_marvin_candidate_selection"
    assert second["identity_source"] == "marvin_session_continuity"
    assert second["identity_confirmed"] is True
    assert second["marvin_continuity"] == {
        "tracker_id": 16,
        "tracker_source": "marvin_continuity_botsort",
        "tracker_generation": CONTINUITY_GENERATION,
    }
    assert semantic.calls == [
        "frame", "select_marvin_candidate", "frame", "frame",
        "frame", "frame", "frame",
    ]


def test_marvin_preview_lost_continuity_falls_back_to_one_gemini_selection():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(16, 17),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    first = manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert first["identity_source"] == "gemini_marvin_candidate_selection"
    assert second["identity_source"] == "gemini_marvin_candidate_selection"
    assert semantic.calls.count("select_marvin_candidate") == 2
    assert manager._marvin_preview_continuity == {
        "tracker_id": 17,
        "tracker_source": "marvin_continuity_botsort",
        "tracker_generation": CONTINUITY_GENERATION,
    }


def test_marvin_preview_generation_change_forces_gemini_reacquisition():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(
            ("generation-a", 1), ("generation-b", 1),
        ),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["identity_source"] == "gemini_marvin_candidate_selection"
    assert semantic.calls.count("select_marvin_candidate") == 2
    assert manager._marvin_preview_continuity == {
        "tracker_generation": "generation-b",
        "tracker_source": "marvin_continuity_botsort",
        "tracker_id": 1,
    }


def test_marvin_preview_source_change_forces_gemini_reacquisition():
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates_with_continuity(16, 16)
    for payload in vision.payloads[3:]:
        payload["detections"][0]["marvin_continuity"]["tracker_source"] = "other_tracker"
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["identity_source"] == "gemini_marvin_candidate_selection"
    assert semantic.calls.count("select_marvin_candidate") == 2


@pytest.mark.parametrize("generation", [None, "", 7])
def test_marvin_preview_missing_or_malformed_generation_cannot_reuse(generation):
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates_with_continuity(16, 16)
    for payload in vision.payloads[3:]:
        payload["detections"][0]["marvin_continuity"]["tracker_generation"] = generation
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["identity_source"] == "gemini_marvin_candidate_selection"
    assert semantic.calls.count("select_marvin_candidate") == 2


def test_marvin_preview_stable_continuity_uses_new_tracker_geometry():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(16, 16),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    box_sets = iter((
        [{"x1": 270, "y1": 110, "x2": 370, "y2": 330}] * 2,
        [{"x1": 350, "y1": 110, "x2": 450, "y2": 330}] * 2,
    ))
    manager.marvin_local_tracker_factory = (
        lambda frame, bbox: PreviewTracker(frame, bbox, next(box_sets))
    )

    first = manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["identity_source"] == "marvin_session_continuity"
    assert first["bbox"] == {"x1": 270, "y1": 110, "x2": 370, "y2": 330}
    assert second["bbox"] == {"x1": 350, "y1": 110, "x2": 450, "y2": 330}
    assert second["horizontal_error"] == 80.0


def test_marvin_preview_cached_continuity_ambiguity_forces_gemini():
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates_with_continuity(16, 16)
    for payload in vision.payloads[3:]:
        payload["detections"][0]["identity_ambiguous"] = True
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["identity_source"] == "gemini_marvin_candidate_selection"
    assert semantic.calls.count("select_marvin_candidate") == 2


def test_marvin_preview_cached_continuity_malformed_proposal_fails_closed():
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates_with_continuity(16)
    vision.payloads.extend([{
        "timestamp": f"bad-{index}", "camera_running": True,
        "detections": [{
            "label": "teddy bear", "confidence": 0.11,
            "x1": 500, "y1": 100, "x2": 400, "y2": 300,
            "image_width": 640, "image_height": 480,
            "marvin_continuity": {
                "tracker_id": 16,
                "tracker_source": "marvin_continuity_botsort",
                "tracker_generation": CONTINUITY_GENERATION,
            },
        }],
    } for index in range(3)])
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["ok"] is False
    assert second["target_found"] is False
    assert "bbox" not in second or second["bbox"] is None
    assert semantic.calls.count("select_marvin_candidate") == 1


def test_marvin_preview_cached_continuity_stale_proposal_fails_closed():
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates_with_continuity(16)
    vision.payloads.extend([vision.payloads[-1]] * 3)
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["ok"] is False
    assert second["target_found"] is False
    assert semantic.calls.count("select_marvin_candidate") == 1


def test_marvin_preview_fast_path_tracker_failure_has_no_result():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(16, 16),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    factories = iter((
        lambda frame, bbox: PreviewTracker(frame, bbox),
        lambda _frame, _bbox: type("LostTracker", (), {"update": lambda self, frame: None})(),
        lambda _frame, _bbox: type("LostTracker", (), {"update": lambda self, frame: None})(),
    ))
    manager.marvin_local_tracker_factory = lambda frame, bbox: next(factories)(frame, bbox)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")

    assert second["ok"] is False
    assert second["target_found"] is False
    assert manager._marvin_preview_continuity is None


def test_marvin_preview_fast_path_remains_usable_by_arrival_policy():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(),
        vision_adapter=marvin_yolo_candidates_with_continuity(16, 16),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    manager.preview_find_object("marvin")
    second = manager.preview_find_object("marvin")
    arrival = evaluate_marvin_visual_arrival(
        second, now=second["source_timestamp"],
    )

    assert second["identity_source"] == "marvin_session_continuity"
    assert arrival["visual_session_authorized"] is True
    assert arrival["geometry_valid"] is True


def test_marvin_preview_omits_missing_or_malformed_continuity_metadata():
    vision = marvin_yolo_candidates()
    for payload in vision.payloads:
        payload["detections"][0]["marvin_continuity"] = {
            "tracker_id": "not-a-real-tracker-id",
            "tracker_source": "marvin_continuity_botsort",
        }
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=MarvinSemanticVision(marvin_result()),
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    result = manager.preview_find_object("marvin")

    assert "marvin_continuity" not in result
    assert "marvin_continuity" not in result["target_observation"]


def test_marvin_preview_never_fabricates_missing_proposal_tracker_metadata():
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
        semantic_vision=MarvinSemanticVision(marvin_result()),
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0
    use_preview_tracker(manager)

    result = manager.preview_find_object("marvin")

    assert "track_id" not in result
    assert "tracker_source" not in result
    assert "track_id" not in result["target_observation"]
    assert "tracker_source" not in result["target_observation"]


def test_marvin_preview_requires_two_fresh_tracker_frames_and_uses_tracker_geometry():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
        semantic_vision=semantic,
    )
    seeds = []
    updates = []

    class Tracker:
        def __init__(self, _frame, bbox):
            seeds.append(dict(bbox))

        def update(self, _frame):
            updates.append(True)
            return {"x1": 270, "y1": 110, "x2": 370, "y2": 330}

    manager.marvin_local_tracker_factory = Tracker
    result = manager.preview_find_object("marvin")

    assert result["source"] == "marvin_local_tracker"
    assert seeds == [{"x1": 382, "y1": 92, "x2": 522, "y2": 312}]
    assert len(updates) == 2
    assert result["bbox"] == {"x1": 270, "y1": 110, "x2": 370, "y2": 330}
    assert result["horizontal_error_pixels"] == 0.0


def test_marvin_tracker_seed_expansion_is_rounded_and_clamped():
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
    )
    assert manager._expand_marvin_tracker_seed_bbox(
        {"x1": 381, "y1": 80, "x2": 533, "y2": 349}, 640, 480,
    ) == {"x1": 351, "y1": 67, "x2": 563, "y2": 362}
    assert manager._expand_marvin_tracker_seed_bbox(
        {"x1": 0, "y1": 0, "x2": 10, "y2": 20}, 20, 30,
    ) == {"x1": 0, "y1": 0, "x2": 12, "y2": 21}
    assert manager._expand_marvin_tracker_seed_bbox(
        {"x1": 10, "y1": 10, "x2": 20, "y2": 30}, 20, 30,
    ) == {"x1": 8, "y1": 9, "x2": 20, "y2": 30}
    with pytest.raises(ValueError):
        manager._expand_marvin_tracker_seed_bbox(
            {"x1": 4, "y1": 4, "x2": 4, "y2": 8}, 20, 20,
        )
    with pytest.raises(ValueError):
        manager._expand_marvin_tracker_seed_bbox(
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, 0, 20,
        )


@pytest.mark.parametrize("label", ["chair", "bed", "refrigerator", "teddy bear", "toilet"])
def test_marvin_proposal_geometry_filter_ignores_label(label):
    diagnostics = {}
    candidate = {
        "label": label,
        "bbox": {"x1": 10, "y1": 20, "x2": 110, "y2": 100},
    }
    assert BehaviorManager._filter_marvin_proposal_geometry(
        [candidate], diagnostics,
    ) == [candidate]
    assert diagnostics["marvin_geometry_candidates_after"] == 1


@pytest.mark.parametrize("bbox,accepted", [
    ({"x1": 303, "y1": 129, "x2": 398, "y2": 302}, True),
    ({"x1": 381, "y1": 80, "x2": 533, "y2": 349}, True),
    ({"x1": 13, "y1": 140, "x2": 446, "y2": 303}, False),
    ({"x1": 2, "y1": 167, "x2": 456, "y2": 300}, False),
])
def test_marvin_proposal_geometry_filter_matches_live_bbox_examples(bbox, accepted):
    candidate = {"label": "chair", "bbox": bbox}
    assert bool(BehaviorManager._filter_marvin_proposal_geometry([candidate])) is accepted


def test_marvin_proposal_geometry_filter_preserves_order_and_ratio_boundary():
    candidates = [
        {"label": "chair", "bbox": {"x1": 0, "y1": 0, "x2": 125, "y2": 100}},
        {"label": "toilet", "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 50}},
        {"label": "bed", "bbox": {"x1": 0, "y1": 0, "x2": 126, "y2": 100}},
    ]
    diagnostics = {}
    filtered = BehaviorManager._filter_marvin_proposal_geometry(candidates, diagnostics)
    assert [item["label"] for item in filtered] == ["chair", "toilet"]
    assert diagnostics["marvin_geometry_candidates_before"] == 3
    assert diagnostics["marvin_geometry_candidates_after"] == 2
    assert diagnostics["marvin_geometry_candidates_rejected"] == 1


def test_marvin_geometry_filter_rejects_known_wide_proposals_before_gemini():
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
    )
    wide = [
        {"label": "chair", "bbox": {"x1": 13, "y1": 140, "x2": 446, "y2": 303}},
        {"label": "toilet", "bbox": {"x1": 2, "y1": 167, "x2": 456, "y2": 300}},
    ]
    diagnostics = {}
    assert manager._filter_marvin_proposal_geometry(wide, diagnostics) == []
    assert diagnostics["marvin_geometry_candidates_rejected"] == 2


def test_marvin_preview_fails_closed_without_gemini_when_all_proposals_are_wide():
    semantic = MarvinSemanticVision(marvin_result())
    vision = marvin_yolo_candidates()
    vision.payloads = [
        {
            "timestamp": f"frame-{index}",
            "camera_running": True,
            "detections": [{
                "label": "chair", "confidence": 0.1,
                "x1": 13, "y1": 140, "x2": 446, "y2": 303,
                "center_x": 229.5, "center_y": 221.5, "area": 70579,
                "image_width": 640, "image_height": 480,
            }],
        }
        for index in range(1, 4)
    ]
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=vision,
        semantic_vision=semantic,
    )
    result = manager.preview_find_object("marvin")
    assert result["ok"] is False
    assert "geometry_invalid" in result["reason"]
    assert semantic.calls == []


def test_marvin_preview_tracker_confirmation_failure_has_no_authoritative_bbox():
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
        semantic_vision=semantic,
    )

    class LostTracker:
        def __init__(self, _frame, _bbox):
            pass

        def update(self, _frame):
            return None

    manager.marvin_local_tracker_factory = LostTracker
    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert result["target_found"] is False
    assert "tracker" in result["reason"]
    assert "bbox" not in result or result["bbox"] is None


def test_marvin_semantic_preview_absent_fails_closed_without_robot_action():
    semantic = MarvinSemanticVision(marvin_result(
        found=False, coarse_direction="UNKNOWN", bbox=None,
    ))
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0

    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert result["target_found"] is False
    assert "identity" in result["reason"]
    assert semantic.calls == ["frame", "select_marvin_candidate"]


def test_marvin_semantic_preview_failure_fails_closed():
    semantic = MarvinSemanticVision(error=TimeoutError("offline timeout"))
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=marvin_yolo_candidates(),
        semantic_vision=semantic,
    )
    manager.TARGET_CONFIRMATION_POLL_SECONDS = 0

    result = manager.preview_find_object("marvin")

    assert result["ok"] is False
    assert result["target_found"] is False
    assert "unavailable" in result["reason"]
    assert semantic.calls == ["frame"]


def test_marvin_semantic_preview_invalid_or_missing_geometry_fails_closed():
    malformed = CandidateVision([
        {"timestamp": "frame-1", "camera_running": True, "detections": [{
            "label": "teddy bear", "confidence": 0.9,
            "x1": 500, "y1": 100, "x2": 400, "y2": 300,
            "center_x": 450, "center_y": 200, "area": 20000,
            "image_width": 640, "image_height": 480,
        }]},
    ])
    semantic = MarvinSemanticVision(marvin_result())
    manager = BehaviorManager(
        robot_client=ReadOnlyRobot(), vision_adapter=malformed,
        semantic_vision=semantic,
    )
    result = manager.preview_find_object("marvin")
    assert result["ok"] is False
    assert result["target_found"] is False


def test_production_confirmation_wrapper_updates_shared_status():
    vision = CandidateVision([
        detection("one"),
        detection("two", cx=146),
        detection("three", cx=147),
    ])
    manager = BehaviorManager(robot_client=ReadOnlyRobot(), vision_adapter=vision)
    manager._last_target_confirmation_status = "sentinel"
    confirmed = manager._confirm_target_candidates("backpack")
    assert confirmed is not None
    assert manager._last_target_confirmation_status == "target_confirmed"


def test_preview_result_builds_tracking_without_runtime_mutation():
    result = {
        "behavior": "FIND_OBJECT",
        "state": "PREVIEW",
        "target": "backpack",
        "target_found": True,
        "target_observation": {
            "label": "backpack",
            "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            "cx": 55,
            "cy": 70,
            "area": 9000,
            "image_width": 640,
            "image_height": 480,
        },
        "target_label": "backpack",
        "target_center_x": 55,
        "target_center_y": 70,
        "target_area": 9000,
        "image_width": 640,
        "image_height": 480,
        "horizontal_error": -265,
        "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
    }
    tracking = build_tracking_state(result, previous=empty_tracking_state())
    assert tracking["state"] == "PREVIEW"
    assert tracking["bbox"]["x2"] == 100.0


class CapturingHandler(RuntimeAPIHandler):
    def __init__(self, runtime):
        self.path = ""
        self.server = SimpleNamespace(runtime=runtime)
        self.responses = []

    def send_json(self, status_code, payload):
        self.responses.append((status_code, payload))


def test_runtime_preview_endpoint_does_not_mutate_runtime_or_execute():
    before_tracking = {"state": "IDLE"}
    calls = []

    class FakeBehavior:
        def preview_find_object(self, target):
            calls.append(target)
            return {
                "ok": True,
                "preview": True,
                "authoritative": False,
                "source": "vision_candidate",
                "target": target,
                "state": "PREVIEW",
                "behavior": "FIND_OBJECT",
                "target_found": True,
                "target_label": target,
                "target_center_x": 320,
                "image_width": 640,
                "target_observation": {},
            }

        def execute(self, _mission):
            raise AssertionError("preview invoked execute")

    class Runtime:
        behavior_manager = FakeBehavior()
        tracking_state = dict(before_tracking)
        active_mission = None
        queue = []
        _control_generation = 0

    before_active_mission = Runtime.active_mission
    before_queue = list(Runtime.queue)

    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-object/preview?target=backpack"
    handler.server = SimpleNamespace(runtime=Runtime())
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_GET()
    assert responses[0][0] == 200
    assert responses[0][1]["preview"] is True
    assert calls == ["backpack"]
    assert handler.server.runtime.tracking_state == before_tracking
    assert handler.server.runtime.active_mission is before_active_mission
    assert handler.server.runtime.queue == before_queue
    assert handler.server.runtime._control_generation == 0


def _call_runtime_preview(result):
    class FakeBehavior:
        def preview_find_object(self, _target):
            return result

    class Runtime:
        behavior_manager = FakeBehavior()

    handler = object.__new__(RuntimeAPIHandler)
    handler.path = "/find-object/preview?target=backpack"
    handler.server = SimpleNamespace(runtime=Runtime())
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    handler.do_GET()
    return responses[0]


def test_runtime_preview_serializes_candidate_confirmation_diagnostics():
    diagnostics = {
        "confirmation_status": "target_confirmed",
        "fetch_attempts": 2,
        "attempts": [{"response_timestamp": "frame-1"}],
    }
    result = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "source": "vision_candidate",
        "target": "backpack",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": True,
        "target_label": "backpack",
        "target_center_x": 320,
        "target_center_y": 300,
        "target_area": 12000,
        "image_width": 640,
        "image_height": 480,
        "target_observation": {
            "label": "backpack",
            "confidence": 0.12,
            "cx": 320,
            "cy": 300,
            "area": 12000,
            "image_width": 640,
            "image_height": 480,
            "bbox": {"x1": 260, "y1": 200, "x2": 380, "y2": 300},
        },
        "confirmation_diagnostics": diagnostics,
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["source"] == "vision_candidate"
    assert payload["authoritative"] is False
    assert payload["confirmation_diagnostics"] == diagnostics
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_runtime_marvin_preview_preserves_tracker_provenance():
    seed_bbox = {"x1": 314, "y1": 87, "x2": 459, "y2": 340}
    tracker_bbox = {"x1": 315, "y1": 88, "x2": 460, "y2": 341}
    diagnostics = {"confirmation_status": "target_confirmed"}
    result = {
        "ok": True,
        "preview": True,
        "authoritative": False,
        "source": "marvin_local_tracker",
        "target": "marvin",
        "behavior": "FIND_OBJECT",
        "state": "PREVIEW",
        "bbox": tracker_bbox,
        "target_found": True,
        "target_label": "marvin",
        "target_center_x": 387.5,
        "target_center_y": 214.5,
        "target_area": 36179,
        "image_width": 640,
        "image_height": 480,
        "detector_target": "teddy bear",
        "detector_confidence": 0.057,
        "geometry_source": "yolo",
        "identity_source": "gemini_marvin_identity",
        "identity_confirmed": True,
        "source_timestamp": "2026-09-25T12:00:00+00:00",
        "vision_timestamp": "2026-09-25T12:00:00+00:00",
        "yolo_seed_bbox": seed_bbox,
        "tracker_seed_bbox": {"x1": 285, "y1": 82, "x2": 489, "y2": 346},
        "tracker_seed_source": "bounded_yolo_proposal_expansion",
        "confirmation_diagnostics": diagnostics,
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["source"] == "marvin_local_tracker"
    assert payload["detector_target"] == "teddy bear"
    assert payload["detector_confidence"] == 0.057
    assert payload["geometry_source"] == "yolo"
    assert payload["identity_source"] == "gemini_marvin_identity"
    assert payload["identity_confirmed"] is True
    assert payload["source_timestamp"] == "2026-09-25T12:00:00+00:00"
    assert payload["vision_timestamp"] == "2026-09-25T12:00:00+00:00"
    assert payload["tracking"]["vision_timestamp"] == "2026-09-25T12:00:00+00:00"
    assert payload["yolo_seed_bbox"] == seed_bbox
    assert payload["tracker_seed_bbox"] == {"x1": 285, "y1": 82, "x2": 489, "y2": 346}
    assert payload["tracker_seed_source"] == "bounded_yolo_proposal_expansion"
    assert payload["confirmation_diagnostics"] == diagnostics
    assert payload["tracking"]["source"] == "marvin_local_tracker"
    assert payload["tracking"]["bbox"] == tracker_bbox


def test_runtime_preview_authoritative_result_keeps_diagnostics_null():
    result = {
        "ok": True,
        "preview": True,
        "authoritative": True,
        "source": "world_model",
        "target": "backpack",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": True,
        "target_label": "backpack",
        "target_center_x": 320,
        "target_center_y": 300,
        "target_area": 12000,
        "image_width": 640,
        "image_height": 480,
        "target_observation": {},
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["authoritative"] is True
    assert payload["confirmation_diagnostics"] is None
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_runtime_preview_failed_result_preserves_supplied_diagnostics():
    diagnostics = {
        "confirmation_status": "target_reconfirmation_failed",
        "terminal_reason": "insufficient_temporal_or_geometric_support",
    }
    result = {
        "ok": False,
        "preview": True,
        "authoritative": False,
        "source": "vision_candidate",
        "target": "backpack",
        "reason": "Target was not confirmed.",
        "state": "PREVIEW",
        "behavior": "FIND_OBJECT",
        "target_found": False,
        "confirmation_diagnostics": diagnostics,
    }
    status, payload = _call_runtime_preview(result)
    assert status == 200
    assert payload["ok"] is False
    assert payload["confirmation_diagnostics"] == diagnostics
    assert "confirmation_diagnostics" not in payload["tracking"]


def test_voice_relay_preview_proxy_forwards_only_read_only_request():
    handler = object.__new__(VoiceRelayHandler)
    handler.path = "/dashboard/find-object-preview?target=backpack"
    responses = []
    handler.send_json = lambda code, payload: responses.append((code, payload))
    with patch(
        "voice_relay.server.request_json",
        return_value={
            "ok": True,
            "status_code": 200,
            "data": {"ok": True, "preview": True, "target": "backpack"},
            "error": None,
        },
    ) as request:
        handler.do_GET()
    assert responses == [(200, {"ok": True, "preview": True, "target": "backpack"})]
    assert request.call_args.args[0] == "GET"
    assert "/find-object/preview?target=backpack" in request.call_args.args[1]
    assert request.call_args.kwargs["timeout"] == FIND_OBJECT_PREVIEW_TIMEOUT_SECONDS


def test_preview_proxy_timeout_is_dedicated_to_read_only_preview():
    assert FIND_OBJECT_PREVIEW_TIMEOUT_SECONDS == 25.0
    assert "timeout=3.0" in inspect.getsource(VoiceRelayHandler.dashboard_status)


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("FIND_OBJECT preview tests passed.")
