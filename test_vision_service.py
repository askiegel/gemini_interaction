#!/usr/bin/env python3

import os
import tempfile

from vision_service import VisionService
from world_model import WorldModel


class FakeVisionAdapter:
    def __init__(self, world_model):
        self.world_model = world_model
        self.vision_url = "http://fake-vision/detections/latest"
        self.calls = 0

    def process_once(self):
        self.calls += 1

        detections = [
            {
                "entity_id": "person-001",
                "label": "person",
                "entity_type": "human",
                "confidence": 0.95,
                "cx": 310,
                "cy": 240,
                "bbox": [250, 100, 370, 380],
                "tracking": True,
            },
            {
                "entity_id": "backpack-001",
                "label": "backpack",
                "entity_type": "object",
                "confidence": 0.88,
                "cx": 460,
                "cy": 270,
                "bbox": [410, 200, 510, 340],
                "tracking": True,
            },
        ]

        self.world_model.environment["vision"] = {
            "camera_running": True,
            "detections_available": True,
            "detection_count": len(detections),
            "vision_status": "DETECTIONS_AVAILABLE",
            "vision_url": self.vision_url,
        }

        return self.world_model.update_from_detections(
            detections,
            source="fake_vision_service",
        )


class FailingVisionAdapter:
    def __init__(self):
        self.vision_url = "http://fake-vision/failure"

    def process_once(self):
        raise RuntimeError("Simulated vision failure.")


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="vision_service_test_"
    )

    world_model_path = os.path.join(
        temporary_directory,
        "world_model_state.json",
    )

    failure_world_model_path = os.path.join(
        temporary_directory,
        "failure_world_model_state.json",
    )

    try:
        print("===== SUCCESSFUL PERCEPTION CYCLE =====")

        world_model = WorldModel(world_model_path)
        adapter = FakeVisionAdapter(world_model)

        service = VisionService(
            world_model=world_model,
            vision_adapter=adapter,
            poll_interval=0.01,
        )

        first_result = service.run_once()
        print(first_result)

        assert first_result["ok"] is True
        assert first_result["cycle"] == 1
        assert first_result["entity_count"] == 2
        assert first_result["entity_ids"] == [
            "person-001",
            "backpack-001",
        ]

        assert service.cycles_completed == 1
        assert service.successful_cycles == 1
        assert service.failed_cycles == 0
        assert service.last_error is None

        entities = world_model.get_entities()

        assert len(entities) == 2

        labels = sorted(
            entity["label"]
            for entity in entities
        )

        assert labels == [
            "backpack",
            "person",
        ]

        assert (
            world_model.environment["vision"]["vision_status"]
            == "DETECTIONS_AVAILABLE"
        )

        assert (
            world_model.robot_state[
                "perception_service_state"
            ]
            == "RUNNING"
        )

        print()
        print("===== SECOND PERCEPTION CYCLE =====")

        second_result = service.run_once()
        print(second_result)

        assert second_result["ok"] is True
        assert second_result["cycle"] == 2
        assert adapter.calls == 2
        assert service.cycles_completed == 2
        assert service.successful_cycles == 2

        status = service.get_status()

        print()
        print("===== SERVICE STATUS =====")
        print(status)

        assert status["running"] is False
        assert status["cycles_completed"] == 2
        assert status["successful_cycles"] == 2
        assert status["failed_cycles"] == 0
        assert status["last_entity_ids"] == [
            "person-001",
            "backpack-001",
        ]

        print()
        print("===== FAILED PERCEPTION CYCLE =====")

        failure_world_model = WorldModel(
            failure_world_model_path
        )

        failing_service = VisionService(
            world_model=failure_world_model,
            vision_adapter=FailingVisionAdapter(),
            poll_interval=0.01,
        )

        failure_result = failing_service.run_once()
        print(failure_result)

        assert failure_result["ok"] is False
        assert failure_result["cycle"] == 1
        assert failure_result["entity_count"] == 0
        assert (
            failure_result["error"]
            == "Simulated vision failure."
        )

        assert failing_service.cycles_completed == 1
        assert failing_service.successful_cycles == 0
        assert failing_service.failed_cycles == 1
        assert (
            failing_service.last_error
            == "Simulated vision failure."
        )

        assert (
            failure_world_model.robot_state[
                "perception_service_state"
            ]
            == "ERROR"
        )

        print()
        print("PASS: vision service executes bounded cycles")
        print("PASS: detections update the World Model")
        print("PASS: repeated cycles preserve service state")
        print("PASS: service status reports cycle metrics")
        print("PASS: vision failures are contained safely")
        print()
        print("Vision Service offline test passed.")

    finally:
        for path in (
            world_model_path,
            failure_world_model_path,
        ):
            if os.path.exists(path):
                os.remove(path)

        if os.path.isdir(temporary_directory):
            os.rmdir(temporary_directory)


if __name__ == "__main__":
    main()
