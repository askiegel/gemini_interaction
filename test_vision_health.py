from world_model import WorldModel
from vision_adapter import VisionAdapter


class FakeWaitingForCameraAdapter(VisionAdapter):
    def fetch_vision_payload(self):
        return {
            "timestamp": None,
            "detections": [],
            "description": "No frame processed yet.",
            "camera_running": False
        }


class FakeNoDetectionsAdapter(VisionAdapter):
    def fetch_vision_payload(self):
        return {
            "timestamp": "2026-01-01T00:00:00Z",
            "detections": [],
            "description": "I do not recognize any common objects.",
            "camera_running": True
        }


class FakeDetectionsAdapter(VisionAdapter):
    def fetch_vision_payload(self):
        return {
            "timestamp": "2026-01-01T00:00:01Z",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "center_x": 320,
                    "center_y": 240,
                    "x1": 250,
                    "y1": 100,
                    "x2": 390,
                    "y2": 380,
                    "image_width": 640,
                    "image_height": 480
                }
            ],
            "description": "I see person.",
            "camera_running": True
        }


wm = WorldModel("test_vision_health_state.json")

adapter = FakeWaitingForCameraAdapter(wm)
adapter.process_once()
assert wm.environment["vision"]["vision_status"] == "WAITING_FOR_CAMERA"

adapter = FakeNoDetectionsAdapter(wm)
adapter.process_once()
assert wm.environment["vision"]["vision_status"] == "NO_DETECTIONS"

adapter = FakeDetectionsAdapter(wm)
entity_ids = adapter.process_once()
assert wm.environment["vision"]["vision_status"] == "DETECTIONS_AVAILABLE"
assert wm.environment["vision"]["detection_count"] == 1
assert len(entity_ids) == 1

print("Vision health test passed.")
print(wm.environment["vision"])
