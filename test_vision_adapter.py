from world_model import WorldModel
from vision_adapter import VisionAdapter


class FakeVisionAdapter(VisionAdapter):
    def fetch_detections(self):
        return [
            {
                "label": "person",
                "confidence": 0.94,
                "cx": 320,
                "cy": 240,
                "bbox": [250, 100, 390, 380],
                "reid": "candidate-a"
            },
            {
                "label": "backpack",
                "confidence": 0.87,
                "cx": 500,
                "cy": 260,
                "bbox": [450, 200, 550, 320]
            }
        ]


wm = WorldModel("test_vision_adapter_state.json")
adapter = FakeVisionAdapter(wm)

entity_ids = adapter.process_once()

print("Vision Adapter entity IDs:", entity_ids)

assert len(entity_ids) == 2

entities = wm.get_entities()

assert len(entities) == 2
assert entities[0]["label"] in ["person", "backpack"]
assert entities[1]["label"] in ["person", "backpack"]

print("")
print("World Model entities:")
for entity in entities:
    print(entity["entity_id"], entity["label"], entity["entity_type"], "history:", len(entity["history"]))

print("")
print("Vision Adapter test passed.")
