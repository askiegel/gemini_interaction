import os
import time
import requests
from typing import Dict, Any, List, Optional

from world_model import WorldModel
from entity_registry import EntityRegistry


DEFAULT_VISION_SERVER_URL = os.getenv(
    "VISION_SERVER_URL",
    "http://localhost:8000/detect"
)

DEFAULT_VISION_IMAGE_PATH = os.getenv(
    "VISION_IMAGE_PATH",
    ""
)


class VisionAdapter:
    """
    Connects the external Vision Server to the cognitive World Model.

    This layer does NOT depend on ROS2.
    It converts vision detections into persistent semantic entities.

    Current Vision Server contract:
    POST /detect with an uploaded image file.
    """

    def __init__(
        self,
        world_model: WorldModel,
        vision_url: str = DEFAULT_VISION_SERVER_URL,
        image_path: str = DEFAULT_VISION_IMAGE_PATH,
        poll_interval: float = 1.0
    ):
        self.world_model = world_model
        self.registry = EntityRegistry(world_model)
        self.vision_url = vision_url
        self.image_path = image_path
        self.poll_interval = poll_interval
        self.running = False

    def fetch_detections(self) -> List[Dict[str, Any]]:
        if not self.image_path:
            raise RuntimeError(
                "VISION_IMAGE_PATH is not set. "
                "Current Vision Server requires an image upload to POST /detect."
            )

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Vision image not found: {self.image_path}")

        with open(self.image_path, "rb") as image_file:
            files = {
                "file": (
                    os.path.basename(self.image_path),
                    image_file,
                    "image/jpeg"
                )
            }

            response = requests.post(
                self.vision_url,
                files=files,
                timeout=10
            )

        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return data

        if "detections" in data:
            return data["detections"]

        if "objects" in data and isinstance(data["objects"], list):
            return data["objects"]

        return []

    def process_detection(self, detection: Dict[str, Any]) -> Optional[str]:
        if isinstance(detection, str):
            detection = {"label": detection, "confidence": 0.0}

        label = (
            detection.get("label")
            or detection.get("class")
            or detection.get("name")
            or detection.get("object")
            or "unknown"
        )

        confidence = float(
            detection.get("confidence")
            or detection.get("conf")
            or detection.get("score")
            or 0.0
        )

        entity_type = self._infer_entity_type(label)
        location = self._extract_location(detection)

        attributes = {
            "raw_detection": detection,
            "targetable": label in ["person", "backpack", "chair", "bottle", "cup"]
        }

        if "reid" in detection:
            attributes["reid"] = detection["reid"]

        if "track_id" in detection:
            attributes["track_id"] = detection["track_id"]

        entity_id = self.registry.register_observation(
            label=label,
            entity_type=entity_type,
            confidence=confidence,
            source="vision_server",
            location=location,
            attributes=attributes
        )

        self.world_model.add_event(
            "vision_detection_processed",
            {
                "entity_id": entity_id,
                "label": label,
                "confidence": confidence
            }
        )

        self.world_model.save()
        return entity_id

    def process_once(self) -> List[str]:
        entity_ids = []
        detections = self.fetch_detections()

        for detection in detections:
            entity_id = self.process_detection(detection)

            if entity_id:
                entity_ids.append(entity_id)

        return entity_ids

    def run_forever(self):
        self.running = True

        print("Vision Adapter started")
        print(f"Vision URL: {self.vision_url}")

        if self.image_path:
            print(f"Vision image source: {self.image_path}")
        else:
            print("Vision image source: NOT SET")

        while self.running:
            try:
                entity_ids = self.process_once()

                if entity_ids:
                    print("Updated entities:", entity_ids)
                else:
                    print("No detections")

            except Exception as exc:
                print("Vision Adapter error:", exc)

            time.sleep(self.poll_interval)

    def stop(self):
        self.running = False

    def _infer_entity_type(self, label: str) -> str:
        if label == "person":
            return "human"

        if label in ["dog", "cat", "animal"]:
            return "animal"

        return "object"

    def _extract_location(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        location = {
            "frame": "camera"
        }

        if "cx" in detection:
            location["cx"] = detection["cx"]

        if "cy" in detection:
            location["cy"] = detection["cy"]

        if "center_x" in detection:
            location["cx"] = detection["center_x"]

        if "center_y" in detection:
            location["cy"] = detection["center_y"]

        if "area" in detection:
            location["area"] = detection["area"]

        if "bbox" in detection:
            location["bbox"] = detection["bbox"]

        if all(k in detection for k in ["x1", "y1", "x2", "y2"]):
            location["bbox"] = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"]
            ]

            location["cx"] = (detection["x1"] + detection["x2"]) / 2
            location["cy"] = (detection["y1"] + detection["y2"]) / 2

        if "image_width" in detection:
            location["image_width"] = detection["image_width"]

        if "image_height" in detection:
            location["image_height"] = detection["image_height"]

        return location
