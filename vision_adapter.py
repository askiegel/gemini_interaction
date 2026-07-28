import os
import time
import requests
from typing import Dict, Any, List, Optional

from world_model import WorldModel
from entity_registry import EntityRegistry
from person_identity_manager import PersonIdentityManager


DEFAULT_VISION_SERVER_URL = os.getenv(
    "VISION_SERVER_URL",
    "http://localhost:8000/detections/latest"
)

DEFAULT_VISION_IMAGE_PATH = os.getenv(
    "VISION_IMAGE_PATH",
    ""
)


class VisionAdapter:
    """
    Connects the external Vision Server to the cognitive World Model.

    Preferred contract:
    GET /detections/latest

    Compatibility contract:
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
        self.identity_manager = PersonIdentityManager()
        self.vision_url = vision_url
        self.image_path = image_path
        self.poll_interval = poll_interval
        self.running = False
        self.last_payload = {}

    def fetch_vision_payload(self) -> Dict[str, Any]:
        if self.image_path:
            detections = self._fetch_from_post_image()
            return {
                "timestamp": None,
                "detections": detections,
                "description": "Image upload detection result.",
                "camera_running": None,
                "vision_status": "IMAGE_UPLOAD_MODE"
            }

        response = requests.get(self.vision_url, timeout=5)
        response.raise_for_status()

        data = response.json()

        if isinstance(data, list):
            return {
                "timestamp": None,
                "detections": data,
                "description": "",
                "camera_running": None,
                "vision_status": "LEGACY_LIST_RESPONSE"
            }

        return data

    def _assign_person_identities(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enrich person detections with persistent identity information.

        Non-person detections pass through unchanged. Frame dimensions from
        the Vision Server payload are copied into person detections when the
        detector did not include them directly.
        """
        enriched = []
        person_positions = []
        person_detections = []

        frame_width = (
            self.last_payload.get("image_width")
            or self.last_payload.get("frame_width")
            or self.last_payload.get("width")
        )

        frame_height = (
            self.last_payload.get("image_height")
            or self.last_payload.get("frame_height")
            or self.last_payload.get("height")
        )

        for detection in detections:
            if not isinstance(detection, dict):
                enriched.append(detection)
                continue

            copied = dict(detection)
            enriched.append(copied)

            label = self._normalize_label(
                copied.get("label")
                or copied.get("class")
                or copied.get("name")
                or copied.get("object")
            )

            if label != "person":
                continue

            if copied.get("image_width") is None and frame_width is not None:
                copied["image_width"] = frame_width

            if copied.get("image_height") is None and frame_height is not None:
                copied["image_height"] = frame_height

            person_positions.append(len(enriched) - 1)
            person_detections.append(copied)

        if not person_detections:
            return enriched

        assigned = self.identity_manager.assign_identities(
            person_detections
        )

        for position, detection in zip(
            person_positions,
            assigned,
        ):
            enriched[position] = detection

        return enriched

    def fetch_detections(self) -> List[Dict[str, Any]]:
        payload = self.fetch_vision_payload()
        self.last_payload = payload
        self.update_vision_health(payload)

        detections = payload.get("detections", [])

        if isinstance(detections, list):
            return [
                dict(detection)
                if isinstance(detection, dict)
                else detection
                for detection in detections
            ]

        objects = payload.get("objects", [])

        if isinstance(objects, list):
            return [{"label": item, "confidence": 0.0} for item in objects]

        return []

    @staticmethod
    def _normalize_label(value: Any) -> str:
        """
        Normalize target and detection labels for reliable comparisons.
        """
        label = str(value or "").strip().lower()
        label = label.replace("_", " ").replace("-", " ")
        label = " ".join(label.split())

        aliases = {
            "back pack": "backpack",
            "rucksack": "backpack",
            "person": "person",
            "human": "person",
        }

        return aliases.get(label, label)

    @staticmethod
    def _extract_bbox(detection: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Convert common YOLO bounding-box formats into x1/y1/x2/y2.
        """
        bbox = detection.get("bbox")

        if isinstance(bbox, dict):
            x1 = bbox.get("x1", bbox.get("left"))
            y1 = bbox.get("y1", bbox.get("top"))
            x2 = bbox.get("x2", bbox.get("right"))
            y2 = bbox.get("y2", bbox.get("bottom"))

            if None not in (x1, y1, x2, y2):
                return {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
            }

        coordinate_sets = (
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        )

        for keys in coordinate_sets:
            if all(key in detection for key in keys):
                return {
                    "x1": float(detection[keys[0]]),
                    "y1": float(detection[keys[1]]),
                    "x2": float(detection[keys[2]]),
                    "y2": float(detection[keys[3]]),
                }

        return None

    def normalize_detection(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize a Vision Server detection for cognitive behavior decisions.

        The raw detection is preserved so provider-specific data is never lost.
        """
        label = self._normalize_label(
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

        bbox = self._extract_bbox(detection)

        cx = detection.get("cx", detection.get("center_x"))
        cy = detection.get("cy", detection.get("center_y"))
        area = detection.get("area")

        if bbox:
            width = max(0.0, bbox["x2"] - bbox["x1"])
            height = max(0.0, bbox["y2"] - bbox["y1"])

            if cx is None:
                cx = bbox["x1"] + width / 2.0

            if cy is None:
                cy = bbox["y1"] + height / 2.0

            if area is None:
                area = width * height

        image_width = (
            detection.get("image_width")
            or self.last_payload.get("image_width")
            or self.last_payload.get("frame_width")
            or self.last_payload.get("width")
        )

        image_height = (
            detection.get("image_height")
            or self.last_payload.get("image_height")
            or self.last_payload.get("frame_height")
            or self.last_payload.get("height")
        )

        return {
            "label": label,
            "confidence": confidence,
            "cx": float(cx) if cx is not None else None,
            "cy": float(cy) if cy is not None else None,
            "area": float(area) if area is not None else None,
            "bbox": bbox,
            "image_width": (
                float(image_width) if image_width is not None else None
            ),
            "image_height": (
                float(image_height) if image_height is not None else None
            ),
            "entity_id": detection.get("entity_id"),
            "identity_id": detection.get("identity_id"),
            "identity_match_score": detection.get(
                "identity_match_score",
                0.0,
            ),
            "identity_status": detection.get("identity_status"),
            "identity_ambiguous": detection.get(
                "identity_ambiguous",
                False,
            ),
            "identity_diagnostics": detection.get(
                "identity_diagnostics"
            ),
            "attributes": dict(
                detection.get("attributes") or {}
            ),
            "raw_detection": detection,
        }

    def find_target(self, target_label: str) -> Dict[str, Any]:
        """
        Fetch the latest detections and return the best matching target.

        All fetched detections are processed into the World Model before the
        target result is returned. The World Model therefore remains the
        platform's single source of truth.
        """
        normalized_target = self._normalize_label(target_label)
        detections = self.process_detection_frame(
            self.fetch_detections()
        )
        candidates = []

        for detection in detections:
            normalized = self.normalize_detection(detection)

            if normalized["label"] == normalized_target:
                candidates.append(normalized)

        if not candidates:
            result = {
                "found": False,
                "target": normalized_target,
                "vision_status": self.last_payload.get(
                    "vision_status",
                    self.world_model.environment
                    .get("vision", {})
                    .get("vision_status", "UNKNOWN"),
                ),
            }

            self.world_model.add_event(
                "vision_target_not_found",
                result,
            )
            self.world_model.save()
            return result

        best = max(
            candidates,
            key=lambda item: (
                item.get("confidence") or 0.0,
                item.get("area") or 0.0,
            ),
        )

        result = {
            "found": True,
            "target": normalized_target,
            **best,
        }

        self.world_model.add_event(
            "vision_target_found",
            {
                "target": normalized_target,
                "label": best["label"],
                "confidence": best["confidence"],
                "cx": best["cx"],
                "cy": best["cy"],
                "area": best["area"],
            },
        )
        self.world_model.save()
        return result

    def update_vision_health(self, payload: Dict[str, Any]):
        camera_running = payload.get("camera_running")
        timestamp = payload.get("timestamp")
        detections = payload.get("detections", [])
        description = payload.get("description", "")

        if camera_running is False:
            vision_status = "WAITING_FOR_CAMERA"
        elif timestamp is None:
            vision_status = "WAITING_FOR_FRAME"
        elif detections:
            vision_status = "DETECTIONS_AVAILABLE"
        else:
            vision_status = "NO_DETECTIONS"

        if payload.get("vision_status"):
            vision_status = payload["vision_status"]

        health = {
            "camera_running": camera_running,
            "last_frame_time": timestamp,
            "detections_available": bool(detections),
            "detection_count": len(detections) if isinstance(detections, list) else 0,
            "description": description,
            "vision_status": vision_status,
            "vision_url": self.vision_url
        }

        self.world_model.environment["vision"] = health

        self.world_model.add_event(
            "vision_health_updated",
            health
        )

        self.world_model.save()

    def _fetch_from_post_image(self) -> List[Dict[str, Any]]:
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
            return [{"label": item, "confidence": 0.0} for item in data["objects"]]

        return []

    def _nested_person_duplicate(
        self,
        first: Dict[str, Any],
        second: Dict[str, Any],
    ) -> bool:
        first_bbox = self._extract_bbox(first)
        second_bbox = self._extract_bbox(second)

        if first_bbox is None or second_bbox is None:
            return False

        intersection_width = max(
            0.0,
            min(first_bbox["x2"], second_bbox["x2"])
            - max(first_bbox["x1"], second_bbox["x1"]),
        )
        intersection_height = max(
            0.0,
            min(first_bbox["y2"], second_bbox["y2"])
            - max(first_bbox["y1"], second_bbox["y1"]),
        )
        intersection = (
            intersection_width * intersection_height
        )

        first_width = max(
            0.0,
            first_bbox["x2"] - first_bbox["x1"],
        )
        first_height = max(
            0.0,
            first_bbox["y2"] - first_bbox["y1"],
        )
        second_width = max(
            0.0,
            second_bbox["x2"] - second_bbox["x1"],
        )
        second_height = max(
            0.0,
            second_bbox["y2"] - second_bbox["y1"],
        )

        first_area = first_width * first_height
        second_area = second_width * second_height
        smaller_area = min(first_area, second_area)
        union = first_area + second_area - intersection

        if smaller_area <= 0.0 or union <= 0.0:
            return False

        containment = intersection / smaller_area
        iou = intersection / union

        first_cx = (
            first_bbox["x1"] + first_bbox["x2"]
        ) / 2.0
        first_cy = (
            first_bbox["y1"] + first_bbox["y2"]
        ) / 2.0
        second_cx = (
            second_bbox["x1"] + second_bbox["x2"]
        ) / 2.0
        second_cy = (
            second_bbox["y1"] + second_bbox["y2"]
        ) / 2.0

        maximum_width = max(first_width, second_width)
        maximum_height = max(first_height, second_height)

        center_x_ratio = (
            abs(first_cx - second_cx) / maximum_width
            if maximum_width > 0.0
            else 1.0
        )
        center_y_ratio = (
            abs(first_cy - second_cy) / maximum_height
            if maximum_height > 0.0
            else 1.0
        )

        return (
            containment >= 0.90
            and iou >= 0.20
            and center_x_ratio <= 0.30
            and center_y_ratio <= 0.30
        )

    def _deduplicate_nested_people(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        deduplicated = []

        for detection in detections:
            if not isinstance(detection, dict):
                deduplicated.append(detection)
                continue

            label = self._normalize_label(
                detection.get("label")
                or detection.get("class")
                or detection.get("name")
                or detection.get("object")
            )

            if label != "person":
                deduplicated.append(detection)
                continue

            duplicate_position = None

            for position, existing in enumerate(
                deduplicated
            ):
                if not isinstance(existing, dict):
                    continue

                existing_label = self._normalize_label(
                    existing.get("label")
                    or existing.get("class")
                    or existing.get("name")
                    or existing.get("object")
                )

                if (
                    existing_label == "person"
                    and self._nested_person_duplicate(
                        existing,
                        detection,
                    )
                ):
                    duplicate_position = position
                    break

            if duplicate_position is None:
                deduplicated.append(detection)
                continue

            existing = deduplicated[duplicate_position]

            existing_rank = (
                float(
                    existing.get("confidence")
                    or existing.get("conf")
                    or existing.get("score")
                    or 0.0
                ),
                float(existing.get("area") or 0.0),
            )
            candidate_rank = (
                float(
                    detection.get("confidence")
                    or detection.get("conf")
                    or detection.get("score")
                    or 0.0
                ),
                float(detection.get("area") or 0.0),
            )

            if candidate_rank > existing_rank:
                deduplicated[duplicate_position] = detection

        return deduplicated

    def process_detection_frame(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Register one perception frame before assigning persistent identities.

        EntityRegistry remains responsible for deterministic World Model entity
        ownership. PersonIdentityManager then receives those resolved entity IDs.
        Identity enrichment updates the already registered observation rather
        than appending a duplicate history entry.
        """
        detections = self._deduplicate_nested_people(
            detections
        )

        registered = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            copied = dict(detection)
            entity_id = self.process_detection(
                copied
            )

            if entity_id:
                copied["entity_id"] = entity_id

            registered.append(copied)

        # Multiple YOLO person boxes may resolve to the same World Model
        # entity in one frame. Collapse those duplicate boxes before identity
        # assignment so assign_identities() does not interpret them as separate
        # people and create NEW_FRAME_CONFLICT identities.
        deduplicated = []
        person_entity_positions = {}

        for detection in registered:
            label = self._normalize_label(
                detection.get("label")
                or detection.get("class")
                or detection.get("name")
                or detection.get("object")
            )

            entity_id = detection.get(
                "entity_id"
            )

            if label != "person" or not entity_id:
                deduplicated.append(detection)
                continue

            existing_position = (
                person_entity_positions.get(
                    entity_id
                )
            )

            if existing_position is None:
                person_entity_positions[
                    entity_id
                ] = len(deduplicated)
                deduplicated.append(detection)
                continue

            existing = deduplicated[
                existing_position
            ]

            existing_rank = (
                float(
                    existing.get("confidence")
                    or existing.get("conf")
                    or existing.get("score")
                    or 0.0
                ),
                float(
                    existing.get("area")
                    or 0.0
                ),
            )

            candidate_rank = (
                float(
                    detection.get("confidence")
                    or detection.get("conf")
                    or detection.get("score")
                    or 0.0
                ),
                float(
                    detection.get("area")
                    or 0.0
                ),
            )

            if candidate_rank > existing_rank:
                deduplicated[
                    existing_position
                ] = detection

        assigned = self._assign_person_identities(
            deduplicated
        )

        identity_fields = (
            "identity_id",
            "identity_match_score",
            "identity_status",
            "identity_ambiguous",
            "identity_diagnostics",
        )

        for detection in assigned:
            if not isinstance(detection, dict):
                continue

            if (
                self._normalize_label(
                    detection.get("label")
                    or detection.get("class")
                    or detection.get("name")
                    or detection.get("object")
                )
                != "person"
            ):
                continue

            entity_id = detection.get(
                "entity_id"
            )

            if not entity_id:
                continue

            updates = dict(
                detection.get("attributes")
                or {}
            )

            for identity_field in identity_fields:
                if identity_field in detection:
                    updates[identity_field] = (
                        detection[identity_field]
                    )

            updates["raw_detection"] = dict(
                detection
            )

            enriched = (
                self.world_model
                .enrich_latest_entity_observation(
                    entity_id=entity_id,
                    attributes=updates,
                )
            )

            if not enriched:
                raise RuntimeError(
                    "Identity enrichment failed for "
                    f"World Model entity '{entity_id}'."
                )

        return assigned

    def process_detection(self, detection: Dict[str, Any]) -> Optional[str]:
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
            "targetable": label in [
                "person",
                "backpack",
                "chair",
                "bottle",
                "cup",
            ],
        }

        detection_attributes = dict(
            detection.get("attributes") or {}
        )

        attributes.update(detection_attributes)

        identity_fields = (
            "identity_id",
            "identity_match_score",
            "identity_status",
            "identity_ambiguous",
            "identity_diagnostics",
        )

        for identity_field in identity_fields:
            if identity_field in detection:
                attributes[identity_field] = detection[
                    identity_field
                ]

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
        detections = self.process_detection_frame(
            self.fetch_detections()
        )

        return [
            detection["entity_id"]
            for detection in detections
            if (
                isinstance(detection, dict)
                and detection.get("entity_id")
            )
        ]

    def run_forever(self):
        self.running = True

        print("Vision Adapter started")
        print(f"Vision URL: {self.vision_url}")

        if self.image_path:
            print(f"Vision image source: {self.image_path}")
        else:
            print("Vision source: latest detections endpoint")

        while self.running:
            try:
                entity_ids = self.process_once()

                vision_health = self.world_model.environment.get("vision", {})
                vision_status = vision_health.get("vision_status", "UNKNOWN")

                if entity_ids:
                    print("Updated entities:", entity_ids, "| vision:", vision_status)
                else:
                    print("No detections | vision:", vision_status)

            except Exception as exc:
                self.world_model.environment["vision"] = {
                    "camera_running": None,
                    "last_frame_time": None,
                    "detections_available": False,
                    "detection_count": 0,
                    "description": str(exc),
                    "vision_status": "VISION_SERVER_ERROR",
                    "vision_url": self.vision_url
                }

                self.world_model.add_event(
                    "vision_health_error",
                    self.world_model.environment["vision"]
                )

                self.world_model.save()

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

        bbox = self._extract_bbox(detection)

        if bbox is not None:
            location["bbox"] = bbox

            if "cx" not in location:
                location["cx"] = (
                    bbox["x1"] + bbox["x2"]
                ) / 2.0

            if "cy" not in location:
                location["cy"] = (
                    bbox["y1"] + bbox["y2"]
                ) / 2.0

            if "area" not in location:
                location["area"] = (
                    max(
                        0.0,
                        bbox["x2"] - bbox["x1"],
                    )
                    * max(
                        0.0,
                        bbox["y2"] - bbox["y1"],
                    )
                )

        if "image_width" in detection:
            location["image_width"] = detection["image_width"]

        if "image_height" in detection:
            location["image_height"] = detection["image_height"]

        return location
