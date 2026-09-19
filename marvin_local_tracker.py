"""Bounded local-only Marvin template tracker seeded by a semantic bbox."""
import math


class MarvinLocalTracker:
    """Track a single seeded patch with bounded, normalized template matching."""

    MATCH_METHOD = "TM_CCOEFF_NORMED"
    MIN_MATCH_QUALITY = 0.80
    SEARCH_RADIUS_RATIO = 0.50
    MIN_SEARCH_RADIUS_PIXELS = 16

    def __init__(self, frame, bbox):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ValueError("marvin_local_tracker_dependencies_unavailable") from exc
        self._cv2 = cv2
        self._np = np
        self.width = self._valid_dimension(getattr(frame, "width", None))
        self.height = self._valid_dimension(getattr(frame, "height", None))
        self.bbox = self._validate_bbox(bbox, self.width, self.height)
        image = self._decode(frame)
        x1, y1, x2, y2 = self.bbox
        self.template = image[y1:y2, x1:x2].copy()
        if self.template.size == 0 or float(self.template.std()) < 1.0:
            raise ValueError("marvin_local_tracker_seed_invalid")
        self.last_quality = None
        self.last_search_roi = None

    @staticmethod
    def _valid_dimension(value):
        if type(value) is not int or value <= 0:
            raise ValueError("marvin_local_tracker_dimensions_invalid")
        return value

    @staticmethod
    def _validate_bbox(bbox, width, height):
        if not isinstance(bbox, dict) or set(bbox) != {"x1", "y1", "x2", "y2"}:
            raise ValueError("marvin_local_tracker_bbox_invalid")
        values = [bbox[key] for key in ("x1", "y1", "x2", "y2")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("marvin_local_tracker_bbox_invalid")
        x1, y1, x2, y2 = (int(round(value)) for value in values)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("marvin_local_tracker_bbox_invalid")
        return x1, y1, x2, y2

    def _decode(self, frame):
        if (
            getattr(frame, "width", None) != self.width
            or getattr(frame, "height", None) != self.height
        ):
            raise ValueError("marvin_local_tracker_dimensions_changed")
        try:
            encoded = self._np.frombuffer(frame.data, dtype=self._np.uint8)
            image = self._cv2.imdecode(encoded, self._cv2.IMREAD_GRAYSCALE)
        except (AttributeError, TypeError, ValueError, self._cv2.error) as exc:
            raise ValueError("marvin_local_tracker_decode_invalid") from exc
        if image is None or image.shape != (self.height, self.width):
            raise ValueError("marvin_local_tracker_decode_invalid")
        return image

    def _search_roi(self):
        x1, y1, _x2, _y2 = self.bbox
        template_height, template_width = self.template.shape
        radius = max(
            self.MIN_SEARCH_RADIUS_PIXELS,
            int(max(template_width, template_height) * self.SEARCH_RADIUS_RATIO),
        )
        min_left = max(0, x1 - radius)
        max_left = min(self.width - template_width, x1 + radius)
        min_top = max(0, y1 - radius)
        max_top = min(self.height - template_height, y1 + radius)
        if min_left > max_left or min_top > max_top:
            raise ValueError("marvin_local_tracker_search_invalid")
        return min_left, min_top, max_left + template_width, max_top + template_height

    def update(self, frame):
        """Return a locally matched bbox, or ``None`` when tracking is lost."""
        try:
            image = self._decode(frame)
            left, top, right, bottom = self._search_roi()
            roi = image[top:bottom, left:right]
            result = self._cv2.matchTemplate(
                roi, self.template, self._cv2.TM_CCOEFF_NORMED,
            )
            _minimum, quality, _min_location, location = self._cv2.minMaxLoc(result)
        except (ValueError, self._cv2.error):
            self.last_quality = None
            return None
        self.last_quality = float(quality)
        self.last_search_roi = (left, top, right, bottom)
        if not math.isfinite(self.last_quality) or self.last_quality < self.MIN_MATCH_QUALITY:
            return None
        template_height, template_width = self.template.shape
        x1 = left + int(location[0])
        y1 = top + int(location[1])
        bbox = (x1, y1, x1 + template_width, y1 + template_height)
        try:
            self.bbox = self._validate_bbox(
                dict(zip(("x1", "y1", "x2", "y2"), bbox)),
                self.width, self.height,
            )
        except ValueError:
            return None
        return dict(x1=x1, y1=y1, x2=x1 + template_width, y2=y1 + template_height)
