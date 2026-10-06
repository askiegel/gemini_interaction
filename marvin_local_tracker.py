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
        self.last_bbox = None
        self.last_candidate_bbox = None
        self.last_source_frame_stamp_ns = None
        self.last_image_width = self.width
        self.last_image_height = self.height
        self.last_reason = "awaiting_match"

    @staticmethod
    def horizontal_error(center_x, image_width):
        """Return the canonical signed pixel offset from image center."""
        return float(center_x) - float(image_width) / 2.0

    def preview_diagnostics(self):
        """Return bounded diagnostics for the most recent update only."""
        bbox = dict(self.last_bbox) if self.last_bbox is not None else None
        quality = (
            self.last_quality
            if self.last_quality is not None and math.isfinite(self.last_quality)
            else None
        )
        center_x = (
            (bbox["x1"] + bbox["x2"]) / 2.0 if bbox is not None else None
        )
        center_y = (
            (bbox["y1"] + bbox["y2"]) / 2.0 if bbox is not None else None
        )
        return {
            "active": True,
            "matched": self.last_reason == "matched",
            "quality": quality,
            "threshold": self.MIN_MATCH_QUALITY,
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "horizontal_error": (
                self.horizontal_error(center_x, self.last_image_width)
                if center_x is not None and self.last_image_width is not None
                else None
            ),
            "image_width": self.last_image_width,
            "image_height": self.last_image_height,
            "source_frame_stamp_ns": self.last_source_frame_stamp_ns,
            "reason": self.last_reason,
        }

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
        self.last_bbox = None
        self.last_candidate_bbox = None
        self.last_search_roi = None
        self.last_source_frame_stamp_ns = self._valid_source_stamp(
            getattr(frame, "source_frame_stamp_ns", None)
        )
        self.last_image_width = getattr(frame, "width", None)
        self.last_image_height = getattr(frame, "height", None)
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
            self.last_reason = "frame_unavailable"
            return None
        self.last_quality = float(quality)
        self.last_search_roi = (left, top, right, bottom)
        template_height, template_width = self.template.shape
        x1 = left + int(location[0])
        y1 = top + int(location[1])
        bbox = (x1, y1, x1 + template_width, y1 + template_height)
        # Retain the best candidate for diagnosis even when it cannot supply
        # motion evidence. Matching remains fixed-scale and quality-gated.
        self.last_candidate_bbox = dict(zip(("x1", "y1", "x2", "y2"), bbox))
        if not math.isfinite(self.last_quality) or self.last_quality < self.MIN_MATCH_QUALITY:
            self.last_reason = "below_threshold"
            return None
        try:
            self.bbox = self._validate_bbox(
                dict(zip(("x1", "y1", "x2", "y2"), bbox)),
                self.width, self.height,
            )
        except ValueError:
            self.last_reason = "invalid_bbox"
            return None
        self.last_bbox = dict(
            x1=x1, y1=y1, x2=x1 + template_width, y2=y1 + template_height
        )
        self.last_reason = "matched"
        return dict(x1=x1, y1=y1, x2=x1 + template_width, y2=y1 + template_height)

    @staticmethod
    def _valid_source_stamp(value):
        if type(value) is int and value >= 0:
            return value
        return None
