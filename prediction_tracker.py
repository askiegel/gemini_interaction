#!/usr/bin/env python3

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class PredictionTracker:
    """
    Estimate short-term target motion in camera-image coordinates.

    The tracker uses a smoothed constant-velocity model:

        predicted_position =
            last_position + velocity * elapsed_time

    It predicts horizontal image position and bounding-box area during
    short target occlusions. It does not perform identity matching.
    """

    DIRECTION_LEFT = "LEFT"
    DIRECTION_RIGHT = "RIGHT"
    DIRECTION_CENTER = "CENTER"

    def __init__(
        self,
        velocity_smoothing: float = 0.65,
        maximum_prediction_seconds: float = 2.0,
        center_band_ratio: float = 0.10,
        minimum_center_band_pixels: float = 40.0,
    ):
        smoothing = float(velocity_smoothing)

        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(
                "velocity_smoothing must be between 0 and 1."
            )

        self.velocity_smoothing = smoothing
        self.maximum_prediction_seconds = float(
            maximum_prediction_seconds
        )
        self.center_band_ratio = float(
            center_band_ratio
        )
        self.minimum_center_band_pixels = float(
            minimum_center_band_pixels
        )

        self.reset()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _now_iso(cls) -> str:
        return (
            cls._now()
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> Optional[datetime]:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(timezone.utc)

    @classmethod
    def _elapsed_seconds(
        cls,
        timestamp: Any,
        now: Optional[datetime] = None,
    ) -> Optional[float]:
        parsed = cls._parse_timestamp(timestamp)

        if parsed is None:
            return None

        current = now or cls._now()

        return max(
            0.0,
            (current - parsed).total_seconds(),
        )

    def reset(self):
        self.last_cx: Optional[float] = None
        self.last_area: Optional[float] = None
        self.last_image_width: Optional[float] = None
        self.last_timestamp: Optional[str] = None

        self.horizontal_velocity = 0.0
        self.area_velocity = 0.0
        self.measurement_count = 0

    def update(
        self,
        cx: Any,
        area: Any,
        image_width: Any,
        timestamp: Any = None,
    ) -> Dict[str, Any]:
        if cx is None or image_width is None:
            return self.snapshot()

        measured_cx = float(cx)
        measured_width = float(image_width)

        measured_area = (
            float(area)
            if area is not None
            else None
        )

        measurement_time = (
            self._parse_timestamp(timestamp)
            or self._now()
        )

        measurement_iso = (
            measurement_time
            .isoformat()
            .replace("+00:00", "Z")
        )

        previous_time = self._parse_timestamp(
            self.last_timestamp
        )

        if (
            previous_time is not None
            and self.last_cx is not None
        ):
            delta_time = (
                measurement_time - previous_time
            ).total_seconds()

            if delta_time > 0.001:
                measured_horizontal_velocity = (
                    measured_cx - self.last_cx
                ) / delta_time

                alpha = self.velocity_smoothing

                self.horizontal_velocity = (
                    alpha
                    * measured_horizontal_velocity
                    + (1.0 - alpha)
                    * self.horizontal_velocity
                )

                if (
                    measured_area is not None
                    and self.last_area is not None
                ):
                    measured_area_velocity = (
                        measured_area - self.last_area
                    ) / delta_time

                    self.area_velocity = (
                        alpha
                        * measured_area_velocity
                        + (1.0 - alpha)
                        * self.area_velocity
                    )

        self.last_cx = measured_cx
        self.last_area = measured_area
        self.last_image_width = measured_width
        self.last_timestamp = measurement_iso
        self.measurement_count += 1

        return self.snapshot()

    def _direction(
        self,
        cx: float,
        image_width: float,
    ) -> str:
        image_center = image_width / 2.0

        center_band = max(
            self.minimum_center_band_pixels,
            image_width * self.center_band_ratio,
        )

        horizontal_error = cx - image_center

        if horizontal_error < -center_band:
            return self.DIRECTION_LEFT

        if horizontal_error > center_band:
            return self.DIRECTION_RIGHT

        return self.DIRECTION_CENTER

    def predict(
        self,
        elapsed_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if (
            self.last_cx is None
            or self.last_image_width is None
            or self.last_timestamp is None
        ):
            return {
                "available": False,
                "reason": (
                    "No target motion history is available."
                ),
            }

        if elapsed_seconds is None:
            elapsed = self._elapsed_seconds(
                self.last_timestamp
            )
        else:
            elapsed = max(
                0.0,
                float(elapsed_seconds),
            )

        if elapsed is None:
            return {
                "available": False,
                "reason": (
                    "The last measurement timestamp "
                    "could not be interpreted."
                ),
            }

        prediction_horizon = min(
            elapsed,
            self.maximum_prediction_seconds,
        )

        predicted_cx = (
            self.last_cx
            + self.horizontal_velocity
            * prediction_horizon
        )

        predicted_cx = min(
            max(predicted_cx, 0.0),
            self.last_image_width,
        )

        predicted_area = None

        if self.last_area is not None:
            predicted_area = max(
                0.0,
                self.last_area
                + self.area_velocity
                * prediction_horizon,
            )

        predicted_direction = self._direction(
            predicted_cx,
            self.last_image_width,
        )

        return {
            "available": True,
            "predicted_cx": predicted_cx,
            "predicted_area": predicted_area,
            "predicted_direction": predicted_direction,
            "image_width": self.last_image_width,
            "horizontal_velocity": (
                self.horizontal_velocity
            ),
            "area_velocity": self.area_velocity,
            "prediction_horizon_seconds": (
                prediction_horizon
            ),
            "measurement_count": (
                self.measurement_count
            ),
            "last_measurement_at": (
                self.last_timestamp
            ),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "last_cx": self.last_cx,
            "last_area": self.last_area,
            "last_image_width": (
                self.last_image_width
            ),
            "last_timestamp": self.last_timestamp,
            "horizontal_velocity": (
                self.horizontal_velocity
            ),
            "area_velocity": self.area_velocity,
            "measurement_count": (
                self.measurement_count
            ),
            "maximum_prediction_seconds": (
                self.maximum_prediction_seconds
            ),
        }
