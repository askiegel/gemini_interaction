#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone

from prediction_tracker import PredictionTracker


def iso_at(seconds):
    return (
        datetime.now(timezone.utc)
        + timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")


tracker = PredictionTracker(
    velocity_smoothing=1.0,
    maximum_prediction_seconds=2.0,
)

start = datetime.now(timezone.utc)

time_1 = start.isoformat().replace(
    "+00:00",
    "Z",
)

time_2 = (
    start + timedelta(seconds=1.0)
).isoformat().replace(
    "+00:00",
    "Z",
)

print("===== FIRST MEASUREMENT =====")

first = tracker.update(
    cx=200.0,
    area=20000.0,
    image_width=640.0,
    timestamp=time_1,
)

print(first)

assert first["measurement_count"] == 1
assert first["last_cx"] == 200.0

print()
print("===== SECOND MEASUREMENT =====")

second = tracker.update(
    cx=300.0,
    area=24000.0,
    image_width=640.0,
    timestamp=time_2,
)

print(second)

assert second["measurement_count"] == 2
assert abs(
    second["horizontal_velocity"] - 100.0
) < 0.001

assert abs(
    second["area_velocity"] - 4000.0
) < 0.001

print()
print("===== ONE-SECOND PREDICTION =====")

prediction = tracker.predict(
    elapsed_seconds=1.0
)

print(prediction)

assert prediction["available"] is True
assert abs(
    prediction["predicted_cx"] - 400.0
) < 0.001

assert abs(
    prediction["predicted_area"] - 28000.0
) < 0.001

assert (
    prediction["predicted_direction"]
    == PredictionTracker.DIRECTION_RIGHT
)

print()
print("===== CLAMP TO IMAGE WIDTH =====")

clamped = tracker.predict(
    elapsed_seconds=10.0
)

print(clamped)

assert clamped["predicted_cx"] <= 640.0
assert (
    clamped["prediction_horizon_seconds"]
    == 2.0
)

print()
print("PASS: target velocity was estimated")
print("PASS: image position was predicted")
print("PASS: target area was predicted")
print("PASS: prediction horizon was limited")
print("PASS: prediction remained inside the image")
print()
print("Prediction Tracker test passed.")
