import os
import requests


DEFAULT_VISION_STATUS_URL = "http://localhost:8000/status"


def get_vision_context():
    url = os.getenv("VISION_STATUS_URL", DEFAULT_VISION_STATUS_URL)

    try:
        response = requests.get(url, timeout=1.5)
        response.raise_for_status()
        data = response.json()

        return {
            "source": "vision_server",
            "online": True,
            "url": url,
            "raw": data,
            "detections": normalize_detections(data),
        }

    except Exception as e:
        return {
            "source": "vision_server",
            "online": False,
            "url": url,
            "error": str(e),
            "detections": [],
        }


def normalize_detections(data):
    detections = []

    raw_detections = (
        data.get("detections")
        or data.get("objects")
        or data.get("targets")
        or []
    )

    for item in raw_detections:
        detections.append({
            "label": item.get("label") or item.get("class") or item.get("name") or "unknown",
            "confidence": item.get("confidence") or item.get("conf"),
            "distance_m": item.get("distance_m") or item.get("distance") or item.get("front"),
            "position": item.get("position") or item.get("location"),
            "tracking": item.get("tracking", False),
        })

    return detections
