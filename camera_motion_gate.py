from datetime import datetime, timezone


def evaluate_camera_gate(payload, now=None):
    now = now or datetime.now(timezone.utc)
    if not isinstance(payload, dict) or payload.get("camera_running") is not True:
        return {"camera_semantic_clear": False, "reason": "camera_unavailable"}
    stamp = payload.get("timestamp")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None: raise ValueError()
    except Exception:
        return {"camera_semantic_clear": False, "reason": "camera_timestamp_invalid"}
    if (now - parsed).total_seconds() > 1.0:
        return {"camera_semantic_clear": False, "reason": "camera_stale"}
    detections = payload.get("detections")
    if not isinstance(detections, list):
        return {"camera_semantic_clear": False, "reason": "camera_detections_invalid"}
    for detection in detections:
        if not isinstance(detection, dict):
            return {"camera_semantic_clear": False, "reason": "camera_detection_invalid"}
        label=str(detection.get("label", detection.get("class", ""))).strip().lower()
        try: confidence=float(detection.get("confidence", 0.0))
        except (TypeError, ValueError): confidence=0.0
        if label not in ("person", "human") or confidence < .25: continue
        width=detection.get("image_width", payload.get("image_width", payload.get("frame_width", payload.get("width"))))
        try: width=float(width)
        except (TypeError, ValueError): width=0.0
        center=detection.get("center_x")
        if center is None and all(key in detection for key in ("x1", "x2")): center=(float(detection["x1"])+float(detection["x2"]))/2
        bbox=detection.get("bbox")
        if center is None and isinstance(bbox, dict):
            left=bbox.get("x1",bbox.get("left")); right=bbox.get("x2",bbox.get("right"))
            if left is not None and right is not None: center=(float(left)+float(right))/2
        if center is None or width <= 0: return {"camera_semantic_clear": False, "reason": "central_person_geometry_unavailable"}
        if .25 <= float(center)/width <= .75: return {"camera_semantic_clear": False, "reason": "central_person_detected"}
    return {"camera_semantic_clear": True, "reason": "camera_semantic_clear"}
