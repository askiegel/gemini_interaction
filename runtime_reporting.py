"""Small, read-only API projections; never traverse retained action evidence."""


def scalar_fields(value):
    if not isinstance(value, dict):
        return {}
    return {
        key: item[:2048] if isinstance(item, str) else item
        for key, item in value.items()
        if item is None or type(item) in (str, bool, int, float)
    }


def result_summary(value):
    if not isinstance(value, dict):
        return None
    result = scalar_fields(value)
    # Some legacy results wrap the behavior's result. Keep its small summary.
    if isinstance(value.get("result"), dict):
        result["result"] = scalar_fields(value["result"])
    result["diagnostics_available"] = isinstance(value.get("progress_diagnostics"), dict)
    return result


def status_summary(value):
    """Compatibility projection for injected runtimes without a summary getter."""
    result = scalar_fields(value)
    result["active_mission"] = scalar_fields(value.get("active_mission")) or None
    queue = value.get("queue") or []
    result["queue"] = [scalar_fields(mission) for mission in queue[:20]]
    result["queue_count"] = value.get("queue_count", len(queue))
    result["last_result"] = result_summary(value.get("last_result"))
    tracking = value.get("tracking") or {}
    result["tracking"] = scalar_fields(tracking)
    if isinstance(tracking.get("bbox"), dict):
        result["tracking"]["bbox"] = scalar_fields(tracking["bbox"])
    for key in ("lidar_perception", "forward_interlock"):
        result[key] = scalar_fields(value.get(key))
    if isinstance(value.get("navigation_shadow"), dict):
        # Small diagnostics only; no certificate/action authority is exposed.
        result["navigation_shadow"] = scalar_fields(value["navigation_shadow"])
    return result
