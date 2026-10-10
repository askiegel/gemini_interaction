"""Bounded image coverage for semantic candidate selection, never authority."""


def marvin_coverage_candidates(width, height, source_frame_stamp_ns):
    """Return four overlapping edge windows and twelve small interior windows.

    Small windows approximate historical Marvin seed sizes. The larger edge
    windows cover their gaps and objects straddling a small-window boundary.
    Coordinates/order depend only on dimensions, never detections or identity.
    """
    if (type(width) is not int or type(height) is not int
            or width < 64 or height < 64
            or type(source_frame_stamp_ns) is not int
            or source_frame_stamp_ns < 0):
        raise ValueError("marvin_coverage_frame_invalid")
    boxes = []
    large_w = max(8, int(round(width * 0.60)))
    large_h = min(height, max(8, int(round(height * 0.70)),
                              (large_w * 4 + 4) // 5))
    for y in (0, height - large_h):
        for x in (0, width - large_w):
            boxes.append((x, y, x + large_w, y + large_h))
    small_w = max(8, int(round(width * 5 / 32)))
    small_h = min(height, max(8, int(round(height * 0.30)),
                              (small_w * 4 + 4) // 5))
    for row in range(1, 4):
        for column in range(1, 5):
            x = max(0, min(width - small_w,
                           int(round(width * column / 5 - small_w / 2))))
            y = max(0, min(height - small_h,
                           int(round(height * row / 4 - small_h / 2))))
            boxes.append((x, y, x + small_w, y + small_h))
    return [{
        "bbox": dict(zip(("x1", "y1", "x2", "y2"), box)),
        "image_width": width, "image_height": height,
        "source_frame_stamp_ns": source_frame_stamp_ns,
        "geometry_source": "deterministic_coverage",
    } for box in boxes]
