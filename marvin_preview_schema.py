"""Pure normalization of the public Marvin Preview representation.

The runtime Preview API deliberately keeps dashboard tracking geometry under
``tracking``.  Policies consume this helper rather than assuming that those
fields were duplicated at the top level.
"""

import math


def normalize_marvin_preview(value):
    """Return canonical current Marvin Preview fields, without mutation.

    A result is returned only for the established Marvin local-tracker Preview
    representation.  ``marvin_continuity`` is retained for diagnostics but is
    never used to create or select an identity.
    """
    if not isinstance(value, dict):
        return None
    tracking = value.get("tracking")
    if not isinstance(tracking, dict):
        tracking = {}
    bbox = _first_dict(value.get("bbox"), tracking.get("bbox"))
    width = _first_number(value.get("image_width"), tracking.get("image_width"))
    height = _first_number(value.get("image_height"), tracking.get("image_height"))
    continuity = _first_dict(
        value.get("marvin_continuity"), tracking.get("marvin_continuity"),
    )
    return {
        "ok": value.get("ok") is True,
        "preview": value.get("preview") is True,
        "authoritative": value.get("authoritative"),
        "target": str(value.get("target") or "").strip().lower(),
        # Public Preview's active Marvin tracking state is the canonical
        # found signal when top-level target_found is intentionally omitted.
        "target_found": (
            value.get("target_found") is True
            or (
                value.get("target_found") is None
                and tracking.get("active") is True
                and str(tracking.get("target_label") or "").strip().lower()
                == "marvin"
            )
        ),
        "identity_confirmed": value.get("identity_confirmed") is True,
        "source": value.get("source") or tracking.get("source"),
        "source_timestamp": value.get("source_timestamp") or tracking.get("vision_timestamp"),
        "vision_timestamp": value.get("vision_timestamp") or tracking.get("vision_timestamp"),
        "bbox": dict(bbox) if isinstance(bbox, dict) else None,
        "image_width": width,
        "image_height": height,
        "horizontal_error": _first_number(
            value.get("horizontal_error"), tracking.get("horizontal_error"),
        ),
        "ambiguous": _ambiguous(value, tracking),
        "marvin_continuity": dict(continuity) if isinstance(continuity, dict) else None,
    }


def _ambiguous(value, tracking):
    for item in (value, tracking, value.get("target_observation"), value.get("selected_proposal")):
        if isinstance(item, dict) and (
            item.get("identity_ambiguous") is True
            or item.get("ambiguous") is True
            or str(item.get("identity_status") or "").strip().upper() == "AMBIGUOUS"
        ):
            return True
    return False


def _first_dict(*values):
    return next((value for value in values if isinstance(value, dict)), None)


def _first_number(*values):
    return next((value for value in values if _number(value)), None)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
