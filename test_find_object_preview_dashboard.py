"""Focused source regression checks for Preview Target overlay precedence."""
from pathlib import Path


HTML = Path("voice_relay/index.html").read_text(encoding="utf-8")
OPERATOR_JS = Path("voice_relay/operator_console.js").read_text(encoding="utf-8")
OPERATOR_CSS = Path("voice_relay/operator_console.css").read_text(encoding="utf-8")


def test_find_object_preview_defaults_to_marvin_and_keeps_generic_input():
    assert 'id="findObjectPreviewTarget"' in HTML
    assert 'id="findObjectPreviewTarget"\n                    type="text"\n                    value="marvin"' in HTML
    assert '"/dashboard/find-object-preview?target="' in HTML
    assert "encodeURIComponent(target)" in HTML
    assert 'value="backpack"' not in HTML


def test_successful_preview_tracking_is_retained_and_sent_to_overlay():
    assert "findObjectPreviewTracking = result.tracking || null;" in HTML
    assert "findObjectPreviewTracking ||\n                        (statusFindObjectTracking ? {} : tracking)" in HTML
    assert "drawTrackingOverlay(\n                detections,\n                activeTarget,\n                displayTracking" in HTML


def test_inactive_find_object_status_cannot_clear_preview_tracking():
    real_tracking = HTML.index("const realFindObjectTracking =")
    clears_preview = HTML.index("findObjectPreviewTracking = null;", real_tracking)
    active_status = HTML.index(
        'String(missions.active.status || "").toUpperCase() ===\n                    "ACTIVE"',
        real_tracking,
    )
    assert active_status < clears_preview
    assert "findObjectPreviewTracking ||\n                        (statusFindObjectTracking ? {} : tracking)" in HTML


def test_active_live_find_object_supersedes_preview_tracking():
    assert (
        'String(missions.active.mission_type || "").toUpperCase() ===\n'
        '                    "FIND_OBJECT" &&\n'
        '                String(missions.active.status || "").toUpperCase() ===\n'
        '                    "ACTIVE"'
    ) in HTML
    assert "if (realFindObjectTracking) {\n                findObjectPreviewTracking = null;" in HTML
    assert 'String(active.target || "").toLowerCase() === "marvin"' in HTML


def test_preview_bbox_uses_existing_find_object_target_box_style():
    assert 'String(tracking.state || "").toUpperCase() ===\n                    "PREVIEW"' in HTML
    assert '"PREVIEW "' in HTML
    assert "context.strokeStyle = \"#f97316\";" in HTML
    assert "context.strokeRect(x1, y1, x2 - x1, y2 - y1);" in HTML


def test_marvin_preview_overlay_labels_and_draws_opencv_tracker_source():
    assert 'String(tracking.source || "") ===\n                                "marvin_local_tracker"' in HTML
    assert '"PREVIEW OPENCV "' in HTML
    assert "const targetBbox =" in HTML


def test_normal_detection_overlay_and_clear_preview_remain_unchanged():
    assert "detections.forEach((detection) =>" in HTML
    assert "function clearFindObjectPreview() {\n        findObjectPreviewTracking = null;" in HTML
    assert "statusFindObjectTracking ? {} : tracking" in HTML
    assert "clearFindObjectPreview\n    );" in HTML


def test_camera_has_no_large_mission_state_banner_but_keeps_geometry_and_status():
    assert "operatorCameraBanner" not in OPERATOR_JS
    assert ".operator-camera-banner" not in OPERATOR_CSS
    assert "operatorLeftArrow" in OPERATOR_JS
    assert "operatorRightArrow" in OPERATOR_JS
    assert "operatorCentered" in OPERATOR_JS
    assert 'id="cameraOverlay"' in HTML
    assert 'id="findMarvinMission"' in HTML
    assert 'id="findMarvinProgress"' in HTML
    assert '"PREVIEW OPENCV "' in HTML
    assert "context.strokeRect(x1, y1, x2 - x1, y2 - y1);" in HTML
    assert "centerX - tolerance" in HTML
    assert "centerX + tolerance" in HTML


def test_marvin_controls_and_guarded_request_contract_remain_present():
    for element_id in (
        "findObjectPreviewButton",
        "marvinOneStepTestButton",
        "marvinCenteringStepButton",
        "marvinGuardedApproachButton",
    ):
        assert f'id="{element_id}"' in HTML
    feature = HTML.split(
        "elements.marvinGuardedApproachButton.addEventListener", 1
    )[1].split("document.querySelectorAll", 1)[0]
    assert 'body: JSON.stringify({execute: isLiveModeEnabled()})' in feature
