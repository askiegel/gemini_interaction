from pathlib import Path
from unittest.mock import patch

from voice_relay.server import VoiceRelayHandler


ROOT = Path(__file__).resolve().parent
HTML = (
    ROOT / "voice_relay" / "index.html"
).read_text(encoding="utf-8")
CSS = (
    ROOT / "voice_relay" / "operator_console.css"
).read_text(encoding="utf-8")
JS = (
    ROOT / "voice_relay" / "operator_console.js"
).read_text(encoding="utf-8")
SERVER = (
    ROOT / "voice_relay" / "server.py"
).read_text(encoding="utf-8")

REVIEW_START = JS.index(
    "/* Read-only candidate map review */"
)
REVIEW_END = JS.index(
    "/* Read-only localized LiDAR map overlay */"
)
REVIEW = JS[REVIEW_START:REVIEW_END]

PROMOTION_START = HTML.index(
    "/* Guarded candidate map promotion */"
)
PROMOTION_END = HTML.index(
    "</script>",
    PROMOTION_START,
)
PROMOTION = HTML[
    PROMOTION_START:PROMOTION_END
]


def test_promotion_workspace_exists():
    assert 'id="promoteCandidateButton"' in HTML
    assert 'id="candidatePromotionState"' in HTML
    assert 'id="candidatePromotionMessage"' in HTML
    assert "Validated Map Replacement" in HTML


def test_promotion_controller_is_isolated():
    assert (
        "/* Guarded candidate map promotion */"
        in HTML
    )
    assert (
        "/* Guarded candidate map promotion */"
        not in JS
    )


def test_candidate_review_remains_get_only():
    assert 'fetch(ENDPOINT' in REVIEW

    for marker in (
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "promote",
        "PROMOTION_ENDPOINT",
    ):
        assert marker not in REVIEW


def test_selection_is_shared_without_mutation():
    assert (
        "list.dataset.selectedCandidateName"
        in REVIEW
    )
    assert (
        "inventory.dataset.selectedCandidateName"
        in PROMOTION
    )


def test_selected_candidate_drives_promotion():
    assert "selectedCandidate()" in PROMOTION
    assert (
        "candidateIsReviewReady(candidate)"
        in PROMOTION
    )
    assert "candidate.name" in PROMOTION


def test_current_validated_candidate_is_blocked():
    assert "candidateMatchesValidated" in PROMOTION
    assert (
        "comparison.cell_count_delta"
        in PROMOTION
    )
    assert (
        '"Already the Validated Map"'
        in PROMOTION
    )


def test_active_runtime_blocks_promotion():
    assert "MAPPING_STATUS_ENDPOINT" in PROMOTION
    assert (
        "LOCALIZATION_STATUS_ENDPOINT"
        in PROMOTION
    )
    assert (
        "mapping.running === false"
        in PROMOTION
    )
    assert (
        "localization.running === false"
        in PROMOTION
    )


def test_explicit_confirmation_is_required():
    assert "window.confirm(" in PROMOTION
    assert "window.prompt(" in PROMOTION
    assert (
        '"PROMOTE REVIEWED CANDIDATE"'
        in PROMOTION
    )


def test_browser_posts_guarded_payload():
    assert 'method: "POST"' in PROMOTION
    assert (
        "candidate_name: candidate.name"
        in PROMOTION
    )
    assert "confirmation," in PROMOTION


def test_success_exposes_backup_and_refreshes():
    assert (
        "promotion.backup_directory"
        in PROMOTION
    )
    assert (
        "window.location.reload()"
        in PROMOTION
    )


def test_server_defines_guarded_proxy():
    assert (
        "def candidate_map_promote(self, payload):"
        in SERVER
    )
    assert (
        'f"{ROBOT_BRIDGE_URL}/map/promote-candidate"'
        in SERVER
    )
    assert (
        'path == "/dashboard/map-promote-candidate"'
        in SERVER
    )


def test_server_forwards_exact_payload():
    handler = object.__new__(VoiceRelayHandler)
    payload = {
        "candidate_name": (
            "mayday_map_candidate_20260809T010000Z"
        ),
        "confirmation": (
            "PROMOTE REVIEWED CANDIDATE"
        ),
    }
    robot_payload = {
        "ok": True,
        "promotion": {"promoted": True},
    }

    with patch(
        "voice_relay.server.request_json",
        return_value={
            "ok": True,
            "status_code": 201,
            "data": robot_payload,
            "error": None,
        },
    ) as request:
        status, result = (
            handler.candidate_map_promote(payload)
        )

    assert status == 201
    assert result == robot_payload

    request.assert_called_once_with(
        "POST",
        (
            "http://minipupperv2.local:8090"
            "/map/promote-candidate"
        ),
        payload=payload,
        timeout=180.0,
    )


def test_wrong_confirmation_is_not_forwarded():
    handler = object.__new__(VoiceRelayHandler)

    with patch(
        "voice_relay.server.request_json"
    ) as request:
        status, result = (
            handler.candidate_map_promote({
                "candidate_name": (
                    "mayday_map_candidate_"
                    "20260809T010000Z"
                ),
                "confirmation": "wrong",
            })
        )

    assert status == 400
    assert result["ok"] is False
    request.assert_not_called()


def test_guarded_styles_are_present():
    assert ".candidate-promotion" in CSS
    assert (
        ".candidate-promotion-button:disabled"
        in CSS
    )
    assert (
        ".candidate-promotion-state.ready"
        in CSS
    )
    assert (
        ".candidate-promotion-state.error"
        in CSS
    )


def test_navigation_and_motion_are_absent():
    for marker in (
        "/cmd_vel",
        "NavigateToPose",
        "navigation_goal",
        "controller_server",
        "bt_navigator",
    ):
        assert marker not in PROMOTION
