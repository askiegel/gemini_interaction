"""Offline tests for the advisory local-obstacle decision policy."""

import copy

import pytest

from local_obstacle_policy import recommend_local_avoidance


def snapshot(*, front="CLEAR", left="CLEAR", right="CLEAR",
             left_clearance=1.0, right_clearance=1.0, session="session-1",
             age=0.05, now=10.0):
    def sector(state, clearance):
        return {
            "state": state,
            "available": True,
            "robust_clearance_m": clearance,
            "minimum_clearance_m": clearance,
        }

    return {
        "available": True,
        "valid": True,
        "reason": "fresh",
        "producer_session": session,
        "received_monotonic_seconds": now,
        "age_at_receipt_seconds": age,
        "effective_age_seconds": age,
        "sectors": {
            "front": sector(front, 0.8),
            "front_left": sector(left, left_clearance),
            "front_right": sector(right, right_clearance),
            "left": sector("CLEAR", 1.0),
            "right": sector("CLEAR", 1.0),
        },
    }


def choose(state, **kwargs):
    return recommend_local_avoidance(state, expected_session="session-1", now=10.0, **kwargs)


def test_fresh_clear_front_recommends_forward_advisory():
    result = choose(snapshot())
    assert result["recommendation"] == "FORWARD"
    assert result["trusted"] is True
    assert result["fresh"] is True
    assert result["reason"] == "front_clear_advisory"


def test_blocked_front_prefers_clear_left():
    result = choose(snapshot(front="CAUTION", left="CLEAR", right="BLOCKED"))
    assert result["recommendation"] == "TURN_LEFT"


def test_blocked_front_prefers_clear_right():
    result = choose(snapshot(front="BLOCKED", left="BLOCKED", right="CLEAR"))
    assert result["recommendation"] == "TURN_RIGHT"


def test_both_clear_sides_choose_greater_robust_clearance():
    left = choose(snapshot(front="CAUTION", left_clearance=1.4, right_clearance=0.9))
    right = choose(snapshot(front="CAUTION", left_clearance=0.9, right_clearance=1.4))
    assert left["recommendation"] == "TURN_LEFT"
    assert right["recommendation"] == "TURN_RIGHT"


def test_near_tie_uses_deterministic_left_preference():
    first = choose(snapshot(front="CAUTION", left_clearance=1.00, right_clearance=0.96))
    second = choose(snapshot(front="CAUTION", left_clearance=0.96, right_clearance=1.00))
    assert first["recommendation"] == "TURN_LEFT"
    assert second["recommendation"] == "TURN_LEFT"


def test_both_sides_unsafe_hold():
    result = choose(snapshot(front="CAUTION", left="CAUTION", right="BLOCKED"))
    assert result["recommendation"] == "HOLD"


@pytest.mark.parametrize("state", [
    None,
    {},
    {"available": True, "valid": True, "reason": "fresh", "producer_session": "session-1", "sectors": []},
])
def test_malformed_or_missing_state_holds(state):
    result = choose(state)
    assert result["recommendation"] == "HOLD"
    assert result["trusted"] is False


def test_unknown_front_holds():
    result = choose(snapshot(front="UNKNOWN"))
    assert result["recommendation"] == "HOLD"


def test_untrusted_front_sector_that_claims_clear_holds():
    state = snapshot()
    state["sectors"]["front"]["available"] = False
    result = choose(state)
    assert result["recommendation"] == "HOLD"


@pytest.mark.parametrize("minimum_clearance", [None, float("nan"), float("inf")])
def test_front_clear_with_invalid_minimum_clearance_holds(minimum_clearance):
    state = snapshot()
    state["sectors"]["front"]["minimum_clearance_m"] = minimum_clearance
    assert choose(state)["recommendation"] == "HOLD"


def test_malformed_clear_left_does_not_win_over_unsafe_right():
    state = snapshot(front="CAUTION", left="CLEAR", right="BLOCKED")
    state["sectors"]["front_left"]["minimum_clearance_m"] = None
    assert choose(state)["recommendation"] == "HOLD"


def test_valid_clear_side_wins_over_malformed_clear_side():
    state = snapshot(front="BLOCKED", left="CLEAR", right="CLEAR")
    state["sectors"]["front_right"]["minimum_clearance_m"] = float("nan")
    assert choose(state)["recommendation"] == "TURN_LEFT"


@pytest.mark.parametrize("field,value", [
    ("robust_clearance_m", None),
    ("robust_clearance_m", float("nan")),
    ("robust_clearance_m", True),
    ("minimum_clearance_m", True),
])
def test_boolean_or_invalid_clearance_metrics_are_not_trusted(field, value):
    state = snapshot()
    state["sectors"]["front"][field] = value
    assert choose(state)["recommendation"] == "HOLD"


def test_stale_state_holds():
    state = snapshot(age=0.31)
    result = recommend_local_avoidance(state, expected_session="session-1", now=10.0)
    assert result["recommendation"] == "HOLD"
    assert result["reason"] == "stale"
    assert result["trusted"] is False
    assert result["fresh"] is False


def test_invalid_and_unavailable_state_holds():
    for state in (snapshot(), snapshot()):
        state["valid"] = False
        state["available"] = False
        state["reason"] = "offline"
        assert choose(state)["recommendation"] == "HOLD"


def test_producer_session_mismatch_holds():
    result = recommend_local_avoidance(snapshot(session="other"), expected_session="session-1", now=10.0)
    assert result["recommendation"] == "HOLD"
    assert result["reason"] == "producer_session_mismatch"


def test_status_exposes_source_and_clearances_without_mutating_input():
    state = snapshot(front="CAUTION", left_clearance=1.2, right_clearance=0.8)
    original = copy.deepcopy(state)
    result = choose(state)
    assert result["producer_session"] == "session-1"
    assert result["effective_age_seconds"] == pytest.approx(0.05)
    assert result["front_state"] == "CAUTION"
    assert result["front_left_robust_clearance_m"] == pytest.approx(1.2)
    assert result["front_right_robust_clearance_m"] == pytest.approx(0.8)
    assert state == original
