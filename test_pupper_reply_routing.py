#!/usr/bin/env python3

from voice_relay.server import (
    set_robot_speech_submitter_for_testing,
    submit_robot_speech,
)


def main():
    calls = []

    def successful_submitter(reply):
        calls.append(reply)

        return {
            "ok": True,
            "action": "speak",
        }

    set_robot_speech_submitter_for_testing(
        successful_submitter
    )

    delivered = submit_robot_speech(
        "The answer is 391."
    )

    assert delivered["ok"] is True
    assert delivered["destination"] == "mini_pupper"
    assert delivered["fallback_required"] is False
    assert calls == ["The answer is 391."]
    print("PASS: validated reply reaches Pupper exactly once")
    print("PASS: successful Pupper speech disables fallback")

    def failed_submitter(reply):
        return {
            "ok": False,
            "error": "speaker unavailable",
        }

    set_robot_speech_submitter_for_testing(
        failed_submitter
    )

    failed = submit_robot_speech("Hello")

    assert failed["ok"] is False
    assert failed["fallback_required"] is True
    assert failed["error"] == "speaker unavailable"
    print("PASS: speech failure requests browser fallback")

    def raising_submitter(reply):
        raise TimeoutError("speech timed out")

    set_robot_speech_submitter_for_testing(
        raising_submitter
    )

    raised = submit_robot_speech("Hello")

    assert raised["ok"] is False
    assert raised["fallback_required"] is True
    assert "timed out" in raised["error"]
    print("PASS: speech exception preserves browser fallback")

    empty = submit_robot_speech("   ")

    assert empty["ok"] is False
    assert empty["skipped"] is True
    assert empty["fallback_required"] is False
    print("PASS: empty replies are not sent or spoken")

    set_robot_speech_submitter_for_testing(None)

    print()
    print("Pupper reply routing test passed.")


if __name__ == "__main__":
    main()
