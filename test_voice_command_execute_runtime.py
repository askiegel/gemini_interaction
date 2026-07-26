#!/usr/bin/env python3

from unittest.mock import patch

import voice_command


class FakeProvider:
    def get_intent(self, user_text):
        if user_text != "follow me":
            raise AssertionError(
                f"Unexpected provider text: {user_text!r}"
            )

        return {
            "intent": "FOLLOW_PERSON",
            "speech": "Okay, I will follow you.",
            "target": "person",
        }


def fake_runtime_submission(**kwargs):
    if kwargs["user_text"] != "Follow me":
        raise AssertionError(
            "Original source text was not preserved."
        )

    if kwargs["intent"]["intent"] != "FOLLOW_PERSON":
        raise AssertionError(
            "Parsed intent was not submitted."
        )

    return {
        "ok": True,
        "accepted": True,
        "submission_mode": "parsed_intent",
        "mission": {
            "mission_id": "mission-execute-runtime-test",
            "mission_type": "FOLLOW_PERSON",
            "status": "ACTIVE",
            "target": "person",
        },
    }


def forbidden_vision_adapter(*args, **kwargs):
    del args
    del kwargs

    raise AssertionError(
        "Legacy --execute constructed a second VisionAdapter."
    )


def main():
    with (
        patch(
            "voice_command.create_provider",
            return_value=FakeProvider(),
        ),
        patch(
            "voice_command.submit_intent_to_runtime",
            side_effect=fake_runtime_submission,
        ) as runtime_submit,
        patch(
            "voice_command.VisionAdapter",
            side_effect=forbidden_vision_adapter,
        ),
    ):
        result = voice_command.run_command(
            user_text="Follow me",
            execute=True,
            submit_runtime=False,
            runtime_url="http://runtime.test:8770",
        )

    if not result.get("accepted"):
        raise AssertionError(
            "Runtime did not accept the legacy execute command."
        )

    if runtime_submit.call_count != 1:
        raise AssertionError(
            "Legacy execute did not submit exactly one mission "
            "to the persistent runtime."
        )

    call = runtime_submit.call_args.kwargs

    if call["runtime_url"] != "http://runtime.test:8770":
        raise AssertionError(
            "Configured runtime URL was not preserved."
        )

    print(
        "PASS: Legacy execute submitted through the "
        "persistent runtime."
    )
    print(
        "PASS: Legacy execute did not construct VisionAdapter."
    )
    print(
        "PASS: Command parsing and addressing metadata "
        "were preserved."
    )


if __name__ == "__main__":
    main()
