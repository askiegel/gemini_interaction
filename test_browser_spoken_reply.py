#!/usr/bin/env python3

from pathlib import Path


def require(text, token, message):
    if token not in text:
        raise AssertionError(message)

    print(f"PASS: {message}")


def main():
    html = Path("voice_relay/index.html").read_text()

    require(
        html,
        'id="robotReply"',
        "Mission Control contains a visible robot reply region",
    )
    require(
        html,
        'aria-live="polite"',
        "robot reply is announced accessibly",
    )
    require(
        html,
        'function speakReply(reply)',
        "spoken reply function is defined",
    )
    require(
        html,
        'new SpeechSynthesisUtterance(',
        "browser speech utterance is constructed",
    )
    require(
        html,
        'window.speechSynthesis.cancel();',
        "previous browser speech is cancelled",
    )
    require(
        html,
        'window.speechSynthesis.speak(utterance);',
        "browser speaks the conversational reply",
    )
    require(
        html,
        'speakReply(result.reply);',
        "successful conversation reply reaches speech output",
    )
    require(
        html,
        'if (!response.ok || result.ok === false)',
        "server failures are rejected before speech output",
    )

    response_check = html.index(
        'if (!response.ok || result.ok === false)'
    )
    speech_call = html.index(
        'speakReply(result.reply);'
    )

    if response_check >= speech_call:
        raise AssertionError(
            "Conversation response must be validated before speaking."
        )

    print(
        "PASS: response validation occurs before speech output"
    )
    print()
    print("Browser spoken conversation reply test passed.")


if __name__ == "__main__":
    main()
