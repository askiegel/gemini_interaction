#!/usr/bin/env python3

from rock_paper_scissors_service import RockPaperScissorsGame


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected!r}\nActual: {actual!r}"
        )

    print(f"PASS: {message}")


def main():
    game = RockPaperScissorsGame(chooser=lambda: "scissors")

    assert_equal(
        game.process("Let's play rock paper scissors"),
        (
            "Let's play rock, paper, scissors. "
            "Choose rock, paper, or scissors."
        ),
        "game invitation starts a pending round",
    )
    assert game.waiting_for_move is True
    print("PASS: game waits for the human move")

    assert_equal(
        game.process("I choose rock"),
        "You chose rock. I chose scissors. You win.",
        "rock beats scissors",
    )
    assert game.waiting_for_move is False
    print("PASS: completed round clears pending state")

    game = RockPaperScissorsGame(chooser=lambda: "rock")
    game.process("play rock paper scissors")
    assert_equal(
        game.process("scissors"),
        "You chose scissors. I chose rock. I win.",
        "rock beats scissors for the robot",
    )

    game = RockPaperScissorsGame(chooser=lambda: "paper")
    assert_equal(
        game.process(
            "Let's play rock paper scissors, I choose paper"
        ),
        "You chose paper. I chose paper. It's a tie.",
        "a start request can include the human move",
    )

    assert_equal(
        game.process("Tell me about rock formations"),
        None,
        "ordinary uses of rock do not start a game",
    )

    invalid = RockPaperScissorsGame(chooser=lambda: "banana")
    invalid.process("play rock paper scissors")

    try:
        invalid.process("paper")
    except ValueError:
        print("PASS: invalid injected robot move is rejected")
    else:
        raise AssertionError("Invalid robot move was accepted.")

    print()
    print("Rock Paper Scissors service test passed.")


if __name__ == "__main__":
    main()
