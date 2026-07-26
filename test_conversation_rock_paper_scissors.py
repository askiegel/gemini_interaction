#!/usr/bin/env python3

from conversation_manager import ConversationManager
from rock_paper_scissors_service import RockPaperScissorsGame


class ForbiddenProvider:
    def __init__(self):
        self.calls = 0

    def get_conversation_decision(self, user_text, history):
        self.calls += 1
        raise AssertionError(
            "Rock Paper Scissors must not call the AI provider."
        )


def main():
    provider = ForbiddenProvider()
    game = RockPaperScissorsGame(chooser=lambda: "scissors")
    manager = ConversationManager(
        provider=provider,
        rock_paper_scissors_game=game,
    )

    started = manager.process("Let's play rock paper scissors")

    assert started.decision_type == "CONVERSATION"
    assert started.has_mission is False
    assert started.reply == (
        "Let's play rock, paper, scissors. "
        "Choose rock, paper, or scissors."
    )
    print("PASS: game starts as deterministic conversation")

    completed = manager.process("I choose rock")

    assert completed.reply == (
        "You chose rock. I chose scissors. You win."
    )
    assert completed.has_mission is False
    assert provider.calls == 0
    print("PASS: completed game bypasses the AI provider")
    print("PASS: game creates no robot mission")

    history = manager.get_history()
    assert [turn["role"] for turn in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    print("PASS: both game turns enter conversation history")

    manager.process("play rock paper scissors")
    assert game.waiting_for_move is True
    manager.clear_history()
    assert game.waiting_for_move is False
    assert manager.get_history() == []
    print("PASS: clearing conversation also resets game state")

    print()
    print("Rock Paper Scissors conversation test passed.")


if __name__ == "__main__":
    main()
