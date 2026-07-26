#!/usr/bin/env python3

import random
import re
from typing import Callable, Optional


MOVES = ("rock", "paper", "scissors")

WINNING_MOVE = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


class RockPaperScissorsGame:
    """Deterministic game rules with an injectable robot move chooser."""

    def __init__(
        self,
        chooser: Optional[Callable[[], str]] = None,
    ):
        self._chooser = chooser or (
            lambda: random.choice(MOVES)
        )
        self._waiting_for_move = False

    @property
    def waiting_for_move(self) -> bool:
        return self._waiting_for_move

    def reset(self) -> None:
        self._waiting_for_move = False

    def process(self, user_text: str) -> Optional[str]:
        normalized = " ".join(
            str(user_text or "").strip().lower().split()
        )

        if not normalized:
            return None

        start_requested = bool(
            re.search(
                r"\b(?:play|start)\b.*\b"
                r"rock\s*,?\s*paper\s*,?\s*(?:or\s+)?scissors\b",
                normalized,
            )
            or re.search(
                r"\brock\s*,?\s*paper\s*,?\s*scissors\b"
                r".*\b(?:game|play)\b",
                normalized,
            )
        )

        move = self._extract_move(normalized)

        if start_requested and move is None:
            self._waiting_for_move = True
            return (
                "Let's play rock, paper, scissors. "
                "Choose rock, paper, or scissors."
            )

        if not self._waiting_for_move and not start_requested:
            return None

        if move is None:
            self._waiting_for_move = True
            return "Choose rock, paper, or scissors."

        robot_move = str(self._chooser() or "").strip().lower()

        if robot_move not in MOVES:
            raise ValueError(
                "RockPaperScissorsGame chooser returned "
                f"an invalid move: {robot_move!r}"
            )

        self._waiting_for_move = False

        if move == robot_move:
            outcome = "It's a tie."
        elif WINNING_MOVE[move] == robot_move:
            outcome = "You win."
        else:
            outcome = "I win."

        return (
            f"You chose {move}. I chose {robot_move}. "
            f"{outcome}"
        )

    @staticmethod
    def _extract_move(user_text: str) -> Optional[str]:
        direct = re.fullmatch(
            r"(?:i\s+(?:choose|pick|chose|picked)\s+)?"
            r"(rock|paper|scissors)[.!]?",
            user_text,
        )

        if direct:
            return direct.group(1)

        selected = re.search(
            r"\b(?:i\s+)?(?:choose|pick|chose|picked)\s+"
            r"(rock|paper|scissors)\b",
            user_text,
        )

        if selected:
            return selected.group(1)

        return None
