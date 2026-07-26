#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from arithmetic_service import answer_arithmetic_question
from symbolic_math_service import answer_symbolic_math_question


ALLOWED_MISSION_TYPES = {
    "FOLLOW_PERSON",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "DESCRIBE_SCENE",
    "FIND_OBJECT",
    "RETURN_HOME",
}

ALLOWED_DECISION_TYPES = {
    "CONVERSATION",
    "MISSION",
    "CLARIFICATION",
    "WORLD_QUERY",
}


ALLOWED_WORLD_QUERY_TYPES = {
    "LATEST_ENTITY",
    "LIST_ENTITIES",
    "CURRENT_MISSION",
    "VISION_STATUS",
}


class ConversationError(RuntimeError):
    """Raised when a conversational provider returns an invalid decision."""


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationResult:
    reply: str
    decision_type: str = "CONVERSATION"
    mission_type: Optional[str] = None
    query_type: Optional[str] = None
    target: Optional[str] = None
    requires_confirmation: bool = False

    @property
    def has_mission(self) -> bool:
        return self.mission_type is not None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationManager:
    """
    Provider-independent conversational decision manager.

    The manager accepts user text and returns a validated conversational
    decision. It never controls motors, submits missions, accesses ROS,
    records audio, or speaks through a text-to-speech system.

    Conversational providers must implement:

        get_conversation_decision(user_text, history)

    and return a dictionary with this shape:

        {
            "reply": "Sure, I'll follow you.",
            "decision_type": "MISSION",
            "mission_type": "FOLLOW_PERSON",
            "target": "person",
            "requires_confirmation": False,
        }

    A mission is optional. Every valid decision must contain a spoken reply.
    """

    def __init__(
        self,
        provider,
        max_history_turns: int = 12,
    ):
        if provider is None:
            raise ValueError("ConversationManager requires a provider.")

        if max_history_turns < 0:
            raise ValueError("max_history_turns cannot be negative.")

        self.provider = provider
        self.max_history_turns = int(max_history_turns)
        self._history: List[ConversationTurn] = []

    def process(self, user_text: str) -> ConversationResult:
        normalized_text = self._normalize_user_text(user_text)

        provider_method = getattr(
            self.provider,
            "get_conversation_decision",
            None,
        )

        if not callable(provider_method):
            raise ConversationError(
                "The provider does not implement "
                "get_conversation_decision(user_text, history)."
            )

        symbolic_reply = answer_symbolic_math_question(
            normalized_text
        )

        if symbolic_reply is not None:
            result = ConversationResult(
                reply=symbolic_reply,
                decision_type="CONVERSATION",
            )

            self._append_turn("user", normalized_text)
            self._append_turn("assistant", result.reply)

            return result

        arithmetic_reply = answer_arithmetic_question(
            normalized_text
        )

        if arithmetic_reply is not None:
            result = ConversationResult(
                reply=arithmetic_reply,
                decision_type="CONVERSATION",
            )

            self._append_turn("user", normalized_text)
            self._append_turn("assistant", result.reply)

            return result

        history_snapshot = self.get_history()

        try:
            raw_decision = provider_method(
                normalized_text,
                history_snapshot,
            )
        except ConversationError:
            raise
        except Exception as exc:
            raise ConversationError(
                f"Conversational provider failed: {exc}"
            ) from exc

        result = self._validate_decision(raw_decision)

        self._append_turn("user", normalized_text)
        self._append_turn("assistant", result.reply)

        return result

    def get_history(self) -> List[Dict[str, str]]:
        return [turn.to_dict() for turn in self._history]

    def clear_history(self) -> None:
        self._history.clear()

    def _append_turn(self, role: str, text: str) -> None:
        if self.max_history_turns == 0:
            return

        self._history.append(
            ConversationTurn(
                role=role,
                text=text,
            )
        )

        maximum_entries = self.max_history_turns * 2

        if len(self._history) > maximum_entries:
            self._history = self._history[-maximum_entries:]

    @staticmethod
    def _normalize_user_text(user_text: str) -> str:
        if not isinstance(user_text, str):
            raise ConversationError("User text must be a string.")

        normalized = user_text.strip()

        if not normalized:
            raise ConversationError("User text cannot be empty.")

        return normalized

    @staticmethod
    def _normalize_optional_text(
        value: Any,
        field_name: str,
    ) -> Optional[str]:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ConversationError(
                f"{field_name} must be a string or null."
            )

        normalized = value.strip()

        return normalized or None

    def _validate_decision(
        self,
        raw_decision: Any,
    ) -> ConversationResult:
        if not isinstance(raw_decision, dict):
            raise ConversationError(
                "Conversational provider must return a dictionary."
            )

        reply = raw_decision.get("reply")

        if not isinstance(reply, str) or not reply.strip():
            raise ConversationError(
                "Conversational decision requires a non-empty reply."
            )

        reply = reply.strip()

        decision_type = raw_decision.get(
            "decision_type",
            "CONVERSATION",
        )

        if not isinstance(decision_type, str):
            raise ConversationError(
                "decision_type must be a string."
            )

        decision_type = decision_type.strip().upper()

        if decision_type not in ALLOWED_DECISION_TYPES:
            raise ConversationError(
                f"Unsupported decision_type: {decision_type}"
            )

        mission_type = self._normalize_optional_text(
            raw_decision.get("mission_type"),
            "mission_type",
        )

        if mission_type is not None:
            mission_type = mission_type.upper()

            if mission_type not in ALLOWED_MISSION_TYPES:
                raise ConversationError(
                    f"Unsupported mission_type: {mission_type}"
                )

        query_type = self._normalize_optional_text(
            raw_decision.get("query_type"),
            "query_type",
        )

        if query_type is not None:
            query_type = query_type.upper()

            if query_type not in ALLOWED_WORLD_QUERY_TYPES:
                raise ConversationError(
                    f"Unsupported query_type: {query_type}"
                )

        target = self._normalize_optional_text(
            raw_decision.get("target"),
            "target",
        )

        requires_confirmation = raw_decision.get(
            "requires_confirmation",
            False,
        )

        if not isinstance(requires_confirmation, bool):
            raise ConversationError(
                "requires_confirmation must be a boolean."
            )

        if decision_type == "MISSION" and mission_type is None:
            raise ConversationError(
                "MISSION decisions require mission_type."
            )

        if decision_type != "MISSION" and mission_type is not None:
            raise ConversationError(
                "Only MISSION decisions may contain mission_type."
            )

        if decision_type == "WORLD_QUERY" and query_type is None:
            raise ConversationError(
                "WORLD_QUERY decisions require query_type."
            )

        if decision_type != "WORLD_QUERY" and query_type is not None:
            raise ConversationError(
                "Only WORLD_QUERY decisions may contain query_type."
            )

        if (
            decision_type not in {"MISSION", "WORLD_QUERY"}
            and target is not None
        ):
            raise ConversationError(
                "Only MISSION or WORLD_QUERY decisions may contain a target."
            )

        if (
            decision_type == "WORLD_QUERY"
            and query_type != "LATEST_ENTITY"
            and target is not None
        ):
            raise ConversationError(
                "Only LATEST_ENTITY world queries may contain a target."
            )

        if query_type == "LATEST_ENTITY" and target is None:
            raise ConversationError(
                "LATEST_ENTITY requires a target."
            )

        if mission_type == "FIND_OBJECT" and target is None:
            raise ConversationError(
                "FIND_OBJECT requires a target."
            )

        if mission_type == "FOLLOW_PERSON" and target is None:
            target = "person"

        return ConversationResult(
            reply=reply,
            decision_type=decision_type,
            mission_type=mission_type,
            query_type=query_type,
            target=target,
            requires_confirmation=requires_confirmation,
        )
