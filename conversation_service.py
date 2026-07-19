#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from config import load_config
from conversation_manager import (
    ConversationError,
    ConversationManager,
    ConversationResult,
)
from provider_factory import create_provider
from voice_command import (
    DEFAULT_RUNTIME_URL,
    submit_intent_to_runtime,
)


@dataclass(frozen=True)
class ConversationServiceResult:
    """
    Complete result from one conversational interaction.

    A conversational response is always returned. Mission submission is
    optional and occurs only after ConversationManager validates the provider
    decision.
    """

    reply: str
    decision_type: str
    mission_type: Optional[str]
    target: Optional[str]
    requires_confirmation: bool
    mission_submitted: bool
    mission_submission: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationService:
    """
    Connect ConversationManager to the persistent Cognitive Runtime.

    Responsibilities:

    - accept user text,
    - obtain a validated ConversationResult,
    - return the robot's conversational reply,
    - convert optional missions into provider-independent intent dictionaries,
    - submit validated missions through the Runtime API.

    This service never controls motors, ROS, the Robot Bridge, or the
    BehaviorManager directly.
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        runtime_url: str = DEFAULT_RUNTIME_URL,
        mission_submitter: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        if conversation_manager is None:
            raise ValueError(
                "ConversationService requires a ConversationManager."
            )

        normalized_runtime_url = str(runtime_url).strip().rstrip("/")

        if not normalized_runtime_url:
            raise ValueError("runtime_url cannot be empty.")

        self.conversation_manager = conversation_manager
        self.runtime_url = normalized_runtime_url
        self.mission_submitter = (
            mission_submitter
            or submit_intent_to_runtime
        )

    def process_text(self, user_text: str) -> ConversationServiceResult:
        """
        Process one user message.

        Conversational and world-query responses do not create missions.
        Mission decisions requiring confirmation are not submitted until a
        future confirmation layer explicitly approves them.
        """
        result = self.conversation_manager.process(user_text)

        if not result.has_mission:
            return self._build_result(
                conversation_result=result,
                mission_submitted=False,
                mission_submission=None,
            )

        if result.requires_confirmation:
            return self._build_result(
                conversation_result=result,
                mission_submitted=False,
                mission_submission=None,
            )

        intent = self._build_runtime_intent(result)

        submission = self.mission_submitter(
            user_text=user_text.strip(),
            intent=intent,
            runtime_url=self.runtime_url,
        )

        if not isinstance(submission, dict):
            raise RuntimeError(
                "Runtime mission submitter must return a dictionary."
            )

        if not submission.get("ok") or not submission.get("accepted"):
            raise RuntimeError(
                "Cognitive Runtime did not accept the mission."
            )

        return self._build_result(
            conversation_result=result,
            mission_submitted=True,
            mission_submission=submission,
        )

    def clear_history(self) -> None:
        self.conversation_manager.clear_history()

    def get_history(self):
        return self.conversation_manager.get_history()

    @staticmethod
    def _build_runtime_intent(
        result: ConversationResult,
    ) -> Dict[str, Any]:
        if not result.has_mission:
            raise ValueError(
                "Cannot create a runtime intent without a mission."
            )

        return {
            "intent": result.mission_type,
            "speech": result.reply,
            "target": result.target,
        }

    @staticmethod
    def _build_result(
        conversation_result: ConversationResult,
        mission_submitted: bool,
        mission_submission: Optional[Dict[str, Any]],
    ) -> ConversationServiceResult:
        return ConversationServiceResult(
            reply=conversation_result.reply,
            decision_type=conversation_result.decision_type,
            mission_type=conversation_result.mission_type,
            target=conversation_result.target,
            requires_confirmation=(
                conversation_result.requires_confirmation
            ),
            mission_submitted=mission_submitted,
            mission_submission=mission_submission,
        )


def create_conversation_service(
    runtime_url: str = DEFAULT_RUNTIME_URL,
    max_history_turns: int = 12,
) -> ConversationService:
    """
    Build the production Conversation Service from project configuration.
    """
    config = load_config()
    provider = create_provider(config)

    manager = ConversationManager(
        provider=provider,
        max_history_turns=max_history_turns,
    )

    return ConversationService(
        conversation_manager=manager,
        runtime_url=runtime_url,
    )


def run_terminal_loop(service: ConversationService) -> None:
    """
    Run the first interactive Conversation Service interface.

    Commands:

    - quit or exit: stop the service
    - /clear: clear conversation history
    - /history: display current in-memory history
    """
    print("============================================")
    print(" Mini Pupper 2 Conversation Service v1")
    print("============================================")
    print()
    print(f"Runtime API: {service.runtime_url}")
    print("Type a message and press Enter.")
    print("Commands: /clear, /history, quit")
    print()

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            print("Conversation input closed.")
            break
        except KeyboardInterrupt:
            print()
            print("Conversation Service stopped.")
            break

        if not user_text:
            continue

        normalized_command = user_text.lower()

        if normalized_command in {"quit", "exit"}:
            print("Robot: Goodbye.")
            break

        if normalized_command == "/clear":
            service.clear_history()
            print("Robot: Conversation history cleared.")
            print()
            continue

        if normalized_command == "/history":
            print(
                json.dumps(
                    service.get_history(),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print()
            continue

        try:
            result = service.process_text(user_text)

            print(f"Robot: {result.reply}")

            if result.requires_confirmation:
                print(
                    "[mission not submitted: confirmation required]"
                )
            elif result.mission_submitted:
                mission = (
                    result.mission_submission or {}
                ).get("mission", {})

                mission_id = mission.get(
                    "mission_id",
                    "unknown",
                )
                mission_status = mission.get(
                    "status",
                    "unknown",
                )

                print(
                    "[mission submitted: "
                    f"{result.mission_type}, "
                    f"id={mission_id}, "
                    f"status={mission_status}]"
                )

            print()

        except ConversationError as exc:
            print(f"Robot: I could not process that safely.")
            print(f"[conversation error: {exc}]")
            print()

        except RuntimeError as exc:
            print(
                "Robot: I understood the request, but I could not "
                "send it to the robot runtime."
            )
            print(f"[runtime error: {exc}]")
            print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Mini Pupper 2 terminal-based "
            "Conversation Service."
        )
    )

    parser.add_argument(
        "--runtime-url",
        default=os.getenv(
            "COGNITIVE_RUNTIME_URL",
            DEFAULT_RUNTIME_URL,
        ),
        help=(
            "Cognitive Runtime base URL. "
            f"Default: {DEFAULT_RUNTIME_URL}"
        ),
    )

    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=12,
        help="Maximum user/assistant exchanges retained in memory.",
    )

    args = parser.parse_args()

    service = create_conversation_service(
        runtime_url=args.runtime_url,
        max_history_turns=args.max_history_turns,
    )

    run_terminal_loop(service)


if __name__ == "__main__":
    main()
