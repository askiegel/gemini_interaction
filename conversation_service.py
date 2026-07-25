#!/usr/bin/env python3

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, Optional

from config import load_config
from conversation_manager import (
    ConversationError,
    ConversationManager,
    ConversationResult,
)
from provider_factory import create_provider
from robot_addressing import AddressedCommand, RobotAddressParser
from robot_fleet import load_robot_fleet
from robot_identity import RobotIdentity, get_robot_identity
from world_model import WorldModel
from world_query_service import (
    WorldQueryError,
    WorldQueryService,
)
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
    query_type: Optional[str]
    target: Optional[str]
    requires_confirmation: bool
    mission_submitted: bool
    mission_submission: Optional[Dict[str, Any]]
    world_query: Optional[Dict[str, Any]] = None
    robot_id: Optional[str] = None
    addressing: Optional[Dict[str, Any]] = None
    accepted: bool = True
    ignored: bool = False

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
        world_query_service: Optional[WorldQueryService] = None,
        local_identity: Optional[RobotIdentity] = None,
        address_parser: Optional[RobotAddressParser] = None,
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
        self.world_query_service = world_query_service

        self.local_identity = (
            local_identity
            or get_robot_identity()
        )

        if not isinstance(self.local_identity, RobotIdentity):
            raise TypeError(
                "local_identity must be a RobotIdentity."
            )

        if address_parser is None:
            fleet = load_robot_fleet(
                local_identity=self.local_identity,
            )

            address_parser = RobotAddressParser(
                local_identity=self.local_identity,
                known_identities=fleet.remote_identities,
            )

        if not isinstance(address_parser, RobotAddressParser):
            raise TypeError(
                "address_parser must be a RobotAddressParser."
            )

        self.address_parser = address_parser

    def process_text(
        self,
        user_text: str,
        submit_missions: bool = True,
    ) -> ConversationServiceResult:
        """
        Process one user message through deterministic robot addressing before
        conversational interpretation.

        Commands addressed to another robot are ignored before the provider,
        conversation history, Runtime API, ROS, or Robot Bridge are contacted.
        """
        if not isinstance(submit_missions, bool):
            raise ValueError("submit_missions must be a boolean.")

        addressed = self.address_parser.parse(user_text)

        if not addressed.is_for(self.local_identity.id):
            return ConversationServiceResult(
                reply=(
                    f"That command is addressed to "
                    f"{addressed.addressed_robot_name or 'another robot'}."
                ),
                decision_type="CONVERSATION",
                mission_type=None,
                query_type=None,
                target=None,
                requires_confirmation=False,
                mission_submitted=False,
                mission_submission=None,
                world_query=None,
                robot_id=self.local_identity.id,
                addressing=addressed.to_dict(),
                accepted=False,
                ignored=True,
            )

        if not addressed.command_text:
            raise ConversationError(
                "A robot name was recognized, but no command followed it."
            )

        result = self._process_local_text(
            command_text=addressed.command_text,
            original_text=addressed.original_text,
            addressed=addressed,
            submit_missions=submit_missions,
        )

        return replace(
            result,
            robot_id=self.local_identity.id,
            addressing=addressed.to_dict(),
            accepted=True,
            ignored=False,
        )

    def _process_local_text(
        self,
        command_text: str,
        original_text: str,
        addressed: AddressedCommand,
        submit_missions: bool,
    ) -> ConversationServiceResult:
        result = self.conversation_manager.process(
            command_text
        )

        if result.decision_type == "WORLD_QUERY":
            return self._process_world_query(
                conversation_result=result,
            )

        if not result.has_mission:
            return self._build_result(
                conversation_result=result,
                mission_submitted=False,
                mission_submission=None,
                world_query=None,
            )

        if result.requires_confirmation:
            return self._build_result(
                conversation_result=result,
                mission_submitted=False,
                mission_submission=None,
                world_query=None,
            )

        if not submit_missions:
            return self._build_result(
                conversation_result=result,
                mission_submitted=False,
                mission_submission=None,
                world_query=None,
            )

        intent = self._build_runtime_intent(result)

        submission = self._submit_runtime_mission(
            original_text=original_text,
            intent=intent,
            addressed=addressed,
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
            world_query=None,
        )

    def _submit_runtime_mission(
        self,
        original_text: str,
        intent: Dict[str, Any],
        addressed: AddressedCommand,
    ) -> Dict[str, Any]:
        """
        Submit mission ownership metadata when supported.

        Legacy test doubles and older submitters that accept only user_text,
        intent, and runtime_url continue to work unchanged.
        """
        kwargs = {
            "user_text": original_text.strip(),
            "intent": intent,
            "runtime_url": self.runtime_url,
        }

        supports_identity = False

        try:
            signature = inspect.signature(
                self.mission_submitter
            )

            parameters = signature.parameters

            supports_identity = (
                "robot_id" in parameters
                or "addressing" in parameters
                or any(
                    parameter.kind
                    == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )
        except (TypeError, ValueError):
            supports_identity = True

        if supports_identity:
            kwargs.update(
                {
                    "robot_id": self.local_identity.id,
                    "addressing": addressed.to_dict(),
                }
            )

        return self.mission_submitter(**kwargs)

    def _process_world_query(
        self,
        conversation_result: ConversationResult,
    ) -> ConversationServiceResult:
        if self.world_query_service is None:
            return self._build_result(
                conversation_result=conversation_result,
                mission_submitted=False,
                mission_submission=None,
                world_query=None,
            )

        try:
            query_result = self.world_query_service.execute(
                query_type=conversation_result.query_type,
                target=conversation_result.target,
            )
        except WorldQueryError as exc:
            return self._build_result(
                conversation_result=ConversationResult(
                    reply=(
                        "I understand that you are asking about "
                        "my World Model, but I could not determine "
                        "which stored information to retrieve."
                    ),
                    decision_type="WORLD_QUERY",
                    mission_type=None,
                    query_type=conversation_result.query_type,
                    target=conversation_result.target,
                    requires_confirmation=False,
                ),
                mission_submitted=False,
                mission_submission=None,
                world_query={
                    "ok": False,
                    "error": str(exc),
                },
            )

        return self._build_result(
            conversation_result=ConversationResult(
                reply=query_result.reply,
                decision_type="WORLD_QUERY",
                mission_type=None,
                query_type=query_result.query_type,
                target=query_result.target,
                requires_confirmation=False,
            ),
            mission_submitted=False,
            mission_submission=None,
            world_query=query_result.to_dict(),
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
        world_query: Optional[Dict[str, Any]],
    ) -> ConversationServiceResult:
        return ConversationServiceResult(
            reply=conversation_result.reply,
            decision_type=conversation_result.decision_type,
            mission_type=conversation_result.mission_type,
            query_type=conversation_result.query_type,
            target=conversation_result.target,
            requires_confirmation=(
                conversation_result.requires_confirmation
            ),
            mission_submitted=mission_submitted,
            mission_submission=mission_submission,
            world_query=world_query,
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

    world_model = WorldModel()

    return ConversationService(
        conversation_manager=manager,
        runtime_url=runtime_url,
        world_query_service=WorldQueryService(
            world_model=world_model,
        ),
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
