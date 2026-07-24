import json

from intent_parser import extract_json, validate_intent
from prompts import CONVERSATION_SYSTEM_PROMPT, SYSTEM_PROMPT
from providers.base import CognitiveProvider
from robot_context import format_robot_context, get_robot_context


class GeminiProvider(CognitiveProvider):
    def __init__(self, api_key, model):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is required for Gemini operation. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_intent(self, user_text):
        """
        Preserve the existing command-oriented intent interface.

        This method is used by voice_command.py and the Cognitive Runtime.
        Conversational processing is intentionally handled separately by
        get_conversation_decision().
        """
        robot_context = get_robot_context()
        context_text = format_robot_context(robot_context)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                SYSTEM_PROMPT,
                context_text,
                f"Human command: {user_text}",
            ],
        )

        raw_text = response.text
        data = extract_json(raw_text)

        return validate_intent(data)

    def get_conversation_decision(self, user_text, history):
        """
        Convert conversational user text into a structured decision.

        This method does not submit missions or control the robot. The returned
        dictionary is validated later by ConversationManager.
        """
        normalized_history = self._normalize_conversation_history(history)

        history_text = json.dumps(
            normalized_history,
            ensure_ascii=False,
            indent=2,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                CONVERSATION_SYSTEM_PROMPT,
                (
                    "Conversation history, oldest to newest:\n"
                    f"{history_text}"
                ),
                f"Current human message: {user_text}",
            ],
        )

        raw_text = response.text

        return extract_json(raw_text)

    @staticmethod
    def _normalize_conversation_history(history):
        if history is None:
            return []

        if not isinstance(history, list):
            raise ValueError("Conversation history must be a list.")

        normalized = []

        for index, turn in enumerate(history):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"Conversation history item {index} must be a dictionary."
                )

            role = turn.get("role")
            text = turn.get("text")

            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"Conversation history item {index} has invalid role."
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Conversation history item {index} requires text."
                )

            normalized.append(
                {
                    "role": role,
                    "text": text.strip(),
                }
            )

        return normalized
