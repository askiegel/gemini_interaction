from google import genai

from prompts import SYSTEM_PROMPT
from intent_parser import extract_json, validate_intent
from robot_context import get_robot_context, format_robot_context
from providers.base import CognitiveProvider


class GeminiProvider(CognitiveProvider):
    def __init__(self, api_key, model):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_intent(self, user_text):
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
