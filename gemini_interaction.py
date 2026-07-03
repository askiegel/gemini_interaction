#!/usr/bin/env python3
import os
import json
from typing import Optional
from pydantic import BaseModel
from google import genai


class RobotIntent(BaseModel):
    intent: str
    object: Optional[str] = None
    recipient: Optional[str] = None
    message: Optional[str] = None
    speech: str


SYSTEM_PROMPT = """
You are the interaction brain for Tony's Mini Pupper 2 robot.

Return ONLY valid JSON with these fields:
intent, object, recipient, message, speech.

Allowed intents:
- CHAT
- FOLLOW_PERSON
- FIND_OBJECT
- STOP
- DESCRIBE_SCENE
- SEND_MESSAGE
- UNKNOWN

Rules:
- Never directly command motors.
- Convert human language into safe robot intent.
- If the user asks to find a backpack, use FIND_OBJECT with object backpack.
- If the user says follow me, use FOLLOW_PERSON.
- If the user says stop, use STOP.
- If the user asks what you see, use DESCRIBE_SCENE.
"""


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY first.")
        print('Example: export GEMINI_API_KEY="your_key_here"')
        return

    client = genai.Client(api_key=api_key)

    print("Gemini Robot Interaction Manager")
    print("Type a command. Type quit to exit.")
    print()

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in ("quit", "exit"):
            break

        prompt = f"{SYSTEM_PROMPT}\n\nHuman command: {user_text}\n\nJSON:"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()

        # Remove markdown fences if Gemini adds them.
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
            intent = RobotIntent(**data)
            print("Robot Intent:")
            print(json.dumps(intent.model_dump(), indent=2))
        except Exception:
            print("Raw Gemini response:")
            print(text)


if __name__ == "__main__":
    main()
