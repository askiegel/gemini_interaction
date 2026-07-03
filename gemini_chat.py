import os
import json
from dotenv import load_dotenv
from google import genai


VALID_INTENTS = {
    "FOLLOW_PERSON",
    "STOP",
    "DESCRIBE_SCENE",
    "FIND_OBJECT",
    "RETURN_HOME",
    "UNKNOWN"
}


SYSTEM_PROMPT = """
You are the Gemini Interaction Manager for a Mini Pupper 2 robot.

You do not directly control motors.

Your job is to convert human language into safe structured robot intent JSON.

Return ONLY valid JSON.

Allowed intents:
- FOLLOW_PERSON
- STOP
- DESCRIBE_SCENE
- FIND_OBJECT
- RETURN_HOME
- UNKNOWN

JSON format:
{
  "intent": "FOLLOW_PERSON",
  "speech": "Okay, I'll follow the person.",
  "target": null
}

Rules:
- If the user asks the robot to follow someone, use FOLLOW_PERSON.
- If the user asks the robot to stop, use STOP.
- If the user asks what the robot sees, use DESCRIBE_SCENE.
- If the user asks to find an object, use FIND_OBJECT and set target.
- If the request is unclear or unsafe, use UNKNOWN.
- Never include markdown.
- Never include explanations outside JSON.
"""


def load_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file")

    return genai.Client(api_key=api_key)


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def validate_intent(data):
    if not isinstance(data, dict):
        raise ValueError("Gemini response is not a JSON object")

    if "intent" not in data:
        raise ValueError("Missing intent field")

    if data["intent"] not in VALID_INTENTS:
        raise ValueError(f"Invalid intent: {data['intent']}")

    if "speech" not in data:
        data["speech"] = ""

    if "target" not in data:
        data["target"] = None

    return {
        "intent": data["intent"],
        "speech": data["speech"],
        "target": data["target"]
    }


def ask_gemini(client, user_text):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            SYSTEM_PROMPT,
            f"Human command: {user_text}"
        ],
    )

    raw_text = response.text
    data = extract_json(raw_text)
    return validate_intent(data)


def main():
    client = load_client()

    print("Gemini Interaction Manager")
    print("Type a command. Type 'quit' to exit.")
    print()

    while True:
        user_text = input("Human > ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        if not user_text:
            continue

        try:
            intent = ask_gemini(client, user_text)
            print("Intent JSON:")
            print(json.dumps(intent, indent=2))
            print()

        except Exception as e:
            fallback = {
                "intent": "UNKNOWN",
                "speech": "I could not safely understand that command.",
                "target": None,
                "error": str(e)
            }

            print("Intent JSON:")
            print(json.dumps(fallback, indent=2))
            print()


if __name__ == "__main__":
    main()
