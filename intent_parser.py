import json


VALID_INTENTS = {
    "FOLLOW_PERSON",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "DESCRIBE_SCENE",
    "FIND_OBJECT",
    "RETURN_HOME",
    "UNKNOWN",
}


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def normalize_target(target):
    if target is None:
        return None

    target = str(target).strip().lower()

    aliases = {
        "back pack": "backpack",
        "book bag": "backpack",
        "person": "person",
        "human": "person",
        "tony": "tony",
    }

    return aliases.get(target, target)


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
        "target": normalize_target(data["target"]),
    }

