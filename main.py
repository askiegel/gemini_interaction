import json

from config import load_config
from gemini_client import GeminiClient
from logger import InteractionLogger


def fallback_intent(error):
    return {
        "intent": "UNKNOWN",
        "speech": "I could not safely understand that command.",
        "target": None,
        "error": str(error),
    }


def main():
    config = load_config()

    gemini = GeminiClient(
        api_key=config["api_key"],
        model=config["model"],
    )

    logger = InteractionLogger(config["log_file"])

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
            intent = gemini.get_intent(user_text)
        except Exception as e:
            intent = fallback_intent(e)

        logger.log(user_text, intent)

        print("Intent JSON:")
        print(json.dumps(intent, indent=2))
        print()


if __name__ == "__main__":
    main()
