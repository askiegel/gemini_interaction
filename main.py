import json

from config import load_config
from provider_factory import create_provider
from logger import InteractionLogger
from mission_manager import MissionManager
from behavior_manager import BehaviorManager


def fallback_intent(error):
    return {
        "intent": "UNKNOWN",
        "speech": "I could not safely understand that command.",
        "target": None,
        "error": str(error),
    }


def main():
    config = load_config()

    provider = create_provider(config)
    logger = InteractionLogger(config["log_file"])
    mission_manager = MissionManager()
    behavior_manager = BehaviorManager()

    print("Cognitive Interface")
    print(f"Provider: {config['provider']}")
    print("Mission Manager Simulator Active")
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
            intent = provider.get_intent(user_text)
        except Exception as e:
            intent = fallback_intent(e)

        mission = mission_manager.handle_intent(intent)
        behavior = behavior_manager.simulate(mission)

        log_entry = {
            "provider": config["provider"],
            "intent": intent,
            "mission": mission.to_dict(),
            "behavior": behavior,
        }

        logger.log(user_text, log_entry)

        print("Intent JSON:")
        print(json.dumps(intent, indent=2))
        print()

        print("Mission:")
        print(json.dumps(mission.to_dict(), indent=2))
        print()

        print(behavior)
        print()


if __name__ == "__main__":
    main()
