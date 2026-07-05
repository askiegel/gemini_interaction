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


def print_mission_state(mission_manager):
    print("Mission State:")
    print(json.dumps(mission_manager.get_state(), indent=2))
    print()


def main():
    config = load_config()

    provider = create_provider(config)
    logger = InteractionLogger(config["log_file"])
    mission_manager = MissionManager()
    behavior_manager = BehaviorManager()

    print("Cognitive Interface")
    print(f"Provider: {config['provider']}")
    print("Mission Queue Simulator Active")
    print("Commands: complete, queue, state, quit")
    print()

    while True:
        user_text = input("Human > ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        if not user_text:
            continue

        if user_text.lower() == "complete":
            completed = mission_manager.complete_active_mission()
            if completed:
                print("Completed Mission:")
                print(json.dumps(completed.to_dict(), indent=2))
            else:
                print("No active mission to complete.")
            print_mission_state(mission_manager)
            continue

        if user_text.lower() in {"queue", "state"}:
            print_mission_state(mission_manager)
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
            "mission_state": mission_manager.get_state(),
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
        print_mission_state(mission_manager)


if __name__ == "__main__":
    main()
