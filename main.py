import json

from config import load_config
from provider_factory import create_provider
from logger import InteractionLogger
from mission_manager import MissionManager
from behavior_manager import BehaviorManager
from event_bus import EventBus
from robot_context import get_world_model
from event_types import (
    EVENT_MISSION_COMPLETE,
    EVENT_TARGET_FOUND,
    EVENT_LOW_BATTERY,
    EVENT_OBSTACLE_DETECTED,
)


def fallback_intent(error):
    return {
        "intent": "UNKNOWN",
        "speech": "I could not safely understand that command.",
        "target": None,
        "error": str(error),
    }


def sync_world_model(mission_manager, event_bus):
    world = get_world_model()

    active = mission_manager.get_active_mission()
    world.update_robot_state(
        mission=active.mission_type if active else "IDLE",
        navigation_state="MISSION_ACTIVE" if active else "STANDBY",
    )

    for event in event_bus.events:
        if event.to_dict() not in world.events:
            world.add_event(event)

    return world


def print_mission_state(mission_manager, event_bus):
    world = sync_world_model(mission_manager, event_bus)

    print("Mission State:")
    print(json.dumps(mission_manager.get_state(), indent=2))
    print("Recent Events:")
    print(json.dumps(event_bus.recent(5), indent=2))
    print("World Model:")
    print(json.dumps(world.snapshot(), indent=2))
    print()


def handle_terminal_event(user_text, event_bus, mission_manager):
    text = user_text.lower().strip()

    if text == "complete":
        event = event_bus.publish(EVENT_MISSION_COMPLETE, {}, source="terminal")
        result = mission_manager.handle_event(event)
        return event, result

    if text.startswith("found "):
        target = text.replace("found ", "", 1).strip()
        event = event_bus.publish(
            EVENT_TARGET_FOUND,
            {"target": target},
            source="terminal",
        )
        result = mission_manager.handle_event(event)
        return event, result

    if text.startswith("battery "):
        value = int(text.replace("battery ", "", 1).strip())
        event = event_bus.publish(
            EVENT_LOW_BATTERY,
            {"battery_percent": value},
            source="terminal",
        )

        world = get_world_model()
        world.update_robot_state(battery_percent=value)

        result = mission_manager.handle_event(event)
        return event, result

    if text == "obstacle":
        event = event_bus.publish(
            EVENT_OBSTACLE_DETECTED,
            {"front_clear": False},
            source="terminal",
        )

        world = get_world_model()
        world.update_robot_state(front_clear=False, nearest_obstacle_m=0.25)

        result = mission_manager.handle_event(event)
        return event, result

    return None, None


def main():
    config = load_config()

    provider = create_provider(config)
    logger = InteractionLogger(config["log_file"])
    mission_manager = MissionManager()
    behavior_manager = BehaviorManager()
    event_bus = EventBus()

    print("Cognitive Interface")
    print(f"Provider: {config['provider']}")
    print("World Model + Event-Driven Mission Queue Active")
    print("Commands: complete, found <target>, battery <percent>, obstacle, queue, state, world, events, quit")
    print()

    while True:
        user_text = input("Human > ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"queue", "state"}:
            print_mission_state(mission_manager, event_bus)
            continue

        if user_text.lower() == "world":
            world = sync_world_model(mission_manager, event_bus)
            print(world.format_for_prompt())
            print()
            continue

        if user_text.lower() == "events":
            print("Event History:")
            print(json.dumps(event_bus.history(), indent=2))
            print()
            continue

        event, event_result = handle_terminal_event(user_text, event_bus, mission_manager)

        if event:
            print("Event:")
            print(json.dumps(event.to_dict(), indent=2))
            print()

            if event_result:
                print("Mission Result:")
                print(json.dumps(event_result.to_dict(), indent=2))
                print()
                print(behavior_manager.simulate(event_result))
                print()
            else:
                print("No mission transition occurred.")
                print()

            print_mission_state(mission_manager, event_bus)
            continue

        sync_world_model(mission_manager, event_bus)

        try:
            intent = provider.get_intent(user_text)
        except Exception as e:
            intent = fallback_intent(e)

        mission = mission_manager.handle_intent(intent)
        behavior = behavior_manager.simulate(mission)
        world = sync_world_model(mission_manager, event_bus)

        log_entry = {
            "provider": config["provider"],
            "intent": intent,
            "mission": mission.to_dict(),
            "mission_state": mission_manager.get_state(),
            "world_model": world.snapshot(),
            "behavior": behavior,
            "recent_events": event_bus.recent(5),
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
        print_mission_state(mission_manager, event_bus)


if __name__ == "__main__":
    main()
