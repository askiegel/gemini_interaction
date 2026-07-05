from world_model import WorldModel


def print_section(title):
    print("")
    print(title)
    print("-" * len(title))


def main():
    wm = WorldModel()
    context = wm.get_context()

    print("")
    print("Mini Pupper Cognitive World Model Status")
    print("========================================")

    print_section("Robot State")
    for key, value in context["robot_state"].items():
        print(f"{key}: {value}")

    print_section("Vision Health")
    vision = context["environment"].get("vision", {})

    if not vision:
        print("No vision health data available.")
    else:
        for key, value in vision.items():
            print(f"{key}: {value}")

    print_section("Entities")
    entities = context["entities"]

    if not entities:
        print("No entities stored.")
    else:
        for entity in entities:
            print(
                f'{entity["entity_id"]} | '
                f'{entity["label"]} | '
                f'{entity["entity_type"]} | '
                f'confidence={entity["confidence"]} | '
                f'history={len(entity["history"])}'
            )

    print_section("Recent Events")
    events = context["recent_events"][-10:]

    if not events:
        print("No recent events.")
    else:
        for event in events:
            print(
                f'{event["timestamp"]} | '
                f'{event["type"]} | '
                f'{event["data"]}'
            )


if __name__ == "__main__":
    main()
