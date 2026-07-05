from world_model import WorldModel
from vision_adapter import VisionAdapter


def main():
    wm = WorldModel()
    adapter = VisionAdapter(wm)

    entity_ids = adapter.process_once()

    print("")
    print("Vision Adapter One-Shot Result")
    print("------------------------------")
    print("Updated entity IDs:", entity_ids)

    print("")
    print("World entities:")
    entities = wm.get_entities()

    if not entities:
        print("No entities currently stored.")

    for entity in entities:
        print(
            entity["entity_id"],
            "|",
            entity["label"],
            "|",
            entity["entity_type"],
            "| confidence:",
            entity["confidence"],
            "| history:",
            len(entity["history"])
        )

    print("")
    print("Recent events:")
    for event in wm.get_recent_events()[-10:]:
        print(event["timestamp"], "|", event["type"], "|", event["data"])


if __name__ == "__main__":
    main()
