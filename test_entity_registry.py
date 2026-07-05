from world_model import WorldModel
from entity_registry import EntityRegistry

wm = WorldModel("test_entity_registry_state.json")
registry = EntityRegistry(wm)

person_a = registry.register_observation(
    label="person",
    entity_type="human",
    confidence=0.92,
    source="test_camera",
    location={"frame": "camera", "cx": 320, "cy": 240},
    attributes={"reid": "candidate-a"}
)

person_a_again = registry.register_observation(
    label="person",
    entity_type="human",
    confidence=0.89,
    source="test_camera",
    location={"frame": "camera", "cx": 335, "cy": 245},
    attributes={"reid": "candidate-a"}
)

backpack = registry.register_observation(
    label="backpack",
    entity_type="object",
    confidence=0.85,
    source="test_camera",
    location={"frame": "camera", "cx": 500, "cy": 270},
    attributes={"targetable": True}
)

print("First person entity:", person_a)
print("Second person observation matched:", person_a_again)
print("Backpack entity:", backpack)

assert person_a == person_a_again
assert backpack != person_a

print("")
print("Registered entities:")
for entity in wm.get_entities():
    print(entity["entity_id"], entity["label"], entity["entity_type"], "history:", len(entity["history"]))

print("")
print("Entity Registry test passed.")
