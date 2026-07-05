from world_model import WorldModel

wm = WorldModel("test_world_model_state.json")

wm.set_robot_state("mission", "TEST_WORLD_MODEL")

wm.update_entity(
    entity_id="person-001",
    label="person",
    entity_type="human",
    confidence=0.91,
    source="test",
    location={"frame": "camera", "cx": 320, "cy": 240},
    attributes={"reid": "candidate"}
)

wm.update_entity(
    entity_id="backpack-001",
    label="backpack",
    entity_type="object",
    confidence=0.84,
    source="test",
    location={"frame": "camera", "cx": 410, "cy": 260},
    attributes={"targetable": True}
)

print("World Model Context:")
print(wm.get_context())
